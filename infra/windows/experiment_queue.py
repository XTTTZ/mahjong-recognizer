"""Durable serial experiment queue. Controls only children created by this process.

Launch independently of SSH with the existing WMI job runner. A restarted queue
never blindly duplicates a previous live child or overwrites an experiment run.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PureWindowsPath
import posixpath
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

TERMINAL = {"succeeded", "failed", "timed_out", "interrupted", "not_started_budget", "not_started_deadline", "skipped_dependency"}
ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}")


def now():
    return datetime.now(timezone.utc).isoformat()


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def config_contract(config):
    if config.get("schemaVersion") != 1 or not ID.fullmatch(config.get("queue_id", "")):
        raise ValueError("Expected schemaVersion1 and a safe queue_id")
    if not 0 < config.get("deadline_hours", 24) <= 24 or config.get("reserve_minutes", 15) < 0:
        raise ValueError("Queue budget must be positive and at most24 hours")
    if config.get("gpu_policy", "exclusive") not in {"exclusive", "disabled"}:
        raise ValueError("gpu_policy must be exclusive or disabled")
    if config.get("poll_seconds", 2) < .05 or config.get("wait_timeout_minutes", 15) <= 0:
        raise ValueError("Invalid polling/wait timeout")
    background_allowlist(config.get("allowed_gpu_process_paths", []))
    jobs = config.get("experiments", [])
    if not jobs or len(jobs) > 100:
        raise ValueError("A bounded nonempty experiment list is required")
    ids, runs = set(), set()
    for job in jobs:
        if not ID.fullmatch(job.get("id", "")) or job["id"] in ids:
            raise ValueError("Experiment IDs must be unique safe names")
        dependencies = job.get("depends_on", [])
        if not isinstance(dependencies, list) or not all(isinstance(v, str) and v in ids for v in dependencies) or len(dependencies) != len(set(dependencies)):
            raise ValueError("Dependencies must be unique experiment IDs appearing earlier in the queue")
        ids.add(job["id"])
        run = str(Path(job["run_dir"]).absolute()).casefold()
        if run in runs or run == str(Path(config["queue_dir"]).absolute()).casefold():
            raise ValueError("Each experiment needs its own run directory, separate from the queue")
        runs.add(run)
        if job.get("estimated_minutes", 0) <= 0 or job.get("timeout_minutes", 0) <= 0:
            raise ValueError("Each experiment needs positive estimate and timeout")
        names = set()
        for step in job.get("steps", []):
            if not ID.fullmatch(step.get("name", "")) or step["name"] in names:
                raise ValueError("Each experiment needs unique safe step names")
            names.add(step["name"])
            if not step.get("argv") or not all(isinstance(v, str) and v for v in step["argv"]):
                raise ValueError("Each step needs an argv string array; shell commands are forbidden")
        if not names:
            raise ValueError("Each experiment needs steps")
    return config


class QueueLock:
    """OS lock releases on process exit; stale files alone never imply activity."""
    def __init__(self, path):
        self.path = Path(path)
        self.stream = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0)
        if not self.stream.read(1):
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("This queue already has a live coordinator")
        return self

    def __exit__(self, *_):
        if self.stream is not None:
            self.stream.close()


def pid_alive(pid):
    if not isinstance(pid, int) or pid <= 0:
        return False
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            # Access denied is an active process whose ownership is unknown.
            return ctypes.get_last_error() == 5
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == 259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def normalize_executable_path(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("GPU background allowlist entries must be absolute executable paths")
    path = value.strip().replace("\\", "/")
    if path.startswith("//?/"):
        path = path[4:]
    windows = PureWindowsPath(path).is_absolute()
    if not windows and not path.startswith("/"):
        raise ValueError("GPU background allowlist must not contain generic executable names")
    normalized = posixpath.normpath(path)
    return normalized.casefold() if windows or os.name == "nt" else normalized


def background_allowlist(paths):
    result = set()
    for value in paths:
        normalized = normalize_executable_path(value)
        executable = normalized.rsplit("/", 1)[-1].casefold()
        if re.fullmatch(r"(?:python(?:w|[0-9.]+)?|torchrun|accelerate)(?:\.exe)?", executable):
            raise ValueError("Python/ML runtimes must not be allowed as desktop GPU backgrounds")
        result.add(normalized)
    return result


def parse_gpu_processes(xml, allowed_paths=()):
    # Driver617/WDDM can report desktop apps as C+G even through the compute
    # query. Never identify background apps merely by executable basename.
    allowed = background_allowlist(allowed_paths)
    root = ET.fromstring(xml)
    gpus = root.findall("gpu")
    if root.tag != "nvidia_smi_log" or not gpus:
        raise ValueError("nvidia-smi XML does not contain GPU process evidence")
    result = {"blocking_processes": [], "allowed_background_processes": [], "graphics_processes": []}
    for gpu in gpus:
        container = gpu.find("processes")
        if container is None or ((container.text or "").strip() and not list(container)):
            raise ValueError("GPU process information is unavailable; idle is unverified")
        for process in container.findall("process_info"):
            pid = (process.findtext("pid") or "").strip()
            if not pid.isdigit():
                raise ValueError("GPU process entry has no valid PID")
            kind = (process.findtext("type") or "unknown").strip().upper()
            name = (process.findtext("process_name") or "unknown").strip()
            record = {"pid": int(pid), "name": name, "type": kind, "gpu_id": gpu.get("id", "unknown")}
            if kind == "G":
                result["graphics_processes"].append(record)
                continue
            try:
                normalized = normalize_executable_path(name)
            except ValueError:
                normalized = None
            if normalized in allowed:
                result["allowed_background_processes"].append(record)
            else:
                result["blocking_processes"].append(record)
    return result


def query_gpu_processes(allowed_paths=()):
    # Parse bytes using the XML-declared encoding, independent of Windows codepage.
    result = subprocess.run(["nvidia-smi", "-q", "-x"], capture_output=True, timeout=15, check=True)
    return parse_gpu_processes(result.stdout, allowed_paths)


def compute_processes(allowed_paths=()):
    return query_gpu_processes(allowed_paths)["blocking_processes"]


def gpu_snapshot():
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=10, check=True)
    return [{"gpu_index": row[0].strip(), "memory_used_mib": row[1].strip(),
             "memory_total_mib": row[2].strip(), "utilization_percent": row[3].strip()}
            for row in csv.reader(result.stdout.splitlines()) if len(row) == 4]


def stop_owned(process):
    """Only called with the current Popen handle, never an inherited PID."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, check=False, timeout=15)
    else:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        process.wait(timeout=10)


def report_summary(job):
    reports = []
    metadata_file = Path(job["run_dir"]) / "run.json"
    try:
        metadata = read_json(metadata_file) if metadata_file.exists() else {}
    except (OSError, ValueError):
        metadata = {}
    for name in job.get("report_paths", []):
        path = Path(name)
        if not path.is_file():
            continue
        try:
            if path.stat().st_size > 4 * 1024**2:
                raise ValueError("Validation report too large")
            value = read_json(path)
            if not isinstance(value, dict):
                raise ValueError("Validation report must be an object")
            if value.get("split") != "validation_only":
                raise ValueError("Only validation-only reports may be aggregated")
            threshold = value.get("bestThreshold")
            points = value.get("thresholds", {}).get(str(threshold), {})
            group = value.get("selectionGroup", "ten_or_more_tiles")
            metrics = points.get(group) or points.get("all") or {}
            if not isinstance(metrics, dict):
                raise ValueError("Validation metrics must be an object")
            for metric in ("hand_exact_match", "model_exact", "accepted_coverage"):
                score = metrics.get(metric)
                if score is not None and (not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1):
                    raise ValueError("Validation rates must be finite values in[0,1]")
            contract = value.get("dataContract", {}).get("splits") or metadata.get("dataContract") or metadata.get("resumeSignature", {}).get("data") or {}
            pixel_hash = contract.get("valid", {}).get("imageTreeSha256")
            annotation_hash = value.get("validationAnnotationSha256")
            cohort_key = (str(annotation_hash) + ":" + pixel_hash) if annotation_hash and pixel_hash else None
            reports.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "validAnnotationSha256": annotation_hash, "validImageTreeSha256": pixel_hash, "cohortKey": cohort_key,
                            "bestThreshold": threshold, "selectionGroup": group,
                            "metrics": metrics, "externalPhoneBenchmark": value.get("externalPhoneBenchmark", False),
                            "partial": value.get("partial", False) or value.get("partialValidation", False)})
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            reports.append({"path": str(path), "error": str(error)})
    return reports


class ExperimentQueue:
    def __init__(self, config, *, clock=time.time, sleep=time.sleep, monotonic=time.monotonic):
        self.config = config_contract(config)
        self.root = Path(config["queue_dir"])
        self.clock, self.sleep, self.monotonic = clock, sleep, monotonic
        self.deadline_monotonic = None
        self.poll = config.get("poll_seconds", 2)
        self.config_sha = signature(config)
        self.state = None
        self.child = None
        self.keep_awake = False

    def event(self, event, **values):
        record = {"at": now(), "event": event, **values}
        with (self.root / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    def save(self, **values):
        self.state.update(values, updated_at=now())
        write_json(self.root / "state.json", self.state)
        jobs = []
        cohorts = {}
        for job in self.config["experiments"]:
            path = self.root / "jobs" / (job["id"] + ".json")
            value = read_json(path) if path.exists() else {"id": job["id"], "state": "pending"}
            jobs.append(value)
            for report in value.get("reports", []):
                key = report.get("cohortKey")
                if key and report.get("metrics") and not report.get("partial"):
                    cohorts.setdefault(key, []).append({"experiment": job["id"], **report})
        for rows in cohorts.values():
            rows.sort(key=lambda r: (r["metrics"].get("hand_exact_match") or 0,
                                     r["metrics"].get("model_exact") or 0,
                                     r["metrics"].get("accepted_coverage") or 0), reverse=True)
        write_json(self.root / "summary.json", {"queue": self.state, "experiments": jobs, "validationCohorts": cohorts,
            "note": "Rank only within identical validation annotation and image-tree hashes; missing provenance is unranked. Research/oracle validation is not field accuracy."})

    def remaining(self):
        wall_remaining = self.state["deadline_epoch_seconds"] - self.clock()
        mono_remaining = self.deadline_monotonic - self.monotonic() if self.deadline_monotonic is not None else wall_remaining
        return max(0, min(wall_remaining, mono_remaining))

    def wait_gate(self, reason, probe, timeout_seconds):
        started = self.monotonic()
        self.save(state="waiting", wait_reason=reason)
        while True:
            ready, detail = probe()
            if ready:
                self.event("wait_released", reason=reason, detail=detail)
                return
            if self.remaining() <= 0 or self.monotonic() - started >= timeout_seconds:
                raise RuntimeError(f"Waiting timed out without taking over another process: {reason}")
            self.save(wait_detail=detail)
            self.sleep(min(self.poll, self.remaining()))

    def wait_for_external(self):
        for filename in self.config.get("wait_for_status", []):
            def probe(path=Path(filename)):
                try:
                    value = read_json(path)
                except (OSError, ValueError):
                    return False, "status missing or being updated"
                state = value.get("state")
                if state in {"succeeded", "failed"}:
                    return True, {"path": str(path), "state": state}
                child = value.get("child_pid")
                if child and not pid_alive(child):
                    return True, {"path": str(path), "state": "orphaned_status_process_exited"}
                return False, {"path": str(path), "state": state, "child_pid": child}
            self.wait_gate("existing experiment " + filename, probe, self.config.get("wait_timeout_minutes", 15) * 60)

    def wait_gpu(self):
        if self.config.get("gpu_policy", "exclusive") == "disabled":
            return
        def probe():
            try:
                processes = query_gpu_processes(self.config.get("allowed_gpu_process_paths", []))
            except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError) as error:
                return False, {"gpu_idle_verified": False, "errorType": type(error).__name__}
            return not processes["blocking_processes"], processes
        self.wait_gate("GPU compute processes must be idle", probe, self.config.get("wait_timeout_minutes", 15) * 60)

    def recover_running(self, receipt):
        child = receipt.get("child_pid")
        if child and pid_alive(child):
            # An inherited PID is deliberately never killed: it could have been
            # reused. Wait and refuse duplicate execution until it disappears.
            self.wait_gate("previous queue child " + str(child), lambda: (not pid_alive(child), {"pid": child}),
                           self.config.get("wait_timeout_minutes", 15) * 60)
        receipt.update(state="interrupted", finished_at=now(),
                       error="Coordinator restarted; partial run retained and not replayed. Use a new experiment ID for an explicit retry.")
        write_json(self.root / "jobs" / (receipt["id"] + ".json"), receipt)
        if Path(receipt["run_dir"]).exists():
            write_json(Path(receipt["run_dir"]) / "status.json", receipt)
        self.event("recovered_interrupted", experiment=receipt["id"])

    def run_job(self, job):
        receipt_path = self.root / "jobs" / (job["id"] + ".json")
        if receipt_path.exists():
            previous = read_json(receipt_path)
            if previous.get("experiment_sha256") != signature(job):
                raise ValueError("Experiment changed after its receipt was written")
            if previous["state"] not in TERMINAL:
                self.recover_running(previous)
            self.event("already_recorded", experiment=job["id"], state=read_json(receipt_path)["state"])
            return
        dependencies = {}
        for identifier in job.get("depends_on", []):
            path = self.root / "jobs" / (identifier + ".json")
            dependencies[identifier] = read_json(path).get("state") if path.exists() else "missing"
        if any(state != "succeeded" for state in dependencies.values()):
            receipt = {"id": job["id"], "state": "skipped_dependency", "experiment_sha256": signature(job),
                       "dependencies": dependencies, "finished_at": now(),
                       "error": "Required earlier experiment did not succeed; independent experiments still continue"}
            write_json(receipt_path, receipt)
            self.event("skipped_dependency", experiment=job["id"], dependencies=dependencies)
            return
        estimate = job["estimated_minutes"] * 60 + self.config.get("reserve_minutes", 15) * 60
        if self.remaining() < estimate:
            receipt = {"id": job["id"], "state": "not_started_budget", "experiment_sha256": signature(job),
                       "remaining_seconds": self.remaining(), "required_estimated_seconds": estimate, "finished_at": now()}
            write_json(receipt_path, receipt)
            self.event("skipped_budget", experiment=job["id"], remaining_seconds=self.remaining())
            return
        self.wait_gpu()
        if self.remaining() < estimate:
            write_json(receipt_path, {"id": job["id"], "state": "not_started_budget", "experiment_sha256": signature(job),
                "remaining_seconds": self.remaining(), "required_estimated_seconds": estimate, "finished_at": now()})
            self.event("skipped_budget_after_gpu_wait", experiment=job["id"])
            return
        run = Path(job["run_dir"])
        error = "Run directory exists without a matching queue receipt" if run.exists() else ("Working directory missing" if not Path(job["cwd"]).is_dir() else None)
        if error:
            receipt = {"id": job["id"], "state": "failed", "experiment_sha256": signature(job), "error": error, "finished_at": now()}
            write_json(receipt_path, receipt)
            self.event("experiment_refused", experiment=job["id"], reason=error)
            return
        run.mkdir(parents=True)
        receipt = {"id": job["id"], "state": "starting", "experiment_sha256": signature(job),
                   "run_dir": str(run), "started_at": now(), "steps": [], "step": None}
        write_json(receipt_path, receipt)
        write_json(run / "status.json", receipt)
        self.save(state="running", current_experiment=job["id"], current_run_dir=str(run), current_step=None)
        self.event("experiment_started", experiment=job["id"])
        job_deadline = self.monotonic() + min(self.remaining(), job["timeout_minutes"] * 60)
        started_monotonic = self.monotonic()
        try:
            for step in job["steps"]:
                environment = os.environ.copy()
                environment.update({str(k): str(v) for k, v in self.config.get("env", {}).items()})
                environment.update({str(k): str(v) for k, v in job.get("env", {}).items()})
                environment.update({str(k): str(v) for k, v in step.get("env", {}).items()})
                environment.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
                receipt.update(state="running", step=step["name"], child_pid=None)
                self.save(current_step=step["name"])
                self.event("step_started", experiment=job["id"], step=step["name"])
                with (run / "stdout.log").open("ab", buffering=0) as stdout, (run / "stderr.log").open("ab", buffering=0) as stderr:
                    marker = ("\n[QUEUE] " + step["name"] + " " + now() + "\n").encode()
                    stdout.write(marker)
                    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
                    process = subprocess.Popen(step["argv"], cwd=step.get("cwd", job["cwd"]), env=environment,
                        stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, shell=False, **kwargs)
                    self.child = process
                    receipt["child_pid"] = process.pid
                    write_json(receipt_path, receipt)
                    write_json(run / "status.json", receipt)
                    self.save(child_pid=process.pid)
                    step_timeout = min(job_deadline, self.monotonic() + step.get("timeout_minutes", job["timeout_minutes"]) * 60)
                    telemetry_at = 0.0
                    while process.poll() is None:
                        if self.config.get("gpu_policy", "exclusive") == "exclusive" and self.monotonic() >= telemetry_at:
                            try:
                                sample = gpu_snapshot()
                                processes = query_gpu_processes(self.config.get("allowed_gpu_process_paths", []))
                                self.event("gpu_sample", experiment=job["id"], step=step["name"], child_pid=process.pid, gpus=sample, processes=processes)
                                self.save(gpu_sample=sample, gpu_processes=processes)
                            except (OSError, ValueError, ET.ParseError, subprocess.SubprocessError):
                                self.event("gpu_telemetry_unavailable", experiment=job["id"], step=step["name"])
                            telemetry_at = self.monotonic() + self.config.get("telemetry_seconds", 60)
                        if self.monotonic() >= step_timeout or self.remaining() <= 0:
                            stop_owned(process)
                            raise TimeoutError("Owned step reached its time/queue budget; saved checkpoints retained")
                        self.sleep(min(self.poll, max(.01, step_timeout - self.monotonic())))
                    code = process.returncode
                    self.child = None
                receipt["steps"].append({"name": step["name"], "exit_code": code, "finished_at": now()})
                self.event("step_finished", experiment=job["id"], step=step["name"], exit_code=code)
                if code != 0:
                    receipt.update(state="failed", exit_code=code, error="Step exited nonzero; see stderr.log")
                    break
            else:
                receipt.update(state="succeeded", exit_code=0)
        except TimeoutError as error:
            receipt.update(state="timed_out", error=str(error), exit_code=self.child.returncode if self.child else None)
            self.child = None
        except KeyboardInterrupt:
            if self.child is not None:
                stop_owned(self.child)
                self.child = None
            receipt.update(state="interrupted", error="Queue coordinator explicitly interrupted")
            raise
        except Exception as error:
            if self.child is not None:
                stop_owned(self.child)
                self.child = None
            receipt.update(state="failed", error=f"{type(error).__name__}: {error}")
        finally:
            reports = report_summary(job)
            if receipt["state"] == "succeeded" and job.get("report_paths") and not any(r.get("metrics") and not r.get("error") for r in reports):
                receipt.update(state="failed", error="Steps exited0 but no valid validation report was produced")
            receipt.update(finished_at=now(), elapsed_seconds=round(self.monotonic() - started_monotonic, 3), child_pid=None, reports=reports)
            write_json(receipt_path, receipt)
            write_json(run / "status.json", receipt)
            self.event("experiment_finished", experiment=job["id"], state=receipt["state"])
            self.save(child_pid=None)

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        with QueueLock(self.root / "queue.lock"):
            state_path = self.root / "state.json"
            if state_path.exists():
                self.state = read_json(state_path)
                if self.state.get("config_sha256") != self.config_sha:
                    raise ValueError("Queue configuration changed; create a new queue directory")
            else:
                created = self.clock()
                self.state = {"queue_id": self.config["queue_id"], "config_sha256": self.config_sha,
                    "created_at": now(), "created_epoch_seconds": created,
                    "deadline_epoch_seconds": created + self.config.get("deadline_hours", 24) * 3600,
                    "state": "starting", "current_experiment": None, "current_run_dir": None, "current_step": None}
                write_json(self.root / "config.json", self.config)
            if os.name == "nt":
                self.keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            self.deadline_monotonic = self.monotonic() + max(0, self.state["deadline_epoch_seconds"] - self.clock())
            self.save(coordinator_pid=os.getpid(), keep_system_awake=self.keep_awake)
            try:
                self.wait_for_external()
                for job in self.config["experiments"]:
                    self.run_job(job)
                    self.save()
                counts = {}
                for job in self.config["experiments"]:
                    status = read_json(self.root / "jobs" / (job["id"] + ".json"))["state"]
                    counts[status] = counts.get(status, 0) + 1
                result = "all_succeeded" if counts.get("succeeded") == len(self.config["experiments"]) else (
                    "completed_with_failures" if any(counts.get(s) for s in ("failed", "interrupted", "timed_out", "skipped_dependency")) else "budget_limited")
                self.save(state="completed", result=result, experiment_counts=counts, finished_at=now(), current_experiment=None, current_step=None,
                          current_run_dir=None, child_pid=None)
                self.event("queue_finished", remaining_seconds=self.remaining())
                return self.state
            except KeyboardInterrupt:
                self.save(state="interrupted", finished_at=now(), child_pid=None)
                raise
            except Exception as error:
                self.save(state="blocked", error=f"{type(error).__name__}: {error}", child_pid=None)
                self.event("queue_blocked", error=self.state["error"])
                raise
            finally:
                if self.keep_awake:
                    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    queue = ExperimentQueue(read_json(args.config))
    try:
        queue.run()
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
