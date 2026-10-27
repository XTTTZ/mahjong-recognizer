# Windows training host

Host: existing trusted SSH alias `win` (Windows 11 / RTX 4070 Ti 12 GB).

The native Windows Conda environment is the currently selected path. WSL is optional and was not installed; no reboot, driver replacement or global PATH/Python registration was performed. Use the clean `mahjong-train` environment. The earlier `mahjong` environment is retained as an investigation backup: it acquired overlapping Conda/PyPI native packages and must not be used for training.

## Environment

- Miniforge: `C:\Users\TRAINING_USER\miniforge3`
- Training Python: `C:\Users\TRAINING_USER\miniforge3\envs\mahjong-train\python.exe`
- Data/code/runs/logs: `D:\mahjong\{data,code,runs,logs}`
- `environment.yml` pins Python, pip and Git; `requirements-torch.txt` selects the official CUDA 12.8 wheels; `requirements-data.txt` pins the initial data/image tools.
- `requirements-training.txt` selects RF-DETR 1.11.2 training extras. Its constraints retain the installed CUDA-enabled torch/torchvision and verified image library versions.
- Runtime snapshots are saved in `verified/` after successful installation; model-specific dependencies should be separately locked before training.

This host had a slow IPv6 path to Python package/CDN hosts. `pip_ipv4.py` chooses IPv4 only inside the pip process, without changing machine networking. The 3.46 GB official torch wheel can be fetched with `download_torch_wheel.py`; it resumes complete range parts and checks the final official SHA256 before installation. Its URL and hash come from [PyTorch's official CUDA 12.8 index](https://download.pytorch.org/whl/cu128/torch/).

From an ordinary Windows shell, use the full Conda path or the Miniforge Prompt. Example:

```powershell
& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" run --no-capture-output -n mahjong-train python --version
```

Reproduction: copy this repository to `D:\mahjong\code`, then run `setup.ps1` in a process permitted to execute local scripts, or send its contents through the existing trusted SSH session with `ssh_ps.py`. The setup does not change the machine's persistent PowerShell execution policy. If invoking a local file from SSH, a process-scoped `-ExecutionPolicy RemoteSigned` permits local unsigned scripts; do not change system-wide policy.

The Miniforge installer is pinned to 26.9.4-0 and checked against the SHA256 from the official release asset metadata. The installer-created base environment had empty `conda-launchers.paths_data.paths`, causing `conda create` failures despite valid launcher binaries. Reinstalling that package with verification enabled repairs metadata; see [official issue and workaround](https://github.com/conda/conda-launchers/issues/35). No Conda source patch or unverified launcher was used.

## GPU environment test

```powershell
& "$env:USERPROFILE\miniforge3\Scripts\conda.exe" run --no-capture-output -n mahjong-train python D:\mahjong\code\infra\windows\gpu_smoke.py --output D:\mahjong\runs\infra-gpu-smoke
```

This runs three synthetic ResNet18 CUDA AMP optimizer steps, saves/reloads model/optimizer/scaler state, checks exact restored model output and CUDA torchvision NMS. It validates the environment; it is not a Mahjong training run or accuracy report.

Verified on 2026-10-25 in the clean environment: Python 3.11.17; torch 2.8.0+cu128; torchvision 0.23.0+cu128; RF-DETR 1.11.2; CUDA available on RTX 4070 Ti. `verified/gpu-smoke.json` records the CUDA/AMP/checkpoint/NMS checks, and `verified/rfdetr-smoke.json` records successful local COCO-pretrained Small construction and one synthetic-image GPU prediction. `verified/pip-check.txt` reports no broken requirements. These checks do not measure Mahjong accuracy.

The pretrained Small cache is `D:\mahjong\models\rf-detr-small.pth`. Set `RF_HOME=D:\mahjong\models` for training. The file was checked against the official package's expected MD5 and its SHA256 is recorded in `verified/assets.json`. The constructor smoke succeeds with `HF_HUB_OFFLINE=1`; the verified weights do not require a live Hugging Face connection.

## Jobs that survive SSH disconnection

Create a JSON file with `run_dir`, `cwd` and an `argv` array. Never put secrets in the spec or command arguments. Use a new immutable run directory for every task. Example:

```json
{
  "run_dir": "D:\\mahjong\\runs\\my-first-run",
  "cwd": "D:\\mahjong\\code",
  "argv": [
    "C:\\Users\\TRAINING_USER\\miniforge3\\Scripts\\conda.exe",
    "run", "--no-capture-output", "-n", "mahjong-train", "python", "path/to/train.py"
  ]
}
```

Start:

```powershell
powershell -NoProfile -ExecutionPolicy RemoteSigned -File D:\mahjong\code\infra\windows\start-job.ps1 -SpecPath D:\mahjong\code\my-spec.json
```

`start-job.ps1` uses local WMI to create the runner outside sshd's process tree. `job_runner.py` writes `status.json`, `runner.lock`, `stdout.log` and `stderr.log`; a completed run retains its lock to prevent accidental reruns. It executes an argument array without a shell and does not terminate unrelated processes. Training code remains responsible for periodic complete checkpoints and explicit resume support.

While a runner is active it requests temporary system-awake status through Windows `SetThreadExecutionState`; the display may still sleep, and the user's persistent power plan is unchanged. The runner records whether this request succeeded plus runner/spec SHA256 values in `status.json`, and releases the request on exit.

Read `status.json` after reconnecting. `succeeded` plus `exit_code: 0` is completion; `running`, a PID or an existing checkpoint alone is not. To resume training, create a new run spec that references the previous complete checkpoint. If Windows reboots, the job is interrupted; auto-restarting arbitrary training is deliberately not enabled.

## 当前全量训练监看（2026-10-26）

Mac终端执行：

```sh
ssh win "D:/mahjong/watch-training-v3.cmd --follow"
```

显示后台状态、epoch、loss及分类/框损失、验证mAP。按Ctrl+C只退出监看；WMI后台训练继续。去掉`--follow`只查一次。正常完成显示`state=succeeded`和`exit_code=0`，早停也属正常完成。最新checkpoint文件可能仍在写入；不要把文件刚出现当作完整可恢复，恢复前使用validate_checkpoint.py检查。

当前run为`D:\mahjong\runs\rfdetr-small960-full-v3-20261026`；原始stdout.log/stderr.log、metrics.csv、run.json、status.json及逐epoch checkpoint均保留。实际配置/观察与预计用时见[运行记录](../../docs/models/training-run-v3.json)。后续读取远端状态，不用此静态记录替代当前进度。
