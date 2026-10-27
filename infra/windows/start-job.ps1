param(
    [Parameter(Mandatory=$true)][string]$SpecPath,
    [string]$Python = "$env:USERPROFILE\miniforge3\envs\mahjong-train\python.exe",
    [string]$Runner = 'D:\mahjong\code\infra\windows\job_runner.py'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
if (!(Test-Path $Python) -or !(Test-Path $Runner) -or !(Test-Path $SpecPath)) { throw 'Missing Python, runner or spec' }
$spec = Get-Content $SpecPath -Raw | ConvertFrom-Json
if ($spec.kind -eq 'offline_experiment_plan' -or $spec.launchable -eq $false) { throw 'Offline experiment plan is not a launchable job. Complete its post-boot preflight first.' }
if (!(Test-Path $spec.cwd)) { throw 'Job working directory is missing' }
if (Test-Path (Join-Path $spec.run_dir 'runner.lock')) { throw 'Run ID already exists. Choose a fresh run directory.' }
# WMI creates the process outside sshd's process tree/job object. The durable
# status file and short disconnect test verify the behavior on this host.
$command = '"' + $Python + '" "' + $Runner + '" --spec "' + $SpecPath + '"'
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$command}
if ($result.ReturnValue -ne 0) { throw ('Process creation failed: ' + $result.ReturnValue) }
[pscustomobject]@{process_id=$result.ProcessId; run_dir=$spec.run_dir; status=(Join-Path $spec.run_dir 'status.json')} | ConvertTo-Json -Compress
