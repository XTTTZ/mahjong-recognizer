param(
    [string]$CondaRoot = "$env:USERPROFILE\miniforge3",
    [string]$WorkRoot = 'D:\mahjong',
    [string]$RepoRoot = 'D:\mahjong\code'
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$conda = Join-Path $CondaRoot 'Scripts\conda.exe'
if (!(Test-Path $conda)) {
    if (Test-Path $CondaRoot) { throw 'Conda directory exists without conda.exe; inspect it before installing.' }
    $installer = Join-Path $env:TEMP 'Miniforge3-26.9.4-0-Windows-x86_64.exe'
    Invoke-WebRequest -UseBasicParsing -TimeoutSec 900 -Uri 'https://github.com/conda-forge/miniforge/releases/download/26.9.4-0/Miniforge3-26.9.4-0-Windows-x86_64.exe' -OutFile $installer
    $expected = '43638698f5ee811debae27b0dc3ab5c0300129bd400a847376c4bd2f6be15ceb'
    if ((Get-FileHash $installer -Algorithm SHA256).Hash -ne $expected) { throw 'Installer SHA256 mismatch' }
    $process = Start-Process -FilePath $installer -ArgumentList @('/InstallationType=JustMe','/RegisterPython=0','/AddToPath=0','/S',('/D='+$CondaRoot)) -Wait -PassThru
    if ($process.ExitCode -ne 0) { throw ('Installer exit '+$process.ExitCode) }
}
New-Item -ItemType Directory -Force (Join-Path $WorkRoot 'code'),(Join-Path $WorkRoot 'data'),(Join-Path $WorkRoot 'runs'),(Join-Path $WorkRoot 'logs') | Out-Null
# Installer-linked conda-launchers can have empty checksum metadata although the
# binaries exist. Reinstall from the package cache with verification enabled.
# Official diagnosis/workaround: conda/conda-launchers#35.
$launcher = Get-ChildItem (Join-Path $CondaRoot 'conda-meta\conda-launchers-*.json') | Select-Object -First 1
if (!$launcher -or !((Get-Content $launcher.FullName -Raw | ConvertFrom-Json).paths_data.paths.Count)) {
    $env:CONDA_SAFETY_CHECKS = 'enabled'
    & $conda install -n base --force-reinstall conda-launchers -y
    if ($LASTEXITCODE -ne 0) { throw 'Conda launcher repair failed' }
}
$python = Join-Path $CondaRoot 'envs\mahjong-train\python.exe'
if (!(Test-Path $python)) {
    & $conda env create -f (Join-Path $RepoRoot 'infra\windows\environment.yml') -y
    if ($LASTEXITCODE -ne 0) { throw 'Conda environment creation failed' }
}
$pipRunner = Join-Path $RepoRoot 'infra\windows\pip_ipv4.py'
& $python (Join-Path $RepoRoot 'infra\windows\download_torch_wheel.py') --directory (Join-Path $WorkRoot 'cache')
if ($LASTEXITCODE -ne 0) { throw 'Official PyTorch wheel download/hash verification failed' }
& $python $pipRunner install (Join-Path $WorkRoot 'cache\torch-2.8.0+cu128-cp311-cp311-win_amd64.whl') 'torchvision==0.23.0' --index-url https://download.pytorch.org/whl/cu128 --disable-pip-version-check --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed' }
& $python $pipRunner install -r (Join-Path $RepoRoot 'infra\windows\requirements-data.txt') --disable-pip-version-check --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw 'Data dependency installation failed' }
& $python $pipRunner install -r (Join-Path $RepoRoot 'infra\windows\requirements-training.txt') --disable-pip-version-check --no-warn-script-location
if ($LASTEXITCODE -ne 0) { throw 'RF-DETR training dependency installation failed' }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency consistency check failed' }
& $python (Join-Path $RepoRoot 'infra\windows\gpu_smoke.py') --output (Join-Path $WorkRoot 'runs\infra-gpu-smoke')
if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke test failed' }
& $python (Join-Path $RepoRoot 'infra\windows\download_rfdetr_small.py') --directory (Join-Path $WorkRoot 'models')
if ($LASTEXITCODE -ne 0) { throw 'RF-DETR weight verification failed' }
$env:RF_HOME = Join-Path $WorkRoot 'models'
& $python (Join-Path $RepoRoot 'infra\windows\rfdetr_smoke.py')
if ($LASTEXITCODE -ne 0) { throw 'RF-DETR constructor/prediction smoke test failed' }
& $python (Join-Path $RepoRoot 'infra\windows\capture_environment.py') --output (Join-Path $WorkRoot 'runs\infra-environment')
if ($LASTEXITCODE -ne 0) { throw 'Environment capture/consistency check failed' }
