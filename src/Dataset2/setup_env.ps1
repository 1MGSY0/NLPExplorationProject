param(
    [switch]$UseCUDA = $false,
    [string]$CudaVariant = "cu121"  # Options include cu118, cu121, etc.
)

Write-Host "[setup_env] Starting setup for Dataset2 environment..." -ForegroundColor Cyan

# Ensure we're running in the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
Write-Host "[setup_env] Working directory: $ScriptDir"

# Detect NVIDIA GPU (best-effort) if UseCUDA not explicitly set
if (-not $PSBoundParameters.ContainsKey('UseCUDA')) {
    $gpuIsNvidia = $false
    try {
        $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($smi) { $gpuIsNvidia = $true }
        else {
            $gpuName = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ", "
            if ($gpuName -match "NVIDIA") { $gpuIsNvidia = $true }
        }
    } catch {}
    if ($gpuIsNvidia) {
        $UseCUDA = $true
        Write-Host "[setup_env] NVIDIA GPU detected. Will install CUDA wheels (variant: $CudaVariant)." -ForegroundColor Green
    } else {
        Write-Host "[setup_env] No NVIDIA GPU detected. Installing CPU wheels." -ForegroundColor Yellow
    }
}

# Create venv if missing
if (-not (Test-Path ".venv")) {
    Write-Host "[setup_env] Creating virtual environment in .venv ..."
    python -m venv .venv
}

# Activate venv (dot-source so it affects current process)
$activate = Join-Path ".venv" "Scripts/Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Error "[setup_env] Activation script not found at $activate"
    exit 1
}
. $activate
Write-Host "[setup_env] Activated virtual environment: $((Get-Command python).Source)"

# Upgrade pip
python -m pip install --upgrade pip

# Base requirements install
if (-not (Test-Path "requirements.txt")) {
    Write-Error "[setup_env] requirements.txt not found in $ScriptDir"
    exit 1
}

Write-Host "[setup_env] Installing base requirements from requirements.txt ..."
pip install -r requirements.txt

# If CUDA requested/detected, install CUDA wheels for PyTorch
if ($UseCUDA) {
    Write-Host "[setup_env] Installing CUDA wheels for PyTorch from $CudaVariant channel ..." -ForegroundColor Green
    try {
        pip uninstall -y torch torchvision torchaudio | Out-Null
    } catch {}
    $indexUrl = "https://download.pytorch.org/whl/$CudaVariant"
    pip install torch==2.2.0 torchvision torchaudio --index-url $indexUrl
}

# Torch/CUDA sanity check
Write-Host "[setup_env] Validating Torch + CUDA availability ..." -ForegroundColor Cyan
python -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

Write-Host "[setup_env] Setup completed." -ForegroundColor Cyan
