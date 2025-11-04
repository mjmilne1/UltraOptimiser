# Setup-Environment.ps1
# Run this to set up Python environment

Write-Host 'Setting up Python environment for UltraOptimiser...' -ForegroundColor Yellow

# Check if Python is installed
if (Get-Command python -ErrorAction SilentlyContinue) {
    Write-Host '✅ Python found' -ForegroundColor Green
    python --version
} else {
    Write-Host '❌ Python not found. Please install Python 3.9+ first' -ForegroundColor Red
    exit 1
}

# Create virtual environment
Write-Host 'Creating virtual environment...' -ForegroundColor Yellow
python -m venv venv

# Activate virtual environment
Write-Host 'Activating virtual environment...' -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install packages
Write-Host 'Installing required packages...' -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host '✅ Environment setup complete!' -ForegroundColor Green
Write-Host 'To activate the environment, run: .\venv\Scripts\Activate.ps1' -ForegroundColor Cyan
