# Run-UltraOptimiser.ps1
# Main runner script for UltraOptimiser

param(
    [Parameter()]
    [ValidateSet("test", "optimize", "setup")]
    [string]$Action = "test"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "===== UltraOptimiser Runner =====" -ForegroundColor Cyan
Write-Host "Action: $Action" -ForegroundColor Yellow

switch ($Action) {
    "setup" {
        Write-Host ""
        Write-Host "Setting up environment..." -ForegroundColor Yellow
        & ".\scripts\Setup-Environment.ps1"
    }
    
    "test" {
        Write-Host ""
        Write-Host "Running tests..." -ForegroundColor Yellow
        
        # Check if venv exists
        if (Test-Path ".\venv\Scripts\python.exe") {
            & ".\venv\Scripts\python.exe" ".\tests\test_optimizer.py"
        } else {
            Write-Host "Virtual environment not found. Run with -Action setup first" -ForegroundColor Red
            exit 1
        }
    }
    
    "optimize" {
        Write-Host ""
        Write-Host "Starting optimization engine..." -ForegroundColor Yellow
        
        # Check if venv exists
        if (Test-Path ".\venv\Scripts\python.exe") {
            & ".\venv\Scripts\python.exe" -c "from core.optimizer import UltraOptimiser; print('UltraOptimiser ready for optimization')"
        } else {
            Write-Host "Virtual environment not found. Run with -Action setup first" -ForegroundColor Red
            exit 1
        }
    }
}

Write-Host ""
Write-Host "Operation completed" -ForegroundColor Green
