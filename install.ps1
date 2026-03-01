<#
.SYNOPSIS
Installs CLIRAG (100% Offline Edge AI Engine) on Windows using PowerShell.

.DESCRIPTION
This script checks for Python, sets up a virtual environment, and pip installs CLIRAG globally 
so the `clirag` command is immediately accessible in the shell sequence.
#>

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  CLIRAG One-Click Installer (Windows)  " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Check for Python
Write-Host "`n[1/4] Checking minimum requirements..." -ForegroundColor Yellow
if (!(Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Python is not installed or not in PATH. Please install Python 3.8+." -ForegroundColor Red
    exit 1
}

$pythonVersion = python --version
Write-Host "Detected: $pythonVersion" -ForegroundColor Green

# 2. Simulate Clone (Assuming we are in the cloned dir currently, but checking)
Write-Host "`n[2/4] Verifying CLIRAG repository..." -ForegroundColor Yellow
if (!(Test-Path "setup.py")) {
    Write-Host "Warning: setup.py not found. Are you in the cloned CLIRAG directory?" -ForegroundColor Yellow
    # In a real external deploy, we would Git clone here:
    # git clone https://github.com/yourusername/clirag.git
    # cd clirag
} else {
    Write-Host "CLIRAG source validated." -ForegroundColor Green
}

# 3. Create Virtual Environment
Write-Host "`n[3/4] Creating an isolated Python Virtual Environment (venv)..." -ForegroundColor Yellow
if (!(Test-Path "env")) {
    python -m venv env
    Write-Host "Virtual environment 'env' created." -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists." -ForegroundColor Green
}

# 4. Install CLIRAG
Write-Host "`n[4/4] Installing CLIRAG dependencies and binding 'clirag' CLI command... (This may take a moment)" -ForegroundColor Yellow
# Activate and install
& .\env\Scripts\python.exe -m pip install --upgrade pip
& .\env\Scripts\python.exe -m pip install -e .

Write-Host "`n==========================================" -ForegroundColor Cyan
Write-Host "  ✅ INSTALLATION COMPLETE  " -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To use CLIRAG, you must activate the virtual environment and type 'clirag':"
Write-Host "  > .\env\Scripts\activate" -ForegroundColor White
Write-Host "  > clirag --help" -ForegroundColor White
Write-Host "  > clirag ingest .\path\to\file.pdf" -ForegroundColor White
Write-Host ""
