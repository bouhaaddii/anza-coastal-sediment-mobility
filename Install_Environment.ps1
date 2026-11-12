$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $ProjectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { py -3 -m venv .venv }
& '.venv\Scripts\python.exe' -m pip install --upgrade pip
& '.venv\Scripts\python.exe' -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed.' }
Write-Host 'Environment ready.'
