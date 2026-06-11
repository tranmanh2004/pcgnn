Write-Host "Starting PCGNN Web..." -ForegroundColor Cyan

$backendPath = "$PSScriptRoot\backend"
$frontendPath = "$PSScriptRoot\frontend"

# Start backend
$backendCommand = @"
`$conda = Get-Command conda -ErrorAction SilentlyContinue
if (-not `$conda) {
    Write-Host 'Conda was not found in PATH. Please install Conda or open this from an initialized Conda shell.' -ForegroundColor Red
    exit 1
}

(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate pcgnn
if (`$LASTEXITCODE -ne 0) {
    Write-Host 'Failed to activate conda environment: pcgnn' -ForegroundColor Red
    exit `$LASTEXITCODE
}

Set-Location -LiteralPath '$backendPath'
uvicorn main:app --reload --host 127.0.0.1 --port 8765
"@

Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-Command", $backendCommand

# Start frontend
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendPath'; npm run dev"

Write-Host "Backend: http://127.0.0.1:8765" -ForegroundColor Green
Write-Host "Frontend: http://localhost:5173" -ForegroundColor Green

Start-Sleep 3
Start-Process "http://localhost:5173"
