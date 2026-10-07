$ErrorActionPreference = "Stop"

$scriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDirectory
$python = Join-Path $scriptDirectory ".venv\Scripts\python.exe"
$entryPoint = Join-Path $scriptDirectory "nnmf_full_fixed_jh_V3.py"

if (-not (Test-Path -LiteralPath $python)) {
    throw "Build environment not found. Create Scripts\.venv and install requirements-build.txt first."
}

if (-not (Test-Path -LiteralPath $entryPoint)) {
    throw "Application entry point not found: $entryPoint"
}

Push-Location $projectRoot
try {
    & $python -m PyInstaller `
        --clean `
        --onedir `
        --console `
        --name "NNMF-Analysis" `
        --distpath (Join-Path $projectRoot "dist") `
        --workpath (Join-Path $projectRoot "build") `
        --specpath (Join-Path $projectRoot "build") `
        $entryPoint

    if ($LASTEXITCODE -ne 0) {
        throw "PyInstaller failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}

Write-Host "Build complete: $(Join-Path $projectRoot 'dist\NNMF-Analysis\NNMF-Analysis.exe')"
