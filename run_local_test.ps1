param(
    [string]$PythonExe = "",
    [switch]$SkipUnitTests
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Wheel = Join-Path $Root "dist\spatial_support_audit-0.1.0-py3-none-any.whl"
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $Root "build"))
$InstallDir = [System.IO.Path]::GetFullPath((Join-Path $BuildRoot "local-test-install"))
$InputCsv = Join-Path $Root "examples\example_support_counts.csv"
$OutputCsv = Join-Path $Root "examples\local_test_output.csv"

if (-not (Test-Path -LiteralPath $Wheel)) {
    throw "Built wheel not found: $Wheel"
}

$Python = $null
$PythonArgs = @()
if ($PythonExe) {
    if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
        throw "Requested Python executable not found: $PythonExe"
    }
    $Python = [System.IO.Path]::GetFullPath($PythonExe)
} else {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($PyLauncher) {
        $Python = $PyLauncher.Source
        $PythonArgs = @("-3")
    } else {
        $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($PythonCommand -and $PythonCommand.Source -notmatch "WindowsApps") {
            $Python = $PythonCommand.Source
        }
    }
    if (-not $Python) {
        $CandidatePatterns = @()
        if ($env:VIRTUAL_ENV) { $CandidatePatterns += Join-Path $env:VIRTUAL_ENV "Scripts\python.exe" }
        if ($env:CONDA_PREFIX) { $CandidatePatterns += Join-Path $env:CONDA_PREFIX "python.exe" }
        if ($env:LOCALAPPDATA) { $CandidatePatterns += Join-Path $env:LOCALAPPDATA "Programs\Python\Python*\python.exe" }
        if ($env:USERPROFILE) { $CandidatePatterns += Join-Path $env:USERPROFILE ".cache\*\*\dependencies\python\python.exe" }
        foreach ($Pattern in $CandidatePatterns) {
            $Candidate = Get-Item -Path $Pattern -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($Candidate) {
                $Python = $Candidate.FullName
                break
            }
        }
    }
}

if (-not $Python) {
    throw "Python 3.10 or newer was not found. Install Python, then run this script again."
}

Write-Host "[1/5] Checking Python"
& $Python @PythonArgs -c "import sys; assert sys.version_info >= (3, 10), sys.version; print(sys.version)"
if ($LASTEXITCODE -ne 0) {
    throw "Python version check failed."
}

Write-Host "[2/5] Preparing isolated test directory"
$ExpectedPrefix = $BuildRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $InstallDir.StartsWith($ExpectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to prepare a test directory outside build/: $InstallDir"
}
if (Test-Path -LiteralPath $InstallDir) {
    Remove-Item -LiteralPath $InstallDir -Recurse -Force
}
New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null

Write-Host "[3/5] Installing the local wheel (offline)"
& $Python @PythonArgs -c "import pathlib, sys, zipfile; wheel=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2]); zipfile.ZipFile(wheel).extractall(target); print(f'Installed {wheel.name} into {target}')" $Wheel $InstallDir
if ($LASTEXITCODE -ne 0) {
    throw "Local wheel installation failed."
}

$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $InstallDir

try {
    if (-not $SkipUnitTests) {
        Write-Host "[4/5] Running unit tests against the installed wheel"
        Push-Location $Root
        try {
            & $Python @PythonArgs -m unittest discover -s tests -v
            if ($LASTEXITCODE -ne 0) {
                throw "Unit tests failed."
            }
        } finally {
            Pop-Location
        }
    } else {
        Write-Host "[4/5] Unit tests skipped by request"
    }

    Write-Host "[5/5] Running the example CSV"
    & $Python @PythonArgs -m spatial_support_audit.cli $InputCsv `
        --output $OutputCsv `
        --group-column species `
        --minimum-stable-cells 1
    if ($LASTEXITCODE -ne 0) {
        throw "Example command-line run failed."
    }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}

Write-Host ""
Write-Host "Local test completed successfully."
Write-Host "Result file: $OutputCsv"
Import-Csv -LiteralPath $OutputCsv | Format-Table -AutoSize
