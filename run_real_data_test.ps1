param(
    [string]$Archive = "",
    [string]$PythonExe = "",
    [switch]$VerifyArchiveHash
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Workspace = Split-Path -Parent (Split-Path -Parent $Root)
$BuildRoot = [System.IO.Path]::GetFullPath((Join-Path $Root "build"))
$ToolDir = [System.IO.Path]::GetFullPath((Join-Path $BuildRoot "real-data-tools"))
$PyArrowWheel = Join-Path $BuildRoot "downloads\pyarrow-24.0.0-cp312-cp312-win_amd64.whl"
$OutputDir = Join-Path $Root "real_data"
$InputCsv = Join-Path $OutputDir "three_state_support_counts.csv"
$DiagnosticsCsv = Join-Path $OutputDir "three_state_support_diagnostics.csv"
$ReferenceCsv = Join-Path $OutputDir "locked_reference_cell_jaccard.csv"
$ValidationJson = Join-Path $OutputDir "validation_report.json"

if (-not $Archive) {
    $ArchiveName = "20260712_full_reproduction_output.tar"
    $ArchiveCandidates = @()
    $RootCandidate = Join-Path $Workspace $ArchiveName
    if (Test-Path -LiteralPath $RootCandidate -PathType Leaf) {
        $ArchiveCandidates += Get-Item -LiteralPath $RootCandidate
    }
    foreach ($Directory in Get-ChildItem -LiteralPath $Workspace -Directory) {
        $Candidate = Join-Path $Directory.FullName $ArchiveName
        if (Test-Path -LiteralPath $Candidate -PathType Leaf) {
            $ArchiveCandidates += Get-Item -LiteralPath $Candidate
        }
    }
    if ($ArchiveCandidates.Count -ne 1) {
        throw "Expected exactly one 20260712_full_reproduction_output.tar under $Workspace; found $($ArchiveCandidates.Count)."
    }
    $Archive = $ArchiveCandidates[0].FullName
}
$Archive = [System.IO.Path]::GetFullPath($Archive)
if (-not (Test-Path -LiteralPath $Archive -PathType Leaf)) {
    throw "Final reproduction archive not found: $Archive"
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
    throw "Python 3.10 or newer was not found."
}

Write-Host "[1/6] Checking Python"
& $Python @PythonArgs -c "import sys; assert sys.version_info >= (3, 10), sys.version; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Python version check failed." }

$ExpectedPrefix = $BuildRoot.TrimEnd([System.IO.Path]::DirectorySeparatorChar) + [System.IO.Path]::DirectorySeparatorChar
if (-not $ToolDir.StartsWith($ExpectedPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to prepare a dependency directory outside build/: $ToolDir"
}
New-Item -ItemType Directory -Path $ToolDir -Force | Out-Null

$PreviousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = $ToolDir
$HasExactPyArrow = $false
try {
    & $Python @PythonArgs -c "import pyarrow; raise SystemExit(0 if pyarrow.__version__ == '24.0.0' else 1)" 2>$null
    $HasExactPyArrow = ($LASTEXITCODE -eq 0)
} catch {
    $HasExactPyArrow = $false
}

if (-not $HasExactPyArrow) {
    Write-Host "[2/6] Installing the frozen PyArrow wheel offline"
    if (-not (Test-Path -LiteralPath $PyArrowWheel -PathType Leaf)) {
        throw "Required local wheel not found: $PyArrowWheel"
    }
    & $Python @PythonArgs -c "import pathlib,sys,zipfile; wheel=pathlib.Path(sys.argv[1]); target=pathlib.Path(sys.argv[2]); zipfile.ZipFile(wheel).extractall(target); print(f'Installed {wheel.name}')" $PyArrowWheel $ToolDir
    if ($LASTEXITCODE -ne 0) { throw "PyArrow wheel extraction failed." }
} else {
    Write-Host "[2/6] Reusing PyArrow 24.0.0 from the isolated build directory"
}

$env:PYTHONPATH = $ToolDir + [System.IO.Path]::PathSeparator + (Join-Path $Root "src")
try {
    Write-Host "[3/6] Checking data-conversion dependencies"
    & $Python @PythonArgs -c "import pandas, numpy, pyarrow; print(f'pandas={pandas.__version__}, numpy={numpy.__version__}, pyarrow={pyarrow.__version__}')"
    if ($LASTEXITCODE -ne 0) { throw "Data-conversion dependency check failed." }

    Write-Host "[4/6] Building the aggregated input from the final TAR"
    $BuilderArguments = @(
        (Join-Path $Root "utilities\build_real_three_state_input.py"),
        "--archive", $Archive,
        "--output-dir", $OutputDir
    )
    if ($VerifyArchiveHash) {
        $BuilderArguments += "--verify-archive-hash"
    }
    & $Python @PythonArgs @BuilderArguments
    if ($LASTEXITCODE -ne 0) { throw "Real-data conversion failed." }

    Write-Host "[5/6] Running spatial-support-audit on all 43 groups"
    & $Python @PythonArgs -m spatial_support_audit.cli $InputCsv `
        --output $DiagnosticsCsv `
        --group-column analysis_group `
        --minimum-effort 10 `
        --minimum-stable-cells 3
    if ($LASTEXITCODE -ne 0) { throw "Real-data diagnostic run failed." }

    Write-Host "[6/6] Comparing species Jaccard values with the locked manuscript table"
    & $Python @PythonArgs (Join-Path $Root "utilities\validate_real_three_state_output.py") `
        --reference $ReferenceCsv `
        --diagnostics $DiagnosticsCsv `
        --report $ValidationJson
    if ($LASTEXITCODE -ne 0) { throw "Locked Jaccard comparison failed." }
} finally {
    $env:PYTHONPATH = $PreviousPythonPath
}

Write-Host ""
Write-Host "Real-data test completed successfully."
Write-Host "Input:       $InputCsv"
Write-Host "Diagnostics: $DiagnosticsCsv"
Write-Host "Validation:  $ValidationJson"
Import-Csv -LiteralPath (Join-Path $OutputDir "conversion_summary.csv") |
    Select-Object state, evaluation_species, thinned_rows, input_rows, effort_jaccard, stable_cells_at_effort_10 |
    Format-Table -AutoSize
