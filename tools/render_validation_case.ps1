param(
    [Parameter(Mandatory)] [string]$Name,
    [Parameter(Mandatory)] [string]$Source,
    [Parameter(Mandatory)] [string]$Palette,
    [ValidateSet("bridge", "onnx")] [string]$Backend = "bridge",
    [string]$ModelDir = "listening\final-validation\dac-onnx",
    [double]$Temperature = 0.47,
    [double]$Threshold = 0.99,
    [double]$Continuity = 0.10,
    [double]$RvqFocus = 0.30,
    [double]$PaletteBias = 1.0,
    [int]$GrainSize = 7,
    [int]$GrainStep = 2,
    [ValidateRange(0, 2)] [int]$SwapMode = 2,
    [ValidateRange(0, 1)] [int]$ProcessingMode = 0,
    [double]$Envelope = 0.05,
    [double]$DryWet = 0.9,
    [double]$OutputGain = -3.0,
    [int]$TimeoutSeconds = 480
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$sourcePath = (Resolve-Path (Join-Path $root $Source)).Path
$palettePaths = ($Palette -split ';' | ForEach-Object { (Resolve-Path (Join-Path $root $_)).Path }) -join ';'
$outputDirectory = Join-Path $root "listening\final-validation\renders\$Backend"
$outputPath = Join-Path $outputDirectory "$Name.wav"
$statusPath = Join-Path $outputDirectory "$Name.status.txt"
$executable = Join-Path $root "build-$Backend\NeuralMorphing_artefacts\Release\Standalone\Neural Morphing.exe"
New-Item -ItemType Directory -Force $outputDirectory | Out-Null
Remove-Item $outputPath, $statusPath -Force -ErrorAction SilentlyContinue

$env:NEURAL_MORPHING_DEMO_SOURCE_FILE = $sourcePath
$env:NEURAL_MORPHING_DEMO_PALETTE_FILES = $palettePaths
$env:NEURAL_MORPHING_DEMO_STATUS_FILE = $statusPath
$env:NEURAL_MORPHING_DEMO_RENDER_FILE = $outputPath
$env:NEURAL_MORPHING_DEMO_DETERMINISTIC = "1"
$env:NEURAL_MORPHING_DEMO_BACKEND = if ($Backend -eq "bridge") { "1" } else { "2" }
if ($Backend -eq "onnx") {
    $env:NEURAL_MORPHING_MODEL_DIR = (Resolve-Path (Join-Path $root $ModelDir)).Path
}
$env:NEURAL_MORPHING_DEMO_PROCESSING_MODE = "$ProcessingMode"
$env:NEURAL_MORPHING_DEMO_BRIDGE_CODEC = "0"
$env:NEURAL_MORPHING_DEMO_SWAP_MODE = "$SwapMode"
$env:NEURAL_MORPHING_DEMO_TEMPERATURE = "$Temperature"
$env:NEURAL_MORPHING_DEMO_THRESHOLD = "$Threshold"
$env:NEURAL_MORPHING_DEMO_CONTINUITY = "$Continuity"
$env:NEURAL_MORPHING_DEMO_RVQ_FOCUS = "$RvqFocus"
$env:NEURAL_MORPHING_DEMO_PALETTE_BIAS = "$PaletteBias"
$env:NEURAL_MORPHING_DEMO_GRAIN_SIZE = "$GrainSize"
$env:NEURAL_MORPHING_DEMO_GRAIN_STEP = "$GrainStep"
$env:NEURAL_MORPHING_DEMO_ENVELOPE = "$Envelope"
$env:NEURAL_MORPHING_DEMO_DRY_WET = "$DryWet"
$env:NEURAL_MORPHING_DEMO_OUTPUT_GAIN = "$OutputGain"

$process = Start-Process $executable -WindowStyle Hidden -PassThru
$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
$status = ""
try {
    do {
        Start-Sleep -Milliseconds 500
        try { $status = Get-Content $statusPath -Raw -ErrorAction Stop } catch { $status = "" }
        if ($status -match "render=(ready|error:)") { break }
        $process.Refresh()
    } while ((Get-Date) -lt $deadline -and !$process.HasExited)
} finally {
    if (!$process.HasExited) { Stop-Process -Id $process.Id -Force }
}

if ($status -notmatch "render=ready" -or !(Test-Path $outputPath) -or ($Backend -eq "onnx" -and $status -notmatch "native \| native")) {
    throw "Validation render failed: $status"
}

[pscustomobject]@{ Name = $Name; Backend = $Backend; Output = $outputPath; Status = $status.Trim() }
