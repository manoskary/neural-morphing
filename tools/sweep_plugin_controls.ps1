param(
    [string]$Executable = "build-bridge\NeuralMorphing_artefacts\Release\Standalone\Neural Morphing.exe",
    [string]$Palette = "examples\demo_percussion_texture.wav",
    [string]$Source = "examples\demo_synth_pulse.wav",
    [int]$TimeoutSeconds = 240
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$executablePath = (Resolve-Path (Join-Path $root $Executable)).Path
$palettePath = (Resolve-Path (Join-Path $root $Palette)).Path
$sourcePath = (Resolve-Path (Join-Path $root $Source)).Path
$logDirectory = Join-Path $root "logs\control-sweep"
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

$defaults = @{
    NEURAL_MORPHING_DEMO_TEMPERATURE = "0.47"
    NEURAL_MORPHING_DEMO_THRESHOLD = "0.99"
    NEURAL_MORPHING_DEMO_CONTINUITY = "0.10"
    NEURAL_MORPHING_DEMO_RVQ_FOCUS = "0.30"
    NEURAL_MORPHING_DEMO_PALETTE_BIAS = "1.0"
    NEURAL_MORPHING_DEMO_GRAIN_SIZE = "7"
    NEURAL_MORPHING_DEMO_GRAIN_STEP = "2"
    NEURAL_MORPHING_DEMO_ENVELOPE = "0.05"
    NEURAL_MORPHING_DEMO_DRY_WET = "0.9"
}

$cases = @(
    @{ Name = "baseline"; Values = @{} },
    @{ Name = "temperature_high"; Values = @{ NEURAL_MORPHING_DEMO_TEMPERATURE = "2.0" } },
    @{ Name = "threshold_low"; Values = @{ NEURAL_MORPHING_DEMO_THRESHOLD = "0.15" } },
    @{ Name = "continuity_high"; Values = @{ NEURAL_MORPHING_DEMO_CONTINUITY = "1.0" } },
    @{ Name = "rvq_focus_high"; Values = @{ NEURAL_MORPHING_DEMO_RVQ_FOCUS = "1.0" } },
    @{ Name = "palette_bias_low"; Values = @{ NEURAL_MORPHING_DEMO_PALETTE_BIAS = "0.0" } },
    @{ Name = "small_grains"; Values = @{ NEURAL_MORPHING_DEMO_GRAIN_SIZE = "2"; NEURAL_MORPHING_DEMO_GRAIN_STEP = "1" } },
    @{ Name = "envelope_full"; Values = @{ NEURAL_MORPHING_DEMO_ENVELOPE = "1.0" } },
    @{ Name = "half_wet"; Values = @{ NEURAL_MORPHING_DEMO_DRY_WET = "0.5" } }
)

$env:NEURAL_MORPHING_DEMO_PALETTE_FILES = $palettePath
$env:NEURAL_MORPHING_DEMO_SOURCE_FILE = $sourcePath
$env:NEURAL_MORPHING_DEMO_DETERMINISTIC = "1"
$results = foreach ($case in $cases) {
    foreach ($entry in $defaults.GetEnumerator()) {
        Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    }
    foreach ($entry in $case.Values.GetEnumerator()) {
        Set-Item -Path "Env:$($entry.Key)" -Value $entry.Value
    }

    $statusPath = Join-Path $logDirectory "$($case.Name).txt"
    Remove-Item -LiteralPath $statusPath -Force -ErrorAction SilentlyContinue
    $env:NEURAL_MORPHING_DEMO_STATUS_FILE = $statusPath
    $process = Start-Process -FilePath $executablePath -WindowStyle Hidden -PassThru
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $status = ""

    try {
        while ((Get-Date) -lt $deadline -and !$process.HasExited) {
            try {
                $status = Get-Content -LiteralPath $statusPath -Raw -ErrorAction Stop
            } catch {
                $status = ""
            }
            if ($status -match "wet=(ready|repeat)" -and $status -match "sig=[0-9a-f]+") {
                break
            }
            Start-Sleep -Milliseconds 500
            $process.Refresh()
        }
    } finally {
        if (!$process.HasExited) {
            Stop-Process -Id $process.Id -Force
        }
    }

    Start-Sleep -Milliseconds 200
    if (Test-Path -LiteralPath $statusPath) {
        $status = Get-Content -LiteralPath $statusPath -Raw
    }
    [pscustomobject]@{ Case = $case.Name; Status = $status.Trim() }
}

$results | Format-Table -AutoSize
if ($results.Where({ $_.Status -notmatch "wet=(ready|repeat)" }).Count -gt 0) {
    throw "One or more control sweeps did not produce wet audio."
}

function Get-WetFingerprint([string]$status) {
    if ($status -match "sig=([0-9a-f]+)") {
        return $Matches[1]
    }
    return ""
}

$baselineFingerprint = Get-WetFingerprint $results[0].Status
foreach ($name in @("temperature_high", "threshold_low", "continuity_high", "rvq_focus_high", "palette_bias_low", "small_grains")) {
    $case = $results.Where({ $_.Case -eq $name })[0]
    if ((Get-WetFingerprint $case.Status) -eq $baselineFingerprint) {
        throw "$name did not change the wet morph fingerprint."
    }
}
foreach ($name in @("envelope_full", "half_wet")) {
    $case = $results.Where({ $_.Case -eq $name })[0]
    if ((Get-WetFingerprint $case.Status) -ne $baselineFingerprint) {
        throw "$name unexpectedly changed the wet morph fingerprint."
    }
}
