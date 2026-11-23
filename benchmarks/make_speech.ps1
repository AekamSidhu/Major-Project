# Speaks the benchmark's interview questions and answers with Windows text-to-speech.
#
#   powershell -ExecutionPolicy Bypass -File benchmarks\make_speech.ps1 -Out M:\openmeet-data\bench-speech
#
# Writes 16 kHz mono 16-bit WAV files (the pipeline's format): <id>_question.wav
# and <id>_answer.wav for every pair in data\interview_pairs.json, and
# <id>_llm_answer.wav for every answer in data\llm_answers.json. Synthetic
# voices only; nothing is recorded from a microphone.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Rate = 0
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
New-Item -ItemType Directory -Force -Path $Out | Out-Null
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
$voices = (New-Object System.Speech.Synthesis.SpeechSynthesizer).GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }

function Speak([string]$text, [string]$path, [string]$voice) {
    $s = New-Object System.Speech.Synthesis.SpeechSynthesizer
    if ($voice) { $s.SelectVoice($voice) }
    $s.Rate = $Rate
    $s.SetOutputToWaveFile($path, $format)
    $s.Speak($text)
    $s.Dispose()
}

# Two voices when installed: one for the interviewer, one for the candidate.
$interviewer = $voices | Select-Object -First 1
$candidate = ($voices | Select-Object -Last 1)
$pairs = (Get-Content -Raw -Encoding UTF8 (Join-Path $here 'data\interview_pairs.json') | ConvertFrom-Json).pairs
foreach ($p in $pairs) {
    Speak $p.question (Join-Path $Out "$($p.id)_question.wav") $interviewer
    Speak $p.answer (Join-Path $Out "$($p.id)_answer.wav") $candidate
}
$llm = Join-Path $here 'data\llm_answers.json'
if (Test-Path $llm) {
    foreach ($a in (Get-Content -Raw -Encoding UTF8 $llm | ConvertFrom-Json).answers) {
        Speak $a.answer (Join-Path $Out "$($a.id)_llm_answer.wav") $candidate
    }
}
Write-Output "voices: interviewer '$interviewer', candidate '$candidate'; files in $Out"
