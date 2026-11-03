param(
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [string]$Voice = 'Microsoft Irina Desktop'
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$fixtureDir = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($fixtureDir) | Out-Null
$referencePath = Join-Path $PSScriptRoot 'fixtures/ru-synthetic.txt'
$referenceText = [IO.File]::ReadAllText($referencePath, [Text.Encoding]::UTF8).Trim()
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speaker.SelectVoice($Voice)
    $speaker.Rate = 0
    $speaker.SetOutputToWaveFile((Join-Path $fixtureDir 'ru-synthetic.wav'))
    $speaker.Speak($referenceText)
} finally {
    $speaker.Dispose()
}
Write-Output (Join-Path $fixtureDir 'ru-synthetic.wav')
