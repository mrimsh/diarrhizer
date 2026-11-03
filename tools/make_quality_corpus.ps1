param([Parameter(Mandatory=$true)][string]$OutputDirectory)
# [SEMANTIC-BEGIN] QUALITY:SYNTHESIZE
# @purpose: Generate non-personal speech and independent SAPI word-onset references on Windows
# @description: Requires the named installed voices; captures SpeakProgress.AudioPosition during synthesis.
# @inputs: corpus-spec.json, output directory, Windows PowerShell 5.1/System.Speech
# @outputs: mono PCM16/16kHz units and JSON voice/text/word markers
# @sideEffects: Writes generated units; uses local speech engines only
# @errors: Stops on missing voice or synthesis/file failure
# @see: QUALITY:CORPUS
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
Add-Type -ReferencedAssemblies System.Speech -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Speech.Synthesis;
using System.Speech.AudioFormat;
public class WordOnset {
    public string word;
    public double start;
}
public class QualitySpeech {
    public static WordOnset[] Render(string voice, string text, string path) {
        var marks = new List<WordOnset>();
        using (var synth = new SpeechSynthesizer()) {
            synth.SelectVoice(voice);
            synth.Rate = 0;
            synth.SpeakProgress += (sender, e) => {
                lock(marks) { marks.Add(new WordOnset { word=e.Text, start=e.AudioPosition.TotalSeconds }); }
            };
            synth.SetOutputToWaveFile(path, new SpeechAudioFormatInfo(16000, AudioBitsPerSample.Sixteen, AudioChannel.Mono));
            synth.Speak(text);
        }
        return marks.ToArray();
    }
}
'@
$corpusDir = [IO.Path]::GetFullPath($OutputDirectory)
$unitDir = Join-Path $corpusDir 'units'
[IO.Directory]::CreateDirectory($unitDir) | Out-Null
$spec = [IO.File]::ReadAllText((Join-Path $PSScriptRoot 'fixtures/corpus-spec.json'), [Text.Encoding]::UTF8) | ConvertFrom-Json
$speechInventory = New-Object System.Speech.Synthesis.SpeechSynthesizer
try { $available = $speechInventory.GetInstalledVoices().VoiceInfo.Name }
finally { $speechInventory.Dispose() }
foreach ($unit in $spec.units) {
    if ($unit.voice -notin $available) { throw "Required voice is not installed: $($unit.voice)" }
}
foreach ($unit in $spec.units) {
    $marks = [QualitySpeech]::Render($unit.voice, $unit.text, (Join-Path $unitDir ($unit.id + '.wav')))
    $metadata = @{voice=$unit.voice; rate=0; sample_rate=16000; text=$unit.text; words=@($marks)}
    [IO.File]::WriteAllText((Join-Path $unitDir ($unit.id + '.json')), ($metadata | ConvertTo-Json -Depth 8), (New-Object Text.UTF8Encoding($false)))
}
Write-Output "Speech units and word-onset references: $unitDir"
# [SEMANTIC-END] QUALITY:SYNTHESIZE
