# VoxConverse dev mini reference fixtures

The three unmodified RTTM files are from [VoxConverse v0.3](https://github.com/joonson/voxconverse),
revision `24bf60be297701cd7e4ef18550c6d390c1b87365`, `dev/`. They retain the dataset's
[Creative Commons Attribution 4.0 license](https://creativecommons.org/licenses/by/4.0/), separately
from Diarrhizer's MIT code license. Audio is downloaded locally and is not distributed in this repository.

Attribution: Joon Son Chung, Jaesung Huh, Arsha Nagrani, Triantafyllos Afouras and Andrew Zisserman,
*Spot the conversation: speaker diarisation in the wild*, Interspeech 2020.

Selection: the first three dev recording IDs alphabetically, chosen before model evaluation.
`abjxc` has one reference speaker, `afjiv` has five without overlap, `ahnss` has four with overlap.
These are complete recordings, not hand-picked successful excerpts or a representative full-dataset score.

The lock records original WAV/RTTM hashes and durations. Preparation creates full-file UEM regions,
including annotated non-speech; no reference is derived from the model under test. RTTM files use LF
to preserve upstream bytes across Windows checkouts. See [DER protocol](../../../docs/diarization-quality.md).
