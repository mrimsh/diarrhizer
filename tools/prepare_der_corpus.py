"""Prepare the pinned VoxConverse dev mini corpus from local WAVs or bounded public ZIP downloads."""
import argparse
import hashlib
import io
import json
import shutil
import struct
import urllib.request
import wave
import zipfile
import zlib
from pathlib import Path

from diarrhizer.utils import write_json_atomic

FIXTURES = Path(__file__).parent / "fixtures" / "voxconverse-dev-mini"


# [SEMANTIC-BEGIN] QUALITY:DER_CORPUS
# @purpose: Prepare a pinned human-speech corpus with independent RTTM references and full-file UEM
# @description: First three alphabetical VoxConverse dev IDs; original WAV/RTTM hashes are mandatory.
#   Optional download reads bounded ZIP ranges from the public source, never the entire 2 GB archive.
# @inputs: pinned corpus-lock.json/RTTM, local WAV directory or explicit --download
# @outputs: local audio, reference/UEM and hash-verified speaker_diarization manifest
# @sideEffects: Reads/copies local files; explicit downloads use public HTTP without credentials
# @errors: ValueError on changed source bytes/format or unsupported range/ZIP response
# @see: QUALITY:DIARIZATION_REFERENCE, QUALITY:DER_SUITE
def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_range(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"})
    with urllib.request.urlopen(request, timeout=40) as response:
        if response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {start}-{end}/"):
            raise ValueError("Server must honor bounded byte ranges; download/extract the archive manually instead")
        content = response.read(end - start + 2)
    if len(content) != end - start + 1:
        raise ValueError("Incomplete or oversized ZIP range")
    return content


def download_audio(lock, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    needed = [c for c in lock["cases"] if not (destination / (c["id"] + ".wav")).exists()
              or digest(destination / (c["id"] + ".wav")) != c["audio_sha256"]]
    if not needed:
        return
    url = lock["audio_archive"]
    with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=40) as response:
        size = int(response.headers["Content-Length"])
    start = max(0, size - 65557)  # EOCD plus maximum ZIP comment; pinned archive central directory fits.
    with zipfile.ZipFile(io.BytesIO(read_range(url, start, size - 1))) as archive:
        for case in needed:
            matches = [i for i in archive.infolist() if Path(i.filename).name == case["id"] + ".wav"]
            if len(matches) != 1:
                raise ValueError(f"Expected one ZIP member for {case['id']}")
            info = matches[0]
            if (info.file_size != case["audio_bytes"] or info.compress_size > info.file_size + 65536
                    or info.compress_type not in (0, 8) or info.flag_bits & 1):
                raise ValueError("Unexpected ZIP member size/compression/encryption")
            # zipfile offsets are relative to the downloaded tail; restore the absolute archive offset.
            offset = info.header_offset + start
            header = struct.unpack("<4s5H3I2H", read_range(url, offset, offset + 29))
            if header[0] != b"PK\x03\x04":
                raise ValueError("Invalid ZIP local header")
            data_start = offset + 30 + header[-2] + header[-1]
            compressed = read_range(url, data_start, data_start + info.compress_size - 1)
            if info.compress_type == 8:
                content = zlib.decompressobj(-15).decompress(compressed, info.file_size + 1)
            else:
                content = compressed
            if (len(content) != info.file_size or zlib.crc32(content) != info.CRC
                    or hashlib.sha256(content).hexdigest() != case["audio_sha256"]):
                raise ValueError(f"Downloaded audio does not match pinned bytes: {case['id']}")
            (destination / (case["id"] + ".wav")).write_bytes(content)


def prepare(audio_dir, output, fixtures=FIXTURES):
    audio_dir, output, fixtures = Path(audio_dir), Path(output), Path(fixtures)
    lock = json.loads((fixtures / "corpus-lock.json").read_text(encoding="utf-8"))
    for case in lock["cases"]:  # Validate every source before publishing any prepared case.
        audio, reference = audio_dir / (case["id"] + ".wav"), fixtures / (case["id"] + ".rttm")
        if digest(audio) != case["audio_sha256"] or digest(reference) != case["reference_sha256"]:
            raise ValueError(f"Pinned corpus hash mismatch: {case['id']}")
        with wave.open(str(audio)) as stream:
            if ((stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1, 2, 16000)
                    or abs(stream.getnframes() / stream.getframerate() - case["duration_seconds"]) > 1 / 16000):
                raise ValueError(f"Unexpected audio format/duration: {case['id']}")
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    for case in lock["cases"]:
        uri = case["id"]
        audio, reference, uem = output / f"{uri}.wav", output / f"{uri}.rttm", output / f"{uri}.uem"
        for source, destination in ((audio_dir / audio.name, audio), (fixtures / reference.name, reference)):
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)
        uem.write_text(f"{uri} 1 0 {case['duration_seconds']:.7f}\n", encoding="utf-8", newline="\n")
        cases.append({"id": uri, "audio": audio.name, "audio_sha256": digest(audio),
                      "reference": reference.name, "reference_sha256": digest(reference),
                      "uem": uem.name, "uem_sha256": digest(uem), "duration_seconds": case["duration_seconds"]})
    manifest = {"schema_version": 1, "task": "speaker_diarization", "source": lock, "cases": cases}
    write_json_atomic(output / "corpus.json", manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--audio-dir", type=Path)
    source.add_argument("--download", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    audio_dir = args.audio_dir or args.out
    if args.download:
        lock = json.loads((FIXTURES / "corpus-lock.json").read_text(encoding="utf-8"))
        download_audio(lock, audio_dir)
    result = prepare(audio_dir, args.out)
    print(f"Prepared {len(result['cases'])} recordings: {args.out / 'corpus.json'}")
# [SEMANTIC-END] QUALITY:DER_CORPUS


if __name__ == "__main__":
    main()
