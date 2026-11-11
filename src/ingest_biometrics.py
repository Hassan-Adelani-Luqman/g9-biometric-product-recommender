"""Copy raw biometric recordings into the repo layout with standard names and formats.

Team members hand in photos and voice clips in whatever their phone produces
(.jpeg/.jpg, .m4a/.aac/.mp4, inconsistent capitalisation, typos). This script
normalises them to the conventions in Plan.md §2.3:

    Images -> data/images/<member>/<member>_<expression>_<take>.jpg
              EXIF orientation applied, then ALL metadata stripped (phone photos
              can carry GPS location), RGB, JPEG quality 95.
    Audio  -> data/audio/<member>/<member>_<phrase>_<take>.wav
              16 kHz, mono, 16-bit PCM (first audio stream only).

Expected source layout (folder names are matched to config.MEMBER_NAMES,
ignoring case, spaces and underscores):

    <source>/Image_features/<Full Name>/<expression>.<jpg|jpeg|png>
    <source>/Audio_features/<Full Name>/<phrase>.<m4a|aac|mp4|wav|mp3|...>

Files that already exist are never overwritten. A new recording of the same
expression/phrase gets the next take number (_02, _03, ...), so the script
can be rerun when members add more takes.

Usage (from the project root):
    python src/ingest_biometrics.py <source_dir> [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageOps

try:
    from src import config
except ModuleNotFoundError:
    import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
AUDIO_SUFFIXES = {".m4a", ".aac", ".mp4", ".mp3", ".wav", ".ogg", ".opus", ".flac", ".3gp", ".amr"}

# Filename keyword -> canonical label (checked in order; handles common typos)
EXPRESSION_PATTERNS = [
    (r"neutral", "neutral"),
    (r"smil", "smiling"),
    (r"su[r]?pris", "surprised"),  # "surprised" and the frequent "suprised"
]
PHRASE_PATTERNS = [
    (r"confirm", "confirm_transaction"),
    (r"approve", "yes_approve"),
]


def _key(text: str) -> str:
    return re.sub(r"[\s_\-]+", "", text).lower()


def resolve_member(folder_name: str) -> str:
    """Map a source folder name (e.g. 'Emma Tiffany Umwari') to a member ID."""
    lookup = {_key(full): mid for mid, full in config.MEMBER_NAMES.items()}
    lookup.update({mid: mid for mid in config.MEMBER_NAMES})
    try:
        return lookup[_key(folder_name)]
    except KeyError:
        raise SystemExit(
            f"Unknown member folder {folder_name!r}. Add the person to "
            f"config.MEMBER_NAMES or rename the folder."
        ) from None


def classify(stem: str, patterns: list[tuple[str, str]]) -> str:
    for pattern, label in patterns:
        if re.search(pattern, stem, flags=re.IGNORECASE):
            return label
    raise SystemExit(f"Cannot tell what {stem!r} is. Expected one of {[p[1] for p in patterns]}.")


def next_path(directory: Path, member: str, label: str, ext: str) -> Path:
    take = 1
    while (path := directory / f"{member}_{label}_{take:02d}{ext}").exists():
        take += 1
    return path


def ingest_image(src: Path, dst: Path) -> str:
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        # Re-encoding without passing exif= drops all EXIF/GPS metadata.
        img.save(dst, "JPEG", quality=95, optimize=True)
        return f"{img.width}x{img.height}"


def ingest_audio(src: Path, dst: Path) -> None:
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-n",
        "-i", str(src), "-map", "0:a:0", "-vn",
        "-ac", "1", "-ar", str(config.SAMPLE_RATE), "-sample_fmt", "s16",
        "-map_metadata", "-1", str(dst),
    ]
    subprocess.run(cmd, check=True)


def plan_jobs(source: Path) -> list[tuple[str, Path, Path, str]]:
    """Return (kind, src, dst, label) for every file to ingest."""
    jobs: list[tuple[str, Path, Path, str]] = []
    reserved: set[Path] = set()
    sections = [
        ("image", "Image_features", IMAGE_SUFFIXES, EXPRESSION_PATTERNS, config.IMAGES_DIR, config.IMAGE_EXT),
        ("audio", "Audio_features", AUDIO_SUFFIXES, PHRASE_PATTERNS, config.AUDIO_DIR, config.AUDIO_EXT),
    ]
    for kind, sub, suffixes, patterns, out_root, ext in sections:
        section_dir = source / sub
        if not section_dir.is_dir():
            print(f"[skip] no {sub}/ folder in {source}")
            continue
        for member_dir in sorted(p for p in section_dir.iterdir() if p.is_dir()):
            member = resolve_member(member_dir.name)
            out_dir = out_root / member
            for src in sorted(member_dir.iterdir()):
                if src.suffix.lower() not in suffixes:
                    print(f"[skip] unsupported file {src}")
                    continue
                label = classify(src.stem, patterns)
                dst = next_path(out_dir, member, label, ext)
                while dst in reserved:  # two sources in one run with the same label
                    take = int(dst.stem.rsplit("_", 1)[1]) + 1
                    dst = out_dir / f"{member}_{label}_{take:02d}{ext}"
                reserved.add(dst)
                jobs.append((kind, src, dst, label))
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="folder containing Image_features/ and Audio_features/")
    parser.add_argument("--dry-run", action="store_true", help="show what would be written, change nothing")
    args = parser.parse_args()

    if not args.source.is_dir():
        sys.exit(f"Not a directory: {args.source}")

    jobs = plan_jobs(args.source)
    for kind, src, dst, _ in jobs:
        rel_dst = dst.relative_to(config.ROOT)
        if args.dry_run:
            print(f"[dry-run] {kind}: {src.parent.name}/{src.name} -> {rel_dst}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if kind == "image":
            size = ingest_image(src, dst)
            print(f"[image] {src.parent.name}/{src.name} -> {rel_dst} ({size}, metadata stripped)")
        else:
            ingest_audio(src, dst)
            print(f"[audio] {src.parent.name}/{src.name} -> {rel_dst} (16 kHz mono wav)")
    print(f"\n{len(jobs)} file(s) {'planned' if args.dry_run else 'ingested'}.")


if __name__ == "__main__":
    main()
