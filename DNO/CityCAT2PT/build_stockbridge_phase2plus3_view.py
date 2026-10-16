#!/usr/bin/env python3
"""Materialize a 300-event Stockbridge Phase 2 + Phase 3 PT training view.

The view contains hardlinks to canonical PT files; source files are only read.
Sidecars and other non-PT files are intentionally excluded.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path


PT_PARENT = Path.home() / "flood/02_data/CityCAT_Winchester/pt"
DEFAULT_PHASE2_ROOT = PT_PARENT / "stockbridge_phase2_core_v01"
DEFAULT_PHASE3_ROOT = PT_PARENT / "stockbridge_phase3_core_v01"
DEFAULT_OUTPUT_ROOT = PT_PARENT / "stockbridge_phase2plus3_300_v01"
DATASET_NAME = "stockbridge_phase2plus3_300_v01"
MANIFEST_NAME = "stockbridge_phase2plus3_300_manifest.csv"
SUMMARY_NAME = "stockbridge_phase2plus3_300_summary.json"
SPLITS = ("train", "valid", "test")
SOURCE_COUNTS = {
    "phase2": {"train": 70, "valid": 15, "test": 15},
    "phase3": {"train": 140, "valid": 30, "test": 30},
}
VIEW_COUNTS = {"train": 210, "valid": 45, "test": 45}
TOTAL_COUNT = 300
MANIFEST_FIELDS = (
    "event_id", "phase", "split", "source_path", "view_path",
    "link_method", "file_size_bytes", "sha256",
)


@dataclass(frozen=True)
class Event:
    phase: str
    split: str
    source: Path

    @property
    def filename(self) -> str:
        return self.source.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-root", type=Path, default=DEFAULT_PHASE2_ROOT,
                        help="Canonical Phase 2 PT dataset root")
    parser.add_argument("--phase3-root", type=Path, default=DEFAULT_PHASE3_ROOT,
                        help="Canonical Phase 3 PT dataset root")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Destination for the materialized view")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate sources and destination without creating a view")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace an existing view at the output root")
    return parser.parse_args()


def source_root(path: Path) -> Path:
    root = path.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Source root is not a directory: {root}")
    return root


def overlaps(a: Path, b: Path) -> bool:
    return a == b or a in b.parents or b in a.parents


def check_destination(phase2: Path, phase3: Path, output_arg: Path,
                      overwrite: bool) -> Path:
    requested = output_arg.expanduser()
    if requested.is_symlink():
        raise ValueError(f"Output root must not be a symlink: {requested}")
    output = requested.resolve()
    if overlaps(phase2, phase3):
        raise ValueError("Phase 2 and Phase 3 source roots must be separate")
    if overlaps(output, phase2) or overlaps(output, phase3):
        raise ValueError(f"Output root overlaps a canonical source: {output}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}; use --overwrite")
        validate_existing_view(output)
    return output


def validate_existing_view(output: Path) -> None:
    """Restrict overwrite to a recognizable view with no unrelated entries."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"Existing output is not a view directory: {output}")
    allowed = set(SPLITS) | {MANIFEST_NAME, SUMMARY_NAME}
    unexpected = {entry.name for entry in output.iterdir()} - allowed
    if unexpected:
        raise ValueError(f"Existing output contains unrelated entries: {sorted(unexpected)}")
    for split in SPLITS:
        directory = output / split
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Existing view has no regular {split} directory: {directory}")
        for entry in directory.iterdir():
            if entry.suffix != ".pt" or entry.is_symlink() or not entry.is_file():
                raise ValueError(f"Existing view contains an unexpected entry: {entry}")
    manifest = output / MANIFEST_NAME
    summary = output / SUMMARY_NAME
    for metadata in (manifest, summary):
        if metadata.is_symlink() or not metadata.is_file():
            raise ValueError(f"Existing view is missing regular metadata: {metadata}")
    with summary.open("r", encoding="utf-8") as handle:
        details = json.load(handle)
    if (not isinstance(details, dict)
            or details.get("dataset_name") != DATASET_NAME
            or details.get("output_root") != str(output)
            or details.get("link_method") != "hardlink"):
        raise ValueError(f"Existing output is not this dataset view: {output}")


def discover(root: Path, phase: str) -> list[Event]:
    events = []
    for split in SPLITS:
        directory = root / split
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError(f"Missing regular source split directory: {directory}")
        files = sorted((entry for entry in directory.iterdir()
                        if entry.name.endswith(".pt")), key=lambda entry: entry.name)
        for entry in files:
            if entry.is_symlink() or not entry.is_file():
                raise ValueError(f"Source PT is not a regular file: {entry}")
        expected = SOURCE_COUNTS[phase][split]
        if len(files) != expected:
            raise ValueError(f"{phase}/{split}: expected {expected} PT files, found {len(files)}")
        events.extend(Event(phase, split, entry) for entry in files)
    return events


def validate_names(events: list[Event]) -> None:
    by_split: dict[str, dict[str, Event]] = {split: {} for split in SPLITS}
    all_names: dict[str, Event] = {}
    for event in events:
        # Case folding also catches names that would collide on a case-insensitive FS.
        key = event.filename.casefold()
        previous = by_split[event.split].get(key)
        if previous is not None:
            raise ValueError(f"Filename collision in {event.split}: "
                             f"{previous.source} and {event.source}")
        by_split[event.split][key] = event
        previous = all_names.get(key)
        if previous is not None:
            raise ValueError(f"Event filename occurs in multiple splits: "
                             f"{previous.source} and {event.source}")
        all_names[key] = event
    for split, expected in VIEW_COUNTS.items():
        actual = len(by_split[split])
        if actual != expected:
            raise ValueError(f"Combined {split}: expected {expected}, found {actual}")
    if len(all_names) != TOTAL_COUNT:
        raise ValueError(f"Expected {TOTAL_COUNT} unique event filenames, "
                         f"found {len(all_names)}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def same_inode(source: Path, view: Path) -> bool:
    original = source.stat()
    linked = view.stat()
    return original.st_dev == linked.st_dev and original.st_ino == linked.st_ino


def validate_view(root: Path, events: list[Event]) -> None:
    expected_by_split = {
        split: {event.filename for event in events if event.split == split}
        for split in SPLITS
    }
    names = []
    for split in SPLITS:
        directory = root / split
        entries = list(directory.iterdir())
        actual = {entry.name for entry in entries}
        if (actual != expected_by_split[split]
                or len(entries) != VIEW_COUNTS[split]
                or any(entry.is_symlink() or not entry.is_file()
                       for entry in entries)):
            raise ValueError(f"View {split} inventory does not match expected files")
        names.extend(actual)
    if len(names) != TOTAL_COUNT or len(set(names)) != TOTAL_COUNT:
        raise ValueError("View does not contain exactly 300 unique event filenames")
    for event in events:
        view = root / event.split / event.filename
        if not same_inode(event.source, view):
            raise ValueError(f"View file is not a hardlink to its source: {view}")


def materialize(staging: Path, output: Path, phase2: Path, phase3: Path,
                events: list[Event]) -> None:
    for split in SPLITS:
        (staging / split).mkdir()
    rows = []
    for event in events:
        staged = staging / event.split / event.filename
        before = event.source.stat()
        source_digest = sha256_file(event.source)
        os.link(event.source, staged, follow_symlinks=False)
        if not same_inode(event.source, staged):
            raise ValueError(f"Hardlink verification failed: {staged}")
        if sha256_file(staged) != source_digest:
            raise ValueError(f"SHA256 mismatch after linking: {staged}")
        after = event.source.stat()
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError(f"Source changed while building the view: {event.source}")
        rows.append({
            "event_id": event.source.stem,
            "phase": event.phase,
            "split": event.split,
            "source_path": str(event.source),
            "view_path": str(output / event.split / event.filename),
            "link_method": "hardlink",
            "file_size_bytes": after.st_size,
            "sha256": source_digest,
        })

    with (staging / MANIFEST_NAME).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "dataset_name": DATASET_NAME,
        "phase2_root": str(phase2),
        "phase3_root": str(phase3),
        "output_root": str(output),
        "link_method": "hardlink",
        "train_count": VIEW_COUNTS["train"],
        "valid_count": VIEW_COUNTS["valid"],
        "test_count": VIEW_COUNTS["test"],
        "total_count": TOTAL_COUNT,
    }
    with (staging / SUMMARY_NAME).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    validate_view(staging, events)


def publish(staging: Path, output: Path, overwrite: bool,
            events: list[Event]) -> None:
    backup = None
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        validate_existing_view(output)
        backup = output.parent / f".{output.name}.old-{uuid.uuid4().hex}"
        output.rename(backup)
    try:
        staging.rename(output)
        validate_view(output, events)
    except Exception:
        if output.exists():
            shutil.rmtree(output)
        if backup is not None:
            backup.rename(output)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def main() -> int:
    args = parse_args()
    staging = None
    try:
        phase2 = source_root(args.phase2_root)
        phase3 = source_root(args.phase3_root)
        output = check_destination(phase2, phase3, args.output_root, args.overwrite)
        events = discover(phase2, "phase2") + discover(phase3, "phase3")
        validate_names(events)
        if args.dry_run:
            print(f"Dry run passed: {VIEW_COUNTS['train']} train, "
                  f"{VIEW_COUNTS['valid']} valid, {VIEW_COUNTS['test']} test; "
                  f"{TOTAL_COUNT} unique PT events. Output: {output}")
            return 0
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-",
                                        dir=output.parent))
        materialize(staging, output, phase2, phase3, events)
        publish(staging, output, args.overwrite, events)
        staging = None
        print(f"Validated {output}: {VIEW_COUNTS['train']} train, "
              f"{VIEW_COUNTS['valid']} valid, {VIEW_COUNTS['test']} test; "
              f"{TOTAL_COUNT} unique PT hardlinks; SHA256 verified.")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    sys.exit(main())
