from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError


STUDENT_KEY = ["code_module", "code_presentation", "id_student"]
COURSE_KEY = ["code_module", "code_presentation"]
RAW_FILES = [
    "assessments.csv",
    "courses.csv",
    "studentAssessment.csv",
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
    "vle.csv",
]
CSV_READ_OPTIONS = {
    "dtype": str,
    "keep_default_na": False,
    "na_filter": False,
}
CSV_WRITE_OPTIONS = {
    "index": False,
    "quoting": csv.QUOTE_ALL,
}


@dataclass
class SplitResult:
    demo_rows: int
    raw_rows: int
    demo_values: dict[str, set[str]]
    raw_values: dict[str, set[str]]


class Command(BaseCommand):
    help = (
        "Move a small, relationally consistent slice of raw OULAD CSV data "
        "into data/demo."
    )

    def add_arguments(self, parser):
        parser.add_argument("--raw-dir", default=None)
        parser.add_argument("--demo-dir", default=None)
        parser.add_argument("--demo-size", type=int, default=100)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--chunksize", type=int, default=250_000)
        parser.add_argument(
            "--overwrite-demo",
            action="store_true",
            help="Overwrite existing CSV files in the demo directory.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be moved without writing files.",
        )

    def handle(self, *args, **options):
        raw_dir = Path(options["raw_dir"]) if options["raw_dir"] else _default_raw_dir()
        demo_dir = (
            Path(options["demo_dir"]) if options["demo_dir"] else _default_demo_dir()
        )
        demo_size = options["demo_size"]
        seed = options["seed"]
        chunksize = options["chunksize"]
        overwrite_demo = options["overwrite_demo"]
        dry_run = options["dry_run"]

        if demo_size <= 0:
            raise CommandError("--demo-size must be greater than 0.")
        if chunksize <= 0:
            raise CommandError("--chunksize must be greater than 0.")

        _validate_raw_files(raw_dir)
        if not dry_run:
            _prepare_demo_dir(demo_dir, overwrite_demo=overwrite_demo)

        students = _read_csv(raw_dir / "studentInfo.csv")
        if students.empty:
            raise CommandError("studentInfo.csv is empty; no demo data can be moved.")

        selected_students = _sample_students(students, demo_size=demo_size, seed=seed)
        selected_keys = selected_students[STUDENT_KEY].drop_duplicates()
        selected_courses = selected_students[COURSE_KEY].drop_duplicates()
        remaining_student_courses = students.loc[
            ~_matching_mask(students, selected_keys, STUDENT_KEY),
            COURSE_KEY,
        ].drop_duplicates()

        if dry_run:
            self.stdout.write(self.style.WARNING("Dry run only; no files were changed."))

        self.stdout.write(
            f"Selected {len(selected_students)} rows from studentInfo.csv "
            f"({len(selected_keys)} unique student enrollments)."
        )

        student_info_result = _split_dataframe_by_keys(
            frame=students,
            selected_keys=selected_keys,
            key_columns=STUDENT_KEY,
            raw_path=raw_dir / "studentInfo.csv",
            demo_path=demo_dir / "studentInfo.csv",
            dry_run=dry_run,
        )

        registration = _read_csv(raw_dir / "studentRegistration.csv")
        registration_result = _split_dataframe_by_keys(
            frame=registration,
            selected_keys=selected_keys,
            key_columns=STUDENT_KEY,
            raw_path=raw_dir / "studentRegistration.csv",
            demo_path=demo_dir / "studentRegistration.csv",
            dry_run=dry_run,
        )

        courses = _read_csv(raw_dir / "courses.csv")
        _write_lookup_split(
            frame=courses,
            key_columns=COURSE_KEY,
            demo_keys=selected_courses,
            raw_keep_keys=remaining_student_courses,
            raw_path=raw_dir / "courses.csv",
            demo_path=demo_dir / "courses.csv",
            dry_run=dry_run,
        )

        assessments = _read_csv(raw_dir / "assessments.csv")
        selected_assessment_pairs = (
            selected_keys.merge(
                assessments[COURSE_KEY + ["id_assessment"]],
                on=COURSE_KEY,
                how="inner",
            )[["id_student", "id_assessment"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        assessment_result = _split_csv_by_keys(
            raw_path=raw_dir / "studentAssessment.csv",
            demo_path=demo_dir / "studentAssessment.csv",
            selected_keys=selected_assessment_pairs,
            key_columns=["id_student", "id_assessment"],
            chunksize=chunksize,
            dry_run=dry_run,
            collect_columns=["id_assessment"],
        )

        demo_assessment_ids = _values_to_frame(
            assessment_result.demo_values.get("id_assessment", set()),
            "id_assessment",
        )
        raw_assessment_ids = _values_to_frame(
            assessment_result.raw_values.get("id_assessment", set()),
            "id_assessment",
        )
        _write_lookup_split(
            frame=assessments,
            key_columns=["id_assessment"],
            demo_keys=demo_assessment_ids,
            raw_keep_keys=raw_assessment_ids,
            raw_path=raw_dir / "assessments.csv",
            demo_path=demo_dir / "assessments.csv",
            dry_run=dry_run,
        )

        vle_result = _split_csv_by_keys(
            raw_path=raw_dir / "studentVle.csv",
            demo_path=demo_dir / "studentVle.csv",
            selected_keys=selected_keys,
            key_columns=STUDENT_KEY,
            chunksize=chunksize,
            dry_run=dry_run,
            collect_columns=["id_site"],
        )

        vle = _read_csv(raw_dir / "vle.csv")
        demo_site_ids = _values_to_frame(
            vle_result.demo_values.get("id_site", set()),
            "id_site",
        )
        raw_site_ids = _values_to_frame(
            vle_result.raw_values.get("id_site", set()),
            "id_site",
        )
        _write_lookup_split(
            frame=vle,
            key_columns=["id_site"],
            demo_keys=demo_site_ids,
            raw_keep_keys=raw_site_ids,
            raw_path=raw_dir / "vle.csv",
            demo_path=demo_dir / "vle.csv",
            dry_run=dry_run,
        )

        self.stdout.write(self.style.SUCCESS("Demo split completed."))
        self.stdout.write(f"Demo dir: {demo_dir}")
        self.stdout.write(f"Raw dir: {raw_dir}")
        self.stdout.write(
            "Moved rows: "
            f"studentInfo={student_info_result.demo_rows}, "
            f"studentRegistration={registration_result.demo_rows}, "
            f"studentAssessment={assessment_result.demo_rows}, "
            f"studentVle={vle_result.demo_rows}"
        )


def _default_raw_dir() -> Path:
    return Path(settings.BASE_DIR) / "data" / "raw"


def _default_demo_dir() -> Path:
    return Path(settings.BASE_DIR) / "data" / "demo"


def _validate_raw_files(raw_dir: Path) -> None:
    if not raw_dir.exists():
        raise CommandError(f"Raw directory does not exist: {raw_dir}")

    missing_files = [name for name in RAW_FILES if not (raw_dir / name).exists()]
    if missing_files:
        raise CommandError(
            "Missing raw CSV files: " + ", ".join(sorted(missing_files))
        )


def _prepare_demo_dir(demo_dir: Path, *, overwrite_demo: bool) -> None:
    demo_dir.mkdir(parents=True, exist_ok=True)
    existing_files = [demo_dir / name for name in RAW_FILES if (demo_dir / name).exists()]
    if existing_files and not overwrite_demo:
        names = ", ".join(path.name for path in existing_files)
        raise CommandError(
            f"Demo files already exist ({names}). Use --overwrite-demo to replace them."
        )

    for path in existing_files:
        path.unlink()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_READ_OPTIONS)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, **CSV_WRITE_OPTIONS)


def _sample_students(
    students: pd.DataFrame,
    *,
    demo_size: int,
    seed: int,
) -> pd.DataFrame:
    sample_size = min(demo_size, len(students))
    return (
        students.sample(n=sample_size, random_state=seed)
        .sort_values(STUDENT_KEY)
        .reset_index(drop=True)
    )


def _split_dataframe_by_keys(
    *,
    frame: pd.DataFrame,
    selected_keys: pd.DataFrame,
    key_columns: list[str],
    raw_path: Path,
    demo_path: Path,
    dry_run: bool,
    collect_columns: list[str] | None = None,
) -> SplitResult:
    mask = _matching_mask(frame, selected_keys, key_columns)
    demo_frame = frame.loc[mask].copy()
    raw_frame = frame.loc[~mask].copy()
    collect_columns = collect_columns or []

    if not dry_run:
        _write_csv(demo_frame, demo_path)
        _write_csv(raw_frame, raw_path)

    return SplitResult(
        demo_rows=len(demo_frame),
        raw_rows=len(raw_frame),
        demo_values=_collect_values(demo_frame, collect_columns),
        raw_values=_collect_values(raw_frame, collect_columns),
    )


def _split_csv_by_keys(
    *,
    raw_path: Path,
    demo_path: Path,
    selected_keys: pd.DataFrame,
    key_columns: list[str],
    chunksize: int,
    dry_run: bool,
    collect_columns: list[str] | None = None,
) -> SplitResult:
    collect_columns = collect_columns or []
    selected_keys = selected_keys[key_columns].drop_duplicates().copy()
    demo_rows = 0
    raw_rows = 0
    demo_values = {column: set() for column in collect_columns}
    raw_values = {column: set() for column in collect_columns}
    temp_raw_path = raw_path.with_name(f"{raw_path.name}.tmp")
    first_chunk_columns: list[str] | None = None
    wrote_demo = False
    wrote_raw = False

    if not dry_run:
        demo_path.parent.mkdir(parents=True, exist_ok=True)
        if temp_raw_path.exists():
            temp_raw_path.unlink()

    for chunk in pd.read_csv(raw_path, chunksize=chunksize, **CSV_READ_OPTIONS):
        first_chunk_columns = list(chunk.columns)
        mask = _matching_mask(chunk, selected_keys, key_columns)
        demo_chunk = chunk.loc[mask]
        raw_chunk = chunk.loc[~mask]

        demo_rows += len(demo_chunk)
        raw_rows += len(raw_chunk)
        _update_collected_values(demo_values, demo_chunk)
        _update_collected_values(raw_values, raw_chunk)

        if dry_run:
            continue

        if not demo_chunk.empty:
            demo_chunk.to_csv(
                demo_path,
                mode="a",
                header=not wrote_demo,
                **CSV_WRITE_OPTIONS,
            )
            wrote_demo = True

        raw_chunk.to_csv(
            temp_raw_path,
            mode="a",
            header=not wrote_raw,
            **CSV_WRITE_OPTIONS,
        )
        wrote_raw = True

    if not dry_run:
        if not wrote_demo:
            empty_demo = pd.DataFrame(columns=first_chunk_columns or [])
            _write_csv(empty_demo, demo_path)
        temp_raw_path.replace(raw_path)

    return SplitResult(
        demo_rows=demo_rows,
        raw_rows=raw_rows,
        demo_values=demo_values,
        raw_values=raw_values,
    )


def _write_lookup_split(
    *,
    frame: pd.DataFrame,
    key_columns: list[str],
    demo_keys: pd.DataFrame,
    raw_keep_keys: pd.DataFrame,
    raw_path: Path,
    demo_path: Path,
    dry_run: bool,
) -> SplitResult:
    demo_mask = _matching_mask(frame, demo_keys, key_columns)
    raw_removable_keys = _anti_join(
        demo_keys[key_columns].drop_duplicates(),
        raw_keep_keys[key_columns].drop_duplicates(),
        key_columns,
    )
    raw_remove_mask = _matching_mask(frame, raw_removable_keys, key_columns)

    demo_frame = frame.loc[demo_mask].copy()
    raw_frame = frame.loc[~raw_remove_mask].copy()

    if not dry_run:
        _write_csv(demo_frame, demo_path)
        _write_csv(raw_frame, raw_path)

    return SplitResult(
        demo_rows=len(demo_frame),
        raw_rows=len(raw_frame),
        demo_values={},
        raw_values={},
    )


def _matching_mask(
    frame: pd.DataFrame,
    selected_keys: pd.DataFrame,
    key_columns: list[str],
) -> pd.Series:
    if selected_keys.empty:
        return pd.Series(False, index=frame.index)

    marker = selected_keys[key_columns].drop_duplicates().copy()
    marker["_demo_match"] = True
    matched = frame[key_columns].merge(marker, on=key_columns, how="left")
    return matched["_demo_match"].notna().set_axis(frame.index)


def _anti_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key_columns: list[str],
) -> pd.DataFrame:
    if left.empty:
        return left.copy()
    if right.empty:
        return left[key_columns].drop_duplicates().copy()

    merged = left[key_columns].drop_duplicates().merge(
        right[key_columns].drop_duplicates(),
        on=key_columns,
        how="left",
        indicator=True,
    )
    return merged.loc[merged["_merge"].eq("left_only"), key_columns]


def _values_to_frame(values: set[str], column: str) -> pd.DataFrame:
    return pd.DataFrame(sorted(values), columns=[column])


def _collect_values(
    frame: pd.DataFrame,
    columns: list[str],
) -> dict[str, set[str]]:
    values = {column: set() for column in columns}
    _update_collected_values(values, frame)
    return values


def _update_collected_values(
    values: dict[str, set[str]],
    frame: pd.DataFrame,
) -> None:
    for column in values:
        if column in frame.columns:
            values[column].update(frame[column].dropna().astype(str).unique())
