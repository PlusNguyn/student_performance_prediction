from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = Path("data") / "demo_oulad"
DEFAULT_STUDENT_COUNT = 200
DEFAULT_SEED = 42

COURSE_MODULES = ("AAA", "BBB", "CCC", "DDD")
B_PRESENTATION_LENGTH = 240
J_PRESENTATION_LENGTH = 269
ASSESSMENT_PLAN = (
    ("TMA", 25, 10),
    ("CMA", 55, 10),
    ("TMA", 95, 20),
    ("CMA", 135, 10),
    ("TMA", 180, 20),
    ("Exam", 240, 30),
)
VLE_ACTIVITY_TYPES = (
    "homepage",
    "oucontent",
    "resource",
    "forumng",
    "subpage",
    "url",
    "quiz",
    "oucollaborate",
)
REGIONS = (
    "East Anglian Region",
    "Scotland",
    "North Western Region",
    "South Region",
    "London Region",
    "Wales",
    "West Midlands Region",
)
EDUCATION_LEVELS = (
    "Lower Than A Level",
    "A Level or Equivalent",
    "HE Qualification",
    "Postgraduate Qualification",
    "No Formal quals",
)
IMD_BANDS = (
    "0-10%",
    "10-20",
    "20-30%",
    "30-40%",
    "40-50%",
    "50-60%",
    "60-70%",
    "70-80%",
    "80-90%",
    "90-100%",
)
AGE_BANDS = ("0-35", "35-55", "55<=")


@dataclass(frozen=True)
class PresentationContext:
    snapshot_date: date
    code_presentation: str
    presentation_start_date: date
    module_presentation_length: int
    observation_day: int
    courses: tuple[tuple[str, str, int], ...]


@dataclass(frozen=True)
class StudentProfile:
    code_module: str
    code_presentation: str
    course_length: int
    id_student: int
    engagement: float
    ability: float
    current_score: float
    learning_state: str
    observation_day: int
    date_registration: int
    date_unregistration: int
    studied_credits: int
    num_of_prev_attempts: int


def generate_oulad_demo_data(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    student_count: int = DEFAULT_STUDENT_COUNT,
    snapshot_date: str | date | None = None,
    presentation_start_date: str | date | None = None,
    observation_day: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    if student_count < 8:
        raise ValueError("student_count must be at least 8 to create a useful demo.")

    rng = random.Random(seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    presentation = resolve_presentation(
        snapshot_date,
        presentation_start_date,
        observation_day,
    )
    assessments = build_assessments(presentation.courses)
    vle_sites = build_vle_sites(presentation.courses)
    students = build_students(rng, student_count, presentation)

    files = {
        "courses": output_path / "courses.csv",
        "assessments": output_path / "assessments.csv",
        "vle": output_path / "vle.csv",
        "studentInfo": output_path / "studentInfo.csv",
        "studentRegistration": output_path / "studentRegistration.csv",
        "studentAssessment": output_path / "studentAssessment.csv",
        "studentVle": output_path / "studentVle.csv",
        "metadata": output_path / "metadata.json",
    }

    write_csv(
        files["courses"],
        ["code_module", "code_presentation", "module_presentation_length"],
        [
            {
                "code_module": code_module,
                "code_presentation": code_presentation,
                "module_presentation_length": course_length,
            }
            for code_module, code_presentation, course_length in presentation.courses
        ],
    )
    write_csv(
        files["assessments"],
        ["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
        assessments,
    )
    write_csv(
        files["vle"],
        ["id_site", "code_module", "code_presentation", "activity_type", "week_from", "week_to"],
        vle_sites,
    )
    write_csv(
        files["studentInfo"],
        [
            "code_module",
            "code_presentation",
            "id_student",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
        ],
        build_student_info_rows(rng, students),
    )
    write_csv(
        files["studentRegistration"],
        [
            "code_module",
            "code_presentation",
            "id_student",
            "date_registration",
            "date_unregistration",
        ],
        build_registration_rows(students),
    )
    write_csv(
        files["studentAssessment"],
        ["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
        build_student_assessment_rows(rng, students, assessments),
    )
    write_csv(
        files["studentVle"],
        ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"],
        build_student_vle_rows(rng, students, vle_sites),
    )

    metadata = {
        "dataset": "OULAD-like synthetic online learning demo",
        "purpose": "Current-time prediction input without end-of-course outcome labels.",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "student_count": student_count,
        "snapshot_date": presentation.snapshot_date.isoformat(),
        "code_presentation": presentation.code_presentation,
        "presentation_start_date": presentation.presentation_start_date.isoformat(),
        "module_presentation_length": presentation.module_presentation_length,
        "observation_day": presentation.observation_day,
        "output_dir": str(output_path),
        "files": {name: str(path) for name, path in files.items()},
        "student_info_has_outcome_label": False,
    }
    files["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return metadata


def resolve_presentation(
    snapshot_date: str | date | None,
    presentation_start_date: str | date | None,
    observation_day: int | None,
) -> PresentationContext:
    resolved_snapshot_date = parse_snapshot_date(snapshot_date)
    resolved_start_date = parse_presentation_start_date(
        presentation_start_date,
        resolved_snapshot_date,
    )
    presentation_year = resolved_start_date.year
    presentation_suffix = "B" if resolved_start_date.month < 9 else "J"
    presentation_length = (
        B_PRESENTATION_LENGTH
        if presentation_suffix == "B"
        else J_PRESENTATION_LENGTH
    )
    observed_day = (
        int(observation_day)
        if observation_day is not None
        else (resolved_snapshot_date - resolved_start_date).days
    )
    observed_day = max(0, min(observed_day, presentation_length - 1))
    code_presentation = f"{presentation_year}{presentation_suffix}"
    courses = tuple(
        (code_module, code_presentation, presentation_length)
        for code_module in COURSE_MODULES
    )
    return PresentationContext(
        snapshot_date=resolved_snapshot_date,
        code_presentation=code_presentation,
        presentation_start_date=resolved_start_date,
        module_presentation_length=presentation_length,
        observation_day=observed_day,
        courses=courses,
    )


def parse_snapshot_date(value: str | date | None) -> date:
    if value is None:
        return date.today()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def parse_presentation_start_date(
    value: str | date | None,
    snapshot_date: date,
) -> date:
    if value is None:
        return date(snapshot_date.year, 1, 1)
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def build_assessments(courses: tuple[tuple[str, str, int], ...]) -> list[dict[str, Any]]:
    rows = []
    assessment_id = 8000
    for code_module, code_presentation, _ in courses:
        for assessment_type, date, weight in ASSESSMENT_PLAN:
            rows.append(
                {
                    "code_module": code_module,
                    "code_presentation": code_presentation,
                    "id_assessment": assessment_id,
                    "assessment_type": assessment_type,
                    "date": date,
                    "weight": weight,
                }
            )
            assessment_id += 1
    return rows


def build_vle_sites(courses: tuple[tuple[str, str, int], ...]) -> list[dict[str, Any]]:
    rows = []
    site_id = 700000
    for code_module, code_presentation, course_length in courses:
        course_weeks = max(1, course_length // 7)
        for activity_index, activity_type in enumerate(VLE_ACTIVITY_TYPES):
            week_from = "" if activity_index < 3 else max(0, activity_index - 2)
            week_to = "" if activity_index < 3 else min(course_weeks, activity_index + 8)
            rows.append(
                {
                    "id_site": site_id,
                    "code_module": code_module,
                    "code_presentation": code_presentation,
                    "activity_type": activity_type,
                    "week_from": week_from,
                    "week_to": week_to,
                }
            )
            site_id += 1
    return rows


def build_students(
    rng: random.Random,
    student_count: int,
    presentation: PresentationContext,
) -> list[StudentProfile]:
    students = []
    for index in range(student_count):
        code_module, code_presentation, course_length = presentation.courses[
            index % len(presentation.courses)
        ]
        ability = clamp(rng.betavariate(2.4, 2.0), 0.02, 0.98)
        engagement = clamp(rng.betavariate(2.0, 2.3), 0.02, 0.98)
        prev_attempts = weighted_choice(rng, [(0, 0.78), (1, 0.16), (2, 0.05), (3, 0.01)])
        studied_credits = weighted_choice(rng, [(30, 0.08), (60, 0.72), (90, 0.14), (120, 0.06)])
        registration = -rng.randint(5, 75)
        late_penalty = 0.08 if registration > -14 else 0
        attempt_penalty = min(prev_attempts * 0.05, 0.15)
        load_penalty = 0.05 if studied_credits >= 90 else 0
        score_signal = (
            0.55 * ability
            + 0.45 * engagement
            - late_penalty
            - attempt_penalty
            - load_penalty
            + rng.gauss(0, 0.06)
        )
        current_score = clamp(score_signal * 100, 0, 100)
        inactive = engagement < 0.22 or current_score < 32
        learning_state = classify_learning_state(current_score, engagement, inactive)
        unregistration = -1
        resolved_observation_day = min(presentation.observation_day, course_length - 1)
        if learning_state == "inactive":
            if resolved_observation_day > 35:
                unregistration = rng.randint(35, resolved_observation_day)
            else:
                unregistration = resolved_observation_day

        students.append(
            StudentProfile(
                code_module=code_module,
                code_presentation=code_presentation,
                course_length=course_length,
                id_student=900000 + index,
                engagement=engagement,
                ability=ability,
                current_score=current_score,
                learning_state=learning_state,
                observation_day=resolved_observation_day,
                date_registration=registration,
                date_unregistration=unregistration,
                studied_credits=studied_credits,
                num_of_prev_attempts=prev_attempts,
            )
        )
    return students


def build_student_info_rows(
    rng: random.Random,
    students: list[StudentProfile],
) -> list[dict[str, Any]]:
    rows = []
    for student in students:
        rows.append(
            {
                "code_module": student.code_module,
                "code_presentation": student.code_presentation,
                "id_student": student.id_student,
                "gender": weighted_choice(rng, [("M", 0.52), ("F", 0.48)]),
                "region": rng.choice(REGIONS),
                "highest_education": weighted_choice(
                    rng,
                    [
                        ("Lower Than A Level", 0.33),
                        ("A Level or Equivalent", 0.42),
                        ("HE Qualification", 0.19),
                        ("Postgraduate Qualification", 0.03),
                        ("No Formal quals", 0.03),
                    ],
                ),
                "imd_band": rng.choice(IMD_BANDS),
                "age_band": weighted_choice(rng, [("0-35", 0.68), ("35-55", 0.27), ("55<=", 0.05)]),
                "num_of_prev_attempts": student.num_of_prev_attempts,
                "studied_credits": student.studied_credits,
                "disability": weighted_choice(rng, [("N", 0.9), ("Y", 0.1)]),
            }
        )
    return rows


def build_registration_rows(students: list[StudentProfile]) -> list[dict[str, Any]]:
    return [
        {
            "code_module": student.code_module,
            "code_presentation": student.code_presentation,
            "id_student": student.id_student,
            "date_registration": student.date_registration,
            "date_unregistration": student.date_unregistration,
        }
        for student in students
    ]


def build_student_assessment_rows(
    rng: random.Random,
    students: list[StudentProfile],
    assessments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    assessments_by_course = group_by_course(assessments)
    rows = []
    for student in students:
        course_key = (student.code_module, student.code_presentation)
        for assessment in assessments_by_course[course_key]:
            assessment_date = int(assessment["date"])
            if assessment_date > student.observation_day:
                continue
            if student.date_unregistration != -1 and assessment_date > student.date_unregistration:
                continue
            submit_probability = 0.96 if student.learning_state in {"steady", "strong"} else 0.78
            if student.learning_state == "inactive":
                submit_probability = max(0.25, student.engagement)
            if rng.random() > submit_probability:
                continue

            late_days = max(0, round(rng.gauss(2.5 - student.engagement * 4, 4)))
            date_submitted = assessment_date + late_days
            if date_submitted > student.observation_day:
                continue
            score_noise = rng.gauss(0, 8 if assessment["assessment_type"] != "Exam" else 10)
            score = round(clamp(student.current_score + score_noise, 0, 100), 1)
            rows.append(
                {
                    "id_assessment": assessment["id_assessment"],
                    "id_student": student.id_student,
                    "date_submitted": date_submitted,
                    "is_banked": 1 if rng.random() < 0.03 else 0,
                    "score": score,
                }
            )
    return rows


def build_student_vle_rows(
    rng: random.Random,
    students: list[StudentProfile],
    vle_sites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    sites_by_course = group_by_course(vle_sites)
    rows = []
    for student in students:
        course_key = (student.code_module, student.code_presentation)
        active_until = (
            student.date_unregistration
            if student.date_unregistration != -1
            else student.observation_day
        )
        active_until = max(10, int(active_until))
        expected_days = int(6 + student.engagement * active_until * 0.42)
        active_days = sorted({rng.randint(0, active_until) for _ in range(expected_days)})
        for day in active_days:
            visits = weighted_choice(rng, [(1, 0.58), (2, 0.28), (3, 0.1), (4, 0.04)])
            for _ in range(visits):
                site = rng.choice(sites_by_course[course_key])
                clicks = int(max(1, round(rng.gauss(2 + student.engagement * 8, 2.2))))
                rows.append(
                    {
                        "code_module": student.code_module,
                        "code_presentation": student.code_presentation,
                        "id_student": student.id_student,
                        "id_site": site["id_site"],
                        "date": day,
                        "sum_click": clicks,
                    }
                )
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)


def group_by_course(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["code_module"]), str(row["code_presentation"]))
        grouped.setdefault(key, []).append(row)
    return grouped


def classify_learning_state(score: float, engagement: float, inactive: bool) -> str:
    if inactive:
        return "inactive"
    if score >= 82 and engagement >= 0.55:
        return "strong"
    if score >= 55:
        return "steady"
    return "at_risk"


def weighted_choice(rng: random.Random, options: list[tuple[Any, float]]) -> Any:
    total = sum(weight for _, weight in options)
    marker = rng.random() * total
    running = 0.0
    for value, weight in options:
        running += weight
        if marker <= running:
            return value
    return options[-1][0]


def count_by(students: list[StudentProfile], attribute: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for student in students:
        value = str(getattr(student, attribute))
        counts[value] = counts.get(value, 0) + 1
    return counts


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an OULAD-like synthetic online learning demo dataset.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--students", type=int, default=DEFAULT_STUDENT_COUNT)
    parser.add_argument(
        "--snapshot-date",
        default=None,
        help="Current snapshot date in YYYY-MM-DD format. Defaults to today.",
    )
    parser.add_argument(
        "--presentation-start-date",
        default=None,
        help="Presentation start date in YYYY-MM-DD format. Defaults to Jan 1 of the snapshot year.",
    )
    parser.add_argument(
        "--observation-day",
        type=int,
        default=None,
        help="Override the day index within the current presentation.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = generate_oulad_demo_data(
        args.output_dir,
        student_count=args.students,
        snapshot_date=args.snapshot_date,
        presentation_start_date=args.presentation_start_date,
        observation_day=args.observation_day,
        seed=args.seed,
    )
    print(
        "Generated "
        f"{summary['student_count']} demo students for "
        f"{summary['code_presentation']} day {summary['observation_day']} "
        f"in {summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
