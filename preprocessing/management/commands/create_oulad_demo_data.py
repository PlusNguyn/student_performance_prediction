from django.core.management.base import BaseCommand

from preprocessing.demo_data import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    DEFAULT_STUDENT_COUNT,
    generate_oulad_demo_data,
)


class Command(BaseCommand):
    help = "Create an OULAD-like synthetic online learning dataset for demos."

    def add_arguments(self, parser):
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

    def handle(self, *args, **options):
        summary = generate_oulad_demo_data(
            options["output_dir"],
            student_count=options["students"],
            snapshot_date=options["snapshot_date"],
            presentation_start_date=options["presentation_start_date"],
            observation_day=options["observation_day"],
            seed=options["seed"],
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"Generated {summary['student_count']} demo students for "
                f"{summary['code_presentation']} day {summary['observation_day']} "
                f"in {summary['output_dir']}."
            )
        )
        self.stdout.write(
            "Prediction snapshot: "
            f"snapshot_date={summary['snapshot_date']}, "
            f"presentation_start_date={summary['presentation_start_date']}, "
            f"observation_day={summary['observation_day']}, "
            "studentInfo has no outcome label column."
        )
