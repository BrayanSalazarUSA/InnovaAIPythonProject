from __future__ import annotations

import argparse
import json

from .monitor import run_monitor_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Innova event monitor MVP.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON config file, e.g. resources/event_monitor/event_rules.example.json",
    )
    args = parser.parse_args()
    summary = run_monitor_from_config(args.config)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

