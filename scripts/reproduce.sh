#!/usr/bin/env bash
# Run the full study from a clean checkout. Requires BHARATCODE_API_KEY.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
python -m pytest tests -q

# 1. shallow interaction arms (comparison with prior work: 0..50 interactions)
python -m src.experiment_runner --main --trials 2 --tag reproduce_main

# 2. deep-horizon stage (token-verified 240k, then the 150k/200k ladder)
python -m src.run_stage --stage deep --max-parallel 4
python -m src.run_stage --stage deep-reinforced --max-parallel 4
python -m src.run_stage --stage deep-neutral --max-parallel 4

# 3. analysis and figures (no API calls)
python -m src.statistics --figures
python -m src.deep_stats
python -m src.paper_tables
