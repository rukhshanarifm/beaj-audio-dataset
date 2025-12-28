# audio-dataset

Short: reproducible pipeline for cleaning human-graded CSVs, extracting AI responses, merging human+AI judgments, and running analysis/visualizations.

## Quick start (macOS, zsh)

1) Clone this repo

   git clone <repo-url>
   cd <repo-dir>

2) Install Homebrew (if needed):

   Follow https://docs.brew.sh/Installation

3) Install Python 3.12 — two options

- Option A — Homebrew (installs latest 3.12.x):

   brew install python@3.12
   # ensure brew python is on your PATH (zsh)
   echo 'export PATH="/opt/homebrew/opt/python@3.12/bin:$PATH"' >> ~/.zshrc
   source ~/.zshrc
   python3.12 --version

- Option B — Recommended if you need exactly Python 3.12.7: use pyenv

   brew install pyenv
   pyenv install 3.12.7
   pyenv local 3.12.7
   python --version

4) Create and activate a virtualenv (zsh)

   python3 -m venv .venv
   source .venv/bin/activate

5) Install dependencies

   pip install -r requirements.txt

6) Verify environment

   python --version
   pip list

## Repository summary

- `requirements.txt` — Python dependencies used by scripts and the analysis notebook.

- `data/`
  - `graded/` — raw grader CSVs (inputs)
  - `clean/` — cleaned and derived CSVs used for analysis (e.g. `graded_combined.csv`, `ai_responses_extracted.csv`, `merged_for_analysis.csv`, `word_level_summary.csv`, `wcpm_by_student.csv`, `final_analysis_dataset.csv`)

- `scripts/`
  - `0. clean_graded_datasets.py` — normalize & combine human-graded CSVs
  - `1. extract_ai_responses.py` — parse AI outputs into a tabular CSV
  - `2. merging.py` — join human and AI tables for analysis
  - `3. analysis.ipynb` — analysis & plotting notebook
  - `helper_functions.py` — utilities used by scripts and notebook

- `images/` — any source images used in tests
- `plots/` — generated figures (e.g. `overall_confusion_matrix.png`, `top10_word_confusion_heatmaps.png`)

## Notes

- If you need precisely Python 3.12.7, use the pyenv option above (pyenv lets you install and pin patch versions).
- The analysis notebook saves outputs to `data/clean/` and figures to `data/plots/` when run end-to-end.
