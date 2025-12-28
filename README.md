# audio-dataset

This folder contains the code, cleaned datasets and plots used to analyze human vs AI pronunciation/fluency grading on short oral reading tasks.

## Overview

- Purpose: provide a reproducible pipeline that cleans raw graded CSVs, extracts AI responses, merges human and AI judgments at utterance and word level, and runs analysis/visualizations.
- Recommended workflow (run in this order):
  1. `scripts/0. clean_graded_datasets.py` — clean and standardize human-graded CSVs from `data/graded/` and produce `data/clean/graded_combined.csv`.
  2. `scripts/1. extract_ai_responses.py` — parse AI model output/feedback and extract per-utterance and per-word fields into `data/clean/ai_responses_extracted.csv`.
  3. `scripts/2. merging.py` — join the cleaned human grades with the AI-extracted responses to produce merged tables and intermediate summaries (see outputs below).
  4. `scripts/3. analysis.ipynb` — exploratory analysis and plotting notebook. Run after steps 0→2; the notebook reads the cleaned CSVs, computes word- and utterance-level metrics (WER, MAE, correlations, fluency/WCPM), and saves plots to `data/plots/`.

## Folder layout (important files)

- `data/graded/` — raw grader CSVs (per-grader, per-question). These are the inputs to the cleaning step.
- `data/clean/` — cleaned and derived datasets used for analysis and reporting. Key files (described below) are written here.
- `data/images/` — source images used in tests (if any).
- `data/plots/` — generated figures (confusion matrices, heatmaps, etc.).
- `scripts/` — pipeline scripts and the analysis notebook:
  - `0. clean_graded_datasets.py` — normalize and combine human-graded CSVs into a single cleaned table.
  - `1. extract_ai_responses.py` — extract and parse the AI model outputs (raw JSON/text responses) into a tabular CSV.
  - `2. merging.py` — join AI responses with cleaned human grades to produce a merged dataset ready for analysis.
  - `3. analysis.ipynb` — analysis notebook that computes metrics, creates visualizations, and exports summary CSVs and figures.
  - `helper_functions.py` — utility functions used by the scripts and notebook (parsers, timing utilities, scoring helpers).

## Recommended run order and what each script produces

1. `0. clean_graded_datasets.py`
   - Purpose: reads the grader CSVs from `data/graded/`, normalizes column names and IDs, and writes a cleaned combined file to `data/clean/graded_combined.csv`.
2. `1. extract_ai_responses.py`
   - Purpose: reads raw AI outputs (JSON/text responses), parses the transcript/word-level offsets where available, and writes `data/clean/ai_responses_extracted.csv`.
3. `2. merging.py`
   - Purpose: aligns/merges the cleaned human grades with the parsed AI responses on student/session/question identifiers and writes `data/clean/merged_for_analysis.csv`.
4. `3. analysis.ipynb`
   - Purpose: analysis notebook that:
     - computes utterance-level and word-level metrics (WER, Levenshtein distance, fluency comparisons),
     - extracts a word-level table and word summaries,
     - computes per-student WCPM (words correct per minute),
     - generates confusion matrices and heatmaps, and
     - writes several CSV and image outputs to `data/clean/` and `data/plots/`.
   - Note: the notebook contains the visualization cells and also saves CSVs/plots when run end-to-end.

## Key datasets (in `data/clean`) and what they mean

- `graded_combined.csv` — cleaned and concatenated human-graded records (one row per utterance with human grader columns standardized). Use this as the canonical source of human judgments.
- `ai_responses_extracted.csv` — parsed AI outputs: model scores, per-word feedback, timestamps/ticks for utterance start/end, and any computed AI-level features.
- `merged_for_analysis.csv` — row-level join of human and AI data (one row per utterance) combining both sources and metadata used by the analysis notebook.
- `word_level_summary.csv` — aggregation at the word level: counts, mean human score, mean AI score, mean absolute error (MAE), WER statistics and other per-word diagnostics used to surface systematic disagreements.
- `wcpm_by_student.csv` — fluency summary at the student/utterance level (words correct per minute, total_words, duration_seconds, and related fluency metrics).
- `final_analysis_dataset.csv` — a cleaned, ready-to-analyze dataset used by the notebook that typically includes normalized AI fluency, human fluency, alignment columns and any features created for hypothesis testing.

## Plots (in `data/plots/`)

- `overall_confusion_matrix.png` — confusion matrix comparing categorical word-level human vs AI labels (counts and/or normalized percentages).
- `top10_word_confusion_heatmaps.png` — compact heatmaps showing top-disagreement words and mean human vs AI scores. The analysis notebook will regenerate these when run.

## Notes on timing units and helper utilities

- Offset / Duration units: AI word-level timing (the `Offset` / `Duration` fields present in model outputs) are typically in 100-nanosecond ticks (the same unit used by some speech-API providers). To convert ticks to seconds: seconds = ticks / 10_000_000. The project helper `compute_utterance_duration_seconds` handles parsing the AI output structure and performs this conversion.
- Fluency normalization: the analysis uses a per-utterance normalized AI fluency score (total_score / total_possible) rather than averaging previously-normalized per-word scores — this was chosen to better reflect weighted contribution of words.

## Re-running the analysis

- The project-level `requirements.txt` lists the Python dependencies used by the scripts and notebook. Use the same Python environment when re-running analysis to avoid version mismatches.
- To re-run the analysis, run all the cells in `3. analysis.ipynb` (or restart & run all). The notebook saves intermediates to `data/clean/`.
