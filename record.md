# Change Record: LLM-Supervisor Validation for the RT-MonoDepth Construction Paper

This record documents the repository and paper changes made to add an
LLM-supervisor validation experiment to the RT-MonoDepth + YOLOv11 construction
safety pipeline. The goal, requested for a top-tier journal submission, was to
run the existing perception script, use its CSV logs, prepare 30 questions, and
quantitatively measure how a locally hosted `llama3.1:8b` model performs as a
safety supervisor reading the system's structured telemetry.

The experiment ran end to end on this machine on **2026-04-27** with:

- **Input video**: `input/depth_estimation_1_20250817_212451.mp4` (3,035 frames @ 25 FPS, 640×360)
- **Models**: RT-MonoDepth (`weights/RTMonoDepth/s/m_640_192/`) + custom YOLOv11n (`custom_yolo11n.pt`)
- **Hardware**: Apple M1 Pro (PyTorch 2.11, MPS backend) + Ollama 0.21.2
- **LLM under test**: `llama3.1:8b` (Q4_K_M quantization, ~4.9 GB)

---

## 1. Headline Result

The locally hosted **Llama 3.1 8B Instruct** model achieved **10/30 (33.3%)
accuracy** on the automatically graded benchmark, with a strong capability
gradient that is now reported in the paper:

| Category                  | Correct | Total | Accuracy |
| ------------------------- | ------: | ----: | -------: |
| Row-level lookup          |       4 |     5 |    80.0% |
| Class identification      |       2 |     3 |    66.7% |
| Aggregate (depth + conf.) |       2 |     5 |    40.0% |
| Safety decision (yes/no)  |       1 |     3 |    33.3% |
| Counts (long-context)     |       1 |     5 |    20.0% |
| Proximity-zone reasoning  |       0 |     6 |     0.0% |
| Distinct-frame reasoning  |       0 |     3 |     0.0% |
| **Overall**               |  **10** |**30** |**33.3%** |

Mean per-question latency: 3.95 s. Total wall-clock: 118.5 s.

The takeaway, now embedded in the paper's Discussion: an 8B local LLM is
trustworthy as a *retrieval / paraphrase* layer over the deterministic CSV
logs but **cannot** be trusted to recompute threshold-based safety statistics
on its own. The deterministic pipeline must remain the system of record.

---

## 2. New / Modified Files in the Repository

### 2.1 New code

| Path                       | Purpose                                                                                                              |
| -------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `process_video_headless.py`| Headless wrapper around `realtime_depth_video.py` that runs the perception pipeline on a video file with no OpenCV GUI calls and writes the depth-log and distance-log CSVs. Reuses `RTMonoDepthModel`, `YOLODetector`, `DepthLogger`, and the auto-calibration logic from the original script. |
| `llm_supervisor_eval.py`   | The 30-question LLM-supervisor benchmark. Loads the CSV logs, computes deterministic ground truth in Python, queries Ollama with a fixed system prompt that requires an `ANSWER: <value>` last line, parses the response with regex + numeric tolerance, and writes machine-readable + human-readable reports. |

### 2.2 New data and reports

| Path                                                | Purpose                                                                              |
| --------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `depth_logs/depth_log_20260427_015450.csv`          | Per-detection log produced by `process_video_headless.py` (1,031 rows, 137 valid detections: 96 Person / 30 Vehicle / 11 Machinery). |
| `depth_logs/distance_log_20260427_015450.csv`       | Pairwise 3D distance log between detected objects in the same frame (19 rows).       |
| `llm_eval/llm_eval_report.json`                     | Per-question records + aggregate metrics in JSON (one entry per question, including raw model response). |
| `llm_eval/llm_eval_summary.csv`                     | Tidy CSV form of the per-question results, suitable for spreadsheet analysis or appendix tables. |
| `llm_eval/llm_eval_summary.md`                      | Human-readable Markdown report (used as the source for the paper's Table 4 and Section 3.5 narrative). |

### 2.3 Existing files left unchanged

`realtime_depth_video.py` was **not modified**. The headless processor imports
from it directly so that any future tweak to the production pipeline (e.g.\
camera intrinsics, calibration smoothing, sampling strategy) automatically
propagates to the validation run.

A Python 3.13 virtual environment was created at `.venv/` (already covered by
the existing `.gitignore`) and the original `requirements.txt` dependencies
were installed alongside `ollama` and `pandas` for the supervisor harness.

---

## 3. Changes to the Paper (`Monocular_Depth_Estimation_Paper/main.tex`)

The main paper file gained one new methods subsection, one new results
subsection (with table), one new discussion subsection, and supporting edits to
the abstract, introduction, conclusion, and bibliography. No previously
reported numbers were changed.

| Section                                | What changed                                                                                                                                                                                 |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Abstract                               | Added a sentence introducing the LLM-supervisor protocol and a sentence summarizing the headline result (80% lookup, 0% threshold, 33.3% overall). Added the keywords *Large Language Models* and *LLM-as-Supervisor*. |
| Introduction (last paragraph)          | Added a new paragraph framing the LLM-supervisor contribution as a direct response to safety-critical AI validation requirements.                                                            |
| Methods §2.7 (new)                     | New subsection **"LLM-Supervisor Validation Protocol"** describing ground-truth construction, the six-category question taxonomy, prompting / inference settings (temperature 0, 8 192 ctx, 256 max tokens), and the automatic grader (5% relative tolerance for floats; exact integer / case-insensitive string match). |
| Experimental Results intro             | Replaced the wording *three-stage system* with a description that also names the proximity-monitoring telemetry and the new LLM-supervisor benchmark.                                        |
| Results §3.5 (new)                     | New subsection **"LLM-Supervisor Evaluation with Llama 3.1 8B"** reporting the run setup, the headline 10/30 = 33.3% number, the per-category capability gradient (with Table 4), and an enumerated discussion of which question types succeed and fail. |
| Table 4 (new, `tab:llm_supervisor`)    | Per-category accuracy of the supervisor.                                                                                                                                                     |
| Discussion §4.1 (new)                  | New subsection **"Implications of the LLM-Supervisor Evaluation"** translating the empirical finding into three concrete deployment guidelines. |
| Conclusion                             | Added a final paragraph summarizing the LLM-supervisor contribution and pointing readers to the released `process_video_headless.py` and `llm_supervisor_eval.py`.                            |
| Bibliography                           | Added two entries: `dubey2024llama` (Dubey et al., 2024 — Llama 3 herd of models) and `ollama2024` (Ollama software).                                                                         |

### Sanity checks performed on the LaTeX

- `\begin{abstract}` / `\end{abstract}`: 1/1
- `\begin{document}` / `\end{document}`: 1/1
- `\begin{table}` / `\end{table}`: 4/4 (unchanged figures, +1 new table)
- `\begin{figure}` / `\end{figure}`: 6/6 (unchanged)
- `\bibitem` count: 16 (was 14, +2 new)
- Linter (Cursor `read_lints`): no errors reported.

---

## 4. How to Reproduce This Run

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt ollama pandas

ollama pull llama3.1:8b

python process_video_headless.py \
    -i input/depth_estimation_1_20250817_212451.mp4 \
    --auto-calib \
    --depth-scale 5.0 \
    --frame-stride 3

python llm_supervisor_eval.py \
    --depth-csv    depth_logs/depth_log_*.csv \
    --distance-csv depth_logs/distance_log_*.csv \
    --model llama3.1:8b \
    --num-ctx 8192 \
    --temperature 0.0
```

Reports land in `llm_eval/`. With temperature 0 and the prompt's strict
`ANSWER: <value>` contract, the run is fully deterministic up to the underlying
GGUF kernel.

---

## 5. Limitations and Future Work Visible from This Experiment

The 33.3% overall figure is dominated by failures on threshold-filter and
long-context counting questions. Two natural follow-ups would tighten the
benchmark and likely raise the achievable score:

1. **Tool-augmented LLM agents.** Allow the LLM to issue pandas queries
   against the CSVs and re-run the same 30 questions. The benchmark would then
   measure tool-use accuracy, not raw arithmetic-over-tokens accuracy.
2. **Larger / fine-tuned models.** Re-run the harness with `llama3.1:70b`,
   `qwen2.5:32b`, or a model fine-tuned on tabular safety telemetry. The
   harness is model-agnostic (`--model <ollama-tag>`) so apples-to-apples
   comparisons require no additional code.

Both are scoped out of the present submission but explicitly invited by
Section 4.1 of the paper.
