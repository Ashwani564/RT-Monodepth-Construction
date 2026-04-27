#!/usr/bin/env python3
"""
LLM-Supervisor evaluation harness for the RT-MonoDepth + YOLOv11 pipeline.

Loads the depth/distance CSVs produced by ``process_video_headless.py`` (or by
``realtime_depth_video.py --log-depth --measure-distance``), defines 30
ground-truth questions about the perception output, queries a local Ollama
model (default: ``llama3.1:8b``), and grades the answers automatically.

The script writes:
    * llm_eval/llm_eval_report.json   - per-question records + aggregate metrics
    * llm_eval/llm_eval_summary.csv   - tidy CSV form for the paper / appendix
    * llm_eval/llm_eval_summary.md    - human-readable report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

try:
    import ollama
except ImportError as e:
    raise SystemExit(
        "The 'ollama' Python client is required. Install with: pip install ollama"
    ) from e


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_logs(depth_csv: Path, distance_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    depth = pd.read_csv(depth_csv)
    dist = pd.read_csv(distance_csv)
    detections = depth[(depth["object_class"] != "none") & (depth["depth_meters"] > 0)].copy()
    return depth, detections, dist


def build_context(detections: pd.DataFrame, distances: pd.DataFrame, max_rows: int = 200) -> str:
    """Serialize the detection and distance logs as a compact text context."""
    cols_det = [
        "frame_count",
        "object_class",
        "confidence",
        "depth_meters",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "center_x",
        "center_y",
    ]
    cols_dist = [
        "frame_count",
        "obj1_class",
        "obj2_class",
        "distance_3d_meters",
        "depth_difference",
        "obj1_depth",
        "obj2_depth",
    ]
    det_view = detections[cols_det].round(3)
    dist_view = distances[cols_dist].round(3)

    if len(det_view) > max_rows:
        det_view = det_view.head(max_rows)
    if len(dist_view) > max_rows:
        dist_view = dist_view.head(max_rows)

    parts: List[str] = []
    parts.append(
        "DETECTION LOG (each row = one valid detected object in one processed frame; "
        "depth is in meters, bbox is pixel coordinates):"
    )
    parts.append(det_view.to_csv(index=False))
    parts.append(
        "\nDISTANCE LOG (each row = pairwise 3D Euclidean distance between two objects in the same frame, in meters):"
    )
    parts.append(dist_view.to_csv(index=False))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Question definitions
# ---------------------------------------------------------------------------

@dataclass
class Question:
    qid: str
    category: str
    answer_type: str  # 'int', 'float', 'string'
    prompt: str
    truth_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame], Any]
    tolerance: float = 0.05  # relative tolerance for floats; absolute when value < 1
    accept: Optional[List[str]] = field(default=None)  # extra string aliases for grading


def q_counts(detections, distances) -> List[Question]:
    return [
        Question(
            qid="Q01",
            category="counts",
            answer_type="int",
            prompt="How many valid detections of class 'Person' are in the detection log?",
            truth_fn=lambda d, det, dist: int((det["object_class"] == "Person").sum()),
        ),
        Question(
            qid="Q02",
            category="counts",
            answer_type="int",
            prompt="How many valid detections of class 'Vehicle' are in the detection log?",
            truth_fn=lambda d, det, dist: int((det["object_class"] == "Vehicle").sum()),
        ),
        Question(
            qid="Q03",
            category="counts",
            answer_type="int",
            prompt="How many valid detections of class 'Machinery' are in the detection log?",
            truth_fn=lambda d, det, dist: int((det["object_class"] == "Machinery").sum()),
        ),
        Question(
            qid="Q04",
            category="counts",
            answer_type="int",
            prompt="How many distinct object classes appear in the detection log (excluding 'none')?",
            truth_fn=lambda d, det, dist: int(det["object_class"].nunique()),
        ),
        Question(
            qid="Q05",
            category="counts",
            answer_type="int",
            prompt="How many rows are in the distance log?",
            truth_fn=lambda d, det, dist: int(len(dist)),
        ),
    ]


def q_aggregates(detections, distances) -> List[Question]:
    return [
        Question(
            qid="Q06",
            category="aggregate-depth",
            answer_type="float",
            prompt="What is the mean depth in meters of all 'Person' detections? Reply with a single number rounded to 2 decimals.",
            truth_fn=lambda d, det, dist: round(float(det.loc[det["object_class"] == "Person", "depth_meters"].mean()), 2),
        ),
        Question(
            qid="Q07",
            category="aggregate-depth",
            answer_type="float",
            prompt="What is the maximum depth in meters of any 'Vehicle' detection? Reply with a single number rounded to 2 decimals.",
            truth_fn=lambda d, det, dist: round(float(det.loc[det["object_class"] == "Vehicle", "depth_meters"].max()), 2),
        ),
        Question(
            qid="Q08",
            category="aggregate-depth",
            answer_type="float",
            prompt="What is the minimum depth in meters of any 'Machinery' detection? Reply with a single number rounded to 2 decimals.",
            truth_fn=lambda d, det, dist: round(float(det.loc[det["object_class"] == "Machinery", "depth_meters"].min()), 2),
        ),
        Question(
            qid="Q09",
            category="aggregate-depth",
            answer_type="float",
            prompt="What is the median depth in meters across ALL valid detections? Reply with a single number rounded to 2 decimals.",
            truth_fn=lambda d, det, dist: round(float(det["depth_meters"].median()), 2),
        ),
        Question(
            qid="Q10",
            category="aggregate-confidence",
            answer_type="float",
            prompt="What is the mean detection confidence of all 'Person' rows? Reply with a single number rounded to 3 decimals.",
            truth_fn=lambda d, det, dist: round(float(det.loc[det["object_class"] == "Person", "confidence"].mean()), 3),
        ),
    ]


def q_lookups(detections, distances) -> List[Question]:
    # Pick deterministic lookups based on first detection rows for each class.
    person_first = detections[detections["object_class"] == "Person"].iloc[0]
    vehicle_first = detections[detections["object_class"] == "Vehicle"].iloc[0]
    machinery_first = detections[detections["object_class"] == "Machinery"].iloc[0]
    closest_dist_row = distances.sort_values("distance_3d_meters").iloc[0]
    farthest_dist_row = distances.sort_values("distance_3d_meters", ascending=False).iloc[0]

    return [
        Question(
            qid="Q11",
            category="lookup",
            answer_type="float",
            prompt=(
                f"In the detection log, what is the depth in meters of the FIRST 'Person' detection "
                f"(the row with the smallest frame_count where object_class is 'Person')? "
                f"Reply with a single number rounded to 2 decimals."
            ),
            truth_fn=lambda d, det, dist: round(float(person_first["depth_meters"]), 2),
        ),
        Question(
            qid="Q12",
            category="lookup",
            answer_type="int",
            prompt=(
                f"What is the frame_count of the first 'Vehicle' detection in the detection log? "
                f"Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: int(vehicle_first["frame_count"]),
        ),
        Question(
            qid="Q13",
            category="lookup",
            answer_type="int",
            prompt=(
                f"What is the frame_count of the first 'Machinery' detection in the detection log? "
                f"Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: int(machinery_first["frame_count"]),
        ),
        Question(
            qid="Q14",
            category="lookup",
            answer_type="float",
            prompt=(
                "What is the smallest 3D distance value in the distance log (in meters)? "
                "Reply with a single number rounded to 3 decimals."
            ),
            truth_fn=lambda d, det, dist: round(float(closest_dist_row["distance_3d_meters"]), 3),
        ),
        Question(
            qid="Q15",
            category="lookup",
            answer_type="float",
            prompt=(
                "What is the largest 3D distance value in the distance log (in meters)? "
                "Reply with a single number rounded to 3 decimals."
            ),
            truth_fn=lambda d, det, dist: round(float(farthest_dist_row["distance_3d_meters"]), 3),
        ),
    ]


def q_distance_reasoning(detections, distances) -> List[Question]:
    person_machinery = distances[
        ((distances["obj1_class"] == "Person") & (distances["obj2_class"] == "Machinery"))
        | ((distances["obj1_class"] == "Machinery") & (distances["obj2_class"] == "Person"))
    ]
    person_vehicle = distances[
        ((distances["obj1_class"] == "Person") & (distances["obj2_class"] == "Vehicle"))
        | ((distances["obj1_class"] == "Vehicle") & (distances["obj2_class"] == "Person"))
    ]
    danger_count = int((distances["distance_3d_meters"] < 1.0).sum())
    caution_count = int(
        ((distances["distance_3d_meters"] >= 1.0) & (distances["distance_3d_meters"] < 3.0)).sum()
    )
    safe_count = int((distances["distance_3d_meters"] >= 3.0).sum())

    pm_min = float(person_machinery["distance_3d_meters"].min()) if len(person_machinery) else float("nan")
    pv_min = float(person_vehicle["distance_3d_meters"].min()) if len(person_vehicle) else float("nan")
    dist_mean = float(distances["distance_3d_meters"].mean())

    return [
        Question(
            qid="Q16",
            category="proximity-zone",
            answer_type="int",
            prompt=(
                "Using the distance log, how many rows have a 3D distance strictly less than 1.0 meter "
                "(the system's 'Danger' zone)? Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: danger_count,
        ),
        Question(
            qid="Q17",
            category="proximity-zone",
            answer_type="int",
            prompt=(
                "Using the distance log, how many rows have a 3D distance in the half-open interval "
                "[1.0, 3.0) meters (the 'Caution' zone)? Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: caution_count,
        ),
        Question(
            qid="Q18",
            category="proximity-zone",
            answer_type="int",
            prompt=(
                "Using the distance log, how many rows have a 3D distance >= 3.0 meters (the 'Safe' zone)? "
                "Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: safe_count,
        ),
        Question(
            qid="Q19",
            category="proximity-zone",
            answer_type="float",
            prompt=(
                "Considering only Person-Machinery pairs in the distance log, what is the SMALLEST "
                "3D distance (in meters)? Reply with a single number rounded to 3 decimals."
            ),
            truth_fn=lambda d, det, dist: round(pm_min, 3),
        ),
        Question(
            qid="Q20",
            category="proximity-zone",
            answer_type="float",
            prompt=(
                "Considering only Person-Vehicle pairs in the distance log, what is the SMALLEST "
                "3D distance (in meters)? Reply with a single number rounded to 3 decimals."
            ),
            truth_fn=lambda d, det, dist: round(pv_min, 3),
        ),
        Question(
            qid="Q21",
            category="proximity-zone",
            answer_type="float",
            prompt=(
                "What is the mean 3D distance across ALL rows of the distance log (in meters)? "
                "Reply with a single number rounded to 3 decimals."
            ),
            truth_fn=lambda d, det, dist: round(dist_mean, 3),
        ),
    ]


def q_classification_reasoning(detections, distances) -> List[Question]:
    # which class has the highest mean confidence?
    mean_conf = detections.groupby("object_class")["confidence"].mean()
    top_conf_class = mean_conf.idxmax()
    # which class has highest mean depth?
    mean_depth = detections.groupby("object_class")["depth_meters"].mean()
    deepest_class = mean_depth.idxmax()
    closest_class = mean_depth.idxmin()
    # number of unique frames with at least one valid detection
    unique_frames = int(detections["frame_count"].nunique())
    # frame with most simultaneous detections
    frame_counts = detections["frame_count"].value_counts()
    busiest_frame = int(frame_counts.idxmax())
    busiest_n = int(frame_counts.max())

    return [
        Question(
            qid="Q22",
            category="reasoning",
            answer_type="string",
            prompt=(
                "Across 'Person', 'Vehicle', and 'Machinery', which object class has the HIGHEST mean "
                "detection confidence? Answer with one word: Person, Vehicle, or Machinery."
            ),
            truth_fn=lambda d, det, dist: str(top_conf_class),
            accept=[top_conf_class.lower()],
        ),
        Question(
            qid="Q23",
            category="reasoning",
            answer_type="string",
            prompt=(
                "Across 'Person', 'Vehicle', and 'Machinery', which object class has the LARGEST mean "
                "depth (i.e. is on average farthest from the camera)? Answer with one word: Person, Vehicle, or Machinery."
            ),
            truth_fn=lambda d, det, dist: str(deepest_class),
            accept=[deepest_class.lower()],
        ),
        Question(
            qid="Q24",
            category="reasoning",
            answer_type="string",
            prompt=(
                "Across 'Person', 'Vehicle', and 'Machinery', which object class has the SMALLEST mean "
                "depth (i.e. is on average closest to the camera)? Answer with one word: Person, Vehicle, or Machinery."
            ),
            truth_fn=lambda d, det, dist: str(closest_class),
            accept=[closest_class.lower()],
        ),
        Question(
            qid="Q25",
            category="reasoning",
            answer_type="int",
            prompt=(
                "How many DISTINCT frame_count values appear in the detection log among rows with a "
                "valid object class (i.e. excluding 'none')? Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: unique_frames,
        ),
        Question(
            qid="Q26",
            category="reasoning",
            answer_type="int",
            prompt=(
                "What is the frame_count of the BUSIEST frame in the detection log (the frame containing "
                "the largest number of valid detections)? Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: busiest_frame,
        ),
        Question(
            qid="Q27",
            category="reasoning",
            answer_type="int",
            prompt=(
                "How many valid detections occur in that busiest frame? Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: busiest_n,
        ),
    ]


def q_safety_decisions(detections, distances) -> List[Question]:
    # any danger event involving a person?
    person_danger = distances[
        (distances["distance_3d_meters"] < 1.0)
        & ((distances["obj1_class"] == "Person") | (distances["obj2_class"] == "Person"))
    ]
    has_person_danger = len(person_danger) > 0
    danger_pm = distances[
        (distances["distance_3d_meters"] < 1.0)
        & (
            ((distances["obj1_class"] == "Person") & (distances["obj2_class"] == "Machinery"))
            | ((distances["obj1_class"] == "Machinery") & (distances["obj2_class"] == "Person"))
        )
    ]
    n_danger_pm = int(len(danger_pm))

    return [
        Question(
            qid="Q28",
            category="safety-decision",
            answer_type="string",
            prompt=(
                "Acting as a safety supervisor: based on the distance log, did ANY pairwise distance "
                "involving a Person fall below 1.0 meter (the Danger threshold)? Answer 'yes' or 'no'."
            ),
            truth_fn=lambda d, det, dist: "yes" if has_person_danger else "no",
        ),
        Question(
            qid="Q29",
            category="safety-decision",
            answer_type="int",
            prompt=(
                "How many distance-log rows are simultaneously (i) Person-Machinery pairs AND "
                "(ii) below the Danger threshold of 1.0 meter? Reply with a single integer."
            ),
            truth_fn=lambda d, det, dist: n_danger_pm,
        ),
        Question(
            qid="Q30",
            category="safety-decision",
            answer_type="string",
            prompt=(
                "Given the data, would you (as a supervisor) recommend a workflow review on this site? "
                "Use this rule: recommend 'yes' if there is at least one distance-log row below 1.0 m "
                "involving a Person, otherwise 'no'. Answer 'yes' or 'no'."
            ),
            truth_fn=lambda d, det, dist: "yes" if has_person_danger else "no",
        ),
    ]


def build_questions(depth: pd.DataFrame, detections: pd.DataFrame, distances: pd.DataFrame) -> List[Question]:
    qs: List[Question] = []
    qs.extend(q_counts(detections, distances))
    qs.extend(q_aggregates(detections, distances))
    qs.extend(q_lookups(detections, distances))
    qs.extend(q_distance_reasoning(detections, distances))
    qs.extend(q_classification_reasoning(detections, distances))
    qs.extend(q_safety_decisions(detections, distances))
    assert len(qs) == 30, f"Expected 30 questions, got {len(qs)}"
    return qs


# ---------------------------------------------------------------------------
# LLM querying + grading
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a meticulous safety-supervisor analyst evaluating outputs from an AI vision system "
    "(monocular depth estimation + YOLOv11 object detection) deployed on a construction site. "
    "Your job is to answer numerical or short-answer questions strictly using the supplied "
    "detection log and distance log. Always show only the final answer in this exact format on the "
    "LAST line: 'ANSWER: <value>'. Do not invent numbers. If the answer is a number, give just the "
    "number without units. If categorical, give a single word."
)


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_answer(text: str, answer_type: str) -> Any:
    """Extract the model's answer from its raw text."""
    candidate: Optional[str] = None
    for line in reversed(text.strip().splitlines()):
        m = re.search(r"answer\s*[:=]\s*([^\n]+)", line, flags=re.IGNORECASE)
        if m:
            candidate = m.group(1).strip()
            break
    if candidate is None:
        candidate = text.strip().splitlines()[-1].strip() if text.strip() else ""

    candidate = candidate.strip("`*_ \t.")

    if answer_type in ("int", "float"):
        m = NUMBER_RE.search(candidate)
        if not m:
            m = NUMBER_RE.search(text)
        if not m:
            return None
        try:
            value = float(m.group(0))
            return int(round(value)) if answer_type == "int" else value
        except ValueError:
            return None

    return candidate.lower().strip().rstrip(".")


def grade(question: Question, predicted: Any, depth: pd.DataFrame, detections: pd.DataFrame,
          distances: pd.DataFrame) -> bool:
    truth = question.truth_fn(depth, detections, distances)
    if predicted is None:
        return False
    if question.answer_type == "int":
        try:
            return int(predicted) == int(truth)
        except (TypeError, ValueError):
            return False
    if question.answer_type == "float":
        try:
            pred = float(predicted)
            ref = float(truth)
        except (TypeError, ValueError):
            return False
        if abs(ref) < 1.0:
            return abs(pred - ref) <= max(question.tolerance, 0.05)
        rel_err = abs(pred - ref) / abs(ref)
        return rel_err <= question.tolerance
    truth_s = str(truth).lower().strip()
    pred_s = str(predicted).lower().strip()
    accept = {truth_s}
    if question.accept:
        accept.update(s.lower() for s in question.accept)
    return pred_s in accept or truth_s in pred_s


def query_llm(client_module, model: str, system_prompt: str, user_prompt: str,
              num_ctx: int = 8192, temperature: float = 0.0) -> str:
    response = client_module.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": 256,
        },
    )
    return response["message"]["content"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-csv", type=Path, required=True)
    parser.add_argument("--distance-csv", type=Path, required=True)
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("llm_eval"))
    parser.add_argument("--max-context-rows", type=int, default=200)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    depth, detections, distances = load_logs(args.depth_csv, args.distance_csv)
    print(f"Loaded depth log ({len(depth)} rows; {len(detections)} valid detections) and "
          f"distance log ({len(distances)} rows)")

    questions = build_questions(depth, detections, distances)
    context = build_context(detections, distances, max_rows=args.max_context_rows)

    print(f"Querying model={args.model}  num_ctx={args.num_ctx}  temperature={args.temperature}")
    print(f"Context length (chars): {len(context)}\n")

    records: List[Dict[str, Any]] = []
    correct = 0
    per_category: Dict[str, Dict[str, int]] = {}
    latencies: List[float] = []

    for i, q in enumerate(questions, start=1):
        truth = q.truth_fn(depth, detections, distances)
        user_prompt = (
            f"You will analyze the following two CSV-formatted logs.\n\n"
            f"{context}\n\n"
            f"Question {q.qid}: {q.prompt}\n"
            f"Remember: respond with reasoning if you must, but the LAST line MUST be 'ANSWER: <value>'."
        )
        t0 = time.time()
        try:
            raw = query_llm(
                ollama,
                args.model,
                SYSTEM_PROMPT,
                user_prompt,
                num_ctx=args.num_ctx,
                temperature=args.temperature,
            )
        except Exception as exc:  # noqa: BLE001
            raw = f"ERROR: {exc}"
        latency = time.time() - t0
        latencies.append(latency)

        predicted = parse_answer(raw, q.answer_type)
        ok = grade(q, predicted, depth, detections, distances)
        correct += int(ok)
        per_category.setdefault(q.category, {"correct": 0, "total": 0})
        per_category[q.category]["total"] += 1
        per_category[q.category]["correct"] += int(ok)

        record = {
            "qid": q.qid,
            "category": q.category,
            "answer_type": q.answer_type,
            "question": q.prompt,
            "ground_truth": truth,
            "predicted": predicted,
            "correct": ok,
            "latency_seconds": round(latency, 3),
            "raw_response": raw,
        }
        records.append(record)
        print(
            f"[{i:02d}/30] {q.qid} ({q.category}) -> truth={truth!r}  "
            f"pred={predicted!r}  {'OK' if ok else 'FAIL'}  ({latency:.1f}s)"
        )

    accuracy = correct / len(questions)
    summary = {
        "model": args.model,
        "num_questions": len(questions),
        "num_correct": correct,
        "accuracy": accuracy,
        "mean_latency_seconds": round(statistics.mean(latencies), 3) if latencies else 0.0,
        "median_latency_seconds": round(statistics.median(latencies), 3) if latencies else 0.0,
        "total_latency_seconds": round(sum(latencies), 3),
        "per_category": {
            cat: {
                "correct": v["correct"],
                "total": v["total"],
                "accuracy": round(v["correct"] / max(v["total"], 1), 3),
            }
            for cat, v in per_category.items()
        },
        "depth_log": str(args.depth_csv),
        "distance_log": str(args.distance_csv),
        "context_chars": len(context),
        "num_ctx": args.num_ctx,
        "temperature": args.temperature,
    }

    json_path = args.out_dir / "llm_eval_report.json"
    csv_path = args.out_dir / "llm_eval_summary.csv"
    md_path = args.out_dir / "llm_eval_summary.md"

    with json_path.open("w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2, default=str)

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)

    md_lines = [
        f"# LLM Supervisor Evaluation ({args.model})",
        "",
        f"- Depth log: `{args.depth_csv}` ({len(depth)} rows, {len(detections)} valid detections)",
        f"- Distance log: `{args.distance_csv}` ({len(distances)} rows)",
        f"- Total questions: **{len(questions)}**",
        f"- Correct: **{correct}** ({accuracy*100:.1f}%)",
        f"- Mean latency / question: {summary['mean_latency_seconds']}s",
        "",
        "## Per-category accuracy",
        "",
        "| Category | Correct | Total | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for cat, v in summary["per_category"].items():
        md_lines.append(f"| {cat} | {v['correct']} | {v['total']} | {v['accuracy']*100:.1f}% |")
    md_lines += [
        "",
        "## Per-question results",
        "",
        "| QID | Category | Ground truth | Predicted | Correct | Latency (s) |",
        "| --- | --- | --- | --- | :-: | ---: |",
    ]
    for r in records:
        md_lines.append(
            f"| {r['qid']} | {r['category']} | {r['ground_truth']} | {r['predicted']} | "
            f"{'✓' if r['correct'] else '✗'} | {r['latency_seconds']} |"
        )
    with md_path.open("w") as f:
        f.write("\n".join(md_lines) + "\n")

    print()
    print(f"Accuracy: {correct}/{len(questions)} = {accuracy*100:.1f}%")
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
