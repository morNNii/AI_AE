#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from generate_labels import ANNOTATION_FIELDS, read_csv, write_csv


HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__DATASET_NAME__ Exposure Labeler</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #111; color: #eee; }
    header { position: sticky; top: 0; z-index: 2; background: #1a1a1a; padding: 12px 18px;
             display: flex; gap: 12px; align-items: center; border-bottom: 1px solid #444; }
    button, select, input { font: inherit; padding: 7px 10px; color: #eee; background: #292929;
                            border: 1px solid #666; border-radius: 5px; }
    button { cursor: pointer; } button:hover { background: #3a3a3a; }
    #save { background: #176f45; border-color: #2aad71; }
    main { max-width: 1480px; margin: 0 auto; padding: 18px; }
    #sheet { width: 100%; height: auto; border: 1px solid #444; display: block; }
    .comparison { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 16px; }
    .comparison > div, .metrics { background: #1c1c1c; padding: 13px; border-radius: 8px; }
    .comparison strong { display: block; color: #8fb8ff; margin-bottom: 5px; }
    .controls { display: grid; grid-template-columns: repeat(3, minmax(230px, 1fr)); gap: 14px;
                margin-top: 18px; }
    .field { background: #1c1c1c; padding: 13px; border-radius: 8px; }
    .field label { display: block; margin-bottom: 8px; color: #bbb; }
    .field select { width: 100%; }
    .footer { margin-top: 16px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    #status { min-height: 24px; color: #7fe0a9; }
    #source { color: #f4c86a; }
    .metrics { margin-top: 14px; overflow: auto; }
    .hdr-panel { margin-top: 18px; padding: 16px; background: #172235; border: 1px solid #35567c;
                 border-radius: 9px; }
    .hdr-panel h2 { margin: 0 0 6px; }
    .hdr-grid { display: grid; grid-template-columns: repeat(4, minmax(180px, 1fr)); gap: 12px;
                margin-top: 13px; }
    .hdr-grid label { color: #b9cbe0; }
    .hdr-grid select, .hdr-grid input, .hdr-grid textarea { width: 100%; margin-top: 5px; }
    .hdr-grid textarea { min-height: 72px; resize: vertical; }
    #saveHdr { background: #315e9d; border-color: #77a8eb; }
    body.hdr-mode .sdr-only { display: none !important; }
    body.sdr-mode .hdr-only { display: none !important; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { padding: 7px 9px; border-bottom: 1px solid #3b3b3b; text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { color: #aaa; font-size: 12px; }
    tr.metric-choice { color: #73dcff; font-weight: 700; }
    tr.original-choice { background: #2b2a1d; }
    #applyMetric { background: #234e78; border-color: #4c91cb; }
    @media (max-width: 850px) { .controls, .comparison, .hdr-grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
<header>
  <button id="prev">← 上一組</button>
  <select id="scenePicker"></select>
  <button id="next">下一組 →</button>
  <button id="nextTodo" class="sdr-only">下一個未確認</button>
  <button id="nextHdrTodo" class="hdr-only">下一個未確認 HDR</button>
  <strong id="progress" class="sdr-only"></strong>
  <strong id="hdrProgress" class="hdr-only"></strong>
  <span id="source" class="sdr-only"></span>
</header>
<main>
  <img id="sheet" alt="__DATASET_NAME__ exposure contact sheet">
  <div class="comparison sdr-only">
    <div><strong>目前 label</strong><span id="originalSummary">無</span></div>
    <div><strong>Metric 建議</strong><span id="metricSummary">尚未產生 metrics</span></div>
  </div>
  <div class="controls sdr-only">
    <div class="field"><label for="dark">可接受的最暗曝光</label><select id="dark"></select></div>
    <div class="field"><label for="preferred">最佳曝光</label><select id="preferred"></select></div>
    <div class="field"><label for="bright">可接受的最亮曝光</label><select id="bright"></select></div>
  </div>
  <div class="footer sdr-only">
    <label>信心值 <input id="confidence" type="number" min="0" max="1" step="0.05" value="0.8"></label>
    <button id="restoreOriginal">還原原選擇</button>
    <button id="applyMetric">套用 metric 建議</button>
    <button id="single">只接受最佳曝光</button>
    <button id="save">儲存並前往下一組</button>
    <span id="status"></span>
  </div>
  <div class="metrics sdr-only">
    <strong>逐曝光品質指標</strong>
    <table><thead><tr><th>Exposure</th><th>Entropy ↑</th><th>Saturated ↓</th><th>Dark ↓</th><th>Mean luma</th><th>Score ↑</th></tr></thead>
      <tbody id="metricRows"></tbody></table>
  </div>
  <section class="hdr-panel hdr-only">
    <h2>Static HDR benefit / enable</h2>
    <div>假設場景靜止：較短曝光是否能救回重要高光，同時較長曝光保有重要暗部？不要因燈泡、太陽等不重要的小面積 clipping 就開 HDR。</div>
    <div class="hdr-grid">
      <label>HDR benefit
        <select id="hdrBenefit"><option value="">-- 請判斷 --</option><option value="0">0：完全沒有幫助</option><option value="0.25">0.25：幫助很小</option><option value="0.5">0.5：不確定／主觀</option><option value="0.75">0.75：明顯有幫助</option><option value="1">1：單張 SDR 明顯不足</option></select>
      </label>
      <label>HDR enable if static
        <select id="hdrEnable"><option value="">Unknown（mask loss）</option><option value="0">Off</option><option value="1">On</option></select>
      </label>
      <label>HDR ratio（目前建議 Unknown）
        <select id="hdrRatio"><option value="">Unknown</option><option value="2">2×</option><option value="4">4×</option><option value="8">8×</option></select>
      </label>
      <label>HDR anchor（目前可留白）<select id="hdrAnchor"></select></label>
      <label>HDR label confidence <input id="hdrConfidence" type="number" min="0" max="1" step="0.05" value="0.8"></label>
      <label style="grid-column:span 3">備註 <textarea id="hdrNotes" maxlength="500" placeholder="例如：窗外高光可由短曝光救回；室內暗部仍需長曝光"></textarea></label>
    </div>
    <div class="footer"><button id="saveHdr">儲存 HDR 並前往下一個未確認</button><span id="hdrStatus"></span></div>
  </section>
</main>
<script>
let dataset = null;
let index = 0;
const ids = ["dark", "preferred", "bright"];
const el = id => document.getElementById(id);
const pct = value => value === undefined ? "—" : `${(100 * Number(value)).toFixed(2)}%`;
const num = value => value === undefined ? "—" : Number(value).toFixed(4);

function optionText(frame) {
  return `e${String(frame.exposure_index).padStart(2, "0")} | ${frame.shutter} | ${frame.filename}`;
}
function fillSelect(select, frames) {
  select.innerHTML = '<option value="">-- 請選擇 --</option>';
  for (const frame of frames) {
    const option = document.createElement("option");
    option.value = frame.filename;
    option.textContent = optionText(frame);
    select.appendChild(option);
  }
}
function render() {
  const scene = dataset.scenes[index];
  const suggestion = scene.metric_suggestion;
  el("scenePicker").value = String(index);
  el("sheet").src = scene.contact_url;
  for (const id of ids) fillSelect(el(id), scene.frames);
  el("dark").value = scene.annotation.acceptable_min_filename || "";
  el("preferred").value = scene.annotation.preferred_filename || "";
  el("bright").value = scene.annotation.acceptable_max_filename || "";
  const reviewed = scene.annotation.label_source === "human_review";
  el("confidence").value = scene.annotation.label_confidence || "0.8";
  el("source").textContent = reviewed ? "已完成第二次人工確認" :
    (scene.annotation.label_source === "metric_review_pending" ? "待第二次人工確認" : "自動建議，尚待人工確認");
  el("progress").textContent = `已確認 ${dataset.reviewed_count}/${dataset.scenes.length}`;
  el("hdrProgress").textContent = `HDR ${dataset.hdr_reviewed_count}/${dataset.scenes.length}`;
  if (suggestion) {
    el("originalSummary").textContent = `dark e${String(suggestion.original_acceptable_min_index).padStart(2,"0")} / preferred e${String(suggestion.original_preferred_index).padStart(2,"0")} / bright e${String(suggestion.original_acceptable_max_index).padStart(2,"0")}`;
    el("metricSummary").textContent = `dark e${String(suggestion.metric_acceptable_min_index).padStart(2,"0")} / preferred e${String(suggestion.metric_preferred_index).padStart(2,"0")} / bright e${String(suggestion.metric_acceptable_max_index).padStart(2,"0")}（Δ index ${Number(suggestion.preferred_index_delta) >= 0 ? "+" : ""}${suggestion.preferred_index_delta}）`;
    el("applyMetric").disabled = false;
  } else {
    el("originalSummary").textContent = "目前選擇已載入";
    el("metricSummary").textContent = "尚未產生 metrics";
    el("applyMetric").disabled = true;
  }
  el("metricRows").innerHTML = scene.frames.map(frame => {
    const classes = [];
    if (suggestion && frame.filename === suggestion.metric_preferred_filename) classes.push("metric-choice");
    if (suggestion && frame.filename === suggestion.original_preferred_filename) classes.push("original-choice");
    return `<tr class="${classes.join(" ")}"><td>${optionText(frame)}</td><td>${num(frame.entropy)}</td>` +
      `<td>${pct(frame.saturated_ratio)}</td><td>${pct(frame.dark_ratio)}</td><td>${num(frame.mean_luma)}</td><td>${num(frame.quality_score)}</td></tr>`;
  }).join("");
  fillSelect(el("hdrAnchor"), scene.frames);
  el("hdrBenefit").value = scene.annotation.hdr_benefit_if_static || "";
  el("hdrEnable").value = scene.annotation.hdr_enable_if_static || "";
  el("hdrRatio").value = scene.annotation.hdr_ratio_class || "";
  el("hdrAnchor").value = scene.annotation.hdr_anchor_filename || "";
  el("hdrConfidence").value = scene.annotation.hdr_label_confidence || "0.8";
  el("hdrNotes").value = scene.annotation.hdr_notes || "";
  el("hdrStatus").textContent = scene.annotation.hdr_reviewed === "1" ? "此組 HDR 已確認" : "";
  el("status").textContent = "";
}
function restoreOriginal() {
  const suggestion = dataset.scenes[index].metric_suggestion;
  if (!suggestion) return;
  el("dark").value = suggestion.original_acceptable_min_filename;
  el("preferred").value = suggestion.original_preferred_filename;
  el("bright").value = suggestion.original_acceptable_max_filename;
}
function applyMetric() {
  const suggestion = dataset.scenes[index].metric_suggestion;
  if (!suggestion) return;
  el("dark").value = suggestion.metric_acceptable_min_filename;
  el("preferred").value = suggestion.metric_preferred_filename;
  el("bright").value = suggestion.metric_acceptable_max_filename;
}
function move(delta) {
  index = (index + delta + dataset.scenes.length) % dataset.scenes.length;
  render();
}
function nextTodo() {
  for (let offset = 1; offset <= dataset.scenes.length; offset++) {
    const candidate = (index + offset) % dataset.scenes.length;
    if (dataset.scenes[candidate].annotation.label_source !== "human_review") {
      index = candidate; render(); return;
    }
  }
  el("status").textContent = "100 組都已人工確認。";
}
function nextHdrTodo() {
  for (let offset = 1; offset <= dataset.scenes.length; offset++) {
    const candidate = (index + offset) % dataset.scenes.length;
    if (dataset.scenes[candidate].annotation.hdr_reviewed !== "1") {
      index = candidate; render(); return;
    }
  }
  el("hdrStatus").textContent = "100 組 HDR 都已人工確認。";
}
async function save() {
  const scene = dataset.scenes[index];
  const payload = {
    scene_id: scene.scene_id,
    preferred_filename: el("preferred").value,
    acceptable_min_filename: el("dark").value,
    acceptable_max_filename: el("bright").value,
    label_confidence: Number(el("confidence").value),
  };
  const response = await fetch("/api/label", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) { el("status").textContent = result.error; return; }
  dataset = await (await fetch("/api/data")).json();
  el("status").textContent = "已儲存";
  nextTodo();
}
async function saveHdr() {
  const scene = dataset.scenes[index];
  const payload = {
    scene_id: scene.scene_id,
    hdr_benefit_if_static: el("hdrBenefit").value,
    hdr_enable_if_static: el("hdrEnable").value,
    hdr_ratio_class: el("hdrRatio").value,
    hdr_anchor_filename: el("hdrAnchor").value,
    hdr_label_confidence: Number(el("hdrConfidence").value),
    hdr_notes: el("hdrNotes").value,
  };
  const response = await fetch("/api/hdr-label", {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) { el("hdrStatus").textContent = result.error; return; }
  dataset = await (await fetch("/api/data")).json();
  el("hdrStatus").textContent = "HDR 已儲存";
  nextHdrTodo();
}

el("prev").onclick = () => move(-1);
el("next").onclick = () => move(1);
el("nextTodo").onclick = nextTodo;
el("nextHdrTodo").onclick = nextHdrTodo;
el("save").onclick = save;
el("saveHdr").onclick = saveHdr;
el("single").onclick = () => { el("dark").value = el("preferred").value; el("bright").value = el("preferred").value; };
el("restoreOriginal").onclick = restoreOriginal;
el("applyMetric").onclick = applyMetric;
el("hdrBenefit").onchange = event => {
  const value = Number(event.target.value);
  el("hdrEnable").value = !event.target.value || value === 0.5 ? "" : (value >= 0.75 ? "1" : "0");
  if (el("hdrEnable").value !== "1") { el("hdrRatio").value = ""; el("hdrAnchor").value = ""; }
};
el("scenePicker").onchange = event => { index = Number(event.target.value); render(); };

fetch("/api/data").then(response => response.json()).then(data => {
  dataset = data;
  document.body.classList.add(data.mode === "hdr" ? "hdr-mode" : "sdr-mode");
  data.scenes.forEach((scene, i) => {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = `${scene.scene_id} ${scene.annotation.label_source === "human_review" ? "✓" : ""}`;
    el("scenePicker").appendChild(option);
  });
  const firstTodo = data.mode === "hdr" ?
    data.scenes.findIndex(scene => scene.annotation.hdr_reviewed !== "1") :
    data.scenes.findIndex(scene => scene.annotation.label_source !== "human_review");
  index = firstTodo < 0 ? 0 : firstTodo;
  render();
});
</script>
</body>
</html>
"""


def format_shutter(microseconds: float) -> str:
    seconds = microseconds / 1_000_000.0
    return f"{seconds:g} s" if seconds >= 1 else f"1/{round(1 / seconds)} s"


class LabelState:
    def __init__(self, draft: Path, contacts: Path, metrics: Path | None = None,
                 suggestions: Path | None = None, mode: str = "sdr",
                 dataset_name: str = "Scene4"):
        self.frames_path = draft / "frame_metadata.csv"
        self.annotations_path = draft / "scene_annotations.csv"
        self.contacts = contacts.resolve()
        self.metrics_path = metrics or draft / "frame_metrics.csv"
        self.suggestions_path = suggestions or draft / "metric_suggestions.csv"
        self.mode = mode
        self.dataset_name = dataset_name
        if not self.frames_path.exists() or not self.annotations_path.exists():
            raise FileNotFoundError("run scripts/prepare_scene4.py before starting the labeler")

    def payload(self) -> dict:
        metrics = {
            (row["scene_id"], row["filename"]): row
            for row in read_csv(self.metrics_path)
        } if self.metrics_path.exists() else {}
        suggestions = {
            row["scene_id"]: row for row in read_csv(self.suggestions_path)
        } if self.suggestions_path.exists() else {}
        frames_by_scene: dict[str, list[dict]] = {}
        for row in read_csv(self.frames_path):
            frame = {
                "filename": row["filename"],
                "exposure_index": int(row["exposure_index"]),
                "exposure_time_us": float(row["exposure_time_us"]),
                "shutter": format_shutter(float(row["exposure_time_us"])),
            }
            metric = metrics.get((row["scene_id"], row["filename"]))
            if metric:
                for field in ("entropy", "saturated_ratio", "dark_ratio", "mean_luma",
                              "quality_score"):
                    frame[field] = float(metric[field])
            frames_by_scene.setdefault(row["scene_id"], []).append(frame)
        annotations = {row["scene_id"]: row for row in read_csv(self.annotations_path)}
        scenes = []
        for scene_id in sorted(frames_by_scene):
            frames = sorted(frames_by_scene[scene_id], key=lambda row: row["exposure_index"])
            scenes.append({
                "scene_id": scene_id,
                "frames": frames,
                "annotation": annotations[scene_id],
                "metric_suggestion": suggestions.get(scene_id),
                "contact_url": f"/contact/{scene_id}.jpg",
            })
        reviewed = sum(
            scene["annotation"].get("label_source") == "human_review" for scene in scenes
        )
        hdr_reviewed = sum(
            scene["annotation"].get("hdr_reviewed") == "1" for scene in scenes
        )
        return {
            "mode": self.mode,
            "dataset": self.dataset_name,
            "reviewed_count": reviewed,
            "hdr_reviewed_count": hdr_reviewed,
            "scenes": scenes,
        }

    def save(self, payload: dict) -> None:
        scene_id = str(payload.get("scene_id", ""))
        rows = read_csv(self.annotations_path)
        annotation = next((row for row in rows if row["scene_id"] == scene_id), None)
        if annotation is None:
            raise ValueError(f"unknown scene_id: {scene_id}")

        scene_frames = [
            row for row in read_csv(self.frames_path) if row["scene_id"] == scene_id
        ]
        exposure_by_name = {
            row["filename"]: float(row["exposure_time_us"]) for row in scene_frames
        }
        preferred = str(payload.get("preferred_filename", ""))
        endpoint_a = str(payload.get("acceptable_min_filename", ""))
        endpoint_b = str(payload.get("acceptable_max_filename", ""))
        for field, filename in (
            ("preferred", preferred), ("acceptable dark", endpoint_a),
            ("acceptable bright", endpoint_b),
        ):
            if filename not in exposure_by_name:
                raise ValueError(f"{field} exposure is missing or invalid")
        low, high = sorted((endpoint_a, endpoint_b), key=exposure_by_name.get)
        target_exposure = exposure_by_name[preferred]
        if not exposure_by_name[low] <= target_exposure <= exposure_by_name[high]:
            raise ValueError("最佳曝光必須落在可接受的最暗與最亮曝光之間")
        confidence = float(payload.get("label_confidence", 0.8))
        if not 0 <= confidence <= 1:
            raise ValueError("信心值必須介於 0 和 1")

        annotation.update({
            "preferred_filename": preferred,
            "acceptable_min_filename": low,
            "acceptable_max_filename": high,
            "label_confidence": f"{confidence:g}",
            "label_source": "human_review",
        })
        temporary = self.annotations_path.with_suffix(".tmp")
        write_csv(temporary, ANNOTATION_FIELDS, rows)
        temporary.replace(self.annotations_path)

    def save_hdr(self, payload: dict) -> None:
        scene_id = str(payload.get("scene_id", ""))
        rows = read_csv(self.annotations_path)
        annotation = next((row for row in rows if row["scene_id"] == scene_id), None)
        if annotation is None:
            raise ValueError(f"unknown scene_id: {scene_id}")
        benefit_text = str(payload.get("hdr_benefit_if_static", "")).strip()
        if not benefit_text:
            raise ValueError("請選擇 HDR benefit")
        benefit = float(benefit_text)
        if benefit not in (0.0, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("HDR benefit 必須是 0、0.25、0.5、0.75 或 1")

        enable_text = str(payload.get("hdr_enable_if_static", "")).strip()
        enable = None if not enable_text else int(enable_text)
        expected_enable = None if benefit == 0.5 else int(benefit >= 0.75)
        if enable != expected_enable:
            expected = "Unknown" if expected_enable is None else ("On" if expected_enable else "Off")
            raise ValueError(f"依標註規則，benefit {benefit:g} 的 enable 應為 {expected}")

        ratio_text = str(payload.get("hdr_ratio_class", "")).strip()
        ratio = None if not ratio_text else int(ratio_text)
        if ratio is not None and ratio not in (2, 4, 8):
            raise ValueError("HDR ratio 必須是 2、4、8 或 Unknown")
        anchor = str(payload.get("hdr_anchor_filename", "")).strip()
        scene_filenames = {
            row["filename"] for row in read_csv(self.frames_path)
            if row["scene_id"] == scene_id
        }
        if anchor and anchor not in scene_filenames:
            raise ValueError("HDR anchor 不屬於此 time step")
        if (ratio is not None or anchor) and enable != 1:
            raise ValueError("只有 HDR On 才能填 ratio 或 anchor")

        confidence = float(payload.get("hdr_label_confidence", 0.8))
        if not 0 <= confidence <= 1:
            raise ValueError("HDR label confidence 必須介於 0 和 1")
        notes = str(payload.get("hdr_notes", "")).strip()
        if len(notes) > 500:
            raise ValueError("HDR 備註不可超過 500 字")
        annotation.update({
            "hdr_benefit_if_static": f"{benefit:g}",
            "hdr_enable_if_static": "" if enable is None else str(enable),
            "hdr_ratio_class": "" if ratio is None else str(ratio),
            "hdr_anchor_filename": anchor,
            "hdr_reviewed": "1",
            "hdr_label_confidence": f"{confidence:g}",
            "hdr_notes": notes,
        })
        temporary = self.annotations_path.with_suffix(".tmp")
        write_csv(temporary, ANNOTATION_FIELDS, rows)
        temporary.replace(self.annotations_path)


class Handler(BaseHTTPRequestHandler):
    state: LabelState

    def send_bytes(self, body: bytes, content_type: str, status: int = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, value: dict, status: int = HTTPStatus.OK) -> None:
        self.send_bytes(json.dumps(value, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            document = HTML.replace("__DATASET_NAME__", self.state.dataset_name)
            self.send_bytes(document.encode("utf-8"), "text/html; charset=utf-8")
            return
        if route == "/api/data":
            self.send_json(self.state.payload())
            return
        if route.startswith("/contact/"):
            name = unquote(route.removeprefix("/contact/"))
            if Path(name).name != name:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            path = self.state.contacts / name
            if path.is_file():
                self.send_bytes(path.read_bytes(), "image/jpeg")
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in ("/api/label", "/api/hdr-label"):
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 65_536:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            if route == "/api/hdr-label":
                self.state.save_hdr(payload)
            else:
                self.state.save(payload)
            self.send_json({"status": "saved"})
        except (ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args) -> None:
        return


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local browser UI for exposure labels.")
    parser.add_argument("--draft", type=Path, default=ROOT / "labels/scene4_draft")
    parser.add_argument("--contact-sheets", type=Path,
                        default=ROOT / "outputs/scene4_labeling/contact_sheets")
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--suggestions", type=Path, default=None)
    parser.add_argument("--mode", choices=("sdr", "hdr"), default="sdr")
    parser.add_argument("--dataset-name", default="Scene4")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = LabelState(
        args.draft, args.contact_sheets, args.metrics, args.suggestions, args.mode,
        args.dataset_name,
    )
    Handler.state = state
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"{args.dataset_name} labeler: {url}")
    print("Press Ctrl+C to stop. Labels are saved after every scene.")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
