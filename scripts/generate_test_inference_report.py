#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ae.dng import read_dng_preview_rgb


FLOAT_FIELDS = (
    "applied_ev", "target_ev", "predicted_target_ev", "absolute_error_ev",
    "acceptable_min_ev", "acceptable_max_ev",
)

OPTIONAL_FLOAT_FIELDS = (
    "predicted_hdr_probability", "hdr_decision_threshold",
)

OPTIONAL_INT_FIELDS = (
    "target_ev_mask", "target_hdr_enable", "hdr_label_mask",
    "predicted_hdr_enable", "hdr_correct",
)


def read_test_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row.get("split") == "test"]
    if not rows:
        raise ValueError(f"{path}: no test predictions found")
    parsed = []
    for row in rows:
        item = dict(row)
        for field in FLOAT_FIELDS:
            item[field] = float(item[field])
        for field in OPTIONAL_FLOAT_FIELDS:
            if item.get(field, "").strip():
                item[field] = float(item[field])
        for field in OPTIONAL_INT_FIELDS:
            if item.get(field, "").strip():
                item[field] = int(item[field])
        item["time_index"] = int(item["time_index"])
        item["exposure_index"] = int(item["exposure_index"])
        item["within_acceptable_interval"] = bool(int(item["within_acceptable_interval"]))
        item["signed_error_ev"] = item["predicted_target_ev"] - item["target_ev"]
        parsed.append(item)
    return sorted(parsed, key=lambda row: row["absolute_error_ev"], reverse=True)


def thumbnail_data_url(path: Path, width: int, quality: int,
                       preview_series: int) -> str:
    if path.suffix.lower() == ".dng":
        preview = Image.fromarray(read_dng_preview_rgb(path, preview_series), "RGB")
    else:
        preview = Image.open(path).convert("RGB")
    preview.thumbnail((width, width), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    preview.save(buffer, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def aggregate_time_steps(samples: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["scene_id"]].append(sample)
    result = []
    for scene_id, rows in grouped.items():
        errors = np.asarray([row["absolute_error_ev"] for row in rows], np.float64)
        result.append({
            "scene_id": scene_id,
            "time_index": rows[0]["time_index"],
            "sample_count": len(rows),
            "mean_absolute_error_ev": float(errors.mean()),
            "max_absolute_error_ev": float(errors.max()),
            "outside_acceptable_count": sum(
                not row["within_acceptable_interval"] for row in rows
            ),
            "worst_filename": max(rows, key=lambda row: row["absolute_error_ev"])[
                "filename"
            ],
        })
    return sorted(result, key=lambda row: row["max_absolute_error_ev"], reverse=True)


def build_report(samples: list[dict], groups: list[dict], output: Path,
                 version: str, dataset: str = "Scene4") -> dict:
    errors = np.asarray([row["absolute_error_ev"] for row in samples], np.float64)
    outside = sum(not row["within_acceptable_interval"] for row in samples)
    worst = samples[0]
    return {
        "status": "PASS",
        "version": version,
        "dataset": dataset,
        "split": "test",
        "sample_count": len(samples),
        "time_step_count": len(groups),
        "mae_ev": float(errors.mean()),
        "median_absolute_error_ev": float(np.median(errors)),
        "p90_absolute_error_ev": float(np.quantile(errors, 0.90)),
        "max_absolute_error_ev": float(errors.max()),
        "outside_acceptable_count": outside,
        "within_acceptable_percent": 100.0 * (len(samples) - outside) / len(samples),
        "worst_sample": {
            key: worst[key] for key in (
                "scene_id", "filename", "time_index", "exposure_index",
                "target_ev", "predicted_target_ev", "absolute_error_ev",
                "acceptable_min_ev", "acceptable_max_ev",
            )
        },
        "report_html": str(output),
    }


def write_html(path: Path, report: dict, samples: list[dict], groups: list[dict]) -> None:
    template = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI AE __DATASET_TEXT__ — __VERSION_TEXT__ Test Inference Images</title>
  <style>
    :root { color-scheme:dark; --bg:#08111d; --panel:#101e30; --line:#29405b;
      --text:#edf4fc; --muted:#96abc3; --accent:#90abff; --good:#45d69b;
      --warn:#ffbd66; --bad:#ff6b78; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--text); background:radial-gradient(circle at 10% 0,#1a3556 0,#08111d 36%);
      font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { max-width:1480px; margin:auto; padding:38px 24px 70px; }
    h1 { margin:5px 0 8px; font-size:clamp(30px,5vw,52px); line-height:1.08; }
    h2 { margin:0 0 16px; font-size:22px; }
    .eyebrow { color:var(--accent); letter-spacing:.13em; text-transform:uppercase; font-weight:800; }
    .sub,.meta { color:var(--muted); }
    .summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:14px; margin:26px 0; }
    .stat,.panel { background:rgba(16,30,48,.95); border:1px solid var(--line); border-radius:16px;
      box-shadow:0 14px 36px rgba(0,0,0,.18); }
    .stat { padding:18px; }
    .stat b { display:block; font-size:24px; margin-top:3px; }
    .panel { padding:20px; margin-top:18px; }
    .controls { display:grid; grid-template-columns:2fr repeat(3,minmax(145px,1fr)); gap:12px; align-items:end; }
    label { color:var(--muted); font-size:13px; }
    input,select,button { width:100%; margin-top:5px; background:#091522; color:var(--text);
      border:1px solid var(--line); border-radius:9px; padding:9px 10px; font:inherit; }
    button { width:auto; cursor:pointer; background:#183252; padding-inline:18px; }
    .check { display:flex; gap:8px; align-items:center; padding:9px 0; }
    .check input { width:auto; margin:0; }
    .gallery { display:grid; grid-template-columns:repeat(auto-fill,minmax(285px,1fr)); gap:16px; margin-top:18px; }
    .case { position:relative; background:var(--panel); border:1px solid var(--line); border-radius:15px; overflow:hidden; }
    .case.outside { border-color:rgba(255,107,120,.8); box-shadow:0 0 0 1px rgba(255,107,120,.2); }
    .preview-grid { display:grid; grid-template-columns:repeat(3,1fr); background:#03080d; }
    figure { margin:0; min-width:0; border-right:1px solid #24364b; }
    figure:last-child { border-right:0; }
    figure img { width:100%; aspect-ratio:3/2; object-fit:contain; display:block; }
    figcaption { padding:5px 4px; min-height:39px; text-align:center; color:var(--muted); font-size:11px; background:#09131f; }
    .rank { position:absolute; z-index:1; left:10px; top:10px; padding:4px 8px; border-radius:7px;
      background:rgba(0,0,0,.78); font-weight:800; }
    .badge { position:absolute; z-index:1; right:10px; top:10px; padding:4px 8px; border-radius:7px;
      background:rgba(7,18,27,.86); color:var(--good); font-weight:800; }
    .outside .badge { color:#fff; background:rgba(202,49,66,.9); }
    .body { padding:14px; }
    .file { font-size:17px; font-weight:800; overflow-wrap:anywhere; }
    .error { font-size:24px; font-weight:850; margin:8px 0; color:var(--warn); }
    .outside .error { color:var(--bad); }
    .kv { display:grid; grid-template-columns:1fr 1fr; gap:5px 12px; font-variant-numeric:tabular-nums; }
    .kv span:nth-child(odd) { color:var(--muted); }
    .table-wrap { overflow:auto; }
    table { width:100%; border-collapse:collapse; }
    th,td { text-align:right; padding:9px 11px; border-bottom:1px solid var(--line); white-space:nowrap; }
    th:first-child,td:first-child { text-align:left; }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; }
    tr[data-scene] { cursor:pointer; }
    tr[data-scene]:hover { background:#162b44; }
    .bar { height:7px; border-radius:10px; background:#1f334b; min-width:120px; overflow:hidden; }
    .bar i { display:block; height:100%; background:linear-gradient(90deg,var(--warn),var(--bad)); }
    .gallery-head { display:flex; justify-content:space-between; align-items:end; gap:18px; flex-wrap:wrap; }
    #visibleCount { color:var(--accent); font-weight:750; }
    footer { color:var(--muted); margin-top:28px; font-size:13px; }
    @media(max-width:820px){ .controls{grid-template-columns:1fr 1fr;} }
    @media(max-width:520px){ main{padding:26px 14px 50px}.controls{grid-template-columns:1fr}.gallery{grid-template-columns:1fr;} }
  </style>
</head>
<body><main>
  <div class="eyebrow">__DATASET_TEXT__ · __VERSION_TEXT__ · Held-out test</div>
  <h1>Inference 圖片錯誤分析</h1>
  <p class="sub">依 absolute EV error 排序。每筆並排顯示實際輸入、prediction 對應的最近 bracket，以及人工 target bracket。紅框代表預測超出人工 acceptable exposure interval。</p>
  <section class="summary" id="summary"></section>

  <section class="panel"><h2>Time-step summary</h2><div class="table-wrap"><table>
    <thead><tr><th>Time step</th><th>Mean error</th><th>Max error</th><th>Outside</th><th>Worst image</th><th>Error scale</th></tr></thead>
    <tbody id="groups"></tbody></table></div></section>

  <section class="panel">
    <h2>圖片篩選</h2><div class="controls">
      <label>搜尋 filename / time step<input id="search" type="search" placeholder="例如 1P0A3305 或 t086"></label>
      <label>Time step<select id="scene"><option value="">全部 test time steps</option></select></label>
      <label>排序<select id="sort"><option value="error-desc">誤差：大到小</option><option value="error-asc">誤差：小到大</option><option value="time">時間順序</option><option value="exposure">曝光序號</option></select></label>
      <label>最低 absolute error：<span id="thresholdText">0.00 EV</span><input id="threshold" type="range" min="0" step="0.05" value="0"></label>
    </div><label class="check"><input id="outsideOnly" type="checkbox">只顯示超出 acceptable range 的圖片</label>
  </section>

  <section class="panel">
    <div class="gallery-head"><div><h2>Test images</h2><div id="visibleCount"></div></div><button id="showAll">顯示全部符合條件的圖片</button></div>
    <div class="gallery" id="gallery"></div>
  </section>
  <footer>圖片為 DNG embedded preview series 2 的縮圖。Predicted nearest 是把連續 predicted EV 映射到該 time step 最近的實拍 bracket，用於視覺比較，不是模型生成或重新曝光的影像。模型只接受 histogram、8×8 luma grid 與 camera-state features。</footer>
</main>
<script>
const report = __REPORT_JSON__;
const samples = __SAMPLES_JSON__;
const groups = __GROUPS_JSON__;
const thumbByFile = Object.fromEntries(samples.map(s=>[s.filename,s.thumbnail]));
const f = (v,n=3) => Number(v).toFixed(n);
const esc = s => String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cards = [
  ['Test images',report.sample_count], ['Time steps',report.time_step_count],
  ['MAE',f(report.mae_ev)+' EV'], ['Median error',f(report.median_absolute_error_ev)+' EV'],
  ['P90 error',f(report.p90_absolute_error_ev)+' EV'], ['Max error',f(report.max_absolute_error_ev)+' EV'],
  ['Outside acceptable',report.outside_acceptable_count], ['Within acceptable',f(report.within_acceptable_percent,2)+'%']
];
document.getElementById('summary').innerHTML = cards.map(([k,v])=>`<div class="stat"><span class="meta">${k}</span><b>${v}</b></div>`).join('');
const maxGroup = Math.max(...groups.map(g=>g.max_absolute_error_ev));
document.getElementById('groups').innerHTML = groups.map(g=>`<tr data-scene="${esc(g.scene_id)}"><td>${esc(g.scene_id)}</td>`+
  `<td>${f(g.mean_absolute_error_ev)}</td><td>${f(g.max_absolute_error_ev)}</td><td>${g.outside_acceptable_count}/${g.sample_count}</td>`+
  `<td>${esc(g.worst_filename)}</td><td><div class="bar"><i style="width:${100*g.max_absolute_error_ev/maxGroup}%"></i></div></td></tr>`).join('');
const sceneSelect=document.getElementById('scene');
[...groups].sort((a,b)=>a.time_index-b.time_index).forEach(g=>sceneSelect.insertAdjacentHTML('beforeend',`<option value="${esc(g.scene_id)}">${esc(g.scene_id)}</option>`));
const search=document.getElementById('search'), sort=document.getElementById('sort'), threshold=document.getElementById('threshold');
const outsideOnly=document.getElementById('outsideOnly'), gallery=document.getElementById('gallery'), showAll=document.getElementById('showAll');
threshold.max = String(Math.ceil(report.max_absolute_error_ev*20)/20);
let expanded=false;
function render(){
  const q=search.value.trim().toLowerCase(), min=Number(threshold.value), scene=sceneSelect.value;
  document.getElementById('thresholdText').textContent=f(min,2)+' EV';
  let rows=samples.filter(s=>(!q||(s.filename+' '+s.scene_id).toLowerCase().includes(q))&&(!scene||s.scene_id===scene)&&
    s.absolute_error_ev>=min&&(!outsideOnly.checked||!s.within_acceptable_interval));
  rows.sort((a,b)=>sort.value==='error-asc'?a.absolute_error_ev-b.absolute_error_ev:
    sort.value==='time'?a.time_index-b.time_index||a.exposure_index-b.exposure_index:
    sort.value==='exposure'?a.exposure_index-b.exposure_index||b.absolute_error_ev-a.absolute_error_ev:
    b.absolute_error_ev-a.absolute_error_ev);
  const shown=expanded?rows:rows.slice(0,60);
  document.getElementById('visibleCount').textContent=`顯示 ${shown.length} / ${rows.length} 張符合條件的圖片`;
  showAll.style.display=rows.length>60&&!expanded?'inline-block':'none';
  gallery.innerHTML=shown.map(s=>`<article class="case ${s.within_acceptable_interval?'':'outside'}">`+
    `<span class="rank">#${s.error_rank}</span><span class="badge">${s.within_acceptable_interval?'IN RANGE':'OUTSIDE'}</span>`+
    `<div class="preview-grid"><figure><img loading="lazy" src="${s.thumbnail}" alt="${esc(s.filename)} input preview"><figcaption>Input<br>e${s.exposure_index}</figcaption></figure>`+
    `<figure><img loading="lazy" src="${thumbByFile[s.predicted_reference_filename]}" alt="predicted nearest exposure"><figcaption>Predicted nearest<br>e${s.predicted_reference_exposure_index}</figcaption></figure>`+
    `<figure><img loading="lazy" src="${thumbByFile[s.target_reference_filename]}" alt="human target exposure"><figcaption>Human target<br>e${s.target_reference_exposure_index}</figcaption></figure></div><div class="body">`+
    `<div class="file">${esc(s.filename)}</div><div class="meta">${esc(s.scene_id)} · exposure index ${s.exposure_index}</div>`+
    `<div class="error">|error| ${f(s.absolute_error_ev)} EV</div><div class="kv">`+
    `<span>Target</span><b>${f(s.target_ev)} EV</b><span>Prediction</span><b>${f(s.predicted_target_ev)} EV</b>`+
    `<span>Signed error</span><b>${s.signed_error_ev>=0?'+':''}${f(s.signed_error_ev)} EV</b>`+
    `<span>Acceptable</span><b>${f(s.acceptable_min_ev)}–${f(s.acceptable_max_ev)}</b>`+
    `<span>Applied input</span><b>${f(s.applied_ev)} EV</b></div></div></article>`).join('');
}
[search,sort,threshold,outsideOnly,sceneSelect].forEach(el=>el.addEventListener('input',()=>{expanded=false;render();}));
showAll.addEventListener('click',()=>{expanded=true;render();});
document.querySelectorAll('tr[data-scene]').forEach(row=>row.addEventListener('click',()=>{sceneSelect.value=row.dataset.scene;expanded=true;render();document.getElementById('gallery').scrollIntoView({behavior:'smooth'});}));
render();
</script></body></html>"""
    document = template.replace(
        "__VERSION_TEXT__", str(report["version"])
    ).replace(
        "__DATASET_TEXT__", str(report.get("dataset", "Scene4"))
    ).replace(
        "__REPORT_JSON__", json.dumps(report, ensure_ascii=False)
    ).replace(
        "__SAMPLES_JSON__", json.dumps(samples, ensure_ascii=False)
    ).replace(
        "__GROUPS_JSON__", json.dumps(groups, ensure_ascii=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def read_hdr_operating_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.DictReader(file) if row.get("split") == "test"]
    parsed = []
    for row in rows:
        if not row.get("target_hdr_enable", "").strip():
            continue
        item = dict(row)
        item["time_index"] = int(item["time_index"])
        item["target_hdr_enable"] = int(item["target_hdr_enable"])
        item["predicted_hdr_enable"] = int(item["predicted_hdr_enable"])
        item["correct"] = int(item["correct"])
        for field in (
            "applied_ev", "target_ev", "distance_to_target_ev",
            "hdr_probability", "decision_threshold",
        ):
            item[field] = float(item[field])
        parsed.append(item)
    return sorted(parsed, key=lambda row: row["time_index"])


def build_hdr_cases(samples: list[dict], operating_rows: list[dict]) -> list[dict]:
    samples_by_scene: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        samples_by_scene[sample["scene_id"]].append(sample)
    cases = []
    for operating in operating_rows:
        scene_rows = sorted(
            samples_by_scene.get(operating["scene_id"], []),
            key=lambda row: row["exposure_index"],
        )
        if not scene_rows:
            raise ValueError(f"missing test images for {operating['scene_id']}")
        operating_sample = min(
            scene_rows,
            key=lambda row: abs(row["applied_ev"] - operating["applied_ev"]),
        )
        target = operating["target_hdr_enable"]
        threshold = operating["decision_threshold"]
        bracket = []
        for row in scene_rows:
            probability = float(row.get("predicted_hdr_probability", 0.0))
            predicted = int(probability >= threshold)
            bracket.append({
                "filename": row["filename"],
                "exposure_index": row["exposure_index"],
                "applied_ev": row["applied_ev"],
                "hdr_probability": probability,
                "predicted_hdr_enable": predicted,
                "correct": int(predicted == target),
                "is_operating_frame": row["filename"] == operating_sample["filename"],
                "thumbnail": row["thumbnail"],
            })
        cases.append({
            "scene_id": operating["scene_id"],
            "time_index": operating["time_index"],
            "target_hdr_enable": target,
            "hdr_probability": operating["hdr_probability"],
            "decision_threshold": threshold,
            "predicted_hdr_enable": operating["predicted_hdr_enable"],
            "correct": operating["correct"],
            "probability_margin": abs(operating["hdr_probability"] - threshold),
            "operating_filename": operating_sample["filename"],
            "operating_exposure_index": operating_sample["exposure_index"],
            "operating_applied_ev": operating["applied_ev"],
            "target_ev": operating["target_ev"],
            "distance_to_target_ev": operating["distance_to_target_ev"],
            "operating_thumbnail": operating_sample["thumbnail"],
            "bracket_wrong_count": sum(not row["correct"] for row in bracket),
            "bracket_count": len(bracket),
            "bracket": bracket,
        })
    return cases


def build_hdr_report(cases: list[dict], output: Path, version: str,
                     dataset: str = "Scene4") -> dict:
    if not cases:
        raise ValueError("no supervised HDR test operating points found")
    target = np.asarray([row["target_hdr_enable"] for row in cases], np.int64)
    predicted = np.asarray([row["predicted_hdr_enable"] for row in cases], np.int64)
    bracket_count = sum(row["bracket_count"] for row in cases)
    bracket_wrong = sum(row["bracket_wrong_count"] for row in cases)
    return {
        "status": "PASS",
        "version": version,
        "dataset": dataset,
        "split": "test",
        "operating_point": "one frame per time step nearest the SDR target EV",
        "time_step_count": len(cases),
        "hdr_on_count": int((target == 1).sum()),
        "hdr_off_count": int((target == 0).sum()),
        "operating_correct_count": int((predicted == target).sum()),
        "operating_accuracy": float((predicted == target).mean()),
        "true_positive": int(((predicted == 1) & (target == 1)).sum()),
        "true_negative": int(((predicted == 0) & (target == 0)).sum()),
        "false_positive": int(((predicted == 1) & (target == 0)).sum()),
        "false_negative": int(((predicted == 0) & (target == 1)).sum()),
        "all_bracket_frame_count": bracket_count,
        "all_bracket_wrong_count": bracket_wrong,
        "all_bracket_accuracy": float((bracket_count - bracket_wrong) / bracket_count),
        "report_html": str(output),
        "note": (
            "The images are captured SDR bracket previews used to explain the decision. "
            "They are not generated HDR merge outputs."
        ),
    }


def write_hdr_html(path: Path, report: dict, cases: list[dict]) -> None:
    template = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI AE __DATASET_TEXT__ — __VERSION_TEXT__ HDR Test Images</title>
  <style>
    :root { color-scheme:dark; --bg:#07121a; --panel:#10232d; --panel2:#0a1922;
      --line:#294654; --text:#eef8fb; --muted:#9bb6c1; --accent:#76d5ff;
      --on:#53dda4; --off:#ffc66d; --bad:#ff6f7d; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--text); background:radial-gradient(circle at 12% 0,#164357 0,#07121a 40%);
      font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { max-width:1540px; margin:auto; padding:38px 24px 70px; }
    h1 { margin:5px 0 9px; font-size:clamp(30px,5vw,52px); line-height:1.08; }
    h2 { margin:0; font-size:21px; }
    .eyebrow { color:var(--accent); letter-spacing:.13em; text-transform:uppercase; font-weight:800; }
    .sub,.muted { color:var(--muted); }
    .warning { padding:13px 15px; margin:20px 0; border:1px solid #735d35; border-radius:12px;
      background:#241d12; color:#ffdea6; }
    .summary { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:13px; margin:24px 0; }
    .stat,.controls,.case { background:rgba(16,35,45,.96); border:1px solid var(--line); border-radius:16px;
      box-shadow:0 15px 36px rgba(0,0,0,.18); }
    .stat { padding:17px; }
    .stat b { display:block; font-size:24px; margin-top:3px; }
    .controls { display:grid; grid-template-columns:2fr repeat(3,minmax(150px,1fr)); gap:12px;
      padding:17px; margin-bottom:18px; }
    label { color:var(--muted); font-size:13px; }
    input,select { width:100%; margin-top:5px; background:#07151d; color:var(--text);
      border:1px solid var(--line); border-radius:9px; padding:9px 10px; font:inherit; }
    .cases { display:grid; gap:18px; }
    .case { overflow:hidden; }
    .case.op-wrong { border-color:var(--bad); box-shadow:0 0 0 1px rgba(255,111,125,.22); }
    .case-head { display:flex; justify-content:space-between; gap:16px; align-items:center;
      padding:16px 18px; border-bottom:1px solid var(--line); }
    .badges { display:flex; flex-wrap:wrap; gap:7px; }
    .badge { padding:5px 9px; border-radius:8px; background:#193844; font-weight:800; }
    .badge.on { color:var(--on); } .badge.off { color:var(--off); }
    .badge.wrong { color:#fff; background:#a93949; }
    .case-main { display:grid; grid-template-columns:minmax(260px,440px) 1fr; gap:20px; padding:18px; }
    .hero { margin:0; background:#03090d; border-radius:12px; overflow:hidden; border:2px solid var(--accent); }
    .hero img { width:100%; max-height:300px; object-fit:contain; display:block; }
    .hero figcaption { padding:8px 10px; text-align:center; color:var(--muted); background:#091720; }
    .decision { display:grid; align-content:center; gap:13px; }
    .probability { font-size:36px; font-weight:850; }
    .bar { position:relative; height:16px; background:#1e3944; border-radius:20px; overflow:visible; }
    .bar .fill { height:100%; border-radius:20px; background:linear-gradient(90deg,#4aa5d0,var(--on)); }
    .bar .threshold { position:absolute; top:-5px; bottom:-5px; width:2px; background:white; }
    .bar-labels { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }
    .kv { display:grid; grid-template-columns:max-content 1fr; gap:7px 14px; font-variant-numeric:tabular-nums; }
    .kv span { color:var(--muted); }
    .bracket-title { display:flex; justify-content:space-between; gap:12px; padding:0 18px 10px; }
    .bracket { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:7px; padding:0 18px 18px; }
    .frame { position:relative; margin:0; min-width:0; overflow:hidden; background:var(--panel2);
      border:1px solid var(--line); border-radius:9px; }
    .frame.wrong { border:2px solid var(--bad); }
    .frame.operating { box-shadow:0 0 0 3px var(--accent); }
    .frame img { width:100%; aspect-ratio:3/2; object-fit:contain; display:block; }
    .frame figcaption { padding:5px; font-size:11px; color:var(--muted); text-align:center; }
    .frame .mark { position:absolute; top:5px; left:5px; padding:2px 5px; border-radius:5px;
      background:rgba(0,0,0,.8); font-size:10px; font-weight:800; }
    .legend { color:var(--muted); font-size:12px; }
    #visibleCount { color:var(--accent); font-weight:750; margin:10px 0 15px; }
    footer { color:var(--muted); margin-top:28px; font-size:13px; }
    @media(max-width:850px){ .controls{grid-template-columns:1fr 1fr}.case-main{grid-template-columns:1fr}.bracket{grid-template-columns:repeat(3,minmax(0,1fr));} }
    @media(max-width:520px){ main{padding:26px 13px 50px}.controls{grid-template-columns:1fr}.bracket{grid-template-columns:repeat(2,minmax(0,1fr));} }
  </style>
</head>
<body><main>
  <div class="eyebrow">__DATASET_TEXT__ · __VERSION_TEXT__ · Held-out HDR test</div>
  <h1>HDR enable 圖片比較</h1>
  <p class="sub">每張卡片是一個 held-out test time step。大圖是最接近人工 SDR target EV、實際用來判斷 HDR 的 frame；下方保留完整 15 張曝光供比較。紅框 bracket frame 表示若在該曝光直接判斷，HDR 分類會錯。</p>
  <div class="warning">這些圖片是原始 SDR exposure bracket，用來解釋 HDR enable decision；目前沒有 HDR merge output，因此頁面不能比較最終 HDR 合成畫質、ghosting 或 ratio。</div>
  <section class="summary" id="summary"></section>
  <section class="controls">
    <label>搜尋 time step / filename<input id="search" type="search" placeholder="例如 t090 或 1P0A...dng"></label>
    <label>人工 label<select id="label"><option value="">全部</option><option value="1">HDR On</option><option value="0">HDR Off</option></select></label>
    <label>操作點結果<select id="status"><option value="">全部</option><option value="wrong">只看錯誤</option><option value="correct">只看正確</option></select></label>
    <label>排序<select id="sort"><option value="bracket">任意曝光錯誤數</option><option value="margin">操作點 margin 小到大</option><option value="time">時間順序</option><option value="probability">HDR probability 大到小</option></select></label>
  </section>
  <div id="visibleCount"></div>
  <section class="cases" id="cases"></section>
  <footer>Probability threshold 只由 validation split 選擇。模型輸入是 DNG embedded preview series 2 提取的 histogram、8×8 luma grid 與 camera-state features；test 沒有參與模型更新或 threshold 選擇。</footer>
</main>
<script>
const report=__REPORT_JSON__;
const cases=__CASES_JSON__;
const f=(v,n=3)=>Number(v).toFixed(n);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct=v=>f(100*v,2)+'%';
const cards=[['Test time steps',report.time_step_count],['Human HDR On / Off',report.hdr_on_count+' / '+report.hdr_off_count],
  ['Operating accuracy',pct(report.operating_accuracy)],['Operating errors',report.time_step_count-report.operating_correct_count],
  ['All bracket accuracy',pct(report.all_bracket_accuracy)],['Wrong bracket frames',report.all_bracket_wrong_count+' / '+report.all_bracket_frame_count]];
document.getElementById('summary').innerHTML=cards.map(([k,v])=>`<div class="stat"><span class="muted">${k}</span><b>${v}</b></div>`).join('');
const search=document.getElementById('search'), label=document.getElementById('label'), status=document.getElementById('status'), sort=document.getElementById('sort');
function render(){
  const q=search.value.trim().toLowerCase();
  let rows=cases.filter(c=>(!q||(c.scene_id+' '+c.operating_filename+' '+c.bracket.map(x=>x.filename).join(' ')).toLowerCase().includes(q))&&
    (!label.value||String(c.target_hdr_enable)===label.value)&&(!status.value||(status.value==='correct')===Boolean(c.correct)));
  rows.sort((a,b)=>sort.value==='margin'?a.probability_margin-b.probability_margin:
    sort.value==='time'?a.time_index-b.time_index:sort.value==='probability'?b.hdr_probability-a.hdr_probability:
    b.bracket_wrong_count-a.bracket_wrong_count||a.probability_margin-b.probability_margin);
  document.getElementById('visibleCount').textContent=`顯示 ${rows.length} / ${cases.length} 個 test time steps`;
  document.getElementById('cases').innerHTML=rows.map(c=>{
    const human=c.target_hdr_enable?'ON':'OFF', predicted=c.predicted_hdr_enable?'ON':'OFF';
    const frames=c.bracket.map(x=>`<figure class="frame ${x.correct?'':'wrong'} ${x.is_operating_frame?'operating':''}" title="${esc(x.filename)} · ${f(x.applied_ev)} EV · p=${f(x.hdr_probability)}">`+
      `${x.is_operating_frame?'<span class="mark">OPERATING</span>':''}<img loading="lazy" src="${x.thumbnail}" alt="${esc(x.filename)}">`+
      `<figcaption>e${x.exposure_index} · p ${f(x.hdr_probability)}<br>${x.predicted_hdr_enable?'ON':'OFF'}${x.correct?'':' · WRONG'}</figcaption></figure>`).join('');
    return `<article class="case ${c.correct?'':'op-wrong'}"><header class="case-head"><div><h2>${esc(c.scene_id)}</h2><span class="muted">${esc(c.operating_filename)}</span></div>`+
      `<div class="badges"><span class="badge ${c.target_hdr_enable?'on':'off'}">Human ${human}</span><span class="badge ${c.predicted_hdr_enable?'on':'off'}">Pred ${predicted}</span>`+
      `<span class="badge ${c.correct?'':'wrong'}">${c.correct?'CORRECT':'WRONG'}</span></div></header>`+
      `<div class="case-main"><figure class="hero"><img loading="lazy" src="${c.operating_thumbnail}" alt="operating frame"><figcaption>Operating frame · e${c.operating_exposure_index} · ${f(c.operating_applied_ev)} EV</figcaption></figure>`+
      `<div class="decision"><div><span class="muted">HDR probability</span><div class="probability">${f(c.hdr_probability)}</div></div>`+
      `<div class="bar"><div class="fill" style="width:${100*c.hdr_probability}%"></div><i class="threshold" style="left:${100*c.decision_threshold}%"></i></div>`+
      `<div class="bar-labels"><span>0 · Off</span><span>threshold ${f(c.decision_threshold)}</span><span>1 · On</span></div>`+
      `<div class="kv"><span>Human label</span><b>HDR ${human}</b><span>Model decision</span><b>HDR ${predicted}</b>`+
      `<span>Probability margin</span><b>${f(c.probability_margin)}</b><span>SDR target distance</span><b>${f(c.distance_to_target_ev)} EV</b>`+
      `<span>Arbitrary-exposure errors</span><b>${c.bracket_wrong_count} / ${c.bracket_count}</b></div></div></div>`+
      `<div class="bracket-title"><b>完整曝光 bracket</b><span class="legend">藍框＝操作點；紅框＝該曝光下判斷錯誤</span></div><div class="bracket">${frames}</div></article>`;
  }).join('');
}
[search,label,status,sort].forEach(el=>el.addEventListener('input',render)); render();
</script></body></html>"""
    document = template.replace(
        "__VERSION_TEXT__", str(report["version"])
    ).replace(
        "__DATASET_TEXT__", str(report.get("dataset", "Scene4"))
    ).replace(
        "__REPORT_JSON__", json.dumps(report, ensure_ascii=False)
    ).replace(
        "__CASES_JSON__", json.dumps(cases, ensure_ascii=False)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def infer_version(predictions: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    report_path = predictions.with_name("report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("version"):
            return str(report["version"])
    return "unknown"


def infer_dataset(predictions: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    report_path = predictions.with_name("report.json")
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("dataset"):
            return str(report["dataset"])
    return "Scene4"


def generate(args: argparse.Namespace) -> dict[str, str]:
    all_samples = read_test_predictions(args.predictions)
    version = infer_version(args.predictions, getattr(args, "version", None))
    dataset = infer_dataset(args.predictions, getattr(args, "dataset", None))
    samples_by_scene: dict[str, list[dict]] = defaultdict(list)
    for sample in all_samples:
        samples_by_scene[sample["scene_id"]].append(sample)
    for sample in all_samples:
        scene_rows = samples_by_scene[sample["scene_id"]]
        target_reference = min(
            scene_rows, key=lambda row: abs(row["applied_ev"] - sample["target_ev"])
        )
        predicted_reference = min(
            scene_rows,
            key=lambda row: abs(row["applied_ev"] - sample["predicted_target_ev"]),
        )
        sample["target_reference_filename"] = target_reference["filename"]
        sample["target_reference_exposure_index"] = target_reference["exposure_index"]
        sample["predicted_reference_filename"] = predicted_reference["filename"]
        sample["predicted_reference_exposure_index"] = predicted_reference["exposure_index"]
        sample["predicted_reference_ev"] = predicted_reference["applied_ev"]
    for rank, sample in enumerate(all_samples, start=1):
        image_path = args.input / sample["filename"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        sample["error_rank"] = rank
        sample["thumbnail"] = thumbnail_data_url(
            image_path, args.thumbnail_width, args.jpeg_quality, args.preview_series
        )
    samples = [row for row in all_samples if row.get("target_ev_mask", 1)]
    if not samples:
        raise ValueError(f"{args.predictions}: no exposure-supervised test predictions found")
    samples.sort(key=lambda row: row["absolute_error_ev"], reverse=True)
    for rank, sample in enumerate(samples, start=1):
        sample["error_rank"] = rank
    groups = aggregate_time_steps(samples)
    report = build_report(samples, groups, args.output, version, dataset)
    write_html(args.output, report, samples, groups)
    summary_path = args.output.with_name(args.output.stem + "_summary.json")
    summary_path.write_text(
        json.dumps({**report, "time_steps": groups}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    result = {
        "sdr_test_image_report": str(args.output),
        "sdr_test_image_summary": str(summary_path),
    }

    hdr_predictions = getattr(args, "hdr_predictions", None)
    if hdr_predictions is None:
        hdr_predictions = args.predictions.with_name("hdr_predictions_at_target_ev.csv")
    hdr_output = getattr(args, "hdr_output", None)
    if hdr_output is None:
        hdr_output = args.output.with_name("test_hdr_inference_report.html")
    operating_rows = read_hdr_operating_predictions(hdr_predictions)
    if operating_rows:
        cases = build_hdr_cases(all_samples, operating_rows)
        hdr_report = build_hdr_report(cases, hdr_output, version, dataset)
        write_hdr_html(hdr_output, hdr_report, cases)
        hdr_summary_path = hdr_output.with_name(hdr_output.stem + "_summary.json")
        compact_cases = []
        for case in cases:
            compact = {key: value for key, value in case.items()
                       if key not in ("operating_thumbnail", "bracket")}
            compact["bracket"] = [
                {key: value for key, value in frame.items() if key != "thumbnail"}
                for frame in case["bracket"]
            ]
            compact_cases.append(compact)
        hdr_summary_path.write_text(
            json.dumps({**hdr_report, "time_steps": compact_cases},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        result.update({
            "hdr_test_image_report": str(hdr_output),
            "hdr_test_image_summary": str(hdr_summary_path),
        })
    if not getattr(args, "quiet", False):
        refresh_training_report_index(args.predictions, result)
    if not getattr(args, "quiet", False):
        print(json.dumps({**report, "image_reports": result},
                         indent=2, ensure_ascii=False))
    return result


def generate_training_reports(predictions: Path, hdr_predictions: Path,
                              input_path: Path, output_dir: Path,
                              version: str, dataset: str = "Scene4") -> dict[str, str]:
    return generate(argparse.Namespace(
        predictions=predictions,
        hdr_predictions=hdr_predictions,
        input=input_path,
        output=output_dir / "test_inference_report.html",
        hdr_output=output_dir / "test_hdr_inference_report.html",
        preview_series=2,
        thumbnail_width=480,
        jpeg_quality=82,
        version=version,
        dataset=dataset,
        quiet=True,
    ))


def refresh_training_report_index(predictions: Path,
                                  image_reports: dict[str, str]) -> None:
    report_path = predictions.with_name("report.json")
    history_path = predictions.with_name("loss_history.csv")
    html_path = predictions.with_name("report.html")
    if not report_path.exists() or not history_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["image_reports"] = image_reports
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    history = []
    with history_path.open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            history.append({
                key: int(value) if key == "epoch" else float(value)
                for key, value in row.items()
            })
    from train_scene4 import write_html_report

    write_html_report(html_path, report, history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a self-contained image report for held-out Scene4 test inference."
    )
    parser.add_argument("--predictions", type=Path,
                        default=ROOT / "outputs/scene4_training/predictions.csv")
    parser.add_argument("--input", type=Path, default=ROOT / "input/Scene4")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs/scene4_training/test_inference_report.html")
    parser.add_argument("--hdr-predictions", type=Path,
                        help="Defaults to hdr_predictions_at_target_ev.csv beside --predictions.")
    parser.add_argument("--hdr-output", type=Path,
                        help="Defaults to test_hdr_inference_report.html beside --output.")
    parser.add_argument("--version", help="Defaults to version in report.json beside predictions.")
    parser.add_argument("--dataset", help="Defaults to dataset in report.json beside predictions.")
    parser.add_argument("--preview-series", type=int, default=2)
    parser.add_argument("--thumbnail-width", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
