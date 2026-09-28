#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_ae.model import MultiHeadMLP


def portable_path(path: Path) -> str:
    """Store project paths without embedding a developer machine location."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def required_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "").strip()
    if not value:
        raise ValueError(f"{row.get('scene_id')}/{row.get('filename')}: missing {key}")
    return float(value)


def optional_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "").strip()
    return default if not value else float(value)


def split_name(time_index: int, train_end: int, validation_end: int) -> str:
    if time_index < train_end:
        return "train"
    if time_index < validation_end:
        return "validation"
    return "test"


def write_split_manifests(rows: list[dict[str, str]], time_indices: np.ndarray,
                          split_names: np.ndarray, output: Path) -> dict[str, dict]:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) + ["split"]
    partitions: dict[str, list[dict[str, str]]] = {
        "train": [], "validation": [], "test": [],
    }
    for row, time_index, name in zip(rows, time_indices, split_names):
        name = str(name)
        partitions[name].append({**row, "split": name})

    summary = {}
    for name, partition in partitions.items():
        path = output / f"{name}.csv"
        with path.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(partition)
        partition_times = [int(row["time_index"]) for row in partition]
        summary[name] = {
            "path": portable_path(path),
            "sample_count": len(partition),
            "time_step_count": len({row["scene_id"] for row in partition}),
            "time_index_min": min(partition_times),
            "time_index_max": max(partition_times),
        }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def resolve_sample_splits(scene_ids: np.ndarray, time_indices: np.ndarray,
                          split_assignments: Path | None, train_end: int,
                          validation_end: int) -> tuple[np.ndarray, dict[str, str]]:
    if split_assignments is None:
        names = np.asarray([
            split_name(int(index), train_end, validation_end)
            for index in time_indices
        ])
        return names, {
            "train": f"t000-t{train_end - 1:03d}",
            "validation": f"t{train_end:03d}-t{validation_end - 1:03d}",
            "test": f"t{validation_end:03d}-t{int(time_indices.max()):03d}",
        }
    assignment_rows = read_csv(split_assignments)
    assignment = {row["scene_id"]: row["split"] for row in assignment_rows}
    if len(assignment) != len(assignment_rows):
        raise ValueError(f"{split_assignments}: duplicate scene_id")
    missing = sorted(set(scene_ids) - set(assignment))
    if missing:
        raise ValueError(
            f"{split_assignments}: missing {len(missing)} scenes; first={missing[0]}"
        )
    names = np.asarray([assignment[scene_id] for scene_id in scene_ids])
    unknown = sorted(set(names) - {"train", "validation", "test"})
    if unknown:
        raise ValueError(f"{split_assignments}: invalid split names {unknown}")
    return names, {
        "train": "scene-level stratified assignment",
        "validation": "scene-level stratified assignment",
        "test": "scene-level stratified assignment",
    }


def evaluate(prediction: np.ndarray, target: np.ndarray, acceptable_min: np.ndarray,
             acceptable_max: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    error = prediction[mask] - target[mask]
    interval_error = np.maximum(acceptable_min[mask] - prediction[mask], 0.0)
    interval_error += np.maximum(prediction[mask] - acceptable_max[mask], 0.0)
    return {
        "sample_count": int(mask.sum()),
        "mae_ev": float(np.mean(np.abs(error))),
        "rmse_ev": float(np.sqrt(np.mean(error ** 2))),
        "within_acceptable_interval": float(np.mean(interval_error == 0)),
        "mean_interval_error_ev": float(np.mean(interval_error)),
    }


def evaluate_binary(probability: np.ndarray, target: np.ndarray,
                    mask: np.ndarray,
                    threshold: float = 0.5) -> dict[str, float | int | None]:
    valid = mask.astype(bool)
    count = int(valid.sum())
    if not count:
        return {"sample_count": 0}
    probability = probability[valid]
    target = target[valid].astype(np.int64)
    predicted = (probability >= threshold).astype(np.int64)
    true_positive = int(((predicted == 1) & (target == 1)).sum())
    true_negative = int(((predicted == 0) & (target == 0)).sum())
    false_positive = int(((predicted == 1) & (target == 0)).sum())
    false_negative = int(((predicted == 0) & (target == 1)).sum())
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    eps = 1e-7
    bce = -np.mean(
        target * np.log(probability + eps)
        + (1 - target) * np.log(1 - probability + eps)
    )
    return {
        "sample_count": count,
        "positive_count": int((target == 1).sum()),
        "negative_count": int((target == 0).sum()),
        "bce": float(bce),
        "threshold": float(threshold),
        "accuracy": float((predicted == target).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2.0),
        "f1": float(2 * precision * recall / max(precision + recall, 1e-12)),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_binary_threshold(probability: np.ndarray,
                            target: np.ndarray) -> dict[str, float]:
    probability = np.asarray(probability, np.float64)
    target = np.asarray(target, np.int64)
    if probability.size == 0 or len(np.unique(target)) < 2:
        return {"threshold": 0.5, "balanced_accuracy": 0.5}
    values = np.unique(probability)
    candidates = np.concatenate((
        np.asarray([0.0]),
        (values[:-1] + values[1:]) / 2.0,
        np.asarray([1.0]),
    ))
    best = None
    for threshold in candidates:
        metrics = evaluate_binary(
            probability, target, np.ones(len(target), dtype=bool), float(threshold)
        )
        candidate = (
            float(metrics["balanced_accuracy"]),
            -abs(float(threshold) - 0.5),
            float(threshold),
        )
        if best is None or candidate > best[0]:
            best = (candidate, metrics)
    assert best is not None
    return {
        "threshold": float(best[1]["threshold"]),
        "balanced_accuracy": float(best[1]["balanced_accuracy"]),
    }


def hdr_time_step_rows(probability: np.ndarray, target: np.ndarray,
                       hdr_mask: np.ndarray, time_indices: np.ndarray,
                       scene_ids: np.ndarray, split_mask: np.ndarray,
                       split_names: np.ndarray) -> list[dict]:
    rows = []
    for time_index in np.unique(time_indices[split_mask]):
        group = split_mask & (time_indices == time_index) & (hdr_mask > 0)
        if not group.any():
            continue
        mean_probability = float(probability[group].mean())
        target_value = int(round(float(target[group][0])))
        predicted = int(mean_probability >= 0.5)
        rows.append({
            "scene_id": str(scene_ids[group][0]),
            "time_index": int(time_index),
            "split": str(split_names[group][0]),
            "target_hdr_enable": target_value,
            "mean_hdr_probability": mean_probability,
            "min_hdr_probability": float(probability[group].min()),
            "max_hdr_probability": float(probability[group].max()),
            "predicted_hdr_enable": predicted,
            "correct": int(predicted == target_value),
        })
    return rows


def hdr_operating_point_rows(probability: np.ndarray, target_hdr: np.ndarray,
                             hdr_mask: np.ndarray, applied_ev: np.ndarray,
                             target_ev: np.ndarray, time_indices: np.ndarray,
                             scene_ids: np.ndarray, split_mask: np.ndarray,
                             split_names: np.ndarray) -> list[dict]:
    """Select the frame nearest the SDR target EV for each labeled time step."""
    rows = []
    for time_index in np.unique(time_indices[split_mask]):
        group = split_mask & (time_indices == time_index) & (hdr_mask > 0)
        indices = np.flatnonzero(group)
        if not len(indices):
            continue
        chosen = int(indices[np.argmin(np.abs(applied_ev[indices] - target_ev[indices]))])
        rows.append({
            "scene_id": str(scene_ids[chosen]),
            "time_index": int(time_index),
            "split": str(split_names[chosen]),
            "target_hdr_enable": int(round(float(target_hdr[chosen]))),
            "applied_ev": float(applied_ev[chosen]),
            "target_ev": float(target_ev[chosen]),
            "distance_to_target_ev": float(abs(applied_ev[chosen] - target_ev[chosen])),
            "hdr_probability": float(probability[chosen]),
        })
    return rows


def assess_overfit(history: list[dict[str, float | int]]) -> dict[str, float | int | str]:
    validation = np.asarray([row["validation_loss"] for row in history], np.float64)
    best_epoch = int(np.argmin(validation))
    best_loss = float(validation[best_epoch])
    final_loss = float(validation[-1])
    deterioration = (final_loss - best_loss) / max(best_loss, 1e-12)
    tail_start = max(0, int((len(history) - 1) * 0.9))
    tail_change = (final_loss - float(validation[tail_start])) / max(
        float(validation[tail_start]), 1e-12
    )
    final_train = float(history[-1]["train_loss"])
    generalization_gap = final_loss - final_train

    if best_epoch < int((len(history) - 1) * 0.9) and deterioration > 0.05:
        status = "overfit_observed"
        recommendation = (
            "Validation loss 已在較早 epoch 觸底後明顯回升；不要直接增加 epochs，"
            "應使用最佳 validation checkpoint 或加入 early stopping。"
        )
    elif best_epoch >= int((len(history) - 1) * 0.95) and tail_change < -0.01:
        status = "still_improving"
        recommendation = (
            "Validation loss 在訓練末段仍持續下降；可另開實驗小幅增加 epochs，"
            "但必須保留 validation 監控與最佳 checkpoint。"
        )
    else:
        status = "plateau_or_no_clear_overfit"
        recommendation = (
            "Validation loss 已接近平臺且沒有明顯惡化；不建議只靠增加 epochs。"
            "優先增加更多 Scene、改善特徵或調整模型容量。"
        )
    return {
        "status": status,
        "best_validation_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "final_validation_loss": final_loss,
        "final_train_loss": final_train,
        "final_generalization_gap": generalization_gap,
        "final_vs_best_validation_percent": deterioration * 100.0,
        "last_10_percent_validation_change_percent": tail_change * 100.0,
        "recommendation": recommendation,
    }


def write_loss_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_html_report(path: Path, report: dict,
                      history: list[dict[str, float | int]]) -> None:
    template = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AI AE __DATASET_TEXT__ — Training Report</title>
  <style>
    :root { color-scheme: dark; --bg:#07111f; --panel:#0f1d30; --line:#263951;
      --text:#e8f0fa; --muted:#93a8c2; --train:#55d6be; --val:#ffb454;
      --accent:#8ba7ff; --good:#3ddc97; --warn:#ffcc66; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(circle at top right,#17305a 0,#07111f 38%);
      color:var(--text); font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif; }
    main { max-width:1180px; margin:auto; padding:42px 24px 64px; }
    h1 { font-size:clamp(28px,5vw,52px); line-height:1.08; margin:8px 0; }
    h2 { font-size:22px; margin:0 0 18px; }
    .eyebrow { color:var(--accent); font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
    .sub { color:var(--muted); max-width:760px; }
    .grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); margin:28px 0; }
    .card,.panel { background:rgba(15,29,48,.94); border:1px solid var(--line); border-radius:16px;
      box-shadow:0 16px 45px rgba(0,0,0,.18); }
    .card { padding:19px; }
    .card .label { color:var(--muted); font-size:13px; }
    .card .value { font-size:25px; font-weight:750; margin-top:4px; }
    .panel { padding:22px; margin-top:18px; overflow:hidden; }
    .status { display:inline-block; padding:5px 10px; border-radius:999px;
      color:#06150e; background:var(--good); font-weight:750; }
    .assessment { border-left:4px solid var(--warn); padding:3px 0 3px 16px; }
    .chart-head { display:flex; align-items:center; justify-content:space-between; gap:16px; flex-wrap:wrap; }
    select { background:#091525; color:var(--text); border:1px solid var(--line); border-radius:8px; padding:7px 10px; }
    canvas { width:100%; height:390px; display:block; margin-top:12px; }
    .legend { display:flex; gap:20px; color:var(--muted); margin-top:8px; }
    .dot { width:10px; height:10px; display:inline-block; border-radius:50%; margin-right:7px; }
    table { width:100%; border-collapse:collapse; }
    th,td { padding:10px 12px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }
    th:first-child,td:first-child { text-align:left; }
    th { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
    .table-wrap { overflow-x:auto; }
    footer { color:var(--muted); margin-top:28px; font-size:13px; }
    code { color:#c9d7ff; }
    .report-links { display:flex; flex-wrap:wrap; gap:12px; }
    .report-links a { color:#061421; background:var(--accent); border-radius:9px;
      padding:9px 13px; text-decoration:none; font-weight:750; }
  </style>
</head>
<body><main>
  <div class="eyebrow" id="versionLabel"></div>
  <h1>AI Auto Exposure 訓練報告</h1>
  <p class="sub"><span id="datasetSummary"></span>。此報告完整保存在單一 HTML，不需要網路或外部圖表套件。</p>
  <span class="status" id="status"></span>

  <section class="grid" id="summaryCards"></section>

  <section class="panel">
    <div class="chart-head"><div><h2>Training / Validation Loss</h2>
      <div class="sub">每個 epoch 使用相同 composite objective；validation 僅監控，不參與梯度更新。</div></div>
      <label>Y 軸 <select id="scale"><option value="log">Log scale</option><option value="linear">Linear scale</option></select></label>
    </div>
    <canvas id="lossChart" aria-label="Training and validation loss chart"></canvas>
    <div class="legend"><span><i class="dot" style="background:var(--train)"></i>Training loss</span>
      <span><i class="dot" style="background:var(--val)"></i>Validation loss</span></div>
  </section>

  <section class="panel"><h2>Overfit 判讀</h2><div class="assessment" id="assessment"></div></section>

  <section class="panel"><h2>最終模型指標</h2><div class="table-wrap"><table>
    <thead><tr><th>Split</th><th>Samples</th><th>MAE (EV)</th><th>RMSE (EV)</th><th>Acceptable</th></tr></thead>
    <tbody id="metrics"></tbody></table></div></section>

  <section class="panel" id="hdrPanel"><h2>HDR enable 指標（AE 接近 target EV 的 operating point）</h2><div class="sub" id="hdrThreshold"></div><div class="table-wrap"><table>
    <thead><tr><th>Split</th><th>Scenes</th><th>On / Off</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>BCE</th></tr></thead>
    <tbody id="hdrMetrics"></tbody></table></div></section>

  <section class="panel" id="imageReportPanel"><h2>Inference / Test 圖片比較</h2>
    <p class="sub">每次訓練自動產生。SDR 頁面比較 predicted exposure 與人工 target；HDR 頁面比較人工 On/Off、模型 probability、threshold 與完整 exposure bracket。</p>
    <div class="report-links" id="imageReportLinks"></div></section>

  <section class="panel"><h2>資料切分</h2><div class="table-wrap"><table>
    <thead><tr><th>Split</th><th>Time range</th><th>Time steps</th><th>Samples</th></tr></thead>
    <tbody id="splits"></tbody></table></div></section>

  <section class="panel"><h2>Loss checkpoints</h2><div class="table-wrap"><table>
    <thead><tr><th>Epoch</th><th>Training loss</th><th>Validation loss</th><th>Train EV MSE</th><th>Validation EV MSE</th><th>Train HDR BCE</th><th>Validation HDR BCE</th></tr></thead>
    <tbody id="checkpoints"></tbody></table></div></section>

  <footer>Input domain: <code id="domain"></code> · <span id="hdrNote"></span> Test split 未用於選擇 epoch 或 HDR threshold。</footer>
</main>
<script>
const report = __REPORT_JSON__;
const history = __HISTORY_JSON__;
const fmt = (v, n=4) => Number(v).toFixed(n);
document.getElementById('versionLabel').textContent = report.dataset + ' · Version ' + report.version;
document.getElementById('datasetSummary').textContent =
  `${report.time_step_count} 個 scene、${report.sample_count.toLocaleString()} 筆 exposure samples`;
document.getElementById('status').textContent = report.status;
const cards = [
  ['版本','Version ' + report.version], ['Epochs',report.epochs.toLocaleString()],
  ['Best validation epoch',report.overfit_assessment.best_validation_epoch],
  ['Best validation loss',fmt(report.overfit_assessment.best_validation_loss)],
  ['Final validation loss',fmt(report.overfit_assessment.final_validation_loss)],
  ['Test MAE',fmt(report.metrics.test.mae_ev) + ' EV']
];
document.getElementById('summaryCards').innerHTML = cards.map(([k,v]) =>
  `<div class="card"><div class="label">${k}</div><div class="value">${v}</div></div>`).join('');
const oa = report.overfit_assessment;
document.getElementById('assessment').innerHTML = `<strong>${oa.status}</strong><br>${oa.recommendation}<br>` +
  `<span class="sub">Final train/validation gap: ${fmt(oa.final_generalization_gap)}；` +
  `final validation 相對最佳值 ${oa.final_vs_best_validation_percent >= 0 ? '+' : ''}${fmt(oa.final_vs_best_validation_percent,2)}%。</span>`;
for (const name of ['train','validation','test']) {
  const m = report.metrics[name];
  document.getElementById('metrics').insertAdjacentHTML('beforeend',
    `<tr><td>${name}</td><td>${m.sample_count}</td><td>${fmt(m.mae_ev)}</td>` +
    `<td>${fmt(m.rmse_ev)}</td><td>${fmt(m.within_acceptable_interval*100,2)}%</td></tr>`);
  const s = report.data_manifests[name];
  document.getElementById('splits').insertAdjacentHTML('beforeend',
    `<tr><td>${name}</td><td>${report.split[name]}</td><td>${s.time_step_count}</td><td>${s.sample_count}</td></tr>`);
}
if (report.hdr_metrics && report.hdr_supervised_time_steps > 0) {
  document.getElementById('hdrThreshold').textContent =
    `Decision threshold ${fmt(report.hdr_metrics.validation_selected_threshold,4)}，僅由 validation operating-point balanced accuracy 選定；任意 input 與 0.5 threshold 指標仍保存在 report.json。`;
  for (const name of ['train','validation','test']) {
    const m=report.hdr_metrics.operating_point.metrics[name];
    document.getElementById('hdrMetrics').insertAdjacentHTML('beforeend',
      `<tr><td>${name}</td><td>${m.sample_count}</td><td>${m.positive_count||0} / ${m.negative_count||0}</td>`+
      `<td>${fmt(m.accuracy*100,2)}%</td><td>${fmt(m.precision*100,2)}%</td><td>${fmt(m.recall*100,2)}%</td>`+
      `<td>${fmt(m.f1*100,2)}%</td><td>${fmt(m.bce)}</td></tr>`);
  }
} else document.getElementById('hdrPanel').style.display='none';
if (report.image_reports) {
  const links=[];
  if (report.image_reports.sdr_test_image_report) links.push(['SDR test 圖片報告','test_inference_report.html']);
  if (report.image_reports.hdr_test_image_report) links.push(['HDR test 圖片報告','test_hdr_inference_report.html']);
  document.getElementById('imageReportLinks').innerHTML=links.map(([label,href])=>`<a href="${href}">${label}</a>`).join('');
} else document.getElementById('imageReportPanel').style.display='none';
const best = oa.best_validation_epoch;
const selected = history.filter(r => r.epoch % 100 === 0 || r.epoch === best || r.epoch === history.length-1);
for (const r of selected) document.getElementById('checkpoints').insertAdjacentHTML('beforeend',
  `<tr${r.epoch===best?' style="color:var(--val);font-weight:700"':''}><td>${r.epoch}${r.epoch===best?' ★':''}</td>` +
  `<td>${fmt(r.train_loss,6)}</td><td>${fmt(r.validation_loss,6)}</td>` +
  `<td>${fmt(r.train_ev_mse,6)}</td><td>${fmt(r.validation_ev_mse,6)}</td>`+
  `<td>${fmt(r.train_hdr_bce,6)}</td><td>${fmt(r.validation_hdr_bce,6)}</td></tr>`);
document.getElementById('domain').textContent = report.input_domain;
document.getElementById('hdrNote').textContent = report.hdr_note;

const canvas = document.getElementById('lossChart'), scale = document.getElementById('scale');
function draw() {
  const cssW = Math.max(320, canvas.clientWidth), cssH = 390, dpr = window.devicePixelRatio || 1;
  canvas.width = cssW*dpr; canvas.height = cssH*dpr;
  const c = canvas.getContext('2d'); c.scale(dpr,dpr);
  const pad={l:66,r:22,t:20,b:44}, w=cssW-pad.l-pad.r, h=cssH-pad.t-pad.b;
  const vals = history.flatMap(r => [r.train_loss,r.validation_loss]);
  const transform = v => scale.value==='log' ? Math.log10(Math.max(v,1e-8)) : v;
  let yMin=Math.min(...vals.map(transform)), yMax=Math.max(...vals.map(transform));
  const margin=(yMax-yMin)*.06 || 1; yMin-=margin; yMax+=margin;
  const x = i => pad.l + i/(history.length-1)*w;
  const y = v => pad.t + (yMax-transform(v))/(yMax-yMin)*h;
  c.strokeStyle='#263951'; c.fillStyle='#93a8c2'; c.font='12px system-ui'; c.lineWidth=1;
  for(let i=0;i<=5;i++){ const yy=pad.t+i*h/5; c.beginPath(); c.moveTo(pad.l,yy); c.lineTo(cssW-pad.r,yy); c.stroke();
    const tv=yMax-i*(yMax-yMin)/5; const label=scale.value==='log'?Math.pow(10,tv).toPrecision(3):tv.toFixed(2);
    c.fillText(label,8,yy+4); }
  for(let i=0;i<=6;i++){ const xx=pad.l+i*w/6; c.fillText(String(Math.round((history.length-1)*i/6)),xx-10,cssH-14); }
  function line(key,color){ c.strokeStyle=color; c.lineWidth=2; c.beginPath(); history.forEach((r,i)=>{ const xx=x(i),yy=y(r[key]); i?c.lineTo(xx,yy):c.moveTo(xx,yy); }); c.stroke(); }
  line('train_loss','#55d6be'); line('validation_loss','#ffb454');
  const bx=x(best), by=y(history[best].validation_loss); c.fillStyle='#ffb454'; c.beginPath(); c.arc(bx,by,5,0,Math.PI*2); c.fill();
  c.fillText('best '+best,Math.min(bx+8,cssW-80),Math.max(by-8,15));
}
scale.addEventListener('change',draw); window.addEventListener('resize',draw); draw();
</script></body></html>"""
    document = template.replace(
        "__REPORT_JSON__", json.dumps(report, ensure_ascii=False)
    ).replace("__HISTORY_JSON__", json.dumps(history, ensure_ascii=False)).replace(
        "__DATASET_TEXT__", str(report.get("dataset", "Scene4"))
    )
    path.write_text(document, encoding="utf-8")


def train(args: argparse.Namespace) -> None:
    if not args.labels.exists():
        raise FileNotFoundError(
            f"{args.labels} does not exist; review labels, then run generate_labels.py build"
        )
    config = json.loads(args.config.read_text(encoding="utf-8"))
    label_rows = read_csv(args.labels)
    labels = {(row["scene_id"], row["filename"]): row for row in label_rows}

    with np.load(args.features) as cache:
        x = cache["features"].astype(np.float32)
        scene_ids = cache["scene_ids"].astype(str)
        filenames = cache["filenames"].astype(str)
        time_indices = cache["time_indices"].astype(np.int64)
        exposure_indices = cache["exposure_indices"].astype(np.int64)
        applied_ev = cache["applied_ev"].astype(np.float32)
        input_domain = str(cache["input_domain"])

    rows = []
    for scene_id, filename in zip(scene_ids, filenames):
        key = (scene_id, filename)
        if key not in labels:
            raise ValueError(f"feature cache has no matching label row: {scene_id}/{filename}")
        rows.append(labels[key])

    sources = {row.get("label_source", "").strip() for row in rows}
    if not args.allow_auto_labels and sources != {"human_review"}:
        human_scenes = {
            row["scene_id"] for row in rows if row.get("label_source", "") == "human_review"
        }
        raise ValueError(
            f"only {len(human_scenes)}/100 time steps are human-reviewed; "
            "finish scripts/label_scene4.py before training"
        )

    y_ev = np.asarray([required_float(row, "target_log_exposure") for row in rows], np.float32)
    ev_mask = np.asarray([
        optional_float(row, "target_ev_mask", default=1.0) for row in rows
    ], np.float32)
    acceptable_min = np.asarray([
        required_float(row, "acceptable_min_log_exposure") for row in rows
    ], np.float32)
    acceptable_max = np.asarray([
        required_float(row, "acceptable_max_log_exposure") for row in rows
    ], np.float32)
    y_hdr = np.asarray([optional_float(row, "hdr_enable_if_static") for row in rows], np.float32)
    hdr_mask = np.asarray([optional_float(row, "hdr_label_mask") for row in rows], np.float32)
    ratio_mask = np.asarray([optional_float(row, "hdr_ratio_mask") for row in rows], np.float32)
    ratio_classes = list(config["ratio_classes"])
    y_ratio = np.zeros(len(rows), np.int64)
    for index, row in enumerate(rows):
        value = row.get("hdr_ratio_class", "").strip()
        if value:
            y_ratio[index] = ratio_classes.index(int(float(value)))
    y_confidence = np.asarray([
        optional_float(row, "label_confidence") for row in rows
    ], np.float32)
    confidence_mask = np.asarray([
        bool(row.get("label_confidence", "").strip()) for row in rows
    ], np.float32)

    train_end = int(config["train_time_end"])
    validation_end = int(config["validation_time_end"])
    sample_splits, split_description = resolve_sample_splits(
        scene_ids, time_indices, args.split_assignments, train_end, validation_end
    )
    train_mask = sample_splits == "train"
    validation_mask = sample_splits == "validation"
    test_mask = sample_splits == "test"
    if not train_mask.any() or not validation_mask.any() or not test_mask.any():
        raise ValueError("train/validation/test split produced an empty partition")
    split_summary = write_split_manifests(
        rows, time_indices, sample_splits, args.split_output
    )

    model = MultiHeadMLP(
        x.shape[1], int(config["hidden_dim"]), len(ratio_classes), int(config["seed"])
    )
    epochs = int(args.epochs if args.epochs is not None else config["epochs"])
    learning_rate = float(config["learning_rate"])
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    best_params: dict[str, np.ndarray] | None = None
    for epoch in range(epochs + 1):
        if epoch:
            model.train(
                x[train_mask], y_ev[train_mask], y_hdr[train_mask],
                y_ratio[train_mask], y_confidence[train_mask], 1, learning_rate,
                ev_mask=ev_mask[train_mask], hdr_mask=hdr_mask[train_mask],
                ratio_mask=ratio_mask[train_mask],
                confidence_mask=confidence_mask[train_mask],
            )
        train_loss = model.loss_components(
            x[train_mask], y_ev[train_mask], y_hdr[train_mask],
            y_ratio[train_mask], y_confidence[train_mask],
            ev_mask=ev_mask[train_mask], hdr_mask=hdr_mask[train_mask],
            ratio_mask=ratio_mask[train_mask],
            confidence_mask=confidence_mask[train_mask],
        )
        validation_loss = model.loss_components(
            x[validation_mask], y_ev[validation_mask], y_hdr[validation_mask],
            y_ratio[validation_mask], y_confidence[validation_mask],
            ev_mask=ev_mask[validation_mask], hdr_mask=hdr_mask[validation_mask],
            ratio_mask=ratio_mask[validation_mask],
            confidence_mask=confidence_mask[validation_mask],
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss["total"],
            "validation_loss": validation_loss["total"],
            "train_ev_mse": train_loss["ev_mse"],
            "validation_ev_mse": validation_loss["ev_mse"],
            "train_hdr_bce": train_loss["hdr_bce"],
            "validation_hdr_bce": validation_loss["hdr_bce"],
        })
        if validation_loss["total"] < best_validation_loss:
            best_validation_loss = validation_loss["total"]
            best_params = {key: value.copy() for key, value in model.params.items()}
    model_output = model.predict(x)
    prediction = model_output["target_ev"]
    hdr_probability = model_output["hdr_benefit"]
    metrics = {
        "train": evaluate(
            prediction, y_ev, acceptable_min, acceptable_max,
            train_mask & (ev_mask > 0),
        ),
        "validation": evaluate(
            prediction, y_ev, acceptable_min, acceptable_max,
            validation_mask & (ev_mask > 0),
        ),
        "test": evaluate(
            prediction, y_ev, acceptable_min, acceptable_max,
            test_mask & (ev_mask > 0),
        ),
    }
    split_masks = {
        "train": train_mask,
        "validation": validation_mask,
        "test": test_mask,
    }
    hdr_sample_metrics_at_0_5 = {
        name: evaluate_binary(hdr_probability, y_hdr, mask & (hdr_mask > 0))
        for name, mask in split_masks.items()
    }
    all_hdr_time_rows = []
    hdr_time_rows_by_split = {}
    hdr_time_metrics_at_0_5 = {}
    for name, mask in split_masks.items():
        time_rows = hdr_time_step_rows(
            hdr_probability, y_hdr, hdr_mask, time_indices, scene_ids, mask,
            sample_splits,
        )
        hdr_time_rows_by_split[name] = time_rows
        all_hdr_time_rows.extend(time_rows)
        if time_rows:
            time_probability = np.asarray(
                [row["mean_hdr_probability"] for row in time_rows], np.float32
            )
            time_target = np.asarray(
                [row["target_hdr_enable"] for row in time_rows], np.float32
            )
            hdr_time_metrics_at_0_5[name] = evaluate_binary(
                time_probability, time_target, np.ones(len(time_rows), dtype=bool)
            )
        else:
            hdr_time_metrics_at_0_5[name] = {"sample_count": 0}

    validation_hdr_rows = hdr_time_rows_by_split["validation"]
    if validation_hdr_rows:
        validation_hdr_probability = np.asarray([
            row["mean_hdr_probability"] for row in validation_hdr_rows
        ], np.float32)
        validation_hdr_target = np.asarray([
            row["target_hdr_enable"] for row in validation_hdr_rows
        ], np.float32)
        threshold_selection = select_binary_threshold(
            validation_hdr_probability, validation_hdr_target
        )
    else:
        threshold_selection = {"threshold": 0.5, "balanced_accuracy": 0.5}
    hdr_threshold = float(threshold_selection["threshold"])
    hdr_sample_metrics = {
        name: evaluate_binary(
            hdr_probability, y_hdr, mask & (hdr_mask > 0), hdr_threshold
        )
        for name, mask in split_masks.items()
    }
    hdr_time_metrics = {}
    for name, time_rows in hdr_time_rows_by_split.items():
        if time_rows:
            time_probability = np.asarray([
                row["mean_hdr_probability"] for row in time_rows
            ], np.float32)
            time_target = np.asarray([
                row["target_hdr_enable"] for row in time_rows
            ], np.float32)
            hdr_time_metrics[name] = evaluate_binary(
                time_probability, time_target, np.ones(len(time_rows), dtype=bool),
                hdr_threshold,
            )
        else:
            hdr_time_metrics[name] = {"sample_count": 0}
    for row in all_hdr_time_rows:
        probability = float(row["mean_hdr_probability"])
        target_value = int(row["target_hdr_enable"])
        row["predicted_hdr_enable_0_5"] = row.pop("predicted_hdr_enable")
        row["correct_0_5"] = row.pop("correct")
        row["decision_threshold"] = hdr_threshold
        row["predicted_hdr_enable"] = int(probability >= hdr_threshold)
        row["correct"] = int(row["predicted_hdr_enable"] == target_value)

    all_hdr_operating_rows = []
    hdr_operating_rows_by_split = {}
    for name, mask in split_masks.items():
        operating_rows = hdr_operating_point_rows(
            hdr_probability, y_hdr, hdr_mask, applied_ev, y_ev, time_indices,
            scene_ids, mask, sample_splits,
        )
        hdr_operating_rows_by_split[name] = operating_rows
        all_hdr_operating_rows.extend(operating_rows)
    validation_operating_rows = hdr_operating_rows_by_split["validation"]
    if validation_operating_rows:
        operating_threshold_selection = select_binary_threshold(
            np.asarray([
                row["hdr_probability"] for row in validation_operating_rows
            ], np.float32),
            np.asarray([
                row["target_hdr_enable"] for row in validation_operating_rows
            ], np.float32),
        )
    else:
        operating_threshold_selection = {
            "threshold": 0.5, "balanced_accuracy": 0.5,
        }
    operating_threshold = float(operating_threshold_selection["threshold"])
    hdr_operating_metrics = {}
    for name, operating_rows in hdr_operating_rows_by_split.items():
        if operating_rows:
            op_probability = np.asarray([
                row["hdr_probability"] for row in operating_rows
            ], np.float32)
            op_target = np.asarray([
                row["target_hdr_enable"] for row in operating_rows
            ], np.float32)
            hdr_operating_metrics[name] = evaluate_binary(
                op_probability, op_target, np.ones(len(operating_rows), dtype=bool),
                operating_threshold,
            )
        else:
            hdr_operating_metrics[name] = {"sample_count": 0}
    for row in all_hdr_operating_rows:
        probability = float(row["hdr_probability"])
        target_value = int(row["target_hdr_enable"])
        row["decision_threshold"] = operating_threshold
        row["predicted_hdr_enable"] = int(probability >= operating_threshold)
        row["correct"] = int(row["predicted_hdr_enable"] == target_value)

    args.output.mkdir(parents=True, exist_ok=True)
    model.save(args.output / "model.npz")
    if best_params is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    np.savez(args.output / "model_best_validation.npz", **best_params)
    write_loss_history(args.output / "loss_history.csv", history)
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "scene_id", "filename", "time_index", "exposure_index", "split",
            "applied_ev", "target_ev", "predicted_target_ev", "absolute_error_ev",
            "target_ev_mask", "acceptable_min_ev", "acceptable_max_ev",
            "within_acceptable_interval",
            "target_hdr_enable", "hdr_label_mask", "predicted_hdr_probability",
            "predicted_hdr_enable_0_5", "hdr_correct_0_5",
            "hdr_decision_threshold", "predicted_hdr_enable", "hdr_correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            within = acceptable_min[index] <= prediction[index] <= acceptable_max[index]
            writer.writerow({
                "scene_id": scene_ids[index],
                "filename": filenames[index],
                "time_index": int(time_indices[index]),
                "exposure_index": int(exposure_indices[index]),
                "split": str(sample_splits[index]),
                "applied_ev": float(applied_ev[index]),
                "target_ev": float(y_ev[index]),
                "predicted_target_ev": float(prediction[index]),
                "absolute_error_ev": float(abs(prediction[index] - y_ev[index])),
                "target_ev_mask": int(ev_mask[index]),
                "acceptable_min_ev": float(acceptable_min[index]),
                "acceptable_max_ev": float(acceptable_max[index]),
                "within_acceptable_interval": int(within),
                "target_hdr_enable": "" if not hdr_mask[index] else int(y_hdr[index]),
                "hdr_label_mask": int(hdr_mask[index]),
                "predicted_hdr_probability": float(hdr_probability[index]),
                "predicted_hdr_enable_0_5": int(hdr_probability[index] >= 0.5),
                "hdr_correct_0_5": (
                    "" if not hdr_mask[index]
                    else int((hdr_probability[index] >= 0.5) == bool(y_hdr[index]))
                ),
                "hdr_decision_threshold": operating_threshold,
                "predicted_hdr_enable": int(hdr_probability[index] >= operating_threshold),
                "hdr_correct": (
                    "" if not hdr_mask[index]
                    else int(
                        (hdr_probability[index] >= operating_threshold)
                        == bool(y_hdr[index])
                    )
                ),
            })

    hdr_time_path = args.output / "hdr_predictions_by_time_step.csv"
    with hdr_time_path.open("w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "scene_id", "time_index", "split", "target_hdr_enable",
            "mean_hdr_probability", "min_hdr_probability", "max_hdr_probability",
            "predicted_hdr_enable_0_5", "correct_0_5", "decision_threshold",
            "predicted_hdr_enable", "correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_hdr_time_rows)

    hdr_operating_path = args.output / "hdr_predictions_at_target_ev.csv"
    with hdr_operating_path.open("w", newline="", encoding="utf-8-sig") as file:
        fields = [
            "scene_id", "time_index", "split", "target_hdr_enable",
            "applied_ev", "target_ev", "distance_to_target_ev", "hdr_probability",
            "decision_threshold", "predicted_hdr_enable", "correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_hdr_operating_rows)

    report = {
        "status": "PASS",
        "version": args.version,
        "dataset": args.dataset_name,
        "input_domain": input_domain,
        "sample_count": len(rows),
        "time_step_count": int(len(np.unique(time_indices))),
        "feature_dim": int(x.shape[1]),
        "epochs": epochs,
        "split": split_description,
        "data_manifests": split_summary,
        "loss_definition": (
            "masked EV MSE + 0.2 * masked HDR BCE + 0.2 * masked ratio CE + "
            "0.1 * confidence BCE"
        ),
        "loss_history_csv": portable_path(args.output / "loss_history.csv"),
        "best_validation_checkpoint": portable_path(
            args.output / "model_best_validation.npz"
        ),
        "initial_loss": history[0]["train_loss"],
        "final_loss": history[-1]["train_loss"],
        "initial_validation_loss": history[0]["validation_loss"],
        "final_validation_loss": history[-1]["validation_loss"],
        "overfit_assessment": assess_overfit(history),
        "metrics": metrics,
        "hdr_metrics": {
            "default_threshold": 0.5,
            "validation_selected_threshold": operating_threshold,
            "threshold_selection": {
                "split": "validation",
                "aggregation": "one frame per time step nearest the SDR target EV",
                "objective": "balanced_accuracy",
                **operating_threshold_selection,
            },
            "operating_point": {
                "definition": "one frame per time step nearest the SDR target EV",
                "metrics": hdr_operating_metrics,
                "predictions_csv": portable_path(hdr_operating_path),
            },
            "all_samples_at_operating_threshold": {
                name: evaluate_binary(
                    hdr_probability, y_hdr, mask & (hdr_mask > 0),
                    operating_threshold,
                )
                for name, mask in split_masks.items()
            },
            "time_step_mean_diagnostic": {
                "validation_selected_threshold": hdr_threshold,
                "threshold_selection": threshold_selection,
                "metrics": hdr_time_metrics,
                "predictions_csv": portable_path(hdr_time_path),
            },
            "samples_at_0_5": hdr_sample_metrics_at_0_5,
            "time_steps_at_0_5": hdr_time_metrics_at_0_5,
        },
        "hdr_supervised_samples": int(hdr_mask.sum()),
        "hdr_supervised_time_steps": int(len(np.unique(time_indices[hdr_mask > 0]))),
        "ev_supervised_samples": int(ev_mask.sum()),
        "ev_supervised_time_steps": int(len(np.unique(time_indices[ev_mask > 0]))),
        "hdr_note": (
            "HDR enable labels describe benefit if static. Ratio loss remains masked "
            "when no ratio labels are provided."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_html_report(args.output / "report.html", report, history)
    if not args.skip_image_reports:
        from generate_test_inference_report import generate_training_reports

        report["image_reports"] = generate_training_reports(
            predictions=args.output / "predictions.csv",
            hdr_predictions=hdr_operating_path,
            input_path=args.input,
            output_dir=args.output,
            version=args.version,
            dataset=args.dataset_name,
        )
        (args.output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        write_html_report(args.output / "report.html", report, history)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NumPy AE baseline.")
    parser.add_argument("--features", type=Path,
                        default=ROOT / "outputs/scene4_labeling/features.npz")
    parser.add_argument("--labels", type=Path,
                        default=ROOT / "labels/scene4_generated/training_samples.csv")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/scene4.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/scene4_training")
    parser.add_argument("--input", type=Path, default=ROOT / "input/Scene4",
                        help="DNG directory used to create test image comparison reports.")
    parser.add_argument("--split-output", type=Path,
                        default=ROOT / "labels/scene4_generated/splits")
    parser.add_argument("--split-assignments", type=Path,
                        help="Optional scene_id,split CSV; overrides time-index ranges.")
    parser.add_argument("--dataset-name", default="Scene4")
    parser.add_argument("--version", default="0")
    parser.add_argument("--epochs", type=int,
                        help="Override config epochs for a controlled comparison.")
    parser.add_argument("--skip-image-reports", action="store_true",
                        help="Skip automatic SDR/HDR test image HTML generation.")
    parser.add_argument("--allow-auto-labels", action="store_true",
                        help="Only for pipeline checks; human-reviewed labels are required by default.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
