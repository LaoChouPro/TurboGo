#!/usr/bin/env python3
"""
Real-Time Web Dashboard Server for TurboGo-v2 (b11c384-nbt22) 100M Pretraining.
Minimalist Monochrome (Black & White + 1px Hairline) UI on port 8765 and JSON API at /api/status.
"""

import calendar
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG_PATH = Path("/content/train_turbogo_mini.log")
METRICS_PATH = Path("/content/checkpoints/turbogo_mini/metrics.jsonl")
CKPT_DIR = Path("/content/checkpoints/turbogo_mini")

STEP_RE = re.compile(
    r"\[Ep\s+(\d+)/(\d+)\s+\|\s+Chunk\s+(\d+)/(\d+)\s+\|\s+Step\s+(\d+)/(\d+)\]\s+"
    r"Loss:\s+([\d.]+)\s+\(Pol:([\d.]+)\s+Val:([\d.]+)\s+Own:([\d.]+)\)\s+\|\s+"
    r"Top1:\s+([\d.]+)%\s+\|\s+LR:\s+([\d.eE+-]+)\s+\|\s+Speed:\s+([\d,]+)\s+pos/s\s+\|\s+VRAM:\s+([\d.]+)G"
)


def collect_status() -> dict:
    gpu_info = {
        "name": "NVIDIA RTX PRO 6000 Blackwell",
        "util": 0,
        "mem_used_mb": 0,
        "mem_total_mb": 97887,
        "power_w": 0.0,
        "temp_c": 0,
    }
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=3,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 6:
            gpu_info = {
                "name": parts[0],
                "util": int(float(parts[1])),
                "mem_used_mb": int(float(parts[2])),
                "mem_total_mb": int(float(parts[3])),
                "power_w": round(float(parts[4]), 1),
                "temp_c": int(float(parts[5])),
            }
    except Exception:
        pass

    train_series = []
    log_tail = []
    latest_train = None
    if LOG_PATH.exists():
        try:
            with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            log_tail = [l.rstrip("\n") for l in lines[-45:]]
            for l in lines:
                m = STEP_RE.search(l)
                if m:
                    pt = {
                        "epoch": int(m.group(1)),
                        "total_epochs": int(m.group(2)),
                        "chunk": int(m.group(3)),
                        "total_chunks": int(m.group(4)),
                        "step": int(m.group(5)),
                        "total_steps": int(m.group(6)),
                        "loss": float(m.group(7)),
                        "pol_loss": float(m.group(8)),
                        "val_loss": float(m.group(9)),
                        "own_loss": float(m.group(10)),
                        "top1": float(m.group(11)),
                        "lr": float(m.group(12)),
                        "speed": int(m.group(13).replace(",", "")),
                        "vram_gb": float(m.group(14)),
                    }
                    train_series.append(pt)
            if train_series:
                latest_train = train_series[-1]
                if len(train_series) > 300:
                    stride = max(1, len(train_series) // 300)
                    sampled = train_series[::stride]
                    if sampled[-1]["step"] != train_series[-1]["step"]:
                        sampled.append(train_series[-1])
                    train_series = sampled
        except Exception:
            pass
    else:
        for dl_p in (Path("/content/bootstrap.log"), Path("/content/dl_50m.log")):
            if dl_p.exists():
                try:
                    with open(dl_p, "r", encoding="utf-8", errors="replace") as f:
                        log_tail = [l.rstrip("\n") for l in f.readlines()[-35:]]
                    break
                except Exception:
                    pass

    val_series = []
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        val_series.append(json.loads(line))
        except Exception:
            pass

    ckpts = []
    if CKPT_DIR.exists():
        for p in sorted(CKPT_DIR.glob("*.pt")):
            try:
                st = p.stat()
                ckpts.append(
                    {
                        "name": p.name,
                        "size_mb": round(st.st_size / (1024 * 1024), 1),
                        "mtime_cst": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime + 8 * 3600)),
                    }
                )
            except Exception:
                pass

    now_ts = time.time()
    if latest_train and latest_train["speed"] > 100:
        steps_left = max(0, latest_train["total_steps"] - latest_train["step"])
        steps_per_sec = latest_train["speed"] / 1024.0
        rem_sec = int(steps_left / max(0.1, steps_per_sec))
    else:
        target_ts = float(calendar.timegm(time.strptime("2026-10-29 14:30:00", "%Y-%m-%d %H:%M:%S")))
        rem_sec = max(0, int(target_ts - now_ts))

    return {
        "server_time_cst": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(now_ts + 8 * 3600)),
        "remaining_sec": rem_sec,
        "gpu": gpu_info,
        "latest_train": latest_train,
        "train_series": train_series,
        "val_series": val_series,
        "checkpoints": ckpts,
        "log_tail": log_tail,
    }


INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>TurboGo-Mini (3.06M) · 100M Pretraining Telemetry</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {
      --bg: #ffffff;
      --ink: #0a0a0a;
      --sub: #555555;
      --muted: #888888;
      --line: #d8d8d8;
      --line-strong: #111111;
      --surface: #fafafa;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "Helvetica Neue", "PingFang SC", sans-serif;
      padding: 24px 28px;
      max-width: 1480px;
      margin: 0 auto;
      -webkit-font-smoothing: antialiased;
    }
    header {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: flex-end;
      gap: 12px;
      border-top: 1.5px solid var(--line-strong);
      border-bottom: 1px solid var(--line-strong);
      padding: 14px 0;
      margin-bottom: 18px;
    }
    .title-group h1 {
      font-size: 17px;
      font-weight: 600;
      letter-spacing: -0.2px;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .dot {
      display: inline-block;
      width: 7px;
      height: 7px;
      background: var(--ink);
      border-radius: 50%;
    }
    .subtitle {
      color: var(--sub);
      font-size: 12px;
      margin-top: 5px;
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    }
    .meta-right {
      text-align: right;
      font-size: 12px;
      color: var(--sub);
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      line-height: 1.6;
    }
    .meta-right strong { color: var(--ink); font-weight: 600; }

    .progress-wrap {
      border-bottom: 1px solid var(--line);
      padding-bottom: 14px;
      margin-bottom: 18px;
    }
    .progress-top {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      margin-bottom: 8px;
      color: var(--sub);
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    }
    .progress-top span strong { color: var(--ink); font-weight: 600; }
    .progress-bar-bg {
      height: 4px;
      background: #eeeeee;
      border: 0.5px solid #cccccc;
      overflow: hidden;
    }
    .progress-bar-fill {
      height: 100%;
      background: #0a0a0a;
      width: 0%;
      transition: width 0.4s ease;
    }

    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      border-top: 1px solid var(--line);
      border-left: 1px solid var(--line);
      margin-bottom: 20px;
    }
    @media (max-width: 1100px) {
      .kpi-grid { grid-template-columns: repeat(3, 1fr); }
    }
    @media (max-width: 640px) {
      .kpi-grid { grid-template-columns: repeat(2, 1fr); }
    }
    .kpi {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 12px 14px;
      background: var(--bg);
    }
    .kpi-label {
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.4px;
      color: var(--sub);
      margin-bottom: 6px;
    }
    .kpi-val {
      font-size: 21px;
      font-weight: 600;
      color: var(--ink);
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      letter-spacing: -0.5px;
    }
    .kpi-sub {
      font-size: 11px;
      color: var(--muted);
      margin-top: 5px;
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    }

    .charts-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      border-top: 1px solid var(--line);
      border-left: 1px solid var(--line);
      margin-bottom: 20px;
    }
    @media (max-width: 960px) {
      .charts-grid { grid-template-columns: 1fr; }
    }
    .chart-card {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 14px 16px;
      background: var(--bg);
    }
    .chart-card h3 {
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 10px;
      color: var(--ink);
      display: flex;
      justify-content: space-between;
      border-bottom: 1px solid #eeeeee;
      padding-bottom: 6px;
    }
    .chart-box { position: relative; height: 245px; width: 100%; }

    .bottom-grid {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      border-top: 1px solid var(--line);
      border-left: 1px solid var(--line);
    }
    @media (max-width: 960px) {
      .bottom-grid { grid-template-columns: 1fr; }
    }
    .panel {
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      padding: 14px 16px;
      background: var(--bg);
      overflow: hidden;
    }
    .panel h3 {
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 10px;
      color: var(--ink);
      border-bottom: 1px solid #eeeeee;
      padding-bottom: 6px;
    }
    .table-wrap { max-height: 320px; overflow-y: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11.5px;
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
    }
    th, td {
      padding: 6px 8px;
      text-align: right;
      border-bottom: 1px solid #ececec;
      color: var(--ink);
    }
    th:first-child, td:first-child { text-align: left; }
    th {
      color: var(--sub);
      font-weight: 500;
      position: sticky;
      top: 0;
      background: var(--bg);
      border-bottom: 1px solid var(--line-strong);
    }
    .badge {
      display: inline-block;
      border: 1px solid #111111;
      color: #111111;
      padding: 0px 4px;
      font-size: 9.5px;
      margin-left: 4px;
    }
    .badge-muted {
      display: inline-block;
      border: 1px solid #999999;
      color: #666666;
      padding: 0px 4px;
      font-size: 9.5px;
      margin-left: 4px;
    }
    pre.log-box {
      background: var(--surface);
      border: 1px solid var(--line);
      padding: 10px 12px;
      font-family: "SFMono-Regular", Menlo, Consolas, monospace;
      font-size: 11px;
      line-height: 1.5;
      color: #222222;
      max-height: 320px;
      overflow-y: auto;
      white-space: pre-wrap;
    }
  </style>
</head>
<body>
  <header>
    <div class="title-group">
      <h1><span class="dot"></span>TURBOGO-MINI (b6c192-nbt12 · 3.06M) · 100M PRETRAINING MONITOR</h1>
      <div class="subtitle">12× Global Attention + 12× SwiGLU · 2D-RoPE (2.0× LR) · 100M Dataset (katago_19x19_chinese_100m.zip) · One-Cycle Cosine LR 1.6e-3 → 1.5e-5</div>
    </div>
    <div class="meta-right">
      <div>TIME (CST): <strong id="server-time">--</strong></div>
      <div>ETA (100M END): <strong id="rem-time">--</strong> · POLL 5S</div>
    </div>
  </header>

  <div class="progress-wrap">
    <div class="progress-top">
      <span>PROGRESS: <strong id="prog-text">Loading...</strong></span>
      <span>SAMPLES SEEN: <strong id="samples-text">--</strong></span>
    </div>
    <div class="progress-bar-bg">
      <div class="progress-bar-fill" id="prog-bar"></div>
    </div>
  </div>

  <div class="kpi-grid">
    <div class="kpi">
      <div class="kpi-label">Val Move Top-1 / Top-3</div>
      <div class="kpi-val" id="kpi-val-top1">--</div>
      <div class="kpi-sub" id="kpi-val-top3">Top-3: --</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Train Move Top-1</div>
      <div class="kpi-val" id="kpi-tr-top1">--</div>
      <div class="kpi-sub" id="kpi-tr-step">Step --</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Val Policy / Total Loss</div>
      <div class="kpi-val" id="kpi-val-ploss">--</div>
      <div class="kpi-sub" id="kpi-val-tloss">Total: --</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Val WLD Acc & Score MAE</div>
      <div class="kpi-val" id="kpi-wld-acc">--</div>
      <div class="kpi-sub" id="kpi-score-mae">MAE: --</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Throughput & LR</div>
      <div class="kpi-val" id="kpi-speed">--</div>
      <div class="kpi-sub" id="kpi-lr">LR: --</div>
    </div>
    <div class="kpi">
      <div class="kpi-label" id="kpi-gpu-title">GPU Telemetry</div>
      <div class="kpi-val" id="kpi-gpu">--</div>
      <div class="kpi-sub" id="kpi-gpu-sub">--</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="chart-card">
      <h3><span>Validation Move Top-1 & Top-3 (%)</span><span style="color:var(--muted);font-weight:400">Higher is better</span></h3>
      <div class="chart-box"><canvas id="accChart"></canvas></div>
    </div>
    <div class="chart-card">
      <h3><span>Validation Loss Curves (Total / Policy / Own MSE)</span><span style="color:var(--muted);font-weight:400">Lower is better</span></h3>
      <div class="chart-box"><canvas id="lossChart"></canvas></div>
    </div>
    <div class="chart-card">
      <h3><span>Live Training Top-1 (%) & Policy Loss</span><span style="color:var(--muted);font-weight:400">120-Step Interval</span></h3>
      <div class="chart-box"><canvas id="trainChart"></canvas></div>
    </div>
    <div class="chart-card">
      <h3><span>Validation WLD Acc (%) & Score MAE (pts)</span><span style="color:var(--muted);font-weight:400">Value Head</span></h3>
      <div class="chart-box"><canvas id="valHeadChart"></canvas></div>
    </div>
  </div>

  <div class="bottom-grid">
    <div class="panel">
      <h3>Validation Evaluation Log (metrics.jsonl · 40,000 Fixed Benchmark Positions)</h3>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Step (Epoch)</th>
              <th>Top-1</th>
              <th>Top-3</th>
              <th>Pol Loss</th>
              <th>Total Loss</th>
              <th>WLD Acc</th>
              <th>MAE</th>
              <th>Own MSE</th>
            </tr>
          </thead>
          <tbody id="val-tbody"></tbody>
        </table>
      </div>
    </div>
    <div class="panel">
      <h3>Terminal Stream (train_turbogo_mini.log)</h3>
      <pre class="log-box" id="log-box">Connecting to stream...</pre>
    </div>
  </div>

  <script>
    Chart.defaults.font.family = '"SFMono-Regular", Menlo, Consolas, monospace';
    Chart.defaults.font.size = 10.5;
    Chart.defaults.color = '#555555';

    const baseGrid = { color: '#eaeaea', lineWidth: 0.75 };
    const chartOpts = {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index', intersect: false },
      elements: { point: { radius: 0, hoverRadius: 3 } },
      plugins: {
        legend: {
          labels: { color: '#111111', boxWidth: 14, boxHeight: 1, font: { size: 10.5 } }
        }
      },
      scales: {
        x: { ticks: { color: '#666666', maxTicksLimit: 8 }, grid: baseGrid, border: { color: '#111111', width: 1 } },
        y: { ticks: { color: '#666666' }, grid: baseGrid, border: { color: '#111111', width: 1 } }
      }
    };

    const accChart = new Chart(document.getElementById('accChart'), {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Val Top-1 (%) [Solid]', data: [], borderColor: '#000000', tension: 0.15, borderWidth: 1.25 },
        { label: 'Val Top-3 (%) [Dashed]', data: [], borderColor: '#666666', borderDash: [4, 3], tension: 0.15, borderWidth: 1.0 }
      ]},
      options: chartOpts
    });

    const lossChart = new Chart(document.getElementById('lossChart'), {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Policy Loss [Solid]', data: [], borderColor: '#000000', tension: 0.15, borderWidth: 1.25 },
        { label: 'Total Loss [Dashed]', data: [], borderColor: '#555555', borderDash: [5, 3], tension: 0.15, borderWidth: 1.0 },
        { label: 'Ownership MSE [Dotted]', data: [], borderColor: '#888888', borderDash: [2, 2], tension: 0.15, borderWidth: 1.0 }
      ]},
      options: chartOpts
    });

    const trainChart = new Chart(document.getElementById('trainChart'), {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'Train Top-1 (%) [Solid]', data: [], borderColor: '#000000', tension: 0.1, borderWidth: 1.1, yAxisID: 'y' },
        { label: 'Train Pol Loss [Dashed]', data: [], borderColor: '#777777', borderDash: [3, 3], tension: 0.1, borderWidth: 1.0, yAxisID: 'y1' }
      ]},
      options: {
        ...chartOpts,
        scales: {
          x: { ticks: { color: '#666666', maxTicksLimit: 8 }, grid: baseGrid, border: { color: '#111111', width: 1 } },
          y: { type: 'linear', position: 'left', ticks: { color: '#000000' }, grid: baseGrid, border: { color: '#111111', width: 1 } },
          y1: { type: 'linear', position: 'right', ticks: { color: '#666666' }, grid: { drawOnChartArea: false }, border: { color: '#111111', width: 1 } }
        }
      }
    });

    const valHeadChart = new Chart(document.getElementById('valHeadChart'), {
      type: 'line',
      data: { labels: [], datasets: [
        { label: 'WLD Acc (%) [Solid]', data: [], borderColor: '#000000', tension: 0.15, borderWidth: 1.25, yAxisID: 'y' },
        { label: 'Score MAE (pts) [Dashed]', data: [], borderColor: '#666666', borderDash: [4, 3], tension: 0.15, borderWidth: 1.0, yAxisID: 'y1' }
      ]},
      options: {
        ...chartOpts,
        scales: {
          x: { ticks: { color: '#666666', maxTicksLimit: 8 }, grid: baseGrid, border: { color: '#111111', width: 1 } },
          y: { type: 'linear', position: 'left', ticks: { color: '#000000' }, grid: baseGrid, border: { color: '#111111', width: 1 } },
          y1: { type: 'linear', position: 'right', ticks: { color: '#666666' }, grid: { drawOnChartArea: false }, border: { color: '#111111', width: 1 } }
        }
      }
    });

    function fmtSec(s) {
      const h = Math.floor(s / 3600);
      const m = Math.floor((s % 3600) / 60);
      return `${h}h ${m}m`;
    }

    async function refresh() {
      try {
        const res = await fetch('/api/status?t=' + Date.now());
        const d = await res.json();

        document.getElementById('server-time').textContent = d.server_time_cst;
        document.getElementById('rem-time').textContent = fmtSec(d.remaining_sec);

        if (d.latest_train) {
          const lt = d.latest_train;
          const pct = (lt.step / lt.total_steps * 100).toFixed(2);
          document.getElementById('prog-text').textContent =
            `Epoch ${lt.epoch}/${lt.total_epochs} (100M) · Chunk ${lt.chunk}/${lt.total_chunks} · Step ${lt.step.toLocaleString()} / ${lt.total_steps.toLocaleString()} (${pct}%)`;
          document.getElementById('samples-text').textContent =
            `${(lt.step * 0.001024).toFixed(2)}M / 99.80M (100M Unique Pool)`;
          document.getElementById('prog-bar').style.width = Math.min(100, pct) + '%';

          document.getElementById('kpi-tr-top1').textContent = lt.top1.toFixed(2) + '%';
          document.getElementById('kpi-tr-step').textContent = `Step ${lt.step.toLocaleString()} · Pol ${lt.pol_loss.toFixed(3)}`;
          document.getElementById('kpi-speed').textContent = lt.speed.toLocaleString() + ' pos/s';
          document.getElementById('kpi-lr').textContent = `LR: ${lt.lr.toExponential(2)} · VRAM: ${lt.vram_gb}G`;
        }

        if (d.gpu) {
          if (d.gpu.name) {
            document.getElementById('kpi-gpu-title').textContent = d.gpu.name;
          }
          document.getElementById('kpi-gpu').textContent = `${d.gpu.util}% · ${d.gpu.power_w}W`;
          document.getElementById('kpi-gpu-sub').textContent =
            `VRAM ${(d.gpu.mem_used_mb/1024).toFixed(1)}/${(d.gpu.mem_total_mb/1024).toFixed(0)}G · ${d.gpu.temp_c}°C`;
        }

        if (d.val_series && d.val_series.length > 0) {
          const lv = d.val_series[d.val_series.length - 1];
          const bestTop1 = Math.max(...d.val_series.map(x => x.val_top1_acc));
          document.getElementById('kpi-val-top1').textContent = `${lv.val_top1_acc.toFixed(2)}%`;
          document.getElementById('kpi-val-top3').textContent = `Top-3: ${lv.val_top3_acc.toFixed(2)}% · Peak: ${bestTop1.toFixed(2)}%`;
          document.getElementById('kpi-val-ploss').textContent = lv.val_p0_loss.toFixed(4);
          const rgText = lv.res_gate !== undefined ? ` · RoPE: ${lv.res_gate.toFixed(3)}` : '';
          document.getElementById('kpi-val-tloss').textContent = `Total: ${lv.val_loss.toFixed(4)}${rgText}`;
          document.getElementById('kpi-wld-acc').textContent = `${lv.val_wld_acc.toFixed(2)}%`;
          document.getElementById('kpi-score-mae').textContent = `MAE: ${lv.val_score_mae_pts.toFixed(2)} pts`;

          const vLabels = d.val_series.map(x => 'S' + x.global_step);
          accChart.data.labels = vLabels;
          accChart.data.datasets[0].data = d.val_series.map(x => x.val_top1_acc);
          accChart.data.datasets[1].data = d.val_series.map(x => x.val_top3_acc);
          accChart.update('none');

          const vsClean = d.val_series.length > 2 ? d.val_series.slice(1) : d.val_series;
          const cLabels = vsClean.map(x => 'S' + x.global_step);
          lossChart.data.labels = cLabels;
          lossChart.data.datasets[0].data = vsClean.map(x => x.val_p0_loss);
          lossChart.data.datasets[1].data = vsClean.map(x => x.val_loss);
          lossChart.data.datasets[2].data = vsClean.map(x => x.val_own_mse);
          lossChart.update('none');

          valHeadChart.data.labels = cLabels;
          valHeadChart.data.datasets[0].data = vsClean.map(x => x.val_wld_acc);
          valHeadChart.data.datasets[1].data = vsClean.map(x => x.val_score_mae_pts);
          valHeadChart.update('none');

          const tbody = document.getElementById('val-tbody');
          tbody.innerHTML = d.val_series.slice().reverse().map(r => {
            const badge = '<span class="badge">MINI·3.06M</span>';
            return `
            <tr>
              <td><strong>${r.global_step.toLocaleString()}</strong> (Ep${r.epoch})${badge}</td>
              <td><strong>${r.val_top1_acc.toFixed(2)}%</strong></td>
              <td>${r.val_top3_acc.toFixed(2)}%</td>
              <td>${r.val_p0_loss.toFixed(4)}</td>
              <td>${r.val_loss.toFixed(4)}</td>
              <td>${r.val_wld_acc.toFixed(2)}%</td>
              <td>${r.val_score_mae_pts.toFixed(2)}</td>
              <td>${r.val_own_mse.toFixed(4)}</td>
            </tr>
          `}).join('');
        }

        if (d.train_series && d.train_series.length > 0) {
          trainChart.data.labels = d.train_series.map(x => x.step);
          trainChart.data.datasets[0].data = d.train_series.map(x => x.top1);
          trainChart.data.datasets[1].data = d.train_series.map(x => x.pol_loss);
          trainChart.update('none');
        }

        if (d.log_tail) {
          const lb = document.getElementById('log-box');
          lb.textContent = d.log_tail.join('\n');
          lb.scrollTop = lb.scrollHeight;
        }
      } catch (e) {
        console.error(e);
      }
    }

    refresh();
    setInterval(refresh, 5000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path.startswith("/api/status"):
            payload = json.dumps(collect_status(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
        elif self.path.startswith("/api/sync/"):
            fname = self.path.split("/api/sync/", 1)[1].split("?")[0]
            allowed = {"metrics.jsonl", "turbogo_mini_best_weights.pt", "turbogo_mini_final.pt"}
            fpath = CKPT_DIR / fname
            if fname in allowed and fpath.exists():
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
        else:
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dashboard server listening on http://127.0.0.1:{port}", flush=True)
    srv.serve_forever()
