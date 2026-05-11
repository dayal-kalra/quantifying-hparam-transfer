#!/usr/bin/env python3
"""Generate interactive/index.html from analysis CSVs.

Run from the transfer-framework/ directory:
    python interactive/build.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.transfer_framework_utils import get_short_filename_raw, PARAM_DISPLAY_NAMES

FILT_THRESH = 1.35
SCALER = 0.002

MARKER_MAP = {
    'sp':            {'symbol': 'circle',        'unicode': '●'},
    'mup_sp':        {'symbol': 'square',        'unicode': '■'},
    'mup_sp_embd':   {'symbol': 'pentagon',      'unicode': '⬠'},
    'sp_embd':       {'symbol': 'x',             'unicode': '✕'},
    'sp_attn_embd':  {'symbol': 'cross',         'unicode': '+'},
    'sp_ln':         {'symbol': 'diamond',       'unicode': '◆'},
    'mup_sp_ln':     {'symbol': 'triangle-up',   'unicode': '▲'},
    'sp_attn':       {'symbol': 'triangle-down', 'unicode': '▼'},
    'sp_last':       {'symbol': 'star',          'unicode': '★'},
    'mup_sp_attn':   {'symbol': 'hexagram',      'unicode': '✡'},
    'sp_lr':         {'symbol': 'circle-open',   'unicode': '○'},
    'sp_attn_last':  {'symbol': 'square-open',   'unicode': '□'},
    'sp_attn_ln':    {'symbol': 'diamond-open',  'unicode': '◇'},
    'sp_embd_ln':    {'symbol': 'triangle-up-open',   'unicode': '△'},
    'sp_embd_last':  {'symbol': 'triangle-down-open', 'unicode': '▽'},
    'sp_ln_last':    {'symbol': 'star-open',     'unicode': '☆'},
}
DEFAULT_MARKER = {'symbol': 'circle-open', 'unicode': '○'}

WD_PALETTE = {
    0.0:    '#1a1a1a',
    0.0001: '#c6dbef',
    0.0003: '#9ecae1',
    0.0006: '#6baed6',
    0.001:  '#4292c6',
    0.003:  '#2171b5',
    0.006:  '#08519c',
    0.01:   '#08306b',
}


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


_BASE_CFG = dict(
    experiment='step',
    wd_mode='fix',
    dataset_name='fineweb',
    model_name='gpt',
    num_layers=12,
    embd_dim=768,
    optim_name='AdamW',
    warmup_steps=2000,
    stable_steps=6000,
    decay_schedule_name='polynomial',
    num_steps=10000,
    gradient_accumulation_steps=16,
    lr_peak=1e-4,
    batch_size=16,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
    grad_clip=0.0,
)


def _load_row(abc_key, wd):
    cfg = _Cfg(**_BASE_CFG, abc=abc_key, weight_decay=wd)
    short = get_short_filename_raw(cfg)
    base = os.path.join('analysis', f'gpt_{abc_key}')
    paths = {
        'raw':    os.path.join(base, f'scaling_laws_raw_{short}.csv'),
        'interp': os.path.join(base, f'scaling_laws_interp_{short}.csv'),
        'curv':   os.path.join(base, f'curvature_interp_{short}.csv'),
        'joint':  os.path.join(base, f'joint_fit_interp_{short}.csv'),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'[{name}] {path}')

    df_raw = pd.read_csv(paths['raw']).iloc[0]
    alpha_raw = float(df_raw['alpha_abc'])
    irr_loss_gap = float(df_raw['irr_loss_gap'])

    df_interp = pd.read_csv(paths['interp'])[['filt_thresh', 'beta_abc']]
    df_curv = pd.read_csv(paths['curv'])[['filt_thresh', 'gamma']]
    df_joint = pd.read_csv(paths['joint'])[['filt_thresh', 'E']]

    df = df_interp.merge(df_curv, on='filt_thresh').merge(df_joint, on='filt_thresh')
    df['alpha_raw'] = alpha_raw
    df['irr_loss_gap'] = irr_loss_gap
    return df


def extract_data():
    wd_lst = np.loadtxt('wd.list', dtype=float)
    abc_lst = [line.strip() for line in open('abc.list') if line.strip()]

    rows = []
    for abc_key in abc_lst:
        for wd in wd_lst:
            try:
                df = _load_row(abc_key, wd)
            except FileNotFoundError as e:
                print(f'  [SKIP] {abc_key}, wd={wd:.4f}: {e}')
                continue
            df['abc_key'] = abc_key
            df['abc'] = PARAM_DISPLAY_NAMES.get(abc_key, abc_key).replace(r'$\mu$', 'μ')
            df['weight_decay'] = wd
            rows.append(df)

    df_all = pd.concat(rows, ignore_index=True)
    df_all = df_all[df_all['filt_thresh'].round(4) == round(FILT_THRESH, 4)].copy()

    neg_mask = df_all['irr_loss_gap'] < 0
    df_all.loc[neg_mask, 'irr_loss_gap'] *= SCALER
    df_all['kappa'] = df_all['alpha_raw'] - 2 * df_all['beta_abc'] + df_all['gamma']

    out = df_all[['abc_key', 'abc', 'weight_decay', 'E', 'irr_loss_gap', 'kappa']].copy()
    out = out.round(6)
    print(f'Extracted {len(out)} rows — {out["abc"].nunique()} parameterizations, {out["weight_decay"].nunique()} wd values')
    return out, sorted(wd_lst.tolist())


def _html(records, wd_values, abc_order_keys):
    data_json = json.dumps(records)
    wd_json = json.dumps(wd_values)

    marker_info = []
    for key in abc_order_keys:
        m = MARKER_MAP.get(key, DEFAULT_MARKER)
        display = PARAM_DISPLAY_NAMES.get(key, key).replace(r'$\mu$', 'μ')
        marker_info.append({'key': key, 'display': display, 'symbol': m['symbol'], 'unicode': m['unicode']})
    marker_js = json.dumps(marker_info)

    wd_palette_js = json.dumps({str(k): v for k, v in WD_PALETTE.items()})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Transfer Framework — Phase Diagrams</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 0; padding: 14px; background: #fff; color: #222; }}
  h2 {{ margin: 0 0 12px; font-size: 15px; font-weight: 600; color: #333; }}
  #wrap {{ display: flex; gap: 12px; align-items: flex-start; }}
  #sidebar {{ width: 185px; flex-shrink: 0; border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px 12px; background: #fafafa; }}
  #sidebar h3 {{ margin: 0 0 6px; font-size: 12px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 0.04em; }}
  .btn-row {{ display: flex; gap: 6px; margin-bottom: 8px; }}
  .btn-row button {{ font-size: 11px; padding: 2px 10px; cursor: pointer; border-radius: 4px; border: 1px solid #ccc; background: #fff; color: #333; }}
  .btn-row button:hover {{ background: #f0f0f0; }}
  .abc-row {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; padding: 2px 0; }}
  .abc-row input {{ cursor: pointer; flex-shrink: 0; }}
  .abc-sym {{ font-size: 13px; width: 16px; text-align: center; flex-shrink: 0; }}
  .abc-row label {{ font-size: 12px; cursor: pointer; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  #wd-section {{ margin-top: 14px; border-top: 1px solid #e8e8e8; padding-top: 10px; }}
  .wd-row {{ display: flex; align-items: center; gap: 6px; margin: 3px 0; }}
  .wd-row input {{ cursor: pointer; flex-shrink: 0; }}
  .wd-row label {{ font-size: 11px; cursor: pointer; font-family: monospace; color: #444; }}
  .wd-dot {{ width: 12px; height: 12px; border-radius: 50%; border: 1px solid rgba(0,0,0,0.3); flex-shrink: 0; display: inline-block; }}
  #plot {{ flex: 1; min-width: 0; }}
</style>
</head>
<body>
<h2>Transfer Framework — Phase Diagrams &nbsp;&middot;&nbsp; experiment=step &nbsp;&middot;&nbsp; filt_thresh={FILT_THRESH}</h2>
<div id="wrap">
  <div id="sidebar">
    <h3>Parameterization</h3>
    <div class="btn-row">
      <button onclick="selectAll()">All</button>
      <button onclick="selectNone()">None</button>
    </div>
    <div id="abc-controls"></div>
    <div id="wd-section">
      <h3>Weight Decay &lambda;</h3>
      <div class="btn-row">
        <button onclick="selectAllWd()">All</button>
        <button onclick="selectNoneWd()">None</button>
      </div>
      <div id="wd-rows"></div>
    </div>
  </div>
  <div id="plot"></div>
</div>

<script>
const DATA         = {data_json};
const WD_VALUES    = {wd_json};
const MARKER_INFO  = {marker_js};
const WD_PALETTE   = {wd_palette_js};

function r4(w) {{ return parseFloat(w.toFixed(4)); }}

function wdColor(wd) {{
  const key   = r4(wd);
  const pkeys = Object.keys(WD_PALETTE).map(Number);
  const best  = pkeys.reduce((a, b) => Math.abs(b - key) < Math.abs(a - key) ? b : a);
  return Math.abs(best - key) < 1e-7 ? WD_PALETTE[best] : '#999';
}}

// ── Filter state ──────────────────────────────────────────────────────────────
const activeAbcs = new Set(MARKER_INFO.map(m => m.key));
const activeWds  = new Set(WD_VALUES.map(w => r4(w)));

// ── Build traces from current filter state ────────────────────────────────────
function buildTraces() {{
  const filtered = DATA.filter(d =>
    activeAbcs.has(d.abc_key) && activeWds.has(r4(d.weight_decay))
  );
  const traces = [];
  MARKER_INFO.forEach(info => {{
    const pts = filtered.filter(d => d.abc_key === info.key);
    if (pts.length === 0) return;
    const col = pts.map(d => wdColor(d.weight_decay));
    const txt = pts.map(d =>
      '<b>' + info.display + '</b><br>' +
      'λ = ' + d.weight_decay + '<br>' +
      'E = ' + d.E.toExponential(3) + '<br>' +
      'R(∞) = ' + d.irr_loss_gap.toFixed(5) + '<br>' +
      'κ = ' + d.kappa.toFixed(5)
    );
    const mk = c => ({{ symbol: info.symbol, size: 10, color: c, line: {{ color: '#333', width: 0.8 }} }});
    const base = {{
      name: info.display, legendgroup: info.key,
      mode: 'markers', type: 'scatter',
      hovertemplate: '%{{text}}<extra></extra>',
      showlegend: false, text: txt,
    }};
    traces.push({{ ...base, x: pts.map(d => d.E),            y: pts.map(d => d.kappa),       marker: mk(col), xaxis: 'x',  yaxis: 'y'  }});
    traces.push({{ ...base, x: pts.map(d => d.E),            y: pts.map(d => d.irr_loss_gap), marker: mk(col), xaxis: 'x2', yaxis: 'y2' }});
    traces.push({{ ...base, x: pts.map(d => d.irr_loss_gap), y: pts.map(d => d.kappa),        marker: mk(col), xaxis: 'x3', yaxis: 'y3' }});
  }});
  return traces;
}}

const layout = {{
  grid: {{ rows: 1, columns: 3, pattern: 'independent' }},
  xaxis:  {{ title: {{ text: 'E' }},    type: 'log', showgrid: true, zeroline: false }},
  yaxis:  {{ title: {{ text: 'κ' }},             showgrid: true, zeroline: true,  zerolinecolor: '#aaa' }},
  xaxis2: {{ title: {{ text: 'E' }},    type: 'log', showgrid: true, zeroline: false }},
  yaxis2: {{ title: {{ text: 'R(∞)' }},          showgrid: true, zeroline: true,  zerolinecolor: '#aaa' }},
  xaxis3: {{ title: {{ text: 'R(∞)' }},          showgrid: true, zeroline: true,  zerolinecolor: '#aaa' }},
  yaxis3: {{ title: {{ text: 'κ' }},             showgrid: true, zeroline: true,  zerolinecolor: '#aaa' }},
  showlegend: false,
  margin: {{ l: 65, r: 20, t: 30, b: 70 }},
  height: 480,
}};

let _ready = false;
function rebuildPlot() {{
  const fn = _ready ? Plotly.react : Plotly.newPlot;
  fn('plot', buildTraces(), layout, {{ responsive: true }});
  _ready = true;
}}

// ── Build sidebar: parameterizations ─────────────────────────────────────────
const ctrl = document.getElementById('abc-controls');
MARKER_INFO.forEach((info, i) => {{
  const row = document.createElement('div');
  row.className = 'abc-row';
  row.innerHTML =
    '<input type="checkbox" id="cba' + i + '" checked onchange="toggleAbc(' + i + ', this.checked)">' +
    '<span class="abc-sym">' + info.unicode + '</span>' +
    '<label for="cba' + i + '" title="' + info.display + '">' + info.display + '</label>';
  ctrl.appendChild(row);
}});

function toggleAbc(i, checked) {{
  const key = MARKER_INFO[i].key;
  checked ? activeAbcs.add(key) : activeAbcs.delete(key);
  rebuildPlot();
}}
function selectAll() {{
  MARKER_INFO.forEach((info, i) => {{
    activeAbcs.add(info.key);
    document.getElementById('cba' + i).checked = true;
  }});
  rebuildPlot();
}}
function selectNone() {{
  MARKER_INFO.forEach((info, i) => {{
    activeAbcs.delete(info.key);
    document.getElementById('cba' + i).checked = false;
  }});
  rebuildPlot();
}}

// ── Build sidebar: weight decay ───────────────────────────────────────────────
const wdContainer = document.getElementById('wd-rows');
WD_VALUES.forEach((wd, j) => {{
  const row = document.createElement('div');
  row.className = 'wd-row';
  row.innerHTML =
    '<input type="checkbox" id="cbw' + j + '" checked onchange="toggleWd(' + wd + ', this.checked)">' +
    '<span class="wd-dot" style="background:' + wdColor(wd) + '"></span>' +
    '<label for="cbw' + j + '">' + wd + '</label>';
  wdContainer.appendChild(row);
}});

function toggleWd(wd, checked) {{
  checked ? activeWds.add(r4(wd)) : activeWds.delete(r4(wd));
  rebuildPlot();
}}
function selectAllWd() {{
  WD_VALUES.forEach((wd, j) => {{
    activeWds.add(r4(wd));
    document.getElementById('cbw' + j).checked = true;
  }});
  rebuildPlot();
}}
function selectNoneWd() {{
  WD_VALUES.forEach((wd, j) => {{
    activeWds.delete(r4(wd));
    document.getElementById('cbw' + j).checked = false;
  }});
  rebuildPlot();
}}

// ── Initial render ────────────────────────────────────────────────────────────
rebuildPlot();
</script>
</body>
</html>
"""


def main():
    df, wd_values = extract_data()

    abc_lst = [line.strip() for line in open('abc.list') if line.strip()]
    present = set(df['abc_key'].unique())
    ordered_keys = (
        (['sp']     if 'sp'     in present else []) +
        (['mup_sp'] if 'mup_sp' in present else []) +
        [k for k in abc_lst if k not in ('sp', 'mup_sp') and k in present]
    )

    html = _html(df.to_dict('records'), wd_values, ordered_keys)

    os.makedirs('interactive', exist_ok=True)
    out = os.path.join('interactive', 'index.html')
    with open(out, 'w') as f:
        f.write(html)
    print(f'Wrote {out}')


if __name__ == '__main__':
    main()
