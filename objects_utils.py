import numpy as np
import colorsys
import json
import os
import webbrowser
from IPython.display import HTML


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _colors_from_true_labels(labels):
    """Map each unique integer label to a distinct RGB color string."""
    arr = np.asarray(labels).ravel()
    uniq = np.unique(arr)
    n = max(1, uniq.size)
    lut = {}
    for i, u in enumerate(uniq):
        h = (i / n) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.60, 0.95)
        lut[int(u)] = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'
    return [lut[int(v)] for v in arr]


def _axis_range(vals_list):
    """Compute a stable [lo, hi] range across a list of value arrays."""
    lo = min(float(np.nanmin(v)) for v in vals_list)
    hi = max(float(np.nanmax(v)) for v in vals_list)
    if not np.isfinite(hi - lo) or (hi - lo) < 1e-12:
        c = (lo + hi) / 2
        return [c - 1e-3, c + 1e-3]
    return [lo, hi]


def _build_payload(evec_seq, colors_seq, labels=None, eigenvalue_seq=None):
    """
    Build the JSON payload consumed by the HTML animation.

    Parameters
    ----------
    evec_seq : list of ndarray, each (n, d>=3)
    colors_seq : list of list-of-str, one color per point per frame
    labels : 1-D array-like, optional
    eigenvalue_seq : list of 1-D array-like, optional
        Eigenvalues at each frame (need not be the same length per frame).
    """
    frames = []
    xs, ys, zs = [], [], []
    # When true labels are provided the JS always uses true_colors, so
    # per-frame colors are never read — skip them to shrink the payload.
    store_frame_colors = labels is None

    for E, cols in zip(evec_seq, colors_seq):
        x = np.round(np.asarray(E[:, 0], float).ravel(), 5)
        y = np.round(np.asarray(E[:, 1], float).ravel(), 5)
        z = np.round(np.asarray(E[:, 2], float).ravel(), 5)
        frame = {'x': x.tolist(), 'y': y.tolist(), 'z': z.tolist()}
        if store_frame_colors:
            frame['colors'] = cols
        frames.append(frame)
        xs.append(x); ys.append(y); zs.append(z)

    # Global y-range for the sorted eigenvector panel (evec0 and evec1 combined)
    evec01_range = _axis_range(xs + ys)

    payload = {
        'frames': frames,
        'ranges': {
            'x': _axis_range(xs),
            'y': _axis_range(ys),
            'z': _axis_range(zs),
        },
        'evec01_range': evec01_range,
    }

    if labels is not None:
        arr = np.asarray(labels).ravel()
        payload['true_colors'] = _colors_from_true_labels(arr)
        payload['labels'] = arr.tolist()

    if eigenvalue_seq is not None:
        # Store only the 1st eigenvalue (index 0) per frame as a time series
        eig1 = [float(np.asarray(ev, float).ravel()[0]) for ev in eigenvalue_seq]
        payload['eig1_series'] = eig1
        lo, hi = min(eig1), max(eig1)
        pad = max(abs(hi - lo) * 0.05, 1e-3)
        payload['eig_range'] = [lo - pad, hi + pad]

    return payload


# ---------------------------------------------------------------------------
# HTML builder
# ---------------------------------------------------------------------------

_STYLE = """
body {
  margin: 0;
  padding: 0.75rem;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f7f7;
  box-sizing: border-box;
}
h1 { font-size: 1.05rem; margin: 0 0 0.4rem 0; }

/* ---- layout ---- */
#layout {
  display: grid;
  gap: 0.6rem;
  align-items: stretch;
  transition: grid-template-columns 0.25s ease;
}
.layout-4       { grid-template-columns: 1fr 1.4fr 1fr 1fr; }
.layout-3       { grid-template-columns: 1fr 1.4fr 1fr; }
.layout-3-right { grid-template-columns: 1.4fr 1fr 1fr; }
.layout-2       { grid-template-columns: 1.4fr 1fr; }

.panel {
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 8px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  overflow: hidden;
  min-width: 0;
}
.panel-plot {
  width: 100%;
  height: 62vh;
  max-height: 760px;
}

/* ---- control bar ---- */
.ctrl-bar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.4rem;
  margin-bottom: 0.5rem;
  padding: 0.3rem 0.6rem;
  background: #fff;
  border: 1px solid #ddd;
  border-radius: 999px;
  box-shadow: 0 1px 2px rgba(0,0,0,0.04);
  width: max-content;
  max-width: 100%;
}
.btn {
  border: 1px solid #d1d5db;
  border-radius: 999px;
  padding: 0.22rem 0.75rem;
  background: #f9fafb;
  font-size: 0.82rem;
  cursor: pointer;
  transition: background 0.12s, box-shadow 0.12s, transform 0.05s, border-color 0.12s;
  white-space: nowrap;
}
.btn:hover  { background: #f3f4f6; box-shadow: 0 1px 2px rgba(0,0,0,0.08); transform: translateY(-0.5px); }
.btn:active { transform: translateY(0); box-shadow: none; }
.btn-primary { background: #2563eb; color: #fff; border-color: #1d4ed8; }
.btn-primary:hover { background: #1d4ed8; }
.btn-active  { background: #065f46; color: #fff; border-color: #047857; }
.btn-sep { width: 1px; height: 1.2rem; background: #e5e7eb; margin: 0 0.1rem; }
.scrub {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  flex: 1 1 220px;
  min-width: 0;
}
.scrub input[type=range] { flex: 1; }
.scrub span { font-size: 0.72rem; color: #4b5563; white-space: nowrap; }
"""

_HTML_TMPL = """\
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"
          onerror="this.onerror=null;this.src='https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly.min.js'"></script>
  <style>{style}</style>
</head>
<body>
  <h1>{title}</h1>

  <!-- control bar -->
  <div class="ctrl-bar" id="ctrlBar">
    <button class="btn btn-primary" id="btnPlay">Play</button>
    <button class="btn" id="btnPause">Pause</button>
    <button class="btn" id="btnRestart">Restart</button>
    <div class="btn-sep"></div>
    <button class="btn" id="btnToggle2D">Hide 2D</button>
    <button class="btn" id="btnToggleEvec">Hide Sorted Evecs</button>
    <div class="btn-sep"></div>
    <span style="font-size:0.78rem;color:#4b5563;white-space:nowrap">2D view:</span>
    <button class="btn btn-active" id="btn2d01" data-view="01">ev0 vs ev1</button>
    <button class="btn" id="btn2d02" data-view="02">ev0 vs ev2</button>
    <button class="btn" id="btn2d12" data-view="12">ev1 vs ev2</button>
    <div class="btn-sep"></div>
    <div class="scrub">
      <input type="range" id="scrub" min="0" value="0" step="1">
      <span id="frameReadout">0 / 0</span>
    </div>
  </div>

  <!-- four-panel grid -->
  <div id="layout" class="layout-4">
    <div class="panel" id="panel2d">
      <div class="panel-plot" id="plot2d"></div>
    </div>
    <div class="panel">
      <div class="panel-plot" id="plot3d"></div>
    </div>
    <div class="panel">
      <div class="panel-plot" id="plotEig"></div>
    </div>
    <div class="panel" id="panelEvec">
      <div class="panel-plot" id="plotEvec"></div>
    </div>
  </div>

<script>
(function(){{
'use strict';

function waitForPlotly(cb, tries) {{
  tries = tries || 0;
  if (typeof Plotly !== 'undefined') {{ cb(); return; }}
  if (tries < 60) {{ setTimeout(function(){{ waitForPlotly(cb, tries+1); }}, 100); }}
  else {{ document.getElementById('plot3d').innerHTML = '<div style="padding:2rem;color:red">Plotly failed to load.</div>'; }}
}}

waitForPlotly(function() {{

const PD = {data_json};

const nFrames = PD.frames.length;
const rx = PD.ranges.x, ry = PD.ranges.y, rz = PD.ranges.z;
const f0 = PD.frames[0] || {{x:[], y:[], z:[], colors:[]}};
const trueColors = (PD.true_colors && PD.true_colors.length) ? PD.true_colors : null;
const hasEig = Array.isArray(PD.eig1_series) && PD.eig1_series.length > 0;

let idx = 0;
let playing = false;
let lastTs = 0;
const INTERVAL = 180;
let dragging = false;
let wasPlayingDrag = false;
let wasPlayingPointer = false;
let show2d = true;
let showEvec = true;
let view2d = '01';  // '01', '02', '12'
let plotsReady = false;
let userCamera = null;  // tracks user-rotated camera position

function updateGridClass() {{
  const el = document.getElementById('layout');
  if (show2d && showEvec)       el.className = 'layout-4';
  else if (show2d && !showEvec) el.className = 'layout-3';
  else if (!show2d && showEvec) el.className = 'layout-3-right';
  else                          el.className = 'layout-2';
}}

// ---- helpers ----
function currentColors() {{
  // color always on – use true label colors if available
  if (trueColors) return trueColors;
  const fr = PD.frames[idx];
  return (fr && fr.colors && fr.colors.length) ? fr.colors : 'rgb(160,160,160)';
}}

function get2dXY(fr) {{
  if (view2d === '01') return {{ x: fr.x, y: fr.y, xLabel: 'evec 0', yLabel: 'evec 1' }};
  if (view2d === '02') return {{ x: fr.x, y: fr.z, xLabel: 'evec 0', yLabel: 'evec 2' }};
  /* '12' */            return {{ x: fr.y, y: fr.z, xLabel: 'evec 1', yLabel: 'evec 2' }};
}}

function xyRange(view) {{
  if (view === '01') return {{ xr: rx, yr: ry }};
  if (view === '02') return {{ xr: rx, yr: rz }};
  /* '12' */          return {{ xr: ry, yr: rz }};
}}

// ---- initialize 3D plot ----
const gd3 = document.getElementById('plot3d');
var p3dReady = Plotly.newPlot(gd3, [{{
  type: 'scatter3d', mode: 'markers',
  x: f0.x, y: f0.y, z: f0.z,
  marker: {{ size: 3, opacity: 0.85, color: trueColors || f0.colors || 'rgb(160,160,160)' }}
}}], {{
  dragmode: 'orbit',
  margin: {{l:0,r:0,b:0,t:0}},
  scene: {{
    bgcolor: '#fff',
    xaxis: {{ title:'evec 0', gridcolor:'#e1e1e1', zerolinecolor:'#ccc', range: rx }},
    yaxis: {{ title:'evec 1', gridcolor:'#e1e1e1', zerolinecolor:'#ccc', range: ry }},
    zaxis: {{ title:'evec 2', gridcolor:'#e1e1e1', zerolinecolor:'#ccc', range: rz }},
    aspectmode: 'cube',
    uirevision: 'lock',
    camera: {{ eye:{{ x:1.6, y:1.6, z:0.9 }} }}
  }}
}}, {{ responsive:true, scrollZoom:true, displaylogo:false,
      modeBarButtonsToRemove:['select2d','lasso2d'] }});

// ---- initialize 2D plot ----
const gd2 = document.getElementById('plot2d');
var p2dReady = Plotly.newPlot(gd2, [{{
  type: 'scattergl', mode: 'markers',
  x: f0.x, y: f0.y,
  marker: {{ size: 4, opacity: 0.85, color: trueColors || f0.colors || 'rgb(160,160,160)' }}
}}], {{
  margin: {{l:40,r:10,b:40,t:10}},
  xaxis: {{ title:'evec 0', gridcolor:'#e1e1e1', range: rx }},
  yaxis: {{ title:'evec 1', gridcolor:'#e1e1e1', range: ry, scaleanchor:'x', scaleratio:1 }},
  plot_bgcolor:'#fff', paper_bgcolor:'#fff'
}}, {{ responsive:true, displaylogo:false,
      modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'] }});

// ---- initialize eigenvalue plot ----
const gdE = document.getElementById('plotEig');
var pERdy = Promise.resolve();
if (hasEig) {{
  const eigRange = PD.eig_range || [null, null];
  pERdy = Plotly.newPlot(gdE, [{{
    type: 'scatter',
    mode: 'lines+markers',
    x: [0],
    y: [PD.eig1_series[0]],
    line: {{ color: '#2563eb', width: 2 }},
    marker: {{ color: '#2563eb', size: 6 }}
  }}], {{
    margin: {{l:50,r:10,b:50,t:10}},
    xaxis: {{ title:'Iteration', gridcolor:'#e1e1e1', range:[0, nFrames-1] }},
    yaxis: {{ title:'Eigenvalue 1', gridcolor:'#e1e1e1', range: eigRange }},
    plot_bgcolor:'#fff', paper_bgcolor:'#fff'
  }}, {{ responsive:true, displaylogo:false }});
}} else {{
  gdE.innerHTML = '<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#9ca3af;font-size:0.9rem">No eigenvalue data</div>';
}}

// ---- initialize sorted eigenvector plot ----
const gdS = document.getElementById('plotEvec');
const nodeIdx = f0.x.map(function(_, i){{ return i; }});
const sorted0_init = f0.x.slice().sort(function(a,b){{ return a-b; }});
const sorted1_init = f0.y.slice().sort(function(a,b){{ return a-b; }});
var pSReady = Plotly.newPlot(gdS, [
  {{ type:'scatter', mode:'lines', name:'evec 1',
     x: nodeIdx, y: sorted0_init,
     line: {{ color:'#2563eb', width:1.5 }} }},
  {{ type:'scatter', mode:'lines', name:'evec 2',
     x: nodeIdx, y: sorted1_init,
     line: {{ color:'#dc2626', width:1.5 }} }}
], {{
  margin: {{l:45,r:10,b:40,t:10}},
  xaxis: {{ title:'Node index', gridcolor:'#e1e1e1' }},
  yaxis: {{ title:'Value', gridcolor:'#e1e1e1', range: PD.evec01_range }},
  legend: {{ x:0.02, y:0.98, bgcolor:'rgba(255,255,255,0.8)', font:{{size:11}} }},
  plot_bgcolor:'#fff', paper_bgcolor:'#fff'
}}, {{ responsive:true, displaylogo:false,
      modeBarButtonsToRemove:['select2d','lasso2d'] }});

// ---- update to frame k ----
function toFrame(k) {{
  if (!plotsReady) return;
  idx = Math.max(0, Math.min(k, nFrames - 1));
  const fr = PD.frames[idx];
  if (!fr) return;

  const cols = currentColors();

  // 3-D
  Plotly.restyle(gd3, {{ x:[fr.x], y:[fr.y], z:[fr.z], 'marker.color':[cols] }}, [0]);

  // 2-D
  if (show2d) {{
    const {{ x, y }} = get2dXY(fr);
    Plotly.restyle(gd2, {{ x:[x], y:[y], 'marker.color':[cols] }}, [0]);
  }}

  // eigenvalue — growing time series up to current frame
  if (hasEig) {{
    const xs = PD.eig1_series.slice(0, idx + 1).map(function(_,i){{ return i; }});
    const ys = PD.eig1_series.slice(0, idx + 1);
    Plotly.restyle(gdE, {{ x:[xs], y:[ys] }}, [0]);
  }}

  // sorted eigenvectors
  if (showEvec) {{
    const s0 = fr.x.slice().sort(function(a,b){{ return a-b; }});
    const s1 = fr.y.slice().sort(function(a,b){{ return a-b; }});
    Plotly.restyle(gdS, {{ y:[s0, s1] }}, [0, 1]);
  }}

  if (scrub) scrub.value = String(idx);
  if (readout) readout.textContent = idx + ' / ' + (nFrames - 1);
}}

// ---- animation loop ----
function raf(ts) {{
  if (!playing) return;
  if (ts - lastTs >= INTERVAL) {{
    lastTs = ts;
    if (idx < nFrames - 1) {{ toFrame(idx + 1); }}
    else {{ playing = false; return; }}
  }}
  requestAnimationFrame(raf);
}}

// ---- wait for all plots ----
Promise.all([p3dReady, p2dReady, pERdy, pSReady]).then(function() {{
  // Keep gd3.layout.scene.camera in sync with user rotation so restyle
  // doesn't snap back to the original camera on each frame update.
  gd3.on('plotly_relayout', function(ed) {{
    if (ed['scene.camera']) {{
      userCamera = ed['scene.camera'];
      if (gd3.layout && gd3.layout.scene)
        gd3.layout.scene.camera = userCamera;
    }}
  }});
  plotsReady = true;
  toFrame(0);
  playing = true;
  requestAnimationFrame(raf);
}});

// ---- pointer-pause on 3D/2D plots ----
function hookPointerPause(el) {{
  if (!el) return;
  el.addEventListener('pointerdown', function() {{
    if (playing) {{ wasPlayingPointer = true; playing = false; }}
  }});
  function resume() {{
    if (wasPlayingPointer) {{ wasPlayingPointer = false; playing = true; requestAnimationFrame(raf); }}
  }}
  el.addEventListener('pointerup', resume);
  el.addEventListener('pointerleave', resume);
}}
hookPointerPause(gd3);
hookPointerPause(gd2);

// ---- controls ----
const scrub   = document.getElementById('scrub');
const readout = document.getElementById('frameReadout');
if (scrub) scrub.max = String(Math.max(0, nFrames - 1));

document.getElementById('btnPlay').onclick = function() {{
  if (!playing) {{ playing = true; requestAnimationFrame(raf); }}
}};
document.getElementById('btnPause').onclick = function() {{ playing = false; }};
document.getElementById('btnRestart').onclick = function() {{ playing = false; toFrame(0); }};

// 2D toggle
document.getElementById('btnToggle2D').onclick = function() {{
  show2d = !show2d;
  const panel = document.getElementById('panel2d');
  const btn = document.getElementById('btnToggle2D');
  const viewBtns = document.querySelectorAll('[data-view]');
  if (show2d) {{
    panel.style.display = '';
    btn.textContent = 'Hide 2D';
    viewBtns.forEach(function(b) {{ b.style.display = ''; }});
    Plotly.Plots.resize(gd2);
  }} else {{
    panel.style.display = 'none';
    btn.textContent = 'Show 2D';
    viewBtns.forEach(function(b) {{ b.style.display = 'none'; }});
  }}
  updateGridClass();
  Plotly.Plots.resize(gd3);
  Plotly.Plots.resize(gdE);
  if (showEvec) Plotly.Plots.resize(gdS);
}};

// Sorted evec toggle
document.getElementById('btnToggleEvec').onclick = function() {{
  showEvec = !showEvec;
  const panel = document.getElementById('panelEvec');
  const btn = document.getElementById('btnToggleEvec');
  if (showEvec) {{
    panel.style.display = '';
    btn.textContent = 'Hide Sorted Evecs';
    Plotly.Plots.resize(gdS);
  }} else {{
    panel.style.display = 'none';
    btn.textContent = 'Show Sorted Evecs';
  }}
  updateGridClass();
  Plotly.Plots.resize(gd3);
  Plotly.Plots.resize(gdE);
  if (show2d) Plotly.Plots.resize(gd2);
}};

// 2D view buttons
['btn2d01','btn2d02','btn2d12'].forEach(function(id) {{
  var btn = document.getElementById(id);
  if (!btn) return;
  btn.onclick = function() {{
    view2d = btn.getAttribute('data-view');
    ['btn2d01','btn2d02','btn2d12'].forEach(function(b) {{
      document.getElementById(b).classList.remove('btn-active');
    }});
    btn.classList.add('btn-active');
    // update 2D axis labels and ranges
    var rng = xyRange(view2d);
    var fr = PD.frames[idx];
    var {{ x, y, xLabel, yLabel }} = get2dXY(fr);
    var cols = currentColors();
    Plotly.restyle(gd2, {{ x:[x], y:[y], 'marker.color':[cols] }}, [0]);
    Plotly.relayout(gd2, {{
      'xaxis.title': xLabel, 'xaxis.range': rng.xr,
      'yaxis.title': yLabel, 'yaxis.range': rng.yr
    }});
  }};
}});

// scrub
if (scrub) {{
  scrub.addEventListener('pointerdown', function() {{
    wasPlayingDrag = playing; playing = false;
  }});
  scrub.addEventListener('input', function() {{ toFrame(+scrub.value); }});
  scrub.addEventListener('pointerup', function() {{
    if (wasPlayingDrag) {{ wasPlayingDrag = false; playing = true; requestAnimationFrame(raf); }}
  }});
}}

// keyboard
window.addEventListener('keydown', function(e) {{
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
  if (e.code === 'Space') {{
    e.preventDefault();
    if (playing) {{ playing = false; }} else {{ playing = true; requestAnimationFrame(raf); }}
  }}
  if (e.code === 'ArrowRight') toFrame(idx + 1);
  if (e.code === 'ArrowLeft')  toFrame(idx - 1);
}});

// resize
window.addEventListener('resize', function() {{
  Plotly.Plots.resize(gd3);
  if (show2d) Plotly.Plots.resize(gd2);
  Plotly.Plots.resize(gdE);
  if (showEvec) Plotly.Plots.resize(gdS);
}});

}}); // waitForPlotly
}})();
</script>
</body>
</html>
"""


def _build_html(payload: dict, title: str) -> str:
    return _HTML_TMPL.format(
        title=title,
        style=_STYLE,
        data_json=json.dumps(payload),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def animate_evecs(evec_seq, eigenvalue_seq=None, labels=None, colors=None,
                  title="Eigenvector Animation"):
    """
    Three-panel interactive animation:

      Left   – 2D scatter (toggleable; cycle between ev0-1, ev0-2, ev1-2)
      Center – 3D scatter (always visible; always colored by true labels)
      Right  – Eigenvalue bar chart (updates each step)

    Parameters
    ----------
    evec_seq : sequence of array-like, each (n, d>=3)
        Eigenvector matrices, one per animation frame.
    eigenvalue_seq : sequence of array-like, optional
        Eigenvalue arrays, one per frame (can vary in length).
    labels : array-like of int, optional
        True node labels used for permanent coloring.
    colors : sequence of list-of-str, optional
        Per-frame color arrays.  Ignored if *labels* is provided.
    title : str
        Title displayed above the animation.

    Returns
    -------
    IPython.display.HTML
    """
    seq = [np.asarray(E, float) for E in evec_seq]
    if not seq:
        raise ValueError("evec_seq must contain at least one frame.")
    n, d = seq[0].shape
    if d < 3:
        raise ValueError(f"Each eigenvector matrix must have >= 3 columns, got {d}.")
    for E in seq:
        if E.shape[0] != n:
            raise ValueError("All frames must have the same number of rows (points).")

    T = len(seq)

    if colors is not None:
        if len(colors) != T:
            raise ValueError("colors must have the same length as evec_seq.")
        colors_seq = list(colors)
    elif labels is not None:
        c = _colors_from_true_labels(labels)
        colors_seq = [c] * T
    else:
        gray = ['rgb(160,160,160)'] * n
        colors_seq = [gray] * T

    if eigenvalue_seq is not None and len(eigenvalue_seq) != T:
        raise ValueError("eigenvalue_seq must have the same length as evec_seq.")

    payload = _build_payload(seq, colors_seq, labels=labels,
                             eigenvalue_seq=eigenvalue_seq)
    return HTML(_build_html(payload, title=title))


# HTML shell that loads data via a <script> tag (works from file:// unlike fetch).
_HTML_FETCH_TMPL = """\
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"
          onerror="this.onerror=null;this.src='https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly.min.js'"></script>
  <script src="{js_file}"></script>
  <style>{style}</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="ctrl-bar" id="ctrlBar">
    <button class="btn btn-primary" id="btnPlay">Play</button>
    <button class="btn" id="btnPause">Pause</button>
    <button class="btn" id="btnRestart">Restart</button>
    <div class="btn-sep"></div>
    <button class="btn" id="btnToggle2D">Hide 2D</button>
    <button class="btn" id="btnToggleEvec">Hide Sorted Evecs</button>
    <div class="btn-sep"></div>
    <span style="font-size:0.78rem;color:#4b5563;white-space:nowrap">2D view:</span>
    <button class="btn btn-active" id="btn2d01" data-view="01">ev0 vs ev1</button>
    <button class="btn" id="btn2d02" data-view="02">ev0 vs ev2</button>
    <button class="btn" id="btn2d12" data-view="12">ev1 vs ev2</button>
    <div class="btn-sep"></div>
    <div class="scrub">
      <input type="range" id="scrub" min="0" value="0" step="1">
      <span id="frameReadout">0 / 0</span>
    </div>
  </div>
  <div id="layout" class="layout-4">
    <div class="panel" id="panel2d"><div class="panel-plot" id="plot2d"></div></div>
    <div class="panel"><div class="panel-plot" id="plot3d"></div></div>
    <div class="panel"><div class="panel-plot" id="plotEig"></div></div>
    <div class="panel" id="panelEvec"><div class="panel-plot" id="plotEvec"></div></div>
  </div>
  <div id="loadMsg" style="padding:1rem;color:#6b7280;font-size:0.9rem">Loading data\u2026</div>
<script>
(function(){{
'use strict';
function waitForPlotly(cb, tries) {{
  tries = tries || 0;
  if (typeof Plotly !== 'undefined') {{ cb(); return; }}
  if (tries < 60) {{ setTimeout(function(){{ waitForPlotly(cb, tries+1); }}, 100); }}
  else {{ document.getElementById('plot3d').innerHTML = '<div style="padding:2rem;color:red">Plotly failed to load.</div>'; }}
}}
waitForPlotly(function() {{
  (function(PD) {{
      var msg = document.getElementById('loadMsg');
      if (msg) msg.remove();

      var nFrames = PD.frames.length;
      var rx = PD.ranges.x, ry = PD.ranges.y, rz = PD.ranges.z;
      var f0 = PD.frames[0] || {{x:[],y:[],z:[],colors:[]}};
      var trueColors = (PD.true_colors && PD.true_colors.length) ? PD.true_colors : null;
      var hasEig = Array.isArray(PD.eig1_series) && PD.eig1_series.length > 0;

      var idx = 0, playing = false, lastTs = 0;
      var INTERVAL = 180;
      var dragging = false, wasPlayingDrag = false, wasPlayingPointer = false;
      var show2d = true, showEvec = true, view2d = '01';
      var plotsReady = false, userCamera = null;

      function updateGridClass() {{
        var el = document.getElementById('layout');
        if (show2d && showEvec)       el.className = 'layout-4';
        else if (show2d && !showEvec) el.className = 'layout-3';
        else if (!show2d && showEvec) el.className = 'layout-3-right';
        else                          el.className = 'layout-2';
      }}

      function currentColors() {{
        if (trueColors) return trueColors;
        var fr = PD.frames[idx];
        return (fr && fr.colors && fr.colors.length) ? fr.colors : 'rgb(160,160,160)';
      }}
      function get2dXY(fr) {{
        if (view2d==='01') return {{x:fr.x,y:fr.y,xLabel:'evec 0',yLabel:'evec 1'}};
        if (view2d==='02') return {{x:fr.x,y:fr.z,xLabel:'evec 0',yLabel:'evec 2'}};
        return {{x:fr.y,y:fr.z,xLabel:'evec 1',yLabel:'evec 2'}};
      }}
      function xyRange(view) {{
        if (view==='01') return {{xr:rx,yr:ry}};
        if (view==='02') return {{xr:rx,yr:rz}};
        return {{xr:ry,yr:rz}};
      }}

      var gd3 = document.getElementById('plot3d');
      var gd2 = document.getElementById('plot2d');
      var gdE = document.getElementById('plotEig');
      var gdS = document.getElementById('plotEvec');

      var p3 = Plotly.newPlot(gd3, [{{type:'scatter3d',mode:'markers',x:f0.x,y:f0.y,z:f0.z,
        marker:{{size:3,opacity:0.85,color:trueColors||f0.colors||'rgb(160,160,160)'}}}}],
        {{dragmode:'orbit',margin:{{l:0,r:0,b:0,t:0}},scene:{{bgcolor:'#fff',
          xaxis:{{title:'evec 0',gridcolor:'#e1e1e1',zerolinecolor:'#ccc',range:rx}},
          yaxis:{{title:'evec 1',gridcolor:'#e1e1e1',zerolinecolor:'#ccc',range:ry}},
          zaxis:{{title:'evec 2',gridcolor:'#e1e1e1',zerolinecolor:'#ccc',range:rz}},
          aspectmode:'cube',uirevision:'lock',camera:{{eye:{{x:1.6,y:1.6,z:0.9}}}}}}}},
        {{responsive:true,scrollZoom:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d']}});

      var p2 = Plotly.newPlot(gd2, [{{type:'scattergl',mode:'markers',x:f0.x,y:f0.y,
        marker:{{size:4,opacity:0.85,color:trueColors||f0.colors||'rgb(160,160,160)'}}}}],
        {{margin:{{l:40,r:10,b:40,t:10}},xaxis:{{title:'evec 0',gridcolor:'#e1e1e1',range:rx}},
          yaxis:{{title:'evec 1',gridcolor:'#e1e1e1',range:ry,scaleanchor:'x',scaleratio:1}},
          plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},
        {{responsive:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d']}});

      var pE = Promise.resolve();
      if (hasEig) {{
        var eigRange=PD.eig_range||[null,null];
        pE = Plotly.newPlot(gdE,[{{type:'scatter',mode:'lines+markers',x:[0],y:[PD.eig1_series[0]],
          line:{{color:'#2563eb',width:2}},marker:{{color:'#2563eb',size:6}}}}],
          {{margin:{{l:50,r:10,b:50,t:10}},
            xaxis:{{title:'Iteration',gridcolor:'#e1e1e1',range:[0,nFrames-1]}},
            yaxis:{{title:'Eigenvalue 1',gridcolor:'#e1e1e1',range:eigRange}},
            plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},
          {{responsive:true,displaylogo:false}});
      }} else {{
        gdE.innerHTML='<div style="display:flex;align-items:center;justify-content:center;height:100%;color:#9ca3af;font-size:0.9rem">No eigenvalue data</div>';
      }}

      // ---- sorted eigenvector plot ----
      var nodeIdx=f0.x.map(function(_,i){{return i;}});
      var s0init=f0.x.slice().sort(function(a,b){{return a-b;}});
      var s1init=f0.y.slice().sort(function(a,b){{return a-b;}});
      var pS = Plotly.newPlot(gdS,[
        {{type:'scatter',mode:'lines',name:'evec 1',x:nodeIdx,y:s0init,line:{{color:'#2563eb',width:1.5}}}},
        {{type:'scatter',mode:'lines',name:'evec 2',x:nodeIdx,y:s1init,line:{{color:'#dc2626',width:1.5}}}}
      ],{{margin:{{l:45,r:10,b:40,t:10}},
        xaxis:{{title:'Node index',gridcolor:'#e1e1e1'}},
        yaxis:{{title:'Value',gridcolor:'#e1e1e1',range:PD.evec01_range}},
        legend:{{x:0.02,y:0.98,bgcolor:'rgba(255,255,255,0.8)',font:{{size:11}}}},
        plot_bgcolor:'#fff',paper_bgcolor:'#fff'}},
        {{responsive:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d']}});

      function toFrame(k) {{
        if (!plotsReady) return;
        idx = Math.max(0,Math.min(k,nFrames-1));
        var fr=PD.frames[idx]; if (!fr) return;
        var cols=currentColors();
        Plotly.restyle(gd3,{{x:[fr.x],y:[fr.y],z:[fr.z],'marker.color':[cols]}},[0]);
        if (show2d) {{
          var xy=get2dXY(fr);
          Plotly.restyle(gd2,{{x:[xy.x],y:[xy.y],'marker.color':[cols]}},[0]);
        }}
        if (hasEig) {{
          var xs=PD.eig1_series.slice(0,idx+1).map(function(_,i){{return i;}});
          var ys=PD.eig1_series.slice(0,idx+1);
          Plotly.restyle(gdE,{{x:[xs],y:[ys]}},[0]);
        }}
        if (showEvec) {{
          var s0=fr.x.slice().sort(function(a,b){{return a-b;}});
          var s1=fr.y.slice().sort(function(a,b){{return a-b;}});
          Plotly.restyle(gdS,{{y:[s0,s1]}},[0,1]);
        }}
        if (scrub) scrub.value=String(idx);
        if (readout) readout.textContent=idx+' / '+(nFrames-1);
      }}

      function raf(ts) {{
        if (!playing) return;
        if (ts-lastTs>=INTERVAL) {{
          lastTs=ts;
          if (idx<nFrames-1) {{ toFrame(idx+1); }} else {{ playing=false; return; }}
        }}
        requestAnimationFrame(raf);
      }}

      Promise.all([p3,p2,pE,pS]).then(function() {{
        gd3.on('plotly_relayout', function(ed) {{
          if (ed['scene.camera']) {{
            userCamera = ed['scene.camera'];
            if (gd3.layout && gd3.layout.scene)
              gd3.layout.scene.camera = userCamera;
          }}
        }});
        plotsReady=true; toFrame(0); playing=true; requestAnimationFrame(raf);
      }});

      function hookPointer(el) {{
        if (!el) return;
        el.addEventListener('pointerdown',function(){{ if(playing){{wasPlayingPointer=true;playing=false;}} }});
        function resume(){{ if(wasPlayingPointer){{wasPlayingPointer=false;playing=true;requestAnimationFrame(raf);}} }}
        el.addEventListener('pointerup',resume); el.addEventListener('pointerleave',resume);
      }}
      hookPointer(gd3); hookPointer(gd2);

      var scrub=document.getElementById('scrub');
      var readout=document.getElementById('frameReadout');
      if (scrub) scrub.max=String(Math.max(0,nFrames-1));

      document.getElementById('btnPlay').onclick=function(){{ if(!playing){{playing=true;requestAnimationFrame(raf);}} }};
      document.getElementById('btnPause').onclick=function(){{ playing=false; }};
      document.getElementById('btnRestart').onclick=function(){{ playing=false;toFrame(0); }};

      document.getElementById('btnToggle2D').onclick=function() {{
        show2d=!show2d;
        var panel=document.getElementById('panel2d');
        var btn=document.getElementById('btnToggle2D');
        var viewBtns=document.querySelectorAll('[data-view]');
        if (show2d) {{
          panel.style.display=''; btn.textContent='Hide 2D';
          viewBtns.forEach(function(b){{b.style.display='';}});
          Plotly.Plots.resize(gd2);
        }} else {{
          panel.style.display='none'; btn.textContent='Show 2D';
          viewBtns.forEach(function(b){{b.style.display='none';}});
        }}
        updateGridClass();
        Plotly.Plots.resize(gd3); Plotly.Plots.resize(gdE);
        if (showEvec) Plotly.Plots.resize(gdS);
      }};

      document.getElementById('btnToggleEvec').onclick=function() {{
        showEvec=!showEvec;
        var panel=document.getElementById('panelEvec');
        var btn=document.getElementById('btnToggleEvec');
        if (showEvec) {{
          panel.style.display=''; btn.textContent='Hide Sorted Evecs';
          Plotly.Plots.resize(gdS);
        }} else {{
          panel.style.display='none'; btn.textContent='Show Sorted Evecs';
        }}
        updateGridClass();
        Plotly.Plots.resize(gd3); Plotly.Plots.resize(gdE);
        if (show2d) Plotly.Plots.resize(gd2);
      }};

      ['btn2d01','btn2d02','btn2d12'].forEach(function(id) {{
        var btn=document.getElementById(id); if (!btn) return;
        btn.onclick=function() {{
          view2d=btn.getAttribute('data-view');
          ['btn2d01','btn2d02','btn2d12'].forEach(function(b){{document.getElementById(b).classList.remove('btn-active');}});
          btn.classList.add('btn-active');
          var rng=xyRange(view2d), fr=PD.frames[idx], xy=get2dXY(fr), cols=currentColors();
          Plotly.restyle(gd2,{{x:[xy.x],y:[xy.y],'marker.color':[cols]}},[0]);
          Plotly.relayout(gd2,{{'xaxis.title':xy.xLabel,'xaxis.range':rng.xr,'yaxis.title':xy.yLabel,'yaxis.range':rng.yr}});
        }};
      }});

      if (scrub) {{
        scrub.addEventListener('pointerdown',function(){{wasPlayingDrag=playing;playing=false;}});
        scrub.addEventListener('input',function(){{toFrame(+scrub.value);}});
        scrub.addEventListener('pointerup',function(){{if(wasPlayingDrag){{wasPlayingDrag=false;playing=true;requestAnimationFrame(raf);}}}});
      }}

      window.addEventListener('keydown',function(e) {{
        if (e.target&&(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')) return;
        if (e.code==='Space') {{ e.preventDefault(); if(playing){{playing=false;}}else{{playing=true;requestAnimationFrame(raf);}} }}
        if (e.code==='ArrowRight') toFrame(idx+1);
        if (e.code==='ArrowLeft')  toFrame(idx-1);
      }});
      window.addEventListener('resize',function(){{
        Plotly.Plots.resize(gd3);
        if(show2d) Plotly.Plots.resize(gd2);
        Plotly.Plots.resize(gdE);
        if(showEvec) Plotly.Plots.resize(gdS);
      }});
  }})(window.ANIMATION_DATA);
}});
}})();
</script>
</body>
</html>
"""


def save_animation(evec_seq, eigenvalue_seq=None, labels=None, colors=None,
                   title="Eigenvector Animation", path="animation"):
    """
    Save the three-panel animation as ``path.html`` + ``path.json`` and open
    it in the browser.  Loading data via fetch is much faster than embedding
    it inline for large datasets.

    Parameters
    ----------
    evec_seq : sequence of array-like, each (n, d>=3)
    eigenvalue_seq : sequence of array-like, optional
    labels : array-like of int, optional
    colors : sequence of list-of-str, optional
    title : str
    path : str
        Base path (without extension).  Two files are written:
        ``path.html`` and ``path_data.js``.
    """
    seq = [np.asarray(E, float) for E in evec_seq]
    if not seq:
        raise ValueError("evec_seq must contain at least one frame.")
    n, d = seq[0].shape
    if d < 3:
        raise ValueError(f"Each eigenvector matrix must have >= 3 columns, got {d}.")

    T = len(seq)
    if colors is not None:
        colors_seq = list(colors)
    elif labels is not None:
        c = _colors_from_true_labels(labels)
        colors_seq = [c] * T
    else:
        gray = ['rgb(160,160,160)'] * n
        colors_seq = [gray] * T

    if eigenvalue_seq is not None and len(eigenvalue_seq) != T:
        raise ValueError("eigenvalue_seq must have the same length as evec_seq.")

    payload = _build_payload(seq, colors_seq, labels=labels,
                             eigenvalue_seq=eigenvalue_seq)

    js_path   = path + '_data.js'
    html_path = path + '.html'
    js_file   = os.path.basename(js_path)

    with open(js_path, 'w') as f:
        f.write('window.ANIMATION_DATA=')
        json.dump(payload, f, separators=(',', ':'))
        f.write(';')

    html = _HTML_FETCH_TMPL.format(
        title=title,
        style=_STYLE,
        js_file=js_file,
    )
    with open(html_path, 'w') as f:
        f.write(html)

    webbrowser.open('file://' + os.path.abspath(html_path))
    print(f"Saved: {html_path}  +  {js_path}")


# ---------------------------------------------------------------------------
# 2D animation
# ---------------------------------------------------------------------------

_HTML_2D_TMPL = """\
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"
          onerror="this.onerror=null;this.src='https://cdn.jsdelivr.net/npm/plotly.js@2.35.2/dist/plotly.min.js'"></script>
  <style>{style}</style>
</head>
<body>
  <h1>{title}</h1>

  <div class="ctrl-bar" id="ctrlBar">
    <button class="btn btn-primary" id="btnPlay">Play</button>
    <button class="btn" id="btnPause">Pause</button>
    <button class="btn" id="btnRestart">Restart</button>
    <div class="btn-sep"></div>
    <div class="scrub">
      <input type="range" id="scrub" min="0" value="0" step="1">
      <span id="frameReadout">0 / 0</span>
    </div>
  </div>

  <div class="panel" style="max-width:700px;">
    <div class="panel-plot" id="plot2d"></div>
  </div>

<script>
(function(){{
'use strict';

function waitForPlotly(cb, tries) {{
  tries = tries || 0;
  if (typeof Plotly !== 'undefined') {{ cb(); return; }}
  if (tries < 60) {{ setTimeout(function(){{ waitForPlotly(cb, tries+1); }}, 100); }}
  else {{ document.getElementById('plot2d').innerHTML = '<div style="padding:2rem;color:red">Plotly failed to load.</div>'; }}
}}

waitForPlotly(function() {{

const PD = {data_json};

const nFrames = PD.frames.length;
const f0 = PD.frames[0] || {{x:[], y:[], colors:[]}};
const trueColors = (PD.true_colors && PD.true_colors.length) ? PD.true_colors : null;
const rx = PD.ranges.x, ry = PD.ranges.y;

let idx = 0;
let playing = false;
let lastTs = 0;
const INTERVAL = 180;
let wasPlayingDrag = false;
let wasPlayingPointer = false;
let plotReady = false;

function currentColors() {{
  if (trueColors) return trueColors;
  const fr = PD.frames[idx];
  return (fr && fr.colors && fr.colors.length) ? fr.colors : 'rgb(160,160,160)';
}}

const gd = document.getElementById('plot2d');

var pReady = Plotly.newPlot(gd, [{{
  type: 'scattergl', mode: 'markers',
  x: f0.x, y: f0.y,
  marker: {{ size: 5, opacity: 0.85, color: trueColors || f0.colors || 'rgb(160,160,160)' }}
}}], {{
  margin: {{l:50,r:20,b:50,t:20}},
  xaxis: {{ title:'Eigenvector 1', gridcolor:'#e1e1e1', range: rx }},
  yaxis: {{ title:'Eigenvector 2', gridcolor:'#e1e1e1', range: ry, scaleanchor:'x', scaleratio:1 }},
  plot_bgcolor:'#fff', paper_bgcolor:'#fff'
}}, {{ responsive:true, displaylogo:false,
      modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'] }});

function toFrame(k) {{
  if (!plotReady) return;
  idx = Math.max(0, Math.min(k, nFrames - 1));
  const fr = PD.frames[idx];
  if (!fr) return;
  Plotly.restyle(gd, {{ x:[fr.x], y:[fr.y], 'marker.color':[currentColors()] }}, [0]);
  if (scrub) scrub.value = String(idx);
  if (readout) readout.textContent = idx + ' / ' + (nFrames - 1);
}}

function raf(ts) {{
  if (!playing) return;
  if (ts - lastTs >= INTERVAL) {{
    lastTs = ts;
    if (idx < nFrames - 1) {{ toFrame(idx + 1); }}
    else {{ playing = false; return; }}
  }}
  requestAnimationFrame(raf);
}}

pReady.then(function() {{
  plotReady = true;
  toFrame(0);
  playing = true;
  requestAnimationFrame(raf);
}});

gd.addEventListener('pointerdown', function() {{
  if (playing) {{ wasPlayingPointer = true; playing = false; }}
}});
function resumePointer() {{
  if (wasPlayingPointer) {{ wasPlayingPointer = false; playing = true; requestAnimationFrame(raf); }}
}}
gd.addEventListener('pointerup', resumePointer);
gd.addEventListener('pointerleave', resumePointer);

const scrub  = document.getElementById('scrub');
const readout = document.getElementById('frameReadout');
if (scrub) scrub.max = String(Math.max(0, nFrames - 1));

document.getElementById('btnPlay').onclick = function() {{
  if (!playing) {{ playing = true; requestAnimationFrame(raf); }}
}};
document.getElementById('btnPause').onclick = function() {{ playing = false; }};
document.getElementById('btnRestart').onclick = function() {{ playing = false; toFrame(0); }};

if (scrub) {{
  scrub.addEventListener('pointerdown', function() {{ wasPlayingDrag = playing; playing = false; }});
  scrub.addEventListener('input', function() {{ toFrame(+scrub.value); }});
  scrub.addEventListener('pointerup', function() {{
    if (wasPlayingDrag) {{ wasPlayingDrag = false; playing = true; requestAnimationFrame(raf); }}
  }});
}}

window.addEventListener('keydown', function(e) {{
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
  if (e.code === 'Space') {{
    e.preventDefault();
    if (playing) {{ playing = false; }} else {{ playing = true; requestAnimationFrame(raf); }}
  }}
  if (e.code === 'ArrowRight') toFrame(idx + 1);
  if (e.code === 'ArrowLeft')  toFrame(idx - 1);
}});

window.addEventListener('resize', function() {{ Plotly.Plots.resize(gd); }});

}}); // waitForPlotly
}})();
</script>
</body>
</html>
"""


def animate_evecs_2d(evec_seq, labels=None, colors=None,
                     title="Eigenvector Animation (2D)"):
    """
    Animated 2D scatter plot of the 1st vs 2nd eigenvector across steps.

    Parameters
    ----------
    evec_seq : sequence of array-like, each (n, d>=2)
        Eigenvector matrices, one per animation frame.
    labels : array-like of int, optional
        True node labels used for permanent coloring.
    colors : sequence of list-of-str, optional
        Per-frame color arrays. Ignored if *labels* is provided.
    title : str
        Title displayed above the animation.

    Returns
    -------
    IPython.display.HTML
    """
    seq = [np.asarray(E, float) for E in evec_seq]
    if not seq:
        raise ValueError("evec_seq must contain at least one frame.")
    n, d = seq[0].shape
    if d < 2:
        raise ValueError(f"Each eigenvector matrix must have >= 2 columns, got {d}.")

    T = len(seq)

    if colors is not None:
        if len(colors) != T:
            raise ValueError("colors must have the same length as evec_seq.")
        colors_seq = list(colors)
    elif labels is not None:
        c = _colors_from_true_labels(labels)
        colors_seq = [c] * T
    else:
        gray = ['rgb(160,160,160)'] * n
        colors_seq = [gray] * T

    frames, xs, ys = [], [], []
    for E, cols in zip(seq, colors_seq):
        x = np.asarray(E[:, 0], float).ravel()
        y = np.asarray(E[:, 1], float).ravel()
        frames.append({'x': x.tolist(), 'y': y.tolist(), 'colors': cols})
        xs.append(x); ys.append(y)

    payload = {
        'frames': frames,
        'ranges': {'x': _axis_range(xs), 'y': _axis_range(ys)},
    }
    if labels is not None:
        payload['true_colors'] = _colors_from_true_labels(np.asarray(labels).ravel())

    html = _HTML_2D_TMPL.format(
        title=title,
        style=_STYLE,
        data_json=json.dumps(payload),
    )
    return HTML(html)


# ---------------------------------------------------------------------------
# Math utilities (kept from old file)
# ---------------------------------------------------------------------------



def comp_vec(v_next, v):
    if sum((v / v_next) < 0) > (len(v_next) / 2):
        return v_next * -1
    return v_next


def Transform(evec_old, evec_new):
    cols = []
    for j in range(3):
        cols.append(comp_vec(evec_new[:, j], evec_old[:, j]).reshape(-1, 1))
    return np.hstack(cols)
