"""Standalone TAPF-MIN Falling Walls executable.

Self-contained local web app for presentation and app-development prototyping.
Uses the repository's real k-ary randomized-response mechanism. The local FER
posterior remains a clearly-labelled demonstration input until a trained FER
checkpoint is supplied.
"""
from __future__ import annotations

import json
import math
import os
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

from tapf.formal_privacy import randomized_response

EMOTIONS = ("ANG", "DIS", "FEA", "HAP", "NEU", "SAD")
NAMES = {"ANG":"Anger","DIS":"Disgust","FEA":"Fear","HAP":"Happiness","NEU":"Neutral","SAD":"Sadness"}

FROZEN_RESULTS = {
    "TAPF-MIN v2.1 frozen validation": {
        "clips": 144, "actors": 24, "emotion_macro_f1": 0.230,
        "emotion_f1_ci95": "0.165–0.297", "clip_identity_auc": 0.573,
        "identity_auc_ci95": "0.474–0.665", "repeated_release_identity_auc": 0.693,
        "release_decision": "BLOCK", "note": "Frozen baseline. No eligible operating point."
    },
    "v3 deep-FER development run": {
        "clips": 144, "actors": 24, "emotion_accuracy": 0.1597,
        "emotion_macro_f1": 0.1505, "emotion_f1_ci95": "0.091–0.212",
        "posterior_identity_auc_warning": "≈0.998 (confounded OOF audit; not a clean leakage estimate)",
        "release_decision": "BLOCK", "note": "Development diagnostic only. Actor-disjoint FER remained near chance."
    },
    "v3.1 hardening status": {
        "engineering_ci": "PASS",
        "architecture": "Face-focused RGB + motion residuals + temporal GRU/attention + delayed GRL",
        "privacy_release": "Minimum disclosure + formal DP + persistent composition architecture",
        "performance_status": "Benchmark not yet claimed",
        "release_decision": "BLOCK until empirical gate passes",
        "note": "No fabricated performance result. Final frozen-model audit still required."
    }
}

_state_lock = threading.Lock()
_state = {"epsilon_spent": 0.0, "release_count": 0}


def normalise(values: list[float]) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.shape != (6,) or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("posterior must contain six finite non-negative values")
    total = float(x.sum())
    return x / total if total > 0 else np.full(6, 1.0 / 6.0)


def release_demo(payload: dict[str, Any]) -> dict[str, Any]:
    posterior = normalise([float(v) for v in payload.get("posterior", [])])
    epsilon = float(payload.get("epsilon", 1.0))
    budget = float(payload.get("budget", 4.0))
    if not math.isfinite(epsilon) or not math.isfinite(budget) or epsilon <= 0 or budget <= 0:
        raise ValueError("epsilon and budget must be positive finite numbers")

    with _state_lock:
        remaining = budget - float(_state["epsilon_spent"])
        if epsilon > remaining + 1e-12:
            return {
                "decision": "BLOCK",
                "reason": "privacy_budget_exhausted",
                "epsilon_spent": _state["epsilon_spent"],
                "remaining_epsilon": max(0.0, remaining),
                "release_count": _state["release_count"],
            }

        local_label = EMOTIONS[int(np.argmax(posterior))]
        rr = randomized_response(
            local_label, EMOTIONS, epsilon=epsilon,
            rng=np.random.default_rng(), accountant=None
        )
        _state["epsilon_spent"] += epsilon
        _state["release_count"] += 1
        return {
            "decision": "RELEASE",
            "local_label": local_label,
            "local_label_name": NAMES[local_label],
            "payload": {
                "type": "categorical_label",
                "value": rr["label"],
                "label_name": NAMES[rr["label"]],
                "payload_bits": int(math.ceil(math.log2(len(EMOTIONS)))),
                "raw_video_released": False,
                "face_crop_released": False,
                "task_latent_released": False,
            },
            "epsilon_spent": _state["epsilon_spent"],
            "remaining_epsilon": max(0.0, budget - _state["epsilon_spent"]),
            "release_count": _state["release_count"],
        }


def reset_demo() -> dict[str, Any]:
    with _state_lock:
        _state["epsilon_spent"] = 0.0
        _state["release_count"] = 0
        return dict(_state)


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TAPF-MIN | Falling Walls Lab</title>
<style>
:root{--bg:#080b12;--panel:#111827;--line:#253047;--text:#f5f7fb;--muted:#9ca3af;--accent:#56d4ff;--good:#35d07f;--bad:#ff637d;--gold:#f6c453}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 20% 0,#132037 0,#080b12 42%);color:var(--text);font:15px/1.5 Inter,Segoe UI,Arial,sans-serif}
.wrap{max-width:1180px;margin:auto;padding:26px}.hero{padding:24px 0 18px}.eyebrow{color:var(--accent);font-weight:700;letter-spacing:.12em;text-transform:uppercase;font-size:12px}h1{font-size:42px;line-height:1.05;margin:8px 0}h2,h3{margin-top:0}.sub{color:var(--muted);max-width:900px}.tabs{display:flex;gap:10px;margin:18px 0}.tab{border:1px solid var(--line);background:#0d1422;color:var(--text);padding:10px 16px;border-radius:10px;cursor:pointer}.tab.active{border-color:var(--accent);color:var(--accent)}.view{display:none}.view.active{display:block}.grid{display:grid;grid-template-columns:1.2fr .8fr;gap:18px}@media(max-width:850px){.grid{grid-template-columns:1fr}}
.card{background:rgba(17,24,39,.92);border:1px solid var(--line);border-radius:16px;padding:20px;box-shadow:0 18px 50px rgba(0,0,0,.2)}.notice{border-left:3px solid var(--gold);background:#1c1820;padding:12px 14px;border-radius:8px;margin-bottom:16px;color:#f7df9b}.row{display:grid;grid-template-columns:150px 1fr 54px;gap:12px;align-items:center;margin:10px 0}input[type=range]{width:100%}.controls{display:grid;grid-template-columns:1fr 1fr;gap:12px}.controls label{display:flex;flex-direction:column;gap:6px;color:var(--muted)}select,input[type=number]{background:#0b1220;border:1px solid var(--line);color:var(--text);padding:10px;border-radius:8px}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0}.metric{background:#0b1220;border:1px solid var(--line);padding:12px;border-radius:10px}.metric b{display:block;font-size:23px}.metric span{color:var(--muted);font-size:12px}.btn{border:0;border-radius:10px;padding:12px 15px;font-weight:700;cursor:pointer}.primary{background:linear-gradient(90deg,#2f8cff,#4dd8ff);color:#06101a;width:100%}.secondary{background:#1a2333;color:var(--text);margin-top:8px;width:100%}.result{margin-top:14px;padding:14px;border-radius:10px;background:#0b1220;border:1px solid var(--line);min-height:110px;white-space:pre-wrap}.release{border-color:var(--good);color:#c9ffe1}.block{border-color:var(--bad);color:#ffd0d8}.boundary{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:14px}.yes,.no{padding:9px 10px;border-radius:8px;background:#0b1220}.yes{color:#bff8d5}.no{color:#ffc5cf}.evidence{margin-bottom:14px}.evidence h3{color:var(--accent)}table{width:100%;border-collapse:collapse}td{padding:7px 8px;border-bottom:1px solid var(--line)}td:first-child{color:var(--muted);width:38%}.arch{white-space:pre;overflow:auto;background:#070c14;border:1px solid var(--line);padding:20px;border-radius:12px;color:#d8f6ff}.claim{background:#0e2b25;border:1px solid #215c4d;padding:16px;border-radius:10px}.foot{color:var(--muted);font-size:12px;margin-top:26px;border-top:1px solid var(--line);padding-top:14px}
</style></head><body><div class="wrap"><div class="hero"><div class="eyebrow">Falling Walls Lab · Research Demonstration</div><h1>Breaking the Wall of Biometric Exposure</h1><div class="sub">TAPF-MIN — Minimum-Disclosure Biometric Privacy Firewall. Private sensing → local inference → minimum sufficient information → formally bounded release.</div></div>
<div class="tabs"><button class="tab active" onclick="show('live',this)">⚡ Live Firewall</button><button class="tab" onclick="show('evidence',this)">📊 Measured Evidence</button><button class="tab" onclick="show('architecture',this)">🧠 Architecture</button></div>
<section id="live" class="view active"><div class="notice"><b>Stage-safe mode.</b> The six-class posterior is a demonstration input to exercise the real privacy release layer. It is not a claimed FER benchmark prediction.</div><div class="grid">
<div class="card"><h2>1. Private local task posterior</h2><div id="sliders"></div><div class="metric" style="margin-top:16px"><span>Local task decision</span><b id="local">Happiness</b><span id="conf">confidence 61.0%</span></div></div>
<div class="card"><h2>2. Minimum-disclosure release</h2><div class="controls"><label>Session ε budget<input id="budget" type="number" min="0.25" max="20" step="0.25" value="4"></label><label>ε per release<select id="eps"><option>.25</option><option>.5</option><option selected>1</option><option>2</option><option>4</option></select></label></div><div class="metrics"><div class="metric"><span>ε spent</span><b id="spent">0.00</b></div><div class="metric"><span>Releases</span><b id="count">0</b></div><div class="metric"><span>Task payload</span><b>3 bits</b></div></div><button class="btn primary" onclick="releaseIt()">🔐 Request protected release</button><button class="btn secondary" onclick="resetIt()">Reset presentation session</button><div id="result" class="result">No release requested yet.</div><div class="boundary"><div class="yes">✅ Protected task label</div><div class="no">❌ Raw video</div><div class="no">❌ Face crop</div><div class="no">❌ Task embedding / latent</div></div></div></div></section>
<section id="evidence" class="view"><div class="notice"><b>Scientific rule:</b> BLOCK means BLOCK. Failed or incomplete experiments are never converted into privacy claims.</div><div id="ev"></div></section>
<section id="architecture" class="view"><div class="card"><h2>System boundary</h2><div class="arch">RAW VIDEO (private device)
      │
      ▼
Face detection + bounded tracking + alignment
      │
      ▼
RGB encoder ─┐
             ├─ Gated fusion → GRU → attention → FER posterior
Motion CNN ──┘
      │
      ▼
Minimum-disclosure selector
      │
      ▼
Formal DP + composition control
      │
      ├── RELEASE: protected task output only
      └── BLOCK: no eligible safe release</div><h3 style="margin-top:18px">Presentation sentence</h3><div class="claim">Instead of sending a patient's face to an AI system, TAPF-MIN keeps the biometric video local and releases only the minimum task information under a measurable privacy budget.</div></div></section>
<div class="foot">Research demonstration — controlled CREMA-D evidence; not clinical validation, not an anonymity guarantee, and not a claim that differential privacy alone prevents every biometric attack.</div></div>
<script>
const codes=['ANG','DIS','FEA','HAP','NEU','SAD'];const names={ANG:'Anger',DIS:'Disgust',FEA:'Fear',HAP:'Happiness',NEU:'Neutral',SAD:'Sadness'};const def=[.06,.05,.04,.61,.17,.07];
function show(id,b){document.querySelectorAll('.view').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));document.getElementById(id).classList.add('active');b.classList.add('active')}
const s=document.getElementById('sliders');codes.forEach((c,i)=>{const r=document.createElement('div');r.className='row';r.innerHTML=`<label>${names[c]} (${c})</label><input type="range" id="p${i}" min="0" max="1" step=".01" value="${def[i]}"><span id="v${i}">${def[i].toFixed(2)}</span>`;s.appendChild(r);document.getElementById('p'+i).oninput=updateLocal});
function posterior(){let x=codes.map((_,i)=>+document.getElementById('p'+i).value);let z=x.reduce((a,b)=>a+b,0)||1;return x.map(v=>v/z)}function updateLocal(){codes.forEach((_,i)=>document.getElementById('v'+i).textContent=(+document.getElementById('p'+i).value).toFixed(2));let p=posterior(),m=Math.max(...p),i=p.indexOf(m);document.getElementById('local').textContent=names[codes[i]];document.getElementById('conf').textContent='confidence '+(m*100).toFixed(1)+'%'}
async function post(path,body){let r=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})});let j=await r.json();if(!r.ok)throw new Error(j.error||'request failed');return j}
async function releaseIt(){let box=document.getElementById('result');try{let j=await post('/api/release',{posterior:posterior(),epsilon:+document.getElementById('eps').value,budget:+document.getElementById('budget').value});document.getElementById('spent').textContent=(j.epsilon_spent||0).toFixed(2);document.getElementById('count').textContent=j.release_count||0;box.className='result '+(j.decision==='RELEASE'?'release':'block');box.textContent=j.decision==='RELEASE'?`RELEASE → ${j.payload.label_name} (${j.payload.value})\n\nOnly protected 3-bit task label crossed the boundary.\nRaw video released: false\nTask latent released: false`:`BLOCK → privacy budget exhausted\nRemaining ε: ${(j.remaining_epsilon||0).toFixed(2)}`;}catch(e){box.className='result block';box.textContent='ERROR → '+e.message}}
async function resetIt(){let j=await post('/api/reset',{});document.getElementById('spent').textContent='0.00';document.getElementById('count').textContent='0';let b=document.getElementById('result');b.className='result';b.textContent='Presentation session reset.'}
fetch('/api/evidence').then(r=>r.json()).then(d=>{let root=document.getElementById('ev');Object.entries(d).forEach(([title,m])=>{let div=document.createElement('div');div.className='card evidence';let rows=Object.entries(m).map(([k,v])=>`<tr><td>${k.replaceAll('_',' ')}</td><td>${v}</td></tr>`).join('');div.innerHTML=`<h3>${title}</h3><table>${rows}</table>`;root.appendChild(div)})});updateLocal();
</script></body></html>'''


class Handler(BaseHTTPRequestHandler):
    server_version = "TAPFMIN/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        if os.environ.get("TAPF_DEMO_VERBOSE") == "1":
            super().log_message(fmt, *args)

    def _send_json(self, obj: Any, status: int = 200) -> None:
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/api/evidence":
            self._send_json(FROZEN_RESULTS)
            return
        if self.path == "/healthz":
            self._send_json({"status": "ok", "app": "TAPF-MIN-Falling-Walls"})
            return
        if self.path in ("/", "/index.html"):
            data = HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self.send_error(404)

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length > 65536:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/api/release":
                self._send_json(release_demo(body))
                return
            if self.path == "/api/reset":
                self._send_json(reset_demo())
                return
            self.send_error(404)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, 400)


def choose_port() -> int:
    raw = os.environ.get("TAPF_DEMO_PORT", "8765").strip()
    port = int(raw)
    if not 1 <= port <= 65535:
        raise ValueError("TAPF_DEMO_PORT must be between 1 and 65535")
    return port


def main() -> int:
    port = choose_port()
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}"
    if os.environ.get("TAPF_DEMO_NO_BROWSER", "0") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
