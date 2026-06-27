"""
Repo Analyzer - 4 agents loop using claude-agent-sdk with custom LM Studio Transport
Auto-push to GitHub Repository via SSH
"""

import asyncio
import json
import re
import sys
import time
import os
import shutil
import tempfile
import subprocess
from dataclasses import dataclass
from typing import Any, AsyncIterator

import aiohttp
import requests

from claude_agent_sdk import (
    ClaudeAgentOptions,
    Transport,
    query,
    AssistantMessage,
    TextBlock,
    ResultMessage,
)

LM_STUDIO_URL = "http://localhost:1234/api/v1/chat"
MODEL = "google/gemma-4-26b-a4b-qat"

# GitHub Target Repo (SSH - usa la clave swarm_bot)
GITHUB_TARGET_REPO_SSH = "git@github.com:serranogallegogerardo/swarm-code.git"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/swarm_bot")

# The Business Goal
GOAL = """
/goal: Generar un veredicto ejecutivo enfocado en negocio. 
Debes alcanzar una puntuacion minima de 9.5/10 en claridad y valor estrategico. 
Traduce las metricas tecnicas a impacto financiero (costo de mantenimiento), 
riesgo operativo (seguridad/estabilidad) y escalabilidad. 
Termina con un claro GO / NO-GO / GO WITH RESERVATIONS.
"""

# ---------------------------------------------------------------------------
# Custom Transport that speaks to LM Studio
# ---------------------------------------------------------------------------
class LMStudioTransport(Transport):
    def __init__(self, api_url: str, model: str, system_prompt: str = ""):
        self.api_url = api_url
        self.model = model
        self.system_prompt = system_prompt
        self._ready = False
        self._q: asyncio.Queue[dict] = asyncio.Queue()

    async def connect(self) -> None:
        self._ready = True

    async def write(self, data: str) -> None:
        msg = json.loads(data)
        t = msg.get("type")

        if t == "control_request":
            rid = msg["request_id"]
            sub = msg["request"].get("subtype", "")
            if sub == "initialize":
                await self._q.put({
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": rid, "response": {}},
                })
            else:
                await self._q.put({
                    "type": "control_response",
                    "response": {"subtype": "error", "request_id": rid, "error": f"unsupported: {sub}"},
                })

        elif t == "user":
            content = msg["message"]["content"]
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "model": self.model,
                        "system_prompt": self.system_prompt,
                        "input": content,
                    }
                    async with session.post(self.api_url, json=payload) as resp:
                        resp.raise_for_status()
                        data = await resp.json()

                resp_text = ""
                if "output" in data:
                    for out in data["output"]:
                        if out.get("type") == "message":
                            resp_text = out.get("content", "")
                            break
                    if not resp_text and data["output"]:
                        last = data["output"][-1]
                        resp_text = last.get("content", "") if isinstance(last, dict) else str(last)
                elif "choices" in data:
                    resp_text = data["choices"][0].get("message", {}).get("content", "")

                await self._q.put({
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": resp_text}],
                        "model": self.model,
                    },
                })
            except Exception as e:
                await self._q.put({
                    "type": "assistant",
                    "message": {
                        "content": [{"type": "text", "text": f"[ERROR] {e}"}],
                        "model": self.model,
                    },
                })

            await self._q.put({
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "duration_ms": 0,
                "duration_api_ms": 0,
                "num_turns": 1,
                "session_id": "",
            })

    def read_messages(self) -> AsyncIterator[dict[str, Any]]:
        return self._read_gen()

    async def _read_gen(self) -> AsyncIterator[dict[str, Any]]:
        while True:
            m = await self._q.get()
            yield m
            if m.get("type") in ("result", "error"):
                break

    async def close(self) -> None:
        self._ready = False

    def is_ready(self) -> bool:
        return self._ready

    async def end_input(self) -> None:
        pass

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------
@dataclass
class Agent:
    name: str
    role: str
    focus: str

AGENTS = [
    Agent(
        "Code Quality Agent",
        "Eres un experto en calidad de codigo y DevOps. Analiza estructura, testing, deuda tecnica y mantenibilidad. Enfocate en el costo a largo plazo para la empresa.",
        "calidad y costo de mantenimiento",
    ),
    Agent(
        "Security Agent",
        "Eres un experto en seguridad. Analiza vulnerabilidades, manejo de secrets y cumplimiento. Enfocate en el riesgo para la reputacion y legal de la empresa.",
        "riesgo de seguridad y compliance",
    ),
    Agent(
        "Architecture Agent",
        "Eres un arquitecto de software senior. Analiza diseno arquitectural, escalabilidad y patrones. Enfocate en si el sistema soportara el crecimiento del negocio.",
        "escalabilidad y alineacion con objetivos de negocio",
    ),
]

# ---------------------------------------------------------------------------
# Repo fetching
# ---------------------------------------------------------------------------
def fetch_repo_info(url: str) -> tuple[str, dict, str | None]:
    m = re.search(r"github\.com/([\w.-]+/[\w.-]+)", url)
    if not m:
        return "", {}, "Invalid GitHub URL"
    repo_path = m.group(1).rstrip("/")

    r = requests.get(f"https://api.github.com/repos/{repo_path}", timeout=10)
    data = r.json() if r.status_code == 200 else {}

    lang_r = requests.get(
        data.get("languages_url", f"https://api.github.com/repos/{repo_path}/languages"),
        timeout=10,
    )
    data["languages"] = lang_r.json() if lang_r.status_code == 200 else {}

    readme = ""
    for ext in ("", ".md", ".rst"):
        rm = requests.get(
            f"https://api.github.com/repos/{repo_path}/readme{ext}",
            headers={"Accept": "application/vnd.github.raw+json"},
            timeout=10,
        )
        if rm.status_code == 200:
            readme = rm.text
            break

    return readme, data, None

def build_context(url: str, readme: str, data: dict) -> str:
    lines = [
        f"URL: {url}",
        f"Descripcion: {data.get('description', 'N/A')}",
        f"Lenguaje: {data.get('language', 'N/A')}",
        f"Lenguajes: {', '.join(data.get('languages', {}).keys()) or 'N/A'}",
        f"Stars: {data.get('stargazers_count', 'N/A')} | Forks: {data.get('forks_count', 'N/A')}",
        f"Topics: {', '.join(data.get('topics', [])) or 'N/A'}",
        f"License: {data.get('license', {}).get('spdx_id', 'N/A') if data.get('license') else 'N/A'}",
        "",
        "README:",
        readme[:1500],
    ]
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Agent runner
# ---------------------------------------------------------------------------
async def run_agent(agent: Agent, context: str, agent_num: int, total: int) -> tuple[str, str]:
    print(f"\n  --- Agente {agent_num}/{total}: {agent.name} ---")

    system_prompt = (
        f"{agent.role}\n\n{GOAL}\n"
        "Da tu veredicto en espanol con este formato:\n"
        "PUNTUACION: X/10\n"
        "FORTALEZAS:\n- ...\n"
        "DEBILIDADES (Impacto de Negocio):\n- ...\n"
        "RECOMENDACIONES (ROI):\n- ...\n"
        "RESUMEN EJECUTIVO:\n...\n"
    )

    prompt = f"Analiza este repositorio de GitHub:\n\n{context}\n\nEnfocate especialmente en: {agent.focus}."

    transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
    options = ClaudeAgentOptions(system_prompt=system_prompt)

    result_text = ""
    async for message in query(prompt=prompt, options=options, transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result_text += block.text
    return agent.name, result_text

# ---------------------------------------------------------------------------
# Synthesis & Looping Evaluation
# ---------------------------------------------------------------------------
async def synthesize(url: str, results: list[tuple[str, str]], feedback: str = "") -> str:
    print("\n  --- Sintetizando veredicto final ---")

    condensed = []
    for name, text in results:
        score = re.search(r"PUNTUACION[:\s]*(\d+)", text, re.IGNORECASE)
        score_str = f"Puntuacion: {score.group(1)}/10" if score else "Puntuacion: N/A"
        summary_m = re.search(r"(?:RESUMEN|CONCLUSION)[:\n]+(.{1,500})", text, re.IGNORECASE | re.DOTALL)
        summary = summary_m.group(1).strip()[:300] if summary_m else text[-300:]
        condensed.append(f"=== {name} ===\n{score_str}\nResumen: {summary}")

    system_prompt = (
        "Eres un Lead Developer y CTO. Sintetiza 3 analisis en un veredicto unico orientado al C-Level. Responde en espanol."
        f"\n{GOAL}"
    )
    
    prompt = (
        f"Repo: {url}\n\nAnalisis:\n\n{chr(10).join(condensed)}\n\n"
        "Veredicto final:\n"
        "PUNTUACION GLOBAL: X/10\n"
        "IMPACTO FINANCIERO Y RIESGO:\n"
        "RECOMENDACIONES PRIORITARIAS (top 3):\n"
        "DECISION FINAL (GO / NO-GO / GO WITH RESERVATIONS):\n"
    )

    if feedback:
        prompt += f"\n\nIMPORTANTE - El evaluador pidio corregir lo siguiente en tu intento anterior:\n{feedback}\nMejora el veredicto basandote en este feedback."

    transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
    options = ClaudeAgentOptions(system_prompt=system_prompt)

    text = ""
    async for message in query(prompt=prompt, options=options, transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text += block.text
    return text

async def evaluate_goal(verdict: str) -> tuple[bool, str]:
    """4th Agent: Evaluates if the verdict meets the business goal."""
    print("\n  --- Evaluando si se alcanzo el Goal ---")
    system_prompt = (
        "Eres un auditor de calidad de software orientado a negocio. Evaluas si un veredicto cumple con el objetivo de claridad para directivos."
        f"\n{GOAL}"
    )
    
    prompt = (
        f"Evalua este veredicto:\n\n{verdict}\n\n"
        "Responde estrictamente en este formato:\n"
        "SCORE: X.X/10\n"
        "FEEDBACK: [Si el score es < 9.5, explica en una linea que falta. Si es >=9.5, escribe 'PERFECTO']"
    )

    transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
    options = ClaudeAgentOptions(system_prompt=system_prompt)

    text = ""
    async for message in query(prompt=prompt, options=options, transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text += block.text

    print(f"  Evaluacion: {text.strip()}")
    score_m = re.search(r"SCORE[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    fb_m = re.search(r"FEEDBACK[:\s]*(.+)", text, re.IGNORECASE | re.DOTALL)
    
    score = float(score_m.group(1)) if score_m else 0.0
    feedback = fb_m.group(1).strip() if fb_m else "No se encontro feedback."
    
    return score >= 9.5, feedback

# ---------------------------------------------------------------------------
# Premium HTML Report Generator (hybrid: template + AI content)
# ---------------------------------------------------------------------------
def extract_score(text: str) -> str:
    m = re.search(r"PUNTUACION[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return f"{m.group(1)}/10" if m else "N/A"

def score_class(val: float) -> str:
    if val <= 3: return "critical"
    if val <= 6: return "warning"
    if val <= 8: return "good"
    return "excellent"

def generate_html(
    repo_url: str,
    repo_data: dict,
    results: list[tuple[str, str]],
    verdict: str,
    output_path: str,
) -> str:
    repo_name = repo_data.get("full_name", repo_url.split("github.com/")[-1] if "github.com" in repo_url else repo_url)
    desc = repo_data.get("description", "")
    stars = repo_data.get("stargazers_count", 0)
    forks = repo_data.get("forks_count", 0)
    langs = ", ".join(repo_data.get("languages", {}).keys()) or "—"
    topics = ", ".join(repo_data.get("topics", [])) or "—"
    license_ = repo_data.get("license", {}).get("spdx_id", "N/A") if repo_data.get("license") else "N/A"
    open_issues = repo_data.get("open_issues_count", 0)
    issues_display = open_issues if isinstance(open_issues, int) else 0
    created = (repo_data.get("created_at", "") or "")[:10]
    pushed = (repo_data.get("pushed_at", "") or "")[:10]

    global_score = extract_score(verdict)
    global_val = float(global_score.split("/")[0]) if global_score != "N/A" else 0
    global_cls = score_class(global_val)

    agent_cards = ""
    for name, text in results:
        s = extract_score(text)
        v = float(s.split("/")[0]) if s != "N/A" else 0
        cls = score_class(v)
        body = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                   .replace('"', "&quot;").replace("\n", "&#10;"))
        short = name.replace(" Agent", "")
        agent_cards += f"""
        <div class="card">
          <div class="card-score {cls}">{s}</div>
          <div class="card-name">{name}</div>
          <div class="card-detail" onclick="this.classList.toggle('open')">
            <span class="card-toggle">Ver analisis</span>
            <div class="card-body">{body}</div>
          </div>
        </div>"""

    verdict_body = (verdict.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                         .replace('"', "&quot;"))

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Repo Analysis - {repo_name}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
*,*::before,*::after{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0a0e1a;color:#e2e8f0;min-height:100vh}}
.container{{max-width:1040px;margin:0 auto;padding:2rem 1.5rem}}
@keyframes fadeUp{{from{{opacity:0;transform:translateY(20px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes pulse{{0%,100%{{transform:scale(1)}}50%{{transform:scale(1.05)}}}}
.header{{animation:fadeUp .6s ease-out;margin-bottom:2rem}}
.header h1{{font-size:1.6rem;font-weight:700;color:#f1f5f9;margin-bottom:.3rem}}
.header .sub{{color:#64748b;font-size:.9rem}}
.badge{{display:inline-block;padding:.25rem .75rem;border-radius:999px;font-size:.75rem;font-weight:600;margin-right:.4rem;margin-bottom:.4rem}}
.badge-lang{{background:rgba(56,189,248,.15);color:#38bdf8;border:1px solid rgba(56,189,248,.25)}}
.badge-topic{{background:rgba(168,85,247,.15);color:#a78bfa;border:1px solid rgba(168,85,247,.25)}}
.repo-card{{background:linear-gradient(135deg,#131827,#1a1f35);border-radius:16px;padding:1.5rem;border:1px solid #1e293b;margin-bottom:2rem;animation:fadeUp .6s ease-out .1s both}}
.repo-card h2{{font-size:.85rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b;margin-bottom:1rem}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:1rem}}
.stat-item{{}}
.stat-value{{font-size:1.2rem;font-weight:700;color:#f1f5f9}}
.stat-label{{font-size:.75rem;color:#64748b;margin-top:.15rem}}
.global-score{{text-align:center;padding:2.5rem 1.5rem;margin-bottom:2rem;border-radius:16px;animation:fadeUp .6s ease-out .2s both;position:relative;overflow:hidden}}
.global-score.critical{{background:linear-gradient(135deg,#1a0e0e,#2d1515);border:1px solid rgba(239,68,68,.25)}}
.global-score.warning{{background:linear-gradient(135deg,#1a170e,#2d2515);border:1px solid rgba(245,158,11,.25)}}
.global-score.good{{background:linear-gradient(135deg,#0e1a0e,#152d15);border:1px solid rgba(132,204,22,.25)}}
.global-score.excellent{{background:linear-gradient(135deg,#0e1a1a,#152d2d);border:1px solid rgba(34,211,238,.25)}}
.global-score .number{{font-size:3.5rem;font-weight:800;line-height:1;margin-bottom:.5rem;animation:pulse 2s ease-in-out infinite}}
.global-score.critical .number{{color:#ef4444}}
.global-score.warning .number{{color:#f59e0b}}
.global-score.good .number{{color:#84cc16}}
.global-score.excellent .number{{color:#22d3ee}}
.global-score .label{{font-size:.85rem;font-weight:500;text-transform:uppercase;letter-spacing:.1em;color:#94a3b8}}
.global-score .sub-label{{font-size:.75rem;color:#64748b;margin-top:.4rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:1.2rem;margin-bottom:2rem;animation:fadeUp .6s ease-out .3s both}}
.card{{background:#131827;border-radius:12px;padding:1.5rem;border:1px solid #1e293b;transition:transform .2s,box-shadow .2s}}
.card:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.3)}}
.card-score{{font-size:2rem;font-weight:800;margin-bottom:.25rem}}
.card-score.critical{{color:#ef4444}}
.card-score.warning{{color:#f59e0b}}
.card-score.good{{color:#84cc16}}
.card-score.excellent{{color:#22d3ee}}
.card-name{{font-size:.85rem;color:#64748b;margin-bottom:.8rem}}
.card-detail{{cursor:pointer}}
.card-toggle{{font-size:.8rem;font-weight:600;color:#38bdf8;transition:color .15s}}
.card-toggle:hover{{color:#7dd3fc}}
.card-body{{display:none;font-size:.82rem;line-height:1.6;color:#94a3b8;margin-top:.7rem;white-space:pre-wrap;max-height:360px;overflow-y:auto;padding:.5rem;background:rgba(0,0,0,.2);border-radius:8px}}
.card-body::-webkit-scrollbar{{width:4px}}
.card-body::-webkit-scrollbar-thumb{{background:#475569;border-radius:2px}}
.card-detail.open .card-body{{display:block}}
.card-detail.open .card-toggle{{display:none}}
.verdict{{background:linear-gradient(135deg,#131827,#1a1f35);border-radius:16px;padding:2rem;border:1px solid #1e293b;margin-bottom:2rem;animation:fadeUp .6s ease-out .4s both}}
.verdict h2{{font-size:.85rem;text-transform:uppercase;letter-spacing:.08em;color:#a78bfa;margin-bottom:1rem}}
.verdict-body{{font-size:.9rem;line-height:1.7;color:#cbd5e1;white-space:pre-wrap}}
.verdict-body strong{{color:#f1f5f9}}
.footer{{text-align:center;color:#334155;font-size:.75rem;padding:2rem 0 0;border-top:1px solid #1e293b;animation:fadeUp .6s ease-out .5s both}}
@media(max-width:640px){{.container{{padding:1rem}}.global-score .number{{font-size:2.5rem}}.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>{repo_name}</h1>
    <div class="sub">{desc}</div>
  </div>

  <div class="repo-card">
    <h2>Repository Info</h2>
    <div class="stats">
      <div class="stat-item"><div class="stat-value">&#9733; {stars}</div><div class="stat-label">Stars</div></div>
      <div class="stat-item"><div class="stat-value">{forks}</div><div class="stat-label">Forks</div></div>
      <div class="stat-item"><div class="stat-value">{issues_display}</div><div class="stat-label">Open Issues</div></div>
      <div class="stat-item"><div class="stat-value">{langs}</div><div class="stat-label">Languages</div></div>
      <div class="stat-item"><div class="stat-value">{license_}</div><div class="stat-label">License</div></div>
    </div>
    <div style="margin-top:1rem">
      {''.join(f'<span class="badge badge-lang">{l}</span>' for l in repo_data.get('languages', {}).keys())}
      {''.join(f'<span class="badge badge-topic">{t}</span>' for t in repo_data.get('topics', []))}
    </div>
  </div>

  <div class="global-score {global_cls}">
    <div class="number">{global_score}</div>
    <div class="label">Global Score</div>
    <div class="sub-label">Repo Analysis Verdict</div>
  </div>

  <div class="grid">{agent_cards}</div>

  <div class="verdict">
    <h2>Final Verdict</h2>
    <div class="verdict-body">{verdict_body}</div>
  </div>

  <div class="footer">Generated by Repo Analyzer &#8212; powered by LM Studio</div>
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  HTML generado: {len(html)} bytes")
    return html

# ---------------------------------------------------------------------------
# Git Push Function (Via SSH con clave específica)
# ---------------------------------------------------------------------------
def push_to_github(filename: str, content: str, repo_ssh_url: str, ssh_key_path: str):
    print(f"\n  --- Subiendo reporte a GitHub via SSH ---")
    
    env = os.environ.copy()
    env["GIT_SSH_COMMAND"] = f'ssh -i "{ssh_key_path}" -o StrictHostKeyChecking=accept-new'
    
    temp_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(temp_dir, "repo")
    
    try:
        print("  [1/4] Clonando repositorio...")
        subprocess.run(["git", "clone", repo_ssh_url, repo_dir], check=True, capture_output=True, env=env)
        
        print("  [2/4] Escribiendo archivo HTML...")
        target_file = os.path.join(repo_dir, filename)
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(content)
            
        print("  [3/4] Haciendo git commit...")
        subprocess.run(["git", "-C", repo_dir, "add", filename], check=True, capture_output=True, env=env)
        commit_msg = f"chore(upload): {filename} - Auto-generated repo analysis"
        subprocess.run(["git", "-C", repo_dir, "commit", "-m", commit_msg], check=True, capture_output=True, env=env)
        
        print("  [4/4] Haciendo git push...")
        subprocess.run(["git", "-C", repo_dir, "push", "origin", "main"], check=True, capture_output=True, env=env)
        print("  [OK] Reporte subido exitosamente a GitHub!")
        
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace")
        print(f"  [ERROR] Falló la subida via Git: {stderr[:500]}")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ---------------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------------
def print_header(text: str, char: str = "=", width: int = 74):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")

async def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} <url_repo_a_analizar>")
        sys.exit(1)

    repo_url = sys.argv[1]

    print_header("REPO ANALYZER - GOAL ORIENTED LOOP + GIT SSH PUSH")
    print(f"  Modelo: {MODEL}")
    print(f"  Repo Target: {repo_url}")
    print(f"  Repo Output: {GITHUB_TARGET_REPO_SSH}")
    print(f"  SSH Key: {SSH_KEY_PATH}")

    # Fetch repo info
    print_header("FETCHING REPO INFO")
    readme, data, err = fetch_repo_info(repo_url)
    if err:
        print(f"  [WARN] {err}")
        ctx = f"URL: {repo_url}"
    else:
        print(f"  {data.get('full_name', 'N/A')}")
        ctx = build_context(repo_url, readme, data)

    # Run 3 agents sequentially
    print_header("EJECUTANDO 3 AGENTES DE ANALISIS")
    results = []
    start = time.time()

    for i, agent in enumerate(AGENTS, 1):
        name, text = await run_agent(agent, ctx, i, len(AGENTS))
        results.append((name, text))
        print(f"\n  --- Resultado: {name} ---")
        print(text)
        await asyncio.sleep(2)

    # Loop of Synthesis + Evaluation until Goal is met (or max 3 iterations)
    print_header("INICIANDO LOOP DE OPTIMIZACION HASTA ALCANZAR EL GOAL")
    
    max_iterations = 3
    verdict = ""
    goal_met = False
    feedback = ""

    for iteration in range(1, max_iterations + 1):
        print(f"\n  >>> Iteracion {iteration}/{max_iterations} <<<")
        verdict = await synthesize(repo_url, results, feedback)
        print("\n  Veredicto Actual:")
        print(verdict)
        
        goal_met, feedback = await evaluate_goal(verdict)
        
        if goal_met:
            print("\n  [SUCCESS] El veredicto alcanzo el Goal (>=9.5/10)!")
            break
        else:
            print(f"\n  [RETRY] Feedback recibido: {feedback}")
            await asyncio.sleep(1)

    # Generate HTML with AI
    print_header("GENERANDO REPORTE HTML FINAL")
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', repo_url.split('/')[-1])
    filename = f"repo_analysis_{clean_name}.html"
    
    html_content = generate_html(repo_url, data, results, verdict, filename)

    # Push to GitHub via SSH
    print_header("SUBIENDO A GITHUB VIA SSH")
    push_to_github(filename, html_content, GITHUB_TARGET_REPO_SSH, SSH_KEY_PATH)

    total = time.time() - start
    print(f"\n{'=' * 74}")
    print(f"  Workflow completado en {total:.0f} segundos")
    print(f"  Revisa tu repo: https://github.com/serranogallegogerardo/swarm-code/blob/main/{filename}")
    print(f"{'=' * 74}")

if __name__ == "__main__":
    asyncio.run(main())