"""
Repo Analyzer - 3 agents using claude-agent-sdk with custom LM Studio Transport
"""

import asyncio
import json
import re
import sys
import time
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


# ---------------------------------------------------------------------------
# Custom Transport that speaks to LM Studio (implements Transport ABC)
# ---------------------------------------------------------------------------
class LMStudioTransport(Transport):
    """Transport that proxies SDK calls directly to LM Studio.

    Instead of spawning the Claude Code CLI, this transport handles the
    control protocol internally and translates user messages into LM Studio
    API calls.
    """

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
        "Eres un experto en calidad de codigo. Analiza estructura, buenas practicas, testing, documentacion y legibilidad.",
        "calidad del codigo",
    ),
    Agent(
        "Security Agent",
        "Eres un experto en seguridad. Analiza vulnerabilidades, manejo de secrets, dependencias peligrosas y practicas de seguridad.",
        "seguridad",
    ),
    Agent(
        "Architecture Agent",
        "Eres un arquitecto de software senior. Analiza diseno arquitectonico, escalabilidad, patrones y organizacion.",
        "arquitectura",
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
# Agent runner using claude-agent-sdk query() + custom transport
# ---------------------------------------------------------------------------
async def run_agent(agent: Agent, context: str, agent_num: int, total: int) -> tuple[str, str]:
    print(f"\n  --- Agente {agent_num}/{total}: {agent.name} ---")

    system_prompt = (
        f"{agent.role}\n\n"
        "Da tu veredicto en espanol con este formato:\n"
        "PUNTUACION: X/10\n"
        "FORTALEZAS:\n- ...\n"
        "DEBILIDADES:\n- ...\n"
        "RECOMENDACIONES:\n- ...\n"
        "RESUMEN:\n...\n"
    )

    prompt = (
        f"Analiza este repositorio de GitHub:\n\n{context}\n\n"
        f"Enfocate especialmente en: {agent.focus}."
    )

    transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
    options = ClaudeAgentOptions(system_prompt=system_prompt)

    result_text = ""
    async for message in query(prompt=prompt, options=options, transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result_text += block.text
        elif isinstance(message, ResultMessage):
            pass  # end of response

    return agent.name, result_text


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------
async def synthesize(url: str, results: list[tuple[str, str]]) -> str:
    print("\n  --- Sintetizando veredicto final ---")

    condensed = []
    for name, text in results:
        score = re.search(r"PUNTUACION[:\s]*(\d+)", text, re.IGNORECASE)
        score_str = f"Puntuacion: {score.group(1)}/10" if score else "Puntuacion: N/A"
        summary_m = re.search(
            r"(?:RESUMEN|CONCLUSION)[:\n]+(.{1,500})", text, re.IGNORECASE | re.DOTALL
        )
        summary = summary_m.group(1).strip()[:300] if summary_m else text[-300:]
        condensed.append(f"=== {name} ===\n{score_str}\nResumen: {summary}")

    system_prompt = (
        "Eres un lead developer. Sintetiza 3 analisis en un veredicto unico. Responde en espanol."
    )
    prompt = (
        f"Repo: {url}\n\nAnalisis:\n\n{chr(10).join(condensed)}\n\n"
        "Veredicto final:\n"
        "PUNTUACION GLOBAL: X/10\n"
        "CONCLUSION:\n"
        "RECOMENDACIONES PRIORITARIAS (top 3):\n"
        "LO RECOMENDARIAS? (Si/No/Con reservas)"
    )

    transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
    options = ClaudeAgentOptions(system_prompt=system_prompt)

    text = ""
    async for message in query(prompt=prompt, options=options, transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text += block.text
    return text


# ---------------------------------------------------------------------------
# HTML Report Generator (4th "agent")
# ---------------------------------------------------------------------------
def extract_score(text: str) -> str:
    m = re.search(r"PUNTUACION[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    return f"{m.group(1)}/10" if m else "N/A"

def extract_section(text: str, title: str) -> str:
    pat = re.compile(
        rf"{title}[:\n]+(.+?)(?=\n\n(?:###\s*)?(?:FORTALEZAS|DEBILIDADES|RECOMENDACIONES|RESUMEN|PUNTUACION|##|$))",
        re.IGNORECASE | re.DOTALL,
    )
    m = pat.search(text)
    if m:
        return m.group(1).strip()
    # fallback: find bullet points under heading
    alt = re.search(
        rf"{title}[:\n]*((?:\s*[-*].+?)*)", re.IGNORECASE | re.DOTALL
    )
    return alt.group(1).strip() if alt else ""


def generate_html(
    repo_url: str,
    repo_data: dict,
    results: list[tuple[str, str]],
    verdict: str,
    output_path: str,
) -> str:
    scores = []
    all_text = ""
    for name, text in results:
        s = extract_score(text)
        scores.append((name, s))
        all_text += f"=== {name} ===\n\n{text}\n\n"

    global_score = extract_score(verdict)

    lines = []
    lines.append("<!DOCTYPE html>")
    lines.append('<html lang="es">')
    lines.append("<head>")
    lines.append('  <meta charset="UTF-8">')
    lines.append('  <meta name="viewport" content="width=device-width, initial-scale=1.0">')
    lines.append(f"  <title>Repo Analysis - {repo_data.get('full_name', 'repo')}</title>")
    lines.append("""  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #0f172a; color: #e2e8f0; padding: 2rem; }
    .container { max-width: 960px; margin: 0 auto; }
    h1 { font-size: 1.8rem; color: #f8fafc; margin-bottom: .25rem; }
    .subtitle { color: #94a3b8; margin-bottom: 2rem; font-size: .95rem; }
    .repo-card { background: #1e293b; border-radius: 12px; padding: 1.5rem;
                 margin-bottom: 2rem; border: 1px solid #334155; }
    .repo-card h2 { color: #38bdf8; font-size: 1.1rem; margin-bottom: .75rem; }
    .stats { display: flex; flex-wrap: wrap; gap: 1rem 2rem; font-size: .9rem; }
    .stats span { color: #cbd5e1; }
    .stats strong { color: #f1f5f9; }
    .score-global { display: inline-block; background: linear-gradient(135deg,#38bdf8,#818cf8);
                    color: #fff; padding: .4rem 1rem; border-radius: 999px;
                    font-weight: 700; font-size: 1.3rem; margin-bottom: .5rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px,1fr));
            gap: 1.2rem; margin-bottom: 2rem; }
    .card { background: #1e293b; border-radius: 12px; padding: 1.5rem;
            border: 1px solid #334155; position: relative; }
    .card h3 { font-size: 1rem; margin-bottom: .5rem; }
    .card .score { font-size: 2rem; font-weight: 800; margin-bottom: .25rem; }
    .card .agent-name { color: #94a3b8; font-size: .85rem; margin-bottom: .8rem; }
    .card details { margin-top: .5rem; }
    .card summary { cursor: pointer; color: #38bdf8; font-size: .85rem; font-weight: 600; }
    .card summary:hover { text-decoration: underline; }
    .card .body { font-size: .85rem; line-height: 1.6; color: #cbd5e1;
                  margin-top: .5rem; white-space: pre-wrap; max-height: 400px; overflow-y: auto; }
    .card .body::-webkit-scrollbar { width: 4px; }
    .card .body::-webkit-scrollbar-thumb { background: #475569; border-radius: 2px; }
    .score-0-3 { color: #ef4444; }
    .score-4-6 { color: #f59e0b; }
    .score-7-8 { color: #84cc16; }
    .score-9-10 { color: #22d3ee; }
    .verdict { background: linear-gradient(135deg,#1e293b,#1a1f35); border-radius: 12px;
               padding: 1.5rem; border: 1px solid #334155; margin-bottom: 2rem; }
    .verdict h2 { color: #a78bfa; font-size: 1.1rem; margin-bottom: 1rem; }
    .verdict .body { font-size: .9rem; line-height: 1.7; color: #cbd5e1; white-space: pre-wrap; }
    .verdict strong { color: #f1f5f9; }
    .footer { text-align: center; color: #475569; font-size: .8rem; padding: 2rem 0 0; }
  </style>""")
    lines.append("</head>")
    lines.append("<body>")
    lines.append('<div class="container">')

    repo_name = repo_data.get("full_name", repo_url.split("github.com/")[-1] if "github.com" in repo_url else repo_url)
    desc = repo_data.get("description", "")
    stars = repo_data.get("stargazers_count", "?")
    forks = repo_data.get("forks_count", "?")
    langs = ", ".join(repo_data.get("languages", {}).keys()) or "?"
    topics = ", ".join(repo_data.get("topics", [])) or "—"

    lines.append(f'  <h1>{repo_name}</h1>')
    lines.append(f'  <div class="subtitle">{desc}</div>')

    lines.append('  <div class="repo-card">')
    lines.append("    <h2>Repository Info</h2>")
    lines.append('    <div class="stats">')
    lines.append(f'      <span>⭐ Stars: <strong>{stars}</strong></span>')
    lines.append(f'      <span>🍴 Forks: <strong>{forks}</strong></span>')
    lines.append(f'      <span>🔤 Languages: <strong>{langs}</strong></span>')
    lines.append(f'      <span>🏷️ Topics: <strong>{topics}</strong></span>')
    lines.append("    </div>")
    lines.append("  </div>")

    lines.append("  <h2 style='margin-bottom:1rem;color:#e2e8f0;font-size:1.05rem;'>Agent Verdicts</h2>")
    lines.append('  <div class="grid">')

    color_map = {"Code Quality": "score-9-10", "Security": "score-4-6", "Architecture": "score-7-8"}

    for name, text in results:
        score = extract_score(text)
        score_val = float(score.split("/")[0]) if score != "N/A" else 0
        if score_val <= 3:
            cls = "score-0-3"
        elif score_val <= 6:
            cls = "score-4-6"
        elif score_val <= 8:
            cls = "score-7-8"
        else:
            cls = "score-9-10"
        short_name = name.replace(" Agent", "")

        lines.append('    <div class="card">')
        lines.append(f'      <div class="score {cls}">{score}</div>')
        lines.append(f'      <div class="agent-name">{name}</div>')
        lines.append("      <details>")
        lines.append("        <summary>View analysis</summary>")
        escaped = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                         .replace('"', "&quot;"))
        lines.append(f'        <div class="body">{escaped}</div>')
        lines.append("      </details>")
        lines.append("    </div>")

    lines.append("  </div>")

    lines.append('  <div class="verdict">')
    lines.append(f'    <div class="score-global">{global_score}</div>')
    lines.append("    <h2>Final Verdict</h2>")
    escaped_v = (verdict.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                       .replace('"', "&quot;"))
    lines.append(f'    <div class="body">{escaped_v}</div>')
    lines.append("  </div>")

    lines.append(f'  <div class="footer">Generated by Repo Analyzer — all models run 100% local via LM Studio</div>')
    lines.append("</div>")
    lines.append("</body>")
    lines.append("</html>")

    html = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_header(text: str, char: str = "=", width: int = 74):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


async def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} <url_repo>")
        sys.exit(1)

    repo_url = sys.argv[1]

    print_header("REPO ANALYZER via claude-agent-sdk + LM Studio Transport")
    print(f"  Modelo: {MODEL}")
    print(f"  Repo:   {repo_url}")

    # Fetch repo info
    print_header("FETCHING REPO INFO")
    readme, data, err = fetch_repo_info(repo_url)
    if err:
        print(f"  [WARN] {err}")
        ctx = f"URL: {repo_url}"
    else:
        print(f"  {data.get('full_name', 'N/A')}")
        print(f"  Stars: {data.get('stargazers_count', 'N/A')} | Forks: {data.get('forks_count', 'N/A')}")
        print(f"  Lenguajes: {', '.join(data.get('languages', {}).keys()) or 'N/A'}")
        ctx = build_context(repo_url, readme, data)

    # Run 3 agents sequentially via SDK query() + custom Transport
    print_header("EJECUTANDO 3 AGENTES con SDK query() + LMStudioTransport")
    results = []
    start = time.time()

    for i, agent in enumerate(AGENTS, 1):
        name, text = await run_agent(agent, ctx, i, len(AGENTS))
        results.append((name, text))
        elapsed = time.time() - start
        print_header(f"RESULTADO: {name}  ({elapsed:.0f}s)")
        print(text)
        await asyncio.sleep(2)

    # Final synthesis
    print_header("SINTETIZANDO VEREDICTO FINAL")
    await asyncio.sleep(1)
    verdict = await synthesize(repo_url, results)

    print_header("VEREDICTO FINAL")
    print(verdict)

    # 4th agent: HTML report generator
    print_header("AGENTE 4: GENERANDO REPORTE HTML")
    output_path = f"repo_analysis_{re.sub(r'[^a-zA-Z0-9]', '_', repo_url.split('/')[-1])}.html"
    path = generate_html(repo_url, data, results, verdict, output_path)
    print(f"  Reporte generado: {path}")

    total = time.time() - start
    print(f"\n{'=' * 74}")
    print(f"  Analisis completado en {total:.0f} segundos")
    print(f"  Reporte HTML: {path}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    asyncio.run(main())
