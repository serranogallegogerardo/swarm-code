"""
Repo Analyzer - 3 agents analysis of a GitHub repository
Uses LM Studio local model with multi-agent orchestration pattern
"""

import requests
import json
import sys
import re
import time
from dataclasses import dataclass
from typing import Optional

# === CONFIG ===
LM_STUDIO_URL = "http://localhost:1234/api/v1/chat"
MODEL = "google/gemma-4-26b-a4b-qat"
MAX_RETRIES = 2
RETRY_DELAY = 3

# === AGENTS ===
@dataclass
class Agent:
    name: str
    role: str
    focus: str

AGENTS = [
    Agent(
        name="Code Quality Agent",
        role="Eres un experto en calidad de codigo. Analiza estructura del proyecto, buenas practicas, testing, documentacion, mantenibilidad y legibilidad.",
        focus="calidad del codigo y buenas practicas"
    ),
    Agent(
        name="Security Agent",
        role="Eres un experto en seguridad. Analiza vulnerabilidades, manejo de secrets, dependencias peligrosas, y practicas de seguridad.",
        focus="seguridad y posibles vulnerabilidades"
    ),
    Agent(
        name="Architecture Agent",
        role="Eres un arquitecto de software senior. Analiza diseno arquitectonico, escalabilidad, patrones usados, organizacion del codigo y decisiones tecnicas.",
        focus="arquitectura y diseno del software"
    ),
]


def fetch_repo_info(url: str) -> tuple[str, dict, Optional[str]]:
    """Fetch README and metadata from GitHub API."""
    match = re.search(r'github\.com/([\w.-]+/[\w.-]+)', url)
    if not match:
        return "", {}, "Invalid GitHub URL"

    repo_path = match.group(1).rstrip("/")

    repo_resp = requests.get(f"https://api.github.com/repos/{repo_path}", timeout=10)
    repo_data = repo_resp.json() if repo_resp.status_code == 200 else {}

    lang_resp = requests.get(
        repo_data.get("languages_url", f"https://api.github.com/repos/{repo_path}/languages"),
        timeout=10
    )
    languages = lang_resp.json() if lang_resp.status_code == 200 else {}

    readme = ""
    for ext in ["", ".md", ".rst"]:
        readme_resp = requests.get(
            f"https://api.github.com/repos/{repo_path}/readme{ext}",
            headers={"Accept": "application/vnd.github.raw+json"},
            timeout=10
        )
        if readme_resp.status_code == 200:
            readme = readme_resp.text
            break

    repo_data["languages"] = languages
    return readme, repo_data, None


def query_lm_studio(system_prompt: str, user_prompt: str) -> str:
    """Send a prompt to the local LM Studio model with retry logic."""
    payload = {
        "model": MODEL,
        "system_prompt": system_prompt,
        "input": user_prompt,
    }

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            resp = requests.post(LM_STUDIO_URL, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()

            if "response" in data and data["response"]:
                return data["response"]
            if "choices" in data and data["choices"]:
                content = data["choices"][0].get("message", {}).get("content", "")
                if content:
                    return content
            return str(data)

        except requests.exceptions.ConnectionError:
            return f"[ERROR] No se pudo conectar a LM Studio en {LM_STUDIO_URL}"
        except requests.exceptions.HTTPError as e:
            last_error = f"[ERROR] {e}"
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    Reintentando en {wait}s... (intento {attempt+2}/{MAX_RETRIES+1})")
                time.sleep(wait)
            else:
                return last_error
        except Exception as e:
            return f"[ERROR] {e}"

    return last_error


def extract_score(text: str) -> str:
    """Extract the score line from an agent analysis."""
    match = re.search(r'PUNTUACION[:\s]*(\d+)\s*(?:/10)?', text, re.IGNORECASE)
    if match:
        return f"Puntuacion: {match.group(1)}/10"
    return "Puntuacion: N/A"


def extract_summary(text: str) -> str:
    """Extract summary/resumen from an agent analysis."""
    match = re.search(r'(?:RESUMEN|CONCLUSION)[:\n]+(.{1,500})', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()[:300]
    return text[-300:] if len(text) > 300 else text


def build_context(repo_url: str, readme: str, repo_data: dict) -> str:
    """Build a concise context string."""
    lines = [
        f"URL: {repo_url}",
        f"Descripcion: {repo_data.get('description', 'N/A')}",
        f"Lenguaje: {repo_data.get('language', 'N/A')}",
        f"Lenguajes: {', '.join(repo_data.get('languages', {}).keys()) or 'N/A'}",
        f"Stars: {repo_data.get('stargazers_count', 'N/A')} | Forks: {repo_data.get('forks_count', 'N/A')}",
        f"Topics: {', '.join(repo_data.get('topics', [])) or 'N/A'}",
        f"License: {repo_data.get('license', {}).get('spdx_id', 'N/A') if repo_data.get('license') else 'N/A'}",
        "",
        "README:",
        readme[:1500],
    ]
    return "\n".join(lines)


def run_agent(agent: Agent, context: str, agent_num: int, total: int) -> tuple[str, str]:
    """Run a single agent analysis."""
    print(f"\n  --- Agente {agent_num}/{total}: {agent.name} ---")
    time.sleep(2)  # Throttle to avoid server overload

    system_prompt = (
        f"{agent.role}\n\n"
        "Da tu veredicto en espanol con este formato EXACTO:\n"
        "PUNTUACION: X/10\n"
        "FORTALEZAS:\n- ...\n"
        "DEBILIDADES:\n- ...\n"
        "RECOMENDACIONES:\n- ...\n"
        "RESUMEN:\n...\n"
    )
    user_prompt = (
        f"Analiza este repositorio de GitHub:\n\n"
        f"{context}\n\n"
        f"Enfocate especialmente en: {agent.focus}."
    )
    result = query_lm_studio(system_prompt, user_prompt)
    return agent.name, result


def synthesize_verdict(repo_url: str, results: list[tuple[str, str]]) -> str:
    """Synthesize all agent analyses into a final verdict (condensed input)."""
    system_prompt = (
        "Eres un lead developer dando el veredicto final sobre un repositorio. "
        "Sintetiza los analisis de 3 expertos en un veredicto unico. "
        "Responde en espanol. Se conciso."
    )

    # Only pass condensed info to avoid hitting context limits
    condensed = []
    for name, text in results:
        score = extract_score(text)
        summary = extract_summary(text)
        condensed.append(f"=== {name} ===\n{score}\nResumen: {summary}")

    user_prompt = (
        f"Repo: {repo_url}\n\n"
        f"Analisis de los 3 agentes:\n\n{chr(10).join(condensed)}\n\n"
        "Genera VEREDICTO FINAL con:\n"
        "PUNTUACION GLOBAL: X/10\n"
        "CONCLUSION:\n"
        "RECOMENDACIONES PRIORITARIAS (top 3):\n"
        "LO RECOMENDARIAS? (Si/No/Con reservas)"
    )
    return query_lm_studio(system_prompt, user_prompt)


def print_header(text: str, char: str = "=", width: int = 74):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def main():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} <url_del_repositorio>")
        print(f"Ej:  python {sys.argv[0]} https://github.com/user/repo")
        sys.exit(1)

    repo_url = sys.argv[1]

    print_header("REPO ANALYZER - 3 AGENTES LOCALES")
    print(f"  Modelo: {MODEL}")
    print(f"  Repo:   {repo_url}")

    # Step 1: Fetch repo info
    print_header("FETCHING REPO INFO")
    readme, repo_data, error = fetch_repo_info(repo_url)

    if error:
        print(f"  [WARN] {error}")
        context = f"URL del repositorio: {repo_url}"
    else:
        print(f"  {repo_data.get('full_name', 'N/A')}")
        print(f"  Stars: {repo_data.get('stargazers_count', 'N/A')} | Forks: {repo_data.get('forks_count', 'N/A')}")
        print(f"  Lenguajes: {', '.join(repo_data.get('languages', {}).keys()) or 'N/A'}")
        print(f"  README: {len(readme)} chars")
        context = build_context(repo_url, readme, repo_data)

    # Step 2: Run 3 agents sequentially
    print_header("EJECUTANDO 3 AGENTES")
    results = []
    start = time.time()

    for i, agent in enumerate(AGENTS, 1):
        name, text = run_agent(agent, context, i, len(AGENTS))
        results.append((name, text))
        elapsed = time.time() - start

        print_header(f"RESULTADO: {name}  ({elapsed:.0f}s)")
        print(text)

    # Step 3: Final verdict
    print_header("SINTETIZANDO VEREDICTO FINAL")
    time.sleep(2)
    verdict = synthesize_verdict(repo_url, results)

    print_header("VEREDICTO FINAL")
    print(verdict)

    elapsed_total = time.time() - start
    print(f"\n{'=' * 74}")
    print(f"  Analisis completado en {elapsed_total:.0f} segundos")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
