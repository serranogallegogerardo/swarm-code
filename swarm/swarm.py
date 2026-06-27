"""
Repo Analyzer - 3 agents analysis usando Claude Agent SDK con proxy a LM Studio
"""

import os
import sys
import re
import time
import requests
import anyio
from dataclasses import dataclass
from typing import Optional

# Importaciones del SDK de Claude
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

# === CONFIGURACIÓN DEL PROXY ===
# Engañamos al Claude Agent SDK para que apunte a LM Studio en lugar de a Anthropic
os.environ["ANTHROPIC_BASE_URL"] = "http://localhost:1234"
os.environ["ANTHROPIC_API_KEY"] = "lm-studio-dummy-key"  # LM Studio no valida la key, pero el SDK exige que exista
os.environ["ANTHROPIC_MODEL"] = "google/gemma-4-26b-a4b-qat" # Modelos a usar

# === AGENTES ===
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
    """Obtiene README y metadata desde la API de GitHub."""
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

def build_context(repo_url: str, readme: str, repo_data: dict) -> str:
    """Construye un string de contexto conciso."""
    lines = [
        f"URL: {repo_url}",
        f"Descripcion: {repo_data.get('description', 'N/A')}",
        f"Lenguaje: {repo_data.get('language', 'N/A')}",
        f"Stars: {repo_data.get('stargazers_count', 'N/A')} | Forks: {repo_data.get('forks_count', 'N/A')}",
        "",
        "README:",
        readme[:1500],
    ]
    return "\n".join(lines)

def extract_score(text: str) -> str:
    match = re.search(r'PUNTUACION[:\s]*(\d+)\s*(?:/10)?', text, re.IGNORECASE)
    if match:
        return f"Puntuacion: {match.group(1)}/10"
    return "Puntuacion: N/A"

def extract_summary(text: str) -> str:
    match = re.search(r'(?:RESUMEN|CONCLUSION)[:\n]+(.{1,500})', text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()[:300]
    return text[-300:] if len(text) > 300 else text

async def ask_lm_studio_via_sdk(system_prompt: str, user_prompt: str) -> str:
    """
    Usa claude_agent_sdk.query para enviar el prompt.
    Max turns = 1 para evitar que el agente intente usar herramientas (Bash, Read, etc.)
    y se comporte como un simple generador de texto.
    """
    options = ClaudeAgentOptions(
        system_prompt=system_prompt,
        max_turns=1,  # Evita bucles de herramientas
        allowed_tools=[] # No le damos herramientas, solo texto
    )

    response_text = []
    
    # Ejecutamos la query asíncrona
    async for message in query(prompt=user_prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    response_text.append(block.text)
        elif hasattr(message, 'result') and message.result:
            # A veces el SDK devuelve el resultado final aquí
            response_text.append(message.result)

    return "".join(response_text).strip()

async def run_agent(agent: Agent, context: str, agent_num: int, total: int) -> tuple[str, str]:
    """Ejecuta un agente individual."""
    print(f"\n  --- Agente {agent_num}/{total}: {agent.name} ---")
    
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
    
    try:
        result = await ask_lm_studio_via_sdk(system_prompt, user_prompt)
        if not result:
            result = "[ERROR] El modelo no devolvió texto."
    except Exception as e:
        result = f"[ERROR SDK] {e}"
        
    return agent.name, result

async def synthesize_verdict(repo_url: str, results: list[tuple[str, str]]) -> str:
    """Sintetiza los resultados de los 3 agentes."""
    system_prompt = (
        "Eres un lead developer dando el veredicto final sobre un repositorio. "
        "Sintetiza los analisis de 3 expertos en un veredicto unico. "
        "Responde en espanol. Se conciso."
    )

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
    return await ask_lm_studio_via_sdk(system_prompt, user_prompt)

def print_header(text: str, char: str = "=", width: int = 74):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")

async def main_async():
    if len(sys.argv) < 2:
        print(f"Uso: python {sys.argv[0]} <url_del_repositorio>")
        print(f"Ej:  python {sys.argv[0]} https://github.com/user/repo")
        sys.exit(1)

    repo_url = sys.argv[1]

    print_header("REPO ANALYZER - 3 AGENTES (VÍA CLAUDE AGENT SDK)")
    print(f"  Proxy: LM Studio (http://localhost:1234)")
    print(f"  Repo:  {repo_url}")

    # Step 1: Fetch repo info
    print_header("FETCHING REPO INFO")
    readme, repo_data, error = fetch_repo_info(repo_url)

    if error:
        print(f"  [WARN] {error}")
        context = f"URL del repositorio: {repo_url}"
    else:
        print(f"  {repo_data.get('full_name', 'N/A')}")
        context = build_context(repo_url, readme, repo_data)

    # Step 2: Run 3 agents sequentially
    print_header("EJECUTANDO 3 AGENTES (ASYNC)")
    results = []
    start = time.time()

    for i, agent in enumerate(AGENTS, 1):
        name, text = await run_agent(agent, context, i, len(AGENTS))
        results.append((name, text))
        elapsed = time.time() - start

        print_header(f"RESULTADO: {name}  ({elapsed:.0f}s)")
        print(text)

    # Step 3: Final verdict
    print_header("SINTETIZANDO VEREDICTO FINAL")
    verdict = await synthesize_verdict(repo_url, results)

    print_header("VEREDICTO FINAL")
    print(verdict)

    elapsed_total = time.time() - start
    print(f"\n{'=' * 74}")
    print(f"  Analisis completado en {elapsed_total:.0f} segundos")
    print(f"{'=' * 74}")

if __name__ == "__main__":
    # Iniciamos el loop asíncrono de anyio
    anyio.run(main_async)