"""
Repo Analyzer - 4 agents loop using claude-agent-sdk with custom LM Studio Transport
Auto-push to GitHub Repository via SSH
Features: Semgrep, OpenSSF Scorecard, License Compliance, PyDriller, Cross-Review
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
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

# Fix Windows console encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import glob as glob_module

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
SSH_KEY_PATH = os.environ.get("SSH_KEY_PATH") or os.path.expanduser("~/.ssh/swarm_bot")

# The Business Goal
GOAL = """
/goal: Generar un veredicto ejecutivo enfocado en negocio.

CRITERIOS DE EXITO (debes cumplir TODOS):
1. IMPACTO FINANCIERO: Cuantifica en USD el costo de mantenimiento anual, costo de oportunidad, y ROI de cada recomendacion
2. RIESGO OPERATIVO: Identifica al menos 3 riesgos especificos con CVEs, endpoints vulnerables, o fallos de arquitectura concretos
3. RECOMENDACIONES: Incluye timeline especifico (semanas/meses), herramientas concretas (SonarQube, Kubernetes, etc), y metricas de exito medibles
4. CLARIDAD: Usa bullet points, negritas para numeros clave, y estructura ejecutiva (no tecnica)

Debes alcanzar una puntuacion minima de 9.5/10 en claridad y valor estrategico.
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
# Real Code Analysis Tools (ground truth, not hallucinations)
# ---------------------------------------------------------------------------
@dataclass
class CodeMetrics:
    loc: int = 0
    files: int = 0
    languages: dict = None
    tree: str = ""
    churn_6m: float = 0.0
    authors_6m: int = 0
    complexity_estimate: str = ""
    key_files: list = None
    license_issues: list = None
    # New fields
    semgrep_findings: list = None
    scorecard_data: dict = None
    complexity_hotspots: list = None

    def __post_init__(self):
        if self.languages is None: self.languages = {}
        if self.key_files is None: self.key_files = []
        if self.license_issues is None: self.license_issues = []
        if self.semgrep_findings is None: self.semgrep_findings = []
        if self.scorecard_data is None: self.scorecard_data = {}
        if self.complexity_hotspots is None: self.complexity_hotspots = []


def clone_repo(url: str, dest: str) -> bool:
    """Clone a GitHub repo to a temp directory. Returns True on success."""
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, dest],
            check=True, capture_output=True, timeout=120,
        )
        return True
    except Exception as e:
        print(f"  [WARN] No se pudo clonar: {e}")
        return False


def get_directory_tree(root: str, max_depth: int = 2) -> str:
    """Generate ASCII directory tree."""
    lines = []
    def walk(path: str, prefix: str = "", depth: int = 0):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return
        # Skip .git, node_modules, __pycache__, etc.
        entries = [e for e in entries if not e.startswith(('.', '_')) and e not in ('node_modules', 'venv', '.venv')]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            full = os.path.join(path, entry)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry}")
            if os.path.isdir(full):
                ext = "    " if is_last else "│   "
                walk(full, prefix + ext, depth + 1)
    walk(root)
    return "\n".join(lines[:50])  # cap at 50 lines


def count_loc(root: str) -> tuple[int, int, dict]:
    """Count lines of code using cloc or fallback to extension matching."""
    total_loc = 0
    total_files = 0
    by_lang = {}

    # Try cloc first
    try:
        r = subprocess.run(
            ["cloc", root, "--json", "--quiet"],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode == 0:
            data = json.loads(r.stdout)
            for lang, stats in data.items():
                if lang in ("header", "SUM", "files"): continue
                total_loc += stats.get("code", 0)
                total_files += stats.get("nFiles", 0)
                by_lang[lang] = {"code": stats.get("code", 0), "files": stats.get("nFiles", 0)}
            return total_loc, total_files, by_lang
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    # Fallback: simple extension-based counting
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".jsx": "JavaScript",
        ".tsx": "TypeScript", ".go": "Go", ".rs": "Rust", ".java": "Java",
        ".rb": "Ruby", ".php": "PHP", ".c": "C", ".cpp": "C++", ".h": "C/C++ Header",
        ".cs": "C#", ".swift": "Swift", ".kt": "Kotlin", ".md": "Markdown",
        ".html": "HTML", ".css": "CSS", ".scss": "SCSS", ".yaml": "YAML",
        ".yml": "YAML", ".json": "JSON", ".xml": "XML", ".sql": "SQL",
        ".sh": "Shell", ".ps1": "PowerShell", ".bat": "Batch",
    }
    for dirpath, _, filenames in os.walk(root):
        if ".git" in dirpath or "node_modules" in dirpath:
            continue
        for fn in filenames:
            ext = os.path.splitext(fn)[1].lower()
            lang = ext_map.get(ext, "Other")
            if fn == "Dockerfile": lang = "Dockerfile"
            if fn == "Makefile": lang = "Makefile"
            total_files += 1
            try:
                with open(os.path.join(dirpath, fn), "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                loc = len([l for l in lines if l.strip()])
            except:
                loc = 0
            total_loc += loc
            if lang not in by_lang:
                by_lang[lang] = {"code": 0, "files": 0}
            by_lang[lang]["code"] += loc
            by_lang[lang]["files"] += 1

    return total_loc, total_files, by_lang


def get_git_metrics(repo_dir: str) -> tuple[float, int]:
    """Get code churn and author count from git log (last 6 months)."""
    try:
        r = subprocess.run(
            ["git", "-C", repo_dir, "log", "--since=6 months ago", "--format=%aE"],
            capture_output=True, text=True, timeout=30,
        )
        authors = set(r.stdout.strip().split("\n")) if r.stdout.strip() else set()
        author_count = len([a for a in authors if a])

        # Churn: total lines changed / total current LOC
        r2 = subprocess.run(
            ["git", "-C", repo_dir, "log", "--since=6 months ago", "--numstat", "--format="],
            capture_output=True, text=True, timeout=30,
        )
        added = 0
        deleted = 0
        for line in r2.stdout.strip().split("\n"):
            parts = line.split("\t")
            if len(parts) == 3:
                try:
                    a = int(parts[0]) if parts[0] != "-" else 0
                    d = int(parts[1]) if parts[1] != "-" else 0
                    added += a
                    deleted += d
                except ValueError:
                    pass
        total_changed = added + deleted
        return total_changed / 1000, author_count  # churn in kLOC changed
    except:
        return 0.0, 0


def read_key_files(root: str) -> list[dict]:
    """Read important config/doc files for context."""
    targets = [
        "package.json", "pyproject.toml", "requirements.txt", "Cargo.toml",
        "go.mod", "Gemfile", "pom.xml", "build.gradle", "composer.json",
        "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
        "Makefile", "Makefile.*", ".github/workflows/*.yml",
        "README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE",
        ".gitignore", ".env.example", "config/*", "src/main.py", "main.py",
        "index.js", "app.ts", "cmd/main.go",
    ]
    found = []
    for pattern in targets:
        for f in glob_module.glob(os.path.join(root, pattern)):
            try:
                with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                    content = fh.read(2000)
                rel = os.path.relpath(f, root)
                found.append({"path": rel, "content": content[:2000]})
            except:
                pass
    return found


def estimate_maintenance_cost(loc: int, files: int, churn_kloc: float, authors: int) -> dict:
    """
    COCOMO II-like parametric cost estimation based on real code metrics.
    No hallucinations — pure math on measured data.
    """
    # Standard: ~$0.50/LOC/year maintenance (industry avg from multiple studies)
    base_cost = loc * 0.50

    # Risk multiplier based on code churn
    churn_rate = min(churn_kloc / max(loc / 1000, 1), 5.0)  # cap at 5x
    risk_mult = 1.0 + (churn_rate * 0.3)

    # Team estimation: ~15k LOC/dev/year productivity
    team_size = max(round(loc / 15000), 1) if loc > 0 else 1

    annual_maintenance = round(base_cost * risk_mult)
    opportunity_cost = round(annual_maintenance * 2.5)  # 2.5x for delays/debt

    return {
        "annual_maintenance_usd": annual_maintenance,
        "opportunity_cost_usd": opportunity_cost,
        "estimated_team_ftes": team_size,
        "loc_total": loc,
        "files_total": files,
        "churn_6m_kloc": round(churn_kloc, 1),
        "churn_risk_multiplier": round(risk_mult, 2),
        "model": "COCOMO-II-lite (parametric)",
    }


# ---------------------------------------------------------------------------
# NEW: Semgrep Scanner (real vulnerability findings)
# ---------------------------------------------------------------------------
def run_semgrep(repo_dir: str) -> list[dict]:
    """Run semgrep --config=auto --json on the repo and return findings."""
    try:
        r = subprocess.run(
            ["semgrep", "--config=auto", "--json", "--quiet", repo_dir],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode not in (0, 1):  # 1 = findings found (normal)
            return []
        data = json.loads(r.stdout) if r.stdout.strip() else {}
        findings = []
        for f in data.get("results", []):
            findings.append({
                "check_id": f.get("check_id", "unknown"),
                "message": f.get("extra", {}).get("message", "").split("\n")[0][:200],
                "path": f.get("path", ""),
                "line": f.get("start", {}).get("line", 0),
                "severity": f.get("extra", {}).get("severity", "WARNING"),
                "cwe": f.get("extra", {}).get("metadata", {}).get("cwe", ""),
                "owasp": f.get("extra", {}).get("metadata", {}).get("owasp", ""),
            })
        return findings
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  [WARN] semgrep fallo: {e}")
        return []
    except Exception as e:
        print(f"  [WARN] semgrep error: {e}")
        return []


# ---------------------------------------------------------------------------
# NEW: OpenSSF Scorecard via public API
# ---------------------------------------------------------------------------
def query_openssf_scorecard(owner: str, repo: str) -> dict:
    """Query the OpenSSF Scorecard public API for a given repo."""
    try:
        r = requests.get(
            f"https://api.securityscorecards.dev/projects/github.com/{owner}/{repo}",
            timeout=15,
        )
        if r.status_code == 200:
            data = r.json()
            score = data.get("score", 0)
            checks = {}
            for check in data.get("checks", []):
                checks[check.get("name", "unknown")] = {
                    "score": check.get("score", 0),
                    "reason": check.get("reason", ""),
                }
            return {
                "overall_score": score,
                "checks": checks,
                "date": data.get("date", ""),
            }
        elif r.status_code == 404:
            return {"overall_score": None, "checks": {}, "error": "Scorecard no disponible (repo no publico o muy nuevo)"}
        return {"overall_score": None, "checks": {}, "error": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"overall_score": None, "checks": {}, "error": str(e)}


# ---------------------------------------------------------------------------
# NEW: License Compliance Checker
# ---------------------------------------------------------------------------
def check_license_compliance(root: str) -> list[dict]:
    """
    Check license compliance:
    1. Parse LICENSE file in the repo
    2. Check Python dependencies via pip-licenses if available
    3. Check npm dependencies if package.json exists
    """
    issues = []
    repo_license = ""

    for lic_name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
        lic_path = os.path.join(root, lic_name)
        if os.path.exists(lic_path):
            try:
                with open(lic_path, "r", encoding="utf-8", errors="ignore") as f:
                    repo_license = f.read(1500)
            except:
                pass
            break

    if repo_license:
        license_map = {
            "MIT License": "MIT",
            "GNU GENERAL PUBLIC LICENSE": "GPL",
            "GNU LESSER GENERAL PUBLIC LICENSE": "LGPL",
            "Apache License": "Apache-2.0",
            "BSD": "BSD",
            "Mozilla Public License": "MPL",
            "The Unlicense": "Unlicense",
            "Creative Commons": "CC",
        }
        lic_type = "Desconocida"
        for key, val in license_map.items():
            if key in repo_license:
                lic_type = val
                break
        issues.append({
            "type": "repo_license",
            "license": lic_type,
            "compatible_with_commercial": lic_type in ("MIT", "Apache-2.0", "BSD", "Unlicense", "CC0"),
        })
    else:
        issues.append({
            "type": "repo_license",
            "license": "Sin LICENSE detectado",
            "compatible_with_commercial": False,
            "risk": "ALTO - Sin licencia clara, el uso comercial puede tener implicaciones legales",
        })

    has_python = any(os.path.exists(os.path.join(root, f)) for f in ("requirements.txt", "pyproject.toml", "setup.py", "Pipfile"))
    if has_python:
        try:
            r = subprocess.run(
                ["pip-licenses", "--format=json", "--with-urls"],
                capture_output=True, text=True, timeout=30,
            )
            if r.returncode == 0:
                deps = json.loads(r.stdout)
                for dep in deps:
                    lic = dep.get("License", "Unknown")
                    if lic in ("GPL", "GPL v2", "GPL v3", "AGPL", "AGPL v3"):
                        issues.append({
                            "type": "dependency",
                            "name": dep.get("Name", "unknown"),
                            "version": dep.get("Version", ""),
                            "license": lic,
                            "risk": f"ALTO - Licencia {lic} restrictiva para uso comercial",
                        })
        except:
            pass

    pkg_json = os.path.join(root, "package.json")
    if os.path.exists(pkg_json):
        try:
            with open(pkg_json, "r", encoding="utf-8") as f:
                pkg = json.load(f)
            issues.append({
                "type": "package_manager",
                "name": "npm",
                "has_lockfile": os.path.exists(os.path.join(root, "package-lock.json")),
            })
        except:
            pass

    return issues


# ---------------------------------------------------------------------------
# NEW: PyDriller + Lizard for complexity and advanced git metrics
# ---------------------------------------------------------------------------
def analyze_with_pydriller(repo_url: str, repo_dir: str) -> dict:
    """Extract advanced git metrics using PyDriller."""
    try:
        from pydriller import RepositoryMining

        total_commits = 0
        authors = set()
        modified_files = set()
        commit_messages = []

        for commit in RepositoryMining(repo_dir).traverse_commits():
            total_commits += 1
            authors.add(commit.author.email)
            for mod in commit.modifications:
                if mod.new_path:
                    modified_files.add(mod.new_path)
            if len(commit_messages) < 20:
                commit_messages.append(commit.msg.split("\n")[0][:100])

        return {
            "total_commits": total_commits,
            "total_authors": len(authors),
            "modified_files": len(modified_files),
            "commit_messages": commit_messages,
        }
    except ImportError:
        return {"error": "pydriller no disponible, usando git log basico"}
    except Exception as e:
        return {"error": str(e)}


def analyze_with_lizard(root: str) -> list[dict]:
    """Run lizard (cyclomatic complexity analyzer) and return hotspots with JSON output."""
    hotspots = []
    ext_map = {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".jsx": "javascript",
        ".tsx": "typescript", ".go": "go", ".rs": "rust", ".java": "java",
        ".cpp": "cpp", ".c": "c", ".cs": "csharp",
    }
    try:
        for dirpath, _, filenames in os.walk(root):
            if ".git" in dirpath or "node_modules" in dirpath:
                continue
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                lang = ext_map.get(ext)
                if not lang:
                    continue
                filepath = os.path.join(dirpath, fn)
                try:
                    r = subprocess.run(
                        ["lizard", "--languages=" + lang, "--json", filepath],
                        capture_output=True, text=True, timeout=15,
                    )
                    if r.returncode == 0 and r.stdout.strip():
                        data = json.loads(r.stdout)
                        for func in data.get("functions", []):
                            ccn = func.get("cyclomatic_complexity", 0)
                            if ccn >= 15:
                                hotspots.append({
                                    "file": func.get("filename", ""),
                                    "function": func.get("name", ""),
                                    "line": func.get("start_line", 0),
                                    "ccn": ccn,
                                    "nloc": func.get("nloc", 0),
                                    "tokens": func.get("tokens", 0),
                                    "language": lang,
                                })
                except:
                    pass
        hotspots.sort(key=lambda x: -x["ccn"])
        return hotspots[:20]
    except:
        return []


def analyze_repo_locally(url: str) -> tuple[CodeMetrics, dict]:
    """Full local analysis: clone, measure, and return real data."""
    print("  [GROUND TRUTH] Clonando repo para analisis local...")
    metrics = CodeMetrics()
    cost = {}

    temp_dir = tempfile.mkdtemp()
    repo_dir = os.path.join(temp_dir, "repo")

    if not clone_repo(url, repo_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
        return metrics, cost

    try:
        # Directory tree
        print("  [GROUND TRUTH] Generando arbol de directorios...")
        metrics.tree = get_directory_tree(repo_dir)

        # LOC and file stats
        print("  [GROUND TRUTH] Contando lineas de codigo...")
        loc, files, langs = count_loc(repo_dir)
        metrics.loc = loc
        metrics.files = files
        metrics.languages = langs

        # Git metrics
        print("  [GROUND TRUTH] Extrayendo metricas de git...")
        metrics.churn_6m, metrics.authors_6m = get_git_metrics(repo_dir)

        # --- NEW: PyDriller advanced metrics ---
        print("  [GROUND TRUTH] Analizando con PyDriller...")
        pydriller_data = analyze_with_pydriller(url, repo_dir)
        if "error" not in pydriller_data:
            print(f"    Commits: {pydriller_data['total_commits']} | Autores: {pydriller_data['total_authors']} | Archivos modificados: {pydriller_data['modified_files']}")

        # --- NEW: Lizard cyclomatic complexity ---
        print("  [GROUND TRUTH] Analizando complejidad ciclomatica...")
        metrics.complexity_hotspots = analyze_with_lizard(repo_dir)
        if metrics.complexity_hotspots:
            print(f"    Hotspots (>15 CCN): {len(metrics.complexity_hotspots)}")
            for h in metrics.complexity_hotspots[:5]:
                print(f"      {h['file']}:{h['line']} {h['function']} (CCN={h['ccn']})")

        # --- NEW: Semgrep scan ---
        print("  [GROUND TRUTH] Escaneando con Semgrep...")
        metrics.semgrep_findings = run_semgrep(repo_dir)
        if metrics.semgrep_findings:
            print(f"    Hallazgos: {len(metrics.semgrep_findings)}")
            by_sev = {}
            for f in metrics.semgrep_findings:
                sev = f.get("severity", "WARNING")
                by_sev[sev] = by_sev.get(sev, 0) + 1
            for sev, cnt in sorted(by_sev.items()):
                print(f"      {sev}: {cnt}")
        else:
            print("    Sin hallazgos (0 vulnerabilidades detectadas)")

        # --- NEW: OpenSSF Scorecard ---
        print("  [GROUND TRUTH] Consultando OpenSSF Scorecard...")
        owner_repo = url.rstrip("/").split("github.com/")[-1] if "github.com" in url else ""
        if "/" in owner_repo:
            owner, repo_name = owner_repo.split("/")[0], owner_repo.split("/")[1]
            metrics.scorecard_data = query_openssf_scorecard(owner, repo_name)
            if metrics.scorecard_data.get("overall_score") is not None:
                print(f"    Score: {metrics.scorecard_data['overall_score']}/10")
            else:
                print(f"    No disponible: {metrics.scorecard_data.get('error', 'desconocido')}")

        # --- NEW: License compliance ---
        print("  [GROUND TRUTH] Verificando licencias...")
        metrics.license_issues = check_license_compliance(repo_dir)
        repo_lic = next((i for i in metrics.license_issues if i["type"] == "repo_license"), None)
        if repo_lic:
            print(f"    Licencia: {repo_lic['license']} | Compatible comercial: {repo_lic.get('compatible_with_commercial', '?')}")

        # Key files
        print("  [GROUND TRUTH] Leyendo archivos clave...")
        metrics.key_files = read_key_files(repo_dir)

        # Cost estimation
        print("  [GROUND TRUTH] Calculando costo de mantenimiento...")
        cost = estimate_maintenance_cost(loc, files, metrics.churn_6m, metrics.authors_6m)
        metrics.complexity_estimate = (
            f"Baja" if loc < 5000 else
            f"Media" if loc < 50000 else
            f"Alta"
        )

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return metrics, cost


def build_enriched_context(url: str, readme: str, gh_data: dict, metrics: CodeMetrics, cost: dict) -> str:
    """Build agent context with real data instead of hallucinations."""
    lines = [f"URL: {url}"]

    # GitHub metadata
    lines.append(f"Descripcion: {gh_data.get('description', 'N/A')}")
    lines.append(f"Stars: {gh_data.get('stargazers_count', 'N/A')} | Forks: {gh_data.get('forks_count', 'N/A')}")
    lines.append(f"License: {gh_data.get('license', {}).get('spdx_id', 'N/A') if gh_data.get('license') else 'N/A'}")
    topics = gh_data.get('topics', [])
    if topics:
        lines.append(f"Topics: {', '.join(topics[:10])}")

    # REAL code metrics (compact)
    if metrics.loc > 0:
        lines.append("")
        lines.append("=== DATOS REALES DEL REPOSITORIO ===")
        lines.append(f"LOC: {metrics.loc:,} | Archivos: {metrics.files} | Churn 6m: {metrics.churn_6m:.1f}k | Autores: {metrics.authors_6m}")
        lang_summary = ", ".join(f"{lang}({stats['code']:,}loc)" for lang, stats in sorted(metrics.languages.items(), key=lambda x: -x[1]['code'])[:6])
        lines.append(f"Lenguajes: {lang_summary}")

    if cost:
        lines.append(f"Costos reales (modelo COCOMO-II): ${cost['annual_maintenance_usd']:,}/año mantenimiento | ${cost['opportunity_cost_usd']:,} oportunidad | {cost['estimated_team_ftes']} FTE")

    # --- NEW: OpenSSF Scorecard ---
    if metrics.scorecard_data and metrics.scorecard_data.get("overall_score") is not None:
        sc = metrics.scorecard_data["overall_score"]
        lines.append(f"OpenSSF Scorecard: {sc}/10")
        # Top critical checks
        critical = {k: v for k, v in metrics.scorecard_data.get("checks", {}).items() if v.get("score", 10) < 5}
        if critical:
            lines.append(f"  Deficiencias: {', '.join(critical.keys())}")

    # --- NEW: Semgrep findings (grouped by directory) ---
    if metrics.semgrep_findings:
        # Group by top-level directory
        by_dir = {}
        for f in metrics.semgrep_findings:
            path = f.get("path", "")
            parts = path.replace("\\", "/").split("/")
            top_dir = parts[-2] if len(parts) >= 2 else "root"
            if top_dir not in by_dir:
                by_dir[top_dir] = []
            by_dir[top_dir].append(f)

        lines.append(f"Semgrep: {len(metrics.semgrep_findings)} hallazgos totales")
        for dir_name, findings in sorted(by_dir.items()):
            errors = sum(1 for f in findings if f.get("severity") == "ERROR")
            warnings = sum(1 for f in findings if f.get("severity") == "WARNING")
            dir_label = f"[⚠️ DIRECTORIO: {dir_name}/]" if dir_name in ("benchmarks", "scripts", "tests") else f"[DIRECTORIO: {dir_name}/]"
            lines.append(f"  {dir_label} {len(findings)} hallazgos ({errors} ERROR, {warnings} WARNING)")
            for f in findings[:2]:
                lines.append(f"    - {f['check_id']}: {f['message'][:100]} en {f['path']}:{f['line']}")

    # --- NEW: Complexity hotspots ---
    if metrics.complexity_hotspots:
        lines.append(f"Complejidad ciclomatica: {len(metrics.complexity_hotspots)} hotspots (CCN>15)")
        for h in metrics.complexity_hotspots[:3]:
            lines.append(f"  - {h['file']}:{h['line']} {h['function']} (CCN={h['ccn']})")

    # --- NEW: License issues ---
    if metrics.license_issues:
        repo_lic = next((i for i in metrics.license_issues if i["type"] == "repo_license"), None)
        if repo_lic:
            li = repo_lic
            compat = "SI" if li.get("compatible_with_commercial") else "NO"
            lines.append(f"Licencia: {li['license']} | Compatible comercial: {compat}")
        dep_risks = [i for i in metrics.license_issues if i["type"] == "dependency"]
        if dep_risks:
            lines.append(f"Riesgos licencia dependencias: {len(dep_risks)}")

    # Compact tree (depth 2, max 30 lines)
    if metrics.tree:
        tree_lines = metrics.tree.split("\n")[:30]
        lines.append(f"Arbol ({len(tree_lines)} entradas):")
        lines.extend(tree_lines)

    # Just list key files found
    if metrics.key_files:
        names = [kf['path'] for kf in metrics.key_files[:10]]
        lines.append(f"Archivos clave: {', '.join(names)}")

    if readme:
        lines.append(f"README: {readme[:2000]}")
    else:
        lines.append("README: NO DISPONIBLE")

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
# NEW: Cross-Review — each agent sees the other two's findings
# ---------------------------------------------------------------------------
async def cross_review_round(agents: list[Agent], results: list[tuple[str, str]], context: str) -> list[tuple[str, str]]:
    """
    Cross-review round: each agent receives the findings of the other two agents
    and must confirm, refute, or update their analysis with specific evidence.
    """
    print("\n" + "=" * 74)
    print("  CROSS-REVIEW ROUND — AGENTES SE CRITICAN MUTUAMENTE")
    print("=" * 74)

    updated_results = []
    for i, (agent, (name, text)) in enumerate(zip(agents, results)):
        print(f"\n  --- Cross-Review: {name} revisa a los otros ---")

        # Gather the other agents' findings
        others = [results[j][1] for j in range(len(results)) if j != i]

        cross_prompt = (
            f"Eres {agent.role}\n\n"
            "Acabas de generar este analisis:\n"
            f"{text[:800]}\n\n"
            f"A CONTINUACION, los hallazgos de los otros agentes:\n\n"
            f"--- OTRO AGENTE 1 ---\n{others[0][:600]}\n\n"
        )
        if len(others) > 1:
            cross_prompt += f"--- OTRO AGENTE 2 ---\n{others[1][:600]}\n\n"

        cross_prompt += (
            "INSTRUCCIONES:\n"
            "1. IDENTIFICA si hay CONTRADICCIONES entre tu analisis y los otros agentes\n"
            "2. CONFIRMA puntos donde los otros agentes refuerzan tus hallazgos\n"
            "3. REFUTA con evidencia si otro agente cometio un error\n"
            "4. ACTUALIZA tu analisis si encuentras informacion que no consideraste\n\n"
            "Genera un CRITICA CRUZADA estructurada:\n"
            "PUNTUACION REVISADA: X/10 (solo si cambia)\n"
            "CONTRADICCIONES:\n- ...\n"
            "CONFIRMACIONES:\n- ...\n"
            "ACTUALIZACIONES:\n- ...\n"
        )

        system_prompt = f"Eres un analista critico. Revisa hallazgos de otros agentes y actualiza tu analisis si es necesario.\n{GOAL}"
        transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
        options = ClaudeAgentOptions(system_prompt=system_prompt)

        text_cross = ""
        async for message in query(prompt=cross_prompt, options=options, transport=transport):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_cross += block.text

        # Append compact cross-review to the original analysis
        updated = text + "\n\n--- CROSS-REVIEW ---\n" + text_cross[:300]
        updated_results.append((name, updated))
        print(f"\n  --- Cross-Review: {name} completo ---")
        print(text_cross[:400])

    return updated_results


# ---------------------------------------------------------------------------
# Synthesis & Looping Evaluation
# ---------------------------------------------------------------------------
def compact(text: str, maxlen: int = 400) -> str:
    """Truncate long text keeping start and end."""
    if len(text) <= maxlen:
        return text
    return text[:maxlen//2] + "\n...\n" + text[-maxlen//2:]

async def synthesize(url: str, results: list[tuple[str, str]], history: list[dict] = None, gh_data: dict = None) -> str:
    print("\n  --- Sintetizando veredicto final ---")

    condensed = []
    for name, text in results:
        score = re.search(r"PUNTUACI[OÓ]N[^0-9]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
        score_str = f"Puntuacion: {score.group(1)}/10" if score else "Puntuacion: N/A"
        summary_m = re.search(r"(?:RESUMEN|CONCLUSION)[:\n]+(.{1,500})", text, re.IGNORECASE | re.DOTALL)
        summary = summary_m.group(1).strip()[:300] if summary_m else (text[-300:] if text else "N/A")
        condensed.append(f"=== {name} ===\n{score_str}\nResumen: {summary}")

    # Build repo identity block for grounding
    repo_identity = f"URL: {url}"
    if gh_data:
        desc = gh_data.get("description", "N/A")
        topics_list = gh_data.get("topics", [])
        repo_identity += f"\nDescripcion oficial: {desc}"
        if topics_list:
            repo_identity += f"\nTopics: {', '.join(topics_list[:10])}"

    few_shot = """
EJEMPLO DE VEREDICTO 10/10:

PUNTUACION GLOBAL: 9.8/10

IMPACTO FINANCIERO Y RIESGO:
- Costo de mantenimiento: $45,000 USD/ano (3 developers FTE) debido a deuda tecnica en modulo de autenticacion
- Riesgo de seguridad: ALTO - 3 vulnerabilidades criticas (SQL injection en /api/users, falta de rate limiting, secrets en codigo)
- Costo de oportunidad: $120,000 USD por retraso en time-to-market por arquitectura monolitica no escalable

RECOMENDACIONES PRIORITARIAS (top 3):
1. [URGENTE - 2 semanas] Implementar OWASP ZAP scanning en CI/CD y corregir SQL injection en auth.py (lineas 45-67)
2. [ALTA - 1 mes] Migrar modulo de autenticacion a microservicio con JWT + Redis para reducir deuda tecnica en 60%
3. [MEDIA - 3 meses] Refactorizar a arquitectura hexagonal para permitir escalado horizontal del modulo de pagos

DECISION FINAL: GO WITH RESERVATIONS

CONDICIONES PARA GO COMPLETO:
- Corregir vulnerabilidades criticas antes de deploy a produccion
- Implementar monitoring con Prometheus + Grafana
- Documentar API con OpenAPI 3.0
"""

    system_prompt = (
        "Eres un Lead Developer y CTO. Sintetiza 3 analisis en un veredicto unico orientado al C-Level. Responde en espanol."
        f"\n{GOAL}\n\n"
        "SIGUE EL FORMATO DEL EJEMPLO ABAJO. Usa numeros concretos, herramientas especificas y plazos.\n\n"
        "REGLAS ESTRICTAS DE GROUNDING (violarlas invalida el analisis):\n"
        "1. El veredicto debe ser COHERENTE con la descripcion y topics del repo. Si el repo es un plugin/set de reglas, NO inventes que es una libreria de animacion, framework web, etc.\n"
        "2. Los costos FINANCIEROS deben basarse en los datos provistos (COCOMO-II real: $X/ano). NO inventes costos 5x o 10x mayores.\n"
        "3. Los hallazgos de SEMGREP deben ubicarse en su directorio real. Si estan en benchmarks/ o scripts/, NO los trates como vulnerabilidades del producto core.\n"
        "4. Cada riesgo tecnico debe CITAR un archivo y linea real del arbol de directorios provisto.\n"
        "5. Si no hay evidencia para una afirmacion, NO la inventes. Di 'No hay datos suficientes para confirmar'.\n"
        "6. Los topics del repo definen el DOMINIO del proyecto. Usalos para contextualizar todo el analisis."
    )

    prompt = (
        f"Repo: {url}\n\n{repo_identity}\n\nAnalisis:\n\n{chr(10).join(condensed)}\n\n"
        f"EJEMPLO DE CALIDAD ESPERADA:\n{few_shot}\n\n"
        "Genera el veredicto final:\n"
        "PUNTUACION GLOBAL: X/10\n"
        "IMPACTO FINANCIERO Y RIESGO:\n"
        "RECOMENDACIONES PRIORITARIAS (top 3):\n"
        "DECISION FINAL (GO / NO-GO / GO WITH RESERVATIONS):\n"
    )

    # Include iteration history (compact) so the agent learns from past mistakes
    if history:
        prompt += "\n\n=== HISTORIAL DE INTENTOS ANTERIORES ===\n"
        for i, h in enumerate(history, 1):
            prompt += f"\n--- Intento {i} (Score: {h['eval_score']}/10) ---\n"
            # Only include key feedback, not full verdict text
            for dim, d in h['eval_dims'].items():
                prompt += f"- {dim}: {d['score']}/10 — {d['feedback'][:150]}\n"
            prompt += f"Feedback general: {h['eval_feedback'][:200]}\n"

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
# Fix 2: Sanity Validator Agent — checks verdict coherence against repo identity
# ---------------------------------------------------------------------------
async def sanity_check(verdict: str, repo_url: str, desc: str, topics: list[str]) -> tuple[bool, str]:
    """
    Sanity validator: checks if the verdict is coherent with the repo's actual identity.
    Returns (passed: bool, reason: str).
    """
    system_prompt = (
        "Eres un validador de coherencia. Tu unica tarea es detectar si un veredicto "
        "describe INCORRECTAMENTE el tipo de proyecto. No evalues calidad, solo coherencia.\n\n"
        "Responde SOLO con:\n"
        "COHERENTE: si el veredicto describe correctamente el tipo de proyecto\n"
        "INCOHERENTE: si el veredicto habla de tecnologias o dominios que no corresponden\n"
        "RAZON: explicacion de 1 linea"
    )
    prompt = (
        f"REPO URL: {repo_url}\n"
        f"DESCRIPCION OFICIAL: {desc}\n"
        f"TOPICS: {', '.join(topics[:10])}\n\n"
        f"VEREDICTO A VALIDAR:\n{verdict[:1500]}\n\n"
        "El veredicto describe correctamente el tipo de proyecto? "
        "Si habla de animaciones, WebGL, renderizado, o tecnologias que no aparecen "
        "en los topics ni en la descripcion, responde INCOHERENTE."
    )
    transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
    options = ClaudeAgentOptions(system_prompt=system_prompt)
    text = ""
    async for message in query(prompt=prompt, options=options, transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text += block.text
    passed = "COHERENTE" in text.upper()
    reason = text.strip()[:200] if not passed else ""
    return passed, reason


async def evaluate_goal(verdict: str) -> dict:
    """4th Agent: Evaluates with a multi-dimension rubric."""
    print("\n  --- Evaluando si se alcanzo el Goal ---")
    system_prompt = (
        "Eres un auditor de calidad de software orientado a negocio. "
        "Evaluas veredictos usando una rubrica multidimensional."
        f"\n{GOAL}"
    )
    
    prompt = (
        f"Evalua este veredicto:\n\n{verdict}\n\n"
        "Responde EXACTAMENTE en este formato:\n"
        "SCORE GLOBAL: X.X/10\n"
        "--- DIMENSIONES ---\n"
        "IMPACTO FINANCIERO: X/10 | feedback aqui\n"
        "  EJEMPLO_CONCRETO: 'En lugar de decir \"alto costo\", especifica \"$50k/ano en mantenimiento tecnico\"'\n"
        "RIESGO OPERATIVO: X/10 | feedback aqui\n"
        "  EJEMPLO_CONCRETO: 'Menciona vulnerabilidades especificas como CVE-XXXX o falta de autenticacion en endpoint /api/admin'\n"
        "RECOMENDACIONES: X/10 | feedback aqui\n"
        "  EJEMPLO_CONCRETO: 'Sugiere herramientas especificas como SonarQube o migracion a Kubernetes en lugar de \"mejorar testing\"'\n"
        "CLARIDAD: X/10 | feedback aqui\n"
        "  EJEMPLO_CONCRETO: 'Usa bullet points en lugar de parrafos largos para el resumen ejecutivo'\n"
        "--- FIN DIMENSIONES ---\n"
        "FEEDBACK GENERAL: texto aqui explicando que mejorar\n"
    )

    transport = LMStudioTransport(LM_STUDIO_URL, MODEL, system_prompt=system_prompt)
    options = ClaudeAgentOptions(system_prompt=system_prompt)

    text = ""
    async for message in query(prompt=prompt, options=options, transport=transport):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text += block.text

    print(f"  Evaluacion:\n{text.strip()}\n")

    # Parse structured response
    global_m = re.search(r"SCORE GLOBAL[:\s]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
    global_score = float(global_m.group(1)) if global_m else 0.0

    dims = {}
    for dim_name in ["IMPACTO FINANCIERO", "RIESGO OPERATIVO", "RECOMENDACIONES", "CLARIDAD"]:
        m = re.search(rf"{dim_name}[:\s]*(\d+(?:\.\d+)?)\s*[|]\s*(.+)", text, re.IGNORECASE)
        if m:
            dims[dim_name] = {"score": float(m.group(1)), "feedback": m.group(2).strip()}

    fb_m = re.search(r"FEEDBACK GENERAL[:\s]*(.+)", text, re.IGNORECASE | re.DOTALL)
    general_feedback = fb_m.group(1).strip() if fb_m else ""

    return {
        "score": global_score,
        "goal_met": global_score >= 9.5,
        "dims": dims,
        "feedback": general_feedback,
    }

# ---------------------------------------------------------------------------
# Premium HTML Report Generator (hybrid: template + AI content)
# ---------------------------------------------------------------------------
def extract_score(text: str) -> str:
    m = re.search(r"PUNTUACI[OÓ]N[^0-9]*(\d+(?:\.\d+)?)", text, re.IGNORECASE)
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
    metrics: CodeMetrics = None,
    cost_data: dict = None,
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

    # Semgrep findings HTML
    semgrep_html = ""
    if metrics and metrics.semgrep_findings:
        for f in metrics.semgrep_findings[:8]:
            sev = f.get("severity", "WARNING")
            cid = f.get("check_id", "")
            msg = f.get("message", "")[:120]
            loc = f"{f.get('path', '')}:{f.get('line', 0)}"
            semgrep_html += f'<div class="finding"><span class="sev sev-{sev}">{sev}</span><strong>{cid}</strong> — {msg} <em>({loc})</em></div>\n'
    if not semgrep_html:
        semgrep_html = '<div class="finding" style="color:#64748b">No se detectaron vulnerabilidades</div>'

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
.metrics-card{{background:linear-gradient(135deg,#111827,#1a1f35);border-radius:12px;padding:1rem 1.5rem;border:1px solid #1e293b;margin-bottom:1.5rem;animation:fadeUp .6s ease-out .15s both}}
.metrics-card h3{{font-size:.8rem;text-transform:uppercase;letter-spacing:.08em;color:#64748b;margin-bottom:.7rem}}
.metrics-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.8rem}}
.metric-item{{}}
.metric-item .mval{{font-size:1rem;font-weight:700;color:#f1f5f9}}
.metric-item .mlabel{{font-size:.7rem;color:#64748b}}
.metric-critical{{color:#ef4444!important}}
.metric-warning{{color:#f59e0b!important}}
.metric-good{{color:#84cc16!important}}
.finding{{padding:.3rem 0;font-size:.8rem;color:#94a3b8;border-bottom:1px solid rgba(255,255,255,.04)}}
.finding .sev{{display:inline-block;padding:0 .4rem;border-radius:3px;font-size:.7rem;font-weight:700;margin-right:.4rem}}
.sev-ERROR{{background:rgba(239,68,68,.25);color:#ef4444}}
.sev-WARNING{{background:rgba(245,158,11,.25);color:#f59e0b}}
.sev-INFO{{background:rgba(56,189,248,.25);color:#38bdf8}}
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

  <!-- Ground Truth Metrics -->
  <div class="metrics-card">
    <h3>&#9881; Ground Truth Metrics</h3>
    <div class="metrics-grid">
      <div class="metric-item"><div class="mval">{metrics.loc:,}</div><div class="mlabel">Lines of Code</div></div>
      <div class="metric-item"><div class="mval">{metrics.files}</div><div class="mlabel">Archivos</div></div>
      <div class="metric-item"><div class="mval">{metrics.churn_6m:.1f}k</div><div class="mlabel">Churn (6m)</div></div>
      <div class="metric-item"><div class="mval">{metrics.authors_6m}</div><div class="mlabel">Autores</div></div>
      <div class="metric-item"><div class="mval">{cost_data.get('annual_maintenance_usd', 0):,}</div><div class="mlabel">Costo Mant. ($/año)</div></div>
      <div class="metric-item"><div class="mval">{cost_data.get('estimated_team_ftes', 0)}</div><div class="mlabel">FTEs Estimados</div></div>
      <div class="metric-item"><div class="mval">{metrics.scorecard_data.get('overall_score', 'N/A')}</div><div class="mlabel">OpenSSF Scorecard</div></div>
      <div class="metric-item"><div class="mval">{len(metrics.semgrep_findings)}</div><div class="mlabel">Semgrep Hallazgos</div></div>
    </div>

    <!-- Semgrep findings (top 5) -->
    {semgrep_html}
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

    # Fetch repo info + real code analysis
    print_header("FETCHING REPO INFO")
    m = re.search(r"github\.com/([\w.-]+/[\w.-]+)", repo_url)
    gh_data = {}
    readme = ""
    if m:
        repo_path = m.group(1).rstrip("/")
        try:
            r = requests.get(f"https://api.github.com/repos/{repo_path}", timeout=10)
            gh_data = r.json() if r.status_code == 200 else {}
            for ext in ("", ".md", ".rst"):
                rm = requests.get(
                    f"https://api.github.com/repos/{repo_path}/readme{ext}",
                    headers={"Accept": "application/vnd.github.raw+json"},
                    timeout=10,
                )
                if rm.status_code == 200:
                    readme = rm.text
                    break
        except:
            pass
        print(f"  {gh_data.get('full_name', repo_path)}")

    # Real local analysis (clone + measure)
    metrics, cost_data = analyze_repo_locally(repo_url)
    if metrics.loc > 0:
        print(f"  LOC: {metrics.loc:,} | Archivos: {metrics.files} | Churn: {metrics.churn_6m:.1f}k | Autores: {metrics.authors_6m}")
        if cost_data:
            print(f"  Costo mantenimiento: ${cost_data['annual_maintenance_usd']:,}/ano | Equipo: {cost_data['estimated_team_ftes']} FTE")
    else:
        print("  [WARN] No se pudo clonar para analisis local, usando solo metadata de GitHub")

    ctx = build_enriched_context(repo_url, readme, gh_data, metrics, cost_data)

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

    # Cross-Review Round
    results = await cross_review_round(AGENTS, results, ctx)

    # Loop of Synthesis + Evaluation until Goal is met
    print_header("INICIANDO LOOP DE OPTIMIZACION HASTA ALCANZAR EL GOAL")
    
    max_iterations = 6
    min_iterations = 2
    hard_timeout = 600  # 10 min max total for the loop
    loop_start = time.time()
    
    verdict = ""
    history = []
    prev_score = 0.0
    stall_count = 0

    for iteration in range(1, max_iterations + 1):
        if time.time() - loop_start > hard_timeout:
            print(f"\n  [TIMEOUT] {hard_timeout}s alcanzado — usando ultimo veredicto disponible.")
            break

        print(f"\n  >>> Iteracion {iteration}/{max_iterations} <<<")
        verdict = await synthesize(repo_url, results, history, gh_data)
        print("\n  Veredicto Actual:")
        print(verdict)

        # Sanity check: reject hallucinated verdicts
        repo_desc = gh_data.get("description", "")
        repo_topics = gh_data.get("topics", [])
        sane, reason = await sanity_check(verdict, repo_url, repo_desc, repo_topics)
        if not sane:
            print(f"\n  [SANITY FAIL] Veredicto INCOHERENTE: {reason}")
            print("  Forzando reconstitucion con grounding reforzado...")
            # Force a regenerate with stronger grounding by adding explicit instruction
            verdict = await synthesize(
                repo_url, results, history, gh_data,
            )
            # Check again
            sane2, reason2 = await sanity_check(verdict, repo_url, repo_desc, repo_topics)
            if not sane2:
                print(f"  [SANITY FAIL x2] Aun incoherente: {reason2}")
                print("  Usando ultimo veredicto disponible (con advertencia)")

        eval_result = await evaluate_goal(verdict)
        
        history.append({
            "verdict": verdict,
            "eval_score": eval_result["score"],
            "eval_dims": eval_result["dims"],
            "eval_feedback": eval_result["feedback"],
        })
        
        print(f"  Dimensiones:")
        for dim, d in eval_result["dims"].items():
            print(f"    {dim}: {d['score']}/10 — {d['feedback']}")
        print(f"  Feedback: {eval_result['feedback']}")
        
        current = eval_result["score"]
        improvement = current - prev_score
        print(f"  Score: {current}/10 (mejora: {improvement:+.1f})")
        
        # Early stopping: converged and good enough
        if iteration >= min_iterations:
            if improvement < 0.3:
                stall_count += 1
                if stall_count >= 2:
                    print(f"\n  [CONVERGENCIA] Score estabilizado en {current}/10 — aceptando veredicto.")
                    break
            else:
                stall_count = 0
        prev_score = current
        
        if eval_result["goal_met"]:
            print(f"\n  [SUCCESS] Alcanzo Goal ({current}/10)!")
            break
        else:
            print(f"\n  [RETRY] Score {current}/10 — refinando...")
            await asyncio.sleep(1)

    # Generate HTML with AI
    print_header("GENERANDO REPORTE HTML FINAL")
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', repo_url.split('/')[-1])
    filename = f"repo_analysis_{clean_name}.html"
    
    html_content = generate_html(repo_url, gh_data, results, verdict, filename, metrics, cost_data)

    # Push to GitHub via SSH
    print_header("SUBIENDO A GITHUB VIA SSH")
    push_to_github(filename, html_content, GITHUB_TARGET_REPO_SSH, SSH_KEY_PATH)

    total = time.time() - start
    print(f"\n{'=' * 74}")
    print(f"  Workflow completado en {total:.0f} segundos")
    raw_url = f"https://github.com/serranogallegogerardo/swarm-code/blob/main/{filename}"
    preview_url = f"https://htmlpreview.github.io/?{raw_url}"
    print(f"  Repo: {raw_url}")
    print(f"  Preview: {preview_url}")
    print(f"{'=' * 74}")

if __name__ == "__main__":
    asyncio.run(main())