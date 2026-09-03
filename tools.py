"""Tool definitions and implementations for nano claude."""
import os
import re
import glob as _glob
import difflib
import subprocess
from pathlib import Path
from typing import Callable, Optional

from tool_registry import ToolDef, register_tool
from tool_registry import execute_tool as _registry_execute

# ── Tool JSON schemas (sent to Claude API) ─────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "Read",
        "description": (
            "Read a file's contents. Returns content with line numbers "
            "(format: 'N\\tline'). Use limit/offset to read large files in chunks."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Absolute file path"},
                "limit":     {"type": "integer", "description": "Max lines to read"},
                "offset":    {"type": "integer", "description": "Start line (0-indexed)"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "Write",
        "description": "Write content to a file, creating parent directories as needed.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "content":   {"type": "string"},
            },
            "required": ["file_path", "content"],
        },
    },
    {
        "name": "Edit",
        "description": (
            "Replace exact text in a file. old_string must match exactly (including whitespace). "
            "If old_string appears multiple times, use replace_all=true or add more context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path":   {"type": "string"},
                "old_string":  {"type": "string", "description": "Exact text to replace"},
                "new_string":  {"type": "string", "description": "Replacement text"},
                "replace_all": {"type": "boolean", "description": "Replace all occurrences"},
            },
            "required": ["file_path", "old_string", "new_string"],
        },
    },
    {
        "name": "Bash",
        "description": "Execute a shell command. Returns stdout+stderr. Stateless (no cd persistence).",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout": {"type": "integer", "description": "Seconds before timeout (default 30)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "Glob",
        "description": "Find files matching a glob pattern. Returns sorted list of matching paths.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. **/*.py"},
                "path":    {"type": "string", "description": "Base directory (default: cwd)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "Grep",
        "description": "Search file contents with regex using ripgrep (falls back to grep).",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern":      {"type": "string", "description": "Regex pattern"},
                "path":         {"type": "string", "description": "File or directory to search"},
                "glob":         {"type": "string", "description": "File filter e.g. *.py"},
                "output_mode":  {
                    "type": "string",
                    "enum": ["content", "files_with_matches", "count"],
                    "description": "content=matching lines, files_with_matches=file paths, count=match counts",
                },
                "case_insensitive": {"type": "boolean"},
                "context":      {"type": "integer", "description": "Lines of context around matches"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "search_papers",
        "description": (
            "在论文语料库中检索与查询最相关的证据片段。内部执行\"稠密检索(BGE-M3)+"
            "稀疏检索(BM25)召回 -> RRF融合排序 -> Cross-Encoder精排\"完整三阶段流程，"
            "返回排好序的证据片段及来源（论文标题、章节、chunk_id）。"
            "query必须是独立、脱离对话上下文也能理解的完整问题——如果用户原话包含"
            "代词或指代（如'这篇论文'、'刚才提到的方法'），请先把它改写成完整问题"
            "再调用本工具。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "独立完整的检索查询，不含未消解的代词或指代",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回的证据片段数量，默认5",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "retrieve_chunk",
        "description": (
            "按chunk_id取回某个证据片段的完整内容，并带上同一篇论文里前后相邻的"
            "chunk扩展上下文。用于search_papers返回的证据片段因为分块边界切得不巧、"
            "需要看更完整上下文的场景，不用于首次检索。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunk_id": {
                    "type": "string",
                    "description": "search_papers返回结果里的chunk_id",
                },
                "context": {
                    "type": "integer",
                    "description": "在目标chunk前后各扩展几个相邻chunk，默认1",
                },
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "WebFetch",
        "description": "Fetch a URL and return its text content (HTML stripped).",
        "input_schema": {
            "type": "object",
            "properties": {
                "url":    {"type": "string"},
                "prompt": {"type": "string", "description": "Hint for what to extract"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "WebSearch",
        "description": "Search the web via DuckDuckGo and return top results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
]

# ── Safe bash commands (never ask permission) ───────────────────────────────

_SAFE_PREFIXES = (
    "ls", "cat", "head", "tail", "wc", "pwd", "echo", "printf", "date",
    "which", "type", "env", "printenv", "uname", "whoami", "id",
    "git log", "git status", "git diff", "git show", "git branch",
    "git remote", "git stash list", "git tag",
    "find ", "grep ", "rg ", "ag ", "fd ",
    "python ", "python3 ", "node ", "ruby ", "perl ",
    "pip show", "pip list", "npm list", "cargo metadata",
    "df ", "du ", "free ", "top -bn", "ps ",
    "curl -I", "curl --head",
)

def _is_safe_bash(cmd: str) -> bool:
    c = cmd.strip()
    # any:一个为真就返回真
    return any(c.startswith(p) for p in _SAFE_PREFIXES)


# ── Diff helpers ──────────────────────────────────────────────────────────
# diff辅助函数，当工具修改文件后生成对比文本
def generate_unified_diff(old, new, filename, context_lines=3):
    old_lines = old.splitlines(keepends=True)  # 保留每行末尾换行符
    new_lines = new.splitlines(keepends=True)
    # unified_diff(旧行, 新行, ...) 对比两串行，产出 git diff 那种格式的差异
    # fromfile / tofile：显示在 diff 头部，标明"这是改前(a/)和改后(b/)的哪个文件"
    # n=context_lines：每个差异块上下各保留几行上下文（默认 3，和 git 默认一致）
    # unified_diff 返回一个生成器（逐行产出）
    diff = difflib.unified_diff(old_lines, new_lines,
        fromfile=f"a/{filename}", tofile=f"b/{filename}", n=context_lines)
    return "".join(diff)

def maybe_truncate_diff(diff_text, max_lines=80):
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    shown = lines[:max_lines]
    remaining = len(lines) - max_lines
    return "\n".join(shown) + f"\n\n[... {remaining} more lines ...]"


# ── Tool implementations ───────────────────────────────────────────────────

def _read(file_path: str, limit: int = None, offset: int = None) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    if p.is_dir():
        return f"Error: {file_path} is a directory"
    try:
        lines = p.read_text(errors="replace").splitlines(keepends=True)  # 按换行符拆成行
        start = offset or 0
        chunk = lines[start:start + limit] if limit else lines[start:] # [a,b]:左闭右开
        if not chunk:
            return "(empty file)"
        return "".join(f"{start + i + 1}\t{l}" for i, l in enumerate(chunk)) # 给每行加上行号，制表符分隔
    except Exception as e:
        return f"Error: {e}"


def _write(file_path: str, content: str) -> str:
    p = Path(file_path)
    try:
        is_new = not p.exists()
        old_content = "" if is_new else p.read_text(errors="replace")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        if is_new:
            lc = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            return f"Created {file_path} ({lc} lines)"
        filename = p.name
        diff = generate_unified_diff(old_content, content, filename)
        truncated = maybe_truncate_diff(diff)
        return f"File updated — {file_path}:\n\n{truncated}"
    except Exception as e:
        return f"Error: {e}"


def _edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    p = Path(file_path)
    if not p.exists():
        return f"Error: file not found: {file_path}"
    try:
        content = p.read_text()
        count = content.count(old_string)
        if count == 0:
            return "Error: old_string not found in file"
        if count > 1 and not replace_all:
            return (f"Error: old_string appears {count} times. "
                    "Provide more context to make it unique, or use replace_all=true.")
        old_content = content
        new_content = content.replace(old_string, new_string) if replace_all else \
                      content.replace(old_string, new_string, 1)
        p.write_text(new_content)
        filename = p.name
        diff = generate_unified_diff(old_content, new_content, filename)
        return f"Changes applied to {filename}:\n\n{diff}"
    except Exception as e:
        return f"Error: {e}"


def _bash(command: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=timeout, cwd=os.getcwd(),
        )
        out = r.stdout
        if r.stderr:
            out += ("\n" if out else "") + "[stderr]\n" + r.stderr
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _glob(pattern: str, path: str = None) -> str:
    # 用 glob 模式查找文件（如 *.py、**/*.py）
    base = Path(path) if path else Path.cwd()   # 基准目录：给了 path 用它，否则用当前目录
    try:
        matches = sorted(base.glob(pattern))    # glob(pattern) 查找匹配文件，sorted 排序
        if not matches:                          # 空列表（没匹配到任何文件）
            return "No files matched"
        return "\n".join(str(m) for m in matches[:500])  # Path 转字符串，最多 500 个，换行拼接
    except Exception as e:
        return f"Error: {e}"


def _has_rg() -> bool:
    try:
        subprocess.run(["rg", "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _grep(pattern: str, path: str = None, glob: str = None,
          output_mode: str = "files_with_matches",
          case_insensitive: bool = False, context: int = 0) -> str:
    use_rg = _has_rg()
    cmd = ["rg" if use_rg else "grep", "--no-heading"]
    if case_insensitive:
        cmd.append("-i")
    if output_mode == "files_with_matches":
        cmd.append("-l")
    elif output_mode == "count":
        cmd.append("-c")
    else:
        cmd.append("-n")
        if context:
            cmd += ["-C", str(context)]
    if glob:
        cmd += (["--glob", glob] if use_rg else ["--include", glob])
    cmd.append(pattern)
    cmd.append(path or str(Path.cwd()))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        out = r.stdout.strip()
        return out[:20000] if out else "No matches found"
    except Exception as e:
        return f"Error: {e}"


def _webfetch(url: str, prompt: str = None) -> str:
    # 抓取网页并转成纯文本（剥掉 HTML 标签），最多返回前 25000 字符
    try:
        import httpx
        r = httpx.get(url, headers={"User-Agent": "NanoClaude/1.0"},  # 伪装浏览器 UA
                      timeout=30, follow_redirects=True)              # 30s 超时，自动跟随跳转
        r.raise_for_status()  # 状态码非 2xx（如 404/500）就抛异常
        ct = r.headers.get("content-type", "")  # 读响应头，判断返回类型
        if "html" in ct:  # 是 HTML → 剥离标签，提取纯文本
            text = re.sub(r"<script[^>]*>.*?</script>", "", r.text,
                          flags=re.DOTALL | re.IGNORECASE)  # ① 删掉 script 块（跨行+忽略大小写）
            text = re.sub(r"<style[^>]*>.*?</style>", "", text,
                          flags=re.DOTALL | re.IGNORECASE)  # ② 删掉 style 块
            text = re.sub(r"<[^>]+>", " ", text)            # ③ 删掉所有剩余 HTML 标签
            text = re.sub(r"\s+", " ", text).strip()        # ④ 连续空白合并成一个空格
        else:
            text = r.text  # 不是 HTML（纯文本/JSON），直接用原文
        return text[:25000]  # 截断，防止海量内容撑爆上下文
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def _websearch(query: str) -> str:
    try:
        import httpx
        url = "https://html.duckduckgo.com/html/"
        r = httpx.get(url, params={"q": query},
                      headers={"User-Agent": "Mozilla/5.0 (compatible)"},
                      timeout=30, follow_redirects=True)
        titles   = re.findall(r'class="result__title"[^>]*>.*?<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
                               r.text, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</div>', r.text, re.DOTALL)
        results = []
        for i, (link, title) in enumerate(titles[:8]):
            t = re.sub(r"<[^>]+>", "", title).strip()
            s = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append(f"**{t}**\n{link}\n{s}")
        return "\n\n".join(results) if results else "No results found"
    except ImportError:
        return "Error: httpx not installed — run: pip install httpx"
    except Exception as e:
        return f"Error: {e}"


def _search_papers(query: str, top_k: int = 5) -> str:
    # 延迟到函数内部import：retrieval.hybrid_search会一路引入sentence-transformers/faiss/
    # torch这些重依赖，只有真正调用search_papers时才需要，避免只用编程工具（Read/Bash等）
    # 的场景也被迫加载这些重量级的检索栈——和下面_webfetch里"函数内部import httpx"是同一个理由
    from retrieval.hybrid_search import search_papers

    query = (query or "").strip()
    if not query:
        return "Error: query is empty"

    results = search_papers(query, top_k=top_k)
    if not results:
        return "(no relevant chunks found)"

    lines = []
    for i, r in enumerate(results):
        lines.append(
            f"[{i+1}] paper={r['title']} section={r['section']} "
            f"chunk_id={r['chunk_id']} score={r['score']:.3f}\n{r['text']}\n"
        )
    return "\n".join(lines)


def _search_papers_tool(params: dict, config: dict) -> str:
    return _search_papers(params.get("query", ""), top_k=params.get("top_k", 5))


def _retrieve_chunk_tool(params: dict, config: dict) -> str:
    from retrieval.hybrid_search import retrieve_chunk

    chunk_id = (params.get("chunk_id") or "").strip()
    if not chunk_id:
        return "Error: chunk_id is empty"

    result = retrieve_chunk(chunk_id, context=params.get("context", 1))
    if result is None:
        return f"Error: chunk_id '{chunk_id}' not found"
    return f"paper={result['title']} section={result['section']}\n{result['text']}"


# ── Dispatcher (backward-compatible wrapper) ──────────────────────────────

def execute_tool(
    name: str,
    inputs: dict,
    permission_mode: str = "auto",
    ask_permission: Optional[Callable[[str], bool]] = None,
) -> str:
    """Dispatch tool execution; ask permission for write/destructive ops.

    Permission checking is done here, then delegation goes to the registry.
    """

    def _check(desc: str) -> bool:
        """Return True if action is allowed."""
        if permission_mode == "accept-all":
            return True
        if ask_permission:
            return ask_permission(desc)
        return True  # headless: allow everything

    # --- permission gate (same logic as before) ---
    if name == "Write":
        if not _check(f"Write to {inputs['file_path']}"):
            return "Denied: user rejected write operation"
    elif name == "Edit":
        if not _check(f"Edit {inputs['file_path']}"):
            return "Denied: user rejected edit operation"
    elif name == "Bash":
        cmd = inputs["command"]
        if permission_mode != "accept-all" and not _is_safe_bash(cmd):
            if not _check(f"Bash: {cmd}"):
                return "Denied: user rejected bash command"

    return _registry_execute(name, inputs, {})


# ── Register built-in tools with the plugin registry ─────────────────────

def _register_builtins() -> None:
    """Register all built-in tools into the central registry."""
    # 按name查schema，而不是像早期版本那样写死TOOL_SCHEMAS[0]/[1]/...这类位置下标——
    # Phase 0清理过一次因为在TOOL_SCHEMAS列表中间插入了新条目、把后面所有下标顶歪导致
    # import直接崩溃的bug，根源就是这种"表的顺序和下标一一对应"的脆弱写法。这里用一个
    # {name: schema}的字典按名字查，往TOOL_SCHEMAS任何位置插入新工具都不会影响其它工具。
    schema_by_name = {s["name"]: s for s in TOOL_SCHEMAS}

    # 每个 ToolDef = 一个工具的"登记条目"：
    #   name    : 工具名（模型调用时用的名字）
    #   schema  : 工具声明（发给模型描述参数）
    #   func    : 真正执行的函数。lambda p, c = 小函数，(p=参数字典, c=config)
    #   lambda  : 匿名函数，p 是模型传来的参数字典，c 是 config
    #             **p 把参数字典解包成关键字参数，如 _read(**p) → _read(file_path=..., limit=...)
    #   concurrent_safe: True=可与其他工具并行执行
    _tool_defs = [
        ToolDef(
            name="Read",
            schema=schema_by_name["Read"],
            func=lambda p, c: _read(**p),  # 参数解包传给 _read(file_path, limit, offset)
            read_only=True,                # 只读
            concurrent_safe=True,
        ),
        ToolDef(
            name="Write",
            schema=schema_by_name["Write"],
            func=lambda p, c: _write(**p),  # 写文件，会修改磁盘
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Edit",
            schema=schema_by_name["Edit"],
            func=lambda p, c: _edit(**p),  # 编辑文件，会修改磁盘
            read_only=False,
            concurrent_safe=False,
        ),
        ToolDef(
            name="Bash",
            schema=schema_by_name["Bash"],
            func=lambda p, c: _bash(p["command"], p.get("timeout", 30)),  # 手动取参数（含默认值）
            read_only=False,  # 执行命令有副作用
            concurrent_safe=False,
        ),
        ToolDef(
            name="Glob",
            schema=schema_by_name["Glob"],
            func=lambda p, c: _glob(p["pattern"], p.get("path")),  # 查找文件，只读
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="Grep",
            schema=schema_by_name["Grep"],
            func=lambda p, c: _grep(  # 多参数：逐个用 p.get() 取，带默认值
                p["pattern"], p.get("path"), p.get("glob"),
                p.get("output_mode", "files_with_matches"),
                p.get("case_insensitive", False),
                p.get("context", 0),
            ),
            read_only=True,  # 搜索内容，只读
            concurrent_safe=True,
        ),
        ToolDef(
            name="search_papers",
            schema=schema_by_name["search_papers"],
            func=_search_papers_tool,
            read_only=True,        # 只读检索，不修改任何状态
            concurrent_safe=True,  # 检索本身无副作用，可以和别的工具并行跑
        ),
        ToolDef(
            name="retrieve_chunk",
            schema=schema_by_name["retrieve_chunk"],
            func=_retrieve_chunk_tool,
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="WebFetch",
            schema=schema_by_name["WebFetch"],
            func=lambda p, c: _webfetch(p["url"], p.get("prompt")),  # 抓网页，只读
            read_only=True,
            concurrent_safe=True,
        ),
        ToolDef(
            name="WebSearch",
            schema=schema_by_name["WebSearch"],
            func=lambda p, c: _websearch(p["query"]),  # 网页搜索，只读
            read_only=True,
            concurrent_safe=True,
        ),
    ]
    for td in _tool_defs:  # 遍历每个 ToolDef，登记进注册表
        register_tool(td)


_register_builtins()

# Domain tools overwrite the legacy wrappers with validated, stable JSON outputs.
from domain_tools import register_domain_tools
register_domain_tools()
