#!/usr/bin/env python3
"""Phase 1 离线脚本：从 arXiv 抓取强化学习方向论文，下载 PDF 并提取正文纯文本。

产出（供 retrieval/chunking.py 在下一步消费）：
    corpus/papers_meta.jsonl   每行一个 JSON 对象：{paper_id, title, authors, abstract, published, pdf_url}
    corpus/papers/pdf/<id>.pdf   原始 PDF（保留一份，方便人工核对/以后换更好的PDF解析器重新提取）
    corpus/papers/text/<id>.txt  pdfplumber 提取出的正文纯文本

用法：
    python corpus/build_corpus.py
"""
from __future__ import annotations

# __future__ import：让本文件里的类型注解（如 list[dict]）在旧版 Python 上也能正常解析，
# 和仓库其它模块（agent.py / compaction.py 等）的写法保持一致。

import re
import json
import time
import xml.etree.ElementTree as ET   # 标准库自带的XML解析器，用来解析 arXiv 返回的 Atom feed
from pathlib import Path

import httpx   # 项目已有依赖（tools.py 的 WebFetch/WebSearch 也用它），这里复用发 HTTP 请求

# ── 路径常量 ────────────────────────────────────────────────────────────────
# Path(__file__).parent：当前文件（build_corpus.py）所在目录，也就是 corpus/ 本身。
# 用相对于文件自身的路径而不是相对于"当前工作目录"，这样不管从哪个目录执行这个脚本
# （比如 `python corpus/build_corpus.py` 还是 `python build_corpus.py`），路径都不会错。
CORPUS_DIR = Path(__file__).parent
PDF_DIR    = CORPUS_DIR / "papers" / "pdf"
TEXT_DIR   = CORPUS_DIR / "papers" / "text"
META_PATH  = CORPUS_DIR / "papers_meta.jsonl"

# ── arXiv 检索参数 ──────────────────────────────────────────────────────────
ARXIV_API   = "https://export.arxiv.org/api/query"   # arXiv 官方 Atom API，不是网页，返回结构化XML
# 查询语法：cat:cs.LG 限定"机器学习"大类，AND abs:"reinforcement learning" 限定摘要包含该短语。
# 这样比只搜关键词更精确，避免搜到跨领域偶然提到"reinforcement learning"一词的无关论文。
SEARCH_QUERY = 'cat:cs.LG AND abs:"reinforcement learning"'
FETCH_COUNT  = 40   # 多取一些做缓冲：PDF下载/解析总有几篇会失败，最终目标是保留约30篇能用的
TARGET_COUNT = 30

# arXiv Atom feed 里所有标签都带命名空间前缀（如 {http://www.w3.org/2005/Atom}entry），
# ElementTree 要求查找标签时把命名空间也写全，这里提前定义好，后面用 f-string 拼接更清晰。
ATOM_NS = "{http://www.w3.org/2005/Atom}"


def fetch_arxiv_metadata(query: str, max_results: int) -> list[dict]:
    """调用 arXiv API，返回论文元数据列表。

    Args:
        query: arXiv 查询语法字符串，如 'cat:cs.LG AND abs:"reinforcement learning"'
        max_results: 最多返回多少条
    Returns:
        每个元素形如 {"paper_id", "title", "authors", "abstract", "published", "pdf_url"}
    """
    # httpx.get(url, params=...) 里的 params 是一个 dict，httpx 会自动把它编码成
    # URL 查询字符串（比如 search_query=... 里的空格/引号会被正确转义），
    # 不需要自己手写 urllib.parse.quote。
    resp = httpx.get(
        ARXIV_API,
        params={
            "search_query": query,
            "sortBy": "relevance",
            "sortOrder": "descending",
            "start": 0,
            "max_results": max_results,
        },
        timeout=30,
    )
    resp.raise_for_status()   # 状态码非 2xx（比如限流429、服务错误500）就直接抛异常，交给上层处理

    # ET.fromstring(text) 把 XML 文本解析成一棵元素树，返回根节点（<feed> 标签）。
    root = ET.fromstring(resp.text)

    papers = []
    # root.findall(f"{ATOM_NS}entry")：在根节点下查找所有 <entry> 子标签（一篇论文对应一个 entry）。
    # 必须带命名空间前缀，否则 ElementTree 找不到任何匹配（这是新手最常踩的坑之一）。
    for entry in root.findall(f"{ATOM_NS}entry"):
        raw_id = entry.findtext(f"{ATOM_NS}id", default="")
        # raw_id 形如 "http://arxiv.org/abs/2401.12345v2"，我们只要不带版本号的短ID "2401.12345"，
        # 因为下载PDF/生成文件名时不需要关心具体是哪个版本，且短ID更适合当 paper_id 用。
        m = re.search(r"abs/([\w.\-]+?)(v\d+)?$", raw_id)
        if not m:
            continue   # 极少数情况下ID格式不符合预期，直接跳过这条，不让脏数据进入语料库
        paper_id = m.group(1)

        title = (entry.findtext(f"{ATOM_NS}title", default="") or "").strip()
        abstract = (entry.findtext(f"{ATOM_NS}summary", default="") or "").strip()
        # 论文的正文摘要里经常有PDF转文本带来的多余换行/空白，这里统一压成单个空格，方便后续存储和展示
        abstract = re.sub(r"\s+", " ", abstract)
        title = re.sub(r"\s+", " ", title)

        published = entry.findtext(f"{ATOM_NS}published", default="")

        # 作者是多个 <author><name>...</name></author> 子标签，用列表推导式一次性收集
        authors = [
            a.findtext(f"{ATOM_NS}name", default="")
            for a in entry.findall(f"{ATOM_NS}author")
        ]

        papers.append({
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": published,
            # arXiv 的PDF直链有固定规律：https://arxiv.org/pdf/<id>.pdf ，
            # 比去 <link type="application/pdf"> 标签里找更简单可靠，直接拼接即可。
            "pdf_url": f"https://arxiv.org/pdf/{paper_id}.pdf",
        })
    return papers


def download_pdf(url: str, dest_path: Path) -> None:
    """下载单个PDF到本地文件。失败时抛异常，由调用方决定如何处理。"""
    resp = httpx.get(url, timeout=60, follow_redirects=True,
                      headers={"User-Agent": "nano-cc-research-agent/0.1 (educational project)"})
    resp.raise_for_status()
    # resp.content 是原始字节（bytes），PDF是二进制格式，必须用 "wb"（写二进制）模式写入，
    # 用普通文本模式 "w" 写会报错或损坏文件。
    dest_path.write_bytes(resp.content)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """用 pdfplumber 把PDF逐页转成纯文本，页与页之间用换行分隔。

    x_tolerance=2：pdfplumber判断"两个相邻字符之间的水平间距要多宽，才认为
    它们之间应该插入一个空格"的阈值，默认值是3。用默认值实测发现相当一部分
    论文（这批语料库里大概2/3）会把整句话的空格全部吃掉，变成
    "descriptionofexplorationratherthanaformalclassification"这种连成一片
    的长字符串——这是部分PDF生成工具导出的字体间距比默认阈值更紧凑导致的。
    调小到2之后，实测同一批论文里"连续25个以上字母粘在一起"的可疑长词
    从每篇几百个降到个位数，且没有明显增加"单词被拆成单字母碎片"的新问题
    （对比测试过x_tolerance=1，那样虽然能修复个别顽固文件，但会让其它本来
    正常的文件出现更多过度切分，属于"改好一个坏一片"，不划算，所以选2作为折中）。
    残留的极少数长词是已知局限，不再进一步调参数解决。

    另一个已知局限（同样是MVP阶段的合理取舍）：论文常见的双栏排版下，
    pdfplumber有时会把左右两栏文字的阅读顺序穿插打乱。曾尝试过"整页对半裁开
    分别提取"和"按字符x坐标密度探测是否双栏"两种方案，但实测发现语料库里
    双栏页和单栏页在这个密度指标上的取值区间有明显重叠，做自动判断反而有把
    正常单栏页面裁坏的风险，所以没有采纳，维持默认整页提取。
    """
    import pdfplumber   # 延迟到函数内部import：这个函数只有真正提取文本时才需要pdfplumber，
                         # 避免脚本一启动（比如只是调用 fetch_arxiv_metadata 做测试时）就强制要求它已安装

    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:   # with语句：无论提取成功与否，退出时都会自动关闭文件句柄
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2) or ""   # 少数页面（纯图片页）会返回None，兜底成空字符串
            pages_text.append(text)
    return "\n".join(pages_text)


def main() -> None:
    # exist_ok=True：目录已存在也不报错；parents=True：连中间缺失的父目录一起创建
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TEXT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] 从 arXiv 检索元数据：{SEARCH_QUERY!r}，请求 {FETCH_COUNT} 条…")
    papers = fetch_arxiv_metadata(SEARCH_QUERY, FETCH_COUNT)
    print(f"      拿到 {len(papers)} 条元数据")

    kept_meta = []   # 只保留PDF下载+文本提取都成功的论文，元数据文件不应该指向不存在的正文
    for i, paper in enumerate(papers):
        if len(kept_meta) >= TARGET_COUNT:
            break   # 已经攒够目标数量，不用再处理剩下的缓冲论文

        pid = paper["paper_id"]
        pdf_path  = PDF_DIR / f"{pid}.pdf"
        text_path = TEXT_DIR / f"{pid}.txt"
        print(f"[2/3] ({i+1}/{len(papers)}) {pid}  {paper['title'][:60]}")

        try:
            if not pdf_path.exists():
                download_pdf(paper["pdf_url"], pdf_path)
            text = extract_text_from_pdf(pdf_path)
            if len(text.strip()) < 500:
                # 提取出来的正文太短，大概率是扫描版PDF（图片，没有可提取的文字层）或提取失败，
                # 这种论文留着也没法支撑"方法细节追问"，直接跳过比留一条空壳数据更干净。
                print(f"      跳过：提取正文过短（{len(text)}字符），可能是扫描版PDF")
                continue
            text_path.write_text(text, encoding="utf-8")
        except Exception as e:
            # 单篇论文的网络错误/解析错误不应该中断整个批处理，打印警告后继续下一篇
            print(f"      跳过：{e}")
            continue

        kept_meta.append(paper)
        time.sleep(1)   # 对 arXiv 服务器的基本礼貌限速，避免短时间内高频请求（PDF下载而非元数据API，
                         # 元数据本身只请求了一次；这里限速的是逐篇PDF下载的频率）

    print(f"[3/3] 写入元数据：{META_PATH}（共 {len(kept_meta)} 篇）")
    with open(META_PATH, "w", encoding="utf-8") as f:
        for paper in kept_meta:
            # ensure_ascii=False：允许输出里包含非ASCII字符（比如作者名里的重音符号）而不转义成\uXXXX，
            # jsonl文件本身就是UTF-8编码，没必要转义
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    print(f"完成：{len(kept_meta)} 篇论文的 PDF 在 {PDF_DIR}，正文文本在 {TEXT_DIR}")


if __name__ == "__main__":
    main()
