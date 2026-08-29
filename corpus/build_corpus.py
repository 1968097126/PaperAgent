#!/usr/bin/env python3
"""Phase 1 离线脚本：从 arXiv 抓取强化学习方向论文，下载 PDF 并提取正文纯文本。

产出（供 retrieval/chunking.py 在下一步消费）：
    corpus/papers_meta.jsonl   每行一个 JSON 对象：{paper_id, title, authors, abstract, published, pdf_url}
    corpus/papers/pdf/<id>.pdf   原始 PDF（保留一份，方便人工核对/以后换更好的PDF解析器重新提取）
    corpus/papers/text/<id>.txt  PyMuPDF 提取出的正文纯文本

用法：
    python corpus/build_corpus.py
"""
from __future__ import annotations

# __future__ import：让本文件里的类型注解（如 list[dict]）在旧版 Python 上也能正常解析，
# 和仓库其它模块（agent.py / compaction.py 等）的写法保持一致。

import re
import json
import time
import unicodedata               # 把PDF里的连字符Unicode字形（如"ﬁ"）规范化回普通字母组合
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


def _classify_block(block: tuple, mid_x: float, tolerance: float = 5.0) -> str:
    """判断一个PyMuPDF文本块相对页面竖直中线落在"左栏"/"右栏"/"通栏"中的哪一种。

    block 是 page.get_text("blocks") 返回的 (x0, y0, x1, y1, text, block_no, block_type) 元组。
    整个块的右边界(x1)在中线左侧 -> 左栏；左边界(x0)在中线右侧 -> 右栏；
    横跨中线（比如标题、单栏排版的图注）-> 通栏。tolerance给5pt的容差，
    避免因为字号/行末标点导致的几像素误差被误判成"跨栏"。
    """
    x0, x1 = block[0], block[2]
    if x1 <= mid_x + tolerance:
        return "left"
    if x0 >= mid_x - tolerance:
        return "right"
    return "full"


def _order_page_text(page) -> str:
    """把PyMuPDF一页里的文本块按"正确阅读顺序"重排后拼成文本。

    PyMuPDF按内部顺序输出的文本块基本是从上到下，但双栏论文里同一水平区间
    左右两栏的块会紧挨着出现在列表里——如果照单全收直接拼接，两栏内容会像
    "左栏第1句 右栏第1句 左栏第2句 右栏第2句"这样交叉穿插，读起来语序错乱
    （这正是本文件早期版本用pdfplumber逐字符提取时遇到的问题，且当时试过两种
    自动检测方案都因信号不可靠而放弃）。

    做法：先按y0（从上到下）把所有文字块排好序，然后扫描一遍——连续出现的
    "左栏/右栏"块先收集成一个"分栏区"(band)，一旦遇到"通栏"块（标题、单栏图注等）
    就把之前收集的分栏区结算掉。每个分栏区内部按"通栏 -> 左栏从上到下 -> 右栏从上到下"
    的顺序重新拼接，而不是保留它们在原始列表里穿插的顺序。单栏页面会退化成
    全部落在"左栏"或"通栏"分类里，顺序和不做任何处理时一致，不受影响。
    """
    blocks = page.get_text("blocks")
    mid_x = page.rect.width / 2

    # block_type==0是文字块（1是图片块，没有文字内容，直接跳过）；
    # 按y0排序确保band切分时是按页面自然的从上到下顺序扫描
    text_blocks = sorted(
        (b for b in blocks if b[6] == 0 and b[4].strip()),
        key=lambda b: b[1],
    )

    bands: list[list[tuple[str, tuple]]] = []
    current: list[tuple[str, tuple]] = []
    for b in text_blocks:
        cls = _classify_block(b, mid_x)
        if cls == "full":
            if current:
                bands.append(current)
                current = []
            bands.append([("full", b)])
        else:
            current.append((cls, b))
    if current:
        bands.append(current)

    lines = []
    for band in bands:
        # 依次取出这个band里的通栏块、左栏块、右栏块——用固定顺序('full','left','right')
        # 遍历一次band，而不是三次单独过滤，是为了保持"同一类别内部"原有的y顺序不被打乱
        for wanted in ("full", "left", "right"):
            for cls, b in band:
                if cls == wanted:
                    lines.append(b[4])
    return "\n".join(lines)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """用 PyMuPDF 把PDF逐页转成纯文本，双栏排版按"左栏后右栏"重排，页与页之间用换行分隔。

    早期版本用pdfplumber逐字符提取，暴露过两个问题：默认参数会把大量单词间的
    空格吃掉（曾经手调x_tolerance缓解，但残留个位数问题），以及双栏论文阅读顺序
    被打乱且找不到可靠的自动检测方案。换成PyMuPDF按"文本块"（block，通常对应
    一个自然段，自带位置信息）而不是逐字符提取和排序后，两个问题一并解决：
    单词间距由PyMuPDF自己的规则处理（实测这批语料库里"25字符以上疑似粘连长词"
    从每篇几百个降到0），块级坐标做左右栏分类也比之前在字符/物理行粒度上做
    双栏检测的信号干净得多——因为一个text block基本就是一个完整段落，
    不会像单个字符或物理行那样被页面顶部的通栏标题稀释掉分类信号。

    unicodedata.normalize("NFKC", ...)：PyMuPDF会保留PDF里的连字符(ligature)
    Unicode字形，比如"field"如果在PDF里用了印刷连字符，"fi"会被编码成
    U+FB01这一个字符而不是"f"+"i"两个字符。NFKC规范化把这类连字符拆回普通
    ASCII字母组合，否则后面BM25分词器（正则只匹配a-zA-Z）会把整个含连字符的
    单词当噪音丢掉，看起来像少了个词。
    """
    import pymupdf   # 延迟到函数内部import：这个函数只有真正提取文本时才需要pymupdf，
                      # 避免脚本一启动（比如只是调用 fetch_arxiv_metadata 做测试时）就强制要求它已安装

    doc = pymupdf.open(pdf_path)
    pages_text = [_order_page_text(page) for page in doc]
    doc.close()   # PyMuPDF的Document不支持with语句自动关闭，需要手动close释放底层文件句柄
    return unicodedata.normalize("NFKC", "\n".join(pages_text))


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
