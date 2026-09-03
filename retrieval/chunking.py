"""Deterministic chunking for the research corpus.

核心函数：
    build_all_chunks()   读 corpus/papers_meta.jsonl + corpus/papers/text/*.txt，
                         写出 corpus/chunks.jsonl（每行一个chunk，供 dense_index.py /
                         sparse_index.py 在下一步分别建索引时消费）

chunk 的粒度选择："按物理行打包到约1000字符，重叠200字符"，理由：
    - 太小（比如按句子）：检索命中率高但单个chunk信息量太少，生成阶段容易断章取义
    - 太大（比如整页/整节）：检索精度下降（一个chunk里混了多个话题，向量代表性变差），
      而且会把大段不相关内容也喂给生成模型，浪费token、增加幻觉风险
    - 1000字符（约150~200个英文单词）是学术论文RAG场景的常见经验值，
      配合200字符overlap，能覆盖大多数"一个论述跨两段"的情况

为什么是"按行"而不是更常见的"按段落"：PDF提取出的文本没有空行来标记段落边界
（pdfplumber只按物理行输出文本，一个段落内部的折行和真正的段落分界在换行符上
没有任何区别），所以这里放弃了"先按空行切段落"的思路，直接按行累积，
详见 split_into_lines_with_sections() 的说明。
"""
from __future__ import annotations

import re
import json
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "corpus"
META_PATH  = CORPUS_DIR / "papers_meta.jsonl"
TEXT_DIR   = CORPUS_DIR / "papers" / "text"
CHUNKS_PATH = CORPUS_DIR / "chunks.jsonl"

CHUNK_SIZE = 1000   # 每个chunk的目标字符数上限
OVERLAP    = 200    # 相邻chunk之间重叠的字符数

# 这两类章节只是引用列表/补充材料，不含论文的实质论述，检索命中了也无法回答
# "方法细节"类问题，直接在分块阶段排除，不进入下游索引
EXCLUDED_SECTIONS = {"References", "Appendix"}


# ── 章节识别（启发式，非精确） ────────────────────────────────────────────
# 常见论文章节标题的正则模式：
#   ^                     行首
#   (?:\d{1,2}\.?\s*)?    可选的编号前缀，如 "3. " 或 "III "（\d{1,2}只匹配阿拉伯数字，
#                         罗马数字章节标题识别不了，属于已知局限，够用即可）
#   (关键词)              章节名关键词，忽略大小写匹配
#   \s*$                  行尾（章节标题通常独占一行，后面不该跟别的文字）
_SECTION_PATTERN = re.compile(
    r"^(?:\d{1,2}\.?\s*)?"
    r"(Abstract|Introduction|Related\s+Work|Background|Preliminaries|"
    r"Method(?:ology)?|Approach|Experiments?|Results?|Evaluation|"
    r"Discussion|Conclusion|Limitations|References|Appendix)"
    r"\s*$",
    re.IGNORECASE,
)


def _fix_hyphenation(text: str) -> str:
    """只修复"单词被行尾连字符拆断"这一种噪音，特意不动其它换行结构。

    章节标题的识别依赖"这一行只有标题几个字、单独成行"这个结构特征，
    如果在这一步之前就把所有单换行压成空格（把标题行和紧跟的正文粘到一起），
    标题就再也不可能被认出来了——这正是本文件早期版本里那个bug的根源，
    这里特意把"修连字符"和"压平换行"拆成两个独立步骤，前者在章节识别之前做，
    后者（_normalize_paragraph）在章节识别之后、对单个段落内部做。
    """
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


def split_into_lines_with_sections(raw_text: str) -> list[tuple[str, str]]:
    """把原始正文按物理行切开，并给每一行标注当时所属的章节。

    为什么是"按行"而不是"按空行分段"：PDF本身不存储"段落"这种语义结构，
    pdfplumber逐行吐出文本时，普通的段内折行和真正的段落分界在换行符上没有
    任何区别——实测这批语料库里几乎不会出现连续两个换行符，如果按
    `\\n\\s*\\n`（空行）去切段落，整篇论文会被当成一个段落，直接导致：
      (a) 章节标题只有全文第一行有机会被检测到，后面的全部检测不到
      (b) "按段落打包chunk"的逻辑名存实亡，退化成对着整篇论文做纯字符硬切
    这是本文件早期版本的真实bug（用手写的、带空行的合成文本做单测时没暴露出来，
    因为那份合成数据的段落间距是我自己加的空行，不是pdfplumber真实的输出形态）。

    做法：
        1. 先修复断行连字符（不影响按行处理）
        2. 按 \\n 切成物理行列表
        3. 依次遍历每一行：如果整行内容匹配章节标题模式，更新current_section
           （标题本身也算进它所代表的新章节里，而不是算进上一个章节）
        4. 跳过空行；返回 (行文本, 所属章节名) 列表

    Returns:
        [(单行文本, 所属章节名), ...]，按原文顺序排列
    """
    text = _fix_hyphenation(raw_text)

    current_section = "Unknown"
    result: list[tuple[str, str]] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # 标题行一般比较短；过滤掉超长行，减少正则在正文长行上误判的概率
        if len(stripped) <= 40:
            m = _SECTION_PATTERN.match(stripped)
            if m:
                current_section = m.group(1).title()
        result.append((stripped, current_section))
    return result


# ── 分块 ────────────────────────────────────────────────────────────────────

def split_into_chunks(
    lines: list[tuple[str, str]],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[tuple[str, str]]:
    """把 (行文本, 章节名) 列表打包成若干 (chunk文本, 章节名)，行与行之间用空格拼接。

    章节标签规则：一个chunk的section取"这个chunk第一行所属的章节"——多数chunk
    不会跨越太多章节，用起始行的章节做近似标签，对科研问答场景（回答里注明
    "该结论大致来自Method部分"）已经够用，不追求字符级精确对齐。

    算法：
        1. 依次把行塞进"当前chunk"，一旦塞入后长度超过chunk_size，
           就把当前chunk定稿，从"上一个chunk末尾overlap个字符"开始累积下一个chunk
        2. 单行本身就超过chunk_size（罕见，比如整段没识别出换行）时，直接按字符硬切
    """
    chunks: list[tuple[str, str]] = []
    current_text = ""
    current_section = "Unknown"

    for line_text, section in lines:
        if len(line_text) > chunk_size:
            if current_text:
                chunks.append((current_text, current_section))
                current_text = ""
            for i in range(0, len(line_text), chunk_size - overlap):
                chunks.append((line_text[i:i + chunk_size], section))
            continue

        candidate = f"{current_text} {line_text}" if current_text else line_text
        if len(candidate) <= chunk_size:
            if not current_text:
                # 当前chunk还是空的，说明这一行是新chunk的第一行，用它的章节给整个chunk定标签
                current_section = section
            current_text = candidate
        else:
            chunks.append((current_text, current_section))
            # 下一个chunk从"上一个chunk末尾overlap个字符" + 当前这一行 开始，
            # 相邻两个chunk之间就有overlap个字符的内容重复，避免边界信息丢失
            tail = current_text[-overlap:] if len(current_text) > overlap else current_text
            current_text = f"{tail} {line_text}"
            current_section = section

    if current_text:
        chunks.append((current_text, current_section))
    return chunks


def build_chunks_for_paper(paper_meta: dict, full_text: str) -> list[dict]:
    """把一篇论文的正文变成一组chunk字典。

    Args:
        paper_meta: build_corpus.py 产出的元数据（paper_id/title/authors/...）
        full_text: 该论文对应 .txt 文件的原始内容
    Returns:
        每个元素形如：
        {chunk_id, paper_id, title, section, chunk_index, text}
    """
    lines = split_into_lines_with_sections(full_text)
    pieces = split_into_chunks(lines)
    # 参考文献列表/附录不是论文的实质内容，只是引用条目/补充材料，留在索引里不会给
    # "方法细节追问"这类问题提供有效证据，反而可能因为偶然命中某个关键词被误召回。
    # 这里过滤掉这两类chunk，不写进最终语料库。
    pieces = [(text, section) for text, section in pieces if section not in EXCLUDED_SECTIONS]

    results = []
    for idx, (text, section) in enumerate(pieces):
        results.append({
            "chunk_id":    f"{paper_meta['paper_id']}_{idx}",
            "paper_id":    paper_meta["paper_id"],
            "title":       paper_meta["title"],
            "section":     section,
            "chunk_index": idx,
            "text":        text,
        })
    return results


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """读取 build_all_chunks() 写出的 chunks.jsonl，逐行反序列化成字典列表。

    dense_index.py / sparse_index.py 建索引前都要先拿到全部chunk，
    这个读取逻辑只写一份放在这里，两边import复用，避免重复代码。
    """
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_all_chunks(
    meta_path: Path = META_PATH,
    text_dir: Path = TEXT_DIR,
    out_path: Path = CHUNKS_PATH,
) -> list[dict]:
    """批量处理 corpus/papers_meta.jsonl 里的所有论文，写出 corpus/chunks.jsonl。

    Returns:
        写入的全部chunk字典列表（同时也是写进文件的内容），方便调用方
        （比如 corpus/build_index.py）直接拿内存里的结果去建索引，不用再读一次盘。
    """
    all_chunks: list[dict] = []
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            paper_meta = json.loads(line)
            text_path = text_dir / f"{paper_meta['paper_id']}.txt"
            if not text_path.exists():
                print(f"  跳过 {paper_meta['paper_id']}：找不到正文文件 {text_path}")
                continue
            full_text = text_path.read_text(encoding="utf-8")
            chunks = build_chunks_for_paper(paper_meta, full_text)
            all_chunks.extend(chunks)

    with open(out_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"共生成 {len(all_chunks)} 个chunk，来自 "
          f"{len({c['paper_id'] for c in all_chunks})} 篇论文，写入 {out_path}")
    return all_chunks


from retrieval.chunking_v2 import (
    build_chunks_for_paper,
    build_chunks_from_source,
    split_into_chunks,
    split_into_lines_with_sections,
    stable_chunk_id,
    write_chunks,
)
from retrieval.chunking_v2 import build_all_chunks as _build_all_chunks_v2


def build_all_chunks(meta_path: Path = META_PATH, text_dir: Path = TEXT_DIR,  # type: ignore[no-redef]
                     out_path: Path = CHUNKS_PATH, *, config=None) -> list[dict]:
    """Backward-compatible wrapper around the configurable implementation."""
    return _build_all_chunks_v2(meta_path, text_dir, out_path, config=config)


if __name__ == "__main__":
    build_all_chunks()
