# 论文语料

1. `selection.json` 是固定的 20 篇论文来源清单，[selected_papers.md](selected_papers.md) 提供可阅读版本。
2. `python -m corpus.acquire_selected` 下载 PDF、提取全文，并在 `papers_meta.jsonl` 中记录版本与校验值。
3. `python -m corpus.build_index --cache-dir .cache/models --device cpu --batch-size 8` 生成分块和检索索引。也可用 `--source /path/to/documents` 索引自己的 PDF、Markdown、TXT 或 JSONL 文件。
4. `papers/`、分块和索引属于本地产物，不提交 Git。保留元数据和索引清单用于核对本次实验的来源与配置。
5. Memory Survey 使用清单标明的作者预印本；其期刊发表信息与下载版本分别记录。
