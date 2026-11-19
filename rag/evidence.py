"""将检索结果转换为可核对的参考资料。"""
from pathlib import Path


NO_EVIDENCE_REPLY = "知识库中没有检索到相关资料，暂时无法给出有依据的回答。"


def source_label(metadata: dict) -> str:
    source = metadata.get("source") or "未知来源"
    label = Path(str(source).replace("\\", "/")).name
    page = metadata.get("page")
    if isinstance(page, int) and not isinstance(page, bool) and page >= 0:
        label += f"（第{page + 1}页）"
    return label


def prepare_evidence(documents) -> tuple[str, list[str]]:
    """只保留有正文的片段，按首次出现顺序列出来源。"""
    snippets = []
    sources = []
    for doc in documents:
        content = doc.page_content.strip()
        if not content:
            continue
        label = source_label(doc.metadata)
        snippets.append(f"【参考资料{len(snippets) + 1}，来源：{label}】\n{content}")
        if label not in sources:
            sources.append(label)
    return "\n\n".join(snippets), sources
