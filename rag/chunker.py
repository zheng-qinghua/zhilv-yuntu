from hashlib import md5
from config import DATA_DIR


def split_markdown_to_chunks(maekdown_text: str, source_name: str) -> list[dict[str,str]]:
  chunks: list[dict[str, str]] = []
  current_title = "文档开头"
  current_lines: list[str] = []
  for line in maekdown_text.splitlines():
    stripped = line.strip()

    if stripped.startswith("## ") or stripped.startswith("### "):
      if current_lines:
        chunks.append({
          "title": current_title,
          "text": "\n".join(current_lines).strip(),
          "source": source_name,
        })
        current_lines = []

      current_title = stripped.lstrip("#").strip()
    elif stripped:
      current_lines.append(stripped)
  
  if current_lines:
    chunks.append({
      "title": current_title,
      "text": "\n".join(current_lines).strip(),
      "source": source_name,
    })

  return chunks

def build_chunk_id(source: str, title: str, text: str) -> str:
  digest = md5(f"{source}|{title}|{text}".encode("utf-8")).hexdigest()
  return f"{source}_{digest}"

def load_all_chunks() -> list[dict[str, str]]:
  all_chunks: list[dict[str, str]] = []
  for guide_file in sorted(DATA_DIR.glob("*.md*")):
    text = guide_file.read_text(encoding="utf-8")

    for chunk in split_markdown_to_chunks(text, guide_file.name):
      all_chunks.append({
        "id": build_chunk_id(chunk["source"], chunk["title"], chunk["text"]),
        "title": chunk["title"],
        "text": chunk["text"],
        "source": chunk["source"],
      })
  return all_chunks