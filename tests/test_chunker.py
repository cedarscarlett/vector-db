# tests/test_chunker.py

from codesem.indexing.hashing import hash_text


# NOTE:
# This file currently tests a local reference chunker implementation.
# Once the real production chunker (codesem/indexing/chunker.py) is
# implemented, these tests should be updated to validate the actual
# chunking logic instead of this placeholder.

def simple_line_chunker(text: str, max_lines: int = 4, overlap: int = 1):
    """
    Minimal line-based chunker used only for testing behavior.

    Not the production chunker — this is a simple deterministic
    reference implementation for testing overlap logic.
    """
    lines = text.splitlines()
    chunks = []

    i = 0
    while i < len(lines):
        chunk_lines = lines[i:i + max_lines]
        if not chunk_lines:
            break

        start_line = i + 1
        end_line = i + len(chunk_lines)

        content = "\n".join(chunk_lines)

        chunks.append(
            {
                "start_line": start_line,
                "end_line": end_line,
                "content": content,
                "content_hash": hash_text(content),
            }
        )

        i += max_lines - overlap

    return chunks


def test_chunk_count_with_overlap():
    text = "\n".join([f"line {i}" for i in range(1, 11)])

    chunks = simple_line_chunker(text, max_lines=4, overlap=1)

    # Expect deterministic chunk count
    # 10 lines, chunk size 4, overlap 1 → stride = 3
    # chunks start at lines: 1, 4, 7, 10
    assert len(chunks) == 4

    assert chunks[0]["start_line"] == 1
    assert chunks[1]["start_line"] == 4
    assert chunks[2]["start_line"] == 7
    assert chunks[3]["start_line"] == 10


def test_chunk_overlap_content():
    text = "\n".join([f"line {i}" for i in range(1, 8)])

    chunks = simple_line_chunker(text, max_lines=3, overlap=1)

    # First chunk: lines 1-3
    assert chunks[0]["content"] == "line 1\nline 2\nline 3"

    # Second chunk should overlap last line of first chunk
    # lines 3-5
    assert chunks[1]["content"].startswith("line 3")
