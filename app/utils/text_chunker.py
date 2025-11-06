def chunk_text(text, max_tokens=800):
    """
    Splits a large block of text into smaller paragraph-based chunks.
    Converts every new line in a paragraph to a bullet point for cleaner downstream responses.
    """
    paragraphs = text.split("\n\n")
    cleaned_chunks = []
    for p in paragraphs:
        bulletified = '\n'.join([f'- {line.strip()}' for line in p.strip().split('\n') if line.strip()])
        if bulletified:
            cleaned_chunks.append({"content": bulletified, "meta": {"source": "upload"}})
    return cleaned_chunks
