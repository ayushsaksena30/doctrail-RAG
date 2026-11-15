import json
import urllib.request
import urllib.error
from typing import List, Dict, Any


def build_rag_prompt(query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Formats retrieved chunks with metadata into a strict RAG prompt."""
    context_blocks = []
    for c in retrieved_chunks:
        body = c.get("text", c.get("snippet", "")).strip()
        context_blocks.append(
            f"[Source: {c['rel_path']} | Page: {c['page']} | ID: {c['chunk_id']}]\n{body}"
        )

    context_str = "\n\n---\n\n".join(context_blocks)

    return f"""You are a helpful assistant answering strictly using the provided context.
        If the answer cannot be determined from the context, state that you do not know. 
        Do not speculate or use outside knowledge. Cite sources where appropriate.

        Context:
        {context_str}

        Question: {query}
        Answer:"""


def generate_answer_ollama(
    prompt: str,
    model: str = "llama3.2:1b",
    endpoint: str = "http://localhost:11434/api/generate"
) -> str:
    """
    Invokes a local Ollama instance using standard library urllib (no third-party SDKs).
    """

    #Make JSON payload for Ollama API
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False
    }).encode("utf-8")

    #HTTP request to Ollama API
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return (
            f"[Generation Error] Could not connect to local Ollama at {endpoint}.\n"
            f"Details: {e.reason}\n"
            f"Make sure Ollama is running (`ollama serve`) and the model is downloaded (`ollama pull {model}`)."
        )