"""Visitor chat over the same PDF, retrieval engine and persona as the CLI."""

import logging
import os
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

import rag_query_engine as rag

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def _pdf():
    path = Path(os.getenv("RAG_PDF") or ROOT / "questions.pdf").expanduser().resolve()
    if not path.is_file():
        raise ValueError("The interview PDF is missing. Check the RAG_PDF setting.")
    return path


def _answer_text(doc):
    return doc.page_content.split("\nA: ", 1)[-1].split("\nTopics: ", 1)[0].strip()


def _sources(label, pdf, docs):
    items = [label]
    for doc in docs:
        metadata = doc.metadata
        heading = " › ".join(filter(None, (metadata["section"], metadata["subtopic"])))
        excerpt = _answer_text(doc)
        if len(excerpt) > 240:
            excerpt = excerpt[:237].rsplit(" ", 1)[0] + "…"
        items.append(f"{pdf.name} · {metadata['qid']}  {heading}\n{excerpt}")
    return "\n\n".join(items)


# A local excerpt search uses subtopics because the subject of an answer is
# often named in an adjacent Q&A (e.g. "that relationship" in the interview).
_SEARCH_STOP = rag.STOP | set(
    "give would could should like please much very really when where why which "
    "has does get some something anything more about s t up".split()
)
_FORMS = {"met": "meet", "meeting": "meet", "growing": "grow", "grew": "grow",
          "employment": "job", "jobs": "job", "favourite": "favorite",
          "spouse": "partner", "wife": "partner", "husband": "partner"}


def _terms(text):
    text = re.sub(r"\bwhere (?:did you |you )?(?:grow|grew) up\b", "childhood home",
                  text, flags=re.I)
    text = re.sub(r"\b(?:grow|grew|growing) up\b", "childhood", text, flags=re.I)
    return [_FORMS.get(word, word) for word in re.findall(r"[^\W_]+", text.casefold())
            if word not in _SEARCH_STOP]


@lru_cache(maxsize=2)
def _load_archive(path, modified, size):
    # Modification time and size invalidate the cache when an interview is replaced.
    docs = rag.parse_pdf(path) or rag.fallback_chunks(path)
    if not docs:
        raise ValueError("The interview PDF contains no readable text.")
    groups = defaultdict(list)
    for doc in docs:
        groups[doc.metadata["group"]].append(doc)
    groups = list(groups.values())
    tokens = [_terms(group[0].metadata["group"] + " " +
                     " ".join(doc.page_content for doc in group)) for group in groups]
    return docs, groups, BM25Okapi(tokens)


def _local_reply(question, archive, path, label):
    _, groups, search = archive
    terms = set(_terms(question))
    if terms:
        scores = search.get_scores(sorted(terms))
        # Sparse or single-group corpora can have zero/negative BM25 scores.
        # Rank by covered question terms first, then by BM25 within ties.
        overlap = [len(terms.intersection(words)) for words in search.doc_freqs]
        best = max(range(len(groups)), key=lambda i: (overlap[i], scores[i]))
        if overlap[best] and overlap[best] / len(terms) >= 0.7:
            group = groups[best]
            # For a general topic, the first record introduces it. For a more
            # specific request, prefer the passage sharing the rarest words.
            record = group[0]
            if terms.difference(_terms(record.metadata["group"])):
                def relevance(doc):
                    matched = terms.intersection(_terms(doc.page_content))
                    return len(matched), sum(search.idf.get(word, 0) for word in matched)

                record = max(group, key=relevance)
            # An exact interview question has a single authoritative answer.
            for doc in group:
                _, separator, rest = doc.page_content.partition("\nQ: ")
                recorded_question = rest.partition("\nA: ")[0]
                if separator and _terms(recorded_question) == _terms(question):
                    record = doc
                    break
            return _answer_text(record), _sources(label, path, [record])
    return ("I couldn’t find a recorded memory that answers that question.",
            _sources(label + " No matching passage found.", path, []))


class _VisitorRetriever(rag.Retriever):
    """Read the existing Pinecone index. Each turn has its own embedding cache."""

    def __init__(self, docs, strict):
        super().__init__(docs, None, None, os.getenv("RAG_NAMESPACE") or "person-001")
        self.strict = strict
        self.label = "OpenAI reply from interview records retrieved with BM25."

    def vector_ids(self, query):
        if not os.getenv("PINECONE_API_KEY"):
            if self.strict:
                raise RuntimeError("PINECONE_API_KEY is missing.")
            return []
        try:
            self.index = _read_index(os.getenv("RAG_INDEX") or "ancestor-life-story")
            self.embeddings = rag.OpenAIEmbeddings(
                model=os.getenv("RAG_EMBEDDING") or "text-embedding-3-small",
                dimensions=rag.EMBED_DIM, request_timeout=20, max_retries=0,
            )
            matches = super().vector_ids(query)
            if not matches:
                raise LookupError("No interview records matched the configured index.")
        except Exception as exc:
            if self.strict:
                raise RuntimeError("Check the Pinecone index, namespace and embedding configuration.") from exc
            logger.warning("Vector retrieval unavailable (%s). Using BM25.", type(exc).__name__)
            return []
        self.label = "OpenAI reply from interview records retrieved with Pinecone and BM25."
        return matches


@lru_cache(maxsize=2)
def _read_index(name):
    # Index(name) looks up its host. No create, upsert or delete on visitor requests.
    return rag.Pinecone().Index(name)


def generate_response(question: str, history: list[tuple[str, str]] | None = None) -> tuple[str, str]:
    """Return a grounded answer and reader-visible evidence for this visitor's turn."""
    question = question.strip()
    if not question:
        raise ValueError("Enter a question about the recorded memories.")
    mode = (os.getenv("RAG_MODE") or "auto").strip().lower()
    if mode not in {"auto", "cloud", "local"}:
        raise ValueError("RAG_MODE must be auto, cloud, or local.")
    path = _pdf()
    stat = path.stat()
    archive = _load_archive(path, stat.st_mtime_ns, stat.st_size)
    label = "Local interview search · verbatim excerpts."
    if mode != "local" and os.getenv("OPENAI_API_KEY"):
        retriever = _VisitorRetriever(archive[0], strict=mode == "cloud")
        try:
            companion = rag.Companion(retriever, os.getenv("RAG_MODEL") or "gpt-4.1-mini",
                                      timeout=35, max_retries=0)
            companion.history = list((history or [])[-3:])
            answer, docs = companion.answer_with_sources(question)
            return answer, _sources(retriever.label, path, docs)
        except Exception as exc:
            if mode == "cloud":
                raise RuntimeError("The cloud reply failed. Check the OpenAI and Pinecone settings and service access.") from exc
            logger.warning("Cloud generation unavailable (%s). Using local excerpts.", type(exc).__name__)
            label += " Cloud generation unavailable."
    elif mode == "cloud":
        raise RuntimeError("Set OPENAI_API_KEY and PINECONE_API_KEY to use cloud mode.")
    return _local_reply(question, archive, path, label)
