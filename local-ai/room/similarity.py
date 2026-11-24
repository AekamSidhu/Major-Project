"""Echo similarity in the room: how close is an answer to what an LLM answers to the same question?

Once the question's transcripts are back during the candidate's answer, the
room's follow-up model (Llama 3.2 through Ollama by default; Gemini in online
mode, which is then sent the interviewer's question only) writes `references`
answers to the interviewer's question, following the recipe in
integrity/signals.py. The same question text reuses those references (see
`SimilarityChecker.references`); a different text starts new ones. When the
answer ends, the answer and the references are embedded with the sidecar's
fastembed model and compared (spec/integrity: the highest cosine).

The thresholds in spec/integrity/thresholds.json were measured with one
reference model and one embedding model. With another embedder the signal is
unavailable; with another reference model (Gemini, or a different Ollama
model) it is still computed but counts as uncalibrated, and says so.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

from integrity import signals
from integrity.text import to_transcript

from . import guardrails
from .followups import Settings

LOG = logging.getLogger("openmeet.room.similarity")
MAX_QUESTION_CHARS = 1500


def same_model(a: str | None, b: str | None) -> bool:
    """`llama3.2` and `llama3.2:latest` are the same Ollama model."""
    strip = lambda m: (m or "").strip().lower().removesuffix(":latest")  # noqa: E731
    return bool(a) and strip(a) == strip(b)


def _text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):  # Gemini can answer in parts
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


class ReferenceWriter:
    """Reference answers from the room's follow-up model, with the calibration's recipe."""

    def __init__(self, mode: str, settings: Settings | None = None,
                 model_factory: Callable[[int], Any] | None = None) -> None:
        self.mode = mode
        self.settings = settings or Settings.load()
        self._factory = model_factory
        self._models: dict[int, Any] = {}

    @property
    def model_name(self) -> str | None:
        return self.settings.model_name(self.mode)

    @property
    def available(self) -> bool:
        return self.mode in ("local", "online")

    def _model(self, i: int) -> Any:
        if i not in self._models:
            self._models[i] = self._factory(i) if self._factory else self._make(i)
        return self._models[i]

    def _make(self, i: int) -> Any:
        if self.mode == "local":
            from langchain_ollama import ChatOllama

            return ChatOllama(model=self.settings.ollama_model, base_url=self.settings.ollama_url,
                              temperature=signals.REFERENCE_TEMPERATURE, num_predict=signals.REFERENCE_MAX_TOKENS,
                              seed=signals.reference_seed(i), client_kwargs={"timeout": 180})
        if self.mode == "online":
            if not self.settings.gemini_key:
                raise RuntimeError("online mode needs OPENMEET_GEMINI_API_KEY")
            from langchain_google_genai import ChatGoogleGenerativeAI

            extra = {"thinking_level": "low"} if self.settings.gemini_model.startswith("gemini-3") else {}
            return ChatGoogleGenerativeAI(model=self.settings.gemini_model, google_api_key=self.settings.gemini_key,
                                          temperature=signals.REFERENCE_TEMPERATURE,
                                          max_output_tokens=signals.REFERENCE_MAX_TOKENS, max_retries=1, timeout=45, **extra)
        raise ValueError(f"no reference model in {self.mode} mode")

    def write(self, question: str, n: int, candidate_name: str = "") -> list[str]:
        """`n` answers to the question. Online, only the question is sent, with the candidate's name,
        emails and phone numbers removed."""
        online = self.mode == "online"
        names = guardrails.name_parts(candidate_name) if online else ()
        q = guardrails.clean(question, MAX_QUESTION_CHARS, online, names)
        messages = [(m["role"] if m["role"] == "system" else "human", m["content"]) for m in signals.reference_messages(q)]
        out = []
        for i in range(n):
            text = _text(self._model(i).invoke(messages)).strip()
            if text:
                out.append(text)
        return out


class SimilarityChecker:
    """References written in the background (one at a time) and cached per question."""

    def __init__(self, writer: ReferenceWriter, embedder: Any, embed_model: str,
                 thresholds: dict[str, Any] | None = None) -> None:
        self.writer, self.embedder, self.embed_model = writer, embedder, embed_model
        self._th = thresholds
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="references")
        self._cache: dict[tuple[str, str], Future] = {}
        self._lock = threading.Lock()
        self._embed_lock = threading.Lock()

    def th(self) -> dict[str, Any]:
        return self._th or signals.thresholds()["echo_similarity"]

    @property
    def calibrated_reference(self) -> bool:
        return same_model(self.writer.model_name, self.th()["reference_model"])

    def references(self, question: str, candidate_name: str = "") -> Future:
        """Start (or reuse) writing the references for a question."""
        key = (" ".join(question.split()).lower(), candidate_name)
        with self._lock:
            job = self._cache.get(key)
            if job is None or (job.done() and job.exception() is not None):  # retry after a failure
                job = self._pool.submit(self.writer.write, question, int(self.th()["references"]), candidate_name)
                self._cache[key] = job
            return job

    def check(self, question: str, answer: str, candidate_name: str = "") -> dict[str, Any]:
        th = self.th()
        base = {"reference_model": self.writer.model_name, "embedding_model": self.embed_model,
                "references": 0}

        def unavailable(why: str) -> dict[str, Any]:
            return {**base, "level": "unavailable", "similarity": None, "calibrated": th["calibrated"], "detail": why}

        if not self.writer.available:
            return unavailable("no language model in rules mode")
        if self.embed_model != th["embedding_model"]:
            return unavailable(f"the thresholds hold for {th['embedding_model']} only")
        if not question.strip():
            return unavailable("the question was not transcribed")
        if not to_transcript(answer):
            return unavailable("no answer text")
        try:
            refs = self.references(question, candidate_name).result(timeout=600)
        except Exception as error:  # model missing, unreachable, out of quota
            LOG.info("reference answers unavailable (%s)", type(error).__name__)
            return unavailable(f"reference model unavailable: {type(error).__name__}")
        texts = [to_transcript(r) for r in refs]
        texts = [t for t in texts if t]
        if not texts:
            return unavailable("the reference model gave no answer")
        try:
            with self._embed_lock:
                vectors = self.embedder.embed([to_transcript(answer)] + texts, self.embed_model)
        except Exception as error:
            return unavailable(f"embeddings unavailable: {type(error).__name__}")
        result = signals.echo_similarity(vectors[0], vectors[1:], th, self.embed_model)
        out = {**base, **result, "references": len(texts)}
        if not self.calibrated_reference:
            out["calibrated"] = False
            out["detail"] = (f"references from {self.writer.model_name}; the thresholds were measured with "
                             f"{th['reference_model']}, so this counts as uncalibrated")
        return out
