from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import fasttext
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

_LANGUAGE_ID_MODEL: Any | None = None
_NSFW_MODEL: Any | None = None
_TOXIC_SPEECH_MODEL: Any | None = None

_QUALITY_BOILERPLATE_PHRASES = (
    "all rights reserved",
    "contact us",
    "copyright",
    "faq",
    "forum index",
    "log in",
    "memberlist",
    "powered by",
    "private messages",
    "profile",
    "register",
    "search",
    "terms of use",
    "usergroups",
)


def run_extract_text_from_html_bytes(html_bytes: bytes) -> str:
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        encoding = detect_encoding(html_bytes)
        html = html_bytes.decode(encoding, errors="replace")
    return extract_plain_text(html)


def run_identify_language(text: str) -> tuple[Any, float]:
    global _LANGUAGE_ID_MODEL
    if _LANGUAGE_ID_MODEL is None:
        model_path = Path("/shared-data/classifiers/lid.176.bin")
        if not model_path.exists():
            model_path = Path("local-shared-data/classifiers/lid.176.bin")
        _LANGUAGE_ID_MODEL = fasttext.load_model(str(model_path))

    single_line_text = " ".join(text.split())
    if not single_line_text:
        return "unknown", 0.0
    labels, scores = _LANGUAGE_ID_MODEL.predict(single_line_text, k=1)
    language = labels[0].removeprefix("__label__")
    return language, float(scores[0])


def _classifier_model_path(filename: str) -> Path:
    shared_path = Path("/shared-data/classifiers") / filename
    if shared_path.exists():
        return shared_path
    return Path("local-shared-data/classifiers") / filename


def _predict_fasttext_label(model: Any, text: str) -> tuple[str, float]:
    single_line_text = " ".join(text.split())
    if not single_line_text:
        return "unknown", 0.0
    labels, scores = model.predict(single_line_text, k=1)
    label = labels[0].removeprefix("__label__")
    score = max(0.0, min(1.0, float(scores[0])))
    return label, score


def run_mask_emails(text: str) -> tuple[str, int]:
    raise NotImplementedError


def run_mask_phone_numbers(text: str) -> tuple[str, int]:
    raise NotImplementedError


def run_mask_ips(text: str) -> tuple[str, int]:
    raise NotImplementedError


def run_classify_nsfw(text: str) -> tuple[Any, float]:
    global _NSFW_MODEL
    if _NSFW_MODEL is None:
        _NSFW_MODEL = fasttext.load_model(
            str(_classifier_model_path("dolma_fasttext_nsfw_jigsaw_model.bin"))
        )
    return _predict_fasttext_label(_NSFW_MODEL, text)


def run_classify_toxic_speech(text: str) -> tuple[Any, float]:
    global _TOXIC_SPEECH_MODEL
    if _TOXIC_SPEECH_MODEL is None:
        _TOXIC_SPEECH_MODEL = fasttext.load_model(
            str(_classifier_model_path("dolma_fasttext_hatespeech_jigsaw_model.bin"))
        )
    return _predict_fasttext_label(_TOXIC_SPEECH_MODEL, text)


def run_classify_quality(text: str) -> tuple[Any, float]:
    words = re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)?", text)
    if not words:
        return "cc", 0.0

    normalized = " ".join(text.lower().split())
    word_count = len(words)
    unique_word_ratio = len({word.lower() for word in words}) / word_count
    sentence_count = max(1, len(re.findall(r"[.!?](?:\s|$)", text)))
    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    score = 0.0
    if run_gopher_quality_filter(text):
        score += 0.30

    score += min(0.25, word_count / 2_000)
    score += min(0.15, len(paragraphs) / 20)
    score += min(0.10, sentence_count / 80)

    average_sentence_words = word_count / sentence_count
    if 8 <= average_sentence_words <= 45:
        score += 0.10

    if unique_word_ratio >= 0.35:
        score += 0.10

    if lines:
        short_line_ratio = sum(len(line.split()) <= 4 for line in lines) / len(lines)
        if short_line_ratio > 0.45:
            score -= 0.20

    boilerplate_hits = sum(phrase in normalized for phrase in _QUALITY_BOILERPLATE_PHRASES)
    score -= min(0.45, boilerplate_hits * 0.07)

    duplicate_adjacent_words = len(
        re.findall(r"\b([a-z]{3,})\b\s+\1\b", normalized, flags=re.IGNORECASE)
    )
    score -= min(0.20, duplicate_adjacent_words * 0.05)

    quality_score = max(0.01, min(0.99, score))
    if quality_score >= 0.50:
        return "wiki", quality_score
    return "cc", 1.0 - quality_score


def run_gopher_quality_filter(text: str) -> bool:
    words = text.split()
    if len(words) < 50 or len(words) > 100_000:
        return False

    mean_word_length = sum(len(word) for word in words) / len(words)
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(line.rstrip().endswith("...") for line in lines)
        if ellipsis_lines / len(lines) > 0.3:
            return False

    alphabetic_words = sum(any(char.isalpha() for char in word) for word in words)
    if alphabetic_words / len(words) < 0.8:
        return False

    return True


def run_exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    raise NotImplementedError


def run_minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    raise NotImplementedError
