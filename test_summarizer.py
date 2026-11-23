"""Tests that need NO API key and cost nothing. Run with:  python test_summarizer.py

A fake model stands in for the real one, so these tests check your code
(loading, chunking, map and reduce, retries, errors), not the quality of the AI's writing.
"""
import os
import tempfile
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda
from pypdf import PdfWriter

import config
import summarizer
from loaders import LoadError, load_document
from summarizer import (
    SummaryError,
    merge_documents,
    split_text,
    summarize_chunk_list,
)

LONG_TEXT = "\n\n".join(
    f"Paragraph {i}. " +
    "The library opened a new reading room for children. " * 6
    for i in range(1, 41)
)


def make_fake_llm(fail_marker=None, fail_always=False, long_reduce=False):
    """A pretend model. It records how many times each step was called."""
    state = {"map": 0, "reduce": 0, "marker_seen": 0, "last_reduce_parts": 0}

    def fake(prompt_value):
        text = prompt_value.to_string()
        if "Partial summaries:" in text:
            state["reduce"] += 1
            state["last_reduce_parts"] = text.count("--- Part ")
            if long_reduce:
                return AIMessage(content="x" * 500)
            return AIMessage(content=f"Summary: REDUCED({state['last_reduce_parts']})")
        state["map"] += 1
        if fail_marker and fail_marker in text:
            state["marker_seen"] += 1
            if fail_always or state["marker_seen"] == 1:
                raise RuntimeError("boom")
        return AIMessage(content="- short bullet summary")

    return RunnableLambda(fake), state


def chunks_of(text, size=800, overlap=80):
    config.CHUNK_SIZE, config.CHUNK_OVERLAP = size, overlap
    document = Document(page_content=text, metadata={"source": "t.txt"})
    return split_text(merge_documents([document]))


def test_chunking():
    chunks = chunks_of(LONG_TEXT)
    assert len(chunks) > 3, len(chunks)
    assert all(len(c.page_content) <= 800 for c in chunks)
    print(f"PASS chunking: {len(chunks)} chunks, none longer than 800")


def test_map_reduce_flow():
    chunks = chunks_of(LONG_TEXT)
    llm, state = make_fake_llm()
    summary = summarize_chunk_list(chunks, llm)
    assert state["map"] == len(chunks)
    assert state["reduce"] == 1
    assert summary == f"Summary: REDUCED({len(chunks)})", summary
    print(
        f"PASS map-reduce: {state['map']} map calls, 1 reduce call, all parts reached the reduce step")


def test_hierarchical_reduce():
    chunks = chunks_of(LONG_TEXT)
    old = config.REDUCE_LIMIT
    config.REDUCE_LIMIT = 150
    try:
        llm, state = make_fake_llm()
        summary = summarize_chunk_list(chunks, llm)
    finally:
        config.REDUCE_LIMIT = old
    assert state["reduce"] > 1, state
    assert summary.startswith("Summary:")
    print(
        f"PASS hierarchical reduce: {state['reduce']} reduce calls ended in one summary")


def test_retry_once():
    chunks = chunks_of(LONG_TEXT)
    marker = chunks[2].page_content[:40]
    llm, state = make_fake_llm(fail_marker=marker)
    summarize_chunk_list(chunks, llm)
    assert state["map"] == len(chunks) + 1, state
    print("PASS retry: a chunk that failed once was retried and the run finished")


def test_chunk_fails_twice():
    chunks = chunks_of(LONG_TEXT)
    marker = chunks[2].page_content[:40]
    llm, _ = make_fake_llm(fail_marker=marker, fail_always=True)
    try:
        summarize_chunk_list(chunks, llm)
    except SummaryError as e:
        assert "Chunk 3" in str(e), str(e)
        print("PASS failure: stops with a clear message ->", e)
    else:
        raise AssertionError("expected SummaryError")


def api_error(error_class, status):
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    request = httpx.Request("POST", "http://example.com")
    response = httpx.Response(status, request=request)
    return error_class("fake error", response=response, body=None)


def test_api_errors_keep_their_type():
    import openai

    chunks = chunks_of(LONG_TEXT)
    cases = [(openai.AuthenticationError, 401), (openai.NotFoundError, 404),
             (openai.RateLimitError, 429)]
    for error_class, status in cases:
        def boom(prompt_value, error_class=error_class, status=status):
            raise api_error(error_class, status)
        try:
            summarize_chunk_list(chunks, RunnableLambda(boom))
        except error_class:
            pass                                     # main.py shows a friendly message for it
        except Exception as e:
            raise AssertionError(
                f"{error_class.__name__} became {type(e).__name__}")
        else:
            raise AssertionError("expected an error")
    print("PASS API errors: wrong key, missing model, and rate limit keep their own type")


def test_reduce_cannot_shrink():
    chunks = chunks_of(LONG_TEXT)
    old = config.REDUCE_LIMIT
    config.REDUCE_LIMIT = 300
    try:
        llm, _ = make_fake_llm(long_reduce=True)
        summarize_chunk_list(chunks, llm)
    except SummaryError as e:
        print("PASS no endless loop: stops with ->", str(e)[:60])
    else:
        raise AssertionError("expected SummaryError")
    finally:
        config.REDUCE_LIMIT = old


def test_config_helper():
    os.environ["TEST_BAD_NUMBER"] = "abc"
    os.environ["TEST_NEGATIVE"] = "-5"
    assert config._int("TEST_BAD_NUMBER", 7) == 7
    assert config._int("TEST_NEGATIVE", 7) == 7
    assert config._int("TEST_MISSING_NUMBER", 7) == 7
    print("PASS config: bad, negative, and missing values fall back to defaults")


def expect_load_error(path, label):
    try:
        load_document(path)
    except LoadError as e:
        print(f"PASS loader ({label}) ->", e)
    else:
        raise AssertionError(f"expected LoadError for {label}")


def test_loaders():
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)

        good = folder / "good.txt"
        good.write_text("Hello. Café, naïve, déjà vu.", encoding="utf-8")
        assert load_document(good)[0].page_content.startswith("Hello")
        print("PASS loader (good txt)")

        expect_load_error(folder / "missing.txt", "missing file")
        expect_load_error(folder / "notes.docx", "unsupported type")

        empty = folder / "empty.txt"
        empty.write_text("   \n", encoding="utf-8")
        expect_load_error(empty, "empty file")

        legacy = folder / "legacy.txt"
        legacy.write_bytes("caf\xe9".encode("cp1252"))
        expect_load_error(legacy, "not UTF-8")

        garbage = folder / "garbage.pdf"
        garbage.write_bytes(b"this is not a pdf")
        expect_load_error(garbage, "corrupted PDF")

        blank = folder / "blank.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with open(blank, "wb") as f:
            writer.write(f)
        expect_load_error(blank, "PDF with no text")

        locked = folder / "locked.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.encrypt("secret")
        with open(locked, "wb") as f:
            writer.write(f)
        expect_load_error(locked, "password-protected PDF")


if __name__ == "__main__":
    test_chunking()
    test_map_reduce_flow()
    test_hierarchical_reduce()
    test_retry_once()
    test_chunk_fails_twice()
    test_api_errors_keep_their_type()
    test_reduce_cannot_shrink()
    test_config_helper()
    test_loaders()
    print("\nAll tests passed.")
