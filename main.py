import os
import sys
import time
from pathlib import Path

import openai

import config
from loaders import LoadError, load_document
from logger_setup import get_logger
from summarizer import (
    SummaryError,
    merge_documents,
    split_text,
    summarize_chunk_list,
)

sys.stdout.reconfigure(encoding="utf-8")
logger = get_logger("doc_summarizer", "doc_summarizer.log")

OUTPUT_DIR = Path("output")


def friendly_api_error(error):
    if isinstance(error, openai.AuthenticationError):
        return "The API key is incorrect. Please check your .env file."
    if isinstance(error, openai.RateLimitError):
        return "Too many requests, or your credits have run out. Please try again later."
    if isinstance(error, openai.NotFoundError):
        return "Model not found. Please check OPENAI_MODEL in your .env file."
    if isinstance(error, openai.BadRequestError):
        return "The request was rejected. Please check the model or the chunk size."
    if isinstance(error, openai.APITimeoutError):
        return "The server is not responding. Please try again later."
    if isinstance(error, openai.APIConnectionError):
        return "Cannot connect to the internet."
    return "There was a problem with the API. Details are in the log file."


def save_summary(source_path, summary):
    OUTPUT_DIR.mkdir(exist_ok=True)
    target = OUTPUT_DIR / f"{Path(source_path).stem}_summary.txt"
    target.write_text(summary + "\n", encoding="utf-8")
    return target


def main():
    if not os.getenv("OPENAI_API_KEY") or not config.MODEL:
        raise SystemExit(
            "Set OPENAI_API_KEY and OPENAI_MODEL in your .env file first.")

    path = sys.argv[1] if len(sys.argv) > 1 else input(
        "File path (.txt or .pdf): ").strip().strip('"')

    # 1. Load
    try:
        docs = load_document(path)
    except LoadError as e:
        print(e)
        return
    except Exception:
        logger.exception("Could not load the file")
        print("There was a problem reading the file. Details are in the log file.")
        return

    # 2. Chunk
    document = merge_documents(docs)
    chunks = split_text(document)
    calls = len(chunks) + 1
    print(
        f"Loaded: {len(document.page_content)} characters, {len(chunks)} chunks, about {calls} AI calls.")
    logger.info("Loaded %s: %s characters, %s chunks", Path(path).name,
                len(document.page_content), len(chunks))

    # 3. Cost guard
    if len(chunks) > config.MAX_CHUNKS:
        answer = input(
            f"This is very large ({len(chunks)} chunks). Continue? (y/n): ").strip().lower()
        if answer != "y":
            logger.info("User cancelled a large job (%s chunks)", len(chunks))
            print("Cancelled.")
            return

    # 4. Summarize
    start = time.perf_counter()
    try:
        summary = summarize_chunk_list(chunks)
    except SummaryError as e:
        logger.warning("Summary failed: %s", e)
        print("Could not create the summary:", e)
        return
    except openai.OpenAIError as e:
        logger.error("API problem: %s", type(e).__name__)
        print(friendly_api_error(e))
        return
    except Exception:
        logger.exception("Unexpected error while summarizing")
        print("An unexpected problem occurred. Details are in the log file.")
        return

    # 5. Show and save
    seconds = time.perf_counter() - start
    target = save_summary(path, summary)
    logger.info("Summary saved to %s in %.1fs", target, seconds)
    print("\n" + summary + "\n")
    print(f"Saved: {target} ({seconds:.1f} seconds)")


if __name__ == "__main__":
    main()
