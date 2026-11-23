# AI Document Summarizer

Load a PDF or TXT file, cut it into chunks, and get a short Bangla summary.

## Setup

```
python -m venv venv
venv\Scripts\activate
python -m pip install langchain langchain-openai langchain-text-splitters pypdf python-dotenv openai
copy .env.example .env
python -m pip freeze > requirements.txt
```

Open `.env` and fill in `OPENAI_API_KEY` and `OPENAI_MODEL`.

## Run

```
python main.py sample_report.txt
python main.py "C:\path\to\your file.pdf"
```

The summary is printed and saved in the `output` folder.

## Test without an API key (free)

```
python test_summarizer.py
```

## How it works

1. `loaders.py` reads the file (TXT or PDF) into Documents.
2. `summarizer.py` joins the pages and splits the text into chunks.
3. Map: every chunk is summarized on its own.
4. Reduce: the chunk summaries are combined into one final summary.
5. `main.py` shows the summary and saves it.

## Settings in .env (all optional)

| Name            | Default | Meaning                                              |
| --------------- | ------- | ---------------------------------------------------- |
| CHUNK_SIZE      | 3000    | Characters per chunk                                 |
| CHUNK_OVERLAP   | 300     | Characters repeated between chunks                   |
| MAX_CHUNKS      | 30      | Ask before using more AI calls than this             |
| MAX_CONCURRENCY | 3       | Chunks sent at the same time                         |
| REDUCE_LIMIT    | 12000   | Maximum characters of summaries combined in one call |
