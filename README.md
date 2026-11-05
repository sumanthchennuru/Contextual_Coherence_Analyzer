# Contextual Coherence Analyzer

A Flask web application that estimates contextual consistency across multiple paragraphs and displays candidate subject–relation–object triplets.

## Features
- Module 1: lowercase normalization, punctuation cleanup, tokenization, stopword removal, and a transparent lightweight lemmatizer.
- Module 2: rule-based candidate subject–relation–object extraction.
- Module 3: TF-IDF cosine similarity, Jaccard word overlap, entity overlap, weighted final scores, consistency labels, and consecutive-paragraph topic-drift alerts.
- Module 4: score table, heatmap, relationship graph, keywords, document themes, and CSV/JSON export options.
- Added a JSON analysis API at `/api/analyze` for programmatic analysis and richer reporting.
- Analyze pasted text or upload up to 20 TXT, Markdown, CSV, JSON, HTML, PDF, or DOCX documents in one batch (up to 100 MB total). Each uploaded document gets its own results and exports.

## Setup (Windows PowerShell)
Open the folder that contains `app.py` and `requirements.txt` in VS Code. In Terminal:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:5000 in your browser. Leave the terminal running while using the site.

If PowerShell activation works in your setup, you may activate `.venv` and use `python app.py`; activation is not required when using the explicit `.venv` Python path above.

Paste text or upload up to 20 `.txt`, `.md`, `.csv`, `.json`, `.html`, `.pdf`, or `.docx` files (up to 100 MB total). Each file is analyzed independently and can contain up to 20 paragraphs. Documents must contain selectable/readable text; scanned PDFs need OCR before they can be analyzed.

## Scoring formula
`Final = 0.5 × cosine + 0.3 × Jaccard + 0.2 × entity overlap`

- Consistent: score >= 0.50
- Moderate: 0.25 <= score < 0.50
- Inconsistent: score < 0.25
- Possible topic drift: consecutive-pair score < 0.25

## Tests
Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Important limitations
The relationship extractor is heuristic, not a full dependency parser. It can produce imperfect triplets, especially with complex grammar, passive voice, and pronouns. Similarity scores estimate overlap and do not prove logical correctness or factual consistency. The Flask development server is for local testing, not production deployment.
