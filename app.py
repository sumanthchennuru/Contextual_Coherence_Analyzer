from flask import Flask, render_template, request, send_file, jsonify
from docx import Document
from pypdf import PdfReader
from pathlib import Path
from collections import Counter
import csv, io, json, re, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
MAX_UPLOAD_FILES = 20
STOPWORDS = set(ENGLISH_STOP_WORDS)

# Transparent, rule-based relation vocabulary. This is an educational extractor,
# not a full syntactic parser.
VERBS = {
    "is":"describes", "are":"describes", "was":"describes", "were":"describes",
    "be":"describes", "has":"possesses", "have":"possesses", "had":"possesses",
    "uses":"uses", "use":"uses", "used":"uses", "contains":"contains",
    "contain":"contains", "includes":"includes", "include":"includes",
    "creates":"creates", "create":"creates", "created":"creates",
    "supports":"supports", "support":"supports", "improves":"improves",
    "improve":"improves", "analyzes":"analyzes", "analyze":"analyzes",
    "analyses":"analyzes", "generates":"generates", "generate":"generates",
    "converts":"converts", "convert":"converts", "provides":"provides",
    "provide":"provides", "enables":"enables", "enable":"enables",
    "produces":"produces", "produce":"produces", "causes":"causes",
    "cause":"causes", "helps":"helps", "help":"helps", "stores":"stores",
    "store":"stores", "connects":"connects", "connect":"connects",
    "reduces":"reduces", "reduce":"reduces", "increases":"increases",
    "increase":"increases", "detects":"detects", "detect":"detects",
    "identifies":"identifies", "identify":"identifies", "makes":"makes",
    "make":"makes", "offers":"offers", "offer":"offers", "powers":"powers",
    "power":"powers", "consists":"consists of", "includes":"includes"
}
DETERMINERS = {"the", "a", "an", "this", "these", "those"}
PRONOUNS = {"it", "they", "we", "he", "she", "i", "you", "them", "us"}

def split_paragraphs(text):
    return [p.strip() for p in re.split(r"\n\s*\n+", (text or "").strip()) if p.strip()]

def _flatten_structured_text(value):
    """Turn a JSON or CSV-like payload into readable text for analysis."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key, item in value.items():
            text = _flatten_structured_text(item)
            if text:
                parts.append(f"{key}: {text}")
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        return "\n".join(_flatten_structured_text(item) for item in value if _flatten_structured_text(item).strip())
    return str(value)


def extract_document_text(upload):
    """Extract readable text from supported document uploads."""
    suffix = Path(upload.filename or "").suffix.lower()
    try:
        if suffix in {".txt", ".md", ".csv"}:
            raw = upload.read()
            if suffix == ".csv":
                try:
                    text = raw.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = raw.decode("latin-1")
                rows = []
                for row in csv.reader(io.StringIO(text)):
                    cleaned = [cell.strip() for cell in row if cell and cell.strip()]
                    if cleaned:
                        rows.append(" ".join(cleaned))
                text = "\n\n".join(rows) or text
            else:
                text = raw.decode("utf-8-sig")
        elif suffix in {".html", ".htm"}:
            raw = upload.read().decode("utf-8-sig", errors="ignore")
            text = re.sub(r"<script.*?</script>", " ", raw, flags=re.I | re.S)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"\s+", " ", text)
        elif suffix == ".json":
            payload = json.loads(upload.read().decode("utf-8-sig"))
            text = _flatten_structured_text(payload)
        elif suffix == ".pdf":
            reader = PdfReader(upload.stream)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
        elif suffix == ".docx":
            document = Document(upload.stream)
            sections = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            sections.extend(
                "\n".join(cell.text for cell in row.cells)
                for table in document.tables
                for row in table.rows
            )
            text = "\n\n".join(sections)
        else:
            raise ValueError("Unsupported document type. Upload a .txt, .md, .csv, .json, .html, .pdf, or .docx file.")
    except UnicodeDecodeError as exc:
        raise ValueError("Text documents must use UTF-8 encoding.") from exc
    except ValueError:
        raise
    except Exception as exc:
        if suffix not in {".pdf", ".docx"}:
            raise
        app.logger.exception("Could not read uploaded %s document", suffix)
        raise ValueError(f"Could not read this {suffix} document. Check that the file is valid and try again.") from exc

    if not text.strip():
        raise ValueError("No readable text was found in the uploaded document.")
    return text

def simple_lemma(word):
    """Small fallback lemmatizer; deliberately conservative and explainable."""
    w = word.lower()
    irregular = {"children":"child", "men":"man", "women":"woman", "people":"person",
                 "better":"good", "worse":"bad", "was":"be", "were":"be", "is":"be", "are":"be"}
    if w in irregular:
        return irregular[w]
    if len(w) > 5 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 5 and w.endswith("ing"):
        base = w[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if len(w) > 4 and w.endswith("ed"):
        return w[:-2]
    if len(w) > 4 and w.endswith("es") and w not in VERBS:
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss") and w not in VERBS:
        return w[:-1]
    return w

def preprocess(text):
    """Module 1: return original text, cleaned text, and tokens."""
    lowered = (text or "").lower()
    normalized = re.sub(r"[^a-z0-9\s'-]", " ", lowered)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    raw_tokens = re.findall(r"[a-z]+(?:'[a-z]+)?|[0-9]+", normalized)
    tokens = []
    for token in raw_tokens:
        if token in STOPWORDS or len(token) < 2:
            continue
        lemma = simple_lemma(token)
        if lemma and lemma not in STOPWORDS:
            tokens.append(lemma)
    return {"original": text or "", "cleaned": normalized, "tokens": tokens}

def _trim_phrase(words):
    words = list(words)
    while words and words[0].lower() in DETERMINERS:
        words.pop(0)
    while words and words[-1].lower() in {"a", "an", "the", "and", "or", "but"}:
        words.pop()
    return " ".join(words).strip(" ,;:")

def extract_triplets(paragraph):
    """Module 2: heuristic subject-relation-object extraction."""
    triplets = []
    sentences = re.split(r"(?<=[.!?])\s+", (paragraph or "").strip())
    for sentence in sentences:
        words = re.findall(r"[A-Za-z][A-Za-z'-]*", sentence)
        lower = [w.lower() for w in words]
        for i, word in enumerate(lower):
            if word not in VERBS or i == 0 or i >= len(words)-1:
                continue
            # Choose a short subject phrase immediately before the relation.
            start = max(0, i-3)
            subject_words = words[start:i]
            while subject_words and subject_words[0].lower() in DETERMINERS | PRONOUNS:
                subject_words = subject_words[1:]
            subject = _trim_phrase(subject_words)
            # Choose a short object phrase after the relation.
            object_words = words[i+1:min(len(words), i+4)]
            obj = _trim_phrase(object_words)
            if not subject:
                # Keep a pronoun subject when the sentence starts with one.
                subject = words[i-1]
            if subject and obj:
                item = {"subject": subject, "relation": VERBS[word], "object": obj,
                        "sentence": sentence.strip()}
                if item not in triplets:
                    triplets.append(item)
                break
    return triplets

def jaccard(a, b):
    a, b = set(a), set(b)
    union = a | b
    return len(a & b) / len(union) if union else 0.0

def entities_from_triplets(triplets):
    entities = set()
    for item in triplets:
        entities.add(item["subject"].lower().strip())
        entities.add(item["object"].lower().strip())
    return {x for x in entities if x}

def _image_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")

def make_heatmap(scores):
    n = len(scores)
    fig, ax = plt.subplots(figsize=(max(4.5, n * .85), max(3.5, n * .7)))
    im = ax.imshow(scores, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(n), [f"P{i+1}" for i in range(n)])
    ax.set_yticks(range(n), [f"P{i+1}" for i in range(n)])
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{scores[i][j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Paragraph consistency heatmap")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    return _image_b64(fig)

def make_graph(triplets):
    graph = nx.MultiDiGraph()
    for group in triplets:
        for item in group:
            graph.add_node(item["subject"])
            graph.add_node(item["object"])
            graph.add_edge(item["subject"], item["object"], label=item["relation"])
    if not graph.number_of_nodes():
        return None
    fig, ax = plt.subplots(figsize=(9, 5.2))
    pos = nx.spring_layout(graph, seed=17)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_size=1900, node_color="#dbeafe",
                           edgecolors="#1d4ed8", linewidths=1.0)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(graph, pos, ax=ax, arrows=True, arrowsize=16,
                           edge_color="#64748b", connectionstyle="arc3,rad=0.08")
    labels = {(u, v): d.get("label", "") for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels, ax=ax, font_size=7)
    ax.set_title("Extracted relationship graph")
    ax.axis("off")
    fig.tight_layout()
    return _image_b64(fig)

def _top_terms(tokens, limit=5):
    return [word for word, _ in Counter(tokens).most_common(limit)]


def _build_summary(overall, drift, token_lists):
    top_terms = _top_terms([term for toks in token_lists for term in toks], limit=8)
    top_phrase = ", ".join(top_terms) if top_terms else "no dominant terms"
    if overall >= 0.50:
        summary = (
            "The document stays largely coherent across paragraphs, with strong overlap in "
            f"key ideas such as {top_phrase}."
        )
        label = "strong coherence"
    elif overall >= 0.25:
        summary = (
            "The document is moderately coherent; some paragraphs share related ideas, but the theme "
            f"is not fully stable across the full text ({top_phrase})."
        )
        label = "moderate coherence"
    else:
        summary = (
            "The document shows weak contextual continuity, suggesting repeated topic drift or a broad shift "
            f"in focus away from {top_phrase}."
        )
        label = "low coherence"

    if drift:
        drift_note = (
            f"There are {len(drift)} low-score transition(s) that may indicate topic drift between "
            + ", ".join(f"P{entry['a']}→P{entry['b']}" for entry in drift[:3])
            + "."
        )
    else:
        drift_note = "No consecutive paragraph pair fell below the topic-drift threshold."
    return {"summary": f"{summary} {drift_note}", "label": label, "top_terms": top_terms}


def analyze(text):
    paragraphs = split_paragraphs(text)
    if len(paragraphs) < 2:
        raise ValueError("Enter at least two paragraphs, separated by a blank line.")
    if len(paragraphs) > 20:
        raise ValueError("For this prototype, please analyze 20 paragraphs or fewer.")
    processed = [preprocess(p) for p in paragraphs]
    token_lists = [x["tokens"] for x in processed]
    cleaned_texts = [" ".join(t) for t in token_lists]

    # TF-IDF can fail when all terms are removed; handle this explicitly.
    try:
        matrix = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b").fit_transform(cleaned_texts)
        cosine_matrix = cosine_similarity(matrix)
    except ValueError:
        cosine_matrix = [[0.0] * len(paragraphs) for _ in paragraphs]

    triplets = [extract_triplets(p) for p in paragraphs]
    entities = [entities_from_triplets(t) for t in triplets]
    n = len(paragraphs)
    scores = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            cosine = float(cosine_matrix[i][j])
            jac = jaccard(token_lists[i], token_lists[j])
            entity_overlap = jaccard(entities[i], entities[j])
            score = 0.5*cosine + 0.3*jac + 0.2*entity_overlap
            verdict = "Consistent" if score >= .50 else ("Moderate" if score >= .25 else "Inconsistent")
            shared = sorted(set(token_lists[i]) & set(token_lists[j]))
            pairs.append({"a": i+1, "b": j+1, "cosine": cosine, "jaccard": jac,
                          "triplet": entity_overlap, "score": score, "verdict": verdict,
                          "shared": shared})
            scores[i][j] = scores[j][i] = score
    drift = [{"a": i+1, "b": i+2, "score": scores[i][i+1]}
             for i in range(n-1) if scores[i][i+1] < .25]
    overall = sum(p["score"] for p in pairs) / len(pairs) if pairs else 0.0
    overall_label = "Consistent" if overall >= .50 else ("Moderate" if overall >= .25 else "Inconsistent")
    keywords = []
    paragraph_themes = []
    for i, toks in enumerate(token_lists):
        common = [word for word, _ in Counter(toks).most_common(8)]
        keywords.append({"paragraph": i+1, "words": common})
        paragraph_themes.append({
            "paragraph": i+1,
            "theme": ", ".join(common[:3]) if common else "No clear theme",
            "keywords": common[:5],
        })
    summary = _build_summary(overall, drift, token_lists)
    highest_pair = max(pairs, key=lambda p: p["score"]) if pairs else None
    return {
        "paragraphs": paragraphs, "processed": processed, "triplets": triplets,
        "pairs": pairs, "scores": scores, "drift": drift, "overall": overall,
        "overall_label": overall_label, "keywords": keywords,
        "heatmap": make_heatmap(scores), "graph_img": make_graph(triplets),
        "summary": summary["summary"], "summary_label": summary["label"],
        "document_themes": summary["top_terms"], "paragraph_themes": paragraph_themes,
        "highest_pair": highest_pair,
    }

@app.route("/", methods=["GET", "POST"])
def index():
    text_input, results, error = "", [], None
    if request.method == "POST":
        text_input = request.form.get("text", "")
        uploaded_files = [
            upload for field in ("documents", "document")
            for upload in request.files.getlist(field)
            if upload and upload.filename
        ]
        if len(uploaded_files) > MAX_UPLOAD_FILES:
            error = f"Upload no more than {MAX_UPLOAD_FILES} files at a time."
        elif uploaded_files:
            for uploaded_file in uploaded_files:
                item = {"filename": uploaded_file.filename, "text": "", "result": None, "error": None}
                try:
                    item["text"] = extract_document_text(uploaded_file)
                    item["result"] = analyze(item["text"])
                except ValueError as exc:
                    item["error"] = str(exc)
                except Exception:
                    app.logger.exception("Analysis failed for uploaded file %s", uploaded_file.filename)
                    item["error"] = "Analysis failed unexpectedly. Check the terminal for details."
                results.append(item)
        else:
            try:
                result = analyze(text_input)
                results.append({
                    "filename": "Pasted text", "text": text_input,
                    "result": result, "error": None,
                })
            except ValueError as exc:
                error = str(exc)
            except Exception:
                app.logger.exception("Analysis failed")
                error = "Analysis failed unexpectedly. Check the terminal for details."
    return render_template("index.html", text_input=text_input, results=results, error=error)

@app.route("/download/csv", methods=["POST"])
def download_csv():
    text_input = request.form.get("text", "")
    try:
        result = analyze(text_input)
    except ValueError as exc:
        return str(exc), 400
    stream = io.StringIO()
    writer = csv.writer(stream)
    writer.writerow(["Paragraph A", "Paragraph B", "Cosine Similarity", "Jaccard Overlap",
                     "Entity Overlap", "Final Score", "Verdict", "Shared Keywords"])
    for p in result["pairs"]:
        writer.writerow([p["a"], p["b"], f'{p["cosine"]:.4f}', f'{p["jaccard"]:.4f}',
                         f'{p["triplet"]:.4f}', f'{p["score"]:.4f}', p["verdict"],
                         ", ".join(p["shared"])])
    mem = io.BytesIO(stream.getvalue().encode("utf-8-sig"))
    mem.seek(0)
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="coherence_report.csv")


@app.route("/download/json", methods=["POST"])
def download_json():
    text_input = request.form.get("text", "")
    try:
        result = analyze(text_input)
    except ValueError as exc:
        return str(exc), 400

    payload = {
        "summary": result["summary"],
        "overall_score": result["overall"],
        "overall_label": result["overall_label"],
        "paragraphs": result["paragraphs"],
        "pairs": result["pairs"],
        "drift": result["drift"],
        "document_themes": result["document_themes"],
    }
    mem = io.BytesIO(json.dumps(payload, indent=2).encode("utf-8"))
    mem.seek(0)
    return send_file(mem, mimetype="application/json", as_attachment=True, download_name="coherence_report.json")


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    payload = request.get_json(silent=True) or {}
    text_input = payload.get("text", "")
    if not text_input.strip():
        return jsonify({"error": "Text input is required."}), 400
    try:
        result = analyze(text_input)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({
        "status": "ok",
        "overall_score": result["overall"],
        "overall_label": result["overall_label"],
        "summary": result["summary"],
        "document_themes": result["document_themes"],
        "pairs": result["pairs"],
        "drift": result["drift"],
    })


@app.route("/health")
def health():
    return {"status": "ok", "application": "Contextual Coherence Analyzer"}

if __name__ == "__main__":
    app.run(debug=True)
