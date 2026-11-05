import unittest
from io import BytesIO
import matplotlib.pyplot as plt
from docx import Document
from werkzeug.datastructures import FileStorage
from app import app, extract_document_text, preprocess, split_paragraphs, extract_triplets, analyze

class AnalyzerTests(unittest.TestCase):
    def test_split_paragraphs(self):
        self.assertEqual(split_paragraphs("First.\n\nSecond."), ["First.", "Second."])

    def test_preprocess_lowercase_and_tokens(self):
        result = preprocess("The Systems are analyzing DATA.")
        self.assertTrue(all(t == t.lower() for t in result["tokens"]))
        self.assertIn("data", result["tokens"])

    def test_extract_triplet(self):
        result = extract_triplets("The system uses data.")
        self.assertTrue(result)
        self.assertIn("subject", result[0])
        self.assertIn("relation", result[0])
        self.assertIn("object", result[0])

    def test_analysis_pair_scores(self):
        result = analyze("AI systems analyze data and make predictions.\n\nMachine learning uses data to improve predictions.")
        self.assertEqual(len(result["pairs"]), 1)
        self.assertGreaterEqual(result["pairs"][0]["score"], 0)
        self.assertLessEqual(result["pairs"][0]["score"], 1)

    def test_requires_two_paragraphs(self):
        with self.assertRaises(ValueError):
            analyze("Only one paragraph.")

    def test_uploads_text_document(self):
        with app.test_client() as client:
            response = client.post(
                "/",
                data={"document": (BytesIO(
                    b"Artificial intelligence analyzes data and finds patterns.\n\n"
                    b"Machine learning uses data to improve predictions."
                ), "notes.txt")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Paragraphs analyzed", response.data)
        self.assertIn(b"Artificial intelligence analyzes data", response.data)

    def test_uploads_and_analyzes_multiple_documents_separately(self):
        with app.test_client() as client:
            response = client.post(
                "/",
                data={"documents": [
                    (BytesIO(b"First file discusses science.\n\nSecond paragraph explains research."), "first.txt"),
                    (BytesIO(b"One paragraph covers music.\n\nAnother paragraph describes songs."), "second.txt"),
                ]},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Analysis results (2 items)", response.data)
        self.assertIn(b"first.txt", response.data)
        self.assertIn(b"second.txt", response.data)

    def test_one_invalid_file_does_not_prevent_other_files_from_being_analyzed(self):
        with app.test_client() as client:
            response = client.post(
                "/",
                data={"documents": [
                    (BytesIO(b"unsupported"), "notes.rtf"),
                    (BytesIO(b"First paragraph uses useful terms.\n\nSecond paragraph explains related ideas."), "valid.txt"),
                ]},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported document type", response.data)
        self.assertIn(b"valid.txt", response.data)
        self.assertIn(b"Paragraphs analyzed", response.data)

    def test_rejects_more_than_twenty_uploaded_files(self):
        files = [
            (BytesIO(b"First paragraph.\n\nSecond paragraph."), f"file-{index}.txt")
            for index in range(21)
        ]
        with app.test_client() as client:
            response = client.post(
                "/",
                data={"documents": files},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload no more than 20 files at a time", response.data)
        self.assertNotIn(b"Analysis results", response.data)

    def test_rejects_unsupported_document_type(self):
        with app.test_client() as client:
            response = client.post(
                "/",
                data={"document": (BytesIO(b"not a supported document"), "notes.rtf")},
                content_type="multipart/form-data",
            )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Unsupported document type", response.data)

    def test_extracts_docx_document(self):
        document = Document()
        document.add_paragraph("First paragraph uses relevant terms.")
        document.add_paragraph("Second paragraph analyzes related terms.")
        buffer = BytesIO()
        document.save(buffer)
        buffer.seek(0)
        text = extract_document_text(FileStorage(stream=buffer, filename="notes.docx"))
        self.assertIn("First paragraph", text)
        self.assertIn("Second paragraph", text)

    def test_extracts_pdf_document(self):
        figure, axis = plt.subplots()
        axis.text(0.1, 0.8, "First paragraph describes useful concepts.")
        axis.text(0.1, 0.6, "Second paragraph explains related concepts.")
        buffer = BytesIO()
        figure.savefig(buffer, format="pdf")
        plt.close(figure)
        buffer.seek(0)
        text = extract_document_text(FileStorage(stream=buffer, filename="notes.pdf"))
        self.assertIn("First paragraph", text)
        self.assertIn("Second paragraph", text)

if __name__ == "__main__":
    unittest.main()
