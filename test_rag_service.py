import io
import json
import unittest
from unittest.mock import patch
from urllib.error import URLError

from services.rag_service import OllamaGenerationError, SlideRAGService


class SlideRAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.rag = SlideRAGService()
        self.rag.index_presentation([
            {"slide": 1, "text": "The project uses pure text slides for analysis."}
        ])

    @patch("services.rag_service.urlopen")
    def test_answer_query_uses_ollama_and_keeps_slide_sources(self, mock_urlopen):
        mock_urlopen.return_value = io.BytesIO(
            json.dumps({"response": "The project uses pure text slides."}).encode("utf-8")
        )

        result = self.rag.answer_query("pure text slides")

        self.assertIn("The project uses pure text slides.", result["answer"])
        self.assertIn("Sources: Slide 1", result["answer"])
        self.assertEqual([citation["slide"] for citation in result["citations"]], [1])
        request = mock_urlopen.call_args.args[0]
        request_data = json.loads(request.data)
        self.assertEqual(request_data["model"], self.rag.ollama_model)
        self.assertIn("pure text slides", request_data["prompt"])

    @patch("services.rag_service.urlopen")
    def test_abstention_does_not_call_ollama(self, mock_urlopen):
        result = self.rag.answer_query("Bitcoin price in 2030")

        self.assertTrue(result["abstained"])
        mock_urlopen.assert_not_called()

    @patch("services.rag_service.urlopen", side_effect=URLError("connection refused"))
    def test_ollama_connection_error_is_explicit(self, mock_urlopen):
        with self.assertRaises(OllamaGenerationError):
            self.rag.answer_query("pure text slides")


if __name__ == "__main__":
    unittest.main()
