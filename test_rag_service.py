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

    @patch("services.rag_service.urlopen")
    def test_summary_uses_all_slides_without_lexical_retrieval(self, mock_urlopen):
        self.rag.index_presentation([
            {"slide": 1, "text": "First topic: renewable energy."},
            {"slide": 2, "text": "Second topic: battery storage."}
        ])
        mock_urlopen.return_value = io.BytesIO(
            json.dumps({"response": "The deck covers renewable energy and battery storage."}).encode("utf-8")
        )

        result = self.rag.answer_query("summerise this presentation")

        self.assertIn("renewable energy and battery storage", result["answer"])
        self.assertIn("Sources: Slide 1, Slide 2", result["answer"])
        self.assertEqual([citation["slide"] for citation in result["citations"]], [1, 2])
        request_data = json.loads(mock_urlopen.call_args.args[0].data)
        self.assertIn("renewable energy", request_data["prompt"])
        self.assertIn("battery storage", request_data["prompt"])

    @patch("services.rag_service.urlopen")
    def test_long_presentation_summary_is_reduced_in_multiple_chunks(self, mock_urlopen):
        self.rag.index_presentation([
            {"slide": 1, "text": "renewable energy " * 200},
            {"slide": 2, "text": "battery storage " * 200}
        ])
        response_body = json.dumps({"response": "A concise partial summary."}).encode("utf-8")
        mock_urlopen.side_effect = lambda *args, **kwargs: io.BytesIO(response_body)

        result = self.rag.answer_query("summarize the presentation")

        self.assertIn("Sources: Slide 1, Slide 2", result["answer"])
        self.assertGreater(mock_urlopen.call_count, 1)
        prompts = [
            json.loads(call.args[0].data)["prompt"]
            for call in mock_urlopen.call_args_list
        ]
        self.assertTrue(any("renewable energy" in prompt for prompt in prompts))
        self.assertTrue(any("battery storage" in prompt for prompt in prompts))

    @patch("services.rag_service.urlopen", side_effect=URLError("connection refused"))
    def test_ollama_connection_error_is_explicit(self, mock_urlopen):
        with self.assertRaises(OllamaGenerationError):
            self.rag.answer_query("pure text slides")


if __name__ == "__main__":
    unittest.main()
