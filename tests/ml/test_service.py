from io import BytesIO
import unittest
from fastapi.testclient import TestClient
from PIL import Image
from ml.inference.service import create_app, normalize_tiles


class FakePredictor:
    version = "contract-test-only"
    calls = 0
    def predict(self, image):
        self.calls += 1
        return [{"classId": "1m", "score": .9, "bbox": [0, 0, 20, 30]}]


class ServiceTests(unittest.TestCase):
    def test_real_image_decode_and_response_contract(self):
        predictor = FakePredictor()
        client = TestClient(create_app(predictor))
        encoded = BytesIO()
        Image.new("RGB", (100, 80), "white").save(encoded, format="JPEG")
        response = client.post("/recognize", files={"image": ("photo.jpg", encoded.getvalue(), "image/jpeg")})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["image"], {"width": 100, "height": 80})
        self.assertEqual(result["modelVersion"], "contract-test-only")
        self.assertEqual(predictor.calls, 1)

    def test_invalid_bytes_do_not_invoke_model(self):
        predictor = FakePredictor()
        client = TestClient(create_app(predictor))
        response = client.post("/recognize", files={"image": ("fake.jpg", b"not an image", "image/jpeg")})
        self.assertEqual(response.status_code, 415)
        self.assertEqual(predictor.calls, 0)

    def test_duplicate_image_fields_are_rejected(self):
        client = TestClient(create_app(FakePredictor()))
        response = client.post("/recognize", files=[("image", ("a", b"a")), ("image", ("b", b"b"))])
        self.assertEqual(response.status_code, 400)

    def test_clipping_and_retake_without_forcing_fourteen(self):
        tiles, reasons = normalize_tiles([{"classId": "back", "score": .99, "bbox": [-5, -5, 20, 30]}], 100, 100, .7)
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0]["bbox"], [0, 0, 15, 25])
        self.assertTrue(tiles[0]["uncertain"])
        self.assertTrue(reasons)


if __name__ == "__main__":
    unittest.main()
