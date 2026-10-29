import importlib.util, json, tempfile, unittest
from pathlib import Path
from PIL import Image
from trustretina.data.validation import file_hash


class ReviewRendererTests(unittest.TestCase):
    def test_rendering_never_silently_clears_a_case_and_rejects_image_mutation(self):
        source = (
            Path(__file__).resolve().parents[1] / "scripts/render_duplicate_review.py"
        )
        spec = importlib.util.spec_from_file_location("review_renderer", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "synthetic.png"
            Image.new("RGB", (60, 40), (80, 40, 20)).save(image)
            endpoint = {
                "patient_id": "synthetic",
                "eye": "left",
                "path": "synthetic.png",
                "sha256": file_hash(image),
            }
            cases = root / "cases.json"
            cases.write_text(
                json.dumps(
                    {
                        "material_cases": [
                            {"a": endpoint, "b": endpoint, "review_index": 0}
                        ]
                    }
                )
            )
            digest = file_hash(cases)
            result = module.render(cases, root, root / "review", cases_sha256=digest)
            self.assertFalse(result["automatic_clearance"])
            self.assertEqual(result["pages"][0]["status"], "unresolved")
            self.assertTrue((root / "review/case-000.jpg").is_file())
            Image.new("RGB", (60, 40)).save(image)
            with self.assertRaises(ValueError):
                module.render(cases, root, root / "changed", cases_sha256=digest)
