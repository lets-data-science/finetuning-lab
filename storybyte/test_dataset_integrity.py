"""Regression checks for StoryByte's protected-name and eval-slice contract.

This test does not regenerate data or model outputs. It inspects the frozen
shipped rows and recomputes documented aggregates from persisted eval details.
"""
from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path


STORYBYTE_DIR = Path(__file__).resolve().parent
LAB_DIR = STORYBYTE_DIR.parent


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "storybyte_dataset_builder", STORYBYTE_DIR / "01_build_requests_dataset.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BUILDER = load_builder()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class DatasetIntegrityTest(unittest.TestCase):
    def test_future_guard_rejects_every_protected_name(self) -> None:
        for name in BUILDER.UNSEEN_NAMES:
            self.assertEqual(BUILDER.protected_name_counts(f"A friend named {name}."), {name: 1})
            self.assertEqual(BUILDER.protected_name_counts(f"{name.lower()} returned."), {name: 1})
            self.assertEqual(BUILDER.protected_name_counts(f"prefix{name}suffix"), {})

        with self.assertRaisesRegex(AssertionError, "protected-name leak"):
            BUILDER.assert_no_protected_names([
                {"request": "Write a dog story.", "story": "A side character was Bella.", "name": "Max"}
            ])

    def test_shipped_rows_match_the_documented_bella_exception(self) -> None:
        train = load_jsonl(LAB_DIR / "data" / "requests_train.jsonl")
        val = load_jsonl(LAB_DIR / "data" / "requests_val.jsonl")
        self.assertEqual((len(train), len(val)), (571, 63))

        for row in train + val:
            self.assertEqual(BUILDER.protected_name_counts(row["request"]), {})
            self.assertEqual(BUILDER.protected_name_counts(row["name"]), {})

        hits = []
        for row in train:
            counts = BUILDER.protected_name_counts(row["story"])
            if counts:
                hits.append((row["animal"], row["name"], counts))

        self.assertEqual(len(hits), 5)
        self.assertEqual(sum(sum(counts.values()) for _, _, counts in hits), 23)
        self.assertEqual(
            Counter((animal, name, counts.get("Bella", 0)) for animal, name, counts in hits),
            Counter({
                ("duck", "Jow", 6): 1,
                ("bear", "Bob", 2): 1,
                ("dog", "Max", 6): 1,
                ("dog", "Timmy", 3): 1,
                ("bear", "Remy", 6): 1,
            }),
        )
        self.assertTrue(all(set(counts) == {"Bella"} for _, _, counts in hits))
        self.assertFalse(any(BUILDER.protected_name_counts(row["story"]) for row in val))

    def test_gold_has_three_requests_per_protected_name(self) -> None:
        gold = json.loads((LAB_DIR / "data" / "gold_requests.json").read_text())
        protected_rows = [row for row in gold if row["name_cond"] == "unseen_name"]
        self.assertEqual(len(protected_rows), 30)
        self.assertEqual(Counter(row["name"] for row in protected_rows), Counter({name: 3 for name in BUILDER.UNSEEN_NAMES}))

    def test_strict_unseen_slice_reaggregates_from_frozen_eval_rows(self) -> None:
        expected = {
            "base": {"ok_character": 29, "ok_name": 65, "ok_format": 70, "ok_full": 23},
            "sft": {"ok_character": 75, "ok_name": 45, "ok_format": 71, "ok_full": 38},
        }
        for model, counts in expected.items():
            rows = json.loads((LAB_DIR / "results" / f"eval_detail_{model}.json").read_text())
            held_out_requested = [row for row in rows if row["name_cond"] == "unseen_name"]
            strict_clean = [row for row in held_out_requested if row["name"] != "Bella"]
            self.assertEqual(len(held_out_requested), 90)
            self.assertEqual(len(strict_clean), 81)
            self.assertEqual(
                {key: sum(bool(row[key]) for row in strict_clean) for key in counts},
                counts,
            )


if __name__ == "__main__":
    unittest.main()
