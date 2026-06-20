from __future__ import annotations

import unittest

from src.pipeline import candidate_recipes


class CandidateRecipeTests(unittest.TestCase):
    def test_raw_count_grid_covers_conservative_and_aggressive_weights(self) -> None:
        recipes = candidate_recipes(has_count_branch=True, default_count_weight=0.45)
        names = [str(recipe["name"]) for recipe in recipes]

        self.assertIn("raw_count_0p25000", names)
        self.assertIn("raw_count_0p55000", names)
        self.assertIn("raw_count_0p75000", names)
        self.assertIn("raw_count_0p80000", names)
        self.assertIn("raw_count_0p90000", names)
        self.assertIn("raw_count_1p00000", names)
        self.assertEqual(names.count("raw_count_0p45000"), 1)


if __name__ == "__main__":
    unittest.main()
