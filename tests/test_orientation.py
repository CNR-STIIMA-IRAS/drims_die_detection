#!/usr/bin/env python3
"""
Unit tests for die_orientation module.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from drims_die_detection.die_orientation import resolve_die_orientation


class TestDieOrientation(unittest.TestCase):
    """Tests for resolve_die_orientation."""

    def test_top_face_only(self):
        res = resolve_die_orientation(top_pips=4)
        self.assertFalse(res["is_fully_determined"])
        self.assertEqual(res["top_pips"], 4)
        self.assertEqual(res["bottom_pips"], 3)
        self.assertIsNone(res["x_pos_pips"])

    def test_top_and_primary_lateral(self):
        # Top=4, Front(+X)=1 -> Right(+Y) should be 5
        res = resolve_die_orientation(top_pips=4, x_pos_pips=1)
        self.assertTrue(res["is_fully_determined"])
        self.assertEqual(res["top_pips"], 4)
        self.assertEqual(res["bottom_pips"], 3)
        self.assertEqual(res["x_pos_pips"], 1)
        self.assertEqual(res["x_neg_pips"], 6)
        self.assertEqual(res["y_pos_pips"], 5)
        self.assertEqual(res["y_neg_pips"], 2)

    def test_top_and_two_laterals(self):
        # Top=1, Front(+X)=2 -> Right(+Y) should be 3
        res = resolve_die_orientation(top_pips=1, x_pos_pips=2, y_pos_pips=3)
        self.assertTrue(res["is_fully_determined"])
        self.assertEqual(res["face_map"]["+Z"], 1)
        self.assertEqual(res["face_map"]["-Z"], 6)
        self.assertEqual(res["face_map"]["+X"], 2)
        self.assertEqual(res["face_map"]["-X"], 5)
        self.assertEqual(res["face_map"]["+Y"], 3)
        self.assertEqual(res["face_map"]["-Y"], 4)


if __name__ == "__main__":
    unittest.main()
