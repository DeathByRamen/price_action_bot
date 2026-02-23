"""Tests for src/scoring/accuracy.py and src/scoring/adaptive.py."""

from __future__ import annotations

from src.scoring.accuracy import AccuracyReport, classify_direction


class TestClassifyDirection:
    def test_up(self):
        assert classify_direction(0.01, threshold=0.005) == "UP"

    def test_down(self):
        assert classify_direction(-0.01, threshold=0.005) == "DOWN"

    def test_flat_positive(self):
        assert classify_direction(0.003, threshold=0.005) == "FLAT"

    def test_flat_negative(self):
        assert classify_direction(-0.003, threshold=0.005) == "FLAT"

    def test_flat_zero(self):
        assert classify_direction(0.0, threshold=0.005) == "FLAT"

    def test_exact_boundary_positive(self):
        assert classify_direction(0.005, threshold=0.005) == "FLAT"

    def test_above_boundary(self):
        assert classify_direction(0.0051, threshold=0.005) == "UP"

    def test_custom_threshold(self):
        assert classify_direction(0.02, threshold=0.03) == "FLAT"
        assert classify_direction(0.04, threshold=0.03) == "UP"


class TestAccuracyReport:
    def test_as_db_row(self):
        report = AccuracyReport(
            run_date="2025-01-01",
            total_scored=100,
            direction_accuracy=0.65,
            magnitude_mae=0.003,
            up_precision=0.7,
            up_recall=0.6,
            down_precision=0.65,
            down_recall=0.55,
            flat_precision=0.5,
            flat_recall=0.7,
            flat_threshold_used=0.005,
        )
        row = report.as_db_row()
        assert len(row) == 11
        assert row[0] == "2025-01-01"
        assert row[1] == 100
        assert row[2] == 0.65

    def test_default_fields(self):
        report = AccuracyReport(run_date="2025-01-01")
        assert report.total_scored == 0
        assert report.per_symbol_accuracy == {}
        assert report.top_symbols == []
