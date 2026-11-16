"""
tests/test_csv_logger.py - Unit tests for logs/csv_logger.py (Stage 4).

Run with:
    python -m pytest tests/test_csv_logger.py -v
"""

import csv
import sys
import pathlib
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from logs.csv_logger import CsvLogger


def _make_logger(tmpdir: pathlib.Path) -> CsvLogger:
    return CsvLogger(run_dir=tmpdir, filename="test_run.csv")


class TestCsvLoggerCreation(unittest.TestCase):

    def test_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.close()
            self.assertTrue((p / "test_run.csv").exists())

    def test_path_property(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.close()
            self.assertEqual(lg.path, p / "test_run.csv")

    def test_header_written(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.close()
            with open(p / "test_run.csv", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.assertIn("timestamp_iso", reader.fieldnames)
                self.assertIn("event_type", reader.fieldnames)

    def test_auto_creates_subdir(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "sub" / "dir"
            lg = CsvLogger(run_dir=p, filename="test.csv")
            lg.close()
            self.assertTrue((p / "test.csv").exists())

    def test_auto_filename_generated(self):
        """When filename is None, an auto-timestamped name is used."""
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = CsvLogger(run_dir=p)
            lg.close()
            files = list(p.glob("run_*.csv"))
            self.assertEqual(len(files), 1)


class TestCsvLoggerLog(unittest.TestCase):

    def test_log_writes_row(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.log("face", {"face_detected": True, "face_confidence": 0.8})
            lg.close()
            with open(p / "test_run.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["event_type"], "face")
            self.assertEqual(rows[0]["face_detected"], "True")

    def test_log_timestamp_present(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.log("test")
            lg.close()
            with open(p / "test_run.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertTrue(len(rows[0]["timestamp_iso"]) > 0)

    def test_log_multiple_rows(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            for event_type in ("start", "face", "speech", "action", "sensor", "stop"):
                lg.log(event_type)
            lg.close()
            with open(p / "test_run.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 6)

    def test_log_missing_columns_filled_empty(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.log("minimal")
            lg.close()
            with open(p / "test_run.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            # All columns present (extra ones filled with empty string)
            self.assertEqual(rows[0]["face_detected"], "")

    def test_log_extra_keys_ignored(self):
        """Extra keys not in schema must not raise."""
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            try:
                lg.log("test", {"unknown_key_xyz": "value"})
            except Exception as e:
                self.fail(f"log() raised on unknown key: {e}")
            lg.close()

    def test_log_no_data_no_crash(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            try:
                lg.log("empty_event")
                lg.log("none_data", None)
            except Exception as e:
                self.fail(f"log() raised with no data: {e}")
            lg.close()

    def test_log_all_event_types(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.log("face", {"face_detected": True})
            lg.log("speech", {"command": "click", "speech_confidence": 0.9})
            lg.log("action", {"action_command": "click", "action_success": True})
            lg.log("sensor", {"sensor_noise_level": 0.03})
            lg.log("error", {"message": "test error"})
            lg.log("start", {"message": "starting"})
            lg.log("stop", {"message": "stopping"})
            lg.close()
            with open(p / "test_run.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            types = [r["event_type"] for r in rows]
            self.assertIn("face", types)
            self.assertIn("speech", types)
            self.assertIn("action", types)
            self.assertIn("sensor", types)
            self.assertIn("error", types)
            self.assertIn("start", types)
            self.assertIn("stop", types)


class TestCsvLoggerClose(unittest.TestCase):

    def test_close_twice_safe(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.close()
            try:
                lg.close()   # Second close must not raise
            except Exception as e:
                self.fail(f"Second close() raised: {e}")

    def test_log_after_close_silent(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.close()
            try:
                lg.log("after_close")   # Must not raise
            except Exception as e:
                self.fail(f"log() after close() raised: {e}")


class TestCsvLoggerThreadSafety(unittest.TestCase):

    def test_concurrent_writes(self):
        """100 concurrent log() calls must all complete, no data corruption."""
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            errors = []

            def writer(n: int):
                for i in range(10):
                    try:
                        lg.log("thread_test", {"message": f"writer-{n}-{i}"})
                    except Exception as e:
                        errors.append(e)

            threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            lg.close()
            self.assertEqual(errors, [], f"Thread errors: {errors}")

            with open(p / "test_run.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 100)

    def test_concurrent_close(self):
        """Multiple threads calling close() must not corrupt state."""
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td)
            lg = _make_logger(p)
            lg.log("before")
            errors = []

            def closer():
                try:
                    lg.close()
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=closer) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)
            self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()

