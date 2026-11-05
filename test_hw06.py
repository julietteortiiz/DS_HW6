"""Public interface checks. Run: python3 -m unittest -v test_hw06

These checks are examples, not an exhaustive grading rubric.
They intentionally fail until the starter TODOs are implemented.
"""
from __future__ import annotations
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from Shannon import Shannon
from Partition import Example, Partition
from encode_decode import process_contents, read_file, write_file
from best_feature import read_arff, parse_args


class ShannonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Shannon({"A": .25, "B": .125, "C": .5, "D": .125})

    def test_reference_codebook(self) -> None:
        self.assertEqual(self.model.get_binary_codes(), {"A": "10", "B": "110", "C": "0", "D": "111"})

    def test_encode_and_decode(self) -> None:
        self.assertEqual(self.model.encode("ABCD"), "101100111")
        self.assertEqual(self.model.decode("101100111"), "ABCD")

    def test_empty_messages(self) -> None:
        self.assertEqual(self.model.encode(""), "")
        self.assertEqual(self.model.decode(""), "")

    def test_singleton(self) -> None:
        model = Shannon({"x": 1.0})
        self.assertEqual(model.encode("xxx"), "000")
        self.assertEqual(model.decode("000"), "xxx")
        with self.assertRaises(ValueError):
            model.decode("1")

    def test_tie_order_and_copy(self) -> None:
        probs = {"b": .5, "a": .5}
        model = Shannon(probs)
        self.assertEqual(model.get_binary_codes(), {"b": "0", "a": "1"})
        self.assertEqual(probs, {"b": .5, "a": .5})
        model.get_binary_codes().clear()
        self.assertEqual(model.encode("ba"), "01")

    def test_fraction_digits(self) -> None:
        self.assertEqual(self.model.convert_binary(.625, 5), "10100")
        self.assertEqual(self.model.convert_binary(.2, 8), "00110011")
        self.assertEqual(self.model.convert_binary(0, 3), "000")
        self.assertEqual(self.model.binary_correct_digits("101", 5), "10100")
        self.assertEqual(self.model.binary_correct_digits("101", 2), "10")

    def test_lengths(self) -> None:
        self.assertEqual(self.model.calc_num_digits(.3), 2)
        self.assertEqual(self.model.calc_num_digits(1), 1)
        self.assertAlmostEqual(self.model.expected_length(), 1.75)

    def test_invalid_messages(self) -> None:
        with self.assertRaises(KeyError):
            self.model.encode("Z")
        for bits in ["1", "11", "102", "0\n"]:
            with self.subTest(bits=bits), self.assertRaises(ValueError):
                self.model.decode(bits)

    def test_invalid_distributions(self) -> None:
        for probs in [{}, {"a": 0.0}, {"ab": 1.0}, {"a": .4}, {"a": float("nan")}]:
            with self.subTest(probs=probs), self.assertRaises(ValueError):
                Shannon(probs)

    def test_probabilities_preserve_whitespace_and_order(self) -> None:
        probs = process_contents("b a\nb")
        self.assertEqual(list(probs), ["b", " ", "a", "\n"])
        self.assertEqual(probs, {"b": .4, " ": .2, "a": .2, "\n": .2})
        with self.assertRaises(ValueError):
            process_contents("")


class PartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.partition = Partition([
            Example({"x": "a"}, -1), Example({"x": "a"}, 1),
            Example({"x": "b"}, 1), Example({"x": "b"}, 1),
            Example({"x": "b"}, 1), Example({"x": "b"}, 1),
        ], {"x": ["a", "b", "unused"]})

    def test_probabilities(self) -> None:
        p = self.partition
        self.assertAlmostEqual(p._prob(-1), 1 / 6)
        self.assertEqual(p._prob(2), 0)
        self.assertAlmostEqual(p._probX("x", "a"), 1 / 3)
        self.assertEqual(p._cond_prob("x", "a", 1), .5)
        self.assertEqual(p._cond_prob("x", "unused", 1), 0)

    def test_weighted_entropy(self) -> None:
        p = self.partition
        self.assertAlmostEqual(p.entropy(), .6500224216483541)
        self.assertEqual(p._cond_entropy("x", "a"), 1)
        self.assertEqual(p._cond_entropy("x", "unused"), 0)
        self.assertAlmostEqual(p.conditional_entropy("x"), 1 / 3)
        self.assertAlmostEqual(p.information_gain("x"), .3166890883150208)

    def test_majority_accuracy(self) -> None:
        self.assertAlmostEqual(self.partition.classification_accuracy("x"), 5 / 6)
        self.assertAlmostEqual(self.partition.classification_error("x"), 1 / 6)

    def test_pure_class(self) -> None:
        p = Partition([Example({"x": "a"}, 1)], {"x": ["a"]})
        self.assertEqual(p.entropy(), 0)
        self.assertEqual(p.information_gain("x"), 0)
        self.assertEqual(p.classification_accuracy("x"), 1)

    def test_zero_gain_ties(self) -> None:
        p = Partition([Example({"z": "v", "a": "v"}, label) for label in [-1, 1]],
                      {"z": ["v"], "a": ["v"]})
        self.assertEqual(p.best_feature(), "z")
        self.assertEqual(p.best_feature_by_accuracy(), "z")

    def test_empty_and_unknown_feature(self) -> None:
        with self.assertRaises(ValueError):
            Partition([], {}).entropy()
        with self.assertRaises(ValueError):
            Partition([Example({}, 1)], {}).best_feature()
        with self.assertRaises(KeyError):
            self.partition.information_gain("missing")

    def test_quiet_and_unchanged(self) -> None:
        p = self.partition
        before = [(e.features.copy(), e.label) for e in p.data]
        domains = {f: values.copy() for f, values in p.F.items()}
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            p.best_feature()
            p.best_feature_by_accuracy()
        self.assertEqual(output.getvalue(), "")
        self.assertEqual(before, [(e.features, e.label) for e in p.data])
        self.assertEqual(domains, p.F)


class ProvidedHelpersTests(unittest.TestCase):
    def test_file_encoding_and_newlines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "nested" / "text.txt")
            write_file(path, "café\r\n", "ISO-8859-1")
            self.assertEqual(read_file(path, "ISO-8859-1"), "café\r\n")

    def test_cli_arguments(self) -> None:
        self.assertEqual(parse_args(["-r", "example.arff"]).train_filename, "example.arff")

    def test_numeric_preprocessing(self) -> None:
        contents = "@relation tiny\n@attribute age numeric\n@attribute class {no,yes}\n@data\n1,no\n2,no\n4,yes\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.arff"
            path.write_text(contents)
            p = read_arff(str(path))
        self.assertEqual(p.F, {"age<=3.0": ["False", "True"]})
        self.assertEqual([e.features for e in p.data], [{"age<=3.0": "True"}, {"age<=3.0": "True"}, {"age<=3.0": "False"}])
        self.assertEqual([e.label for e in p.data], [-1, -1, 1])


if __name__ == "__main__":
    unittest.main()
