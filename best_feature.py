"""Provided ARFF preprocessing and CLI, adapted from Sara Mathieson/Ameet Soni.

Continuous features become categorical threshold indicators using the original
lab's label-change rule. Only these preprocessing helpers mutate examples;
Partition's entropy/selection methods must not mutate their inputs.
"""
from __future__ import annotations
import argparse
import csv
import re
from typing import Optional
from Partition import Example, FeatureDomains, FeatureValues, Partition


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Provided: require -r/--train_filename; None reads the terminal arguments."""
    parser = argparse.ArgumentParser(description="Select informative features")
    parser.add_argument("-r", "--train_filename", required=True)
    return parser.parse_args(argv)


def read_arff(filename: str) -> Partition:
    """Provided reader for dense ARFF with binary class labels in the last column.

    The first declared class maps to -1 and the second to +1. Nominal values
    remain strings. Numeric/real/integer features are discretized by convert_one.
    Preserve declaration order; skip blank lines and full-line percent comments.
    """
    attributes: list[tuple[str, Optional[list[str]]]] = []
    data: list[Example] = []
    in_data = False
    pattern = r"@attribute\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))\s+(.+)"
    with open(filename, encoding="utf-8") as stream:
        for raw in stream:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            if not in_data:
                if line.lower().startswith("@attribute"):
                    match = re.fullmatch(pattern, line, re.IGNORECASE)
                    if match is None:
                        raise ValueError("Invalid ARFF attribute")
                    name = next(x for x in match.groups()[:3] if x is not None)
                    declaration = match.group(4).strip()
                    if declaration.startswith("{") and declaration.endswith("}"):
                        values = [v.strip().strip("\"'") for v in next(csv.reader([declaration[1:-1]]))]
                        attributes.append((name, values))
                    elif declaration.lower() in {"numeric", "real", "integer"}:
                        attributes.append((name, None))
                    else:
                        raise ValueError("Unsupported ARFF attribute type")
                elif line.lower() == "@data":
                    if not attributes or attributes[-1][0] != "class" or attributes[-1][1] is None or len(attributes[-1][1]) != 2:
                        raise ValueError("Expected a final binary class attribute")
                    in_data = True
                continue
            row = [v.strip().strip("\"'") for v in next(csv.reader([line]))]
            if len(row) != len(attributes):
                raise ValueError("Wrong number of ARFF columns")
            features: FeatureValues = {}
            for (name, domain), value in zip(attributes[:-1], row[:-1]):
                if domain is None:
                    features[name] = float(value)
                else:
                    if value not in domain:
                        raise ValueError("Value outside its declared domain")
                    features[name] = value
            classes = attributes[-1][1]
            assert classes is not None
            if row[-1] not in classes:
                raise ValueError("Unknown class label")
            data.append(Example(features, -1 if row[-1] == classes[0] else 1))
    if not in_data:
        raise ValueError("Missing @data section")
    discrete: FeatureDomains = {}
    for name, domain in attributes[:-1]:
        if domain is None:
            convert_one(name, data, discrete)
        else:
            discrete[name] = domain.copy()
    return Partition(data, discrete)


def convert_one(f: str, data: list[Example], F_disc: FeatureDomains) -> None:
    """Provided label-aware discretization, preserving the original lab rule.

    MUTATES data and F_disc: add f<=midpoint indicators with string values
    'True'/'False', then remove numeric feature f. Sorted adjacent unique values
    produce a threshold if their labels differ or either value has mixed labels.
    Empty/constant numeric features produce no thresholds. Train-data use only.
    """
    combined = sorted((float(example.features[f]), example.label) for example in data)
    labels: dict[float, Optional[int]] = {}
    for value, label in combined:
        if value not in labels:
            labels[value] = label
        elif labels[value] != label:
            labels[value] = None
    unique = list(labels)
    switches = [(left + right) / 2 for left, right in zip(unique, unique[1:])
                if labels[left] != labels[right] or labels[left] is None]
    for threshold in switches:
        name = f + "<=" + str(threshold)
        for example in data:
            example.features[name] = "True" if float(example.features[f]) <= threshold else "False"
        F_disc[name] = ["False", "True"]
    for example in data:
        del example.features[f]


def main(argv: Optional[list[str]] = None) -> None:
    """Provided reporting; selection methods themselves must remain quiet."""
    args = parse_args(argv)
    partition = read_arff(args.train_filename)
    i = 1
    # for example in partition.data:
    #     print(str(i) + str(example.features) + "LABEL: " + str(example.label))
    #     i += 1
    # exit(0)
    print("Feature | Information gain (bits) | Training accuracy")
    for feature in partition.F:
        print(f"{feature} | {partition.information_gain(feature):.6f} | {partition.classification_accuracy(feature):.6f}")
    print("best feature:", partition.best_feature())
    print("best feature by accuracy:", partition.best_feature_by_accuracy())


if __name__ == "__main__":
    main()
