"""HW06 text experiments, adapted from Allison Gong's original course code."""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional
from Shannon import Shannon


def process_contents(contents: str) -> dict[str, float]:
    """Return character frequencies divided by len(contents), in first-seen order.

    Include all whitespace, punctuation, and newlines. Empty input raises
    ValueError. Return probabilities without printing or changing the input.
    """
    raise NotImplementedError("TODO: implement process_contents")


def read_file(filename: str, encode: str = "utf-8") -> str:
    """Provided: read text without newline conversion (CRLF stays CRLF)."""
    with open(filename, "r", encoding=encode, newline="") as stream:
        return stream.read()


def write_file(filename: str, output: str, encode: str = "utf-8") -> None:
    """Provided: write text exactly, creating parent folders when needed."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w", encoding=encode, newline="") as stream:
        stream.write(output)


def run_encoding(source_filename: str, message_filename: str,
                 secret_filename: str, output_dir: str) -> tuple[str, str]:
    """Provided pipeline: write required outputs and return (encoded, decoded_secret).

    Source/message use ISO-8859-1; secret/output files use UTF-8. No printing.
    Raises ValueError if the message round trip fails. Paths are caller-supplied.
    """
    model = Shannon(process_contents(read_file(source_filename, "ISO-8859-1")))
    message = read_file(message_filename, "ISO-8859-1")
    encoded = model.encode(message)
    if model.decode(encoded) != message:
        raise ValueError("Encoding round trip did not recover the original message")
    decoded = model.decode(read_file(secret_filename))
    write_file(str(Path(output_dir) / "vaccine_tweets_encoded.txt"), encoded)
    write_file(str(Path(output_dir) / "secret_message_decoded.txt"), decoded)
    return encoded, decoded


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Provided optional paths; defaults match the assignment's file names."""
    parser = argparse.ArgumentParser(description="Shannon encoding experiments")
    parser.add_argument("--source", default="data/vaccine_tweets_codes_source.csv")
    parser.add_argument("--message", default="data/vaccine_tweets_to_encode.csv")
    parser.add_argument("--secret", default="data/secrete_message_to_decode.txt")
    parser.add_argument("--output-dir", default="output")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """Provided CLI; passing argv makes it callable without patching sys.argv."""
    args = parse_args(argv)
    encoded, decoded = run_encoding(args.source, args.message, args.secret, args.output_dir)
    print(f"Encoded message: {len(encoded)} bits")
    print(f"Decoded secret: {len(decoded)} characters")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
