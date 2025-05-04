#!/usr/bin/env python3
"""
Test script for validating JSONL data files against schema.
"""
import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path
from dataclasses import dataclass, fields, field, asdict
from typing import Dict, List, Optional, Any, Union

# Add the parent directory to the path so we can import compliance
script_dir = Path(__file__).parent.resolve()
parent_dir = script_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from compliance.data import Question, ModelResponse, ComplianceAnalysis
from compliance.utils.testing import test_round_trip
from dataclasses import fields


logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_data")

def print_schema_info(cls):
    """Print info about a schema class"""
    print(f"\nSchema for {cls.__name__}:")

    required_fields = []
    optional_fields = []

    for field in fields(cls):
        if field.default == field.default_factory == dataclasses._MISSING_TYPE:
            required_fields.append(field.name)
        else:
            optional_fields.append(field.name)

    print("Required fields:")
    for field in required_fields:
        print(f"  - {field}")

    print("\nOptional fields:")
    for field in optional_fields:
        print(f"  - {field}")


def main():
    parser = argparse.ArgumentParser(description="Test JSONL files against schema")

    parser.add_argument("--file", help="Path to the JSONL file to test")
    parser.add_argument("--dir", help="Directory of JSONL files to test")
    parser.add_argument("--type", choices=["question", "response", "analysis"],
                      help="Type of data to test against")
    parser.add_argument("--info", action="store_true",
                      help="Show schema information")

    args = parser.parse_args()

    # Select the appropriate class
    cls_map = {
        "question": Question,
        "response": ModelResponse,
        "analysis": ComplianceAnalysis
    }

    # Show schema info if requested
    if args.info:
        if args.type:
            print_schema_info(cls_map[args.type])
        else:
            for cls_type, cls in cls_map.items():
                print_schema_info(cls)
        return

    # Require file or directory
    if not args.file and not args.dir:
        parser.error("Either --file or --dir must be specified")

    # Require type
    if not args.type:
        parser.error("--type must be specified")

    cls = cls_map[args.type]

    # Test single file
    if args.file:
        file_path = Path(args.file)
        success = test_round_trip(file_path, cls)
        sys.exit(0 if success else 1)

    # Test directory
    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error(f"Not a directory: {dir_path}")
            sys.exit(1)

        files = list(dir_path.glob("*.jsonl"))
        if not files:
            logger.error(f"No JSONL files found in {dir_path}")
            sys.exit(1)

        logger.info(f"Testing {len(files)} files in {dir_path}")

        all_success = True
        for file_path in files:
            logger.info(f"Testing {file_path}")
            success = test_round_trip(file_path, cls)
            if not success:
                all_success = False

        sys.exit(0 if all_success else 1)


if __name__ == "__main__":
    main()
