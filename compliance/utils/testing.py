"""
Testing utilities for the compliance package.
"""
from dataclasses import fields
import json
from pathlib import Path
from typing import Any, Dict, List, Type, Union
import logging

from ..data.jsonl_handler import JSONLHandler

logger = logging.getLogger(__name__)


def test_round_trip(file_path: Union[str, Path], cls: Type) -> bool:
    """
    Test loading and saving data to ensure consistency.
    
    Args:
        file_path: Path to JSONL file
        cls: Dataclass to use for deserialization
        
    Returns:
        True if round-trip was successful, False otherwise
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        return False
    
    logger.info(f"Testing round-trip for {path}")
    
    # Direct JSON load
    with path.open('r', encoding='utf-8') as f:
        original_entries = []
        errors = 0
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                original_entries.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing line {i}: {e}")
                errors += 1
    
    if errors > 0:
        logger.warning(f"Found {errors} parsing errors in original file")
    
    # Load the file
    entries = []
    unknown_fields = {}
    
    with path.open('r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
                
            try:
                data = json.loads(stripped)
                
                # Extract the data class fields
                cls_fields = {f.name for f in fields(cls)}
                
                # Find unknown fields
                extra_fields = set(data.keys()) - cls_fields
                if extra_fields:
                    unknown_fields[i] = extra_fields
                
                # Create a filtered dictionary with only known fields
                filtered_data = {k: v for k, v in data.items() if k in cls_fields}
                
                # Create the dataclass instance
                obj = cls(**filtered_data)
                entries.append(obj)
                
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Error on line {i}: {e}")
                errors += 1
    
    # Report on unknown fields
    if unknown_fields:
        total_unknown = sum(len(fields) for fields in unknown_fields.values())
        all_unknown = set().union(*unknown_fields.values())
        logger.warning(f"Found {total_unknown} unknown field occurrences across {len(unknown_fields)} lines")
        logger.warning(f"Unknown field names: {sorted(all_unknown)}")
    
    # Convert back to dicts
    round_trip_entries = [entry.to_dict() for entry in entries]
    
    # Compare only fields that should have been preserved
    differences = 0
    
    if len(original_entries) != len(round_trip_entries):
        logger.warning(f"Count mismatch: {len(original_entries)} original vs {len(round_trip_entries)} round-trip")
        differences += 1
    
    for i, (orig, rt) in enumerate(zip(original_entries, round_trip_entries)):
        # Get fields in both the original and the schema
        cls_fields = {f.name for f in fields(cls)}
        valid_orig_keys = set(orig.keys()) & cls_fields
        
        # Check for missing valid keys
        missing_valid_keys = valid_orig_keys - set(rt.keys())
        if missing_valid_keys:
            logger.warning(f"Entry {i}: Missing valid keys after round-trip: {missing_valid_keys}")
            differences += 1
        
        # Check values for common keys
        for key in valid_orig_keys & set(rt.keys()):
            if orig[key] != rt[key]:
                logger.warning(f"Entry {i}: Value mismatch for key '{key}':")
                logger.warning(f"  Original: {orig[key]}")
                logger.warning(f"  Round-trip: {rt[key]}")
                differences += 1
    
    if differences == 0:
        logger.info(f"Round-trip test passed: {len(entries)} entries processed with no differences")
        if unknown_fields:
            logger.warning(f"Note: {len(unknown_fields)} entries had unknown fields that were not included in the schema")
        return True
    else:
        logger.warning(f"Round-trip test found {differences} differences")
        return False
