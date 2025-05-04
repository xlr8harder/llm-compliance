"""
JSONL file handling utilities for compliance evaluation.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import logging
from dataclasses import fields as dataclass_fields

logger = logging.getLogger(__name__)


class JSONLHandler:
    """Utilities for reading and writing JSONL files."""
    
    @staticmethod
    def load_jsonl(file_path: Union[str, Path], cls: Optional[Type] = None, 
                  strict: bool = False) -> List[Any]:
        """
        Load JSONL file into a list of dataclass instances or dicts.
        
        Args:
            file_path: Path to JSONL file
            cls: Optional dataclass to deserialize into
            strict: If False, ignore fields not in the dataclass
            
        Returns:
            List of objects (dataclass instances or dicts)
        """
        path = Path(file_path)
        results = []
        
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return results
            
        with path.open('r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                stripped = line.strip()
                if not stripped:
                    continue
                    
                try:
                    # Parse the JSON
                    data = json.loads(stripped)
                    
                    # Convert to class if requested
                    if cls:
                        try:
                            if strict:
                                # Use as-is
                                obj = cls(**data)
                            else:
                                # Filter to only fields in the dataclass
                                cls_fields = {f.name for f in dataclass_fields(cls)}
                                filtered_data = {k: v for k, v in data.items() if k in cls_fields}
                                obj = cls(**filtered_data)
                            results.append(obj)
                        except TypeError as e:
                            if strict:
                                logger.error(f"Error parsing line {i+1} into {cls.__name__}: {e}")
                            else:
                                # Try harder to make it work by handling only the required fields
                                try:
                                    # Get required fields (those without default values)
                                    import dataclasses
                                    required_fields = {f.name for f in dataclass_fields(cls) 
                                                     if f.default == dataclasses._MISSING_TYPE}
                                    
                                    # Check if we have all required fields
                                    missing = required_fields - set(data.keys())
                                    if missing:
                                        logger.error(f"Line {i+1}: Missing required fields {missing}")
                                        continue
                                        
                                    # Create with only the fields we know about
                                    filtered_data = {k: v for k, v in data.items() if k in cls_fields}
                                    obj = cls(**filtered_data)
                                    results.append(obj)
                                except Exception as inner_e:
                                    logger.error(f"Error parsing line {i+1} despite relaxed handling: {inner_e}")
                                    continue
                    else:
                        results.append(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON at line {i+1}: {e}")
                    continue
                    
        return results
    
    @staticmethod
    def save_jsonl(objects: List[Any], file_path: Union[str, Path], append: bool = False) -> bool:
        """
        Save a list of objects to a JSONL file.
        
        Args:
            objects: List of objects with to_dict() method or dicts
            file_path: Path to JSONL file
            append: Whether to append to the file
            
        Returns:
            True if successful, False otherwise
        """
        path = Path(file_path)
        
        # Create parent directory if it doesn't exist
        path.parent.mkdir(exist_ok=True, parents=True)
        
        try:
            if append and path.exists():
                # Direct append for existing files - no temp file needed
                with path.open('a', encoding='utf-8') as f:
                    for obj in objects:
                        if hasattr(obj, 'to_dict'):
                            data = obj.to_dict()
                        else:
                            data = obj
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                return True
            else:
                # For new files or overwrites, use the atomic approach with temp file
                temp_path = path.with_suffix(f".{datetime.now().strftime('%Y%m%d%H%M%S')}.tmp")
                
                with temp_path.open('w', encoding='utf-8') as f:
                    for obj in objects:
                        if hasattr(obj, 'to_dict'):
                            data = obj.to_dict()
                        else:
                            data = obj
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # Rename temp file to target file (atomic operation)
                temp_path.replace(path)
                return True
        except Exception as e:
            logger.error(f"Error saving JSONL file: {e}")
            # Clean up temp file if it exists
            if 'temp_path' in locals() and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as cleanup_error:
                    logger.error(f"Error cleaning up temp file: {cleanup_error}")
            return False

    @staticmethod
    def load_jsonl_to_dict(file_path: Union[str, Path], key_field: str = "id", 
                          cls: Optional[Type] = None) -> Dict[str, Any]:
        """
        Load JSONL file into a dictionary keyed by the specified field.
        
        Args:
            file_path: Path to JSONL file
            key_field: Field to use as dictionary key
            cls: Optional dataclass to deserialize into
            
        Returns:
            Dictionary of objects keyed by key_field
        """
        entries = JSONLHandler.load_jsonl(file_path, cls)
        result = {}
        
        for entry in entries:
            key = entry.get(key_field) if isinstance(entry, dict) else getattr(entry, key_field, None)
            if key is not None:
                result[key] = entry
            else:
                logger.warning(f"Entry missing key field '{key_field}', skipping")
                
        return result
