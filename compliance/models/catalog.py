"""
Model catalog management for tracking LLM models and providers.
"""
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import threading

from ..data.jsonl_handler import JSONLHandler

logger = logging.getLogger(__name__)


@dataclass
class ModelProvider:
    """Representation of a provider for a specific model."""
    provider_name: str
    provider_model_id: str
    priority: int = 0  # Higher = more preferred
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelCatalogEntry:
    """Representation of a model in the catalog with its providers."""
    canonical_name: str
    providers: List[ModelProvider] = field(default_factory=list)
    release_date: Optional[str] = None
    description: Optional[str] = None
    parameters: Optional[int] = None
    context_window: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_provider(self, provider: str, model_id: str, priority: int = 0) -> None:
        """
        Add a provider for this model or update if it already exists.
        
        Args:
            provider: Provider name
            model_id: Provider-specific model ID
            priority: Provider priority (higher = more preferred)
        """
        # Check if provider already exists
        for existing in self.providers:
            if existing.provider_name == provider and existing.provider_model_id == model_id:
                existing.priority = priority  # Update priority if needed
                return
        
        # Add new provider
        self.providers.append(ModelProvider(provider, model_id, priority))
        # Sort providers by priority (highest first)
        self.providers.sort(key=lambda p: p.priority, reverse=True)
    
    def get_preferred_provider(self) -> Optional[ModelProvider]:
        """Return the highest priority provider or None if no providers."""
        return self.providers[0] if self.providers else None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary, excluding empty fields."""
        result = asdict(self)
        # Convert provider objects to dicts
        result['providers'] = [p.to_dict() for p in self.providers]
        # Clean up empty fields
        return {k: v for k, v in result.items() if v or isinstance(v, (bool, int, float))}


class ModelCatalog:
    """Catalog of models with thread-safe access and persistence."""
    
    def __init__(self, catalog_path: Union[str, Path] = "model_catalog.jsonl"):
        """
        Initialize the model catalog.
        
        Args:
            catalog_path: Path to the catalog JSONL file
        """
        self.catalog_path = Path(catalog_path)
        self.models: Dict[str, ModelCatalogEntry] = {}
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._load_catalog()
    
    def _load_catalog(self) -> None:
        """Load the model catalog from disk."""
        if not self.catalog_path.exists():
            return
            
        with self._lock:
            entries = JSONLHandler.load_jsonl(self.catalog_path, cls=None)
            for entry_dict in entries:
                try:
                    # Convert providers list from dict to ModelProvider objects
                    providers = []
                    for provider_dict in entry_dict.get('providers', []):
                        providers.append(ModelProvider(**provider_dict))
                    
                    entry_dict['providers'] = providers
                    entry = ModelCatalogEntry(**entry_dict)
                    self.models[entry.canonical_name] = entry
                except Exception as e:
                    logger.error(f"Error loading model catalog entry: {e}")
    
    def save_catalog(self) -> bool:
        """
        Save the model catalog to disk.
        
        Returns:
            True if successful, False otherwise
        """
        with self._lock:
            return JSONLHandler.save_jsonl(
                [model.to_dict() for model in self.models.values()],
                self.catalog_path
            )
    
    def get_model(self, canonical_name: str) -> Optional[ModelCatalogEntry]:
        """
        Get a model by its canonical name.
        
        Args:
            canonical_name: Canonical model name
            
        Returns:
            ModelCatalogEntry or None if not found
        """
        with self._lock:
            return self.models.get(canonical_name)
    
    def add_or_update_model(self, canonical_name: str, provider: str = None, 
                           provider_model_id: str = None, priority: int = 0,
                           **metadata) -> ModelCatalogEntry:
        """
        Add a new model or update an existing one.
        
        Args:
            canonical_name: Canonical model name
            provider: Provider name
            provider_model_id: Provider-specific model ID
            priority: Provider priority (higher = more preferred)
            **metadata: Additional metadata fields
            
        Returns:
            The new or updated ModelCatalogEntry
        """
        with self._lock:
            if canonical_name in self.models:
                entry = self.models[canonical_name]
                # Update metadata if provided
                for key, value in metadata.items():
                    if value is not None:
                        if key == 'metadata' and isinstance(value, dict):
                            entry.metadata.update(value)
                        else:
                            setattr(entry, key, value)
            else:
                entry = ModelCatalogEntry(canonical_name=canonical_name, **metadata)
                self.models[canonical_name] = entry
            
            # Add provider if specified
            if provider and provider_model_id:
                entry.add_provider(provider, provider_model_id, priority)
            
            return entry
    
    def get_provider_for_model(self, canonical_name: str, 
                              preferred_provider: str = None) -> Optional[ModelProvider]:
        """
        Get the best provider for a model.
        
        Args:
            canonical_name: Canonical model name
            preferred_provider: Optional specific provider to use
            
        Returns:
            ModelProvider object or None if not found
        """
        with self._lock:
            entry = self.get_model(canonical_name)
            if not entry or not entry.providers:
                return None
            
            if preferred_provider:
                # Find the specific provider if requested
                for provider in entry.providers:
                    if provider.provider_name == preferred_provider:
                        return provider
            
            # Return highest priority provider
            return entry.get_preferred_provider()
    
    def resolve_model(self, canonical_name: str = None, provider: str = None, 
                     provider_model_id: str = None) -> tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Resolve model information based on provided parameters.
        
        This can be called with various combinations of inputs:
        - canonical_name only: Find the best provider/model_id
        - provider and provider_model_id only: Find or create canonical_name
        - canonical_name, provider, provider_model_id: Register the mapping
        
        Args:
            canonical_name: Canonical model name
            provider: Provider name
            provider_model_id: Provider-specific model ID
            
        Returns:
            Tuple of (canonical_name, provider, provider_model_id)
        """
        with self._lock:
            # Case 1: All parameters provided - register the mapping
            if canonical_name and provider and provider_model_id:
                self.add_or_update_model(
                    canonical_name=canonical_name,
                    provider=provider,
                    provider_model_id=provider_model_id
                )
                return canonical_name, provider, provider_model_id
                
            # Case 2: Only canonical_name provided - find best provider
            if canonical_name and not (provider and provider_model_id):
                model_provider = self.get_provider_for_model(
                    canonical_name=canonical_name,
                    preferred_provider=provider
                )
                if model_provider:
                    return canonical_name, model_provider.provider_name, model_provider.provider_model_id
                return canonical_name, None, None
                
            # Case 3: Only provider and provider_model_id provided - find canonical_name
            if provider and provider_model_id and not canonical_name:
                # Search for matching provider/model_id
                for name, entry in self.models.items():
                    for p in entry.providers:
                        if p.provider_name == provider and p.provider_model_id == provider_model_id:
                            return name, provider, provider_model_id
                
                # If not found, create a new entry with provider/model_id as canonical name
                default_canonical = f"{provider}/{provider_model_id}"
                self.add_or_update_model(
                    canonical_name=default_canonical,
                    provider=provider,
                    provider_model_id=provider_model_id
                )
                return default_canonical, provider, provider_model_id
                
            # Case 4: Insufficient information
            return None, None, None
