"""
Schema and loader for Likert-style survey definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json


@dataclass
class SurveyScale:
    """Likert-style scale definition with labeled anchors.

    - min_value, max_value define the numeric range (inclusive)
    - labels maps a stringified integer (e.g., "1") to a description
    """

    min_value: int
    max_value: int
    labels: Dict[str, str]

    def validate(self) -> None:
        if self.min_value >= self.max_value:
            raise ValueError("SurveyScale: min_value must be < max_value")
        # Ensure all labels for each integer in range exist
        for v in range(self.min_value, self.max_value + 1):
            if str(v) not in self.labels:
                raise ValueError(f"SurveyScale: missing label for value {v}")

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SurveyItem:
    """One item in a survey.

    - id: unique id for the item
    - text: the question or statement to rate
    - category: optional grouping id for aggregated scores
    - inversion: if True, invert scoring when aggregating
    """

    id: str
    text: str
    category: Optional[str] = None
    inversion: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SurveyDefinition:
    """Complete survey definition.

    - name: stable identifier for the survey (filename-friendly)
    - title: human-readable title
    - initial_prompt: preface to be included with each item when asked
    - scale: Likert scale configuration
    - items: list of SurveyItem objects
    """

    name: str
    title: Optional[str]
    initial_prompt: str
    scale: SurveyScale
    items: List[SurveyItem] = field(default_factory=list)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("SurveyDefinition: name is required")
        if not self.initial_prompt:
            raise ValueError("SurveyDefinition: initial_prompt is required")
        if not self.items:
            raise ValueError("SurveyDefinition: at least one item is required")
        self.scale.validate()
        seen: set[str] = set()
        for it in self.items:
            if it.id in seen:
                raise ValueError(f"SurveyDefinition: duplicate item id {it.id}")
            seen.add(it.id)

    def to_dict(self) -> Dict:
        d = asdict(self)
        return d


def load_survey(path: Path) -> SurveyDefinition:
    """Load and validate a survey JSON file into a SurveyDefinition."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    # Build nested objects
    scale = SurveyScale(
        min_value=int(data["scale"]["min"]),
        max_value=int(data["scale"]["max"]),
        labels={str(k): str(v) for k, v in data["scale"]["labels"].items()},
    )
    items = [
        SurveyItem(
            id=str(it["id"]),
            text=str(it["text"]),
            category=str(it["category"]) if it.get("category") is not None else None,
            inversion=bool(it.get("inversion", False)),
        )
        for it in data.get("items", [])
    ]
    survey = SurveyDefinition(
        name=str(data["name"]),
        title=str(data.get("title")) if data.get("title") is not None else None,
        initial_prompt=str(data["initial_prompt"]),
        scale=scale,
        items=items,
    )
    survey.validate()
    return survey

