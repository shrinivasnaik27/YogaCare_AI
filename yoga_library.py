"""
YogaCare AI - Yoga Practice Library Module
Provides structured access, categorization, filtering, and sequential navigation
for all 40 yoga practices across 7 standardized categories.
"""

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class YogaPractice:
    """Represents a single yoga practice within the 40-practice library."""
    id: str
    sequence_number: int
    category_id: str
    category_name: str
    category_number: int
    name: str
    sanskrit_name: str
    difficulty: str
    image_filename: str
    local_image_path: str
    description: str
    instructions: List[str] = field(default_factory=list)
    instructions_kannada: List[str] = field(default_factory=list)
    alignment_tips: List[str] = field(default_factory=list)
    breathing_guidance: str = ""
    beginner_modification: str = ""
    safety_reminder: str = "Breathe naturally and avoid strain."
    suggested_duration_seconds: int = 30
    is_repetition: bool = False
    requires_sides: bool = False
    practice_type: str = "static"
    has_ai_evaluation: bool = False
    applicable_evaluator_key: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class YogaCategory:
    """Represents a categorized phase containing a curated set of yoga practices."""
    category_id: str
    category_number: int
    name: str
    description: str
    thumbnail_path: str
    practice_count: int
    practice_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CATEGORY_METADATA = [
    {
        "category_id": "cat_warmup",
        "category_number": 1,
        "name": "Warm-up & Mobility",
        "description": "Prepare the body with gentle movement, mobility, and breathing.",
        "thumbnail_path": "assets/category_thumbnails/cat_01_warmup.jpg",
    },
    {
        "category_id": "cat_standing",
        "category_number": 2,
        "name": "Standing Poses",
        "description": "Build lower-body stability, balance, alignment, and grounded strength.",
        "thumbnail_path": "assets/category_thumbnails/cat_02_standing.jpg",
    },
    {
        "category_id": "cat_seated",
        "category_number": 3,
        "name": "Seated Poses",
        "description": "Enhance hip mobility, spinal elongation, and calm grounding.",
        "thumbnail_path": "assets/category_thumbnails/cat_03_seated.jpg",
    },
    {
        "category_id": "cat_kneeling",
        "category_number": 4,
        "name": "Kneeling & Back Mobility",
        "description": "Mobilize the spine, alleviate stiffness, and strengthen posterior chain.",
        "thumbnail_path": "assets/category_thumbnails/cat_04_kneeling.jpg",
    },
    {
        "category_id": "cat_core",
        "category_number": 5,
        "name": "Strength & Core",
        "description": "Engage abdominal, arm, and stabilizer muscles for functional endurance.",
        "thumbnail_path": "assets/category_thumbnails/cat_05_core.jpg",
    },
    {
        "category_id": "cat_flexibility",
        "category_number": 6,
        "name": "Flexibility & Recovery",
        "description": "Release deep muscular tension, soothe connective tissue, and restore calm.",
        "thumbnail_path": "assets/category_thumbnails/cat_06_flexibility.jpg",
    },
    {
        "category_id": "cat_breathing",
        "category_number": 7,
        "name": "Breathing & Meditation",
        "description": "Cultivate breath control, quiet the nervous system, and achieve mental stillness.",
        "thumbnail_path": "assets/category_thumbnails/cat_07_breathing.jpg",
    },
]


class YogaLibrary:
    """
    Core library orchestrator for the 40 Yoga Practices.
    Manages loading, category groupings, practice lookups, sequential navigation, and validation.
    """

    def __init__(self, data_path: Optional[str] = None) -> None:
        if data_path:
            self.data_path = Path(data_path)
        else:
            base_dir = Path(__file__).resolve().parent
            self.data_path = base_dir / "data" / "yoga_practices.json"

        self.practices: List[YogaPractice] = []
        self._practice_map: Dict[str, YogaPractice] = {}
        self._sequence_map: Dict[int, YogaPractice] = {}
        self._category_practices: Dict[str, List[YogaPractice]] = {}
        self.categories: List[YogaCategory] = []
        self._category_map: Dict[str, YogaCategory] = {}

        self._load_practices()
        self._build_categories()

    def _load_practices(self) -> None:
        """Parses practices from the JSON configuration."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Practices configuration not found at: {self.data_path}")

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.practices = []
        for item in raw_data:
            practice = YogaPractice(
                id=item["id"],
                sequence_number=item["sequence_number"],
                category_id=item["category_id"],
                category_name=item["category_name"],
                category_number=item["category_number"],
                name=item["name"],
                sanskrit_name=item["sanskrit_name"],
                difficulty=item.get("difficulty", "Beginner"),
                image_filename=item["image_filename"],
                local_image_path=item["local_image_path"],
                description=item.get("description", ""),
                instructions=item.get("instructions", []),
                instructions_kannada=item.get("instructions_kannada", []),
                alignment_tips=item.get("alignment_tips", []),
                breathing_guidance=item.get("breathing_guidance", ""),
                beginner_modification=item.get("beginner_modification", ""),
                safety_reminder=item.get("safety_reminder", "Breathe naturally and avoid strain."),
                suggested_duration_seconds=item.get("suggested_duration_seconds", 30),
                is_repetition=item.get("is_repetition", False),
                requires_sides=item.get("requires_sides", False),
                practice_type=item.get("practice_type", "static"),
                has_ai_evaluation=item.get("has_ai_evaluation", False),
                applicable_evaluator_key=item.get("applicable_evaluator_key"),
            )
            self.practices.append(practice)
            self._practice_map[practice.id] = practice
            self._sequence_map[practice.sequence_number] = practice
            self._category_practices.setdefault(practice.category_id, []).append(practice)

    def _build_categories(self) -> None:
        """Constructs category metadata and associates practices with each category."""
        self.categories = []
        for cat_meta in CATEGORY_METADATA:
            cid = cat_meta["category_id"]
            assigned_practices = self._category_practices.get(cid, [])
            cat = YogaCategory(
                category_id=cid,
                category_number=cat_meta["category_number"],
                name=cat_meta["name"],
                description=cat_meta["description"],
                thumbnail_path=cat_meta["thumbnail_path"],
                practice_count=len(assigned_practices),
                practice_ids=[p.id for p in assigned_practices],
            )
            self.categories.append(cat)
            self._category_map[cid] = cat

    def get_categories(self) -> List[YogaCategory]:
        """Returns all 7 categories in standard sequence."""
        return list(self.categories)

    def get_category_by_id(self, category_id: str) -> Optional[YogaCategory]:
        """Looks up a category by its identifier."""
        return self._category_map.get(category_id)

    def get_category_by_number(self, category_number: int) -> Optional[YogaCategory]:
        """Looks up a category by sequence number (1 to 7)."""
        for cat in self.categories:
            if cat.category_number == category_number:
                return cat
        return None

    def get_all_practices(self) -> List[YogaPractice]:
        """Returns all 40 practices in sequence order."""
        return list(self.practices)

    def get_practices_by_category(self, category_id: str) -> List[YogaPractice]:
        """Returns all practices within a specific category."""
        return list(self._category_practices.get(category_id, []))

    def get_practice_by_id(self, practice_id: str) -> Optional[YogaPractice]:
        """Looks up a practice by its unique ID."""
        return self._practice_map.get(practice_id)

    def get_practice_by_sequence(self, sequence_number: int) -> Optional[YogaPractice]:
        """Looks up a practice by sequence number (1 to 40)."""
        return self._sequence_map.get(sequence_number)

    def get_next_practice(self, current_practice_id: str, within_category: bool = True) -> Optional[YogaPractice]:
        """
        Retrieves the next practice.
        If within_category is True, only returns the next practice within the same category (or None if at the end of category).
        If within_category is False, advances to the next practice globally (or None if at 40).
        """
        curr = self.get_practice_by_id(current_practice_id)
        if not curr:
            return None

        if within_category:
            cat_practices = self.get_practices_by_category(curr.category_id)
            for idx, p in enumerate(cat_practices):
                if p.id == curr.id:
                    if idx + 1 < len(cat_practices):
                        return cat_practices[idx + 1]
                    return None
            return None
        else:
            return self.get_practice_by_sequence(curr.sequence_number + 1)

    def get_previous_practice(self, current_practice_id: str, within_category: bool = True) -> Optional[YogaPractice]:
        """
        Retrieves the previous practice.
        If within_category is True, only returns the previous practice within the same category (or None if at the start).
        If within_category is False, moves to the previous practice globally (or None if at 1).
        """
        curr = self.get_practice_by_id(current_practice_id)
        if not curr:
            return None

        if within_category:
            cat_practices = self.get_practices_by_category(curr.category_id)
            for idx, p in enumerate(cat_practices):
                if p.id == curr.id:
                    if idx > 0:
                        return cat_practices[idx - 1]
                    return None
            return None
        else:
            return self.get_practice_by_sequence(curr.sequence_number - 1)

    def validate_library(self, base_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Validates the integrity of all 40 practices, 7 categories, and image assets on disk.
        """
        if base_dir is None:
            base_dir = Path(__file__).resolve().parent

        issues: List[str] = []
        expected_category_counts = {
            1: 5,  # Warm-up
            2: 7,  # Standing
            3: 6,  # Seated
            4: 6,  # Kneeling
            5: 5,  # Core
            6: 6,  # Flexibility
            7: 5,  # Breathing
        }

        # Validate total count
        if len(self.practices) != 40:
            issues.append(f"Expected 40 practices, found {len(self.practices)}")

        # Validate total categories
        if len(self.categories) != 7:
            issues.append(f"Expected 7 categories, found {len(self.categories)}")

        # Validate category counts
        for cat in self.categories:
            expected = expected_category_counts.get(cat.category_number, 0)
            if cat.practice_count != expected:
                issues.append(
                    f"Category {cat.category_number} ({cat.name}): expected {expected} practices, got {cat.practice_count}"
                )
            # Check category thumbnail
            thumb_path = base_dir / cat.thumbnail_path
            if not thumb_path.exists():
                issues.append(f"Missing category thumbnail: {cat.thumbnail_path}")

        # Validate individual practices
        seen_ids = set()
        seen_seqs = set()
        for idx, p in enumerate(self.practices, start=1):
            if p.sequence_number != idx:
                issues.append(f"Practice {p.id}: sequence_number {p.sequence_number} != expected {idx}")
            if p.id in seen_ids:
                issues.append(f"Duplicate practice ID: {p.id}")
            seen_ids.add(p.id)
            if p.sequence_number in seen_seqs:
                issues.append(f"Duplicate sequence number: {p.sequence_number}")
            seen_seqs.add(p.sequence_number)

            # Check image existence
            img_path = base_dir / p.local_image_path
            if not img_path.exists():
                issues.append(f"Missing practice image: {p.local_image_path} for {p.name}")

            # Check bilingual instructions
            if not p.instructions:
                issues.append(f"Practice {p.id} has empty English instructions.")
            if not p.instructions_kannada:
                issues.append(f"Practice {p.id} has empty Kannada instructions.")

        return {
            "is_valid": len(issues) == 0,
            "total_practices": len(self.practices),
            "total_categories": len(self.categories),
            "issues": issues,
        }
