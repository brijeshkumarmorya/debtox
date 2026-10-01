from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import hashlib

@dataclass
class DatasetProvenanceRecord:
    dataset_name: str
    official_publication: str
    journal_or_venue: str
    year: int
    doi: str
    authors: List[str]
    source_url: str
    license: str
    retrieval_date: str
    checksum_md5: Optional[str] = None
    checksum_sha256: Optional[str] = None
    raw_size_bytes: Optional[int] = None
    total_projects: Optional[int] = None
    total_samples: Optional[int] = None
    available_smells: List[str] = field(default_factory=list)
    unavailable_smells: List[str] = field(default_factory=list)
    label_methodology: str = ""
    feature_source: str = ""
    notes: str = ""

DATASET_REGISTRY: Dict[str, DatasetProvenanceRecord] = {
    "smellycode_plus_plus": DatasetProvenanceRecord(
        dataset_name="SmellyCode++",
        official_publication="SmellyCode++: Multi-Label Dataset for Code Smell Detection",
        journal_or_venue="Scientific Data (Nature)",
        year=2025,
        doi="10.1038/s41597-025-05465-z",
        authors=[
            "Nawaf Alomari",
            "Amal Alazba",
            "Hamoud Aljamaan",
            "Mohammad Alshayeb"
        ],
        source_url="https://doi.org/10.6084/m9.figshare.28519385.v1",
        license="Creative Commons Zero 1.0 Universal (CC0)",
        retrieval_date="2026-10-01T06:17:30Z",
        checksum_md5="0d39aed4be0099cfba5ab73343c70039",
        checksum_sha256="b368922aff91eef64bb3de3ba78afd7b1d05e9af16581c686cd2175c29ec7a9c",
        raw_size_bytes=114094024,
        total_projects=26,
        total_samples=12400,
        available_smells=["god_class", "data_class", "long_method", "feature_envy"],
        unavailable_smells=["brain_class", "brain_method"],
        label_methodology="Multi-label consensus from static analysis advisors with stratified manual validation",
        feature_source="external_supplied_14_metrics",
        notes="Primary dataset for DebtOx empirical research. Brain Class and Brain Method are marked UNAVAILABLE and strictly excluded from empirical evaluation."
    ),
    "crowdsmelling": DatasetProvenanceRecord(
        dataset_name="Crowdsmelling",
        official_publication="Crowdsmelling: A preliminary study on using collective knowledge in code smells detection",
        journal_or_venue="Empirical Software Engineering (Springer)",
        year=2022,
        doi="10.1007/s10664-021-10110-5",
        authors=[
            "J. P. D. Reis",
            "F. B. E. Abreu",
            "G. D. F. Carneiro"
        ],
        source_url="https://github.com/dataset-cs-surveys/Crowdsmelling",
        license="Open Academic Research (MIT / CC-BY)",
        retrieval_date="2026-10-01T06:06:27Z",
        total_projects=3,
        total_samples=1946,
        available_smells=["god_class", "long_method", "feature_envy"],
        unavailable_smells=["brain_class", "brain_method", "data_class"],
        label_methodology="Human ground-truth consensus collected over 3 years (2018-2020) from developer crowds",
        feature_source="external_supplied_ck_metrics",
        notes="Human-validated secondary benchmark dataset. Data Class, Brain Class, and Brain Method are marked UNAVAILABLE in this dataset."
    )
}

def verify_file_checksum(file_path: Path, expected_md5: Optional[str] = None) -> Dict[str, str]:
    """Calculates MD5 and SHA256 for provenance verification."""
    md5_hash = hashlib.md5()
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            md5_hash.update(chunk)
            sha256_hash.update(chunk)
    actual_md5 = md5_hash.hexdigest()
    actual_sha256 = sha256_hash.hexdigest()
    return {
        "md5": actual_md5,
        "sha256": actual_sha256,
        "md5_matches": str(actual_md5.lower() == expected_md5.lower()) if expected_md5 else "unspecified"
    }
