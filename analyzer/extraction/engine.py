import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
from analyzer.java.parser import JavaSourceParser, JavaClassInfo, JavaMethodInfo
from analyzer.metrics.complexity import (
    compute_lines_of_code,
    compute_cyclomatic_complexity,
    compute_max_nested_blocks,
    compute_halstead_metrics
)
from analyzer.metrics.structural import compute_lcom5, compute_cbo, compute_atfd_fdp
from backend.app.schemas.analysis import MetricRecord, EntityGranularity

class MetricExtractionEngine:
    def __init__(self):
        self.parser = JavaSourceParser()

    def analyze_repository_path(
        self,
        repo_dir: Path,
        repo_id: str = "local_repo",
        commit_hash: str = "HEAD"
    ) -> List[MetricRecord]:
        """
        Scans all .java files under repo_dir, builds ASTs, extracts metrics,
        and returns a normalized list of class-level and method-level MetricRecords.
        """
        java_files = list(repo_dir.rglob("*.java"))
        records: List[MetricRecord] = []
        
        # Step 1: Pre-scan to collect all known class names and inheritance links
        known_classes: Set[str] = set()
        inheritance_map: Dict[str, str] = {}  # child -> parent
        children_map: Dict[str, Set[str]] = {} # parent -> children
        
        parsed_files: List[Tuple[Path, str, List[JavaClassInfo]]] = []
        
        for file_path in java_files:
            try:
                rel_path = str(file_path.relative_to(repo_dir))
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    source_code = f.read()
                classes = self.parser.parse_file(str(file_path), source_code)
                parsed_files.append((file_path, rel_path, classes))
                
                for c in classes:
                    known_classes.add(c.name)
                    if c.extends_name:
                        inheritance_map[c.name] = c.extends_name
                        children_map.setdefault(c.extends_name, set()).add(c.name)
            except Exception:
                continue

        # Step 2: Compute Depth of Inheritance Tree (DIT)
        def get_dit(cls_name: str) -> float:
            depth = 0
            curr = cls_name
            visited = set()
            while curr in inheritance_map and curr not in visited:
                visited.add(curr)
                depth += 1
                curr = inheritance_map[curr]
            return float(depth)

        # Step 3: Extract Class & Method Metrics
        for file_path, rel_path, classes in parsed_files:
            for cls in classes:
                # Class Size & Complexity
                cls_loc = compute_lines_of_code(cls.source)
                cls_cc = compute_cyclomatic_complexity(cls.source)
                cls_mnb = compute_max_nested_blocks(cls.source)
                cls_hal = compute_halstead_metrics(cls.source)
                
                # CK Metrics
                # WMC is sum of method cyclomatic complexities
                method_ccs = [compute_cyclomatic_complexity(m.source) for m in cls.methods]
                wmc = sum(method_ccs) if method_ccs else cls_cc
                
                # CBO
                cbo = compute_cbo(cls.name, cls.referenced_types, known_classes)
                
                # LCOM5
                methods_field_access = [m.field_accesses for m in cls.methods]
                lcom5 = compute_lcom5(methods_field_access, cls.fields)
                
                # RFC: methods + external calls
                rfc_invocations = set()
                for m in cls.methods:
                    rfc_invocations.update(m.invocations)
                rfc = float(len(cls.methods) + len(rfc_invocations))
                
                dit = get_dit(cls.name)
                noc = float(len(children_map.get(cls.name, set())))
                
                class_entity_id = f"{repo_id}:{rel_path}:{cls.name}"
                
                class_record = MetricRecord(
                    entity_id=class_entity_id,
                    file_path=rel_path,
                    class_name=cls.name,
                    method_name=None,
                    granularity=EntityGranularity.CLASS,
                    start_line=cls.start_line,
                    end_line=cls.end_line,
                    WMC=float(wmc),
                    CBO=cbo,
                    LCOM=lcom5,
                    LCOM5=lcom5,
                    RFC=rfc,
                    DIT=dit,
                    NOC=noc,
                    ATFD=0.0,
                    FDP=0.0,
                    SLOC=cls_loc["SLOC"],
                    LLOC=cls_loc["LLOC"],
                    cyclomatic_complexity=cls_cc,
                    MNB=cls_mnb,
                    halstead_length=cls_hal["halstead_length"],
                    halstead_volume=cls_hal["halstead_volume"],
                    halstead_difficulty=cls_hal["halstead_difficulty"],
                    halstead_effort=cls_hal["halstead_effort"]
                )
                records.append(class_record)
                
                # Method level extraction
                for m in cls.methods:
                    m_loc = compute_lines_of_code(m.source)
                    m_cc = compute_cyclomatic_complexity(m.source)
                    m_mnb = compute_max_nested_blocks(m.source)
                    m_hal = compute_halstead_metrics(m.source)
                    
                    atfd_fdp = compute_atfd_fdp(m.source, cls.fields, known_classes)
                    m_cbo = float(len(m.parameters) + (1 if m.return_type else 0))
                    
                    method_entity_id = f"{repo_id}:{rel_path}:{cls.name}:{m.name}:{m.start_line}"
                    
                    method_record = MetricRecord(
                        entity_id=method_entity_id,
                        file_path=rel_path,
                        class_name=cls.name,
                        method_name=m.name,
                        granularity=EntityGranularity.METHOD,
                        start_line=m.start_line,
                        end_line=m.end_line,
                        WMC=m_cc,
                        CBO=m_cbo,
                        LCOM=0.0,
                        LCOM5=0.0,
                        RFC=float(len(m.invocations)),
                        DIT=dit,
                        NOC=0.0,
                        ATFD=atfd_fdp["ATFD"],
                        FDP=atfd_fdp["FDP"],
                        SLOC=m_loc["SLOC"],
                        LLOC=m_loc["LLOC"],
                        cyclomatic_complexity=m_cc,
                        MNB=m_mnb,
                        halstead_length=m_hal["halstead_length"],
                        halstead_volume=m_hal["halstead_volume"],
                        halstead_difficulty=m_hal["halstead_difficulty"],
                        halstead_effort=m_hal["halstead_effort"]
                    )
                    records.append(method_record)
                    
        return records
