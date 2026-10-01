from typing import Dict, List, Set, Any
import re

def compute_lcom5(methods_field_access: List[Set[str]], field_names: Set[str]) -> float:
    """
    Computes Henderson-Sellers LCOM5:
    m = number of methods
    a = number of fields (attributes)
    If m <= 1 or a == 0, LCOM5 = 0.0
    Otherwise:
    sum_a = sum over each attribute of how many methods access it
    LCOM5 = ( (1/a) * sum_a - m ) / (1 - m)
    Range: [0, 1] where 0 is high cohesion and 1 is total lack of cohesion.
    """
    m = len(methods_field_access)
    a = len(field_names)
    if m <= 1 or a == 0:
        return 0.0
        
    sum_a = 0
    for field in field_names:
        count = sum(1 for method_fields in methods_field_access if field in method_fields)
        sum_a += count
        
    num = (sum_a / float(a)) - float(m)
    denom = 1.0 - float(m)
    
    val = num / denom if denom != 0 else 0.0
    return float(max(0.0, min(1.0, round(val, 3))))

def compute_cbo(class_name: str, referenced_types: Set[str], known_classes: Set[str]) -> float:
    """
    Coupling Between Objects (CBO):
    Number of unique other classes to which this class is coupled.
    Excludes primitives and standard java.lang built-ins.
    """
    standard_types = {
        'int', 'long', 'short', 'byte', 'char', 'boolean', 'float', 'double', 'void',
        'String', 'Object', 'Integer', 'Long', 'Boolean', 'Double', 'Float',
        'List', 'Map', 'Set', 'ArrayList', 'HashMap', 'HashSet', 'Collection',
        'Exception', 'Throwable', 'RuntimeException', 'Class', 'System'
    }
    
    coupled = set()
    for t in referenced_types:
        if t != class_name and t not in standard_types:
            coupled.add(t)
            
    return float(len(coupled))

def compute_atfd_fdp(method_source: str, class_fields: Set[str], foreign_classes: Set[str]) -> Dict[str, float]:
    """
    Computes method-level Feature Envy indicators:
    ATFD (Access to Foreign Data): number of foreign fields or foreign getters accessed.
    FDP (Foreign Data Providers): number of distinct foreign classes providing that data.
    """
    # Look for calls like object.getSomething() or object.field
    foreign_calls = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\.(get[A-Z][A-Za-z0-9_]*|[a-z_][A-Za-z0-9_]*)', method_source)
    
    foreign_data_accessed = set()
    foreign_providers = set()
    
    for obj_name, member in foreign_calls:
        if obj_name not in ('this', 'super', 'System', 'Math', 'Arrays', 'Collections', 'logger', 'log'):
            if member not in class_fields:
                foreign_data_accessed.add(f"{obj_name}.{member}")
                foreign_providers.add(obj_name)
                
    return {
        "ATFD": float(len(foreign_data_accessed)),
        "FDP": float(len(foreign_providers))
    }
