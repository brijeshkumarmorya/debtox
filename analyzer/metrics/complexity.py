import math
import re
from typing import Dict, Tuple

def compute_lines_of_code(source: str) -> Dict[str, float]:
    """
    Computes SLOC (source lines of code) and LLOC (logical lines of code).
    Excludes pure comments and blank lines.
    """
    lines = source.splitlines()
    total_lines = len(lines)
    sloc = 0
    in_block_comment = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if in_block_comment:
            if "*/" in stripped:
                in_block_comment = False
                after = stripped.split("*/", 1)[1].strip()
                if after:
                    sloc += 1
            continue
        if stripped.startswith("/*"):
            if "*/" in stripped:
                after = stripped.split("*/", 1)[1].strip()
                if after:
                    sloc += 1
            else:
                in_block_comment = True
            continue
        if stripped.startswith("//"):
            continue
        sloc += 1
        
    # LLOC counts executable statement terminators (;, {, })
    lloc = len(re.findall(r'[;{}]', source))
    lloc = max(1.0, float(lloc if lloc > 0 else sloc))
    
    return {
        "SLOC": float(sloc),
        "LLOC": float(lloc)
    }

def compute_cyclomatic_complexity(source: str) -> float:
    """
    Computes McCabe's Cyclomatic Complexity:
    CC = 1 + number of branching decision points.
    Keywords: if, for, while, case, catch, &&, ||, ?, throw
    """
    # Remove strings and comments to avoid false positives
    cleaned = re.sub(r'//.*', '', source)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'"(\\.|[^"\\])*"', '""', cleaned)
    cleaned = re.sub(r"'(\\.|[^'\\])*'", "''", cleaned)
    
    keywords = [
        r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\bcase\b',
        r'\bcatch\b', r'&&', r'\|\|', r'\?', r'\bthrow\b'
    ]
    
    decisions = 0
    for kw in keywords:
        decisions += len(re.findall(kw, cleaned))
        
    return float(1 + decisions)

def compute_max_nested_blocks(source: str) -> float:
    """
    Computes Maximum Nested Blocks (MNB) depth of braces.
    """
    cleaned = re.sub(r'//.*', '', source)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'"(\\.|[^"\\])*"', '""', cleaned)
    
    max_depth = 0
    current_depth = 0
    for char in cleaned:
        if char == '{':
            current_depth += 1
            if current_depth > max_depth:
                max_depth = current_depth
        elif char == '}':
            if current_depth > 0:
                current_depth -= 1
                
    return float(max(0, max_depth - 1))  # Subtract 1 for enclosing method/class brace

def compute_halstead_metrics(source: str) -> Dict[str, float]:
    """
    Computes Halstead complexity suite:
    n1: distinct operators
    n2: distinct operands
    N1: total operators
    N2: total operands
    Vocabulary: n = n1 + n2
    Length: N = N1 + N2
    Volume: V = N * log2(n)
    Difficulty: D = (n1 / 2) * (N2 / n2)
    Effort: E = D * V
    """
    cleaned = re.sub(r'//.*', '', source)
    cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
    
    # Operators in Java
    operator_patterns = [
        r'(\+\+|--|\+=|-=|\*=|/=|%=|&=|\|=|\^=|>>>=|>>=|<<=)',
        r'(==|!=|<=|>=|&&|\|\||<<|>>>|>>)',
        r'(\+|-|\*|/|%|<|>|=|!|&|\||\^|~|\?|:|\.)',
        r'\b(new|instanceof|return|throw)\b'
    ]
    
    operators_found = []
    temp_code = cleaned
    for pattern in operator_patterns:
        matches = re.findall(pattern, temp_code)
        operators_found.extend(matches)
        temp_code = re.sub(pattern, ' ', temp_code)
        
    # Operands: identifiers, literals, numbers
    operands_found = re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b|\b\d+(\.\d+)?\b', temp_code)
    operands_found = [op[0] if isinstance(op, tuple) else op for op in operands_found if op]
    
    n1 = len(set(operators_found))
    n2 = len(set(operands_found))
    N1 = len(operators_found)
    N2 = len(operands_found)
    
    n = n1 + n2
    N = N1 + N2
    
    volume = (N * math.log2(n)) if n > 1 else 0.0
    difficulty = ((n1 / 2.0) * (N2 / float(n2))) if n2 > 0 and n1 > 0 else 1.0
    effort = difficulty * volume
    
    return {
        "halstead_length": float(N),
        "halstead_volume": float(round(volume, 2)),
        "halstead_difficulty": float(round(difficulty, 2)),
        "halstead_effort": float(round(effort, 2))
    }
