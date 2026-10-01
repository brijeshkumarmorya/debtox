import pytest
from analyzer.metrics.complexity import (
    compute_lines_of_code,
    compute_cyclomatic_complexity,
    compute_max_nested_blocks,
    compute_halstead_metrics
)
from analyzer.metrics.structural import compute_lcom5, compute_cbo, compute_atfd_fdp

def test_lines_of_code():
    code = """
    // Single line comment
    /* Multi
       line
       comment */
    public class Test {
        int a = 1;
        int b = 2;
    }
    """
    loc = compute_lines_of_code(code)
    assert loc["SLOC"] >= 4
    assert loc["LLOC"] >= 2

def test_cyclomatic_complexity():
    code = """
    public void check(int x) {
        if (x > 0 && x < 10) {
            for (int i=0; i<x; i++) {
                while (true) break;
            }
        }
    }
    """
    cc = compute_cyclomatic_complexity(code)
    # 1 base + if (1) + && (1) + for (1) + while (1) = 5
    assert cc >= 5

def test_max_nested_blocks():
    code = """
    public void nesting() {
        if (true) {
            while (true) {
                for (;;) {
                    // level 3
                }
            }
        }
    }
    """
    mnb = compute_max_nested_blocks(code)
    assert mnb >= 3

def test_halstead_metrics():
    code = "int a = 10; int b = a + 5;"
    h = compute_halstead_metrics(code)
    assert h["halstead_length"] > 0
    assert h["halstead_volume"] > 0
    assert h["halstead_difficulty"] > 0
    assert h["halstead_effort"] > 0

def test_lcom5():
    # 2 methods accessing completely disjoint fields -> lack of cohesion
    methods = [{'fieldA'}, {'fieldB'}]
    fields = {'fieldA', 'fieldB'}
    lcom5 = compute_lcom5(methods, fields)
    assert lcom5 >= 0.5

def test_atfd_fdp():
    code = """
    public void report(Customer c, Order o) {
        String name = c.getName();
        String id = c.getId();
        double amt = o.getTotal();
    }
    """
    res = compute_atfd_fdp(code, set(), {'Customer', 'Order'})
    assert res["ATFD"] >= 3
    assert res["FDP"] >= 2
