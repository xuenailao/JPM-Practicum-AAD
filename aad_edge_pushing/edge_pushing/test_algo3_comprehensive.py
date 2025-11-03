"""
Comprehensive test suite for Algorithm 3 (block form) Hessian computation.
Tests various function types and edge cases.
"""

import numpy as np
import sys
from typing import Tuple, List
import traceback

# Add parent directory to path
sys.path.insert(0, '/home/junruw2/AAD')

from aad_edge_pushing.aad.core.var import ADVar
from aad_edge_pushing.aad.core.tape import global_tape
from aad_edge_pushing.edge_pushing.algo3_block import algo3_block


class TestCase:
    """Container for test case information"""
    def __init__(self, name: str, func, inputs: List[float], expected_hessian: np.ndarray, 
                 description: str = "", tolerance: float = 1e-10):
        self.name = name
        self.func = func
        self.inputs = inputs
        self.expected_hessian = expected_hessian
        self.description = description
        self.tolerance = tolerance


def run_test(test_case: TestCase) -> Tuple[bool, str]:
    """Run a single test case and return (passed, message)"""
    try:
        # Reset tape
        global_tape.reset()
        
        # Create input variables
        vars = []
        for i, val in enumerate(test_case.inputs):
            var = ADVar(val, name=f'x{i}', requires_grad=True)
            vars.append(var)
        
        # Compute function
        output = test_case.func(*vars)
        
        # Compute Hessian
        H = algo3_block(output, vars)
        
        # Check result
        diff = np.abs(H - test_case.expected_hessian).max()
        if diff < test_case.tolerance:
            return True, f"PASS (max diff: {diff:.2e})"
        else:
            return False, f"FAIL\n  Computed:\n{H}\n  Expected:\n{test_case.expected_hessian}\n  Max diff: {diff:.2e}"
            
    except Exception as e:
        return False, f"ERROR: {str(e)}\n{traceback.format_exc()}"


def create_test_cases() -> List[TestCase]:
    """Create all test cases"""
    tests = []
    
    # 1. Simple quadratic functions
    tests.append(TestCase(
        "x^2",
        lambda x: x * x,
        [2.0],
        np.array([[2.0]]),
        "Simple quadratic"
    ))
    
    tests.append(TestCase(
        "x^2 + y^2 at (0,0)",
        lambda x, y: x*x + y*y,
        [0.0, 0.0],
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        "Sum of squares at origin"
    ))
    
    tests.append(TestCase(
        "x^2 + y^2 at (1,1)",
        lambda x, y: x*x + y*y,
        [1.0, 1.0],
        np.array([[2.0, 0.0], [0.0, 2.0]]),
        "Sum of squares at (1,1)"
    ))
    
    # 2. Mixed terms
    tests.append(TestCase(
        "xy",
        lambda x, y: x * y,
        [3.0, 4.0],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        "Simple product"
    ))
    
    tests.append(TestCase(
        "x^2*y",
        lambda x, y: x * x * y,
        [2.0, 3.0],
        np.array([[6.0, 4.0], [4.0, 0.0]]),
        "Mixed quadratic"
    ))
    
    # 3. Cubic functions
    tests.append(TestCase(
        "x^3",
        lambda x: x * x * x,
        [2.0],
        np.array([[12.0]]),
        "Simple cubic"
    ))
    
    tests.append(TestCase(
        "x^3*y at (-1,2)",
        lambda x, y: x * x * x * y,
        [-1.0, 2.0],
        np.array([[-12.0, 3.0], [3.0, 0.0]]),
        "Critical test case B"
    ))
    
    # 4. Higher order
    tests.append(TestCase(
        "x^4",
        lambda x: x * x * x * x,
        [1.0],
        np.array([[12.0]]),
        "Quartic at x=1"
    ))
    
    tests.append(TestCase(
        "x^2*y^2",
        lambda x, y: x * x * y * y,
        [1.0, 2.0],
        np.array([[8.0, 8.0], [8.0, 2.0]]),
        "Bivariate quartic"
    ))
    
    # 5. Three variables
    tests.append(TestCase(
        "xyz",
        lambda x, y, z: x * y * z,
        [1.0, 2.0, 3.0],
        np.array([[0.0, 3.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]]),
        "Three-way product"
    ))
    
    tests.append(TestCase(
        "x^2 + y^2 + z^2",
        lambda x, y, z: x*x + y*y + z*z,
        [1.0, 2.0, 3.0],
        np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]),
        "Sum of squares (3D)"
    ))
    
    # 6. Complex expressions
    tests.append(TestCase(
        "(x+y)^2",
        lambda x, y: (x + y) * (x + y),
        [1.0, 2.0],
        np.array([[2.0, 2.0], [2.0, 2.0]]),
        "Squared sum"
    ))
    
    tests.append(TestCase(
        "x^2 + 2xy + y^2",
        lambda x, y: x*x + 2*x*y + y*y,
        [1.0, 2.0],
        np.array([[2.0, 2.0], [2.0, 2.0]]),
        "Expanded squared sum"
    ))
    
    tests.append(TestCase(
        "(x-y)^2",
        lambda x, y: (x - y) * (x - y),
        [3.0, 1.0],
        np.array([[2.0, -2.0], [-2.0, 2.0]]),
        "Squared difference"
    ))
    
    # 7. Edge cases
    tests.append(TestCase(
        "Zero function",
        lambda x, y: x * 0.0,
        [1.0, 2.0],
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        "Constant zero"
    ))
    
    tests.append(TestCase(
        "Linear function",
        lambda x, y: x + y,
        [1.0, 2.0],
        np.array([[0.0, 0.0], [0.0, 0.0]]),
        "Linear has zero Hessian"
    ))
    
    # 8. Nested operations
    tests.append(TestCase(
        "x*(x*(x+1))",
        lambda x: x * (x * (x + 1)),
        [1.0],
        np.array([[8.0]]),  # Fixed: f(x)=x³+x², f''(x)=6x+2, at x=1: f''(1)=8
        "Nested multiplications"
    ))
    
    tests.append(TestCase(
        "(x*y)*(x+y)",
        lambda x, y: (x * y) * (x + y),
        [2.0, 3.0],
        np.array([[6.0, 10.0], [10.0, 4.0]]),  # Fixed: verified with finite differences
        "Product of product and sum"
    ))
    
    # 9. Negative values
    tests.append(TestCase(
        "x^2 at x=-2",
        lambda x: x * x,
        [-2.0],
        np.array([[2.0]]),
        "Quadratic with negative input"
    ))
    
    tests.append(TestCase(
        "x*y at (-1,-1)",
        lambda x, y: x * y,
        [-1.0, -1.0],
        np.array([[0.0, 1.0], [1.0, 0.0]]),
        "Product with negative inputs"
    ))
    
    # 10. Chain of multiplications
    tests.append(TestCase(
        "x*x*x*x",
        lambda x: x * x * x * x,
        [2.0],
        np.array([[48.0]]),
        "Chain of 4 multiplications"
    ))
    
    return tests


def run_all_tests():
    """Run all test cases and report results"""
    print("=" * 70)
    print("COMPREHENSIVE TEST SUITE FOR ALGORITHM 3")
    print("=" * 70)
    
    test_cases = create_test_cases()
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test.name}")
        if test.description:
            print(f"  Description: {test.description}")
        print(f"  Inputs: {test.inputs}")
        
        success, message = run_test(test)
        
        if success:
            passed += 1
            print(f"  Result: {message}")
        else:
            failed += 1
            print(f"  Result: {message}")
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"Success rate: {100 * passed / len(test_cases):.1f}%")
    print("=" * 70)
    
    return passed == len(test_cases)


def debug_failing_case():
    """Debug a specific failing case with detailed output"""
    print("\n" + "=" * 70)
    print("DEBUGGING SPECIFIC CASE: x^2*y^2 at (1,2)")
    print("=" * 70)
    
    global_tape.reset()
    x = ADVar(1.0, name='x', requires_grad=True)
    y = ADVar(2.0, name='y', requires_grad=True)
    
    # f(x,y) = x²y²
    x2 = x * x
    y2 = y * y
    f = x2 * y2
    
    print(f"\nTape structure ({len(global_tape.nodes)} nodes):")
    for i, node in enumerate(global_tape.nodes):
        print(f"  Node {i}: {node.op_tag}")
        for j, (parent, deriv) in enumerate(node.parents):
            parent_name = parent.name if hasattr(parent, 'name') else f'temp_{id(parent)}'
            print(f"    parent[{j}]: {parent_name}, deriv={deriv}")
    
    # Expected Hessian for f = x²y²:
    # ∂f/∂x = 2xy²
    # ∂f/∂y = 2x²y
    # ∂²f/∂x² = 2y² = 8
    # ∂²f/∂x∂y = 4xy = 8
    # ∂²f/∂y² = 2x² = 2
    
    print("\nComputing Hessian...")
    H = algo3_block(f, [x, y])
    print(f"Computed Hessian:\n{H}")
    
    expected = np.array([[8.0, 8.0], [8.0, 2.0]])
    print(f"Expected Hessian:\n{expected}")
    
    diff = np.abs(H - expected).max()
    print(f"Max difference: {diff}")


if __name__ == "__main__":
    # Run all tests
    all_passed = run_all_tests()
    
    # If any test failed, run debug
    if not all_passed:
        debug_failing_case()