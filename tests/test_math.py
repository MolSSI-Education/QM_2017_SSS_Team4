"""
 Testing for the math.py module.
 """
from fcm.math import add, mult
import pytest
 
def test_add():
    assert add(5, 2) == 7
    assert add(2, 5) == 7
    assert add(1, 2) == 3
 
testdata  = [
     (2, 5, 10),
     (1, 2, 2),
     (11, 9, 99),
     (11, 0, 0),
     (0, 0, 0),
]
@pytest.mark.parametrize("a,b,expected", testdata)
def test_mult(a, b, expected):
    assert mult(a, b) == expected
    assert mult(b, a) == expected
