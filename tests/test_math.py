"""
Testing for the math.py module.
"""
import cuddly_octo_invention as coi
import pytest

def test_add():
    assert coi.math.add(5,2) == 7
    assert coi.math.add(2,5) == 7
testdata = [
    (2,5,10),
    (1,2,2),
    (11,9,99),
    (10,9,90),
    (6,3,18),
 ]

@pytest.mark.parametrize("a,b,expected", testdata)
def test_mult(a,b,expected):
    assert coi.math.mult(a,b) == expected 
