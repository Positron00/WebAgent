"""AST code findings table (SEC-043)."""
import pytest

from app.security.code_checks import find_findings


@pytest.mark.parametrize(
    "code,expected",
    [
        ("import requests\nrequests.get('http://x')", {"network"}),
        ("import subprocess\nsubprocess.run(['ls'])", {"process"}),
        ("import os\nos.system('ls')", {"process"}),
        ("import os\nos.getenv('HOME')", {"env"}),
        ("eval('1+1')", {"eval"}),
        ("import ctypes", {"native"}),
        ("open('/etc/passwd', 'w')", {"filesystem"}),
        ("not valid python {{{", {"unparseable"}),
        ("print('hello')", set()),
    ],
)
def test_code_findings(code, expected):
    assert set(find_findings(code)) == expected
