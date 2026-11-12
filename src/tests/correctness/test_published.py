"""``published`` cases: expected values printed in the defining publication or a textbook worked example."""

import pytest

import correctness.cases


@pytest.mark.parametrize('case', correctness.cases.case_params('published'))
# auditme: accept[paired-sibling-helpers] one test module per kind of evidence, as the spec lays out the suite
def test_published(case: correctness.cases.Case) -> None:
    correctness.cases.check_case(case, correctness.cases.run_case(case))
