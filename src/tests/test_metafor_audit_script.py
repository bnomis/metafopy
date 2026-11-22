"""Tests of ``bin/metafor-audit.py``, the entry point of the audit: ``--metafopy-src`` is read before anything imports metafopy."""

import importlib.util
import pathlib
import sys
import types

import hypothesis
import hypothesis.strategies

SCRIPT_PATH = pathlib.Path(__file__).resolve().parents[2] / 'bin' / 'metafor-audit.py'
OPTION = '--metafopy-src'


def load_script() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location('metafor_audit_script', SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script = load_script()

# arguments of an argv that are not the option: they never start with it
ARGUMENT = hypothesis.strategies.text(max_size=12).filter(lambda a: not a.startswith(OPTION))


def test_metafopy_source_reads_the_option_in_both_spellings() -> None:
    """``metafopy_source``: ``--metafopy-src PATH`` and ``--metafopy-src=PATH`` give PATH; no option gives ``None``."""
    assert script.metafopy_source(['--metafopy-src', '/old/src']) == '/old/src'
    assert script.metafopy_source(['--metafopy-src=/old/src']) == '/old/src'
    assert script.metafopy_source(['--quick']) is None
    assert script.metafopy_source([]) is None


def test_metafopy_source_ignores_an_option_without_its_value() -> None:
    """``metafopy_source``: ``--metafopy-src`` as the last argument has no value, so it gives ``None``."""
    assert script.metafopy_source(['--quick', OPTION]) is None


def test_metafopy_source_keeps_an_equals_sign_in_the_value() -> None:
    """``metafopy_source``: only the first ``=`` splits the option from its value."""
    assert script.metafopy_source([f'{OPTION}=a=b']) == 'a=b'


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(
    value=hypothesis.strategies.text(max_size=20),
    before=hypothesis.strategies.lists(ARGUMENT, max_size=4),
    after=hypothesis.strategies.lists(ARGUMENT, max_size=4),
)
def test_metafopy_source_gives_the_value_in_both_spellings(value: str, before: list[str], after: list[str]) -> None:
    """``metafopy_source`` (T2): for any value and surrounding argv, both spellings give the value."""
    assert script.metafopy_source([*before, OPTION, value, *after]) == value
    assert script.metafopy_source([*before, f'{OPTION}={value}', *after]) == value


@hypothesis.settings(max_examples=100, deadline=None)
@hypothesis.given(argv=hypothesis.strategies.lists(ARGUMENT, max_size=6))
def test_metafopy_source_is_none_without_the_option(argv: list[str]) -> None:
    """``metafopy_source`` (T2): an argv without the option gives ``None``."""
    assert script.metafopy_source(argv) is None
