"""Regression test for a Windows-only failure: legacy console code pages cannot encode the report's characters."""

import io

import pytest

from pda.cli import make_output_safe

REPORT_LINE = "> \"it broke\"  \n> — quote, 2★, 2026-07-03 · mean rating 2.5 …"


def legacy_console() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict", write_through=True)


def test_report_characters_crash_a_legacy_windows_console_without_the_fix():
    out = legacy_console()
    with pytest.raises(UnicodeEncodeError):
        out.write(REPORT_LINE)  # this is exactly what `pda themes` did on Windows PowerShell


def test_make_output_safe_replaces_instead_of_crashing_and_keeps_ascii():
    out = legacy_console()
    make_output_safe(out)
    out.write(REPORT_LINE)
    out.flush()
    text = out.buffer.getvalue().decode("cp1252")
    assert "2?," in text and "it broke" in text and "mean rating 2.5" in text


def test_streams_without_reconfigure_are_left_alone():
    make_output_safe(object(), io.StringIO())  # must not raise
