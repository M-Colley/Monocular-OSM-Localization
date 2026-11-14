"""Descriptor receipts are checked against the run's own declared retrieval source,
whichever path separator the run recorded (Windows runs write ``monocular_osm\\x.py``)."""

from scripts import assisted_goal_audit as audit

SHA, OTHER = "9a" * 32, "0b" * 32


def receipts(*shas):
    return [{"receipt": {"kv_source_sha256": sha}} for sha in shas]


def test_windows_recorded_source_keys_match_their_receipts():
    declared = {"monocular_osm\\kartaview_vpr.py": SHA, "main.py": OTHER}
    assert audit.posix_sources(declared) == {"monocular_osm/kartaview_vpr.py": SHA, "main.py": OTHER}
    assert audit.descriptor_receipts_match_source(receipts(SHA, SHA), declared)
    assert audit.descriptor_receipts_match_source(receipts(SHA), {"monocular_osm/kartaview_vpr.py": SHA})


def test_a_receipt_from_another_retrieval_source_is_refused():
    assert not audit.descriptor_receipts_match_source(receipts(SHA, OTHER), {"monocular_osm\\kartaview_vpr.py": SHA})


def test_missing_receipt_and_undeclared_source_never_match_each_other():
    # Before the fix both sides of the comparison were None for Windows runs.
    assert not audit.descriptor_receipts_match_source([{"receipt": {}}], {"monocular_osm\\other.py": SHA})
    assert not audit.descriptor_receipts_match_source(receipts(SHA), {"main.py": SHA})
    assert audit.descriptor_receipts_match_source([], {"main.py": SHA})          # nothing to check
