"""The admission occurrence binding capability is advertised (cond-0842)."""

from __future__ import annotations


def test_capabilities_advertise_admission_occurrence_binding(client):
    advertised = client.get("/managed-launch/capabilities").json()
    assert advertised["admission_occurrence_binding"] is True
