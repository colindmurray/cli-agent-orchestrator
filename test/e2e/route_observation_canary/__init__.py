"""M10 Codex route-observation exact-build canary authoring (lane C3).

The spec-§7 installed canaries for the dark ``route_observation_codex``
adapter are authored here as code: the five case definitions
(:mod:`cases`), the fixture surfaces (:mod:`fixtures`), and the M10
positive-receipt validator (:mod:`receipt`).  The deterministic unit mirror
in ``test/services/test_route_observation_canary_unit.py`` drives every case
against fake panes; LIVE execution (real codex binary, tmux, paid turns) is
out of scope for this lane and belongs to the M17 activation lane, which
wires the entry points named by ``cases.py`` via :mod:`runner`.
"""
