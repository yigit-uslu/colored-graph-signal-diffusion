"""Source provenance — **stdlib only, and that is the whole point of the module existing.**

⭐ **Plan revision 22, from the rev-21 review, which found the provenance race a second time.**

Revision 21 moved the timing closure's capture to `fmri_a0_timing`'s module body, reasoning that
import is when the process binds its code. That was the right principle applied one level too late:
by the time a module's own body executes, **every module it imports has already been imported**. So
`SOURCE_AT_IMPORT` was computed after `fmri_data`, `fmri_arms`, `ccd.*` and the rest were live, and
an edit landing in that window is seen by *both* the before and after captures — they agree, and the
artifact certifies bytes the process never executed. Exactly the defect the guard was built for,
moved from a multi-hour window to a one-second one.

Closing it requires capturing the closure **before importing any of it**, which requires the
capturing code to import none of it. Hence this module: it computes the digests, and it depends on
nothing but the standard library. `fmri_protocol` and `fmri_seal` now source their provenance
primitives here rather than defining them, so there is one implementation, not three.

The AST walk is a pure function of the *source text* — it never imports what it hashes — so the
launcher (`fmri_timing_launch.py`) can fingerprint the whole timing closure while `torch` is still
unloaded.

**The irreducible residue is this file itself.** Nothing can hash the hasher before the hasher runs.
The launcher therefore reads *these bytes* with `hashlib` directly, before importing this module, and
re-checks them at the end — a five-line stdlib preamble that reduces the exposure from 33 files to
zero.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

__all__ = ["CALIBRATION_ENTRY", "FLOOR_ENTRY", "HEADS_ENTRY", "LaunchGuard", "REPO_ROOT",
           "fingerprint_is_self_consistent",
           "SEAL_ENTRY", "SOURCE_ROOTS", "TIMING_ENTRY", "canonical_json", "closure_contamination",
           "sha256_file", "source_fingerprint", "source_race", "transitive_sources"]

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Repo-local roots the closure may reach, as `(path, is_package)`.
#:
#: ⭐ Rev 20, from the step-8 review: `baselines/gad` was missing and it is load-bearing. Both
#: scoring modules `sys.path`-insert it and then `from signal_metrics import …` — `gad_eval` takes
#: the descriptor and MMD implementations, `fmri_bootstrap` takes `_pairwise_sq_dists`, which is the
#: kernel every floor and every M0 score is built from. It is vendored *because* it must not drift,
#: and that is exactly the code a fingerprint has to cover. Flat, like `scripts/`.
SOURCE_ROOTS: tuple[tuple[str, bool], ...] = (
    ("src", True), ("scripts", False), ("baselines/gad", False),
)

#: Entry points whose transitive closure an artifact's fingerprint covers. A *closure*, not a list.
#: Every writer names its own: a caller that takes someone else's records a fingerprint of code it
#: did not run, which is the defect the step-6 review returned.
SEAL_ENTRY: tuple[str, ...] = ("fmri_seal",)

#: The timing harness's entry point. Defined **here** rather than in `fmri_a0_timing` so the launcher
#: can name the closure without importing the module that constitutes it — and so there is one
#: definition instead of a launcher-side copy that must track it.
TIMING_ENTRY: tuple[str, ...] = ("fmri_a0_timing",)

#: Stage A's writers, here for the same reason `TIMING_ENTRY` is: a launcher must be able to name
#: the closure without importing the module that constitutes it (D6, revision-5 review).
CALIBRATION_ENTRY: tuple[str, ...] = ("fmri_stage_a",)
HEADS_ENTRY: tuple[str, ...] = ("fmri_heads",)
#: §3.5's reference floor. It writes scientific output from the fixed data, so it gets the same
#: capture-before-import treatment as the other two rather than the weaker one `fmri_bandwidths`
#: predates — the exposure is smaller, but "smaller" was explicitly rejected as the standard.
FLOOR_ENTRY: tuple[str, ...] = ("fmri_reference_floor",)


def canonical_json(obj: Any) -> str:
    """Key-sorted, whitespace-free JSON. The hash input, so it must be order-insensitive."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _module_file(dotted: str) -> Path | None:
    """Resolve a dotted module name to a repo-local file, or None if it is not ours."""
    parts = dotted.split(".")
    for root, packaged in SOURCE_ROOTS:
        base = REPO_ROOT / root
        if packaged:
            cand = base.joinpath(*parts)
            for p in (cand.with_suffix(".py"), cand / "__init__.py"):
                if p.is_file():
                    return p
        elif len(parts) == 1 and (base / f"{parts[0]}.py").is_file():
            return base / f"{parts[0]}.py"
    return None


def _imports_of(path: Path, package: tuple[str, ...]) -> set[str]:
    """Dotted names imported by one module, with relative imports resolved against `package`."""
    out: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:                                  # `from .rate import ColouringRate`
                base = package[:len(package) - node.level + 1]
                head = ".".join(base + ((node.module,) if node.module else ()))
            else:
                head = node.module or ""
            if not head:
                continue
            out.add(head)
            # `from ccd.forward import bands` — `bands` is a submodule, not an attribute, and it is
            # the file that holds the band functions. Try both readings; `_module_file` discards
            # whichever does not exist.
            out.update(f"{head}.{a.name}" for a in node.names)
    return out


def transitive_sources(entry: Sequence[str]) -> tuple[str, ...]:
    """Every repo-local module reachable from `entry` by imports, as sorted repo-relative paths.

    ⭐ **Rev 3, from the step-6 review, and the fix is a class of fix rather than four more lines in
    a list.** `SOURCE_FILES` was a hand-written tuple built for the *seal*, and `fmri_g0_audit`
    reused it — so the G0 fingerprint covered the sealer's dependencies and **not its own**:
    `fmri_g0_audit.py`, `fmri_kstar_audit.py` (which builds the clocks), `ccd/forward/clock_audit.py`
    and `ccd/denoiser/denoiser.py` were all absent. Changing the G0 classification or the
    Gaussianisation formula would have left `combined_sha256` unchanged, which is the exact opposite
    of what the field claims.

    A hand-written *audit-specific* list would have been the same mistake in a new place — this repo
    has now been bitten eight times by one. So the closure is derived instead, by parsing imports.

    **Static, not `sys.modules`.** Walking the live import table would also be complete, but it
    depends on what else happened to be loaded — the same script would fingerprint differently under
    pytest than from the CLI, and a provenance record that changes with its observer is not one.
    An AST walk is a deterministic function of the source, and — the property revision 22 needs —
    it can be evaluated **without importing anything it describes**.

    **It over-covers, deliberately.** A module imported for an unrelated reason still enters the
    closure, so an irrelevant edit moves the fingerprint. That is the safe direction: over-coverage
    reports "different code" when nothing relevant changed (an annoyance the reader can check),
    while under-coverage reports "same code" when the formula moved (a false provenance claim).

    🛑 **Rev 7 — it was UNDER-covering, which is the unsafe direction, and the paragraph above was
    wrong about that.** Resolving `ccd.operators.backends` to one file ignores the `__init__.py`
    chain Python executes first, and `ccd/operators/__init__.py` eagerly re-exports `circulant` and
    `bracket`. **12 of 14 entry points were short by 4-6 files, across 33 sealed artifacts.** No
    measured number was wrong and no `protocol_id` moved -- but an edit to `operators/__init__.py`
    would have left every `combined_sha256` unchanged, which is precisely the false "same code" this
    docstring claims cannot happen. Reported by the `hmclock` stream
    (`docs/agent_summaries/fmri_provenance_closure_defect_handoff_2026-08-13.md`); fixed below.

    ⭐ **The repair is not what makes this safe — the CONTROL is.** Their walker and this one agreed
    for weeks because they share a design, and agreement between two implementations of one
    algorithm cannot test the algorithm. What can is a different *modality*:
    `tests/test_fmri_provenance_runtime.py` imports each entry in a FRESH SUBPROCESS and asserts
    `runtime subset-of static`. One-sided on purpose -- a static walk may legitimately over-cover
    (an import inside a never-called function still counts), and over-coverage is checkable while
    under-coverage is a false provenance claim.
    """
    seen: dict[str, Path] = {}
    queue = list(entry)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        path = _module_file(name)
        if path is None:
            continue                                        # third-party or stdlib; not ours to hash
        seen[name] = path
        package = tuple(name.split("."))
        if path.name == "__init__.py":
            package = package + ("",)                       # `.x` inside a package __init__ is `pkg.x`
        queue.extend(_imports_of(path, package))
        # 🛑 **ANCESTOR PACKAGES — the rev-7 repair.** Importing `ccd.operators.backends` does not
        # execute one file: Python first executes `ccd/__init__.py`, then
        # `ccd/operators/__init__.py`, and the latter EAGERLY re-exports `circulant` and `bracket`.
        # Four files ran and none were in the closure. Enqueuing the ancestor package NAMES (rather
        # than manipulating paths) lets the loop above resolve them through `_module_file` — which
        # already maps `ccd.operators` to its `__init__.py` — and walk their imports with the right
        # package tuple, so the re-exports come in too.
        parts = name.split(".")
        queue.extend(".".join(parts[:i]) for i in range(1, len(parts)))
    return tuple(sorted({p.relative_to(REPO_ROOT).as_posix() for p in seen.values()}))


def source_fingerprint(entry: Sequence[str] = SEAL_ENTRY) -> dict:
    """Per-file digests of the implementation, **recorded and deliberately not hashed**.

    ⭐ Rev 2, from the reviewer's scope finding: *the seal binds configuration and data, not source
    code.* `protocol_id` covers what the benchmark was configured to do and which bytes it reads. It
    does not cover how any of it is computed — a changed formula, a rewritten seed resolution or a
    different reduction order leaves the id untouched while the numbers move.

    Hashing source into `protocol_id` would be the wrong fix: a comment change would then invalidate
    every measurement, and the id would stop meaning "the same experiment". The right fix is to make
    the boundary visible and give downstream records an implementation identifier to carry, which is
    what this is. A measurement whose `source_fingerprint` differs from the seal's is not invalid —
    it is a measurement produced by different code, and that is a fact worth being able to see.

    ⭐ **Rev 3: `entry` is the artifact's own entry point**, and the file list is its transitive
    import closure (`transitive_sources`) rather than a shared hand-written tuple. Every writer must
    pass its own — the default covers the sealer, and a caller that takes the default while doing
    something else records a fingerprint of code it did not run.
    """
    files = transitive_sources(entry)
    out = {rel: (sha256_file(REPO_ROOT / rel) if (REPO_ROOT / rel).is_file() else None)
           for rel in files}
    combined = canonical_json(out).encode("utf-8")
    return {"entry": list(entry), "files": out,
            "combined_sha256": hashlib.sha256(combined).hexdigest(),
            "note": ("NOT part of protocol_id. The seal binds configuration and data; this "
                     "identifies the implementation that produced the measurements beside it. "
                     "The file list is the transitive import closure of `entry`, derived by AST "
                     "walk rather than hand-maintained.")}


def fingerprint_is_self_consistent(fp: dict) -> bool:
    """Does a RECORDED fingerprint's `files` map actually hash to its own `combined_sha256`?

    ⭐ Rev 7, from the `hmclock` exchange. Every consumer compares a record's `combined_sha256`
    against a fresh live walk, which is what catches drift in the code — but nothing checked the
    record against *itself*. An artifact whose `files` map had been edited while its digest was left
    intact would pass every gate here, because no gate ever reads that map.

    This is a record-integrity check, not a provenance one: it cannot tell you the code is the same,
    only that the document is not internally contradictory. Both are wanted, and they fail on
    different mutations.
    """
    files = fp.get("files")
    if not isinstance(files, dict) or "combined_sha256" not in fp:
        return False
    return hashlib.sha256(canonical_json(files).encode("utf-8")).hexdigest() == fp["combined_sha256"]


def source_race(before: dict, after: dict) -> tuple[str, ...]:
    """Paths whose digest moved between two captures of a closure.

    Reports appearances and disappearances as well as edits: a file entering the closure changes
    what the next run would do, and one leaving it means the recorded digest describes nothing.
    """
    b, a = before["files"], after["files"]
    return tuple(sorted(k for k in set(b) | set(a) if b.get(k) != a.get(k)))


# ------------------------------------------------------------------ the launcher-side guard
#
# Revision 22 built this inside `fmri_timing_launch`. Revision 5's D6 requires the same guarantee for
# Stage A's writers, and three copies of a capture-order check is how this repository has produced
# nine silent drifts (see the `derive-dont-hand-maintain` rule): the copies stay correct only until
# one of them is fixed. So it lives here, in the one module that is already stdlib-only and already
# hashed before it is imported, and each launcher is a five-line instantiation.

#: Modules exempt from the contamination check, and why each one is unavoidable.
EXEMPT_FROM_CONTAMINATION: dict = {
    "fmri_provenance": "the hasher itself — covered instead by the launcher's pre-import digest",
}


def closure_contamination(fp: dict, *, modules: dict | None = None) -> tuple[str, ...]:
    """Files of `fp` that were **already imported** when it was captured — i.e. captured too late.

    This is what makes the guard a fix rather than a restatement. "Captured early enough" is not
    something a comment can establish: an import added above the capture moves it back after the code
    it describes — exactly how revision 21 regressed — and nothing would notice. So the invariant is
    measured against the live import table.

    It **reports** rather than raises. Raising at import time was the first attempt and it was
    order-dependent: under pytest a sibling module has usually imported `fmri_protocol` before the
    launcher loads, so the launcher became unimportable depending on collection order. A non-empty
    result travels into the run's validity record instead and withholds the verdict, like every other
    invalidating condition. Under the real CLI it is empty, because nothing has been imported yet.

    `modules` is injectable so a test can demonstrate it firing; a gate never shown to reject
    anything is this repository's most-repeated defect.
    """
    import sys

    mods = sys.modules if modules is None else modules
    files = {str((REPO_ROOT / rel).resolve()) for rel in fp["files"]}
    return tuple(sorted(
        name for name, mod in list(mods.items())
        if name not in EXEMPT_FROM_CONTAMINATION
        and (f := getattr(mod, "__file__", None)) and str(Path(f).resolve()) in files
    ))


class LaunchGuard:
    """One writer's pre-import capture: the closure, the contamination check, and the hasher's race.

    Constructed in a launcher's module body **before** the measured module is imported, and handed to
    that module so its artifact records what was captured rather than re-deriving it later. The three
    fields together are the claim "these bytes ran": `source_before` says which bytes,
    `capture_contamination` says the capture was not already too late, and `provenance_race()` says
    the hasher itself did not move underneath the run.
    """

    def __init__(self, entry: Sequence[str], provenance_digest: str) -> None:
        self.entry = tuple(entry)
        #: The hasher's own bytes as the launcher read them, before importing this module.
        self.provenance_at_start = str(provenance_digest)
        #: The closure as it stood before one byte of the measured code was imported.
        self.source_before = source_fingerprint(self.entry)
        #: Empty on the CLI path, which is the one that writes artifacts. Non-empty under pytest —
        #: and then the artifact says so rather than the import blowing up.
        self.capture_contamination = closure_contamination(self.source_before)

    def provenance_race(self) -> tuple[str, ...]:
        """Did the hasher's own source move while we were using it?"""
        now = hashlib.sha256((REPO_ROOT / "scripts" / "fmri_provenance.py").read_bytes()).hexdigest()
        return () if now == self.provenance_at_start else ("scripts/fmri_provenance.py",)

    def race_since_capture(self) -> tuple[str, ...]:
        """Closure files that moved between the capture and now. Disqualifying, never a warning."""
        return source_race(self.source_before, source_fingerprint(self.entry))
