r"""Application backends for functions of ``S`` — matrix-free, or exact-and-dense.

Two objects with one method, so a call site can be routed per callable without knowing which it
holds. Both default to nothing: the caller chooses, and :class:`ChebyshevBackend` reproduces exactly
what the code did before backends existed.

**Why a dense route exists at all** (implementation plan §2.1b, Remedy C — RULED §7.13). On a
ridge-dominated spectrum some band functions cannot be resolved at any admissible Chebyshev degree:

* ``c_out`` carries ``√(λ−τ)``, a branch point **exactly at** the bracket's lower endpoint;
* ``c_skip`` runs ``0 → 1`` across a boundary layer of relative width ~8e-8;
* the terminal ``S^{1/2}`` carries ``√λ``, a branch point a distance ``τ`` below the bracket;
* and — measured at implementation, §2.1b-bis — **every φ-dependent function on a mildly-pinned
  arm**, because the two-point pin's shift ``s = (λ_max − ρλ_min)/(ρ−1)`` collapses as ``|a| → 0``
  and drags φ's own branch point up against the bracket edge.

On this problem's shape (``n > d``) the dense route is also **80× cheaper** than the Chebyshev one
it replaces (5.65 ms vs 452 ms at K=32), so nothing is traded away but claim scope.

**This is not an eigendecomposition.** :class:`SpectralBackend` takes ``(evals, evecs)`` **from the
caller** and never computes them — the same contract as :mod:`ccd.forward.clock_audit`, which is why
this module needs no matrix-free pragma. In particular it must never be built from
``CovarianceOperator._dense()``, whose contract is TEST/ORACLE ONLY; the driver's own recomputed and
validated eigensystem is the source.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import Tensor

from .chebyshev import apply_fun
from .linop import CovarianceOperator

__all__ = ["Backend", "ChebyshevBackend", "SpectralBackend"]

ScalarFn = Callable[[Tensor], Tensor]


class Backend:
    """Interface: ``apply(f, v)`` returns ``f(S) v``."""

    exact: bool = False

    def apply(self, fn: ScalarFn, v: Tensor) -> Tensor:  # pragma: no cover - interface
        raise NotImplementedError


class ChebyshevBackend(Backend):
    """Today's matrix-free path, unchanged. ``K`` matvecs per apply."""

    exact = False

    def __init__(self, op: CovarianceOperator, lo: float, hi: float, K: int | None = None):
        self.op, self.lo, self.hi, self.K = op, float(lo), float(hi), K

    def apply(self, fn: ScalarFn, v: Tensor) -> Tensor:
        return apply_fun(fn, self.op, v, self.lo, self.hi, self.K)

    def __repr__(self) -> str:
        return f"ChebyshevBackend(K={self.K}, bracket=[{self.lo:.4g}, {self.hi:.4g}])"


class SpectralBackend(Backend):
    """Exact application ``U diag(f(λ)) Uᵀ v`` from a caller-supplied eigensystem.

    Exact up to the arithmetic's own precision — there is no approximation to audit, which is why
    the callables routed here are recorded in the seal as *not audited* rather than as passing.

    ``validate_against`` is strongly recommended: an eigensystem that does not diagonalise the
    operator the rest of the pipeline applies would make this backend silently exact for the *wrong*
    matrix, and nothing downstream would notice. The tolerance is dtype-scoped (plan §1.9 S3): the
    float64 construction is checked at ``rtol``, and 1e-12 is unreachable in float32.
    """

    exact = True

    def __init__(self, evals: Tensor, evecs: Tensor, *,
                 validate_against: CovarianceOperator | None = None, rtol: float = 1e-12):
        evals = torch.as_tensor(evals).reshape(-1)
        evecs = torch.as_tensor(evecs)
        if evecs.ndim != 2 or evecs.shape[0] != evecs.shape[1] or evecs.shape[0] != evals.numel():
            raise ValueError(
                f"need a square evecs matching evals; got evecs{tuple(evecs.shape)}, "
                f"{evals.numel()} eigenvalues"
            )
        if not (torch.isfinite(evals).all() and torch.isfinite(evecs).all()):
            raise ValueError("eigensystem contains non-finite entries")
        self.evals, self.evecs = evals, evecs
        self._cache: dict[tuple, tuple[Tensor, Tensor]] = {}

        if validate_against is not None:
            self._validate(validate_against, rtol)

    def _validate(self, op: CovarianceOperator, rtol: float) -> None:
        """``UᵀU ≈ I`` **and** ``S U ≈ U diag(λ)``, the latter by MATVEC — never by forming ``S``.

        Checking against `op` and not against a dense matrix is the point: it ties this backend to
        the same operator the matrix-free residue applies, which is what makes routing a function
        here rather than there a numerical choice instead of a change of subject.

        **Both clauses are needed.** Correspondence alone accepts an arbitrarily **rescaled**
        eigenvector: scaling a column preserves ``S u = λ u`` exactly, while ``U diag(f) Uᵀ`` is then
        wrong by that column's scale squared. The orthonormality clause is what closes it.
        """
        d = self.evals.numel()
        U64 = self.evecs.to(torch.float64)
        # `device=U64.device` is load-bearing, not defensive: `torch.eye` defaults to CPU, so on a
        # CUDA eigensystem this line raised "Expected all tensors to be on the same device" and the
        # backend could not be CONSTRUCTED at all. The correspondence clause below never had the bug
        # because it moves `U` onto `op.device` explicitly. Nothing caught it because every test and
        # every smoke run to date built this backend on CPU, where the default is already right.
        ortho = float(torch.linalg.norm(U64.transpose(0, 1) @ U64
                                        - torch.eye(d, dtype=torch.float64,
                                                    device=U64.device))) / d ** 0.5
        if not (ortho <= rtol):
            raise ValueError(
                f"supplied eigenvectors are not orthonormal: ||U^T U - I||_F/sqrt(d) = {ortho:.3e} "
                f"> rtol={rtol:.1e}. A rescaled column still satisfies S u = lambda u, so the "
                "correspondence check below cannot catch it -- but U diag(f) U^T would be wrong."
            )
        U = self.evecs.to(dtype=op.dtype, device=op.device)
        lhs = op.matvec(U)
        rhs = U * self.evals.to(dtype=op.dtype, device=op.device)[None, :]
        num = float(torch.linalg.norm(lhs - rhs))
        den = float(torch.linalg.norm(rhs))
        rel = num / den if den > 0 else num
        if not (rel <= rtol):
            raise ValueError(
                f"supplied eigensystem does not diagonalise the operator: "
                f"||SU - U diag(lam)||/||U diag(lam)|| = {rel:.3e} > rtol={rtol:.1e} "
                f"(dtype {op.dtype}). A dense apply against the wrong eigensystem is exact for the "
                "wrong matrix and nothing downstream would notice."
            )

    def _for(self, v: Tensor) -> tuple[Tensor, Tensor]:
        key = (v.dtype, v.device)
        if key not in self._cache:
            self._cache[key] = (self.evals.to(dtype=v.dtype, device=v.device),
                                self.evecs.to(dtype=v.dtype, device=v.device))
        return self._cache[key]

    def apply(self, fn: ScalarFn, v: Tensor) -> Tensor:
        """``f(S) v`` for ``v`` of shape ``(d,)`` or ``(d, batch)``. **Rank-preserving.**

        A 1-D input used to be broadcast into a ``(d, d)`` matrix by the unconditional ``f[:, None]``
        — silently, and with a plausible-looking result. Rank is preserved explicitly and the
        leading dimension is validated, because a shape bug on this path corrupts every draw rather
        than raising.
        """
        lam, U = self._for(v)
        d = lam.numel()
        if v.ndim not in (1, 2):
            raise ValueError(f"v must be (d,) or (d, batch); got {tuple(v.shape)}")
        if v.shape[0] != d:
            raise ValueError(f"v has leading dimension {v.shape[0]}, but the eigensystem has d={d}")
        f = fn(lam)
        if f.shape != lam.shape:
            raise ValueError(f"band function returned {tuple(f.shape)} for {tuple(lam.shape)} evals")

        vec = v.ndim == 1
        w = v.unsqueeze(1) if vec else v
        out = U @ (f[:, None] * (U.transpose(0, 1) @ w))
        return out.squeeze(1) if vec else out

    def __repr__(self) -> str:
        return f"SpectralBackend(d={self.evals.numel()}, exact)"
