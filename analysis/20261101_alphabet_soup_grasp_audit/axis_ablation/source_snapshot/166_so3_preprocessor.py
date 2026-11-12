"""Small, independently verified SO(3) tolerance-box preprocessor.

No joint variables or robot objective are used here. The default objective is
half the squared Euclidean distance to the latest Euler box. Zero/one
dimensional problems are analytic; multi-axis expansion is a local approximation.
Optional TSR scalar recovery instead scans one common face-expansion coefficient.
An interval upper bound certifies incompatibility before any box is expanded.
Unresolved searches fail closed and are distinguished from proven emptiness.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
import time

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from ..geometry import fixed_xyz_rotation_angles


@dataclass(frozen=True)
class SO3PreprocessorSettings:
    maximum_step_rad: float = float(np.sqrt(3.) * np.radians(4.5))
    semantic_lower: tuple[float, float, float] = tuple(-np.radians((30., 30., 30.)))
    semantic_upper: tuple[float, float, float] = tuple(np.radians((30., 30., 30.)))
    acceptance_mode: str = 'monitor'
    max_iterations: int = 30
    certificate_nodes: int = 256
    verification_tolerance_rad: float = 1.e-8
    publication_tolerance_rad: float = 1.e-6
    expansion_objective: str = 'squared_l2'
    max_starts: int = 3
    recovery_strategy: str = 'joint'
    scalar_scan_steps: int = 16
    scalar_bisection_steps: int = 24

    def __post_init__(self):
        lo, hi = np.array(self.semantic_lower), np.array(self.semantic_upper)
        if (lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi]))
                or np.any(lo > 0.) or np.any(hi < 0.) or np.any(lo >= hi)
                or np.any(abs(lo[[0, 2]]) >= np.pi) or np.any(abs(hi[[0, 2]]) >= np.pi)
                or abs(lo[1]) >= np.pi/2 or abs(hi[1]) >= np.pi/2):
            raise ValueError('Semantic bounds must contain zero on a nonsingular local XYZ branch')
        if not np.isfinite(self.maximum_step_rad) or not 0. < self.maximum_step_rad < np.pi:
            raise ValueError('SO(3) step budget must be in (0, pi)')
        if self.acceptance_mode not in ('monitor', 'reject'):
            raise ValueError('SO(3) acceptance mode must be monitor or reject')
        if self.max_iterations < 1 or self.certificate_nodes < 1 or not 1 <= self.max_starts <= 3:
            raise ValueError('Search limits must be positive')
        if self.expansion_objective not in ('squared_l2', 'l1'):
            raise ValueError('Expansion objective must be squared_l2 or l1')
        if self.recovery_strategy not in ('joint', 'tsr_scalar'):
            raise ValueError('Recovery strategy must be joint or tsr_scalar')
        if (not isinstance(self.scalar_scan_steps, int) or not 1 <= self.scalar_scan_steps <= 256
                or not isinstance(self.scalar_bisection_steps, int)
                or not 1 <= self.scalar_bisection_steps <= 64):
            raise ValueError('Invalid scalar search limits')
        if not 0. < self.verification_tolerance_rad < 1.e-3:
            raise ValueError('Invalid SO(3) verification tolerance')
        if not 0. < self.publication_tolerance_rad < 1.e-3:
            raise ValueError('Invalid FK publication tolerance')


@dataclass(frozen=True)
class SO3BoxResult:
    lower: np.ndarray
    upper: np.ndarray
    witness: np.ndarray | None
    witness_angle_rad: float | None
    expansion_rad: float | None
    status: str
    latest_compatible: bool | None
    incompatibility_proven: bool
    global_minimum_proven: bool
    elapsed_ms: float
    dimension: int
    certificate_nodes: int = 0
    secondary_rule: str = 'among verified candidates: minimum carry-coordinate squared distance, then lexicographic'
    # Keep expansion_rad as the sum of face expansions for older log readers.
    expansion_squared_rad2: float | None = None
    expansion_objective: str = 'squared_l2'
    objective_value: float | None = None
    optimization_starts: int = 0
    recovery_strategy: str = 'joint'
    scalar_coefficient: float | None = None
    scalar_safe_coefficient: float | None = None
    scalar_semantic_cap: float | None = None
    scalar_safe_source: str | None = None
    scalar_angle_evaluations: int = 0
    tsr_projection_distance_rad: float | None = None
    tsr_strict_residual_rad: float | None = None

    @property
    def usable(self):
        return self.witness is not None

    def as_dict(self):
        return {name: value.tolist() if isinstance(value, np.ndarray) else value
                for name, value in self.__dict__.items()}


def rotation_distance(a, b):
    """Principal SO(3) angle, stable at zero/pi and quaternion-sign invariant."""
    q = Rotation.from_matrix(np.asarray(a) @ np.asarray(b).T).as_quat()
    return float(2.*np.arctan2(np.linalg.norm(q[:3]), abs(q[3])))


def _quat_and_jac(e):
    """Scalar-first quaternion of Rz Ry Rx and its exact 4x3 derivative."""
    x, y, z = e / 2.
    a, b, c = np.cos((x, y, z))
    d, f, g = np.sin((x, y, z))
    q = np.array((a*b*c+d*f*g, d*b*c-a*f*g, a*f*c+d*b*g, a*b*g-d*f*c))
    jac = .5*np.array((
        (-d*b*c+a*f*g, -a*f*c+d*b*g, -a*b*g+d*f*c),
        (a*b*c+d*f*g, -d*f*c-a*b*g, -d*b*g-a*f*c),
        (-d*f*c+a*b*g, a*b*c-d*f*g, -a*f*g+d*b*c),
        (-d*b*g-a*f*c, -a*f*g-d*b*c, a*b*c+d*f*g),
    ))
    return q, jac


def _intervals(a, b, lower, upper, threshold):
    radius = float(np.hypot(a, b))
    if radius < threshold - 2.e-15:
        return []
    halfwidth = 2.*np.arccos(np.clip(threshold / max(radius, 1.e-300), -1., 1.))
    center = 2.*np.arctan2(b, a)
    result = []
    for k in range(-3, 4):
        left, right = max(lower, center+2*k*np.pi-halfwidth), min(upper, center+2*k*np.pi+halfwidth)
        if left <= right + 2.e-14:
            result.append((left, max(left, right)))
    return result


def _box_expansion_objective(e, lower, upper, kind='squared_l2'):
    """Value and gradient in radians; squared_l2 has units rad^2.

    Eliminating nonnegative lower/upper face-expansion variables gives exactly
    this residual. For squared_l2 the gradient is continuous at both faces.
    """
    residual = e-np.clip(e, lower, upper)
    if kind == 'squared_l2':
        return float(.5*(residual@residual)), residual
    if kind == 'l1':
        return float(np.sum(abs(residual))), np.sign(residual)
    raise ValueError('Unknown expansion objective')


def preprocess_so3_box(previous_rotation, reference_rotation, frame, carried_coordinates,
                       lower, upper, eligible, settings=SO3PreprocessorSettings()):
    started = time.perf_counter()
    previous, reference, frame = (np.asarray(v, dtype=float) for v in
                                   (previous_rotation, reference_rotation, frame))
    for r in (previous, reference, frame):
        if (r.shape != (3, 3) or not np.all(np.isfinite(r))
                or not np.allclose(r.T@r, np.eye(3), atol=1.e-6) or np.linalg.det(r) < 0.):
            raise ValueError('Expected proper finite rotation matrices')
    lower, upper, carry = (np.asarray(v, dtype=float).copy() for v in (lower, upper, carried_coordinates))
    eligible = np.asarray(eligible, dtype=bool)
    if any(v.shape != (3,) for v in (lower, upper, carry, eligible)):
        raise ValueError('Bounds, carry and mask must have three components')
    if (not np.all(np.isfinite([lower, upper, carry])) or np.any(lower > 0.)
            or np.any(upper < 0.)):
        raise ValueError('Finite carry and signed bounds containing zero required')
    sem_lo = np.where(eligible, settings.semantic_lower, 0.)
    sem_hi = np.where(eligible, settings.semantic_upper, 0.)
    lower, upper = np.where(eligible, lower, 0.), np.where(eligible, upper, 0.)
    if np.any(lower < sem_lo-1.e-12) or np.any(upper > sem_hi+1.e-12):
        raise ValueError('Latest prediction exceeds configured stage semantic bounds')
    axes = np.flatnonzero(eligible)
    dimension = len(axes)
    target = frame.T @ previous @ reference.T @ frame
    xyzw = Rotation.from_matrix(target).as_quat()
    target_q = np.r_[xyzw[3], xyzw[:3]]
    target_coordinates = fixed_xyz_rotation_angles(target)
    threshold = np.cos(settings.maximum_step_rad/2.)
    nodes_used = 0
    starts_used = 0
    scalar_diagnostics = {}

    def dot_grad(e):
        q, jac = _quat_and_jac(e)
        return float(target_q@q), target_q@jac

    def theta(e):
        # Independent quaternion composition, rather than the search dot test.
        candidate = Rotation.from_euler('xyz', e).as_matrix()
        return rotation_distance(candidate, target)

    def valid(e, lo, hi):
        return (e is not None and np.all(e >= lo-1.e-10) and np.all(e <= hi+1.e-10)
                and theta(e) <= settings.maximum_step_rad+settings.verification_tolerance_rad)

    def distance(e):
        return float(np.sum(np.maximum.reduce((lower-e, np.zeros(3), e-upper))))

    def cost(e):
        return _box_expansion_objective(e, lower, upper, settings.expansion_objective)[0]

    def finish(e, status, compatible, proven, global_minimum):
        if e is None:
            effective_lo, effective_hi, angle, expansion = lower, upper, None, None
        else:
            assert valid(e, sem_lo, sem_hi)
            effective_lo, effective_hi = np.minimum(lower, e), np.maximum(upper, e)
            effective_lo[~eligible] = effective_hi[~eligible] = 0.
            angle, expansion = theta(e), distance(e)
            if compatible:
                effective_lo, effective_hi = lower.copy(), upper.copy()
        squared = None if e is None else float(np.sum((e-np.clip(e, lower, upper))**2))
        return SO3BoxResult(effective_lo, effective_hi, e, angle, expansion, status,
                            compatible, proven, global_minimum,
                            (time.perf_counter()-started)*1000., dimension, nodes_used,
                            expansion_squared_rad2=squared,
                            expansion_objective=settings.expansion_objective,
                            objective_value=None if e is None else cost(e),
                            optimization_starts=starts_used,
                            recovery_strategy=settings.recovery_strategy,
                            **scalar_diagnostics)

    def analytic_box(lo, hi):
        free = np.flatnonzero(hi-lo > 1.e-13)
        e = (lo+hi)/2.
        if not len(free):
            return e if valid(e, lo, hi) else None
        if len(free) != 1:
            raise ValueError('Analytic solve requires at most one free coordinate')
        axis = free[0]
        e[axis] = 0.
        a = dot_grad(e)[0]
        e[axis] = np.pi
        b = dot_grad(e)[0]
        choices = []
        for left, right in _intervals(a, b, lo[axis], hi[axis], threshold):
            candidate = ((lo+hi)/2.).copy()
            candidate[axis] = np.clip(carry[axis], left, right)
            if valid(candidate, lo, hi):
                choices.append(candidate)
        return min(choices, key=lambda v: (cost(v), np.sum((v-carry)**2), tuple(v))) if choices else None

    def certificate_empty(lo, hi):
        """Taylor interval upper bound on |q(e) dot q_target|, not a grid claim.

        Every quaternion second partial derivative has norm <=1/4. Taylor
        remainder <= (sum half-widths)^2/8, including mixed partials.
        A bounded search may return unknown; it never labels unknown empty.
        """
        nonlocal nodes_used
        # XYZ pitch is the elevation of the body's X direction. Its change
        # cannot exceed the angle between those unit directions, hence cannot
        # exceed the full SO(3) distance. This exact lower bound resolves very
        # thin incompatible boxes without subdividing them.
        pitch_gap = max(lo[1]-target_coordinates[1], 0., target_coordinates[1]-hi[1])
        if pitch_gap > settings.maximum_step_rad+2.e-12:
            return True
        queue, serial = [], itertools.count()

        def insert(a, b):
            middle, radius = (a+b)/2., (b-a)/2.
            value, grad = dot_grad(middle)
            bound = abs(value)+float(abs(grad)@radius)+float(np.sum(radius)**2)/8.+2.e-14
            if bound >= threshold:
                heapq.heappush(queue, (-bound, next(serial), a, b))
        insert(lo, hi)
        for _ in range(settings.certificate_nodes):
            if not queue:
                return True
            _, _, a, b = heapq.heappop(queue)
            nodes_used += 1
            axis = int(np.argmax(b-a))
            if b[axis]-a[axis] < 1.e-10:
                return False
            mid = (a[axis]+b[axis])/2.
            left, right = b.copy(), a.copy()
            left[axis], right[axis] = mid, mid
            insert(a, left); insert(right, b)
        return not queue

    def find_in_box(lo, hi):
        free = np.flatnonzero(hi-lo > 1.e-13)
        if len(free) <= 1:
            return analytic_box(lo, hi), True
        base = (lo+hi)/2.
        qcoords = target_coordinates
        seeds = [np.clip(qcoords, lo, hi), np.clip(carry, lo, hi), base]
        for seed in seeds:
            if valid(seed, lo, hi):
                return seed, True
        def evaluate(v):
            e = base.copy(); e[free] = v
            value, grad = dot_grad(e)
            return -value*value, -2.*value*grad[free]
        for seed in seeds[:2]:
            result = minimize(evaluate, seed[free], jac=True, method='L-BFGS-B',
                              bounds=list(zip(lo[free], hi[free])),
                              options={'maxiter': settings.max_iterations, 'ftol': 1.e-14, 'gtol': 1.e-10})
            e = base.copy(); e[free] = result.x
            if valid(e, lo, hi):
                return e, True
        return None, certificate_empty(lo, hi)

    exact, classified = find_in_box(lower, upper)
    if exact is not None:
        return finish(exact, 'latest_compatible', True, False, True)
    if not classified:
        return finish(None, 'latest_compatibility_unresolved', None, False, False)
    if settings.recovery_strategy == 'tsr_scalar':
        # This path deliberately does not minimize over independent face widths.
        # All latest-box searches above remain joint, so inside axes are free.
        b = target_coordinates
        p = np.clip(b, lower, upper)
        delta = np.where(eligible, b-p, 0.)
        g = float(np.linalg.norm(delta))
        fixed = float(np.linalg.norm(b[~eligible]))
        cap = 1.
        for axis in axes:
            if delta[axis] > 0.:
                cap = min(cap, (sem_hi[axis]-p[axis])/delta[axis])
            elif delta[axis] < 0.:
                cap = min(cap, (sem_lo[axis]-p[axis])/delta[axis])
        cap = float(np.clip(cap, 0., 1.))
        scalar_diagnostics.update(scalar_semantic_cap=cap,
            tsr_projection_distance_rad=float(np.linalg.norm(b-p)),
            tsr_strict_residual_rad=fixed, scalar_angle_evaluations=0)

        def path(alpha):
            return p+alpha*delta

        def reachable(alpha):
            scalar_diagnostics['scalar_angle_evaluations'] += 1
            # Use the actual principal angle, including strict-axis residuals.
            return theta(path(alpha)) <= settings.maximum_step_rad

        safe = None
        rho = settings.maximum_step_rad/np.sqrt(3.)
        if fixed <= rho and g > 0.:
            analytic = max(0., 1.-np.sqrt(max(0., rho*rho-fixed*fixed))/g)
            if analytic <= cap and reachable(analytic):
                safe = analytic
                scalar_diagnostics['scalar_safe_source'] = 'euler_l2_bound'
        # The conservative Euler bound may fail even when the SO(3) angle fits.
        if safe is None and reachable(cap):
            safe = cap
            scalar_diagnostics['scalar_safe_source'] = 'verified_semantic_endpoint'
        if safe is None:
            # Nonmonotone paths may have a feasible interior but no feasible end.
            for alpha in np.linspace(0., cap, settings.scalar_scan_steps+1)[1:-1]:
                if reachable(float(alpha)):
                    safe = float(alpha)
                    scalar_diagnostics['scalar_safe_source'] = 'verified_scan'
                    break
        if safe is None:
            # A feasible semantic pose elsewhere is not a certificate on this
            # fixed ray. Report its limitation rather than expand new directions.
            semantic, resolved = find_in_box(sem_lo, sem_hi)
            status = ('semantic_set_empty' if semantic is None and resolved
                      else 'scalar_path_unresolved')
            return finish(None, status, False, True, False)
        scalar_diagnostics['scalar_safe_coefficient'] = safe
        left, right = 0., safe
        for alpha in np.linspace(0., safe, settings.scalar_scan_steps+1)[1:]:
            if reachable(float(alpha)):
                right = float(alpha)
                break
            left = float(alpha)
        # Retain the verified right endpoint. No monotonicity or earliest
        # crossing is assumed; finite scans can miss narrow earlier components.
        for _ in range(settings.scalar_bisection_steps):
            middle = (left+right)/2.
            if reachable(middle):
                right = middle
            else:
                left = middle
        scalar_diagnostics['scalar_coefficient'] = right
        return finish(path(right), 'expanded_tsr_scalar', False, True, False)
    witness, semantic_classified = find_in_box(sem_lo, sem_hi)
    if witness is None:
        return finish(None, 'semantic_set_empty' if semantic_classified else 'semantic_search_unresolved',
                      False, True, False)
    if dimension <= 1:
        # Find the entire reachable interval, then globally minimize the exact
        # scalar interval distance and apply the deterministic secondary rule.
        if dimension == 1:
            axis = axes[0]; zero = np.zeros(3)
            a = dot_grad(zero)[0]; zero[axis] = np.pi; b = dot_grad(zero)[0]
            choices = []
            for left, right in _intervals(a, b, sem_lo[axis], sem_hi[axis], threshold):
                overlap_lo, overlap_hi = max(left, lower[axis]), min(right, upper[axis])
                value = (np.clip(carry[axis], overlap_lo, overlap_hi) if overlap_lo <= overlap_hi
                         else right if right < lower[axis] else left)
                e = np.zeros(3); e[axis] = value
                if valid(e, sem_lo, sem_hi):
                    choices.append(e)
            witness = min(choices, key=lambda v: (cost(v), np.sum((v-carry)**2), tuple(v)))
        return finish(witness, 'expanded_analytic', False, True, True)

    # A feasible start is retained even if every local minimizer fails.
    # All candidate outputs are independently verified in SO(3).
    candidates = [witness]
    # Deterministic feasible fallback toward the newest box. Every retained
    # point is checked; no monotonicity or globally shortest path is assumed.
    toward = np.clip(witness, lower, upper)
    left, right = 0., 1.
    radial = witness.copy()
    for _ in range(22):
        middle = (left+right)/2.
        trial = witness+middle*(toward-witness)
        value, _ = dot_grad(trial)
        if abs(value) >= threshold:
            left, radial = middle, trial
        else:
            right = middle
    if valid(radial, sem_lo, sem_hi):
        candidates.append(radial)
    def objective(v):
        e = np.zeros(3); e[axes] = v
        value, gradient = _box_expansion_objective(e, lower, upper, settings.expansion_objective)
        return value, gradient[axes]
    def constraint(v):
        e = np.zeros(3); e[axes] = v
        value, grad = dot_grad(e)
        return (value*value-threshold*threshold)*100., 200.*value*grad[axes]
    seeds = []
    for seed in (radial, witness, toward):
        if not any(np.allclose(seed, other, rtol=0., atol=1.e-12) for other in seeds):
            seeds.append(seed)
    for seed in seeds[:settings.max_starts]:
        starts_used += 1
        result = minimize(objective, seed[axes], jac=True, method='SLSQP',
                          bounds=list(zip(sem_lo[axes], sem_hi[axes])),
                          constraints=[dict(type='ineq', fun=lambda v: constraint(v)[0],
                                            jac=lambda v: constraint(v)[1])],
                          options={'maxiter': settings.max_iterations, 'ftol': 1.e-11})
        e = np.zeros(3); e[axes] = result.x
        if valid(e, sem_lo, sem_hi):
            candidates.append(e)
    best_cost = min(cost(e) for e in candidates)
    tie_tolerance = 1.e-12 if settings.expansion_objective == 'squared_l2' else 1.e-9
    tied = [e for e in candidates if cost(e) <= best_cost+tie_tolerance]
    best = min(tied, key=lambda v: (float(np.sum((v-carry)**2)), tuple(v)))
    return finish(best, 'expanded_verified_local', False, True, False)
