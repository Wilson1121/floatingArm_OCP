from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pinocchio as pin


@dataclass
class ReachabilityConfig:
    urdf_path: str
    base_frame: str
    ee_frame: str
    eps_pos: float = 0.01
    eps_rot: float = np.deg2rad(5.0)
    max_iters: int = 100
    n_seeds: int = 16
    seed_sigma: float = 0.3
    dt: float = 0.5
    damping: float = 1e-3
    dq_max: float = 0.2
    limit_tolerance: float = 1e-6
    rng_seed: Optional[int] = None


@dataclass
class ReachabilityResult:
    reachable: bool
    q_best: np.ndarray
    err_pos: float
    err_rot: float
    iters: int
    success_seeds: int
    total_seeds: int
    message: str


class EEReachability:
    def __init__(self, cfg: ReachabilityConfig):
        self.cfg = cfg
        self.model = pin.buildModelFromUrdf(cfg.urdf_path)
        self.data = self.model.createData()
        self.base_id = self._require_frame_id(cfg.base_frame)
        self.ee_id = self._require_frame_id(cfg.ee_frame)
        self.lower = np.array(self.model.lowerPositionLimit).astype(float)
        self.upper = np.array(self.model.upperPositionLimit).astype(float)
        self.q_neutral = pin.neutral(self.model)

    def _require_frame_id(self, name: str) -> int:
        frame_id = self.model.getFrameId(name)
        if frame_id < 0 or frame_id >= self.model.nframes:
            raise ValueError(f"Frame '{name}' not found in model.")
        return frame_id

    def _clamp_q(self, q: np.ndarray) -> np.ndarray:
        q_clamped = q.copy()
        for i in range(self.model.nq):
            lo = self.lower[i]
            hi = self.upper[i]
            if np.isfinite(lo):
                q_clamped[i] = max(q_clamped[i], lo)
            if np.isfinite(hi):
                q_clamped[i] = min(q_clamped[i], hi)
        return q_clamped

    def _within_limits(self, q: np.ndarray) -> bool:
        tol = self.cfg.limit_tolerance
        for i in range(self.model.nq):
            lo = self.lower[i]
            hi = self.upper[i]
            if np.isfinite(lo) and q[i] < lo - tol:
                return False
            if np.isfinite(hi) and q[i] > hi + tol:
                return False
        return True

    def _goal_to_se3(self, goal_T_base) -> pin.SE3:
        if isinstance(goal_T_base, pin.SE3):
            return goal_T_base
        goal_T_base = np.asarray(goal_T_base)
        if goal_T_base.shape == (4, 4):
            return pin.SE3(goal_T_base)
        raise ValueError("goal_T_base must be pin.SE3 or 4x4 numpy array.")

    def solve_ik_pose(
        self, goal_T_base: pin.SE3, q_seed: np.ndarray
    ) -> Tuple[bool, np.ndarray, float, float, int]:
        goal_T_base = self._goal_to_se3(goal_T_base)
        if q_seed.shape[0] != self.model.nq:
            raise ValueError(f"q_seed must have size {self.model.nq}.")

        q = self._clamp_q(q_seed)
        best_q = q.copy()
        best_err = (np.inf, np.inf)
        best_iters = 0

        for it in range(self.cfg.max_iters):
            # Forward kinematics and frame placements
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)

            # Current EE pose expressed in base_frame:
            # base_T_ee = (world_T_base)^-1 * world_T_ee
            world_T_base = self.data.oMf[self.base_id]
            world_T_ee = self.data.oMf[self.ee_id]
            base_T_ee = world_T_base.inverse() * world_T_ee

            # Pose error in SE(3) using log map.
            # Note: log6 gives error expressed in the current EE frame.
            err_local = pin.log6(base_T_ee.inverse() * goal_T_base)
            # Convert error to base_frame coordinates for consistency.
            err_base = base_T_ee.act(err_local)

            # Pinocchio Motion is [angular, linear]; we build e6 as [pos, rot]
            # to match the required definition: e_pos = e6[:3], e_rot = e6[3:].
            err_pos = float(np.linalg.norm(err_base.linear))
            err_rot = float(np.linalg.norm(err_base.angular))
            e6 = np.hstack([err_base.linear, err_base.angular])

            if err_pos < best_err[0] or (err_pos == best_err[0] and err_rot < best_err[1]):
                best_err = (err_pos, err_rot)
                best_q = q.copy()
                best_iters = it + 1

            # Check convergence (pose reachability)
            if err_pos <= self.cfg.eps_pos and err_rot <= self.cfg.eps_rot:
                if self._within_limits(q):
                    return True, q, err_pos, err_rot, it + 1

            # EE Jacobian in base frame
            J_local = pin.computeFrameJacobian(
                self.model, self.data, q, self.ee_id, pin.ReferenceFrame.LOCAL
            )
            J_base = base_T_ee.action @ J_local
            # Reorder Jacobian rows to match e6 = [pos; rot] (linear then angular).
            J_reordered = np.vstack([J_base[3:6, :], J_base[0:3, :]])

            # Damped least squares / LM: dq = J^T (J J^T + λ^2 I)^-1 e
            JJt = J_reordered @ J_reordered.T
            A = JJt + (self.cfg.damping ** 2) * np.eye(6)
            dq = J_reordered.T @ np.linalg.solve(A, e6)

            # Step size scaling and per-joint clamping
            dq = self.cfg.dt * dq
            dq = np.clip(dq, -self.cfg.dq_max, self.cfg.dq_max)

            # Integrate and clamp to joint limits
            q = q + dq
            q = self._clamp_q(q)

        return False, best_q, best_err[0], best_err[1], best_iters

    def is_reachable(
        self, goal_T_base: pin.SE3, q0: Optional[np.ndarray] = None
    ) -> ReachabilityResult:
        rng = np.random.default_rng(self.cfg.rng_seed)
        total_seeds = 0
        success_seeds = 0

        best_q = self.q_neutral.copy()
        best_err = (np.inf, np.inf)
        best_iters = 0

        seeds = []
        if q0 is not None:
            seeds.append(q0.copy())
        if len(seeds) < self.cfg.n_seeds:
            for _ in range(self.cfg.n_seeds - len(seeds)):
                noise = rng.normal(0.0, self.cfg.seed_sigma, size=self.model.nq)
                seeds.append(self.q_neutral + noise)

        for q_seed in seeds:
            total_seeds += 1
            success, q_sol, err_pos, err_rot, iters = self.solve_ik_pose(goal_T_base, q_seed)
            if err_pos < best_err[0] or (err_pos == best_err[0] and err_rot < best_err[1]):
                best_err = (err_pos, err_rot)
                best_q = q_sol.copy()
                best_iters = iters

            if success:
                success_seeds += 1
                return ReachabilityResult(
                    reachable=True,
                    q_best=q_sol.copy(),
                    err_pos=err_pos,
                    err_rot=err_rot,
                    iters=iters,
                    success_seeds=success_seeds,
                    total_seeds=total_seeds,
                    message="Reachable: found IK solution within joint limits.",
                )

        return ReachabilityResult(
            reachable=False,
            q_best=best_q,
            err_pos=best_err[0],
            err_rot=best_err[1],
            iters=best_iters,
            success_seeds=success_seeds,
            total_seeds=total_seeds,
            message="Unreachable: no IK seed converged within limits.",
        )


def main():
    # Minimal usage example (replace URDF path and frame names with real ones).
    urdf_path = "/home/wzx/WholeBodyRL_WS/RMPlus/Go2Arm_description/urdf/piper_description_mjc_NoGripper.urdf"
    base_frame = "base_link"
    ee_frame = "link6"

    cfg = ReachabilityConfig(
        urdf_path=urdf_path,
        base_frame=base_frame,
        ee_frame=ee_frame,
    )

    # Construct a goal pose expressed in base_frame.
    goal_T_base = pin.SE3(
        pin.utils.rotate("z", np.deg2rad(45.0)),
        np.array([0.0, 0.0, 0.4]),
    )

    try:
        reach = EEReachability(cfg)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    result = reach.is_reachable(goal_T_base)
    print("reachable:", result.reachable)
    print("q_best:", result.q_best)
    print("err_pos:", result.err_pos, "err_rot:", result.err_rot)
    print("iters:", result.iters)
    print("success_seeds/total_seeds:", result.success_seeds, "/", result.total_seeds)


if __name__ == "__main__":
    main()
