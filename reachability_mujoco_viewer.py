from __future__ import annotations

import time
from typing import Tuple

import numpy as np
import pinocchio as pin

try:
    import mujoco
    import mujoco.viewer
except Exception as e:  # pragma: no cover
    raise ImportError("mujoco is required to run this viewer.") from e

from reachability_v1 import EEReachability, ReachabilityConfig


def _quat_wxyz_from_rot(R: np.ndarray) -> np.ndarray:
    q = pin.Quaternion(R).coeffs()  # x, y, z, w
    return np.array([q[3], q[0], q[1], q[2]], dtype=float)


def _rot_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _set_qpos_from_pin(model: mujoco.MjModel, data: mujoco.MjData, pin_model: pin.Model, q: np.ndarray) -> None:
    missing = []
    for j in range(1, pin_model.njoints):
        joint = pin_model.joints[j]
        if joint.nq != 1:
            continue
        name = pin_model.names[j]
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            missing.append(name)
            continue
        qpos_adr = model.jnt_qposadr[jid]
        data.qpos[qpos_adr] = q[joint.idx_q]
    if missing:
        print("[warn] joints not found in MJCF:", missing)


def _world_T_base(model: mujoco.MjModel, data: mujoco.MjData, base_body_name: str) -> pin.SE3:
    base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
    if base_body_id >= 0:
        base_pos = data.xpos[base_body_id].copy()
        base_quat = data.xquat[base_body_id].copy()  # wxyz
        return pin.SE3(_rot_from_quat_wxyz(base_quat), base_pos)
    else:
        print("[warn] base body not found in MJCF, using world frame as base.")
        return pin.SE3.Identity()


def _add_goal_marker_to_viewer(viewer, goal_pos: np.ndarray, size: float = 0.02) -> None:
    if not hasattr(viewer, "user_scn"):
        print("[warn] viewer has no user_scn; cannot draw goal marker.")
        return
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        print("[warn] user_scn full; cannot add goal marker.")
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([size, 0.0, 0.0], dtype=float),
        goal_pos.astype(float),
        np.eye(3, dtype=float).reshape(9),
        np.array([1.0, 0.0, 0.0, 0.6], dtype=float),
    )
    scn.ngeom += 1


def _add_floor_to_viewer(viewer, size: float = 5.0) -> None:
    if not hasattr(viewer, "user_scn"):
        print("[warn] viewer has no user_scn; cannot draw floor.")
        return
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        print("[warn] user_scn full; cannot add floor.")
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_PLANE,
        np.array([size, size, 0.1], dtype=float),
        np.array([0.0, 0.0, 0.0], dtype=float),
        np.eye(3, dtype=float).reshape(9),
        np.array([0.7, 0.7, 0.7, 1.0], dtype=float),
    )
    scn.ngeom += 1


def _add_frame_axes_to_viewer(
    viewer,
    origin: np.ndarray,
    R: np.ndarray,
    axis_len: float = 0.15,
    radius: float = 0.0035,
    alpha: float = 1.0,
) -> None:
    if not hasattr(viewer, "user_scn"):
        print("[warn] viewer has no user_scn; cannot draw axes.")
        return
    scn = viewer.user_scn
    axes = [
        (R[:, 0], np.array([1.0, 0.0, 0.0, alpha])),  # X red
        (R[:, 1], np.array([0.0, 1.0, 0.0, alpha])),  # Y green
        (R[:, 2], np.array([0.0, 0.0, 1.0, alpha])),  # Z blue
    ]
    for direction, color in axes:
        if scn.ngeom + 1 > scn.maxgeom:
            print("[warn] user_scn full; cannot add axes.")
            return
        geom = scn.geoms[scn.ngeom]
        z_axis = direction / (np.linalg.norm(direction) + 1e-9)
        x_axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(np.abs(np.dot(z_axis, x_axis)), 1.0, atol=1e-6):
            x_axis = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        R_axis = np.vstack([x_axis, y_axis, z_axis]).T
        pos = origin  # for mjGEOM_ARROW, place tail at origin
        if hasattr(mujoco.mjtGeom, "mjGEOM_ARROW"):
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_ARROW,
                np.array([radius, radius, axis_len], dtype=float),
                pos.astype(float),
                R_axis.reshape(9),
                color.astype(float),
            )
        else:
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                np.array([radius, axis_len * 0.5, 0.0], dtype=float),
                pos.astype(float),
                R_axis.reshape(9),
                color.astype(float),
            )
        scn.ngeom += 1
    # small sphere at origin to emphasize intersection
    if scn.ngeom < scn.maxgeom:
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius * 1.5, 0.0, 0.0], dtype=float),
            origin.astype(float),
            np.eye(3, dtype=float).reshape(9),
            np.array([1.0, 1.0, 1.0, alpha], dtype=float),
        )
        scn.ngeom += 1


def _add_grid_to_viewer(
    viewer,
    half_size: float = 2.0,
    step: float = 0.2,
    z: float = 0.0,
    thickness: float = 0.003,
) -> None:
    if not hasattr(viewer, "user_scn"):
        print("[warn] viewer has no user_scn; cannot draw grid.")
        return
    scn = viewer.user_scn
    coords = np.arange(-half_size, half_size + 1e-9, step)
    color = np.array([0.5, 0.5, 0.5, 0.5], dtype=float)
    for x in coords:
        if scn.ngeom >= scn.maxgeom:
            print("[warn] user_scn full; grid truncated.")
            return
        geom = scn.geoms[scn.ngeom]
        size = np.array([half_size, thickness, thickness], dtype=float)
        pos = np.array([0.0, x, z], dtype=float)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            size,
            pos,
            np.eye(3, dtype=float).reshape(9),
            color,
        )
        scn.ngeom += 1
    for y in coords:
        if scn.ngeom >= scn.maxgeom:
            print("[warn] user_scn full; grid truncated.")
            return
        geom = scn.geoms[scn.ngeom]
        size = np.array([thickness, half_size, thickness], dtype=float)
        pos = np.array([y, 0.0, z], dtype=float)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_BOX,
            size,
            pos,
            np.eye(3, dtype=float).reshape(9),
            color,
        )
        scn.ngeom += 1


def _base_T_ee_from_pin(reach: EEReachability, q: np.ndarray) -> pin.SE3:
    pin.forwardKinematics(reach.model, reach.data, q)
    pin.updateFramePlacements(reach.model, reach.data)
    world_T_base = reach.data.oMf[reach.base_id]
    world_T_ee = reach.data.oMf[reach.ee_id]
    return world_T_base.inverse() * world_T_ee


def main() -> None:
    # Paths and frames (adjust if needed)
    urdf_path = "/home/wzx/WholeBodyRL_WS/RMPlus/Go2Arm_description/urdf/piper_description_mjc_NoGripper.urdf"
    mjcf_path = "/home/wzx/WholeBodyRL_WS/RMPlus/Go2Arm_description/mjcf/piper_description_mjc_NoGripper.xml"
    base_frame = "base_link"
    ee_frame = "link6"

    # Goal pose in base_frame: Rz(45 deg), t = [0.3, 0.2, 0.4] (m)
    # goal_T_base = pin.SE3(
    #     pin.utils.rotate("z", np.deg2rad(0.0)),
    #     np.array([0.5, 0.0, 0.3]),
    # )
    goal_T_base = pin.SE3(
        np.array([[ 0.91758449, -0.39214651,  0.06526726],
                  [ 0.37790065,  0.91138208,  0.16301473],
                  [-0.12340907, -0.12491525,  0.98446248]]),
        np.array([ 0.43136562, -0.2648881,   0.38199008])
    )

    cfg = ReachabilityConfig(
        urdf_path=urdf_path,
        base_frame=base_frame,
        ee_frame=ee_frame,
        debug=False,
    )

    reach = EEReachability(cfg)
    result = reach.is_reachable(goal_T_base)
    print("reachable:", result.reachable)
    print("q_best:", result.q_best)
    print("err_pos:", result.err_pos, "err_rot:", result.err_rot)
    base_T_ee = _base_T_ee_from_pin(reach, result.q_best)
    print("ee_in_base_position:", base_T_ee.translation)
    print("ee_in_base_rotation_matrix:\n", base_T_ee.rotation)
    # Optional: force q0 = q_best and re-check reachability deterministically
    # result_check = reach.is_reachable(goal_T_base, q0=result.q_best)
    # print("reachable_with_q0=q_best:", result_check.reachable)
    # print("err_pos_with_q0:", result_check.err_pos, "err_rot_with_q0:", result_check.err_rot)

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    _set_qpos_from_pin(model, data, reach.model, result.q_best)
    mujoco.mj_forward(model, data)
    world_T_base = _world_T_base(model, data, base_frame)
    world_T_goal = world_T_base * goal_T_base
    goal_pos = world_T_goal.translation

    with mujoco.viewer.launch_passive(model, data) as viewer:
        _add_floor_to_viewer(viewer)
        _add_grid_to_viewer(viewer)
        _add_frame_axes_to_viewer(viewer, np.zeros(3), np.eye(3), axis_len=0.3, alpha=0.8)
        _add_goal_marker_to_viewer(viewer, goal_pos)
        _add_frame_axes_to_viewer(
            viewer, goal_pos, world_T_goal.rotation, axis_len=0.2, alpha=0.9
        )
        base_T_ee = _base_T_ee_from_pin(reach, result.q_best)
        world_T_ee = world_T_base * base_T_ee
        _add_frame_axes_to_viewer(
            viewer, world_T_ee.translation, world_T_ee.rotation, axis_len=0.2, alpha=0.9
        )
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
