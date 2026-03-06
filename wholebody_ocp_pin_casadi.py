#!/usr/bin/env python3
"""
Whole-body Reference Trajectory Planning (single-file demo)
Using Pinocchio + CasADi (Opti/IPOPT)

State (9D):
    x = [z, phi, theta, varphi1..varphi6]^T
Control (9D):
    u = [dot_z, dot_phi, dot_theta, dot_varphi1..dot_varphi6]^T
Dynamics:
    xdot = u
Discretization:
    N segments, dt = T/N, default explicit Euler (optional RK4 switch)

Important modeling assumption (consistent with the referenced design):
    base x_b = 0, y_b = 0, yaw psi_b = 0 are fixed,
    only z, phi, theta are optimized.

Base transform is strictly:
    ^W T_B = [ R_x(phi) R_y(theta), [0,0,z]^T ]
(Please keep this exact rotation order Rx then Ry.)
"""

import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass

import casadi as ca
import numpy as np
import pinocchio as pin

# -----------------------------
# Top-level user parameters
# -----------------------------
URDF_PATH = "/home/wzx/WholeBodyRL_WS/RMPlus/Go2Arm_description/urdf/piper_description_mjc_NoGripper.urdf"  # 机械臂 URDF 路径
BASE_FRAME_NAME = "base"  # 论文中的 base 对应 URDF 帧名
ARM_BASE_FRAME_NAME = "base_link"  # 机械臂基座帧名（arm root）
EE_FRAME_NAME = "link6"  # 末端执行器帧名
ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]  # varphi1..6 与 URDF 关节的显式顺序映射

T = 1.0  # 规划总时长 [s]
N = 100  # 离散段数（状态点数为 N+1）
INTEGRATOR = "euler"  # 离散积分方法：euler(默认) 或 rk4

R = np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # 运行代价中 u^T R u 的权重矩阵（9x9）
W_EE_POS = 1000.0  # 终端位置误差项权重（||e_pos||^2 前系数）
W_EE_ORI = 1000.0  # 终端姿态误差项权重（||e_ori||^2 前系数）
W_BARRIER = 10.0  # 运行阶段屏障项总权重（L_B 前系数）

X0 = np.array([0.30, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)  # 初始状态 [z,phi,theta,varphi1..6]

BASE_X_MIN = np.array([0.30, -np.pi/6, -np.pi/6], dtype=float)  # base 状态下界 [z(m), phi(rad), theta(rad)]
BASE_X_MAX = np.array([0.50,  np.pi/6,  np.pi/6], dtype=float)  # base 状态上界 [z(m), phi(rad), theta(rad)]

BASE_U_MIN = np.array([-0.40, -1.20, -1.20], dtype=float)  # base 控制下界 [dot_z(m/s), dot_phi(rad/s), dot_theta(rad/s)]
BASE_U_MAX = np.array([ 0.40,  1.20,  1.20], dtype=float)  # base 控制上界 [dot_z(m/s), dot_phi(rad/s), dot_theta(rad/s)]
# arm 的位置/速度上下界由 URDF 关节限位自动读取（按 ARM_JOINT_NAMES 顺序）

MU_BASE = 5e-3  # base 相关 relaxed barrier 系数 mu
DELTA_BASE = 1e-4  # base 相关 relaxed barrier 分段阈值 delta
MU_ARM = 5e-3  # arm 相关 relaxed barrier 系数 mu
DELTA_ARM = 1e-4  # arm 相关 relaxed barrier 分段阈值 delta

P_GOAL = np.array([0.20, 0.00, 0.80], dtype=float)  # 末端目标位置 ^W p_goal [m]
Q_GOAL_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)  # 末端目标姿态四元数 [w,x,y,z]

USE_QUAT_SIGN_CORRECTION = True  # 姿态误差计算时是否启用符号修正（dot(q,qhat)<0 时取 -qhat）
PLOT_ERRORS = False  # 是否绘制误差随时间曲线（需 matplotlib）

MUJOCO_VISUALIZE = True  # 是否在求解后使用 MuJoCo 播放优化轨迹
MUJOCO_MODEL_PATH = "/home/wzx/WholeBodyRL_WS/RMPlus/Go2Arm_description/urdf/vis_piper_description_mjc_NoGripper.urdf"  # MuJoCo 加载模型路径（可为 URDF 或 MJCF）
MUJOCO_LOOP = False  # 轨迹播放结束后是否循环播放
MUJOCO_REALTIME_FACTOR = 0.1  # 播放速度倍率：1.0 实时，2.0 两倍速
MUJOCO_ADD_ORIGIN_AXES = True  # 是否在 viewer 中显示世界原点坐标系
MUJOCO_ORIGIN_AXIS_LENGTH = 0.25  # 原点坐标轴长度 [m]
MUJOCO_ORIGIN_AXIS_WIDTH_PX = 4.0  # 原点坐标轴线宽 [pixel]
MUJOCO_ADD_GOAL_AXES = True  # 是否在 viewer 中显示目标位姿坐标系（位于 P_GOAL）
MUJOCO_GOAL_AXIS_LENGTH = 0.18  # 目标坐标轴长度 [m]
MUJOCO_GOAL_AXIS_WIDTH_PX = 4.0  # 目标坐标轴线宽 [pixel]

IPOPT_MAX_ITER = 1000  # IPOPT 最大迭代次数
IPOPT_TOL = 1e-6  # IPOPT 收敛容差
IPOPT_PRINT_LEVEL = 5  # IPOPT 日志详细级别（0 更安静）


_ARM_FK_FUN = None  # 全局：arm 符号 FK 函数句柄（由 pinocchio.casadi 构建）
_BASE_TO_ARMBASE_P = None  # 全局：固定外参 ^base p_arm_base
_BASE_TO_ARMBASE_Q = None  # 全局：固定外参 ^base q_arm_base（wxyz）


@dataclass
class OCPParams:
    urdf_path: str
    base_frame_name: str
    arm_base_frame_name: str
    ee_frame_name: str
    T: float
    N: int
    integrator: str
    R: np.ndarray
    w_ee_pos: float
    w_ee_ori: float
    w_barrier: float
    x0: np.ndarray
    x_min: np.ndarray
    x_max: np.ndarray
    u_min: np.ndarray
    u_max: np.ndarray
    mu_base: float
    delta_base: float
    mu_arm: float
    delta_arm: float
    p_goal: np.ndarray
    q_goal_wxyz: np.ndarray
    use_quat_sign_correction: bool
    plot_errors: bool
    ipopt_max_iter: int
    ipopt_tol: float
    ipopt_print_level: int


def cross3(a: ca.MX, b: ca.MX) -> ca.MX:
    return ca.vertcat(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def quat_normalize_wxyz(q: ca.MX, eps: float = 1e-12) -> ca.MX:
    return q / ca.sqrt(ca.fmax(ca.dot(q, q), eps))


def quat_mul_wxyz(q1: ca.MX, q2: ca.MX) -> ca.MX:
    w1, v1 = q1[0], q1[1:4]
    w2, v2 = q2[0], q2[1:4]
    w = w1 * w2 - ca.dot(v1, v2)
    v = w1 * v2 + w2 * v1 + cross3(v1, v2)
    return ca.vertcat(w, v)


def quat_to_rotmat_wxyz(q: ca.MX) -> ca.MX:
    qn = quat_normalize_wxyz(q)
    w, x, y, z = qn[0], qn[1], qn[2], qn[3]
    return ca.vertcat(
        ca.horzcat(1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        ca.horzcat(2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        ca.horzcat(2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )


def rotmat_to_quat_wxyz(Rm: ca.MX, eps: float = 1e-12) -> ca.MX:
    """Convert 3x3 rotation matrix to quaternion in wxyz order (CasADi symbolic)."""
    r00, r01, r02 = Rm[0, 0], Rm[0, 1], Rm[0, 2]
    r10, r11, r12 = Rm[1, 0], Rm[1, 1], Rm[1, 2]
    r20, r21, r22 = Rm[2, 0], Rm[2, 1], Rm[2, 2]

    tr = r00 + r11 + r22

    S1 = 2.0 * ca.sqrt(ca.fmax(tr + 1.0, eps))
    q1 = ca.vertcat(
        0.25 * S1,
        (r21 - r12) / S1,
        (r02 - r20) / S1,
        (r10 - r01) / S1,
    )

    S2 = 2.0 * ca.sqrt(ca.fmax(1.0 + r00 - r11 - r22, eps))
    q2 = ca.vertcat(
        (r21 - r12) / S2,
        0.25 * S2,
        (r01 + r10) / S2,
        (r02 + r20) / S2,
    )

    S3 = 2.0 * ca.sqrt(ca.fmax(1.0 + r11 - r00 - r22, eps))
    q3 = ca.vertcat(
        (r02 - r20) / S3,
        (r01 + r10) / S3,
        0.25 * S3,
        (r12 + r21) / S3,
    )

    S4 = 2.0 * ca.sqrt(ca.fmax(1.0 + r22 - r00 - r11, eps))
    q4 = ca.vertcat(
        (r10 - r01) / S4,
        (r02 + r20) / S4,
        (r12 + r21) / S4,
        0.25 * S4,
    )

    cond1 = tr > 0
    cond2 = ca.logic_and(r00 > r11, r00 > r22)
    cond3 = r11 > r22

    q = ca.if_else(cond1, q1, ca.if_else(cond2, q2, ca.if_else(cond3, q3, q4)))
    return quat_normalize_wxyz(q)


def base_quat_wxyz(phi: ca.MX, theta: ca.MX) -> ca.MX:
    """
    Closed-form quaternion for R_WB = R_x(phi) * R_y(theta), in wxyz.
    Avoids branch switching in rotmat_to_quat for the base orientation.
    """
    cph2 = ca.cos(0.5 * phi)
    sph2 = ca.sin(0.5 * phi)
    cth2 = ca.cos(0.5 * theta)
    sth2 = ca.sin(0.5 * theta)
    q = ca.vertcat(
        cph2 * cth2,  # w
        sph2 * cth2,  # x
        cph2 * sth2,  # y
        sph2 * sth2,  # z
    )
    return quat_normalize_wxyz(q)


def orientation_error_vec(q_wxyz: ca.MX, qhat_wxyz: ca.MX, use_sign_correction: bool) -> ca.MX:
    """
    e_ori = qw*[qhat_xyz] - qhat_w*[q_xyz] + qhat_xyz x q_xyz
    Quaternion order is strictly wxyz.
    """
    qh = qhat_wxyz
    if use_sign_correction:
        qh = ca.if_else(ca.dot(q_wxyz, qhat_wxyz) >= 0.0, qhat_wxyz, -qhat_wxyz)

    qw, qv = q_wxyz[0], q_wxyz[1:4]
    qhw, qhv = qh[0], qh[1:4]
    return qw * qhv - qhw * qv + cross3(qhv, qv)


def relaxed_barrier(h: ca.MX, mu: float, delta: float) -> ca.MX:
    """
    Relaxed barrier for h >= 0 canonical constraints:
      B(h) = -mu*log(h), if h >= delta
           = mu/2 * (((h - 2*delta)/delta)^2 - 1) - mu*log(delta), else

    Implemented via casadi.if_else with numerical safety on log input.
    """
    h_safe = ca.fmax(h, 1e-12)
    log_branch = -mu * ca.log(h_safe)
    quad_branch = (mu / 2.0) * ((((h - 2.0 * delta) / delta) ** 2) - 1.0) - mu * ca.log(delta)
    return ca.if_else(h >= delta, log_branch, quad_branch)


def load_model(urdf: str, base_frame: str, arm_base_frame: str, ee_frame: str):
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()

    base_id = model.getFrameId(base_frame)
    arm_base_id = model.getFrameId(arm_base_frame)
    ee_id = model.getFrameId(ee_frame)
    if base_id >= model.nframes:
        all_frames = [f.name for f in model.frames]
        raise ValueError(
            f"Base frame '{base_frame}' not found in URDF. Available frames: {all_frames}"
        )
    if arm_base_id >= model.nframes:
        all_frames = [f.name for f in model.frames]
        raise ValueError(
            f"Arm base frame '{arm_base_frame}' not found in URDF. Available frames: {all_frames}"
        )
    if ee_id >= model.nframes:
        all_frames = [f.name for f in model.frames]
        raise ValueError(
            f"EE frame '{ee_frame}' not found in URDF. Available frames: {all_frames}"
        )

    if model.nq != 6 or model.nv != 6:
        raise ValueError(
            f"This script assumes a 6-DoF arm (nq=nv=6), but got nq={model.nq}, nv={model.nv}."
        )

    # Numeric extraction of fixed transform ^base T_arm_base from URDF frames.
    q0 = np.zeros(model.nq)
    pin.forwardKinematics(model, data, q0)
    pin.updateFramePlacements(model, data)
    M_ob = data.oMf[base_id]
    M_oa = data.oMf[arm_base_id]
    M_ba = M_ob.inverse() * M_oa
    p_ba = ca.DM(M_ba.translation.reshape((3, 1)))
    q_ba = rotmat_to_quat_wxyz(ca.DM(M_ba.rotation))

    return model, data, base_id, arm_base_id, ee_id, p_ba, q_ba


def _build_fk_arm_casadi(model: pin.Model, arm_base_frame_id: int, ee_id: int):
    """
    Build symbolic arm FK function with pinocchio.casadi.
    Returns fk_fun(q_arm)->(p_AE, q_AE_wxyz), where A = arm_base_frame (e.g. base_link).
    """
    try:
        import pinocchio.casadi as cpin
    except Exception as e:
        raise RuntimeError(
            "pinocchio.casadi is not available. "
            "OCP requires differentiable arm FK p_ee,q_ee in CasADi. "
            "Numeric pinocchio FK alone cannot provide symbolic gradients for Opti/IPOPT. "
            "Please install/enable pinocchio with CasADi bindings in env_eeRM."
        ) from e

    try:
        cmodel = cpin.Model(model)
        cdata = cmodel.createData()
        # pinocchio.casadi expects SX symbols in this binding.
        q_arm = ca.SX.sym("q_arm", model.nq)
        cpin.forwardKinematics(cmodel, cdata, q_arm)
        cpin.updateFramePlacements(cmodel, cdata)

        M_oa = cdata.oMf[arm_base_frame_id]
        M_oe = cdata.oMf[ee_id]
        M_ae = M_oa.inverse() * M_oe
        p_AE = M_ae.translation
        R_AE = M_ae.rotation
        q_AE_wxyz = rotmat_to_quat_wxyz(R_AE)

        fk_fun = ca.Function(
            "fk_arm_pin_casadi",
            [q_arm],
            [p_AE, q_AE_wxyz],
            ["q_arm"],
            ["p_AE", "q_AE_wxyz"],
        )
        return fk_fun
    except Exception as e:
        raise RuntimeError(
            "Failed to build symbolic FK with pinocchio.casadi. "
            "Common causes: Pinocchio/CasADi ABI mismatch, unsupported build options, or API version mismatch. "
            "Check that `import pinocchio.casadi as cpin` works and that `cpin.Model(model)` + "
            "`cpin.forwardKinematics(...)` run with a CasADi SX symbol."
        ) from e


def fk_arm(varphi: ca.MX):
    """Return ^arm_base p_EE, ^arm_base q_EE(wxyz) using symbolic Pinocchio FK."""
    global _ARM_FK_FUN
    if _ARM_FK_FUN is None:
        raise RuntimeError("Arm FK function not initialized. Call build_and_solve_ocp() first.")
    p_AE, q_AE_wxyz = _ARM_FK_FUN(varphi)
    return p_AE, quat_normalize_wxyz(q_AE_wxyz)


def base_transform(z: ca.MX, phi: ca.MX, theta: ca.MX):
    """
    Return ^W T_B components: R_WB, p_WB.

    IMPORTANT:
    Rotation order is STRICTLY R_x(phi) then R_y(theta), i.e.:
        R_WB = R_x(phi) * R_y(theta)
    and translation p_WB = [0,0,z]^T.
    """
    cphi, sphi = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)

    R_x = ca.vertcat(
        ca.horzcat(1, 0, 0),
        ca.horzcat(0, cphi, -sphi),
        ca.horzcat(0, sphi, cphi),
    )
    R_y = ca.vertcat(
        ca.horzcat(cth, 0, sth),
        ca.horzcat(0, 1, 0),
        ca.horzcat(-sth, 0, cth),
    )

    # Strict order required by the paper design:
    R_WB = R_x @ R_y
    p_WB = ca.vertcat(0, 0, z)
    return R_WB, p_WB


def compose_fk(z: ca.MX, phi: ca.MX, theta: ca.MX, varphi: ca.MX):
    """
    ^W T_EE(x) = ^W T_base(z,phi,theta) * ^base T_arm_base * ^arm_base T_EE(varphi)
    Returns p_ee and q_ee(wxyz).
    """
    global _BASE_TO_ARMBASE_P, _BASE_TO_ARMBASE_Q
    if _BASE_TO_ARMBASE_P is None or _BASE_TO_ARMBASE_Q is None:
        raise RuntimeError("Base-to-arm-base fixed transform not initialized. Call build_and_solve_ocp() first.")

    R_WB, p_WB = base_transform(z, phi, theta)
    p_BA = _BASE_TO_ARMBASE_P
    q_BA = quat_normalize_wxyz(_BASE_TO_ARMBASE_Q)
    R_BA = quat_to_rotmat_wxyz(q_BA)
    p_AE, q_AE = fk_arm(varphi)

    p_BE = p_BA + R_BA @ p_AE
    p_ee = p_WB + R_WB @ p_BE
    # Closed-form quaternion for base orientation (more stable than branch-based rotmat->quat here).
    q_WB = base_quat_wxyz(phi, theta)
    q_ee = quat_normalize_wxyz(quat_mul_wxyz(quat_mul_wxyz(q_WB, q_BA), q_AE))
    return p_ee, q_ee


def _read_arm_bounds_from_urdf(urdf_path: str, arm_joint_names):
    """
    Read arm joint position/velocity limits from URDF via Pinocchio model:
      q_min/q_max from model.lowerPositionLimit / upperPositionLimit
      dq limits from model.velocityLimit, mapped to [-v, +v]
    Joint order is explicitly defined by arm_joint_names.
    """
    model = pin.buildModelFromUrdf(urdf_path)
    q_lower = np.array(model.lowerPositionLimit, dtype=float).reshape(-1)
    q_upper = np.array(model.upperPositionLimit, dtype=float).reshape(-1)
    v_limit = np.array(model.velocityLimit, dtype=float).reshape(-1)

    arm_x_min = []
    arm_x_max = []
    arm_u_min = []
    arm_u_max = []
    seen = set()

    for joint_name in arm_joint_names:
        if joint_name in seen:
            raise ValueError(f"Duplicate joint in ARM_JOINT_NAMES: '{joint_name}'")
        seen.add(joint_name)

        if not model.existJointName(joint_name):
            raise ValueError(
                f"Joint '{joint_name}' not found in URDF model. Available joints: {list(model.names)}"
            )

        jid = model.getJointId(joint_name)
        jmodel = model.joints[jid]
        if int(jmodel.nq) != 1 or int(jmodel.nv) != 1:
            raise ValueError(
                f"Joint '{joint_name}' is not 1-DoF (nq={jmodel.nq}, nv={jmodel.nv}). "
                "This script expects scalar varphi_i joints."
            )

        iq = int(model.idx_qs[jid])
        iv = int(model.idx_vs[jid])

        qmin_i = float(q_lower[iq])
        qmax_i = float(q_upper[iq])
        vmax_i = float(v_limit[iv])

        if (not np.isfinite(qmin_i)) or (not np.isfinite(qmax_i)) or qmin_i >= qmax_i:
            raise ValueError(f"Invalid position limits for joint '{joint_name}': [{qmin_i}, {qmax_i}]")
        if (not np.isfinite(vmax_i)) or vmax_i <= 0.0:
            raise ValueError(f"Invalid velocity limit for joint '{joint_name}': {vmax_i}")

        arm_x_min.append(qmin_i)
        arm_x_max.append(qmax_i)
        arm_u_min.append(-vmax_i)
        arm_u_max.append(vmax_i)

    arm_x_min = np.array(arm_x_min, dtype=float)
    arm_x_max = np.array(arm_x_max, dtype=float)
    arm_u_min = np.array(arm_u_min, dtype=float)
    arm_u_max = np.array(arm_u_max, dtype=float)
    return arm_x_min, arm_x_max, arm_u_min, arm_u_max


def _make_xu_bounds(arm_x_min: np.ndarray, arm_x_max: np.ndarray, arm_u_min: np.ndarray, arm_u_max: np.ndarray):
    # final 9D bounds = manual base(3) + URDF arm(6)
    x_min = np.concatenate([BASE_X_MIN, arm_x_min])
    x_max = np.concatenate([BASE_X_MAX, arm_x_max])
    u_min = np.concatenate([BASE_U_MIN, arm_u_min])
    u_max = np.concatenate([BASE_U_MAX, arm_u_max])
    return x_min, x_max, u_min, u_max


def _quat_mul_wxyz_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def _quat_to_rotmat_wxyz_np(q: np.ndarray) -> np.ndarray:
    qn = q / max(np.linalg.norm(q), 1e-12)
    w, x, y, z = qn
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _base_quat_wxyz_np(phi: float, theta: float) -> np.ndarray:
    cph2 = np.cos(0.5 * phi)
    sph2 = np.sin(0.5 * phi)
    cth2 = np.cos(0.5 * theta)
    sth2 = np.sin(0.5 * theta)
    q = np.array([cph2 * cth2, sph2 * cth2, cph2 * sth2, sph2 * sth2], dtype=float)
    return q / max(np.linalg.norm(q), 1e-12)


def _prepare_mujoco_model_path(model_path: str):
    """
    MuJoCo URDF importer may resolve mesh paths relative to the URDF directory with basename only.
    For URDF input, create a temp folder:
      - copy URDF
      - symlink all referenced mesh basenames into the same temp folder
    Return (prepared_model_path, temp_dir_or_None).
    """
    ext = os.path.splitext(model_path)[1].lower()
    if ext != ".urdf":
        return model_path, None

    src_dir = os.path.dirname(os.path.abspath(model_path))
    with open(model_path, "r", encoding="utf-8") as f:
        urdf_text = f.read()

    tmp_dir = tempfile.mkdtemp(prefix="mujoco_urdf_")
    prepared_urdf = os.path.join(tmp_dir, os.path.basename(model_path))
    with open(prepared_urdf, "w", encoding="utf-8") as f:
        f.write(urdf_text)

    mesh_files = re.findall(r"filename\s*=\s*['\"]([^'\"]+)['\"]", urdf_text)
    for mesh_rel in mesh_files:
        mesh_abs = os.path.abspath(os.path.join(src_dir, mesh_rel))
        if not os.path.exists(mesh_abs):
            raise FileNotFoundError(f"Mesh file referenced by URDF not found: {mesh_abs}")
        mesh_link = os.path.join(tmp_dir, os.path.basename(mesh_abs))
        if not os.path.exists(mesh_link):
            os.symlink(mesh_abs, mesh_link)

    return prepared_urdf, tmp_dir


def _add_origin_axes_to_viewer(viewer, mj, length: float = 0.25, width_px: float = 4.0) -> None:
    """
    Add a visual-only XYZ coordinate frame at world origin in viewer.user_scn.
    X: red, Y: green, Z: blue.
    """
    if not hasattr(viewer, "user_scn"):
        print("[warn] viewer has no user_scn; cannot draw origin axes.")
        return

    scn = viewer.user_scn
    needed = 3
    if scn.ngeom + needed > scn.maxgeom:
        print("[warn] user_scn full; cannot add origin axes.")
        return

    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    ends = [
        (np.array([length, 0.0, 0.0], dtype=float), np.array([1.0, 0.1, 0.1, 1.0], dtype=float)),  # X
        (np.array([0.0, length, 0.0], dtype=float), np.array([0.1, 1.0, 0.1, 1.0], dtype=float)),  # Y
        (np.array([0.0, 0.0, length], dtype=float), np.array([0.1, 0.3, 1.0, 1.0], dtype=float)),  # Z
    ]

    for p_to, rgba in ends:
        geom = scn.geoms[scn.ngeom]
        # Initialize visual attrs then overwrite geometry as connector line.
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_LINE,
            np.array([1.0, 1.0, 1.0], dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).reshape(9),
            rgba,
        )
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_LINE, width_px, origin, p_to)
        scn.ngeom += 1


def _add_goal_axes_to_viewer(
    viewer,
    mj,
    goal_xyz: np.ndarray,
    goal_q_wxyz: np.ndarray,
    length: float = 0.18,
    width_px: float = 4.0,
) -> None:
    """
    Add visual-only XYZ axes at goal pose in viewer.user_scn.
    Axes orientation follows goal_q_wxyz (wxyz).
    """
    if not hasattr(viewer, "user_scn"):
        print("[warn] viewer has no user_scn; cannot draw goal axes.")
        return

    scn = viewer.user_scn
    needed = 3
    if scn.ngeom + needed > scn.maxgeom:
        print("[warn] user_scn full; cannot add goal axes.")
        return

    origin = np.array(goal_xyz, dtype=float).reshape(3)
    R_goal = _quat_to_rotmat_wxyz_np(np.array(goal_q_wxyz, dtype=float).reshape(4))
    ends = [
        (origin + R_goal @ np.array([length, 0.0, 0.0], dtype=float), np.array([1.0, 0.2, 0.2, 1.0], dtype=float)),  # X
        (origin + R_goal @ np.array([0.0, length, 0.0], dtype=float), np.array([0.2, 1.0, 0.2, 1.0], dtype=float)),  # Y
        (origin + R_goal @ np.array([0.0, 0.0, length], dtype=float), np.array([0.2, 0.4, 1.0, 1.0], dtype=float)),  # Z
    ]

    for p_to, rgba in ends:
        geom = scn.geoms[scn.ngeom]
        mj.mjv_initGeom(
            geom,
            mj.mjtGeom.mjGEOM_LINE,
            np.array([1.0, 1.0, 1.0], dtype=float),
            np.zeros(3, dtype=float),
            np.eye(3, dtype=float).reshape(9),
            rgba,
        )
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_LINE, width_px, origin, p_to)
        scn.ngeom += 1


def visualize_trajectory_mujoco(
    x_traj: np.ndarray,
    dt: float,
    model_path: str,
    arm_joint_names,
    goal_point: np.ndarray | None = None,
    goal_quat_wxyz: np.ndarray | None = None,
    loop: bool = False,
    realtime_factor: float = 1.0,
):
    """
    Play optimized trajectory in MuJoCo viewer.
    Mapping:
      - x[:, 0:3] -> virtual base pose (z, phi, theta), applied to root body transform
      - x[:, 3:9] -> arm joints by explicit arm_joint_names order
    """
    try:
        import mujoco as mj
        import mujoco.viewer as mj_viewer
    except Exception as e:
        raise RuntimeError("MuJoCo or mujoco.viewer is not available in current environment.") from e

    if x_traj.ndim != 2 or x_traj.shape[1] != 9:
        raise ValueError(f"x_traj must have shape (N+1, 9), got {x_traj.shape}.")

    model_file, tmp_dir = _prepare_mujoco_model_path(model_path)
    try:
        model = mj.MjModel.from_xml_path(model_file)
        data = mj.MjData(model)

        # Map arm joints by explicit name order.
        qpos_adrs = []
        for jn in arm_joint_names:
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                available = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
                raise ValueError(f"Joint '{jn}' not found in MuJoCo model. Available joints: {available}")
            qpos_adrs.append(int(model.jnt_qposadr[jid]))

        # Preferred mode: if URDF has explicit movable root joints, map x[0:3] directly to them.
        root_joint_names = ["root_z", "root_roll", "root_pitch"]
        root_joint_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, n) for n in root_joint_names]
        has_explicit_root_joints = all(jid >= 0 for jid in root_joint_ids)

        if has_explicit_root_joints:
            root_qpos_adrs = [int(model.jnt_qposadr[jid]) for jid in root_joint_ids]
            print(
                f"MuJoCo viewer: model='{model_file}', root_joint_mode='explicit({','.join(root_joint_names)})', "
                f"logical_base_frame='{BASE_FRAME_NAME}', logical_arm_base_frame='{ARM_BASE_FRAME_NAME}', "
                f"frames={x_traj.shape[0]}, dt={dt:.6f}s"
            )
        else:
            # Fallback mode for fixed-base URDF:
            # apply virtual base z/phi/theta to runtime root body transform.
            root_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, arm_joint_names[0])
            root_body_id = int(model.jnt_bodyid[root_joint_id])
            root_pos_nom = np.array(model.body_pos[root_body_id], dtype=float)
            root_quat_nom = np.array(model.body_quat[root_body_id], dtype=float)
            base_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, BASE_FRAME_NAME)
            arm_base_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, ARM_BASE_FRAME_NAME)
            print(
                f"MuJoCo viewer: model='{model_file}', runtime_root_body='{mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, root_body_id)}', "
                f"logical_base_frame='{BASE_FRAME_NAME}', logical_arm_base_frame='{ARM_BASE_FRAME_NAME}', "
                f"frames={x_traj.shape[0]}, dt={dt:.6f}s"
            )
            if base_body_id < 0 or arm_base_body_id < 0:
                print(
                    "[INFO] MuJoCo URDF importer collapsed fixed-link frames (e.g. base/base_link). "
                    "Using first dynamic body as runtime root is expected."
                )

        goal_xyz = np.array(P_GOAL if goal_point is None else goal_point, dtype=float).reshape(3)
        goal_q = np.array(Q_GOAL_WXYZ if goal_quat_wxyz is None else goal_quat_wxyz, dtype=float).reshape(4)

        with mj_viewer.launch_passive(model, data) as viewer:
            if MUJOCO_ADD_ORIGIN_AXES:
                with viewer.lock():
                    _add_origin_axes_to_viewer(
                        viewer,
                        mj,
                        length=MUJOCO_ORIGIN_AXIS_LENGTH,
                        width_px=MUJOCO_ORIGIN_AXIS_WIDTH_PX,
                    )
            if MUJOCO_ADD_GOAL_AXES:
                with viewer.lock():
                    _add_goal_axes_to_viewer(
                        viewer,
                        mj,
                        goal_xyz=goal_xyz,
                        goal_q_wxyz=goal_q,
                        length=MUJOCO_GOAL_AXIS_LENGTH,
                        width_px=MUJOCO_GOAL_AXIS_WIDTH_PX,
                    )
            while viewer.is_running():
                for k in range(x_traj.shape[0]):
                    if not viewer.is_running():
                        break
                    tic = time.time()

                    z = float(x_traj[k, 0])
                    phi = float(x_traj[k, 1])
                    theta = float(x_traj[k, 2])

                    if has_explicit_root_joints:
                        # Directly write base DoFs to root joints: z, roll(phi), pitch(theta).
                        data.qpos[root_qpos_adrs[0]] = z
                        data.qpos[root_qpos_adrs[1]] = phi
                        data.qpos[root_qpos_adrs[2]] = theta
                    else:
                        q_base = _base_quat_wxyz_np(phi, theta)
                        p_base = np.array([0.0, 0.0, z], dtype=float)
                        R_base = _quat_to_rotmat_wxyz_np(q_base)

                        # Apply virtual base transform to fallback root body.
                        model.body_pos[root_body_id] = p_base + R_base @ root_pos_nom
                        model.body_quat[root_body_id] = _quat_mul_wxyz_np(q_base, root_quat_nom)

                    # Apply arm joint trajectory.
                    for j, qadr in enumerate(qpos_adrs):
                        data.qpos[qadr] = float(x_traj[k, 3 + j])

                    mj.mj_forward(model, data)
                    viewer.sync()

                    target = dt / max(realtime_factor, 1e-6)
                    sleep_t = max(0.0, target - (time.time() - tic))
                    if sleep_t > 0.0:
                        time.sleep(sleep_t)
                if not loop:
                    break
    finally:
        if tmp_dir is not None and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)


def _barrier_stage_cost(xk: ca.MX, uk: ca.MX, params: OCPParams) -> ca.MX:
    lb = 0
    for i in range(9):
        if i < 3:
            mu = params.mu_base
            delta = params.delta_base
        else:
            mu = params.mu_arm
            delta = params.delta_arm

        # Canonical h >= 0 constraints:
        # h1 = x - xmin, h2 = xmax - x, h3 = xdot - xdot_min, h4 = xdot_max - xdot
        # Here xdot = u due to dynamics xdot = u.
        h1 = xk[i] - params.x_min[i]
        h2 = params.x_max[i] - xk[i]
        h3 = uk[i] - params.u_min[i]
        h4 = params.u_max[i] - uk[i]

        lb += relaxed_barrier(h1, mu, delta)
        lb += relaxed_barrier(h2, mu, delta)
        lb += relaxed_barrier(h3, mu, delta)
        lb += relaxed_barrier(h4, mu, delta)
    return lb


def build_and_solve_ocp(params: OCPParams):
    global _ARM_FK_FUN, _BASE_TO_ARMBASE_P, _BASE_TO_ARMBASE_Q

    model, data, base_id, arm_base_id, ee_id, p_ba, q_ba = load_model(
        params.urdf_path,
        params.base_frame_name,
        params.arm_base_frame_name,
        params.ee_frame_name,
    )

    # Must be differentiable in CasADi for OCP; otherwise raise explicit actionable error.
    _ARM_FK_FUN = _build_fk_arm_casadi(model, arm_base_id, ee_id)
    _BASE_TO_ARMBASE_P = p_ba
    _BASE_TO_ARMBASE_Q = q_ba

    dt = params.T / params.N

    opti = ca.Opti()
    X = opti.variable(9, params.N + 1)
    U = opti.variable(9, params.N)

    x0_dm = ca.DM(params.x0)
    x_min_dm = ca.DM(params.x_min)
    x_max_dm = ca.DM(params.x_max)
    u_min_dm = ca.DM(params.u_min)
    u_max_dm = ca.DM(params.u_max)
    R_dm = ca.DM(params.R)
    p_goal_dm = ca.DM(params.p_goal)
    q_goal_dm = quat_normalize_wxyz(ca.DM(params.q_goal_wxyz))

    # Initial condition
    opti.subject_to(X[:, 0] == x0_dm)

    # Hard bounds for better convergence / barrier safety.
    for k in range(params.N + 1):
        opti.subject_to(opti.bounded(x_min_dm, X[:, k], x_max_dm))
    for k in range(params.N):
        opti.subject_to(opti.bounded(u_min_dm, U[:, k], u_max_dm))

    def f_dyn(x, u):
        return u

    # Dynamics constraints
    for k in range(params.N):
        xk = X[:, k]
        uk = U[:, k]

        if params.integrator.lower() == "rk4":
            k1 = f_dyn(xk, uk)
            k2 = f_dyn(xk + 0.5 * dt * k1, uk)
            k3 = f_dyn(xk + 0.5 * dt * k2, uk)
            k4 = f_dyn(xk + dt * k3, uk)
            x_next = xk + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        else:
            # Default explicit Euler
            x_next = xk + dt * uk

        opti.subject_to(X[:, k + 1] == x_next)

    # Objective
    J = 0
    for k in range(params.N):
        xk = X[:, k]
        uk = U[:, k]

        running_u = ca.mtimes([uk.T, R_dm, uk])
        L_B = _barrier_stage_cost(xk, uk, params)
        J += dt * (running_u + params.w_barrier * L_B)

    pN, qN = compose_fk(X[0, params.N], X[1, params.N], X[2, params.N], X[3:9, params.N])
    e_pos = pN - p_goal_dm
    e_ori = orientation_error_vec(qN, q_goal_dm, params.use_quat_sign_correction)
    terminal_cost = params.w_ee_pos * ca.dot(e_pos, e_pos) + params.w_ee_ori * ca.dot(e_ori, e_ori)

    J += terminal_cost
    opti.minimize(J)

    # Initial guess
    x_init = np.tile(params.x0.reshape(-1, 1), (1, params.N + 1))
    u_init = np.zeros((9, params.N))
    opti.set_initial(X, x_init)
    opti.set_initial(U, u_init)

    # IPOPT settings
    p_opts = {"expand": True}
    s_opts = {
        "max_iter": int(params.ipopt_max_iter),
        "tol": float(params.ipopt_tol),
        "print_level": int(params.ipopt_print_level),
    }
    opti.solver("ipopt", p_opts, s_opts)

    sol = opti.solve()

    x_traj = np.array(sol.value(X)).T  # (N+1, 9)
    u_traj = np.array(sol.value(U)).T  # (N, 9)

    # Build helper for post evaluation
    x_sym = ca.SX.sym("x", 9)
    p_sym, q_sym = compose_fk(x_sym[0], x_sym[1], x_sym[2], x_sym[3:9])
    epos_sym = p_sym - p_goal_dm
    eori_sym = orientation_error_vec(q_sym, q_goal_dm, params.use_quat_sign_correction)
    ee_eval = ca.Function("ee_eval", [x_sym], [p_sym, q_sym, epos_sym, eori_sym])

    p_term, q_term, epos_term, eori_term = ee_eval(x_traj[-1, :])
    pos_err_norm = float(np.linalg.norm(np.array(epos_term).reshape(-1)))
    ori_err_norm = float(np.linalg.norm(np.array(eori_term).reshape(-1)))

    print("Solved OCP.")
    print(f"terminal position error norm: {pos_err_norm:.6e}")
    print(f"terminal orientation error norm: {ori_err_norm:.6e}")

    result = {
        "x_traj": x_traj,
        "u_traj": u_traj,
        "terminal_p_ee": np.array(p_term).reshape(-1),
        "terminal_q_ee_wxyz": np.array(q_term).reshape(-1),
        "terminal_pos_err_norm": pos_err_norm,
        "terminal_ori_err_norm": ori_err_norm,
        "dt": dt,
    }

    if params.plot_errors:
        try:
            import matplotlib.pyplot as plt

            pos_err_curve = []
            ori_err_curve = []
            for k in range(params.N + 1):
                _, _, epos_k, eori_k = ee_eval(x_traj[k, :])
                pos_err_curve.append(np.linalg.norm(np.array(epos_k).reshape(-1)))
                ori_err_curve.append(np.linalg.norm(np.array(eori_k).reshape(-1)))

            t_grid = np.linspace(0.0, params.T, params.N + 1)
            plt.figure(figsize=(8, 4))
            plt.plot(t_grid, pos_err_curve, label="||e_pos||")
            plt.plot(t_grid, ori_err_curve, label="||e_ori||")
            plt.xlabel("time [s]")
            plt.ylabel("error norm")
            plt.title("Terminal-tracking error curves")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"[WARN] plot_errors=True but plotting failed: {e}")

    return result


def main():
    if not os.path.isfile(URDF_PATH):
        print(f"URDF file not found: {URDF_PATH}")
        print("Please modify URDF_PATH at the top of this script.")
        return

    # Build unified bounds:
    # - base(3) bounds are manually configured above
    # - arm(6) bounds are read from URDF joint limits
    arm_x_min, arm_x_max, arm_u_min, arm_u_max = _read_arm_bounds_from_urdf(URDF_PATH, ARM_JOINT_NAMES)
    x_min, x_max, u_min, u_max = _make_xu_bounds(arm_x_min, arm_x_max, arm_u_min, arm_u_max)

    q_goal = Q_GOAL_WXYZ / np.linalg.norm(Q_GOAL_WXYZ)

    params = OCPParams(
        urdf_path=URDF_PATH,
        base_frame_name=BASE_FRAME_NAME,
        arm_base_frame_name=ARM_BASE_FRAME_NAME,
        ee_frame_name=EE_FRAME_NAME,
        T=T,
        N=N,
        integrator=INTEGRATOR,
        R=R,
        w_ee_pos=W_EE_POS,
        w_ee_ori=W_EE_ORI,
        w_barrier=W_BARRIER,
        x0=X0,
        x_min=x_min,
        x_max=x_max,
        u_min=u_min,
        u_max=u_max,
        mu_base=MU_BASE,
        delta_base=DELTA_BASE,
        mu_arm=MU_ARM,
        delta_arm=DELTA_ARM,
        p_goal=P_GOAL,
        q_goal_wxyz=q_goal,
        use_quat_sign_correction=USE_QUAT_SIGN_CORRECTION,
        plot_errors=PLOT_ERRORS,
        ipopt_max_iter=IPOPT_MAX_ITER,
        ipopt_tol=IPOPT_TOL,
        ipopt_print_level=IPOPT_PRINT_LEVEL,
    )

    # Print model assumption explicitly to avoid ambiguity.
    print("Model assumption: x_b=y_b=psi_b=0 fixed, optimize only z/phi/theta + 6 arm joints.")
    print(f"URDF frame mapping: paper base -> '{BASE_FRAME_NAME}', arm base -> '{ARM_BASE_FRAME_NAME}'.")
    print("Bounds: base(3) from manual constants; arm(6) limits from URDF by explicit ARM_JOINT_NAMES order.")
    print("Base rotation order is strictly R_x(phi) then R_y(theta). Quaternion order is wxyz.")

    res = build_and_solve_ocp(params)

    print("x_traj shape:", res["x_traj"].shape)
    print("u_traj shape:", res["u_traj"].shape)

    if MUJOCO_VISUALIZE:
        try:
            visualize_trajectory_mujoco(
                x_traj=res["x_traj"],
                dt=res["dt"],
                model_path=MUJOCO_MODEL_PATH,
                arm_joint_names=ARM_JOINT_NAMES,
                goal_point=params.p_goal,
                goal_quat_wxyz=params.q_goal_wxyz,
                loop=MUJOCO_LOOP,
                realtime_factor=MUJOCO_REALTIME_FACTOR,
            )
        except Exception as e:
            print(f"[WARN] MuJoCo visualization failed: {e}")


if __name__ == "__main__":
    main()
