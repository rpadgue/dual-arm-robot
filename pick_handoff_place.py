
import random
import time
from collections import Counter, defaultdict

import numpy as np
import pybullet as p
import pybullet_data

from end_effectors import (
    create_two_finger_gripper,
    create_suction_gripper
)
from object_interaction import ObjectInteraction


# ============================================================
# CONFIGURATION
# ============================================================

DT = 1.0 / 240.0
END_LINK = 6
JOINT_COUNT = 7

MOVE_TIME = 0.85
MAX_VELOCITY = 2.0
MOTOR_FORCE = 200

SETTLE_TOLERANCE = 0.015
SETTLE_MAX_STEPS = 120

TABLE_HEIGHT = 0.35
CUBE_SIZE = 0.10
CUBE_Z = TABLE_HEIGHT + CUBE_SIZE / 2

TABLE_A_X = -1.05
TABLE_B_X = 1.05

MAX_OBJECTS = 5
RANDOM_SEED = 42

if RANDOM_SEED is not None:
    random.seed(RANDOM_SEED)

GRIPPER_TCP = np.array([0.0, 0.0, 0.105])
SUCTION_TCP = np.array([0.0, 0.0, 0.065])

DOWN = p.getQuaternionFromEuler([np.pi, 0, 0])
SUCTION_SIDE = p.getQuaternionFromEuler(
    [0, -np.pi / 2, 0]
)

PREFERRED_HANDOFF = np.array([-0.20, 0.20, 0.75])
PREFERRED_PLACE = np.array([1.12, 0.05, CUBE_Z])

HANDOFF_ORIENTATIONS = [
    p.getQuaternionFromEuler([0, 0, 0]),
    p.getQuaternionFromEuler([0, 0, np.pi / 2]),
    p.getQuaternionFromEuler([0, np.pi / 2, 0]),
    p.getQuaternionFromEuler([0, -np.pi / 2, 0]),
]

PLACE_ORIENTATIONS = [
    p.getQuaternionFromEuler([roll, pitch, yaw])
    for roll in [0, np.pi / 2, -np.pi / 2, np.pi]
    for pitch in [0, np.pi / 2, -np.pi / 2]
    for yaw in [0, np.pi / 2, np.pi]
]

APPROACH_HEIGHTS = [0.12, 0.18, 0.25]

ARM_CLEARANCE = 0.001
CUBE_CLEARANCE = 0.005
PLACEMENT_CONTACT_TOLERANCE = 0.008

TRAJECTORY_SAMPLES = 24
JOINT_LIMIT_MARGIN = 0.12

# During simultaneous movement, keep additional
# separation between the two robot arms.
ROBOT_ROBOT_CLEARANCE = 0.025

# Robot B stops 12 cm before the final suction contact.
PREHANDOFF_OFFSET = np.array([0.12, 0, 0])

PRINT_COLLISION_SUMMARY = True

# Performance tracking.
TIMINGS = defaultdict(float)

# Reuse previously successful IK solutions as starting
# guesses. They are never used without recalculation.
IK_SEED_CACHE = {
    robot_key: []
    for robot_key in ["A", "B"]
}


# ============================================================
# INITIALIZE PYBULLET
# ============================================================

physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(DT)
p.setRealTimeSimulation(0)

p.loadURDF("plane.urdf")


# ============================================================
# TABLES
# ============================================================

def create_table(x, color):

    half = [0.175, 0.20, TABLE_HEIGHT / 2]

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half,
        rgbaColor=color
    )

    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=[x, 0, TABLE_HEIGHT / 2]
    )


table_a = create_table(TABLE_A_X, [0.2, 0.45, 0.9, 1])
table_b = create_table(TABLE_B_X, [0.2, 0.8, 0.45, 1])


# ============================================================
# ROBOTS
# ============================================================

robot_a = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[-0.7, 0, 0],
    useFixedBase=True
)

robot_b = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0.7, 0, 0],
    baseOrientation=p.getQuaternionFromEuler(
        [0, 0, np.pi]
    ),
    useFixedBase=True
)

gripper_a = create_two_finger_gripper(robot_a)
suction_b = create_suction_gripper(robot_b)

for robot, tool in [
    (robot_a, gripper_a),
    (robot_b, suction_b)
]:
    for i in range(-1, p.getNumJoints(robot)):
        for j in range(-1, p.getNumJoints(tool)):
            p.setCollisionFilterPair(
                robot, tool, i, j, 0
            )


# ============================================================
# PHYSICS HELPERS
# ============================================================

def step():
    p.stepSimulation()


def simulate(steps=30):
    for _ in range(steps):
        step()


def robot_name(robot):
    return "A" if robot == robot_a else "B"


def quintic(t):
    t = np.clip(t, 0.0, 1.0)
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def timed(label, function, *args, **kwargs):

    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start

    TIMINGS[label] += elapsed

    print(f"{label}: {elapsed:.3f} seconds")

    return result


# ============================================================
# ROBOT STATE
# ============================================================

def get_joints(robot):

    return np.array([
        p.getJointState(robot, j)[0]
        for j in range(JOINT_COUNT)
    ])


def get_wrist_pose(robot):

    state = p.getLinkState(
        robot,
        END_LINK,
        computeForwardKinematics=True
    )

    return state[4], state[5]


def get_cube_pose(cube):
    return p.getBasePositionAndOrientation(cube)


def get_limits(robot):

    lower = []
    upper = []

    for j in range(JOINT_COUNT):
        info = p.getJointInfo(robot, j)
        lower.append(info[8])
        upper.append(info[9])

    return np.array(lower), np.array(upper)


def orientation_error(q1, q2):

    dot = abs(np.dot(q1, q2))
    dot = np.clip(dot, -1.0, 1.0)

    return 2 * np.arccos(dot)


# ============================================================
# IK PLANNING CLIENT
# ============================================================

ik_client = p.connect(p.DIRECT)

p.setAdditionalSearchPath(
    pybullet_data.getDataPath(),
    physicsClientId=ik_client
)

ik_robot_a = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[-0.7, 0, 0],
    useFixedBase=True,
    physicsClientId=ik_client
)

ik_robot_b = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0.7, 0, 0],
    baseOrientation=p.getQuaternionFromEuler(
        [0, 0, np.pi]
    ),
    useFixedBase=True,
    physicsClientId=ik_client
)

IK_COPY = {
    robot_a: ik_robot_a,
    robot_b: ik_robot_b
}

ik_plane = p.loadURDF(
    "plane.urdf",
    physicsClientId=ik_client
)

IK_TABLES = {}

for name, x in [
    ("Table A", TABLE_A_X),
    ("Table B", TABLE_B_X)
]:

    shape = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[0.175, 0.20, TABLE_HEIGHT / 2],
        physicsClientId=ik_client
    )

    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=shape,
        basePosition=[x, 0, TABLE_HEIGHT / 2],
        physicsClientId=ik_client
    )

    IK_TABLES[name] = body


# ============================================================
# IK HELPERS
# ============================================================

def set_ghost_joints(ghost, joints):

    for j in range(JOINT_COUNT):
        p.resetJointState(
            ghost,
            j,
            float(joints[j]),
            physicsClientId=ik_client
        )


def validate_fk(
    ghost,
    angles,
    position,
    orientation
):

    set_ghost_joints(ghost, angles)

    state = p.getLinkState(
        ghost,
        END_LINK,
        computeForwardKinematics=True,
        physicsClientId=ik_client
    )

    actual_pos = np.array(state[4])
    actual_ori = state[5]

    pos_error = np.linalg.norm(
        actual_pos - np.array(position)
    )

    ori_error = orientation_error(
        actual_ori,
        orientation
    )

    return pos_error, ori_error


def remember_ik(robot, angles):

    key = robot_name(robot)
    cache = IK_SEED_CACHE[key]

    if all(
        np.linalg.norm(angles - old) > 0.1
        for old in cache
    ):
        cache.insert(0, angles.copy())

    del cache[8:]


def calculate_ik(
    robot,
    position,
    orientation,
    reference=None,
    quick=False
):

    position = np.array(position, dtype=float)
    current = get_joints(robot)

    if reference is None:
        reference = current

    reference = np.array(reference, dtype=float)

    ghost = IK_COPY[robot]
    lower, upper = get_limits(robot)
    ranges = upper - lower

    seeds = [
        reference,
        current,
    ]

    if not quick:
        seeds += IK_SEED_CACHE[robot_name(robot)][:3]
        seeds += [
            np.zeros(7),
            np.array([0, 0.3, 0, -1, 0, 0.5, 0]),
            np.array([0, -0.3, 0, -1, 0, 0.5, 0])
        ]

    best = None
    best_score = float("inf")

    for seed in seeds:

        seed = np.clip(seed, lower, upper)
        set_ghost_joints(ghost, seed)

        result = p.calculateInverseKinematics(
            bodyUniqueId=ghost,
            endEffectorLinkIndex=END_LINK,
            targetPosition=position.tolist(),
            targetOrientation=list(orientation),
            lowerLimits=lower.tolist(),
            upperLimits=upper.tolist(),
            jointRanges=ranges.tolist(),
            restPoses=seed.tolist(),
            maxNumIterations=120 if quick else 150,
            residualThreshold=1e-4,
            physicsClientId=ik_client
        )

        angles = np.array(result[:7], dtype=float)

        if not np.all(np.isfinite(angles)):
            continue

        if np.any(angles < lower - 1e-4):
            continue

        if np.any(angles > upper + 1e-4):
            continue

        pos_error, ori_error = validate_fk(
            ghost,
            angles,
            position,
            orientation
        )

        if pos_error > 0.03 or ori_error > 0.20:
            continue

        score = (
            np.linalg.norm(angles - reference)
            + 2 * pos_error
            + ori_error
        )

        if score < best_score:
            best = angles.copy()
            best_score = score

    if best is None:
        raise RuntimeError(
            f"Robot {robot_name(robot)}: "
            f"No valid IK for {np.round(position, 4)}"
        )

    remember_ik(robot, best)

    return best


# ============================================================
# OBJECT TRANSFORMS
# ============================================================

def wrist_for_tcp(tcp_world, orientation, tcp):

    offset = p.rotateVector(
        orientation,
        list(tcp)
    )

    return np.array(tcp_world) - np.array(offset)


def relative_cube_pose(robot, cube):

    wp, wq = get_wrist_pose(robot)
    cp, cq = get_cube_pose(cube)

    inv_p, inv_q = p.invertTransform(wp, wq)

    return p.multiplyTransforms(
        inv_p, inv_q, cp, cq
    )


def wrist_for_cube_pose(
    cube_position,
    cube_orientation,
    relative_position,
    relative_orientation
):

    inv_p, inv_q = p.invertTransform(
        relative_position,
        relative_orientation
    )

    wp, wq = p.multiplyTransforms(
        list(cube_position),
        list(cube_orientation),
        inv_p,
        inv_q
    )

    return np.array(wp), wq


# ============================================================
# COLLISION CHECKING
# ============================================================

def cube_vertical_half_extent(orientation):

    matrix = np.array(
        p.getMatrixFromQuaternion(orientation)
    ).reshape(3, 3)

    return float(
        (CUBE_SIZE / 2)
        * np.sum(np.abs(matrix[2, :]))
    )


def cube_pose_from_ghost(ghost, relative_b):

    state = p.getLinkState(
        ghost,
        END_LINK,
        computeForwardKinematics=True,
        physicsClientId=ik_client
    )

    cp, cq = p.multiplyTransforms(
        state[4],
        state[5],
        relative_b[0],
        relative_b[1]
    )

    return np.array(cp), cq


def cube_clearance_valid(
    cube_position,
    cube_orientation,
    allow_table_b_contact=False
):

    cube_bottom = (
        cube_position[2]
        - cube_vertical_half_extent(cube_orientation)
    )

    for name, x in [
        ("Table A", TABLE_A_X),
        ("Table B", TABLE_B_X)
    ]:

        overlap = (
            abs(cube_position[0] - x)
            < 0.175 + CUBE_SIZE / 2
            and abs(cube_position[1])
            < 0.20 + CUBE_SIZE / 2
        )

        if not overlap:
            continue

        required = TABLE_HEIGHT + CUBE_CLEARANCE

        if name == "Table B" and allow_table_b_contact:
            required = (
                TABLE_HEIGHT
                - PLACEMENT_CONTACT_TOLERANCE
            )

        if cube_bottom < required:
            return (
                False,
                f"carried cube / {name}",
                cube_bottom - TABLE_HEIGHT
            )

    return True, None, None


def configuration_valid(
    robot,
    joints,
    relative_b=None,
    allow_table_b_contact=False
):

    lower, upper = get_limits(robot)

    if np.any(joints < lower - 1e-4):
        return False, "joint lower limit", None

    if np.any(joints > upper + 1e-4):
        return False, "joint upper limit", None

    ghost = IK_COPY[robot]
    set_ghost_joints(ghost, joints)

    p.performCollisionDetection(
        physicsClientId=ik_client
    )

    for name, table in IK_TABLES.items():

        contacts = p.getClosestPoints(
            ghost,
            table,
            distance=ARM_CLEARANCE,
            physicsClientId=ik_client
        )

        moving = [
            c for c in contacts
            if c[3] != -1
        ]

        if moving:
            closest = min(moving, key=lambda c: c[8])
            return (
                False,
                f"robot arm / {name} (link {closest[3]})",
                float(closest[8])
            )

    contacts = p.getClosestPoints(
        ghost,
        ik_plane,
        distance=ARM_CLEARANCE,
        physicsClientId=ik_client
    )

    moving = [
        c for c in contacts
        if c[3] != -1
    ]

    if moving:
        closest = min(moving, key=lambda c: c[8])
        return (
            False,
            f"robot arm / ground (link {closest[3]})",
            float(closest[8])
        )

    if relative_b is not None:

        cp, cq = cube_pose_from_ghost(
            ghost,
            relative_b
        )

        valid, reason, distance = cube_clearance_valid(
            cp,
            cq,
            allow_table_b_contact
        )

        if not valid:
            return False, reason, distance

    return True, None, None


def validate_joint_path(
    robot,
    start,
    target,
    relative_b=None,
    allow_final_contact=False
):

    start = np.array(start)
    target = np.array(target)

    for t in np.linspace(0, 1, TRAJECTORY_SAMPLES):

        joints = start + quintic(t) * (target - start)

        valid, reason, distance = configuration_valid(
            robot,
            joints,
            relative_b=relative_b,
            allow_table_b_contact=allow_final_contact
        )

        if not valid:
            return False, reason, distance

    return True, None, None


def robots_separated(joints_a, joints_b):

    set_ghost_joints(ik_robot_a, joints_a)
    set_ghost_joints(ik_robot_b, joints_b)

    contacts = p.getClosestPoints(
        ik_robot_a,
        ik_robot_b,
        distance=ROBOT_ROBOT_CLEARANCE,
        physicsClientId=ik_client
    )

    moving_contacts = [
        c for c in contacts
        if c[3] != -1 and c[4] != -1
    ]

    if moving_contacts:
        return False

    return True


def validate_coordinated_path(
    start_a,
    target_a,
    start_b,
    target_b
):

    for t in np.linspace(0, 1, TRAJECTORY_SAMPLES):

        s = quintic(t)

        qa = start_a + s * (target_a - start_a)
        qb = start_b + s * (target_b - start_b)

        if not robots_separated(qa, qb):
            return False

    return True


# ============================================================
# JOINT MOTION
# ============================================================

def command_joints(robot, target):

    for j in range(JOINT_COUNT):
        p.setJointMotorControl2(
            robot,
            j,
            p.POSITION_CONTROL,
            targetPosition=float(target[j]),
            force=MOTOR_FORCE,
            maxVelocity=MAX_VELOCITY
        )


def required_duration(start, target, requested):

    max_change = np.max(
        np.abs(target - start)
    )

    minimum = (
        1.875 * max_change / MAX_VELOCITY
    )

    return max(requested, minimum)


def settle_robots(targets):

    for _ in range(SETTLE_MAX_STEPS):

        for robot, target in targets:
            command_joints(robot, target)

        step()

        if all(
            np.linalg.norm(get_joints(robot) - target)
            < SETTLE_TOLERANCE
            for robot, target in targets
        ):
            break

    for robot, target in targets:

        error = np.linalg.norm(
            get_joints(robot) - target
        )

        print(
            f"Robot {robot_name(robot)} "
            f"joint error: {error:.4f} rad"
        )

        if error > 0.20:
            raise RuntimeError(
                f"Robot {robot_name(robot)} "
                "failed to reach joint target"
            )


def move_joints(robot, target, duration=MOVE_TIME):

    start = get_joints(robot)
    target = np.array(target[:7], dtype=float)

    duration = required_duration(
        start,
        target,
        duration
    )

    steps = max(1, int(duration / DT))

    for k in range(steps):

        s = quintic((k + 1) / steps)

        command_joints(
            robot,
            start + s * (target - start)
        )

        step()

    settle_robots([(robot, target)])


def move_both(
    target_a,
    target_b,
    duration=MOVE_TIME
):

    start_a = get_joints(robot_a)
    start_b = get_joints(robot_b)

    target_a = np.array(target_a, dtype=float)
    target_b = np.array(target_b, dtype=float)

    if not validate_coordinated_path(
        start_a,
        target_a,
        start_b,
        target_b
    ):
        raise RuntimeError(
            "Coordinated handoff trajectory "
            "has a robot-robot collision risk"
        )

    duration = max(
        required_duration(start_a, target_a, duration),
        required_duration(start_b, target_b, duration)
    )

    steps = max(1, int(duration / DT))

    print(
        "Moving Robot A and Robot B simultaneously"
    )

    for k in range(steps):

        s = quintic((k + 1) / steps)

        qa = start_a + s * (target_a - start_a)
        qb = start_b + s * (target_b - start_b)

        command_joints(robot_a, qa)
        command_joints(robot_b, qb)

        step()

    settle_robots([
        (robot_a, target_a),
        (robot_b, target_b)
    ])


def move_pose(robot, position, orientation):

    target = calculate_ik(
        robot,
        position,
        orientation
    )

    move_joints(robot, target)

    actual_pos, actual_ori = get_wrist_pose(robot)

    pos_error = np.linalg.norm(
        np.array(actual_pos) - np.array(position)
    )

    ori_error = orientation_error(
        actual_ori,
        orientation
    )

    print(
        f"Robot {robot_name(robot)} "
        f"position error: {pos_error:.4f} m | "
        f"orientation error: {ori_error:.4f} rad"
    )

    if pos_error > 0.06 or ori_error > 0.25:
        raise RuntimeError(
            f"Robot {robot_name(robot)} "
            "failed to reach requested pose"
        )


def move_tcp(robot, target, orientation, tcp):

    wrist = wrist_for_tcp(
        target,
        orientation,
        tcp
    )

    print(
        f"Robot {robot_name(robot)} "
        "moving TCP to:",
        np.round(target, 4)
    )

    move_pose(robot, wrist, orientation)


# ============================================================
# RANDOM PICKUP
# ============================================================

def random_pick_position():

    for _ in range(100):

        position = np.array([
            random.uniform(-1.12, -0.98),
            random.uniform(-0.08, 0.08),
            CUBE_Z
        ])

        approach = wrist_for_tcp(
            position + [0, 0, 0.20],
            DOWN,
            GRIPPER_TCP
        )

        pickup = wrist_for_tcp(
            position,
            DOWN,
            GRIPPER_TCP
        )

        try:

            q = calculate_ik(
                robot_a,
                approach,
                DOWN
            )

            calculate_ik(
                robot_a,
                pickup,
                DOWN,
                reference=q
            )

            print(
                "Validated pickup position:",
                np.round(position, 4)
            )

            return position

        except RuntimeError:
            continue

    raise RuntimeError(
        "Could not find reachable pickup position"
    )


def create_cube(position):

    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[CUBE_SIZE / 2] * 3
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[CUBE_SIZE / 2] * 3,
        rgbaColor=[0.9, 0.2, 0.1, 1]
    )

    cube = p.createMultiBody(
        baseMass=0.2,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position.tolist()
    )

    for body in [
        robot_a,
        robot_b,
        gripper_a,
        suction_b
    ]:
        for link in range(-1, p.getNumJoints(body)):
            p.setCollisionFilterPair(
                body,
                cube,
                link,
                -1,
                0
            )

    return cube


# ============================================================
# HANDOFF PLANNING
# ============================================================

def find_handoff(relative_a):

    print("Searching for handoff...")

    positions = [
        PREFERRED_HANDOFF,
        PREFERRED_HANDOFF + [0, -0.10, 0],
        PREFERRED_HANDOFF + [0.10, 0, 0],
        PREFERRED_HANDOFF + [0, 0, -0.10],
    ]

    positions += [
        np.array([x, y, z])
        for x in [-0.20, -0.10, 0, 0.10]
        for y in [0, 0.10, 0.20]
        for z in [0.65, 0.75]
    ]

    for position in positions:

        for orientation in HANDOFF_ORIENTATIONS:

            a_wp, a_wq = wrist_for_cube_pose(
                position,
                orientation,
                *relative_a
            )

            try:

                calculate_ik(robot_a, a_wp, a_wq)

                contact = (
                    position
                    + np.array([CUBE_SIZE / 2, 0, 0])
                )

                b_wp = wrist_for_tcp(
                    contact,
                    SUCTION_SIDE,
                    SUCTION_TCP
                )

                b_joints = calculate_ik(
                    robot_b,
                    b_wp,
                    SUCTION_SIDE
                )

                pre_wp = wrist_for_tcp(
                    contact + PREHANDOFF_OFFSET,
                    SUCTION_SIDE,
                    SUCTION_TCP
                )

                calculate_ik(
                    robot_b,
                    pre_wp,
                    SUCTION_SIDE,
                    reference=b_joints
                )

                print(
                    "Selected handoff:",
                    np.round(position, 4)
                )

                return position, orientation

            except RuntimeError:
                continue

    raise RuntimeError(
        "No shared handoff pose found"
    )


# ============================================================
# FAST PLACEMENT PLANNING
# ============================================================

def placement_positions():

    preferred = [
        PREFERRED_PLACE,
        np.array([1.12, -0.05, CUBE_Z]),
        np.array([1.06, 0.05, CUBE_Z]),
        np.array([1.06, -0.05, CUBE_Z]),
        np.array([1.00, 0.05, CUBE_Z]),
        np.array([1.12, 0.00, CUBE_Z]),
    ]

    fallback = [
        np.array([x, y, CUBE_Z])
        for x in [0.94, 1.00, 1.06, 1.12]
        for y in [0, -0.05, 0.05]
    ]

    result = []
    seen = set()

    for position in preferred + fallback:

        key = tuple(np.round(position, 4))

        if key not in seen:
            seen.add(key)
            result.append(position)

    return result


def waypoint_positions(current, target):

    middle = (np.array(current) + np.array(target)) / 2

    return [
        np.array([middle[0], middle[1], 0.80]),
        np.array([0.25, 0.10, 0.80]),
        np.array([0.35, 0.00, 0.80]),
        np.array([0.45, 0.10, 0.75]),
        np.array([0.55, 0.00, 0.75]),
        np.array([0.65, 0.00, 0.70]),
    ]


def try_placement(
    position,
    orientation,
    height,
    relative_b,
    current_joints,
    current_cube_pos,
    current_cube_ori,
    rejection_counts,
    quick=True
):

    above = position + np.array([0, 0, height])

    above_wp, above_wq = wrist_for_cube_pose(
        above,
        orientation,
        *relative_b
    )

    place_wp, place_wq = wrist_for_cube_pose(
        position,
        orientation,
        *relative_b
    )

    try:

        above_joints = calculate_ik(
            robot_b,
            above_wp,
            above_wq,
            reference=current_joints,
            quick=quick
        )

        place_joints = calculate_ik(
            robot_b,
            place_wp,
            place_wq,
            reference=above_joints,
            quick=quick
        )

    except RuntimeError:
        rejection_counts["IK failure"] += 1
        return None

    final_ok, reason, _ = configuration_valid(
        robot_b,
        place_joints,
        relative_b=relative_b,
        allow_table_b_contact=True
    )

    if not final_ok:
        rejection_counts["destination: " + reason] += 1
        return None

    lowering_ok, reason, _ = validate_joint_path(
        robot_b,
        above_joints,
        place_joints,
        relative_b=relative_b,
        allow_final_contact=True
    )

    if not lowering_ok:
        rejection_counts["lowering: " + reason] += 1
        return None

    transport_ok, reason, _ = validate_joint_path(
        robot_b,
        current_joints,
        above_joints,
        relative_b=relative_b
    )

    if transport_ok:

        return {
            "position": position,
            "height": height,
            "orientation": orientation,
            "path": [above_joints, place_joints],
            "path_type": "direct"
        }

    rejection_counts["transport: " + reason] += 1

    above_cube = position + np.array([0, 0, height])

    for waypoint in waypoint_positions(
        current_cube_pos,
        above_cube
    ):

        wp, wq = wrist_for_cube_pose(
            waypoint,
            current_cube_ori,
            *relative_b
        )

        try:

            waypoint_joints = calculate_ik(
                robot_b,
                wp,
                wq,
                reference=current_joints,
                quick=True
            )

        except RuntimeError:
            continue

        first_ok, _, _ = validate_joint_path(
            robot_b,
            current_joints,
            waypoint_joints,
            relative_b=relative_b
        )

        if not first_ok:
            continue

        second_ok, _, _ = validate_joint_path(
            robot_b,
            waypoint_joints,
            above_joints,
            relative_b=relative_b
        )

        if not second_ok:
            continue

        return {
            "position": position,
            "height": height,
            "orientation": orientation,
            "path": [
                waypoint_joints,
                above_joints,
                place_joints
            ],
            "path_type": "waypoint"
        }

    return None


def find_place(relative_b, cube):

    print("Searching for placement (fast progressive mode)...")

    start_time = time.perf_counter()

    current_joints = get_joints(robot_b)
    current_cube_pos, current_cube_ori = get_cube_pose(cube)

    current_cube_pos = np.array(current_cube_pos)

    positions = placement_positions()

    rejection_counts = Counter()
    checked = 0

    # First pass: preferred locations and a small
    # orientation set, with the shortest approach.
    fast_positions = positions[:4]
    fast_orientations = PLACE_ORIENTATIONS[:12]

    for position in fast_positions:

        for orientation in fast_orientations:

            checked += 1

            result = try_placement(
                position,
                orientation,
                0.12,
                relative_b,
                current_joints,
                current_cube_pos,
                current_cube_ori,
                rejection_counts,
                quick=True
            )

            if result is not None:

                elapsed = time.perf_counter() - start_time

                print("FAST PLACEMENT FOUND")
                print("Candidates checked:", checked)
                print("Planning time:", round(elapsed, 3), "s")
                print(
                    "Selected placement:",
                    np.round(result["position"], 4)
                )
                print("Path type:", result["path_type"])

                return result

    # Fallback: all positions, orientations, and heights.
    print("Fast search failed; expanding search...")

    for position in positions:

        for orientation in PLACE_ORIENTATIONS:

            for height in APPROACH_HEIGHTS:

                checked += 1

                result = try_placement(
                    position,
                    orientation,
                    height,
                    relative_b,
                    current_joints,
                    current_cube_pos,
                    current_cube_ori,
                    rejection_counts,
                    quick=False
                )

                if result is not None:

                    elapsed = time.perf_counter() - start_time

                    print("FALLBACK PLACEMENT FOUND")
                    print("Candidates checked:", checked)
                    print(
                        "Planning time:",
                        round(elapsed, 3),
                        "s"
                    )
                    print(
                        "Selected placement:",
                        np.round(result["position"], 4)
                    )
                    print("Path type:", result["path_type"])

                    return result

    if PRINT_COLLISION_SUMMARY:

        print("\nPATH REJECTION SUMMARY")

        for reason, count in rejection_counts.most_common():
            print(f"{reason}: {count}")

    raise RuntimeError(
        "No validated placement path found"
    )


# ============================================================
# ADAPTIVE RETREAT
# ============================================================

def retreat_robot_b(cube, relative_b):

    current_pos, current_ori = get_cube_pose(cube)
    current_pos = np.array(current_pos)

    offsets = [
        [0.05, 0, 0],
        [0.10, 0, 0],
        [0.05, 0, 0.05],
        [0, 0, 0.05],
        [0, -0.05, 0],
        [0, 0.05, 0],
    ]

    for offset in offsets:

        wp, wq = wrist_for_cube_pose(
            current_pos + np.array(offset),
            current_ori,
            *relative_b
        )

        try:

            calculate_ik(robot_b, wp, wq)

            print("Valid retreat offset:", offset)

            move_pose(robot_b, wp, wq)
            return

        except RuntimeError:
            print("Rejected retreat:", offset)

    raise RuntimeError("Robot B cannot retreat")


# ============================================================
# OBJECT TRANSFER
# ============================================================

home_a = get_joints(robot_a)
home_b = get_joints(robot_b)


def transfer_object(cube):

    interaction = ObjectInteraction(
        robot_a,
        robot_b,
        cube
    )

    pick_pos, _ = get_cube_pose(cube)
    pick_pos = np.array(pick_pos)

    print(
        "\nPicking object at:",
        np.round(pick_pos, 4)
    )

    print("\nSTEP 1: Robot A approaching")

    move_tcp(
        robot_a,
        pick_pos + [0, 0, 0.20],
        DOWN,
        GRIPPER_TCP
    )

    print("\nSTEP 2: Robot A lowering")

    move_tcp(
        robot_a,
        pick_pos,
        DOWN,
        GRIPPER_TCP
    )

    print("\nSTEP 3: Robot A grasping")

    if not interaction.robot_a_grasp():
        raise RuntimeError("Robot A grasp failed")

    simulate(30)

    relative_a = relative_cube_pose(robot_a, cube)

    print("\nSTEP 4: Planning handoff")

    handoff_pos, handoff_ori = timed(
        "Handoff planning",
        find_handoff,
        relative_a
    )

    print("\nSTEP 5: Robot A lifting")

    current_ori = get_cube_pose(cube)[1]

    lift_wp, lift_wq = wrist_for_cube_pose(
        pick_pos + [0, 0, 0.25],
        current_ori,
        *relative_a
    )

    move_pose(robot_a, lift_wp, lift_wq)

    print("\nSTEP 6: Coordinated handoff approach")

    a_wp, a_wq = wrist_for_cube_pose(
        handoff_pos,
        handoff_ori,
        *relative_a
    )

    # Predict the contact position at the handoff.
    contact = (
        handoff_pos
        + np.array([CUBE_SIZE / 2, 0, 0])
    )

    precontact = contact + PREHANDOFF_OFFSET

    b_wp = wrist_for_tcp(
        precontact,
        SUCTION_SIDE,
        SUCTION_TCP
    )

    target_a = calculate_ik(
        robot_a,
        a_wp,
        a_wq
    )

    target_b = calculate_ik(
        robot_b,
        b_wp,
        SUCTION_SIDE
    )

    timed(
        "Coordinated movement",
        move_both,
        target_a,
        target_b
    )

    print("\nSTEP 7: Robot B final approach")

    # Recalculate from the actual cube pose to
    # compensate for small physics tracking errors.
    actual_cube_pos = np.array(
        get_cube_pose(cube)[0]
    )

    actual_contact = (
        actual_cube_pos
        + np.array([CUBE_SIZE / 2, 0, 0])
    )

    move_tcp(
        robot_b,
        actual_contact,
        SUCTION_SIDE,
        SUCTION_TCP
    )

    print("\nSTEP 8: Transferring object")

    if not interaction.robot_b_grasp():
        raise RuntimeError("Robot B grasp failed")

    simulate(30)

    relative_b = relative_cube_pose(robot_b, cube)

    print("\nSTEP 9: Robot A retreating")

    move_joints(robot_a, home_a)

    print("\nSTEP 10: Robot B retreating")

    retreat_robot_b(cube, relative_b)

    print("\nSTEP 11: Fast placement planning")

    placement = timed(
        "Placement planning",
        find_place,
        relative_b,
        cube
    )

    print("\nSTEP 12: Transporting cube")

    for target in placement["path"][:-1]:
        move_joints(robot_b, target)

    print("\nSTEP 13: Lowering cube")

    move_joints(
        robot_b,
        placement["path"][-1]
    )

    print("\nSTEP 14: Releasing cube")

    if not interaction.release_object(robot_b):
        raise RuntimeError("Robot B release failed")

    simulate(120)

    print("\nSTEP 15: Returning Robot B home")

    move_joints(robot_b, home_b)

    print("\nOBJECT TRANSFER COMPLETE")
    print(
        "Final cube position:",
        get_cube_pose(cube)[0]
    )


# ============================================================
# MAIN LOOP
# ============================================================

p.resetDebugVisualizerCamera(
    cameraDistance=3.2,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0, 0, 0.45]
)

completed = 0

try:

    while MAX_OBJECTS is None or completed < MAX_OBJECTS:

        print("\n================================")
        print(f"OBJECT {completed + 1}")
        print("================================")

        position = random_pick_position()
        cube = create_cube(position)

        simulate(30)

        try:

            transfer_object(cube)
            completed += 1

            p.removeBody(cube)

        except RuntimeError as error:

            print("\nTRANSFER FAILED:", error)
            break

    print(
        "\nFinished processing:",
        completed,
        "objects"
    )

    print("\nPERFORMANCE SUMMARY")

    for label, seconds in TIMINGS.items():
        print(f"{label}: {seconds:.3f} s total")

except RuntimeError as error:

    print("\nSIMULATION STOPPED:", error)

except KeyboardInterrupt:

    print("\nSimulation interrupted")

finally:

    print(
        "\nSimulation remains open. "
        "Press Ctrl+C to exit."
    )

    try:
        while p.isConnected(physics_client):
            step()
    except KeyboardInterrupt:
        pass

    if p.isConnected(physics_client):
        p.disconnect(physics_client)

    if p.isConnected(ik_client):
        p.disconnect(ik_client)
