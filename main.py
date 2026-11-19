
import time
import pybullet as p
import pybullet_data
import numpy as np
from scipy.optimize import minimize
from object_interaction import ObjectInteraction

from end_effectors import (
    create_two_finger_gripper,
    create_suction_gripper
)

from object_interaction import ObjectInteraction

#connect to the simulator.
physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 480)

# Add a ground plane
ground = p.loadURDF("plane.urdf")

# Load two robots facing each other.
robot_a = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[-0.7, 0, 0],
    useFixedBase=True
)

robot_b = p.loadURDF(
    "kuka_iiwa/model.urdf",
    basePosition=[0.7, 0, 0],
    baseOrientation=p.getQuaternionFromEuler([0, 0, 3.14159]),
    useFixedBase=True
)
# NEW: Attach end effectors to both robots.

# Robot A gets the two-finger gripper.
gripper_a = create_two_finger_gripper(robot_a)

# Robot B gets the suction gripper.
suction_b = create_suction_gripper(robot_b)

# Disable collisions between each robot and its own tool.
for robot_id, tool_id in [
    (robot_a, gripper_a),
    (robot_b, suction_b)
]:
    for robot_link in range(-1, p.getNumJoints(robot_id)):
        for tool_link in range(-1, p.getNumJoints(tool_id)):
            p.setCollisionFilterPair(
                robot_id,
                tool_id,
                robot_link,
                tool_link,
                enableCollision=0
            )
            
# Create a small test object.
collision_shape = p.createCollisionShape(
    p.GEOM_BOX,
    halfExtents=[0.05, 0.05, 0.05]
)

visual_shape = p.createVisualShape(
    p.GEOM_BOX,
    halfExtents=[0.05, 0.05, 0.05],
    rgbaColor=[0.9, 0.3, 0.2, 1]
)

test_object = p.createMultiBody(
    baseMass=0.2,
    baseCollisionShapeIndex=collision_shape,
    baseVisualShapeIndex=visual_shape,
    basePosition=[0, 0, 0.05]
)
# Initialize the object interaction controller.
interaction = ObjectInteraction(
    robot_a,
    robot_b,
    test_object
)
# Position the camera.
p.resetDebugVisualizerCamera(
    cameraDistance=2.5,
    cameraYaw=45,
    cameraPitch=-35,
    cameraTargetPosition=[0, 0, 0.4]
)

print("Simulation started successfully.")
print("Robot A joints:", p.getNumJoints(robot_a))
print("Robot B joints:", p.getNumJoints(robot_b))


# Inspect the joint limits of Robot A.
print("\nRobot A joint limits:")

for joint_index in range(p.getNumJoints(robot_a)):
    joint_info = p.getJointInfo(robot_a, joint_index)

    joint_name = joint_info[1].decode("utf-8")
    lower_limit = joint_info[8]
    upper_limit = joint_info[9]

    print(
        f"Joint {joint_index} ({joint_name}): "
        f"[{lower_limit:.3f}, {upper_limit:.3f}] rad"
    )




# Move Robot A using inverse kinematics.

# KUKA iiwa's final link is the end effector.
end_effector_index = 6

# Desired end-effector position in world coordinates (meters).
target_position = [-0.45, 0.0, 0.65]

# Joint limits from the robot's URDF.
lower_limits = []
upper_limits = []

for joint_index in range(7):
    info = p.getJointInfo(robot_a, joint_index)
    lower_limits.append(info[8])
    upper_limits.append(info[9])

# Joint ranges and preferred resting configuration.
joint_ranges = [
    upper - lower
    for lower, upper in zip(lower_limits, upper_limits)
]

rest_poses = [0.0, 0.3, 0.0, -1.2, 0.0, 0.5, 0.0]

# Calculate IK using joint-limit information.
joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=robot_a,
    endEffectorLinkIndex=end_effector_index,
    targetPosition=target_position,
    lowerLimits=lower_limits,
    upperLimits=upper_limits,
    jointRanges=joint_ranges,
    restPoses=rest_poses,
    maxNumIterations=1000,
    residualThreshold=1e-6
)

print("Calculated IK joint angles:", joint_angles)


# STEP 1: Calculate initial IK solution.
joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=robot_a,
    endEffectorLinkIndex=end_effector_index,
    targetPosition=target_position,
    lowerLimits=lower_limits,
    upperLimits=upper_limits,
    jointRanges=joint_ranges,
    restPoses=rest_poses,
    maxNumIterations=1000,
    residualThreshold=1e-6
)

print("Calculated IK joint angles:", joint_angles)


# STEP 2: Refine the IK solution with joint limits.
initial_angles = np.array(joint_angles[:7])
joint_bounds = list(zip(lower_limits, upper_limits))

def position_error_squared(angles):
    # Temporarily set candidate joint angles.
    for i, angle in enumerate(angles):
        p.resetJointState(robot_a, i, float(angle))

    # Calculate forward kinematics.
    link_state = p.getLinkState(
        robot_a,
        end_effector_index,
        computeForwardKinematics=True
    )

    actual_position = np.array(link_state[4])
    desired_position = np.array(target_position)

    return np.sum((desired_position - actual_position) ** 2)

result = minimize(
    position_error_squared,
    initial_angles,
    method="SLSQP",
    bounds=joint_bounds,
    options={
        "maxiter": 500,
        "ftol": 1e-12
    }
)

joint_angles = result.x

print("Optimization successful:", result.success)
print("Refined joint angles:", joint_angles)
print(
    "Predicted positioning error:",
    np.sqrt(result.fun) * 1000,
    "mm"
)


# STEP 3: Command Robot A to the refined joint angles.
for joint_index in range(7):
    p.setJointMotorControl2(
        bodyUniqueId=robot_a,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=float(joint_angles[joint_index]),
        force=200
    )

    # ROBOT B: Calculate inverse kinematics.

target_position_b = [0.45, 0.0, 0.65]

joint_angles_b = p.calculateInverseKinematics(
    bodyUniqueId=robot_b,
    endEffectorLinkIndex=end_effector_index,
    targetPosition=target_position_b,
    lowerLimits=lower_limits,
    upperLimits=upper_limits,
    jointRanges=joint_ranges,
    restPoses=rest_poses,
    maxNumIterations=1000,
    residualThreshold=1e-6
)

print("Robot B IK angles:", joint_angles_b)

# Command Robot B to move.
for joint_index in range(7):
    p.setJointMotorControl2(
        bodyUniqueId=robot_b,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=joint_angles_b[joint_index],
        force=200
    )

# Draw Robot B's target marker.
p.addUserDebugLine(
    lineFromXYZ=target_position_b,
    lineToXYZ=[
        target_position_b[0],
        target_position_b[1],
        target_position_b[2] + 0.15
    ],
    lineColorRGB=[0, 1, 0],
    lineWidth=3
)

# STEP 4: Draw the target marker.
p.addUserDebugLine(
    lineFromXYZ=target_position,
    lineToXYZ=[
        target_position[0],
        target_position[1],
        target_position[2] + 0.15
    ],
    lineColorRGB=[1, 0, 0],
    lineWidth=3
)


# Command Robot A's joints to the calculated angles.
for joint_index in range(7):
    p.setJointMotorControl2(
        bodyUniqueId=robot_a,
        jointIndex=joint_index,
        controlMode=p.POSITION_CONTROL,
        targetPosition=joint_angles[joint_index],
        force=200
    )

# Draw a visual marker at the target location.
p.addUserDebugLine(
    lineFromXYZ=target_position,
    lineToXYZ=[
        target_position[0],
        target_position[1],
        target_position[2] + 0.15
    ],
    lineColorRGB=[1, 0, 0],
    lineWidth=3
)

#--------------------------------

#Reset both arms to known starting configurations.
start_angles_a = [0.0] * 7
start_angles_b = [0.0] * 7

#Store the target configurations for synchronized movement.
goal_angles_a = np.array(joint_angles[:7], dtype=float)
goal_angles_b = np.array(joint_angles_b[:7], dtype=float)

#Convert starting configurations to NumPy arrays.
start_angles_a = np.array(start_angles_a, dtype=float)
start_angles_b = np.array(start_angles_b, dtype=float)

#Set the trajectory duration.
movement_duration = 1.0

for joint_index in range(7):
    p.resetJointState(
        robot_a,
        joint_index,
        start_angles_a[joint_index]
    )

    p.resetJointState(
        robot_b,
        joint_index,
        start_angles_b[joint_index]
    )

print("Both arms reset to starting configurations.")

# Run the physics simulation.

# Run the physics simulation.

SIMULATION_HZ = 240
SIMULATION_DT = 1.0 / SIMULATION_HZ

p.setTimeStep(SIMULATION_DT)
p.setRealTimeSimulation(0)

try:
    step_count = 0

    while p.isConnected():

        # Calculate elapsed simulation time.
        elapsed_time = step_count * SIMULATION_DT

        # Normalized trajectory progress.
        progress = min(
            elapsed_time / movement_duration,
            1.0
        )

        # Smooth acceleration and deceleration.
        smooth_progress = (
            3 * progress**2 - 2 * progress**3
        )

        # Calculate joint targets.
        current_angles_a = (
            start_angles_a
            + smooth_progress
            * (goal_angles_a - start_angles_a)
        )

        current_angles_b = (
            start_angles_b
            + smooth_progress
            * (goal_angles_b - start_angles_b)
        )

        # Command BOTH robots.
        for joint_index in range(7):

            p.setJointMotorControl2(
                bodyUniqueId=robot_a,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(
                    current_angles_a[joint_index]
                ),
                force=200,
                maxVelocity=1.5
            )

            p.setJointMotorControl2(
                bodyUniqueId=robot_b,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=float(
                    current_angles_b[joint_index]
                ),
                force=200,
                maxVelocity=1.5
            )

        # Advance physics every iteration.
        p.stepSimulation()
        step_count += 1

        # Print status once per simulated second.
        if step_count % SIMULATION_HZ == 0:

            actual_a = np.array([
                p.getJointState(robot_a, i)[0]
                for i in range(7)
            ])

            actual_b = np.array([
                p.getJointState(robot_b, i)[0]
                for i in range(7)
            ])

            error_a = np.linalg.norm(
                actual_a - goal_angles_a
            )

            error_b = np.linalg.norm(
                actual_b - goal_angles_b
            )

            print(
                f"Time: {elapsed_time:.1f}s | "
                f"Progress: {progress:.2f} | "
                f"Robot A joint error: {error_a:.4f} | "
                f"Robot B joint error: {error_b:.4f}"
            )

        # Allow GUI to update without excessive delay.
        #time.sleep(0.001)

except KeyboardInterrupt:
    print("Simulation stopped by user.")

finally:
    if p.isConnected():
        p.disconnect()
