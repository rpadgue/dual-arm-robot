
import time
import pybullet as p
import pybullet_data
import numpy as np
from scipy.optimize import minimize

#connect to the simulator.
physics_client = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)

# Add a ground plane.
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

# Run the physics simulation.
try:
    step_count = 0

    while p.isConnected():
        p.stepSimulation()
        step_count += 1

        # Measure once per second of simulated time.
        if step_count % 240 == 0:
            # Get actual end-effector position.
            link_state = p.getLinkState(
                robot_a,
                end_effector_index,
                computeForwardKinematics=True
            )

            actual_position = np.array(link_state[4])
            desired_position = np.array(target_position)

            # Calculate Euclidean positioning error.
            error = np.linalg.norm(
                desired_position - actual_position
            )

            print(
                f"End-effector error: {error * 1000:.2f} mm"
            )

            # Measure Robot B's end-effector positioning error.
            link_state_b = p.getLinkState(
                robot_b,
                end_effector_index,
                computeForwardKinematics=True
            )

            actual_position_b = np.array(link_state_b[4])
            desired_position_b = np.array(target_position_b)

            error_b = np.linalg.norm(
                desired_position_b - actual_position_b
            )

            print(f"Robot B end-effector error: {error_b * 1000:.2f} mm")

            # Compare actual joint angles against IK targets.
            for joint_index in range(7):
                actual_angle = p.getJointState(
                    robot_a,
                    joint_index
                )[0]

                target_angle = joint_angles[joint_index]

                print(
                    f"Joint {joint_index}: "
                    f"Target = {target_angle:.3f}, "
                    f"Actual = {actual_angle:.3f}"
    )

        time.sleep(1 / 240)

except KeyboardInterrupt:
    pass

finally:
    if p.isConnected():
        p.disconnect()
