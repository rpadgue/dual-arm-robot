
import pybullet as p

# KUKA iiwa final link index
END_LINK = 6


def attach_tool(robot_id, tool_id):
    """
    Attach an end effector to the final link
    of a KUKA iiwa robot.
    """

    # Get the final link's current world pose.
    link_state = p.getLinkState(
        robot_id,
        END_LINK,
        computeForwardKinematics=True
    )

    link_position = link_state[4]
    link_orientation = link_state[5]

    # Position the tool at the robot's wrist.
    p.resetBasePositionAndOrientation(
        tool_id,
        link_position,
        link_orientation
    )

    # Create fixed attachment.
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=END_LINK,
        childBodyUniqueId=tool_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0]
    )

    return constraint_id


def create_two_finger_gripper(robot_id):
    """
    Create a simple two-finger gripper
    for Robot A.
    """

    # Gripper body dimensions
    body_half = [0.045, 0.035, 0.025]

    # Finger dimensions
    finger_half = [0.008, 0.008, 0.045]

    # Gripper body
    body_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=body_half
    )

    body_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=body_half,
        rgbaColor=[0.15, 0.35, 0.85, 1]
    )

    # Two fingers
    finger_collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=finger_half
    )

    finger_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=finger_half,
        rgbaColor=[0.9, 0.75, 0.15, 1]
    )

    # Create gripper with two fixed fingers.
    gripper_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=body_collision,
        baseVisualShapeIndex=body_visual,
        basePosition=[0, 0, 1.5],

        linkMasses=[0.02, 0.02],

        linkCollisionShapeIndices=[
            finger_collision,
            finger_collision
        ],

        linkVisualShapeIndices=[
            finger_visual,
            finger_visual
        ],

        linkPositions=[
            [0, -0.045, 0.065],
            [0,  0.045, 0.065]
        ],

        linkOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],

        linkInertialFramePositions=[
            [0, 0, 0],
            [0, 0, 0]
        ],

        linkInertialFrameOrientations=[
            [0, 0, 0, 1],
            [0, 0, 0, 1]
        ],

        linkParentIndices=[0, 0],

        linkJointTypes=[
            p.JOINT_FIXED,
            p.JOINT_FIXED
        ],

        linkJointAxis=[
            [0, 0, 1],
            [0, 0, 1]
        ]
    )

    attach_tool(robot_id, gripper_id)

    print("Two-finger gripper attached to Robot A.")

    return gripper_id


def create_suction_gripper(robot_id):
    """
    Create a suction gripper for Robot B.
    """

    # Suction tool body
    body_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=0.025,
        height=0.08
    )

    body_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.025,
        length=0.08,
        rgbaColor=[0.2, 0.7, 0.3, 1]
    )

    # Suction cup
    cup_collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=0.04,
        height=0.015
    )

    cup_visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=0.04,
        length=0.015,
        rgbaColor=[0.1, 0.1, 0.1, 1]
    )

    suction_id = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=body_collision,
        baseVisualShapeIndex=body_visual,
        basePosition=[0, 0, 1.5],

        linkMasses=[0.02],

        linkCollisionShapeIndices=[
            cup_collision
        ],

        linkVisualShapeIndices=[
            cup_visual
        ],

        linkPositions=[
            [0, 0, 0.055]
        ],

        linkOrientations=[
            [0, 0, 0, 1]
        ],

        linkInertialFramePositions=[
            [0, 0, 0]
        ],

        linkInertialFrameOrientations=[
            [0, 0, 0, 1]
        ],

        linkParentIndices=[0],

        linkJointTypes=[
            p.JOINT_FIXED
        ],

        linkJointAxis=[
            [0, 0, 1]
        ]
    )

    attach_tool(robot_id, suction_id)

    print("Suction gripper attached to Robot B.")

    return suction_id
