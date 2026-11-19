
import pybullet as p


class ObjectInteraction:
    def __init__(self, robot_a, robot_b, object_id):
        self.robot_a = robot_a
        self.robot_b = robot_b
        self.object_id = object_id

        self.end_effector_link = 6

        self.grasp_constraint = None
        self.object_owner = None

        # Maximum permitted grasp distance (meters).
        self.grasp_tolerance = 0.15

    def get_distance(self, robot_id):
        """Distance from robot wrist to object center."""
        link_state = p.getLinkState(
            robot_id,
            self.end_effector_link,
            computeForwardKinematics=True
        )

        ee_position = link_state[4]
        object_position, _ = p.getBasePositionAndOrientation(
            self.object_id
        )

        return sum(
            (a - b) ** 2
            for a, b in zip(ee_position, object_position)
        ) ** 0.5

    def attach_object(self, robot_id):
        """Attach object while preserving its world pose."""
        if self.grasp_constraint is not None:
            print("Object is already attached.")
            return False

        distance = self.get_distance(robot_id)

        if distance > self.grasp_tolerance:
            print(
                f"Grasp failed: distance {distance:.3f} m"
            )
            return False

        link_state = p.getLinkState(
            robot_id,
            self.end_effector_link,
            computeForwardKinematics=True
        )

        ee_position = link_state[4]
        ee_orientation = link_state[5]

        obj_position, obj_orientation = (
            p.getBasePositionAndOrientation(self.object_id)
        )

        # Express the object's current pose in the
        # end-effector link coordinate frame.
        inverse_position, inverse_orientation = (
            p.invertTransform(
                ee_position,
                ee_orientation
            )
        )

        relative_position, relative_orientation = (
            p.multiplyTransforms(
                inverse_position,
                inverse_orientation,
                obj_position,
                obj_orientation
            )
        )

        self.grasp_constraint = p.createConstraint(
            parentBodyUniqueId=robot_id,
            parentLinkIndex=self.end_effector_link,
            childBodyUniqueId=self.object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=relative_position,
            parentFrameOrientation=relative_orientation,
            childFramePosition=[0, 0, 0],
            childFrameOrientation=[0, 0, 0, 1]
        )

        self.object_owner = robot_id

        print("Object attached successfully.")
        return True

    def robot_a_grasp(self):
        """Robot A picks up the object."""
        print("Robot A attempting grasp...")
        return self.attach_object(self.robot_a)

    def robot_b_grasp(self):
        """Transfer the object from A to B."""
        if self.object_owner != self.robot_a:
            print("Handoff failed: A does not hold object.")
            return False

        if self.get_distance(self.robot_b) > (
            self.grasp_tolerance
        ):
            print("Handoff failed: B is too far away.")
            return False

        # Save old ownership.
        old_constraint = self.grasp_constraint

        # Attach B before releasing A.
        self.grasp_constraint = None
        self.object_owner = None

        success = self.attach_object(self.robot_b)

        if success:
            p.removeConstraint(old_constraint)
            print("Handoff successful: A -> B.")
            return True

        # Restore previous ownership if B fails.
        self.grasp_constraint = old_constraint
        self.object_owner = self.robot_a
        return False

    def release_object(self, robot_id):
        """Release object from the owning robot."""
        if self.object_owner != robot_id:
            print("Release failed: robot does not own object.")
            return False

        p.removeConstraint(self.grasp_constraint)

        self.grasp_constraint = None
        self.object_owner = None

        print("Object released.")
        return True
