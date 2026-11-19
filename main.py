"""Entry point. Keep Aleks's end-effectors.py and object_interaction.py unchanged."""
import time
import pybullet as p
import pybullet_data
from end_effectors import create_two_finger_gripper, create_suction_gripper
from object_interaction import ObjectInteraction
from arm_controller import ArmController, DT
from motion_sequence import run_initial_movement_test
from conveyor import CircularConveyor


def main():
    p.connect(p.GUI)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(DT)
        p.setRealTimeSimulation(0)
        p.loadURDF("plane.urdf")
        robot_a = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[-0.7, 0, 0], useFixedBase=True)
        robot_b = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0.7, 0, 0],
                             baseOrientation=p.getQuaternionFromEuler([0, 0, 3.14159]),
                             useFixedBase=True)
        gripper_a = create_two_finger_gripper(robot_a)
        suction_b = create_suction_gripper(robot_b)
        for robot_id, tool_id in [(robot_a, gripper_a), (robot_b, suction_b)]:
            for robot_link in range(-1, p.getNumJoints(robot_id)):
                for tool_link in range(-1, p.getNumJoints(tool_id)):
                    p.setCollisionFilterPair(robot_id, tool_id, robot_link, tool_link, 0)

        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05]*3)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05]*3,
                                     rgbaColor=[0.9, 0.3, 0.2, 1])
        cube = p.createMultiBody(baseMass=0.2, baseCollisionShapeIndex=collision,
                                 baseVisualShapeIndex=visual, basePosition=[0, 0, 0.05])
        interaction = ObjectInteraction(robot_a, robot_b, cube)


        # Create two circular conveyor belts.
        # Belt A is left of Robot A.
        # Belt B is right of Robot B.

        conveyor_a = CircularConveyor(
            center=[-1.65, 0.0],
            radius_x=0.45,
            radius_y=0.75,
            angular_speed=0.25,
            name="Conveyor A"
        )

        conveyor_b = CircularConveyor(
            center=[1.65, 0.0],
            radius_x=0.45,
            radius_y=0.75,
            angular_speed=0.25,
            name="Conveyor B"
        )

        # Belt A should contain squares.
        # Initially, it contains a mixture.
        conveyor_a.add_object("square", angle=0.0)
        conveyor_a.add_object("circle", angle=1.5)
        conveyor_a.add_object("square", angle=3.0)
        conveyor_a.add_object("circle", angle=4.5)

        # Belt B should contain circles.
        # Initially, it also contains a mixture.
        conveyor_b.add_object("circle", angle=0.0)
        conveyor_b.add_object("square", angle=1.5)
        conveyor_b.add_object("circle", angle=3.0)
        conveyor_b.add_object("square", angle=4.5)


        # Keep a reference to the interaction controller for later handoff integration.
        _ = interaction
        p.resetDebugVisualizerCamera(
            cameraDistance=5.0,
            cameraYaw=45,
            cameraPitch=-70,
            cameraTargetPosition=[0, 0, 0.3]
        )
        arm_a = ArmController(robot_a, "Robot A")
        arm_b = ArmController(robot_b, "Robot B")
        print("Robot A joints:", p.getNumJoints(robot_a))
        print("Robot B joints:", p.getNumJoints(robot_b))
        print("Initial movement test succeeded:", run_initial_movement_test(arm_a, arm_b))

        # Run the simulation as fast as possible.
        while p.isConnected():

            # Keep both arms at their current positions.
            arm_a.command(arm_a.angles())
            arm_b.command(arm_b.angles())

            # Rotate objects around the conveyors.
            conveyor_a.update(DT)
            conveyor_b.update(DT)

            # Advance physics.
            p.stepSimulation()


            # No sleep — run at maximum simulation speed.
    except KeyboardInterrupt:
        print("Simulation stopped.")
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()