"""Orchestration. First verify the original movement before adding grasping."""
from arm_controller import move_both
from robot_poses import HOME_JOINTS, TEST_TARGET_A, TEST_TARGET_B


def run_initial_movement_test(arm_a, arm_b):
    arm_a.reset(HOME_JOINTS)
    arm_b.reset(HOME_JOINTS)
    goal_a = arm_a.solve_ik(TEST_TARGET_A, refine=True)
    goal_b = arm_b.solve_ik(TEST_TARGET_B)
    print("Robot A target joints:", goal_a)
    print("Robot B target joints:", goal_b)
    return move_both(arm_a, goal_a, arm_b, goal_b, duration=0.01, timeout=5.0)


# Next milestone (not implemented yet):
# A: HOME -> PRE_PICK -> PICK -> (grasp confirmation) -> LIFT -> HANDOFF_A
# B: HOME -> PRE_HANDOFF_B -> HANDOFF_B -> (grasp confirmation)
# A: (release confirmation) -> RETREAT_A
# B: ROTATE -> PRE_PLACE -> PLACE -> (release confirmation) -> RETREAT_B
# Both: HOME