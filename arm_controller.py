"""Ramon's joint control, IK, and trajectory utilities."""
import time
import numpy as np
import pybullet as p
from scipy.optimize import minimize

END_EFFECTOR_LINK = 6
SIMULATION_HZ = 240
DT = 1.0 / SIMULATION_HZ


class ArmController:
    def __init__(self, robot_id, name):
        self.robot_id = robot_id
        self.name = name
        self.joints = list(range(7))
        info = [p.getJointInfo(robot_id, j) for j in self.joints]
        self.lower = np.array([item[8] for item in info])
        self.upper = np.array([item[9] for item in info])
        self.ranges = self.upper - self.lower

    def angles(self):
        return np.array([p.getJointState(self.robot_id, j)[0] for j in self.joints])

    def reset(self, angles):
        for j, angle in zip(self.joints, angles):
            p.resetJointState(self.robot_id, j, float(angle))

    def command(self, angles):
        for j, angle in zip(self.joints, angles):
            p.setJointMotorControl2(self.robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=float(angle), force=200,
                                    maxVelocity=10.0)

    def solve_ik(self, position, refine=False):
        """Position-only IK for the initial test; orientation comes later."""
        original = self.angles()
        guess = p.calculateInverseKinematics(
            self.robot_id, END_EFFECTOR_LINK, position,
            lowerLimits=self.lower.tolist(),
            upperLimits=self.upper.tolist(),
            jointRanges=self.ranges.tolist(),
            restPoses=original.tolist(),
            maxNumIterations=1000, residualThreshold=1e-6)
        candidate = np.clip(np.asarray(guess[:7]), self.lower, self.upper)
        if not refine:
            return candidate

        def objective(q):
            self.reset(q)
            actual = np.asarray(p.getLinkState(self.robot_id, END_EFFECTOR_LINK,
                                                computeForwardKinematics=True)[4])
            return float(np.sum((actual - np.asarray(position)) ** 2))

        try:
            result = minimize(objective, candidate, method="SLSQP",
                              bounds=list(zip(self.lower, self.upper)),
                              options={"maxiter": 500, "ftol": 1e-12})
            if not result.success:
                print(f"{self.name}: IK refinement warning: {result.message}")
            return np.clip(result.x if result.success else candidate,
                           self.lower, self.upper)
        finally:
            # Optimization evaluates poses by resetting joints; always restore.
            self.reset(original)


def move_both(arm_a, goal_a, arm_b, goal_b, duration=5.0, timeout=8.0):
    """Smooth joint-space targets; wait for convergence after interpolation."""
    start_a, start_b = arm_a.angles(), arm_b.angles()
    goal_a, goal_b = np.asarray(goal_a), np.asarray(goal_b)
    if duration <= 0 or timeout <= duration:
        raise ValueError("Require 0 < duration < timeout")
    steps = int(timeout * SIMULATION_HZ)
    for step in range(steps):
        t = step * DT
        progress = min(t / duration, 1.0)
        smooth = 3 * progress**2 - 2 * progress**3
        arm_a.command(start_a + smooth * (goal_a - start_a))
        arm_b.command(start_b + smooth * (goal_b - start_b))
        p.stepSimulation()
        if step % SIMULATION_HZ == 0:
            print(f"t={t:.1f}s  A error={np.linalg.norm(arm_a.angles()-goal_a):.4f} rad  "
                  f"B error={np.linalg.norm(arm_b.angles()-goal_b):.4f} rad")
        if t >= duration and max(np.max(np.abs(arm_a.angles()-goal_a)),
                                 np.max(np.abs(arm_b.angles()-goal_b))) < 0.02:
            return True
        time.sleep(DT)
    print("WARNING: trajectory timed out before reaching joint tolerance")
    return False