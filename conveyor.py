
import math
import pybullet as p


class CircularConveyor:
    def __init__(
        self,
        center,
        radius_x=0.45,
        radius_y=0.75,
        angular_speed=0.25,
        height=0.12,
        color=(0.25, 0.25, 0.30, 1),
        name="Conveyor"
    ):
        self.center = center
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.angular_speed = angular_speed
        self.height = height
        self.name = name
        self.objects = []

        # Build a visible oval conveyor track.
        segments = 96
        track_width = 0.14
        track_thickness = 0.04

        for i in range(segments):
            angle = 2 * math.pi * i / segments

            x = center[0] + radius_x * math.cos(angle)
            y = center[1] + radius_y * math.sin(angle)

            # Tangent direction of the ellipse.
            dx = -radius_x * math.sin(angle)
            dy = radius_y * math.cos(angle)

            orientation_angle = math.atan2(dy, dx)

            orientation = p.getQuaternionFromEuler([
                0, 0, orientation_angle - math.pi / 2
            ])

            # Approximate length of this segment.
            segment_length = (
                math.hypot(dx, dy) * 2 * math.pi / segments
            )

            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[
                    track_width / 2,
                    segment_length * 0.55,
                    track_thickness / 2
                ],
                rgbaColor=color
            )

            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, height],
                baseOrientation=orientation
            )


    def add_object(self, shape, angle, color=None):
        """Add a square or circular object to the conveyor."""

        if color is None:
            color = (
                (0.2, 0.5, 1.0, 1.0)
                if shape == "square"
                else (1.0, 0.6, 0.1, 1.0)
            )

        if shape == "square":
            collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[0.035, 0.035, 0.035]
            )
            visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[0.035, 0.035, 0.035],
                rgbaColor=color
            )

        elif shape == "circle":
            collision = p.createCollisionShape(
                p.GEOM_CYLINDER,
                radius=0.035,
                height=0.07
            )
            visual = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=0.035,
                length=0.07,
                rgbaColor=color
            )

        else:
            raise ValueError("Shape must be square or circle")

        x = self.center[0] + self.radius_x * math.cos(angle)
        y = self.center[1] + self.radius_y * math.sin(angle)

        body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, self.height + 0.055]
        )

        self.objects.append({
            "id": body,
            "shape": shape,
            "angle": angle,
            "attached": False
        })

        return body

    def update(self, dt):
        """Advance objects around the circular conveyor."""

        for obj in self.objects:
            if obj["attached"]:
                continue

            obj["angle"] = (
                obj["angle"] + self.angular_speed * dt
            ) % (2 * math.pi)

            angle = obj["angle"]

            x = self.center[0] + self.radius_x * math.cos(angle)
            y = self.center[1] + self.radius_y * math.sin(angle)

            p.resetBasePositionAndOrientation(
                obj["id"],
                [x, y, self.height + 0.055],
                [0, 0, 0, 1]
            )

    def get_object_position(self, object_id):
        position, _ = p.getBasePositionAndOrientation(
            object_id
        )
        return position

    def detach_object_from_belt(self, object_id):
        """Stop conveyor updates when an arm takes an object."""
        for obj in self.objects:
            if obj["id"] == object_id:
                obj["attached"] = True
                return
        raise ValueError("Object not found on this conveyor")
