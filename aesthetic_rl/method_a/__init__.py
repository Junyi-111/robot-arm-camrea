"""ArtiMuse feature-augmented SAC for the PiPER wrist camera.

The package is intentionally isolated from the historical experiments in this
workspace.  Importing it never starts ROS, opens a camera, contacts a remote
service, or enables the robot.
"""

from .config import MethodAConfig, load_config

__all__ = ["MethodAConfig", "load_config"]
