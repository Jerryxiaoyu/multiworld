import pybullet
import numpy as np

from multiworld.math import Pose

def plot_pose(
              pose,
              axis_length=1.0,
              text=None,
              text_size=1.0,
              text_color=[0, 0, 0]):
    """Plot a 6-DoF pose or a frame in the debugging visualizer.

    Args:
        pose: The pose to be plot.
        axis_length: The length of the axes.
        text: Text showing up next to the frame.
        text_size: Size of the text.
        text_color: Color of the text.
    """
    if not isinstance(pose, Pose):
        pose = Pose(pose)

    origin = pose.position
    x_end = origin + np.dot([axis_length, 0, 0], pose.matrix3.T)
    y_end = origin + np.dot([0, axis_length, 0], pose.matrix3.T)
    z_end = origin + np.dot([0, 0, axis_length], pose.matrix3.T)

    pybullet.addUserDebugLine(
        origin,
        x_end,
        lineColorRGB=[1, 0, 0],
        lineWidth=2)

    pybullet.addUserDebugLine(
        origin,
        y_end,
        lineColorRGB=[0, 1, 0],
        lineWidth=2)

    pybullet.addUserDebugLine(
        origin,
        z_end,
        lineColorRGB=[0, 0, 1],
        lineWidth=2)

    if text is not None:
        pybullet.addUserDebugText(
            text,
            origin,
            text_color,
            text_size)

def plot_line(
              start,
              end,
              line_color=[0, 0, 0],
              line_width=1):
    """Plot a pose or a frame in the debugging visualizer.

    Args:
        start: Starting point of the line.
        end: Ending point of the line.
        line_color: Color of the line.
        line_width: Width of the line.
    """
    pybullet.addUserDebugLine(
        start,
        end,
        lineColorRGB=line_color,
        lineWidth=line_width)

def clear_visualization():
    """Clear all visualization items."""
    pybullet.removeAllUserDebugItems()