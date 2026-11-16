"""Reading-gaze detection for OpenMeet's answer-integrity check.

Every gaze source (an eye tracker in a public dataset, or MediaPipe running on
a candidate's video tile) is converted to a `GazeTrace`. Traces are cut into
windows, each window becomes a handful of scale-free features, and a detector
turns those into a reading score. See README.md for the plan.
"""

__version__ = "0.1.0"
