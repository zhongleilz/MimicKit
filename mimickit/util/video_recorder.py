from __future__ import annotations

import numpy as np
import os
import tempfile
from typing import TYPE_CHECKING, Any
import wandb
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from util.logger import Logger

if TYPE_CHECKING:
    import engines.engine as engine
    from omni.replicator.core import Annotator, RenderProduct


class VideoRecorder:
    """Records video frames from the simulation and uploads to WandB.
    
    Works with Isaac Lab engine in headless mode using the omni.replicator
    annotator API to capture viewport images.

    The recorder manages its own camera controls, independent of the environment's
    visualization camera. This allows recording without interfering with visualization.
    
    Args:
        engine: The simulation engine (e.g. IsaacLabEngine).
        camera_config: Dict with optional keys:
            - cam_pos: np.ndarray, camera position [x, y, z]. Defaults to [0, -5, 3].
            - cam_target: np.ndarray, camera target [x, y, z]. Defaults to [0, 0, 0].
        resolution: Tuple (width, height) for the captured frames.
        fps: Frames per second for the output video.
    """

    VIDEO_CAM_PATH = "/OmniverseKit_Persp"  # Use viewport camera (works in headless mode)

    def __init__(self, engine: engine.Engine, camera_config: dict = {},
                 resolution: tuple[int, int] = (640, 480), fps: int = 30) -> None:
        self._engine: engine.Engine = engine
        self._resolution: tuple[int, int] = resolution
        self._fps: int = fps

        # Create dedicated camera prim for video recording
        self._cam_prim_path: str = self._build_camera()

        # Camera control (from camera_config dict)
        self._cam_pos = camera_config.get("cam_pos", np.array([0.0, -5.0, 3.0]))
        self._cam_target = camera_config.get("cam_target", np.array([0.0, 0.0, 0.0]))

        self._recorded_frames: list[np.ndarray] = []
        self._recording: bool = False

        self._annotator: Any | None = None
        self._render_product: Any | None = None

        self._logger_step_tracker: Any | None = None

        return

    def _build_camera(self) -> str:
        """Get ViewportCameraState for the viewport camera (shared with visualization)."""
        # Use engine's camera state (viewport camera works in headless mode)
        self._camera_state = self._engine._camera_state
        Logger.print("[VideoRecorder] Using viewport camera at {}".format(self.VIDEO_CAM_PATH))
        return self.VIDEO_CAM_PATH

    def set_logger_step_tracker(self, logger: Any) -> None:
        """
        A temporary hack to get the step value from the logger.
        """
        self._logger_step_tracker = logger
        return

    def _set_camera_pose(self, pos: np.ndarray, target: np.ndarray) -> None:
        """Set the video camera pose using ViewportCameraState (same pattern as engine.set_camera_pose)."""
        env_offset = self._engine._env_offsets[0].cpu().numpy()
        cam_pos = pos.copy()
        cam_target = target.copy()

        cam_pos[:2] += env_offset
        cam_target[:2] += env_offset
        
        self._camera_state.set_position_world(cam_pos.tolist(), True)
        self._camera_state.set_target_world(cam_target.tolist(), True)
        return

    def _ensure_annotator(self) -> None:
        """Lazily create the render product and RGB annotator."""
        if self._annotator is not None:
            return
        
        import omni.replicator.core as rep

        self._render_product = rep.create.render_product(
            self._cam_prim_path, self._resolution
        )
        self._annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        self._annotator.attach([self._render_product])
        Logger.print("[VideoRecorder] Created RGB annotator for {}".format(self._cam_prim_path))
        return

    def _capture_frame(self) -> None:
        """Capture a single RGB frame from the viewport.
        
        Saves and restores camera state to avoid conflicts with visualization.
        """
        self._ensure_annotator()

        # Save current camera state (in case visualization is active)
        saved_pos = self._camera_state.position_world
        saved_target = self._camera_state.target_world
        
        # Set video camera position
        self._set_camera_pose(self._cam_pos, self._cam_target)
        
        # Render the scene to update the viewport
        self._engine._sim.render()

        rgb_data: Any = self._annotator.get_data()
        if rgb_data is None or rgb_data.size == 0:
            # Renderer still warming up
            frame: np.ndarray = np.zeros((self._resolution[1], self._resolution[0], 3), dtype=np.uint8)
        else:
            frame = np.frombuffer(rgb_data, dtype=np.uint8).reshape(*rgb_data.shape)
            frame = frame[:, :, :3]  # drop alpha channel

        self._recorded_frames.append(frame)
        
        # Restore visualization camera state
        self._camera_state.set_position_world(saved_pos, True)
        self._camera_state.set_target_world(saved_target, True)
        
        return

    def start_recording(self) -> None:
        """Begin a new video recording."""
        if self._recording:
            Logger.print("[VideoRecorder] Already recording, stopping previous recording first")
            self.stop_recording()
        
        self._recorded_frames = []
        self._recording = True
        Logger.print("[VideoRecorder] Started recording")
        return

    def capture_frame(self) -> None:
        """Capture a frame during recording. Call this each step while recording."""
        if self._recording:
            self._capture_frame()
        return

    def stop_recording(self) -> None:
        """Stop recording, create video, upload to WandB, and clean up."""
        if not self._recording:
            return
        
        self._stop_recording()
        return

    def _stop_recording(self) -> None:
        """Stop recording, create video, upload to WandB, and clean up."""
        if not self._recording or len(self._recorded_frames) == 0:
            self._recording = False
            return

        self._recording = False

        try:
            if len(self._recorded_frames) == 0:
                Logger.print("[VideoRecorder] No frames recorded, skipping video creation")
                return

            clip: ImageSequenceClip = ImageSequenceClip(self._recorded_frames, fps=self._fps)

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as tmp:
                temp_path: str = tmp.name
            
                clip.write_videofile(temp_path, logger=None)
                if wandb.run is not None:
                    step_val = self._logger_step_tracker.get_current_step()
                    wandb.log({
                        "video": wandb.Video(temp_path, format="mp4"),
                    }, step=step_val)
                    Logger.print("[VideoRecorder] Uploaded video to WandB ({} frames, step {})".format(
                        len(self._recorded_frames), step_val))
                else:
                    Logger.print("[VideoRecorder] WandB not initialized, skipping upload")
        except ImportError as e:
            Logger.print("[VideoRecorder] Missing dependency: {}. Video not saved.".format(e))
        except Exception as e:
            Logger.print("[VideoRecorder] Error creating video: {}".format(e))

        self._recorded_frames = []
        return
