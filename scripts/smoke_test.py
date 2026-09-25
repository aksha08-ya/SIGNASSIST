"""Smoke test script for SIGNASSIST Phase 1 infrastructure."""

import sys
from pathlib import Path

# Ensure repository root is in sys.path when executed directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import ConfigManager
from src.utils.logger import setup_logger


def main():
    logger = setup_logger(name="signassist_smoke")
    config_mgr = ConfigManager()
    cfg = config_mgr.load_all()

    logger.info("=== SIGNASSIST Infrastructure Smoke Test ===")
    logger.info(
        f"System: {cfg.system.name} v{cfg.system.version} "
        f"(env: {cfg.system.environment}, requested_device: {cfg.system.requested_device}, resolved_device: {cfg.system.device})"
    )
    logger.info(f"Target FPS: {cfg.video.target_fps}")
    logger.info(
        f"Perception Space: {cfg.model.dynamic_model.raw_landmarks} raw MediaPipe Holistic landmarks"
    )
    logger.info(
        f"Recognition Graph: {cfg.model.dynamic_model.num_nodes} nodes (21 LH + 21 RH + 33 Pose + 40 Face) "
        f"x {cfg.model.dynamic_model.landmark_channels} channels (x, y, z, vx, vy)"
    )
    logger.info(
        f"Static Model: {cfg.model.static_model.name} "
        f"(input_dim: {cfg.model.static_model.input_dim} = {cfg.model.static_model.num_nodes} nodes x {cfg.model.static_model.spatial_channels} spatial coords)"
    )
    logger.info(
        f"Dynamic Model: {cfg.model.dynamic_model.name} (sequence_length: {cfg.model.dynamic_model.sequence_length})"
    )
    logger.info(f"Confidence Threshold: {cfg.model.inference.confidence_threshold}")
    logger.info("Smoke test completed successfully!")


if __name__ == "__main__":
    main()
