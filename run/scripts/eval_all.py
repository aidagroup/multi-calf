from src import run_path
import subprocess
import concurrent.futures
from pathlib import Path
from loguru import logger
import os

checkpoints_path = (
    run_path / "artifacts" / "ppo_VisualPendulumClassicReward_10" / "checkpoints"
)


def evaluate_checkpoint(checkpoint_path: Path):
    """Evaluate a single checkpoint."""
    logger.info(f"Evaluating checkpoint: {checkpoint_path}")
    output_path = checkpoints_path.parent / "videos" / checkpoint_path.stem

    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "run/eval.py",
                "--checkpoint_path",
                str(checkpoint_path),
                "--output_path",
                str(output_path),
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.success(f"Successfully evaluated {checkpoint_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to evaluate {checkpoint_path.name}: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def main():
    # Get all checkpoint files
    checkpoint_files = sorted(list(checkpoints_path.glob("*.zip")))
    logger.info(f"Found {len(checkpoint_files)} checkpoints to evaluate")

    # Determine the number of workers based on CPU count
    max_workers = min(2, len(checkpoint_files))
    logger.info(f"Using {max_workers} workers for parallel evaluation")

    # Run evaluations in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(evaluate_checkpoint, checkpoint_files))

    # Report results
    successful = sum(results)
    logger.info(f"Evaluation complete: {successful}/{len(checkpoint_files)} successful")


if __name__ == "__main__":
    main()
