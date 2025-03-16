import argparse
import torch
from stable_baselines3 import PPO
import os


def perturb_model(model_path, perturb_factor):
    """
    Load a model, perturb its weights, and save it with a modified name.

    :param model_path: Path to the model file.
    :param perturb_factor: The standard deviation of the Gaussian noise for perturbation.
    """
    # Load the model
    model = PPO.load(model_path)

    # Perturb the model's policy weights
    for param in model.policy.parameters():
        param.data += torch.randn_like(param) * perturb_factor  # Add Gaussian noise

    # Generate the new file name
    base_dir, base_name = os.path.split(model_path)
    base_name, ext = os.path.splitext(base_name)
    perturb_tag = f"_perturb{abs(perturb_factor):.2f}"
    perturbed_name = os.path.join(base_dir, f"{base_name}{perturb_tag}{ext}")

    # Save the perturbed model
    model.save(perturbed_name)
    print(f"Perturbed model saved as: {perturbed_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perturb a Stable-Baselines model and save it."
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the model file (e.g., 'folder/model.zip').",
    )
    parser.add_argument(
        "--factor", type=float, required=True, help="Perturbation factor (e.g., -0.02)."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.model_path):
        print(f"Error: The file '{args.model_path}' does not exist.")
    else:
        perturb_model(args.model_path, args.factor)
