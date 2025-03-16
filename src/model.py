import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for stacked frame input.
    """

    def __init__(self, observation_space, features_dim: int = 256, num_frames: int = 4):
        # Get the shape of the input from the observation space
        input_channels = num_frames * 3  # Assuming RGB images
        super(CustomCNN, self).__init__(observation_space, features_dim)

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the size of the output from the CNN
        with torch.no_grad():
            dummy_input = torch.zeros(
                1,
                input_channels,
                observation_space.shape[1],
                observation_space.shape[2],
            )
            n_flatten = self.cnn(dummy_input).shape[1]

        # Linear layer to map the CNN output to the desired feature size
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

    def get_layer_features(self, x):
        """
        Extract intermediate feature maps from the CNN.

        Args:
            x (torch.Tensor): Input tensor to the CNN.

        Returns:
            dict: A dictionary containing feature maps at different layers.
        """
        features = {}

        # Pass through the first Conv2d layer
        x = self.cnn[0](x)  # Conv2d(32, ...)
        features["layer1"] = x.clone()  # Save feature map
        # print(f"Layer1 output shape: {x.shape}")  # Optional debug print

        # Apply ReLU activation
        x = self.cnn[1](x)  # ReLU

        # Pass through the second Conv2d layer
        x = self.cnn[2](x)  # Conv2d(64, ...)
        features["layer2"] = x.clone()  # Save feature map
        # print(f"Layer2 output shape: {x.shape}")  # Optional debug print

        # Apply ReLU activation
        x = self.cnn[3](x)  # ReLU

        # Pass through the third Conv2d layer
        x = self.cnn[4](x)  # Conv2d(128, ...)
        features["layer3"] = x.clone()  # Save feature map
        # print(f"Layer3 output shape: {x.shape}")  # Optional debug print

        # Apply ReLU activation
        x = self.cnn[5](x)  # ReLU

        # Return all saved feature maps
        return features
