import torch
from torch import nn

# Neural Network using DDQN algorithm
class MarioNet(nn.Module):
    # input -> (conv2d + relu) x 3 -> flatten -> (dense + relu) x 2 -> output
    def __init__(self, input_dim, output_dim):
        super().__init__()
        c, h, w = input_dim

        # Validate height and width
        if h <= 0 or w <= 0:
            raise ValueError(f"Height and Width must be > 0. Got: {h}x{w}")

        self.online = self._build_cnn(c, h, w, output_dim)

        self.target = self._build_cnn(c, h, w, output_dim)
        self.target.load_state_dict(self.online.state_dict())

        # Q target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)
        
    def _get_conv_output_shape(self, shape, conv_block):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            dummy_output = conv_block(dummy_input)
            return dummy_output.numel()

    def _build_cnn(self, c, h, w, output_dim):
        # Convolution feature layers
        feature_layer = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Input features for decision layer
        num_flatten_features = self._get_conv_output_shape((c, h, w), feature_layer)
        
        # Fully connected decision layers
        decision_layer = nn.Sequential(
            nn.Linear(num_flatten_features, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

        return nn.Sequential(
            feature_layer,
            decision_layer
        )