"""
Module Mạng Neural CNN (MarioNet & MarioDuelingNet) cho dự án Mario DQN.
Được thiết kế theo chuẩn Python Pro 3.12+ và tối ưu hóa cho CPU.
"""

import torch
from torch import nn


class MarioNet(nn.Module):
    """
    Mạng Neural Tích chập (CNN) trích xuất đặc trưng hình ảnh và dự đoán giá trị Q-value cho từng hành động.
    Kiến trúc Vanilla DQN truyền thống sử dụng lớp kết nối đầy đủ (Fully Connected) duy nhất.
    """

    def __init__(self, input_shape: tuple[int, int, int], n_actions: int) -> None:
        """
        Khởi tạo kiến trúc mạng MarioNet.

        Parameters:
            input_shape (tuple): Kích thước đầu vào (channels, height, width). Mặc định (4, 84, 84).
            n_actions (int): Số lượng hành động đầu ra.
        """
        super().__init__()
        channels, _, _ = input_shape

        # Khối trích xuất đặc trưng hình ảnh (Feature Extractor)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Tính toán kích thước đầu ra của khối Conv2d để đưa vào lớp Fully Connected
        conv_out_size = self._get_conv_out(input_shape)

        # Khối quyết định hành động (Decision Maker / Fully Connected)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

        # Khởi tạo trọng số mạng theo chuẩn Kaiming Normal (He initialization) giúp hội tụ nhanh hơn
        self._init_weights()

    def _get_conv_out(self, shape: tuple[int, int, int]) -> int:
        """
        Hàm hỗ trợ tính toán tự động kích thước đầu ra của khối Conv2d.
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))

    def _init_weights(self) -> None:
        """
        Khởi tạo trọng số cho các lớp Conv2d và Linear bằng Kaiming Normal.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện tính toán tiến (forward pass).
        """
        # Chuẩn hóa giá trị pixel từ [0, 255] về [0.0, 1.0]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x.float() / 255.0

        conv_out = self.conv(x)
        flatten_out = conv_out.view(x.size(0), -1)
        return self.fc(flatten_out)


class MarioDuelingNet(nn.Module):
    """
    Mạng Neural Tích chập (CNN) cải tiến sử dụng kiến trúc Dueling DQN.
    Phân tách đầu ra thành 2 luồng tính toán song song: State Value V(s) và Action Advantage A(s, a).
    """

    def __init__(self, input_shape: tuple[int, int, int], n_actions: int) -> None:
        """
        Khởi tạo kiến trúc mạng MarioDuelingNet.
        """
        super().__init__()
        channels, _, _ = input_shape

        # Khối trích xuất đặc trưng hình ảnh (Feature Extractor)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Nhánh State Value (V) và Nhánh Advantage (A) cho Dueling DQN
        self.value_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        # Khởi tạo trọng số mạng theo chuẩn Kaiming Normal (He initialization)
        self._init_weights()

    def _get_conv_out(self, shape: tuple[int, int, int]) -> int:
        """
        Hàm hỗ trợ tính toán tự động kích thước đầu ra của khối Conv2d.
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(torch.prod(torch.tensor(o.size())))

    def _init_weights(self) -> None:
        """
        Khởi tạo trọng số bằng Kaiming Normal.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Thực hiện tính toán tiến (forward pass) kết hợp Dueling Q-value.
        """
        # Chuẩn hóa giá trị pixel từ [0, 255] về [0.0, 1.0]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x.float() / 255.0

        conv_out = self.conv(x)
        flatten_out = conv_out.view(x.size(0), -1)

        values = self.value_stream(flatten_out)
        advantages = self.advantage_stream(flatten_out)

        # Công thức kết hợp Dueling Q-value: Q = V + (A - mean(A))
        q_values = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
