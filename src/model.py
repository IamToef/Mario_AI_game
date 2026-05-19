"""
Module Mạng Neural CNN (MarioNet) cho dự án Mario DQN.
Được thiết kế theo chuẩn Python Pro 3.12+ và tối ưu hóa cho CPU.
"""

import torch
from torch import nn


class MarioNet(nn.Module):
    """
    Mạng Neural Tích chập (CNN) trích xuất đặc trưng hình ảnh và dự đoán giá trị Q-value cho từng hành động.
    Kiến trúc được tinh gọn tối đa để đảm bảo tốc độ tính toán forward/backward cực nhanh trên CPU.
    """

    def __init__(self, input_shape: tuple[int, int, int], n_actions: int) -> None:
        """
        Khởi tạo kiến trúc mạng MarioNet.

        Parameters:
            input_shape (tuple): Kích thước đầu vào (channels, height, width). Mặc định (4, 84, 84).
            n_actions (int): Số lượng hành động đầu ra (ví dụ 7 hành động trong SIMPLE_MOVEMENT).
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

        Parameters:
            x (torch.Tensor): Tensor đầu vào (batch_size, 4, 84, 84). Chuẩn hóa giá trị về [0, 1].

        Returns:
            torch.Tensor: Tensor chứa giá trị Q-value cho từng hành động (batch_size, n_actions).
        """
        # Chuẩn hóa giá trị pixel từ [0, 255] về [0.0, 1.0]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        elif x.max() > 1.0:
            x = x.float() / 255.0

        conv_out = self.conv(x)
        flatten_out = conv_out.view(x.size(0), -1)
        return self.fc(flatten_out)
