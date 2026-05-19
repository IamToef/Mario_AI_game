"""
Module Bộ nhớ Kinh nghiệm (Replay Buffer) cho dự án Mario DQN.
Được thiết kế theo chuẩn Python Pro 3.12+ với Type Hints và Docstrings đầy đủ.
"""

from collections import deque
import random
from typing import Any
import numpy as np
import torch


class ReplayMemory:
    """
    Bộ nhớ lưu trữ các trải nghiệm (experiences) của Agent trong quá trình tương tác với môi trường.
    Sử dụng cấu trúc deque (hàng đợi hai đầu) để tự động loại bỏ các trải nghiệm cũ khi vượt quá maxlen.
    """

    def __init__(self, capacity: int = 30000) -> None:
        """
        Khởi tạo bộ nhớ với sức chứa tối đa `capacity`.
        """
        self.memory: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Lưu trữ một bộ kinh nghiệm (state, action, reward, next_state, done) vào bộ nhớ.

        Parameters:
            state (np.ndarray): Trạng thái hiện tại (4, 84, 84).
            action (int): Hành động được chọn.
            reward (float): Phần thưởng nhận được.
            next_state (np.ndarray): Trạng thái tiếp theo (4, 84, 84).
            done (bool): Cờ báo hiệu kết thúc episode (terminated hoặc truncated).
        """
        self.memory.append((state, action, float(reward), next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lấy mẫu ngẫu nhiên một mini-batch các trải nghiệm từ bộ nhớ và chuyển đổi sang PyTorch Tensors.

        Parameters:
            batch_size (int): Số lượng trải nghiệm cần lấy mẫu.
            device (torch.device): Thiết bị tính toán (CPU hoặc GPU).

        Returns:
            tuple[torch.Tensor, ...]: Các tensors (states, actions, rewards, next_states, dones) đã sẵn sàng cho tính toán Loss.
        """
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Chuyển đổi sang PyTorch Tensors
        states_tensor = torch.tensor(
            np.array(states), dtype=torch.float32, device=device
        )
        actions_tensor = torch.tensor(
            actions, dtype=torch.int64, device=device
        ).unsqueeze(1)
        rewards_tensor = torch.tensor(
            rewards, dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states_tensor = torch.tensor(
            np.array(next_states), dtype=torch.float32, device=device
        )
        dones_tensor = torch.tensor(
            dones, dtype=torch.float32, device=device
        ).unsqueeze(1)

        return (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )

    def __len__(self) -> int:
        """
        Trả về số lượng trải nghiệm hiện có trong bộ nhớ.
        """
        return len(self.memory)
