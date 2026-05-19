"""
Module DQN Agent (MarioDQN) cho dự án Mario DQN.
Được thiết kế theo chuẩn Python Pro 3.12+ quản lý Epsilon-Greedy, Loss, Optimizer, Target Sync và Checkpoints.
"""

import copy
import os
from typing import Any
from memory import ReplayMemory
from model import MarioNet
import numpy as np
import torch
from torch import nn


class MarioDQN:
    """
    DQN Agent quản lý toàn bộ vòng lặp học tập, lưu trữ kinh nghiệm, chọn hành động và cập nhật trọng số.
    """

    def __init__(
        self,
        state_dim: tuple[int, int, int],
        action_dim: int,
        save_dir: str = "checkpoints",
    ) -> None:
        """
        Khởi tạo MarioDQN Agent.

        Parameters:
            state_dim (tuple): Kích thước state đầu vào (channels, height, width).
            action_dim (int): Số lượng hành động đầu ra.
            save_dir (str): Thư mục lưu trữ trọng số (checkpoints).
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Ưu tiên sử dụng CPU theo cấu hình máy
        self.device = torch.device("cpu")

        # Khởi tạo Mạng Online và Mạng Target
        self.online_net = MarioNet(self.state_dim, self.action_dim).to(
            self.device
        )
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()  # Mạng Target chỉ dùng để tính toán, không cập nhật gradient

        # Bộ nhớ kinh nghiệm (Replay Buffer)
        self.memory = ReplayMemory(capacity=30000)

        # Các siêu tham số (Hyperparameters) của DQN
        self.gamma: float = 0.90  # Hệ số chiết khấu (Discount factor)
        self.epsilon: float = 1.0  # Tỷ lệ khám phá ngẫu nhiên ban đầu
        self.epsilon_min: float = 0.05  # Tỷ lệ khám phá ngẫu nhiên tối thiểu
        self.epsilon_decay: float = 0.999999  # Tốc độ giảm epsilon qua từng step

        self.batch_size: int = 32
        self.sync_rate: int = (
            10000  # Số steps để đồng bộ mạng Online sang Target
        )
        self.curr_step: int = 0  # Đếm tổng số bước đã thực hiện

        # Bộ tối ưu và Hàm mất mát
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=0.00025
        )
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, state: np.ndarray) -> int:
        """
        Chọn hành động dựa theo chính sách Epsilon-Greedy.

        Parameters:
            state (np.ndarray): Trạng thái hiện tại (4, 84, 84).

        Returns:
            int: Hành động được chọn.
        """
        # Khám phá ngẫu nhiên (Exploration)
        if np.random.rand() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            # Khai thác tri thức (Exploitation)
            state_tensor = (
                torch.tensor(state, dtype=torch.float32, device=self.device)
                .unsqueeze(0)
            )
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
                action = int(torch.argmax(q_values, dim=1).item())

        # Giảm dần epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.curr_step += 1
        return action

    def cache(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Lưu trữ trải nghiệm vào Replay Memory.
        """
        self.memory.push(state, action, reward, next_state, done)

    def learn(self) -> float | None:
        """
        Lấy mẫu từ Replay Memory và cập nhật trọng số mạng Online.

        Returns:
            float | None: Giá trị Loss tính toán được (hoặc None nếu chưa đủ dữ liệu trong buffer).
        """
        # Chưa đủ dữ liệu trong buffer thì chưa học
        if len(self.memory) < self.batch_size:
            return None

        # Đồng bộ Mạng Online sang Target định kỳ
        if self.curr_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        # Lấy mẫu mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Tính Q-values hiện tại: Q_online(s, a)
        curr_q = self.online_net(states).gather(1, actions)

        # Tính Q-values mục tiêu: r + gamma * max(Q_target(s', a')) * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        # Tính toán Loss và cập nhật gradient
        loss = self.loss_fn(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())

    def save_checkpoint(self, filename: str = "mario_dqn.pt") -> None:
        """
        Lưu trữ trọng số mạng Online và các trạng thái học tập.
        """
        save_path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                "curr_step": self.curr_step,
                "epsilon": self.epsilon,
                "model_state_dict": self.online_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            save_path,
        )

    def load_checkpoint(self, filename: str = "mario_dqn.pt") -> bool:
        """
        Tải lại trọng số mạng Online và các trạng thái học tập từ file checkpoint.

        Returns:
            bool: True nếu tải thành công, False nếu không tìm thấy file.
        """
        save_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(save_path):
            return False

        checkpoint = torch.load(save_path, map_location=self.device)
        self.curr_step = checkpoint.get("curr_step", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
        self.online_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return True
