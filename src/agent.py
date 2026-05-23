"""
Module DQN Agent (MarioDQN) cho dự án Mario DQN.
Được thiết kế theo chuẩn Python Pro 3.12+ quản lý Epsilon-Greedy, Loss, Optimizer, Target Sync và Checkpoints.
"""

import copy
import os
from typing import Any
from src.memory import ReplayMemory
from src.model import MarioNet, MarioDuelingNet
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
        model_type: str = "dueling",
    ) -> None:
        """
        Khởi tạo MarioDQN Agent.

        Parameters:
            state_dim (tuple): Kích thước state đầu vào (channels, height, width).
            action_dim (int): Số lượng hành động đầu ra.
            save_dir (str): Thư mục lưu trữ trọng số (checkpoints).
            model_type (str): Kiểu mô hình mạng nơ-ron sử dụng ('vanilla' hoặc 'dueling').
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model_type = model_type
        # Tự động phân tách thư mục checkpoints theo loại mô hình đang sử dụng
        self.save_dir = os.path.join(save_dir, self.model_type)
        os.makedirs(self.save_dir, exist_ok=True)

        # Ưu tiên sử dụng GPU (CUDA) nếu có, nếu không thì dùng CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Khởi tạo Mạng Online và Mạng Target dựa trên model_type
        if self.model_type == "vanilla":
            self.online_net = MarioNet(self.state_dim, self.action_dim).to(self.device)
        else:
            self.online_net = MarioDuelingNet(self.state_dim, self.action_dim).to(self.device)

        self.target_net = copy.deepcopy(self.online_net)
        self.target_net.eval()  # Mạng Target chỉ dùng để tính toán, không cập nhật gradient

        # Bộ nhớ kinh nghiệm (Replay Buffer)
        self.memory = ReplayMemory(capacity=100000)

        # Các siêu tham số (Hyperparameters) của DQN
        self.gamma: float = 0.99  # Hệ số chiết khấu (Discount factor) - tăng từ 0.90 để học dài hạn
        self.epsilon: float = 1.0  # Tỷ lệ khám phá ngẫu nhiên ban đầu
        self.epsilon_min: float = 0.05  # Tỷ lệ khám phá ngẫu nhiên tối thiểu
        self.epsilon_decay: float = 0.9999962  # Tốc độ giảm epsilon qua từng step

        self.batch_size: int = 64  # Tăng batch size từ 32 lên 64 giúp gradient ổn định hơn
        self.sync_rate: int = 10000  # Số steps để đồng bộ mạng Online sang Target
        self.curr_step: int = 0  # Đếm tổng số bước đã thực hiện
        self.last_sync: int = 0  # Lần đồng bộ gần nhất

        # Bộ tối ưu và Hàm mất mát
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(), lr=0.00025
        )
        self.loss_fn = nn.SmoothL1Loss()

    def act(self, state: np.ndarray) -> np.ndarray | int:
        """
        Chọn hành động dựa theo chính sách Epsilon-Greedy. Hỗ trợ cả 1 state (play.py) và batch states (main.py).

        Parameters:
            state (np.ndarray): Trạng thái hiện tại. Có thể là 3D (4, 84, 84) hoặc 4D (num_envs, 4, 84, 84).

        Returns:
            np.ndarray | int: Hành động được chọn.
        """
        is_single = state.ndim == 3
        if is_single:
            state = np.expand_dims(state, axis=0)

        num_envs = state.shape[0]

        # Khám phá ngẫu nhiên (Exploration)
        explore_mask = np.random.rand(num_envs) < self.epsilon
        rand_actions = np.random.randint(self.action_dim, size=num_envs)

        # Khai thác tri thức (Exploitation)
        if not np.all(explore_mask):
            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            )
            with torch.no_grad():
                q_values = self.online_net(state_tensor)
                model_actions = torch.argmax(q_values, dim=1).cpu().numpy()
        else:
            model_actions = rand_actions

        # Trộn ngẫu nhiên và mô hình
        actions = np.where(explore_mask, rand_actions, model_actions)

        # Giảm dần epsilon và tăng step
        for _ in range(num_envs):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.curr_step += num_envs
        
        return int(actions[0]) if is_single else actions

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

        # Đồng bộ Mạng Online sang Target định kỳ an toàn (vì curr_step nhảy theo num_envs)
        if self.curr_step - self.last_sync >= self.sync_rate:
            self.target_net.load_state_dict(self.online_net.state_dict())
            self.last_sync = self.curr_step

        # Lấy mẫu mini-batch
        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Tính Q-values hiện tại: Q_online(s, a)
        curr_q = self.online_net(states).gather(1, actions)

        # Tính Q-values mục tiêu bằng Double DQN (DDQN)
        with torch.no_grad():
            # Chọn hành động tốt nhất ở trạng thái tiếp theo bằng mạng Online
            best_next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
            # Lấy giá trị Q-value của hành động đó từ mạng Target
            next_q = self.target_net(next_states).gather(1, best_next_actions)
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
            bool: True nếu tải thành công, False nếu không tìm thấy file hoặc lỗi kiến trúc.
        """
        save_path = os.path.join(self.save_dir, filename)
        if not os.path.exists(save_path):
            return False

        try:
            checkpoint = torch.load(save_path, map_location=self.device)
            self.online_net.load_state_dict(checkpoint["model_state_dict"])
            self.target_net = copy.deepcopy(self.online_net)
            self.target_net.eval()
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.curr_step = checkpoint.get("curr_step", 0)
            self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
            return True
        except Exception as e:
            print(f"[!] Cảnh báo: Không thể tải checkpoint {filename} do lỗi hoặc không khớp kiến trúc mô hình.")
            print(f"    Chi tiết lỗi: {e}")
            print("[*] Agent sẽ khởi động với mô hình mới hoàn toàn.")
            return False
