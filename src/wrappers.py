"""
Module Tiền xử lý Môi trường (Environment Wrappers) cho dự án Mario DQN.
Được thiết kế theo chuẩn Python Pro 3.12+ và tương thích hoàn toàn với Gym 0.26.2.
"""

from collections import deque
from typing import Any
import cv2
import gymnasium as gym
from gymnasium.spaces import Box
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np


class SkipFrame(gym.Wrapper):
    """
    Wrapper bỏ qua N frames liên tiếp.
    Hành động được chọn sẽ lặp lại trong suốt N frames này.
    Giúp tăng tốc độ chơi game và giảm tải tính toán cho CPU.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self._skip = skip

    def step(self, action: int) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Lặp lại hành động trong `skip` frames và cộng dồn phần thưởng (reward).
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info: dict[str, Any] = {}

        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break

        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    Wrapper chuyển đổi ảnh màu RGB sang ảnh xám (Grayscale).
    Giảm 2/3 dung lượng bộ nhớ đầu vào cho mạng CNN.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Chuyển đổi mảng RGB thành Grayscale bằng OpenCV.
        """
        return cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """
    Wrapper thu nhỏ kích thước khung hình về (width, height) chuẩn.
    Mặc định: (84, 84) theo chuẩn bài báo Nature DQN.
    """

    def __init__(self, env: gym.Env, shape: tuple[int, int] = (84, 84)) -> None:
        super().__init__(env)
        self._shape = shape
        self.observation_space = Box(low=0, high=255, shape=self._shape, dtype=np.uint8)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        """
        Thu nhỏ ảnh bằng phương pháp INTER_AREA của OpenCV.
        """
        return cv2.resize(observation, self._shape, interpolation=cv2.INTER_AREA)


class FrameStack(gym.Wrapper):
    """
    Wrapper gộp N frames liên tiếp thành 1 observation duy nhất (N, width, height).
    Giúp AI nhận biết được hướng di chuyển và vận tốc của các đối tượng trong game.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
        super().__init__(env)
        self._num_stack = num_stack
        self._frames: deque[np.ndarray] = deque(maxlen=num_stack)

        # Định nghĩa lại observation space: (num_stack, 84, 84)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[np.newaxis, ...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset môi trường và làm đầy buffer frames bằng frame đầu tiên.
        """
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._num_stack):
            self._frames.append(obs)
        return self._get_observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Thực hiện step và đẩy frame mới vào buffer frames.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Gộp các frames trong deque thành mảng numpy 3D (num_stack, 84, 84).
        """
        return np.stack(list(self._frames), axis=0)


def create_mario_env(
    world: int = 1,
    stage: int = 1,
    actions: list[list[str]] = SIMPLE_MOVEMENT,
    render_mode: str | None = None,
) -> gym.Env:
    """
    Khởi tạo và bọc (wrap) môi trường Super Mario Bros với các lớp tiền xử lý tối ưu.

    Parameters:
        world (int): Thế giới trong Mario (1-8). Mặc định 1.
        stage (int): Màn chơi trong thế giới (1-4). Mặc định 1.
        actions (list): Danh sách các hành động cho phép. Mặc định SIMPLE_MOVEMENT (7 hành động).
        render_mode (str | None): Chế độ hiển thị ('human' hoặc None). Mặc định None.

    Returns:
        gym.Env: Môi trường Mario đã được wrap đầy đủ.
    """
    env_name = f"SuperMarioBros-{world}-{stage}-v0"
    if render_mode:
        env = gym_super_mario_bros.make(
            env_name, render_mode=render_mode, disable_env_checker=True
        )
    else:
        env = gym_super_mario_bros.make(env_name, disable_env_checker=True)

    # 1. Giới hạn nút bấm (JoypadSpace)
    env = JoypadSpace(env, actions)
    # 2. Bỏ qua frame (SkipFrame)
    env = SkipFrame(env, skip=4)
    # 3. Chuyển ảnh xám (GrayScaleObservation)
    env = GrayScaleObservation(env)
    # 4. Thu nhỏ 84x84 (ResizeObservation)
    env = ResizeObservation(env, shape=(84, 84))
    # 5. Gộp 4 frames liên tiếp (FrameStack)
    env = FrameStack(env, num_stack=4)

    return env
