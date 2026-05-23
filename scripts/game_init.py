import gym_super_mario_bros
from gym_super_mario_bros.smb_env import SuperMarioBrosEnv
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Khởi tạo trực tiếp môi trường Mario màn 1-1 gốc để tránh lỗi wrapper gym cũ
env = SuperMarioBrosEnv(rom_mode='vanilla', lost_levels=False, target=(1, 1))
# Giới hạn số nút bấm (chỉ lấy các nút di chuyển đơn giản: Phải, Nhảy, chạy nhanh...)
env = JoypadSpace(env, SIMPLE_MOVEMENT)