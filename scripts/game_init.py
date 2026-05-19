import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# Khởi tạo môi trường Mario màn 1-1 gốc
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# Giới hạn số nút bấm (chỉ lấy các nút di chuyển đơn giản: Phải, Nhảy, chạy nhanh...)
env = JoypadSpace(env, SIMPLE_MOVEMENT)