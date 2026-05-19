"""
Kịch bản Đánh giá Trực quan (Visual Evaluation) cho dự án Mario DQN.
Khởi tạo môi trường ở chế độ 'human', tải trọng số đã học và cho Mario tự chạy.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
from agent import MarioDQN
from wrappers import create_mario_env


def play() -> None:
    """
    Hàm chạy thử nghiệm trực quan với trọng số đã huấn luyện.
    """
    print("\n" + "=" * 60)
    print(" " * 15 + "CHẾ ĐỘ ĐÁNH GIÁ TRỰC QUAN MARIO DQN")
    print("=" * 60 + "\n")

    # Khởi tạo môi trường với chế độ hiển thị 'human'
    env = create_mario_env(world=1, stage=1, render_mode="human")

    # Khởi tạo Agent
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    agent = MarioDQN(state_dim, action_dim, save_dir="checkpoints")

    # Tải trọng số tốt nhất (nếu có) hoặc trọng số mới nhất
    if agent.load_checkpoint("mario_dqn_best.pt"):
        print("[*] Đã tải trọng số mario_dqn_best.pt thành công!")
    elif agent.load_checkpoint("mario_dqn.pt"):
        print("[*] Đã tải trọng số mario_dqn.pt thành công!")
    else:
        print(
            "[!] CẢNH BÁO: Không tìm thấy file trọng số nào. Agent sẽ chạy ngẫu nhiên (Random Agent)!"
        )

    # Tắt hoàn toàn cơ chế khám phá ngẫu nhiên
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0

    episodes = 5
    for e in range(1, episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0.0

        print(f"\n---> Bắt đầu Episode {e}...")
        while not done:
            # Chọn hành động tối ưu theo Q-value
            action = agent.act(state)

            # Thực thi hành động
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)

            # Ngủ 0.02s để tốc độ hiển thị vừa mắt người xem
            time.sleep(0.02)

        print(
            f"[*] Kết thúc Episode {e} | Tổng phần thưởng đạt được: {total_reward:.1f}"
        )

    env.close()
    print("\n[*] Hoàn tất quá trình đánh giá!")


if __name__ == "__main__":
    play()
