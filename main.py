"""
Kịch bản Huấn luyện chính (Training Loop) & MLOps Tracking cho dự án Mario DQN.
Được thiết kế theo chuẩn Python Pro 3.12+ với ghi log CSV và in tiến trình trực quan.
"""

import csv
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
from agent import MarioDQN
from wrappers import create_mario_env


def train() -> None:
    """
    Vòng lặp huấn luyện chính cho Mario DQN.
    """
    # Khởi tạo môi trường
    env = create_mario_env(world=1, stage=1)

    # Khởi tạo Agent
    state_dim = (4, 84, 84)
    action_dim = env.action_space.n
    agent = MarioDQN(state_dim, action_dim, save_dir="checkpoints")

    # Thử tải lại checkpoint cũ nếu có để tiếp tục học (Resume training)
    if agent.load_checkpoint("mario_dqn.pt"):
        print(
            f"[*] Đã tải lại checkpoint thành công! Tiếp tục học từ bước: {agent.curr_step}"
        )
    else:
        print("[*] Khởi tạo mô hình mới hoàn toàn.")

    # Cấu hình MLOps Tracking (CSV Log)
    log_file = "training_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                ["Episode", "Step", "Reward", "Epsilon", "Loss", "Time"]
            )

    episodes = 10000
    save_every = 100  # Lưu checkpoint mỗi 100 episodes
    best_reward = -9999.0

    print("\n" + "=" * 60)
    print(" " * 15 + "BẮT ĐẦU HUẤN LUYỆN MARIO DQN")
    print("=" * 60 + "\n")

    for e in range(1, episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0.0
        total_loss = 0.0
        loss_count = 0
        start_time = time.time()

        while not done:
            # Chọn hành động
            action = agent.act(state)

            # Tương tác môi trường
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Lưu vào bộ nhớ
            agent.cache(state, action, reward, next_state, done)

            # Học tập
            loss = agent.learn()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            state = next_state

        duration = time.time() - start_time
        avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

        # In log ra console
        print(
            f"[Episode {e:4d}] | Steps: {agent.curr_step:7d} | Reward: {total_reward:6.1f} | "
            f"Epsilon: {agent.epsilon:.4f} | Loss: {avg_loss:.4f} | Time: {duration:.1f}s"
        )

        # Ghi log ra CSV
        with open(log_file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    e,
                    agent.curr_step,
                    total_reward,
                    agent.epsilon,
                    avg_loss,
                    duration,
                ]
            )

        # Lưu checkpoint định kỳ hoặc khi đạt kỷ lục mới
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save_checkpoint("mario_dqn_best.pt")
            print(
                f" ---> [!] Kỷ lục mới: {best_reward:.1f}! Đã lưu mario_dqn_best.pt"
            )

        if e % save_every == 0:
            agent.save_checkpoint("mario_dqn.pt")
            print(f" ---> [*] Đã lưu checkpoint định kỳ tại episode {e}.")

    env.close()
    print("\n[*] Quá trình huấn luyện đã hoàn tất!")


if __name__ == "__main__":
    train()
