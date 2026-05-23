import csv
import os
import sys
import time

import gymnasium as gym
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
from src.agent import MarioDQN
from src.wrappers import create_mario_env


def make_env():
    """Hàm khởi tạo môi trường cho từng worker."""
    return create_mario_env(world=1, stage=1)


def train() -> None:
    """
    Vòng lặp huấn luyện chính cho Mario DQN sử dụng Multi-Worker.
    """
    num_envs = 8  # 8 workers để tận dụng sức mạnh CPU 10 cores / 16 luồng của bạn
    
    print("\n" + "=" * 60)
    print(" " * 10 + f"BẮT ĐẦU HUẤN LUYỆN MARIO DQN ({num_envs} WORKERS)")
    print("=" * 60 + "\n")

    # Khởi tạo môi trường Vector
    env_fns = [make_env for _ in range(num_envs)]
    env = gym.vector.AsyncVectorEnv(env_fns)

    # Khởi tạo Agent sử dụng mô hình Dueling DQN mới
    state_dim = (4, 84, 84)
    action_dim = env.single_action_space.n
    agent = MarioDQN(state_dim, action_dim, save_dir="checkpoints", model_type="dueling")

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
                ["Episode", "Step", "Reward", "Epsilon", "Loss", "Time", "Model"]
            )

    episodes = 2000
    save_every = 100  # Lưu checkpoint mỗi 100 episodes hoàn thành
    best_reward = -9999.0

    states, infos = env.reset()
    
    # Biến theo dõi tiến trình cho từng worker
    episode_rewards = np.zeros(num_envs)
    episode_losses = np.zeros(num_envs)
    episode_loss_counts = np.zeros(num_envs) 
    start_times = [time.time() for _ in range(num_envs)]
    
    completed_episodes = 0

    while completed_episodes < episodes:
        # Chọn hành động (Batch size = num_envs)
        actions = agent.act(states)

        # Tương tác môi trường
        next_states, rewards, terminateds, truncateds, next_infos = env.step(actions)
        
        # Học tập 1 lần cho mỗi step tổng
        # (Bạn có thể lặp agent.learn() nhiều lần nếu muốn học nhiều hơn)
        loss = agent.learn()

        for i in range(num_envs):
            done = terminateds[i] or truncateds[i]

            # Xử lý True Next State do AsyncVectorEnv tự động reset
            if done and "final_observation" in next_infos and next_infos["_final_observation"][i]:
                true_next_state = next_infos["final_observation"][i]
            else:
                true_next_state = next_states[i]

            # Lưu vào bộ nhớ
            agent.cache(states[i], actions[i], rewards[i], true_next_state, done)

            episode_rewards[i] += float(rewards[i])
            if loss is not None:
                episode_losses[i] += loss
                episode_loss_counts[i] += 1

            if done:
                completed_episodes += 1
                duration = time.time() - start_times[i]
                avg_loss = episode_losses[i] / episode_loss_counts[i] if episode_loss_counts[i] > 0 else 0.0

                # In log ra console
                print(
                    f"[Ep {completed_episodes:4d} | W{i}] | Steps: {agent.curr_step:7d} | "
                    f"Reward: {episode_rewards[i]:6.1f} | Epsilon: {agent.epsilon:.4f} | "
                    f"Loss: {avg_loss:.4f} | Time: {duration:.1f}s"
                )

                # Ghi log ra CSV
                with open(log_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            completed_episodes,
                            agent.curr_step,
                            episode_rewards[i],
                            agent.epsilon,
                            avg_loss,
                            duration,
                            "DuelingDDQN" if agent.model_type == "dueling" else "VanillaDQN",
                        ]
                    )

                # Lưu checkpoint định kỳ hoặc khi đạt kỷ lục mới
                if episode_rewards[i] > best_reward:
                    best_reward = episode_rewards[i]
                    agent.save_checkpoint("mario_dqn_best.pt")
                    print(
                        f" ---> [!] Kỷ lục mới: {best_reward:.1f}! Đã lưu mario_dqn_best.pt"
                    )

                if completed_episodes % save_every == 0:
                    agent.save_checkpoint("mario_dqn.pt")
                    print(f" ---> [*] Đã lưu checkpoint định kỳ tại episode {completed_episodes}.")

                # Reset tracking cho worker này
                episode_rewards[i] = 0.0
                episode_losses[i] = 0.0
                episode_loss_counts[i] = 0.0
                start_times[i] = time.time()

        states = next_states

    env.close()
    print("\n[*] Quá trình huấn luyện đã hoàn tất!")


if __name__ == "__main__":
    train()
