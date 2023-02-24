import gym

# Khởi tạo môi trường
from train_agent import Agent

# Chương trình chính

env = gym.make("CartPole-v1")
state, _ = env.reset()

# Định nghĩa state_size và action_size
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Định nghĩa tham số khác
n_episodes = 10000
n_timesteps = 500
batch_size = 64

# Khởi tạo agent
my_agent = Agent(state_size, action_size)
total_time_step = 0

for ep in range(n_episodes):

    ep_rewards = 0
    state, _ = env.reset()

    for t in range(n_timesteps):

        total_time_step += 1
        # Cập nhật lại target NN mỗi my_agent.update_targetnn_rate
        if total_time_step % my_agent.update_targetnn_rate == 0:
            # Có thể chọn cách khác = weight của targetnetwork = 0 * weight của targetnetwork  + 1  * weight của mainnetwork
            my_agent.target_network.set_weights(my_agent.main_network.get_weights())

        action = my_agent.make_decision(state)
        next_state, reward, terminal, _, _ = env.step(action)
        my_agent.save_experience(state, action, reward, next_state, terminal)

        state = next_state
        ep_rewards += reward

        if terminal:
            print("Ep ", ep + 1, " reach terminal with reward = ", ep_rewards)
            break

        if len(my_agent.replay_buffer) > batch_size:
            my_agent.train_main_network(batch_size)

    if my_agent.epsilon > my_agent.epsilon_min:
        my_agent.epsilon = my_agent.epsilon * my_agent.epsilon_decay

    if ep % 100 ==0:
        # Save weights
        print("save weight at ep ",ep )
        my_agent.main_network.save("train_agent.h5")
