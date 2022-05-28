import gym
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor
from IPython.display import SVG, display

import pogema
from pogema import GridConfig
import model.agent as Agent
agent = Agent.Model()

# Define random configuration
grid_config = GridConfig(num_agents=30, # количество агентов на карте
                         size=80,      # размеры карты
                         density=0.3,  # плотность препятствий
                         seed=5,       # сид генерации задания
                         max_episode_steps=256,  # максимальная длина эпизода
                         obs_radius=5, # радиус обзора
                        )

env = gym.make("Pogema-v0", grid_config=grid_config)
env = AnimationMonitor(env)

# обновляем окружение
obs = env.reset()

done = [False for k in range(len(obs))]
steps = 0

while not all(done):
    # Используем случайную стратегию
    #print(reward)
    #move = agent.act(obs, done, positions_xy, [])
    # 1 - вверх, 2 - вниз, 3 -влево, 4 - вправо, 0 - пропуск
    steps += 1
    obs, reward, done, info = env.step(agent.act(obs, done, env.get_agents_xy_relative(), env.get_targets_xy_relative()))
# сохраняем анимацию и рисуем ее
print(steps)
env.save_animation("render.svg", egocentric_idx=None)
display(SVG('render.svg'))