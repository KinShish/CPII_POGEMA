import gym
from pogema.wrappers.multi_time_limit import MultiTimeLimit
from pogema.animation import AnimationMonitor

from datetime import datetime

import pogema
from pogema import GridConfig
import model.agent as Agent
def start(s,d,n):
    # Define random configuration
    agent = Agent.Model()
    print(s,d,n)
    grid_config = GridConfig(num_agents=n, # количество агентов на карте
                             size=s,      # размеры карты
                             density=d,  # плотность препятствий
                             seed=0,       # сид генерации задания
                             max_episode_steps=1000,  # максимальная длина эпизода
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
    #name="render"+datetime.now().strftime("%m.%d.%Y_%H:%M:%S")+".svg"
    name = "render.svg"
    env.save_animation(name, egocentric_idx=None)
    #display(SVG(name))
    return [steps,name]