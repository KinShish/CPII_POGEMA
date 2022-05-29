import numpy as np
from pogema import GridConfig
from heapq import heappop, heappush

class Node:
    def __init__(self, coord: (int, int) = (0, 0), g: int = 0, h: int = 0):
        self.i, self.j = coord
        self.g = g
        self.h = h
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f or ((self.f == other.f) and (self.g < other.g))

class AStar:
    def __init__(self):
        self.start = (0, 0)
        self.goal = (0, 0)
        self.max_steps = 10000  # due to the absence of information about the map size we need some other stop criterion
        self.OPEN = list()
        self.CLOSED = dict()
        self.obstacles = set()
        self.other_agents = set()
        self.compass = []
        self.last_move = None
        self.stay = 0
        self.old_position = (0, 0)
        self.Zzz_position = (0, 0)
        self.repeat = 0
        self.stop = 0
        self.block_history = 0
        self.inaction = 0
        self.obs47=set()
        self.obs38 = set()
        self.agents47=set()

    def compute_shortest_path(self, start, goal):
        self.start = start
        self.goal = goal
        self.CLOSED = dict()
        self.OPEN = list()
        heappush(self.OPEN, Node(self.start))
        u = Node()
        steps = 0
        while len(self.OPEN) > 0 and steps < self.max_steps and (u.i, u.j) != self.goal:
            u = heappop(self.OPEN)
            steps += 1
            for d in self.compass:
                n = (u.i+d[0], u.j + d[1])
                h = abs(n[0] - self.goal[0]) + abs(
                    n[1] - self.goal[1])  # Manhattan distance as a heuristic function
                if n not in self.obstacles and n not in self.CLOSED and (n not in self.other_agents or d[0] == 1 or d[1] == 1):# or d[0] == 1 or d[1] == 1
                    heappush(self.OPEN, Node(n, u.g + 1, h))
                    self.CLOSED[n] = (u.i, u.j)  # store information about the predecessor

    def get_open_steps(self):
        matrix = self.obs47+self.agents47
        result = []
        # 1 - вверх, 2 - вниз, 3 -влево, 4 - вправо, 0 - пропуск
        if self.obs47[0, 1]+self.agents47[0, 1] == 0 and (
                self.obs38[0,2]+self.obs38[1,1]+self.obs38[1,3]!=3 or self.obs38[2,1]+self.obs38[2,3]==0): result.append(1)
        if self.obs47[2, 1]+self.agents47[2, 1] == 0 and (
                self.obs38[4,2]+self.obs38[3,1]+self.obs38[3,3]!=3 or self.obs38[2,1]+self.obs38[2,3]==0): result.append(2)
        if self.obs47[1, 0]+self.agents47[1, 0] == 0 and (
                self.obs38[1,1]+self.obs38[2,0]+self.obs38[3,1]!=3 or self.obs38[1,2]+self.obs38[3,2]==0): result.append(3)
        if self.obs47[1, 2]+self.agents47[1, 2] == 0 and (
                self.obs38[1,3]+self.obs38[2,4]+self.obs38[3,3]!=3 or self.obs38[1,2]+self.obs38[3,2]==0): result.append(4)
        if len(result) == 0: result.append(0)
        for i in result:
            if i==self.last_move:
                return [self.last_move]
        return result

    def get_next_node(self):
        next_node = self.start  # if path not found, current start position is returned
        if self.goal in self.CLOSED:  # if path found
            next_node = self.goal
            while self.CLOSED[next_node] != self.start:  # get node in the path with start node as a predecessor
                next_node = self.CLOSED[next_node]
        return next_node

    def get_time_stop(self):
        return (6 - np.sum(self.obs47)+self.block_history)//2

    def watch_repeat(self, move):
        # 1 - вверх, 2 - вниз, 3 -влево, 4 - вправо, 0 - пропуск
        if move != 0:
            if (move == 1 and self.last_move == 2) or (
                    move == 2 and self.last_move == 1) or (
                    move == 3 and self.last_move == 4) or (
                    move == 4 and self.last_move == 3):
                self.repeat += 1
                self.stay += 1
            else:
                self.repeat = 0
            self.last_move = move

    def reset_position(self, obs, positions_xy):
        if self.Zzz_position == positions_xy:
            self.stay += 1
        else:
            if self.stay>0:
                self.stay -= 1
            else:
                self.stay = 0
        self.Zzz_position = positions_xy
        if self.old_position == (0, 0) and self.last_move:
            if (self.last_move == 3 and np.sum(np.matrix(obs)[:, 0]) == 11) or (self.last_move == 4 and np.sum(np.matrix(obs)[:, 10]) == 11) or (self.last_move == 1 and np.sum(np.matrix(obs)[:0, ]) == 11) or (self.last_move == 2 and np.sum(np.matrix(obs)[:10, ]) == 11):
                self.old_position = positions_xy
                obstacles = set()
                while len(self.obstacles) > 1:
                    obstacle = self.obstacles.pop()
                    obstacles.add((obstacle[0] - positions_xy[0], obstacle[1] - positions_xy[1]))
                self.obstacles = obstacles

    def update_compass(self, comp):
        self.compass.clear()
        compass = np.transpose(np.nonzero(comp))[0]
        if compass[1] > 5:
            self.compass.extend([(-1, 0), (1, 0)])
        else:
            self.compass.extend([(1, 0), (-1, 0)])
        if compass[0] > 5:
            self.compass.extend([(0, 1), (0, -1)])
        else:
            self.compass.extend([(0, -1), (0, 1)])

    def update_obstacles(self, obs, other_agents, n):
        obstacles = np.transpose(np.nonzero(obs))  # get the coordinates of all obstacles in current observation
        for obstacle in obstacles:
            self.obstacles.add((n[0] + obstacle[0], n[1] + obstacle[1]))  # save them with correct coordinates
        self.other_agents.clear()  # forget previously seen agents as they move
        agents = np.transpose(np.nonzero(other_agents))  # get the coordinates of all agents that are seen
        for agent in agents:
            #if abs(agent[0] - 5) < len(agents) and abs(agent[1] - 5) < len(agents):
            self.other_agents.add((n[0] + agent[0], n[1] + agent[1]))  # save them with correct coordinates


class Model:
    def __init__(self):
        self.agents = None
        self.actions = {tuple(GridConfig().MOVES[i]): i for i in
                        range(len(GridConfig().MOVES))}  # make a dictionary to translate coordinates of actions into id

    def act(self, obs, dones, positions_xy, targets_xy) -> list:
        if self.agents is None:
            self.agents = [AStar() for _ in range(len(obs))]  # create a planner for each of the agents
        actions = []
        for k in range(len(obs)):
            if positions_xy[k] == targets_xy[k]:  # don't waste time on the agents that have already reached their goals
                actions.append(0)  # just add useless action to save the order and length of the actions
                continue
            if self.agents[k].repeat > 2:
                self.agents[k].stop = self.agents[k].get_time_stop()
                self.agents[k].repeat = 0
            self.agents[k].obs47 = np.matrix(obs[k][0])[4:7, 4:7]
            self.agents[k].obs38 = np.matrix(obs[k][0])[3:8, 3:8]
            self.agents[k].agents47 = np.matrix(obs[k][1])[4:7, 4:7]
            self.agents[k].reset_position(obs[k][0], positions_xy[k])
            if (self.agents[k].stop == 0 and self.agents[k].stay < 3) or len(self.agents[k].other_agents) == 1:
                position = (positions_xy[k][0] - self.agents[k].old_position[0], positions_xy[k][1] - self.agents[k].old_position[1])
                target = (targets_xy[k][0] - self.agents[k].old_position[0], targets_xy[k][1] - self.agents[k].old_position[1])
                self.agents[k].update_obstacles(obs[k][0], obs[k][1], (position[0] - 5, position[1] - 5))
                self.agents[k].update_compass(obs[k][2])
                self.agents[k].compute_shortest_path(start=position, goal=target)
                next_node = self.agents[k].get_next_node()
                actions.append(self.actions[(next_node[0] - position[0], next_node[1] - position[1])])
            else:
                if self.agents[k].stop > 0:
                    self.agents[k].stop -= 1
                else:
                    self.agents[k].stop = 0
                if self.agents[k].stay>1:
                    arr = self.agents[k].get_open_steps()
                    if len(arr)>1:
                        actions.append(np.random.choice(arr))
                    else:
                        actions.append(arr[0])
                else:
                    actions.append(0)
                self.agents[k].inaction += 1
            self.agents[k].watch_repeat(actions[k])
            self.agents[k].block_history = (6 - np.sum(self.agents[k].obs47))
        return actions
