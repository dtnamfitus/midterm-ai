from collections import deque
import matplotlib.pyplot as plt
import numpy as np
import time
import heapq

# Define the 15x15 maze grid: 0 = open, 1 = blocked
maze = np.array([
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]
])

start = (0, 0)
goal = (14, 14)

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def measure_time(func, *args):
    start_time = time.perf_counter()
    result = func(*args)
    end_time = time.perf_counter()
    return result, end_time - start_time

def a_star(maze, start, goal):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (current[0] + dr, current[1] + dc)
            if 0 <= neighbor[0] < 15 and 0 <= neighbor[1] < 15 and maze[neighbor] == 0:
                tentative_g_score = g_score[current] + 1
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    came_from[neighbor] = current
    
    path = []
    if goal in came_from:
        current = goal
        while current:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path
    return None

# Chạy A*
path_a_star, exec_time_a_star = measure_time(a_star, maze, start, goal)
print(f"A* Execution Time: {exec_time_a_star:.6f} seconds")
if path_a_star:
    print("A* Path:", path_a_star)

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(maze, cmap='gray_r', origin='upper')
    
    path_y, path_x = zip(*path_a_star)
    ax.plot(path_x, path_y, color='red', linewidth=2, label='A* Path')
    
    ax.scatter(start[1], start[0], c='green', s=100, label='Start (S)')
    ax.scatter(goal[1], goal[0], c='blue', s=100, label='Goal (G)')
    
    ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 15, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
    
    ax.legend()
    ax.set_title('A* Maze Path')
    plt.show()