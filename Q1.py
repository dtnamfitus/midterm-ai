from collections import deque
import matplotlib.pyplot as plt
import numpy as np

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

def bfs(maze, start, goal):
    queue = deque([start])
    visited = {start: None}

    while queue:
        current = queue.popleft()
        if current == goal:
            break

        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            r, c = current[0] + dr, current[1] + dc
            if 0 <= r < 15 and 0 <= c < 15 and maze[r, c] == 0 and (r, c) not in visited:
                visited[(r, c)] = current
                queue.append((r, c))

    path = []
    if goal in visited:
        current = goal
        while current:
            path.append(current)
            current = visited[current]
        path.reverse()
        return path
    return None

def dfs(maze, start, goal):
    stack = [start]
    visited = {start: None}
    failed_paths = []

    while stack:
        current = stack.pop()
        if current == goal:
            break

        moves = [(-1,0), (1,0), (0,1), (0,-1)]  # north, south, east, west
        for dr, dc in moves:
            r, c = current[0] + dr, current[1] + dc
            if 0 <= r < 15 and 0 <= c < 15 and maze[r, c] == 0 and (r, c) not in visited:
                visited[(r, c)] = current
                stack.append((r, c))
        else:
            failed_paths.append(current)

    path = []
    if goal in visited:
        current = goal
        while current:
            path.append(current)
            current = visited[current]
        path.reverse()
        return path, failed_paths
    return None, failed_paths

# Chọn thuật toán bạn muốn chạy ở đây (bfs hoặc dfs):
algorithm = 'dfs'  # Thay đổi 'bfs' hoặc 'dfs' để chạy thuật toán tương ứng

if algorithm == 'bfs':
    path = bfs(maze, start, goal)
else:
    path, failed_paths = dfs(maze, start, goal)

# Visualization
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(maze, cmap='gray_r', origin='upper')

if path:
    path_y, path_x = zip(*path)
    ax.plot(path_x, path_y, color='red', linewidth=2, label=f'{algorithm.upper()} Path')

if algorithm == 'dfs':
    failed_y, failed_x = zip(*failed_paths)
    ax.scatter(failed_x, failed_y, color='orange', s=20, label='DFS Failed Paths')

ax.scatter(start[1], start[0], c='green', s=100, label='Start (S)')
ax.scatter(goal[1], goal[0], c='blue', s=100, label='Goal (G)')

ax.set_xticks(np.arange(-0.5, 15, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 15, 1), minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

ax.legend()
ax.set_title(f'{algorithm.upper()} Maze Path')
plt.show()

# Output path coordinates
if path:
    print("Path found:", path)
else:
    print("No path found.")
