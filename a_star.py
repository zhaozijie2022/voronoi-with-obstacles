import numpy as np
import heapq


# def calculate_shortest_distance(start, goal, obstacle_map):
#     # Define possible movements (up, down, left, right)
#     movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]
#
#     # Get the dimensions of the obstacle map
#     height, width = obstacle_map.shape
#
#     # Create a priority queue (heap) for the open set
#     open_set = []
#     heapq.heappush(open_set, (0, start))
#
#     # Create a dictionary to store the cost from the start to each cell
#     cost_from_start = {start: 0}
#
#     # Create a dictionary to store the previous cell in the optimal path
#     previous = {}
#
#     while open_set:
#         # Pop the cell with the lowest cost from the open set
#         current_cost, current_cell = heapq.heappop(open_set)
#
#         # Check if we have reached the goal
#         if current_cell == goal:
#             break
#
#         # Explore neighbors
#         for movement in movements:
#             # Calculate the neighbor's coordinates
#             neighbor = current_cell[0] + movement[0], current_cell[1] + movement[1]
#
#             # Check if the neighbor is within the map boundaries
#             if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
#                 # Check if the neighbor is an obstacle
#                 if obstacle_map[neighbor] == 1:
#                     continue
#
#                 # Calculate the cost to reach the neighbor from the current cell
#                 neighbor_cost = cost_from_start[current_cell] + 1
#
#                 # Check if the neighbor has not been visited or if a shorter path has been found
#                 if neighbor not in cost_from_start or neighbor_cost < cost_from_start[neighbor]:
#                     # Update the cost and previous cell for the neighbor
#                     cost_from_start[neighbor] = neighbor_cost
#                     previous[neighbor] = current_cell
#
#                     # Calculate the heuristic (Manhattan distance) from the neighbor to the goal
#                     heuristic = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
#
#                     # Calculate the total cost (cost from start + heuristic) for the neighbor
#                     total_cost = neighbor_cost + heuristic
#
#                     # Add the neighbor to the open set
#                     heapq.heappush(open_set, (total_cost, neighbor))
#
#     # Reconstruct the optimal path
#     path = []
#     current = goal
#     while current in previous:
#         path.append(current)
#         current = previous[current]
#     path.append(start)
#     path.reverse()
#
#     return path


def get_navi_path(obstacle_map, start, goal, known_shortest_distance=None):
    # Define possible movements (up, down, left, right)
    movements = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Get the dimensions of the obstacle map
    height, width = obstacle_map.shape

    # Create a priority queue (heap) for the open set
    open_set = []
    heapq.heappush(open_set, (0, start))

    # Create a dictionary to store the cost from the start to each cell
    cost_from_start = {start: 0}

    # Create a dictionary to store the previous cell in the optimal path
    previous = {}

    while open_set:
        # Pop the cell with the lowest cost from the open set
        current_cost, current_cell = heapq.heappop(open_set)

        # Check if we have reached the goal
        if current_cell == goal:
            break

        # Check if the current path length exceeds known shortest distance K
        if known_shortest_distance is not None and current_cost > known_shortest_distance:
            return None  # Return None to indicate that no path within K exists

        # Explore neighbors
        for movement in movements:
            # Calculate the neighbor's coordinates
            neighbor = current_cell[0] + movement[0], current_cell[1] + movement[1]

            # Check if the neighbor is within the map boundaries
            if 0 <= neighbor[0] < height and 0 <= neighbor[1] < width:
                # Check if the neighbor is an obstacle
                if obstacle_map[neighbor] == 1:
                    continue

                # Calculate the cost to reach the neighbor from the current cell
                neighbor_cost = cost_from_start[current_cell] + 1

                # Check if the neighbor has not been visited or if a shorter path has been found
                if neighbor not in cost_from_start or neighbor_cost < cost_from_start[neighbor]:
                    # Update the cost and previous cell for the neighbor
                    cost_from_start[neighbor] = neighbor_cost
                    previous[neighbor] = current_cell

                    # Calculate the heuristic (Manhattan distance) from the neighbor to the goal
                    heuristic = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])

                    # Calculate the total cost (cost from start + heuristic) for the neighbor
                    total_cost = neighbor_cost + heuristic

                    # Add the neighbor to the open set
                    heapq.heappush(open_set, (total_cost, neighbor))

    # Reconstruct the optimal path
    if current_cell != goal:
        return None  # No path found
    path = []
    current = goal
    while current in previous:
        path.append(current)
        current = previous[current]
    path.append(start)
    path.reverse()

    return path
















