import cv2
import numpy as np
import heapq
import matplotlib.pyplot as plt

# -----------------------------
# A* ALGORITHM
# -----------------------------
def heuristic(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def astar(grid, start, goal):
    rows, cols = grid.shape
    open_list = []
    heapq.heappush(open_list, (0, start))

    came_from = {}
    g_score = {start: 0}

    directions = [(0,1),(1,0),(0,-1),(-1,0)]

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1]

        for d in directions:
            neighbor = (current[0]+d[0], current[1]+d[1])

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor] == 1:
                    continue

                new_cost = g_score[current] + 1

                if neighbor not in g_score or new_cost < g_score[neighbor]:
                    g_score[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor, goal)
                    heapq.heappush(open_list, (priority, neighbor))
                    came_from[neighbor] = current

    return []

# -----------------------------
# LOAD YOLO
# -----------------------------
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

print("🚀 Autonomous Navigation Started")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape

    # YOLO detection
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416,416), (0,0,0), True)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            confidence = max(scores)

            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append((x, y, w, h))

                # Draw detection
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    # -----------------------------
    # CREATE GRID (10x10)
    # -----------------------------
    grid = np.zeros((10,10))

    for (x, y, w, h) in boxes:
        gx = int((x / width) * 10)
        gy = int((y / height) * 10)

        if 0 <= gy < 10 and 0 <= gx < 10:
            grid[gy][gx] = 1

    start = (0,0)
    goal = (9,9)

    path = astar(grid, start, goal)

    # -----------------------------
    # SHOW PATH (matplotlib)
    # -----------------------------
    plt.clf()
    plt.imshow(grid, cmap='gray_r')

    if path:
        x_coords = [p[1] for p in path]
        y_coords = [p[0] for p in path]
        plt.plot(x_coords, y_coords)

    plt.title("Autonomous Navigation")
    plt.pause(0.01)

    # Show camera
    cv2.imshow("YOLO Navigation", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
plt.close()