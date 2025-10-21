import sys
import random
import math

N, K = int(sys.argv[1]), int(sys.argv[2])
assert K >= 2

NA = int(N * 10 / 11)
NB = N - NA
assert NA + NB == N

# Assign group counts per cluster (evenly spread)
points_per_cluster = N // K
remainder = N % K
radius = 10

centers = [
    (radius * math.cos(2 * math.pi * i / K), radius * math.sin(2 * math.pi * i / K))
    for i in range(K)
]

total_A, total_B = 0, 0

for i, (cx, cy) in enumerate(centers):
    cluster_size = points_per_cluster + (1 if i < remainder else 0)

    nA = int(cluster_size * 0.95) if total_A + int(cluster_size * 0.95) <= NA else NA - total_A
    nB = cluster_size - nA


    if total_B + nB > NB:
        nB = NB - total_B
        nA = cluster_size - nB

    for _ in range(nA):
        x = random.gauss(cx, 0.8)
        y = random.gauss(cy, 0.8)
        print(f"{x:.4f},{y:.4f},A")
    for _ in range(nB):
        x = random.gauss(cx, 2.0)  # more spread out B → higher Φ_std
        y = random.gauss(cy, 2.0)
        print(f"{x:.4f},{y:.4f},B")

    total_A += nA
    total_B += nB
