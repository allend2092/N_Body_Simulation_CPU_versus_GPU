import torch
import time
import matplotlib.pyplot as plt

# Number of particles
N = 1000

# Initialize positions and velocities
positions = torch.rand(N, 2)  # x, y coordinates
velocities = torch.rand(N, 2)  # x, y velocities

# Move data to GPU
positions_gpu = positions.clone().to('cuda')
velocities_gpu = velocities.clone().to('cuda')

# Gravitational constant
G = 1.0

# Time step
dt = 0.01

# Function to update positions and velocities (N-Body simulation)
def update(positions, velocities, G, dt):
    for i in range(len(positions)):
        dx = positions[i, 0] - positions[:, 0]
        dy = positions[i, 1] - positions[:, 1]
        distance = torch.sqrt(dx**2 + dy**2)
        force = G / (distance + 1e-9)**2
        force[force > 1] = 0  # Remove self-interaction
        ax = torch.sum(force * dx / (distance + 1e-9))
        ay = torch.sum(force * dy / (distance + 1e-9))
        velocities[i, 0] += ax * dt
        velocities[i, 1] += ay * dt
    positions += velocities * dt

# CPU Simulation and Visualization Loop
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter(positions[:, 0], positions[:, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

print("Running CPU Simulation...")
for i in range(100):
    update(positions, velocities, G, dt)
    sc.set_offsets(positions)
    plt.pause(0.1)

plt.ioff()
plt.show()

# GPU Simulation and Visualization Loop
plt.ion()
fig, ax = plt.subplots()
sc = ax.scatter(positions_gpu[:, 0].cpu(), positions_gpu[:, 1].cpu())
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

print("Running GPU Simulation...")
for i in range(100):
    update(positions_gpu, velocities_gpu, G, dt)
    sc.set_offsets(positions_gpu.cpu())
    plt.pause(0.1)

plt.ioff()
plt.show()
