import torch  # Import the PyTorch library for tensor operations
import time  # Import the time library to measure execution time

# Number of particles in the simulation
N = 300000

# Initialize positions and velocities of particles
# Each particle has an x and y coordinate, and an x and y velocity component
positions = torch.rand(N, 2)  # Randomly initialize x, y coordinates
velocities = torch.rand(N, 2)  # Randomly initialize x, y velocities

# Clone the positions and velocities to GPU for parallel processing
positions_gpu = positions.clone().to('cuda')
velocities_gpu = velocities.clone().to('cuda')

# Gravitational constant for the simulation
G = 1.0

# Time step for the simulation
dt = 0.01


# Function to update positions and velocities based on gravitational interactions
def update(positions, velocities, G, dt):
    # Loop through each particle to update its position and velocity
    for i in range(len(positions)):
        # Calculate the distance components between the i-th particle and all other particles
        dx = positions[i, 0] - positions[:, 0]
        dy = positions[i, 1] - positions[:, 1]

        # Calculate the actual distance
        distance = torch.sqrt(dx ** 2 + dy ** 2)

        # Calculate the gravitational force according to Newton's law of gravitation
        force = G / (distance + 1e-9) ** 2  # Added epsilon to avoid division by zero

        # Remove self-interaction by setting the force to zero for the particle itself
        force[force > 1] = 0

        # Calculate the acceleration components
        ax = torch.sum(force * dx / (distance + 1e-9))
        ay = torch.sum(force * dy / (distance + 1e-9))

        # Update the velocity of the i-th particle
        velocities[i, 0] += ax * dt
        velocities[i, 1] += ay * dt

    # Update the positions of all particles based on their velocities
    positions += velocities * dt


# Measure the time taken for the simulation on CPU
start_time_cpu = time.time()
update(positions, velocities, G, dt)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu

# Measure the time taken for the simulation on GPU
start_time_gpu = time.time()
update(positions_gpu, velocities_gpu, G, dt)
end_time_gpu = time.time()
gpu_time = end_time_gpu - start_time_gpu

# Print the time taken for both CPU and GPU simulations
print(f"Time taken on CPU: {cpu_time:.6f} seconds")
print(f"Time taken on GPU: {gpu_time:.6f} seconds")
