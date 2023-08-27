---

## N-Body Simulation using PyTorch and CUDA

This code demonstrates an N-Body simulation using both the CPU and GPU. The simulation models the gravitational interactions between particles in a 2D space.

### Dependencies

- PyTorch
- time (Python Standard Library)

### Code Overview

1. **Import Libraries**: The code imports PyTorch for tensor operations and the time library for measuring execution time.

    ```python
    import torch
    import time
    ```

2. **Initialize Parameters**: The number of particles (`N`), their positions, and velocities are initialized. The gravitational constant (`G`) and time step (`dt`) for the simulation are also set.

    ```python
    N = 300000
    positions = torch.rand(N, 2)
    velocities = torch.rand(N, 2)
    G = 1.0
    dt = 0.01
    ```

3. **Data Transfer to GPU**: The positions and velocities are cloned to the GPU for parallel processing.

    ```python
    positions_gpu = positions.clone().to('cuda')
    velocities_gpu = velocities.clone().to('cuda')
    ```

4. **Update Function**: This function updates the positions and velocities of all particles based on their mutual gravitational interactions.

    ```python
    def update(positions, velocities, G, dt):
        ...
    ```

5. **Time Measurement**: The code measures and prints the time taken for the simulation to run on both the CPU and GPU.

    ```python
    start_time_cpu = time.time()
    update(positions, velocities, G, dt)
    end_time_cpu = time.time()
    cpu_time = end_time_cpu - start_time_cpu

    start_time_gpu = time.time()
    update(positions_gpu, velocities_gpu, G, dt)
    end_time_gpu = time.time()
    gpu_time = end_time_gpu - start_time_gpu

    print(f"Time taken on CPU: {cpu_time:.6f} seconds")
    print(f"Time taken on GPU: {gpu_time:.6f} seconds")
    ```

### How to Run

1. Install the required dependencies.
2. Run the script.

---
