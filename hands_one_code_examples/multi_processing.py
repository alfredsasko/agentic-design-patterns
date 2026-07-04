import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor

def fetch_data(task_id):
    print(f"Process: Starting Task {task_id} on Process ID {multiprocessing.current_process().pid}")
    time.sleep(__import__('random').randint(1, 3))  # Simulates heavy work or a blocking operation
    print(f"Process: Finished Task {task_id}")
    return f"Result {task_id}"

# Guard required for multiprocessing on many operating systems
if __name__ == '__main__':
    start_time = time.time()
    # Create a pool of separate processes
    with ProcessPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(fetch_data, [1, 2, 3]))

    print(f"Multi-processing completed in: {time.time() - start_time:.2f} seconds")
