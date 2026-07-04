import threading
import time
from concurrent.futures import ThreadPoolExecutor

def fetch_data(task_id):
    print(f"Thread: Starting Task {task_id} on {threading.current_thread().name}")
    time.sleep(__import__('random').randint(1, 3))  # Simulates a blocking I/O operation
    print(f"Thread: Finished Task {task_id}")
    return f"Result {task_id}"

start_time = time.time()
# Create a pool of worker threads
with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(fetch_data, [1, 2, 3]))

print(f"Multi-threading completed in: {time.time() - start_time:.2f} seconds")
