import asyncio
import time

async def fetch_data(task_id):
    print(f"Async: Starting Task {task_id}")
    await asyncio.sleep(__import__('random').randint(1, 3))
    print(f"Async: Finished Task {task_id}")
    return f"Result {task_id}"

async def main():
    start_time = time.time()
    # Schedule all three tasks concurrently
    results = await asyncio.gather(fetch_data(1), fetch_data(2), fetch_data(3))
    print(f"Async completed in: {time.time() - start_time:.2f} seconds")

# Run the event loop
asyncio.run(main())
