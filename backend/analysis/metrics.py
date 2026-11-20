from time import perf_counter
import tracemalloc


def measure_algorithm(function, *args) -> tuple[dict, dict]:
    tracemalloc.start()
    start_time = perf_counter()
    try:
        result = function(*args)
        elapsed_seconds = perf_counter() - start_time
        _, peak_bytes = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    metrics = {
        "elapsed_seconds": elapsed_seconds,
        "peak_memory_bytes": peak_bytes,
        "explored_nodes": result["explored_nodes"],
        "path_cost": result["cost"],
    }
    return result, metrics
