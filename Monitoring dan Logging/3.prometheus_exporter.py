from prometheus_client import start_http_server, Summary
import random
import time

# Create a metric to track time spent
INFERENCE_TIME = Summary('inference_duration_seconds', 'Time spent on inference')

@INFERENCE_TIME.time()
def simulate_inference():
    time.sleep(random.uniform(0.1, 0.5))  # Simulasi waktu inferensipython 3.

if __name__ == '__main__':
    start_http_server(8000)
    print("Prometheus exporter running at http://localhost:8000")

    while True:
        simulate_inference()
