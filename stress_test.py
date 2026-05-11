"""
stress_test.py v2 — Load tester avec collecte CPU/RAM serveur via /metrics
Usage :
    python stress_test.py --workers 10 --total 200 --output results.csv
"""

import requests
import numpy as np
import base64
import json
import random
import time
import csv
import argparse
import threading
from io import BytesIO
from PIL import Image
from datetime import datetime
from collections import deque

DEFAULT_URL      = "http://rnn.dokpoly.claveille.fr"
DEFAULT_WORKERS  = 5
DEFAULT_TOTAL    = 100
DEFAULT_OUTPUT   = "stress_results.csv"
METRICS_INTERVAL = 1.0
PRINT_EVERY      = 10

lock           = threading.Lock()
results        = []
server_metrics = []
req_count      = 0
err_count      = 0
start_global   = None
recent_rtt     = deque(maxlen=20)
stop_monitor   = threading.Event()


def array_to_base64(image_array):
    img = Image.fromarray((image_array * 255).astype(np.uint8), mode='L')
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def monitor_server(metrics_url):
    session = requests.Session()
    while not stop_monitor.is_set():
        ts = datetime.utcnow().isoformat()
        try:
            r = session.get(metrics_url, timeout=5)
            if r.status_code == 200:
                m = r.json()
                row = {
                    "timestamp":    ts,
                    "cpu_percent":  m.get("cpu_percent", ""),
                    "mem_percent":  m.get("mem_percent", ""),
                    "mem_used_mb":  m.get("mem_used_mb", ""),
                    "swap_percent": m.get("swap_percent", ""),
                    "load_avg_1m":  m.get("load_avg_1m", ""),
                }
                with lock:
                    server_metrics.append(row)
        except Exception:
            pass
        time.sleep(METRICS_INTERVAL)
    session.close()


def send_request(session, url, image_array, label):
    payload  = json.dumps({"image": array_to_base64(image_array)})
    headers  = {"Content-Type": "application/json"}
    ts_start = time.perf_counter()
    ts_wall  = datetime.utcnow().isoformat()
    status   = None
    error    = None

    try:
        r      = session.post(url, data=payload, headers=headers, timeout=30)
        rtt_ms = (time.perf_counter() - ts_start) * 1000
        status = r.status_code
    except requests.exceptions.Timeout:
        rtt_ms = (time.perf_counter() - ts_start) * 1000
        error  = "timeout"
    except Exception as e:
        rtt_ms = (time.perf_counter() - ts_start) * 1000
        error  = str(e)[:80]

    return {
        "timestamp":   ts_wall,
        "label":       int(label),
        "rtt_ms":      round(rtt_ms, 2),
        "status_code": status,
        "error":       error or "",
    }


def worker(worker_id, task_queue, predict_url, x_test, y_test):
    global req_count, err_count
    session = requests.Session()

    while True:
        with lock:
            if not task_queue:
                break
            idx = task_queue.pop()

        image = x_test[idx] / 255.0
        label = y_test[idx]
        row   = send_request(session, predict_url, image, label)
        row["worker"] = worker_id

        with lock:
            req_count += 1
            if row["error"] or (row["status_code"] and row["status_code"] >= 400):
                err_count += 1
            results.append(row)
            recent_rtt.append(row["rtt_ms"])
            if req_count % PRINT_EVERY == 0:
                _print_live_stats()

    session.close()


def _print_live_stats():
    elapsed  = time.perf_counter() - start_global
    rps      = req_count / elapsed if elapsed > 0 else 0
    avg_rtt  = np.mean(list(recent_rtt)) if recent_rtt else 0
    p95_rtt  = np.percentile(list(recent_rtt), 95) if len(recent_rtt) >= 5 else avg_rtt
    err_rate = err_count / req_count * 100 if req_count else 0

    cpu_str = ""
    if server_metrics:
        last = server_metrics[-1]
        if last["cpu_percent"] != "":
            cpu_str = f" | CPU {last['cpu_percent']:>5.1f}% RAM {last['mem_percent']:>5.1f}%"

    print(
        f"\r[{req_count:>5} req | {elapsed:>6.1f}s | "
        f"{rps:>5.1f} rps | avg {avg_rtt:>7.1f} ms | "
        f"p95 {p95_rtt:>7.1f} ms | err {err_rate:>4.1f}%{cpu_str}]",
        end="", flush=True
    )


def print_final_stats():
    rtts    = [r["rtt_ms"] for r in results]
    errors  = [r for r in results if r["error"] or (r["status_code"] and r["status_code"] >= 400)]
    elapsed = time.perf_counter() - start_global
    cpu_vals = [m["cpu_percent"] for m in server_metrics if m["cpu_percent"] != ""]
    mem_vals = [m["mem_percent"] for m in server_metrics if m["mem_percent"] != ""]

    print("\n\n" + "═" * 65)
    print("  RÉSULTATS FINAUX")
    print("═" * 65)
    print(f"  Requêtes totales  : {len(results)}")
    print(f"  Durée totale      : {elapsed:.2f} s")
    print(f"  Débit moyen       : {len(results)/elapsed:.2f} req/s")
    print(f"  Erreurs           : {len(errors)} ({len(errors)/len(results)*100:.1f}%)")
    print(f"  Latence min       : {np.min(rtts):.2f} ms")
    print(f"  Latence moy       : {np.mean(rtts):.2f} ms")
    print(f"  Latence médiane   : {np.median(rtts):.2f} ms")
    print(f"  Latence p95       : {np.percentile(rtts, 95):.2f} ms")
    print(f"  Latence p99       : {np.percentile(rtts, 99):.2f} ms")
    print(f"  Latence max       : {np.max(rtts):.2f} ms")
    if cpu_vals:
        print(f"  CPU moy (serveur) : {np.mean(cpu_vals):.1f}%")
        print(f"  CPU max (serveur) : {np.max(cpu_vals):.1f}%")
        print(f"  RAM moy (serveur) : {np.mean(mem_vals):.1f}%")
        print(f"  RAM max (serveur) : {np.max(mem_vals):.1f}%")
    print("═" * 65)


def save_csv(base_path):
    fields_req = ["timestamp", "worker", "label", "rtt_ms", "status_code", "error"]
    with open(base_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_req)
        w.writeheader()
        w.writerows(results)
    print(f"\n  Latences → {base_path}")

    if server_metrics:
        metrics_path = base_path.replace(".csv", "_server.csv")
        fields_srv = ["timestamp", "cpu_percent", "mem_percent", "mem_used_mb", "swap_percent", "load_avg_1m"]
        with open(metrics_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields_srv)
            w.writeheader()
            w.writerows(server_metrics)
        print(f"  Serveur   → {metrics_path}")
    else:
        print("  (aucune métrique serveur — /metrics disponible ?)")


def main():
    global start_global

    parser = argparse.ArgumentParser(description="Stress tester MNIST avec métriques serveur")
    parser.add_argument("--url",     default=DEFAULT_URL)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--total",   type=int, default=DEFAULT_TOTAL)
    parser.add_argument("--output",  default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    predict_url = f"{args.url}/predict"
    metrics_url = f"{args.url}/metrics"

    print(f"\n  Target    : {predict_url}")
    print(f"  Métriques : {metrics_url}")
    print(f"  Workers   : {args.workers}")
    print(f"  Requêtes  : {args.total}")
    print(f"  Sortie    : {args.output}\n")

    try:
        r = requests.get(metrics_url, timeout=3)
        print("  /metrics OK ✓" if r.status_code == 200 else f"  /metrics répond {r.status_code}")
    except Exception:
        print("  /metrics injoignable — collecte CPU désactivée (test continue)")

    print("  Chargement MNIST...", end=" ", flush=True)
    try:
        from tensorflow.keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
    except ImportError:
        data = np.load("mnist.npz")
        x_test, y_test = data["x_test"], data["y_test"]
    print(f"OK ({len(x_test)} images)")

    task_queue   = [random.randint(0, len(x_test) - 1) for _ in range(args.total)]
    start_global = time.perf_counter()

    mon = threading.Thread(target=monitor_server, args=(metrics_url,), daemon=True)
    mon.start()

    threads = []
    for wid in range(args.workers):
        t = threading.Thread(
            target=worker,
            args=(wid, task_queue, predict_url, x_test, y_test),
            daemon=True,
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    stop_monitor.set()
    mon.join(timeout=2)

    print_final_stats()
    save_csv(args.output)


if __name__ == "__main__":
    main()


# charge légère
# python stress_test.py --workers 5 --total 100

# charge sérieuse
# python stress_test.py --workers 20 --total 500 --output run_20w.csv

# marteau
# python stress_test.py --workers 50 --total 2000 --output run_50w.csv