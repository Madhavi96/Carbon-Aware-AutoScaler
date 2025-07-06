# coding=utf-8
import urllib.parse, requests, os, datetime, time, re
import pandas as pd, numpy as np

###############################################################################
# 1.  ──  CONFIG  (no services constant any more)  ─────────────────────────── #
###############################################################################
NAMESPACE  = "default"
INTERVAL   = 120                # seconds
METRICS    = ["pod", "vCPU", "cpu", "mem_", "mem", "res", "req"]

PROM_API       = "http://localhost:9090/api/v1/query?query="
PROM_API_ISTIO = "http://localhost:9091/api/v1/query?query="

# PromQL templates: {0}=namespace, {1}=regex for pod|destination_app, {2}=step
TEMPLATE = {
    # ---- cAdvisor / kubelet ------------------------------------------------
    "vCPU": ("sum by(pod)(rate(container_cpu_usage_seconds_total"
             "{{namespace='{0}',pod=~'{1}'}}[1m]))"),
    "cpu":  ("sum by(pod)(irate(container_cpu_usage_seconds_total"
             "{{namespace='{0}',pod=~'{1}'}}[1m])) "
             "/ sum by(pod)(container_spec_cpu_quota"
             "{{namespace='{0}',pod=~'{1}'}} "
             "/ container_spec_cpu_period{{namespace='{0}',pod=~'{1}'}})"),
    "mem_": ("sum by(pod)(container_memory_usage_bytes"
             "{{namespace='{0}',pod=~'{1}'}})"),
    "mem":  ("sum by(pod)(container_memory_usage_bytes"
             "{{namespace='{0}',pod=~'{1}'}}) "
             "/ sum by(pod)(container_spec_memory_limit_bytes"
             "{{namespace='{0}',pod=~'{1}'}})"),
    "pod":  ("count by(pod)(container_spec_cpu_period"
             "{{namespace='{0}',pod=~'{1}'}})"),
    # ---- Istio -------------------------------------------------------------
    "res":  ("sum by(destination_app)(rate("
             "istio_request_duration_milliseconds_sum"
             "{{reporter='destination',destination_workload_namespace='{0}',"
             "destination_app=~'{1}'}}[{2}])) "
             "/ sum by(destination_app)(rate("
             "istio_request_duration_milliseconds_count"
             "{{reporter='destination',destination_workload_namespace='{0}',"
             "destination_app=~'{1}'}}[{2}])) / 1000"),
    "req":  ("sum by(destination_app)(rate("
             "istio_requests_total"
             "{{destination_workload_namespace='{0}',"
             "destination_app=~'{1}'}}[{2}]))"),
}

###############################################################################
# 2.  ──  HELPERS  ─────────────────────────────────────────────────────────── #
###############################################################################
_svc_re = re.compile(r"^([a-z0-9-]+?)(?:-[0-9a-f]{4,}|-[0-9]+-[0-9a-f]{5,})?$").match

def _service_from_pod(pod: str, services: list[str]) -> str | None:
    """
    Return the service prefix this pod belongs to, or None.

    A pod matches a service if it is *exactly the same* or starts with
    "<service>-", which is how every Kubernetes replica-name looks.
    """
    for svc in services:                # the list you passed to save_all_fetch_data
        if pod == svc or pod.startswith(svc + "-"):
            return svc
    return None

def _prom_query(expr: str, ts: float, istio: bool = False) -> list[dict]:
    url = (PROM_API_ISTIO if istio else PROM_API) \
          + urllib.parse.quote_plus(expr) + f"&time={ts}"
    return requests.get(url, timeout=20).json()["data"]["result"]

###############################################################################
# 3.  ──  CORE  ────────────────────────────────────────────────────────────── #
###############################################################################
def _collect_metric(mode: str,
                    t0: datetime.datetime,
                    seconds: int,
                    interval: int,
                    services: list[str]) -> dict[str, list[float]]:
    """Query ONE metric for ALL requested services."""
    out = {svc: [] for svc in services}

    # 1. build regex identical to the original “svc.*” wildcard
    regex = "(" + "|".join(f"{svc}.*" for svc in services) + ")"

    # 2. ready-made PromQL
    expr = TEMPLATE[mode].format(NAMESPACE, regex, f"{interval}s")
    is_istio = mode in ("res", "req")

    # 3. loop over the timeline
    for offs in range(0, seconds, interval):
        ts   = t0 + datetime.timedelta(seconds=offs)
        vect = _prom_query(expr, time.mktime(ts.timetuple()), istio=is_istio)

        snap = {svc: 0.0 for svc in services}          # initialise with 0
        for item in vect:                              # Prometheus vector
            raw = item["value"][1]
            val = float(raw) if raw not in ("NaN", "+Inf", "-Inf") else 0.0

            # Which label holds the name we want?
            label = (item["metric"].get("destination_app")            # Istio metrics
                    if is_istio else
                    item["metric"].get("pod", ""))                   # cAdvisor/K8s

            svc = (label if is_istio else _service_from_pod(label, services))
            if svc in snap:                       # ignore pods we didn’t ask for
                snap[svc] += val                  # <── SUM per-prefix here

        # append (with zero-fill) to the output series
        for svc in services:
            out[svc].append(snap[svc])

    return out


def save_all_fetch_data(times: list[tuple[str, str | int]],
                        start_iter: int,
                        # root_dir: str,
                        interval: int = INTERVAL,
                        output_file: list[str] | None = None,
                        services: list[str] | None = None,
                        metrics: list[str] = METRICS) -> None:
    """High-level driver. *services* is now REQUIRED."""
    if not services:
        raise ValueError("Please pass a non-empty list of services.")
    # os.makedirs(root_dir, exist_ok=True)

    all_data = []

    for start_str, end_or_len in times:
        t0 = datetime.datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
        seconds = (
            int((datetime.datetime.strptime(end_or_len, "%Y-%m-%d %H:%M:%S") - t0).total_seconds())
            if isinstance(end_or_len, str) else int(end_or_len)
        )

        matrix = {m: _collect_metric(m, t0, seconds, interval, services)
                  for m in metrics}

        df = pd.DataFrame({
            f"{svc}_{m}": matrix[m][svc]
            for m in metrics
            for svc in services
        })
        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(f"{output_file}.csv", index=False)


###############################################################################
# 4.  ──  Loader (needs explicit *services* too)  ─────────────────────────── #
###############################################################################
def load_fetch_data(root_dir: str,
                    services: list[str],
                    start_iter: int = 1,
                    end_iter: int | None = None,
                    metrics: list[str] = METRICS) -> pd.DataFrame:
    if end_iter is None:
        end_iter = start_iter
    data = {svc: {m: [] for m in metrics} for svc in services}

    for it in range(start_iter, end_iter + 1):
        for svc in services:
            for m in metrics:
                with open(f"{root_dir}{it}_{svc}_{m}.log") as f:
                    data[svc][m] += [float(x) for x in f.read().splitlines()]

    array = [data[svc][m] for svc in services for m in metrics]
    cols  = [f"{svc}_{m}" for svc in services for m in metrics]
    return pd.DataFrame(np.array(array).T, columns=cols)

import json, sys

if __name__ == '__main__':
    services = json.loads(sys.argv[1])  # Parse the services array
    output_file_path = sys.argv[4]

    start_time = datetime.datetime.fromisoformat(sys.argv[2])  # Convert string to datetime object
    end_time = datetime.datetime.fromisoformat(sys.argv[3])  # Convert string to datetime object
    
    print(f"\nServices: {services}\n")

    times = [(start_time.strftime('%Y-%m-%d %H:%M:%S'), end_time.strftime('%Y-%m-%d %H:%M:%S'))]
    
    save_all_fetch_data(times, 
                        interval=60, 
                        start_iter=1,
                        services=services, 
                        output_file=output_file_path)
    print("Data collection completed.")
