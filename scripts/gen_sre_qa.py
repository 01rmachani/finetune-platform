#!/usr/bin/env python3
"""
Build the SRE fine-tuning Q&A set, GROUNDED in the bundled observability tables.

Reads data/sre-tables/{predictions,kubernetes_events,alert_log}.csv and turns real
rows into {question, reference_answer, context} training examples, so the fine-tuned
model answers questions about the same pods/incidents the chat can show. Output:
data/sre-tables-train/sre_qa.jsonl  (also selectable in the Train dataset dropdown).

Usage:  python3 scripts/gen_sre_qa.py
"""
import csv, json, os, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "sre-tables")
OUTDIR = os.path.join(ROOT, "data", "sre-tables-train")
os.makedirs(OUTDIR, exist_ok=True)
rng = random.Random(42)

# Root cause + remediation + prevention per ML use case (the answer key).
REMEDIATION = {
    "Memory Leak Detection": ("a steadily climbing working-set with no plateau — a memory leak that will OOMKill the pod",
        "capture a heap profile, roll back the most recent deploy if the leak is new, and raise the memory limit as a stop-gap",
        "set memory requests/limits from p95+headroom, alert at 80% of limit, and add a startupProbe so slow boots aren't mistaken for crashes"),
    "OOM Risk Forecast": ("the container approaching its memory limit; an OOMKill (exit 137) is imminent",
        "raise limits.memory above observed peak and restart; if it recurs, profile for a leak",
        "size limits from real percentiles and load-test before release so the ceiling is known"),
    "Pod CrashLoop Prediction": ("repeated restarts (CrashLoopBackOff) — usually a failing dependency, bad config, or ungraceful shutdown after eviction",
        "check `kubectl logs --previous`, fix the missing config/dependency, and run >=2 replicas so a restart doesn't take the service down",
        "validate required env at boot, add a PodDisruptionBudget, and handle SIGTERM with a preStop hook"),
    "CPU Throttling": ("CFS throttling from a tight CPU limit; latency rises while average CPU looks low",
        "raise or remove the CPU limit (keep a realistic request) so the app can burst",
        "prefer CPU requests over tight limits for latency-sensitive services and alert on the throttled-periods ratio"),
    "Disk Pressure": ("node ephemeral-storage filling up (DiskPressure) — often verbose logs written to the container filesystem",
        "clear/rotate disk and move logs to stdout collected off-node",
        "set ephemeral-storage requests/limits and alert on node disk usage before the eviction threshold"),
    "Node NotReady Risk": ("the node's kubelet/PLEG going unhealthy under load — its pods will be evicted",
        "cordon and drain the node, let pods reschedule, then replace/reboot it",
        "spread replicas across nodes/zones with topology spread constraints and alert on node load + NotReady"),
    "Network Latency Spike": ("rising p99 latency and packet drops — a network or upstream-dependency regression",
        "check the dependency and node network; fail over if a single AZ is affected",
        "add retries with backoff, set sensible timeouts, and alert on p99 latency + drop rate"),
    "DNS Resolution Failures": ("CoreDNS degradation causing intermittent 'no such host'",
        "restart/repair CoreDNS and check its Corefile and resources",
        "run CoreDNS with >=2 replicas + node-local DNS cache and alert on DNS error rate"),
    "Replica Starvation": ("ready replicas below desired — the service is under-provisioned or pods aren't passing readiness",
        "scale up and fix whatever readiness reports unhealthy",
        "set HPA targets with headroom and alert when ready < desired for more than a minute"),
    "GC Pause Storms": ("clustered JVM GC pauses from heap pressure, stalling requests",
        "raise heap / tune GC and right-size the container memory",
        "monitor GC pause time and keep heap well below the container limit"),
    "Cert Expiry": ("an upcoming TLS certificate expiry that will break connections",
        "rotate the certificate before expiry",
        "automate cert rotation (cert-manager) and alert weeks ahead"),
    "Connection Pool Exhaustion": ("the DB/HTTP connection pool saturating — often all replicas restarting together after a drain",
        "size the pool correctly, add retry-with-backoff, and stagger restarts",
        "run >=2 replicas with a PDB and graceful shutdown so a reschedule doesn't stampede the pool"),
    "HTTP Error Surge": ("a spike in 5xx responses — a bad deploy or failing dependency",
        "roll back the recent deploy and check upstream health",
        "gate rollouts on error-rate SLOs and alert on 5xx anomalies"),
    "Cluster Capacity": ("the cluster running low on schedulable CPU/memory headroom",
        "free capacity or add nodes / enable cluster-autoscaler",
        "set requests accurately, autoscale with headroom, and alert on Pending pods"),
    "Security Policy Violation": ("unauthorized access or policy breaches detected",
        "investigate the source, revoke access, and apply the relevant NetworkPolicy/PSP",
        "enforce least-privilege RBAC + admission policies and alert on violations"),
}
RISK_HINT = {"critical":"act now","high":"act within the action window","medium":"monitor closely","low":"informational"}

def read(name):
    p = os.path.join(SRC, f"{name}.csv")
    with open(p) as f:
        return list(csv.DictReader(f))

def main():
    preds = read("predictions")
    rows = []
    # 1) grounded per-incident Q&A from real high-risk predictions (multiple angles each)
    QFORMS = [
        ("Our pod {pod} in namespace {ns} was flagged for '{uc}' at {risk} risk. What's the root cause and how do we remediate it?",
         "Root cause: {cause}. Severity: {risk} ({hint}). Remediation: {fix}. Preventive measure: {prevent}."),
        ("{pod} ({ns}) triggered a '{uc}' prediction. What should the on-call SRE do?",
         "This means {cause}. {hint_cap}. First: {fix}. Then prevent recurrence: {prevent}."),
        ("Why was {pod} flagged for '{uc}', and what's the preventive fix?",
         "Because {cause}. Remediate by: {fix}. Prevent it long-term: {prevent}."),
    ]
    for r in preds:
        uc = r["use_case_name"]; rem = REMEDIATION.get(uc)
        if not rem: continue
        if float(r["leak_probability"]) < 0.5: continue
        pod, ns, cl, risk = r["pod_name"], r["namespace"], r["cluster_name"], r["risk_level"]
        cause, fix, prevent = rem
        ctx = (f"pod={pod} namespace={ns} cluster={cl} use_case={uc} "
               f"risk={risk} leak_probability={r['leak_probability']} "
               f"time_to_impact_s={r.get('time_to_impact_seconds') or 'n/a'}")
        qf, af = rng.choice(QFORMS)
        rows.append({"question": qf.format(pod=pod, ns=ns, uc=uc, risk=risk),
                     "reference_answer": af.format(cause=cause, fix=fix, prevent=prevent,
                         risk=risk, hint=RISK_HINT.get(risk,"review"),
                         hint_cap=RISK_HINT.get(risk,"review").capitalize()),
                     "context": ctx})
        if len(rows) >= 1000: break

    # 2) general SRE knowledge Q&A (one per use case, several phrasings)
    phr = ["What does the '{uc}' signal mean and how do I respond?",
           "How should an SRE handle a '{uc}' alert?",
           "Explain '{uc}' and the standard remediation."]
    for uc,(cause,fix,prevent) in REMEDIATION.items():
        for t in phr:
            rows.append({"question": t.format(uc=uc),
                "reference_answer": f"It indicates {cause}. Remediation: {fix}. Prevention: {prevent}.",
                "context": f"use_case={uc}"})

    rng.shuffle(rows)
    out = os.path.join(OUTDIR, "sre_qa.jsonl")
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} Q&A rows -> {out}")

if __name__ == "__main__":
    main()
