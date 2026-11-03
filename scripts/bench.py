#!/usr/bin/env python3
"""Reproducible benchmark for the Freja llama-server (stdlib only).

Measures, per request:
  * TTFT (time to first token, client side, via streaming)
  * server-side prefill and decode speed (llama.cpp `timings`)
  * MTP draft acceptance rate
and reports the median over several runs after a warm-up. With --concurrency N
it fires N requests at once and also reports aggregate decode throughput, which
is what matters for the 3 parallel slots.

Examples:
  python3 bench.py                       # 1 warm-up + 5 measured runs
  python3 bench.py --concurrency 3       # load all parallel slots
  python3 bench.py --cache-prompt        # include prompt-cache reuse in TTFT
"""
import argparse
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

DEFAULT_PROMPT = '''Förklara i detalj hur en relationsdatabas indexerar data med B-träd och LSM-träd.
Beskriv skillnaderna i läs- och skrivprestanda, hur siddelningar (page splits) och komprimering fungerar,
samt varför LSM-träd ofta föredras i distribuerade databaser som Cassandra och RocksDB medan B-träd
dominerar i traditionella relationsdatabaser som PostgreSQL och MySQL.'''


def run_once(args, prompt, seed):
    payload = {
        'model': args.model,
        'messages': [{'role': 'user', 'content': prompt}],
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'seed': seed,
        'stream': True,
        'stream_options': {'include_usage': True},
        # Disable prompt-cache reuse by default so every run pays the full prefill
        # and TTFT stays comparable between runs.
        'cache_prompt': args.cache_prompt,
    }
    req = urllib.request.Request(
        args.url.rstrip('/') + '/v1/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
    )
    t0 = time.perf_counter()
    ttft = None
    timings, usage = {}, {}
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        for raw in resp:
            line = raw.decode('utf-8').strip()
            if not line.startswith('data:'):
                continue
            data = line[5:].strip()
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            if ttft is None:
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content') or delta.get('reasoning_content'):
                        ttft = time.perf_counter() - t0
                        break
            timings = chunk.get('timings') or timings
            usage = chunk.get('usage') or usage
    wall = time.perf_counter() - t0
    return {
        'wall': wall,
        'ttft': ttft if ttft is not None else wall,
        'prompt_n': timings.get('prompt_n', usage.get('prompt_tokens', 0)),
        'prompt_tps': timings.get('prompt_per_second', 0.0),
        'gen_n': timings.get('predicted_n', usage.get('completion_tokens', 0)),
        'gen_tps': timings.get('predicted_per_second', 0.0),
        'draft_n': timings.get('draft_n', 0),
        'draft_acc': timings.get('draft_n_accepted', 0),
    }


def run_round(args, prompt, round_idx):
    if args.concurrency == 1:
        t0 = time.perf_counter()
        res = [run_once(args, prompt, args.seed + round_idx)]
        return res, time.perf_counter() - t0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        t0 = time.perf_counter()
        futures = [pool.submit(run_once, args, prompt, args.seed + round_idx * 100 + i)
                   for i in range(args.concurrency)]
        res = [f.result() for f in futures]
        return res, time.perf_counter() - t0


def fmt_run(r):
    acc = f"{r['draft_acc']}/{r['draft_n']} ({100 * r['draft_acc'] / r['draft_n']:.0f}%)" if r['draft_n'] else 'n/a'
    return (f"TTFT {r['ttft'] * 1000:7.0f} ms | prefill {r['prompt_n']:>5} tok @ {r['prompt_tps']:7.1f} t/s | "
            f"decode {r['gen_n']:>4} tok @ {r['gen_tps']:6.2f} t/s | draft {acc} | wall {r['wall']:.2f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--url', default='http://127.0.0.1:11434')
    ap.add_argument('--model', default='gemma4:12b')
    ap.add_argument('--prompt-file', help='read prompt from file (e.g. a long document to test prefill)')
    ap.add_argument('--max-tokens', type=int, default=350)
    ap.add_argument('--temperature', type=float, default=0.7)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--runs', type=int, default=5)
    ap.add_argument('--warmup', type=int, default=1)
    ap.add_argument('--concurrency', type=int, default=1)
    ap.add_argument('--cache-prompt', action='store_true', help='allow server prompt-cache reuse')
    ap.add_argument('--timeout', type=float, default=600)
    args = ap.parse_args()

    prompt = open(args.prompt_file, encoding='utf-8').read() if args.prompt_file else DEFAULT_PROMPT

    for i in range(args.warmup):
        run_round(args, prompt, -1 - i)

    all_runs, aggregates = [], []
    for i in range(args.runs):
        res, wall = run_round(args, prompt, i)
        all_runs.extend(res)
        agg = sum(r['gen_n'] for r in res) / wall
        aggregates.append(agg)
        for r in res:
            print(f'run {i + 1}: {fmt_run(r)}')
        if args.concurrency > 1:
            print(f'run {i + 1}: aggregate decode {agg:.2f} t/s over {args.concurrency} concurrent requests')

    med = statistics.median
    draft_n = sum(r['draft_n'] for r in all_runs)
    draft_acc = sum(r['draft_acc'] for r in all_runs)
    print('-' * 100)
    print(f"median over {len(all_runs)} requests (concurrency {args.concurrency}, warm-up {args.warmup}):")
    print(f"  TTFT            {med(r['ttft'] for r in all_runs) * 1000:8.0f} ms")
    print(f"  prefill speed   {med(r['prompt_tps'] for r in all_runs):8.1f} t/s")
    print(f"  decode speed    {med(r['gen_tps'] for r in all_runs):8.2f} t/s per request "
          f"({1000 / max(med(r['gen_tps'] for r in all_runs), 1e-9):.2f} ms/token)")
    if args.concurrency > 1:
        print(f"  aggregate       {med(aggregates):8.2f} t/s total")
    if draft_n:
        print(f"  draft accepted  {100 * draft_acc / draft_n:8.1f} %  ({draft_acc}/{draft_n})")
    print(f"  wall per req    {med(r['wall'] for r in all_runs):8.2f} s")


if __name__ == '__main__':
    main()
