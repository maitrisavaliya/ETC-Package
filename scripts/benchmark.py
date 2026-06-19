#!/usr/bin/env python3
"""
scripts/benchmark.py
====================
Reproduces Tables 3-6 of the ETC paper (Johnson 2002 methodology):
fresh ExactReal instance per precision, minimum of 3 wall-clock runs.

Usage:  python scripts/benchmark.py
"""
from __future__ import annotations
import math, time
from fractions import Fraction
from etc import ExactReal, sign, benchmark_eval

def _section(title):
    print(); print("=" * 68); print(title); print("=" * 68)

def table3_pi():
    _section("TABLE 3: pi computation (8-1024 bits)")
    print(f"{'Precision':>16}  {'Time (ms)':>10}  {'Doubling exp.':>14}")
    print("-" * 46)
    results = benchmark_eval(ExactReal.pi(), [8,16,32,64,128,256,512,1024], repeat=3)
    prev_n = prev_t = None
    for n, t in results:
        exp = f"+{math.log(t/prev_t)/math.log(n/prev_n):.2f}" if prev_t and t > 0 else "---"
        print(f"{n:>16}  {t*1000:>10.3f}  {exp:>14}")
        prev_n, prev_t = n, t

def table4_transcendental():
    _section("TABLE 4: Transcendental constant timings (ms)")
    precs = [16,32,64,128,256,512]
    constants = [("pi  (Machin) ", ExactReal.pi()), ("e   (Taylor) ", ExactReal.e()),
                 ("ln2 (series) ", ExactReal.log2()), ("sqrt2 (Newton)", ExactReal.sqrt2())]
    hdr = f"{'Constant':<16}" + "".join(f"  {p:>6}b" for p in precs)
    print(hdr); print("-"*len(hdr))
    for name, func in constants:
        times = benchmark_eval(func, precs, repeat=3)
        print(f"{name:<16}" + "".join(f"  {t*1000:>6.3f}" for _,t in times))

def table5_arithmetic():
    _section("TABLE 5: Arithmetic operation timings (ms)")
    a = ExactReal.from_rational(77617); b = ExactReal.from_rational(33096)
    print(f"{'Precision':>16}  {'Add':>8}  {'Mul':>8}  {'Sqrt':>8}")
    print("-" * 46)
    for prec in [8,16,32,64,128,256,512]:
        t_a = min(time.perf_counter().__class__.__new__(time.perf_counter().__class__) or
                  (lambda: (lambda t0: time.perf_counter()-t0)(time.perf_counter()) or
                   (a.add(b).eval(prec), time.perf_counter()))[1]
                  for _ in range(3))
        def _t(fn):
            t0=time.perf_counter(); fn(); return time.perf_counter()-t0
        ta = min(_t(lambda: a.add(b).eval(prec)) for _ in range(3))
        tm = min(_t(lambda: a.mul(b).eval(prec)) for _ in range(3))
        ts = min(_t(lambda: a.sqrt().eval(prec)) for _ in range(3))
        print(f"{prec:>16}  {ta*1000:>8.3f}  {tm*1000:>8.3f}  {ts*1000:>8.3f}")

def table6_sign():
    _section("TABLE 6: Certified sign determination")
    cases = [("pi - 3", ExactReal.pi().sub(ExactReal.from_rational(3))),
             ("e - 2.7", ExactReal.e().sub(ExactReal.from_rational(Fraction(27,10)))),
             ("+1e-6",  ExactReal.from_rational(Fraction(1,1_000_000))),
             ("-1e-6",  ExactReal.from_rational(Fraction(-1,1_000_000)))]
    print(f"{'Expression':<14}  {'Sign':>5}  {'Time (ms)':>10}  {'Valid':>5}")
    print("-" * 40)
    for label, expr in cases:
        t0=time.perf_counter(); r=sign(expr); elapsed=time.perf_counter()-t0
        print(f"{label:<14}  {r.value:>+5}  {elapsed*1000:>10.3f}  {'yes' if r.is_valid() else 'NO':>5}")

if __name__ == "__main__":
    print("ETC Benchmark Suite -- Johnson (2002) methodology")
    table3_pi(); table4_transcendental(); table5_arithmetic(); table6_sign()
    print("\nBenchmark complete.")
