# Python Fluency Cheat Sheet

## 1. Data Structures

### List
```python
xs = [1, 2, 3]
xs.append(4)            # [1,2,3,4]
xs.extend([5, 6])       # [1,2,3,4,5,6]
xs.insert(0, 0)         # [0,1,2,3,4,5,6]
xs.pop()                # removes & returns last
xs.pop(0)               # removes & returns index 0
xs.remove(3)            # removes first occurrence of value 3
xs.index(2)             # first index of value 2
xs.count(2)             # number of occurrences
xs.reverse()            # in-place
xs.sort()               # in-place; sorted(xs) returns new list
len(xs); 3 in xs
```

### Dict
```python
d = {"a": 1, "b": 2}
d["c"] = 3
d.get("x", 0)           # default if missing
d.setdefault("y", [])   # set & return if missing
d.pop("a")              # remove & return
d.update({"e": 5})
list(d); list(d.keys()); list(d.values()); list(d.items())
{**d1, **d2}            # merge (d2 wins)
d1 | d2                 # 3.9+ merge
```

### Set
```python
s = {1, 2, 3}
s.add(4); s.discard(2)  # discard: no error if missing
s & t                   # intersection
s | t                   # union
s - t                   # difference
s ^ t                   # symmetric difference
s <= t                  # subset
frozenset([1, 2])       # hashable / immutable
```

### Tuple
```python
t = (1, 2, 3)
a, b, c = t             # unpacking
a, *rest = t            # rest = [2, 3]
a, b = b, a             # swap
```

## 2. Slicing

```python
xs[a:b]         # [a, b)
xs[a:b:step]
xs[::-1]        # reverse
xs[-3:]         # last 3
xs[:-3]         # all but last 3
xs[::2]         # every other
xs[:] = [1,2]   # replace contents in place
```

## 3. Comprehensions

```python
[x*x for x in xs]
[x for x in xs if x > 0]
[x*y for x in xs for y in ys]           # nested
{x: x*x for x in xs}                    # dict comp
{x for x in xs}                         # set comp
(x*x for x in xs)                       # generator (lazy)
[a if a > 0 else -a for a in xs]        # ternary inside comp
```

## 4. Iteration Idioms

```python
for i, v in enumerate(xs, start=0): ...
for a, b in zip(xs, ys): ...
for a, b in zip(xs, ys, strict=True): ...   # 3.10+ length check
for k, v in d.items(): ...
list(reversed(xs))
sorted(xs, key=lambda x: x[1], reverse=True)
sorted(items, key=lambda x: (-x.score, x.name))   # multi-key
any(x > 0 for x in xs); all(x > 0 for x in xs)
sum(x*x for x in xs)
min(xs, key=abs); max(d, key=d.get)
```

## 5. String

```python
s.split(",")            # ["a","b"]
",".join(parts)
s.strip(); s.lstrip(); s.rstrip()
s.replace("a", "b")
s.startswith("x"); s.endswith("y")
s.lower(); s.upper(); s.title()
s.find("x")             # -1 if missing
s.zfill(4)              # "0042"
"-".join(map(str, [1,2,3]))
f"{x:.3f}"; f"{x:,}"; f"{x:>10}"; f"{x:<10}"; f"{x:^10}"
f"{x:0>4}"              # "0042"
f"{name=}"              # "name='alice'" (3.8+)
```

## 6. Numbers / Math

```python
import math
math.inf; math.nan; math.isnan(x); math.isinf(x)
math.log(x); math.log2(x); math.log10(x); math.exp(x)
math.sqrt(x); math.pow(x, y); x ** 0.5
math.floor(x); math.ceil(x); round(x, 2)
divmod(7, 2)            # (3, 1)
abs(x); pow(x, y, mod)
```

## 7. Functions

```python
def f(a, b=1, *args, key=None, **kwargs): ...
f(1, 2, 3, 4, key="x", extra=5)

# lambda
sq = lambda x: x*x

# unpacking
f(*args, **kwargs)

# default mutable arg pitfall
def bad(xs=[]): xs.append(1); return xs   # shared!
def good(xs=None): xs = [] if xs is None else xs

# closures
def make_adder(n):
    def add(x): return x + n
    return add
```

## 8. Useful Built-ins

```python
range(start, stop, step)
list, tuple, set, dict, frozenset
map(f, xs); filter(pred, xs)
zip(xs, ys)
enumerate(xs, start)
sorted(xs, key=...); reversed(xs)
sum, min, max, any, all
abs, round, pow, divmod
len, type, isinstance(x, (int, float))
hasattr(o, "x"); getattr(o, "x", default); setattr(o, "x", v)
id, hash
print("a", "b", sep="|", end="\n")
```

## 9. collections

```python
from collections import Counter, defaultdict, deque, OrderedDict, namedtuple

c = Counter("aabbbc")            # {'b':3, 'a':2, 'c':1}
c.most_common(2)                 # [('b',3), ('a',2)]
c1 + c2; c1 - c2; c1 & c2; c1 | c2

dd = defaultdict(list)
dd["k"].append(1)                # no KeyError

dq = deque([1,2,3], maxlen=10)
dq.append(4); dq.appendleft(0)
dq.pop(); dq.popleft()           # O(1) both ends
dq.rotate(1)

Pt = namedtuple("Pt", ["x", "y"])
p = Pt(1, 2); p.x; p._asdict()
```

## 10. itertools

```python
from itertools import (
    chain, product, permutations, combinations,
    combinations_with_replacement, accumulate,
    groupby, count, cycle, repeat, islice, tee, starmap,
)

list(chain([1,2], [3,4]))                # [1,2,3,4]
list(product([1,2], [3,4]))              # [(1,3),(1,4),(2,3),(2,4)]
list(permutations([1,2,3], 2))           # all 2-perms
list(combinations([1,2,3], 2))           # [(1,2),(1,3),(2,3)]
list(accumulate([1,2,3,4]))              # [1,3,6,10]
list(accumulate([1,2,3,4], max))         # [1,2,3,4]
list(islice(count(0), 5))                # [0,1,2,3,4]
# groupby requires sorted input on key
for k, grp in groupby(sorted(xs, key=key), key=key): ...
```

## 11. functools

```python
from functools import reduce, lru_cache, cache, partial, cmp_to_key

reduce(lambda a, b: a + b, xs, 0)

@cache                                 # 3.9+, unbounded
def fib(n): return n if n < 2 else fib(n-1) + fib(n-2)

@lru_cache(maxsize=128)
def f(x): ...

f10 = partial(f, x=10)

sorted(xs, key=cmp_to_key(lambda a, b: a - b))
```

## 12. Control Flow

```python
# for/while-else: runs if loop didn't break
for x in xs:
    if cond(x): break
else:
    print("not found")

# walrus (3.8+)
while (line := f.readline()):
    process(line)

# match (3.10+)
match cmd:
    case "go": ...
    case "stop" | "halt": ...
    case [x, y, *rest]: ...
    case {"k": v}: ...
    case Point(x=0, y=0): ...
    case _: ...
```

## 13. Exceptions

```python
try:
    risky()
except (ValueError, KeyError) as e:
    handle(e)
except Exception:
    raise           # re-raise
else:
    print("no error")
finally:
    cleanup()

raise ValueError("msg") from prev_exc
assert cond, "message"
```

## 14. Files & Context Managers

```python
with open("f.txt", "r", encoding="utf-8") as f:
    for line in f: ...
    data = f.read()

with open("f.txt", "w") as f:
    f.write("hello\n")

# multiple context managers
with open("a") as a, open("b") as b: ...
```

## 15. Classes

```python
class Foo:
    cls_var = 0                         # class attr

    def __init__(self, x):
        self.x = x

    def method(self): ...

    @classmethod
    def from_str(cls, s): return cls(int(s))

    @staticmethod
    def util(x): return x + 1

    @property
    def doubled(self): return self.x * 2

    def __repr__(self): return f"Foo({self.x})"
    def __eq__(self, o): return isinstance(o, Foo) and self.x == o.x
    def __hash__(self): return hash(self.x)
    def __len__(self): return 1
    def __iter__(self): yield self.x

class Bar(Foo):
    def __init__(self, x, y):
        super().__init__(x)
        self.y = y

from dataclasses import dataclass, field
@dataclass(frozen=True)
class Point:
    x: int
    y: int = 0
    tags: list = field(default_factory=list)
```

## 16. Common Idioms

```python
# swap
a, b = b, a

# default to empty
xs = xs or []

# group by key
from collections import defaultdict
groups = defaultdict(list)
for x in xs: groups[key(x)].append(x)

# count items
from collections import Counter
Counter(xs)

# dedupe preserving order
list(dict.fromkeys(xs))

# flatten one level
[y for sub in xss for y in sub]

# transpose
list(zip(*matrix))

# chunk a list
[xs[i:i+n] for i in range(0, len(xs), n)]

# cumulative sum
from itertools import accumulate
list(accumulate(xs))

# argmax / argmin
i_max = max(range(len(xs)), key=lambda i: xs[i])

# top-k
import heapq
heapq.nlargest(k, xs)
heapq.nsmallest(k, xs, key=lambda x: x.score)
```

## 17. heapq (min-heap)

```python
import heapq
h = []
heapq.heappush(h, 3)
heapq.heappush(h, 1)
heapq.heappop(h)            # 1 (smallest)
h[0]                        # peek
heapq.heapify(xs)           # in place
heapq.nlargest(k, xs); heapq.nsmallest(k, xs)
# max-heap: push -x, pop -x
```

## 18. bisect (sorted list)

```python
import bisect
bisect.bisect_left(xs, v)   # leftmost insert pos
bisect.bisect_right(xs, v)  # rightmost insert pos
bisect.insort(xs, v)        # insert keeping sorted
```

## 19. Truthiness & Identity

```python
# falsy: 0, 0.0, "", [], {}, set(), None, False
if xs: ...                  # not empty
if x is None: ...           # identity check, not ==
a == b                      # value equality
a is b                      # same object

# chained comparison
if 0 < x < 10: ...
```

## 20. Typing (3.9+)

```python
from typing import Optional, Union, Callable, Iterable, Any

def f(x: int, y: Optional[str] = None) -> list[int]: ...
xs: list[int] = []
d: dict[str, int] = {}
cb: Callable[[int, int], int] = lambda a, b: a + b

# 3.10+
def g(x: int | None) -> list[int] | None: ...
```

## 21. Generators

```python
def gen(n):
    for i in range(n):
        yield i * i

g = gen(5)
next(g); next(g, default)

# yield from
def flatten(xss):
    for xs in xss:
        yield from xs

# generator expression
sum(x*x for x in xs)
```

## 22. Random / Misc

```python
import random
random.random()                 # [0, 1)
random.randint(a, b)            # [a, b]
random.uniform(a, b)
random.choice(xs)
random.choices(xs, k=3)         # with replacement
random.sample(xs, k=3)          # without replacement
random.shuffle(xs)              # in place
random.seed(42)

import time
time.time(); time.perf_counter()

import os, sys
os.path.join("a", "b"); os.environ.get("VAR")
sys.argv; sys.exit(1)
```

## 23. Decorators

A decorator is a callable that **takes a function and returns a (usually wrapped) function**. `@deco` above `def f` is just sugar for `f = deco(f)`.

### Decorator without arguments — 1 layer

The decorator itself is the wrapper factory: it receives `func` directly.

```python
from functools import wraps

def log_calls(func):                    # <- receives the function
    @wraps(func)                        # preserves name/docstring/signature
    def wrapper(*args, **kwargs):
        print(f"calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"  -> {result}")
        return result
    return wrapper                      # <- returns replacement function

@log_calls
def add(a, b):
    return a + b

add(2, 3)
# calling add
#   -> 5
```

Equivalent without `@`:
```python
def add(a, b): return a + b
add = log_calls(add)
```

### Decorator with arguments — 2 layers

You need an **extra outer layer** to capture the arguments. The outer call returns the real decorator.

```python
from functools import wraps

def repeat(n):                          # <- layer 1: takes deco args
    def decorator(func):                # <- layer 2: takes the function
        @wraps(func)
        def wrapper(*args, **kwargs):   # <- layer 3: takes call args
            for _ in range(n):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)                              # repeat(3) returns `decorator`
def greet(name):
    print(f"hi {name}")

greet("alice")     # prints 3 times
```

Equivalent without `@`:
```python
def greet(name): print(f"hi {name}")
greet = repeat(3)(greet)                # call once, then apply
```

### Why the extra layer?

| Form               | What `@deco` evaluates to | Layers |
|--------------------|---------------------------|--------|
| `@log_calls`       | `log_calls` (the deco)    | 1      |
| `@repeat(3)`       | `repeat(3)` → a deco      | 2      |

`@repeat(3)` runs `repeat(3)` first, which **must return a decorator** — i.e. a callable that accepts a function. That's why arg-decorators are always one nesting level deeper.

### Decorator that works with OR without arguments

A common pattern — detect whether the first arg is a function:

```python
def deco(func=None, *, prefix=">>"):
    if func is None:                    # called as @deco(prefix="!!")
        return lambda f: deco(f, prefix=prefix)
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(prefix, func.__name__)
        return func(*args, **kwargs)
    return wrapper

@deco
def f(): ...                            # works

@deco(prefix="!!")
def g(): ...                            # also works
```

### Stacking decorators

Applied **bottom-up** (closest to `def` runs first):

```python
@a
@b
@c
def f(): ...
# == f = a(b(c(f)))
```

### Class as decorator

Any callable works — a class with `__call__` can be a decorator and lets you keep state cleanly:

```python
class CountCalls:
    def __init__(self, func):
        self.func = func
        self.n = 0
        wraps(func)(self)               # copy metadata onto self
    def __call__(self, *a, **kw):
        self.n += 1
        return self.func(*a, **kw)

@CountCalls
def ping(): ...
ping(); ping()
ping.n          # 2
```

### Common gotchas

- **Forgetting `@wraps(func)`** → decorated function loses `__name__`, `__doc__`, and tracebacks get confusing.
- **`@deco` vs `@deco()`** — if the decorator takes no args, do **not** add `()`. If it does, you **must** add `()` (even if all args have defaults), unless using the dual-mode pattern above.
- **Decorators run at definition time**, not call time — `@repeat(3)` executes `repeat(3)` once when the module is imported.

## 24. Concurrency: threading vs multiprocessing vs asyncio

### The core constraint: the GIL

CPython has a **Global Interpreter Lock**: only one thread executes Python bytecode at a time. So:

- Multiple **threads** cannot run Python code in parallel — but the GIL **is released** during blocking I/O (`read`, `recv`, `sleep`) and inside many C extensions (NumPy, PyTorch, BLAS, `zlib`, `hashlib`).
- Multiple **processes** each have their own interpreter and own GIL → true parallelism, at the cost of memory and IPC overhead.
- **asyncio** is single-threaded cooperative multitasking — no parallelism at all, just efficient overlap of waits.

> Python 3.13+ has an experimental **free-threaded** build (no GIL). Until that's standard and your deps support it, assume the GIL.

### Decision tree

```
Is the work CPU-bound (number crunching, parsing, compression, ML)?
├── Yes → multiprocessing  (or release-GIL C extensions like NumPy)
└── No  → it's I/O-bound (network, disk, DB, subprocess)
        ├── Many concurrent ops (100s–10Ks) & you control the code
        │       → asyncio
        └── Few–moderate ops, or stuck with blocking libraries (requests, psycopg2-sync, …)
                → threading  (or concurrent.futures.ThreadPoolExecutor)
```

### Side-by-side

| Aspect                | threading             | multiprocessing                  | asyncio                          |
|-----------------------|-----------------------|----------------------------------|----------------------------------|
| Parallelism           | No (GIL)              | **Yes** (separate interpreters)  | No (single thread)               |
| Best for              | Blocking I/O          | CPU-bound work                   | High-concurrency I/O             |
| Memory                | Cheap (shared)        | Expensive (copied per process)   | Cheapest                         |
| Startup cost          | Low                   | High (fork/spawn + imports)      | Lowest                           |
| Data sharing          | Trivial (shared mem)  | IPC: `Queue`, `Pipe`, `shm`, pickle | Trivial (shared mem)          |
| Switch trigger        | OS preemption         | OS scheduler (true parallel)     | `await` (cooperative)            |
| Cancellation          | Hard (no kill)        | `proc.terminate()`               | `task.cancel()` (clean)          |
| Debugging             | Race conditions, locks| Pickling errors, zombie procs    | "coroutine never awaited" traps  |
| Library compatibility | Any                   | Any (fn must be picklable)       | Needs **async-aware** libs       |
| Scales to             | ~hundreds of threads  | ~num CPU cores                   | tens of thousands of tasks       |

### Code sketches

**Threading — blocking I/O fan-out**
```python
from concurrent.futures import ThreadPoolExecutor
import requests

with ThreadPoolExecutor(max_workers=32) as ex:
    results = list(ex.map(requests.get, urls))
```

**Multiprocessing — CPU-bound fan-out**
```python
from concurrent.futures import ProcessPoolExecutor

def heavy(x): return sum(i*i for i in range(x))

with ProcessPoolExecutor() as ex:        # defaults to os.cpu_count()
    results = list(ex.map(heavy, inputs))
```

**asyncio — many concurrent I/O ops**
```python
import asyncio, aiohttp

async def fetch(session, url):
    async with session.get(url) as r:
        return await r.text()

async def main(urls):
    async with aiohttp.ClientSession() as s:
        return await asyncio.gather(*(fetch(s, u) for u in urls))

asyncio.run(main(urls))
```

### Trade-off rules of thumb

1. **CPU-bound pure Python → `multiprocessing`.** Threads won't help (GIL); asyncio is irrelevant.
2. **CPU-bound but already in NumPy / PyTorch / BLAS → threads can help**, because those libs release the GIL. Don't reach for multiprocessing reflexively if you're already in vectorized C code.
3. **I/O-bound, blocking libraries → `threading`.** Pragmatic and requires no rewrite. Scales to a few hundred concurrent ops.
4. **I/O-bound, very high concurrency (10K connections, web scraper, proxy) → `asyncio`.** But every library in the call path must be async-aware — one blocking call (`time.sleep`, sync DB driver) stalls the whole event loop. Use `asyncio.to_thread(fn, ...)` to offload an unavoidable blocking call.
5. **Mixed CPU + I/O → combine.** e.g. an async server that offloads heavy work to a `ProcessPoolExecutor` via `loop.run_in_executor`.
6. **Don't pay for parallelism you don't need.** Process pools cost ~tens of MB per worker and seconds of startup; spinning one up for a 10ms task is a loss.

### Common pitfalls

- **`asyncio` + blocking call** — `time.sleep`, `requests.get`, sync file I/O inside a coroutine freezes the entire loop. Use `asyncio.sleep`, `aiohttp`, `aiofiles`, or `await asyncio.to_thread(blocking_fn, ...)`.
- **Mutable shared state across threads** — needs `threading.Lock`, `queue.Queue`, or atomic primitives. Race conditions are silent and intermittent.
- **`multiprocessing` and unpicklable args** — closures, lambdas, local functions, open file handles can't cross the process boundary. Top-level functions only.
- **Windows / macOS spawn semantics** — `multiprocessing` re-imports the module in each child, so always guard the entry point with `if __name__ == "__main__":`.
- **Forking after threads / CUDA / opened sockets** is unsafe. Prefer `multiprocessing.set_start_method("spawn")` when in doubt (default on Windows/macOS already).
- **Thread pool sizing** — for I/O, oversubscribe (`max_workers` ≫ cores) is fine; for CPU, cap at `os.cpu_count()`.

### TL;DR

> **CPU-bound → processes. I/O-bound + blocking libs → threads. I/O-bound + async libs + high concurrency → asyncio.**
> The GIL is the reason this question exists at all.

## 25. Generators & `yield` (deep dive)

### What a generator actually is

A function that contains `yield` is a **generator function**. Calling it does **not** run the body — it returns a **generator object** (an iterator). The body runs in chunks, pausing at each `yield` and resuming on each `next()`.

```python
def gen():
    print("start")
    yield 1
    print("middle")
    yield 2
    print("end")

g = gen()           # nothing prints — body hasn't run
next(g)             # prints "start", returns 1
next(g)             # prints "middle", returns 2
next(g)             # prints "end", raises StopIteration
```

Key mental model:
- A generator is a **resumable function** — local variables and the instruction pointer are frozen between `yield`s.
- It produces values **lazily, one at a time, on demand**.
- It's iterated with `for`, `next()`, or consumed by `list()`, `sum()`, `any()`, etc.

### Generator function vs generator expression

```python
# generator function
def squares(n):
    for i in range(n):
        yield i * i

# generator expression — same thing, terser
squares = (i * i for i in range(n))
```

Use a **generator expression** for one-liners; use a **generator function** when you need branches, state, or multiple `yield`s.

### Why use generators

| Benefit                | Why it matters                                                      |
|------------------------|---------------------------------------------------------------------|
| **Memory**             | Process a 10 GB file line-by-line without loading it into RAM       |
| **Lazy / infinite**    | Streams, infinite sequences (`itertools.count`), early termination  |
| **Composability**      | Pipelines: each stage pulls from the previous one                   |
| **Decouples produce/consume** | Producer doesn't need to know how many items consumer wants  |

### `list` vs generator — the canonical comparison

```python
# Eager: builds the whole list in memory, even if you only need the first match
def find_eager(rows):
    return [r for r in rows if expensive(r)]

first = find_eager(huge_rows)[0]      # processes everything

# Lazy: stops as soon as you do
def find_lazy(rows):
    for r in rows:
        if expensive(r):
            yield r

first = next(find_lazy(huge_rows))    # processes until first match
```

### `yield from` — delegate to a sub-iterable

```python
def flatten(xss):
    for xs in xss:
        yield from xs                 # equiv: for x in xs: yield x

list(flatten([[1,2], [3,4], [5]]))    # [1,2,3,4,5]
```

`yield from` also forwards `.send()`, `.throw()`, and the return value — important for coroutine-style generators and `asyncio` internals.

### Generators as **pipelines**

Each stage is a generator that consumes the previous one. Nothing is materialized until the final consumer asks.

```python
def read_lines(path):
    with open(path) as f:
        for line in f:
            yield line.rstrip()

def only_errors(lines):
    for line in lines:
        if "ERROR" in line:
            yield line

def parse(lines):
    for line in lines:
        ts, _, msg = line.partition(" ")
        yield (ts, msg)

for ts, msg in parse(only_errors(read_lines("app.log"))):
    print(ts, msg)
# Memory: O(1) regardless of file size.
```

### The generator protocol — under the hood

A generator object supports four operations:

```python
g = gen()
next(g)             # resume until next yield; returns yielded value
g.send(value)       # resume, and `yield` evaluates to `value` inside the generator
g.throw(ExcType)    # resume by raising ExcType at the yield point
g.close()           # raise GeneratorExit at the yield point (cleanup)
```

`send` lets the **caller push data into** the generator — this is how generators served as primitive coroutines before `async`/`await`:

```python
def echo():
    while True:
        x = yield                     # `yield` here is an expression
        print("got", x)

g = echo()
next(g)             # advance to the first yield (priming)
g.send("hi")        # prints "got hi"
g.send("there")     # prints "got there"
```

### Cleanup with `try/finally`

Generators run cleanup when garbage-collected or when `close()` is called (which raises `GeneratorExit` at the yield point):

```python
def reader(path):
    f = open(path)
    try:
        for line in f:
            yield line
    finally:
        f.close()                     # runs even if consumer breaks early
```

A `with` block inside a generator does the right thing automatically.

### `return` inside a generator

`return value` raises `StopIteration(value)`. The value is normally invisible, but `yield from` exposes it:

```python
def inner():
    yield 1
    yield 2
    return "done"

def outer():
    result = yield from inner()       # captures the return value
    print("inner returned:", result)
    yield 3

list(outer())
# prints "inner returned: done"
# returns [1, 2, 3]
```

### Common pitfalls

- **One-shot consumption.** A generator is exhausted after one pass — you can't iterate it twice. Wrap in `list(...)` if you need to reuse.
  ```python
  g = (x*x for x in range(3))
  list(g)        # [0, 1, 4]
  list(g)        # []   <-- already exhausted
  ```
- **Late binding in closures.** `gens = [(lambda: i) for i in range(3)]` — all return 2. Same trap inside generator expressions referring to outer mutable state.
- **`len()` doesn't work** on generators (no known length). Use `sum(1 for _ in g)` if you really need a count — but that consumes it.
- **Don't `return` a value expecting a normal caller to see it** — only `yield from` captures it; plain `for` loops drop it.
- **Exceptions inside generators** propagate to the consumer at the `next()` call where they happen, not at definition time. Stack traces can look weird because the generator's frame is paused.
- **Mixing `yield` and `return value` in async-style code** — in modern Python, prefer `async def` + `await` over hand-rolled coroutine generators.

### When NOT to use a generator

- You need **random access** (`xs[5]`) or `len()` → use a list.
- You'll iterate the data **multiple times** → materialize once, or write a class with `__iter__`.
- The dataset is **small and fits in memory** and you don't need composability → a list is simpler and faster per-item.
- You need to **share** the iteration across threads → generators aren't thread-safe; wrap with a lock or use a queue.

## 26. Context managers & `with`

### What `with` actually does

A context manager is any object that defines **`__enter__`** and **`__exit__`**. The `with` statement guarantees that `__exit__` runs even if the body raises — making it the standard tool for **paired setup/teardown** (open/close, lock/unlock, begin/commit, push/pop).

```python
with EXPR as VAR:
    BODY
```

is roughly equivalent to:

```python
mgr = EXPR
VAR = mgr.__enter__()
try:
    BODY
except:
    if not mgr.__exit__(*sys.exc_info()):
        raise
else:
    mgr.__exit__(None, None, None)
```

Key points:
- `__enter__` returns the value bound by `as` (often `self`, but not always — e.g. `open()` returns the file).
- `__exit__(exc_type, exc_val, tb)` receives `(None, None, None)` on a clean exit, or the exception triple if one was raised.
- If `__exit__` returns a **truthy** value, the exception is **suppressed**. Returning `None` / falsy lets it propagate (the usual case).

### Class-based context manager

```python
class Timer:
    def __enter__(self):
        import time
        self.t0 = time.perf_counter()
        return self                       # bound to `as t`
    def __exit__(self, exc_type, exc, tb):
        import time
        self.elapsed = time.perf_counter() - self.t0
        # return None → don't suppress exceptions
        return False

with Timer() as t:
    do_work()
print(t.elapsed)
```

Real-world examples that ship as classes:
```python
open("f.txt")                             # file handle
threading.Lock()                          # acquire / release
sqlite3.connect(":memory:")               # commit / rollback
torch.no_grad()                           # disable autograd
```

### Function-based with `contextlib.contextmanager`

For one-shot CMs, write a generator: code **before `yield`** is `__enter__`, code **after `yield`** is `__exit__`. The yielded value is what `as` binds.

```python
from contextlib import contextmanager

@contextmanager
def timer():
    import time
    t0 = time.perf_counter()
    try:
        yield                             # body of `with` runs here
    finally:
        print(f"elapsed: {time.perf_counter() - t0:.3f}s")

with timer():
    do_work()
```

The `try/finally` is important — without it, an exception in the body skips your teardown. Wrap the `yield` whenever cleanup must always run.

### Multiple managers in one `with`

```python
with open("in") as a, open("out", "w") as b:
    b.write(a.read())

# 3.10+: parenthesized form for long lists
with (
    open("a") as a,
    open("b") as b,
    open("c") as c,
):
    ...
```

Equivalent to nested `with` blocks; `__exit__`s run in **reverse** order (LIFO), even on exception.

### `contextlib` toolbox

```python
from contextlib import (
    contextmanager,        # generator → CM (above)
    closing,               # ensures .close() is called on anything with one
    suppress,              # silently ignore listed exceptions
    redirect_stdout,       # temporarily redirect sys.stdout
    redirect_stderr,
    nullcontext,           # no-op CM (handy as a default)
    ExitStack,             # dynamic / variable-number CMs
    AsyncExitStack,        # async version
)
```

#### `closing` — for objects with `.close()` but no CM protocol
```python
from contextlib import closing
from urllib.request import urlopen
with closing(urlopen("https://example.com")) as resp:
    data = resp.read()
```

#### `suppress` — clean replacement for `try/except: pass`
```python
from contextlib import suppress
with suppress(FileNotFoundError):
    os.remove("maybe_missing.tmp")
```

#### `redirect_stdout` — capture or silence prints
```python
import io
from contextlib import redirect_stdout
buf = io.StringIO()
with redirect_stdout(buf):
    print("captured")
buf.getvalue()                            # "captured\n"
```

#### `nullcontext` — conditional `with`
```python
from contextlib import nullcontext
cm = open(path) if path else nullcontext()
with cm as f:
    ...                                   # works either way
```
Common in tests / debug toggles, or when a switch decides whether locking is needed.

#### `ExitStack` — variable number of CMs
When the count of resources is decided at runtime (e.g. open N files):
```python
from contextlib import ExitStack
with ExitStack() as stack:
    files = [stack.enter_context(open(p)) for p in paths]
    # all files closed when block exits, in reverse order, even on error
```

`ExitStack` also supports:
- `stack.callback(fn, *args, **kw)` — register an arbitrary cleanup function
- `stack.push(cm)` — register an already-entered CM
- `stack.pop_all()` — transfer ownership (e.g. return resources from a factory)

### Class-based with `contextlib.ContextDecorator`

A CM that also works as a `@decorator`:

```python
from contextlib import ContextDecorator

class log_section(ContextDecorator):
    def __init__(self, name): self.name = name
    def __enter__(self): print(f"--- {self.name} ---"); return self
    def __exit__(self, *exc): print(f"--- end {self.name} ---")

with log_section("load"):
    load_data()

@log_section("train")                     # also valid — wraps the function call
def train(): ...
```

### Async context managers

Use `async with` and define `__aenter__` / `__aexit__`. `contextlib.asynccontextmanager` is the async sibling.

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def session():
    s = await open_session()
    try:
        yield s
    finally:
        await s.close()

async def main():
    async with session() as s:
        await s.fetch(...)
```

### Suppressing exceptions — the truthy-`__exit__` trick

```python
class IgnoreKeyError:
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, tb):
        return exc_type is KeyError       # True → suppress

with IgnoreKeyError():
    {}["missing"]                         # silently swallowed
print("still runs")
```

Use sparingly — `contextlib.suppress` is almost always cleaner.

### Common pitfalls

- **Forgetting `try/finally` in `@contextmanager`** — if the body raises, teardown after `yield` is skipped. Always wrap `yield` when cleanup is required.
- **`yield`ing more than once** in `@contextmanager` raises `RuntimeError`. A context manager is single-shot.
- **Assuming `as VAR` binds the manager** — it binds whatever `__enter__` returns. `open()` returns the file, but a custom CM might return `self`, a child resource, or nothing.
- **Returning truthy from `__exit__` by accident** — silently swallows exceptions. Return `None` (or `False`) unless you mean to suppress.
- **Reusing a `@contextmanager` instance** — it's a one-shot generator under the hood; call the factory each time.
- **Not using `ExitStack` for dynamic resource counts** — manually nesting `try/finally` for N resources gets ugly fast and is easy to leak on partial failure.

### When to reach for what

| Need                                        | Use                                |
|---------------------------------------------|------------------------------------|
| Quick paired setup/teardown                 | `@contextmanager` + `try/finally`  |
| Stateful resource that's reused             | Class with `__enter__`/`__exit__`  |
| Object has `.close()` but no CM protocol    | `closing(obj)`                     |
| Ignore a specific exception                 | `suppress(ExcType)`                |
| Capture / silence stdout                    | `redirect_stdout(buf)`             |
| Conditional `with`                          | `nullcontext()`                    |
| Variable / runtime number of resources      | `ExitStack`                        |
| Same code as both CM and decorator          | `ContextDecorator`                 |
| Async resources                             | `async with` + `asynccontextmanager` |

## 27. Iterators & the iterator protocol

### Iterable vs iterator — the crucial distinction

| Term         | Defines        | What it does                                    | Example                  |
|--------------|----------------|-------------------------------------------------|--------------------------|
| **Iterable** | `__iter__`     | Can produce a fresh iterator on demand          | `list`, `dict`, `str`, `range` |
| **Iterator** | `__next__` (and `__iter__` returning `self`) | Holds the cursor; produces values one at a time | a generator object, `iter(xs)` |

Rule of thumb: **iterables are reusable, iterators are one-shot.**

```python
xs = [1, 2, 3]                # iterable
it = iter(xs)                 # iterator over xs
next(it)                      # 1
next(it)                      # 2
next(it)                      # 3
next(it)                      # StopIteration

iter(xs) is iter(xs)          # False — each call returns a fresh iterator
iter(it)  is it               # True  — an iterator returns itself
```

### What `for` actually does

```python
for x in obj:
    BODY
```

is equivalent to:

```python
it = iter(obj)                # calls obj.__iter__()
while True:
    try:
        x = next(it)          # calls it.__next__()
    except StopIteration:
        break
    BODY
```

`StopIteration` is the **signal** that iteration is complete — it's not an error, just the protocol's "I'm done."

### The two-method protocol

```python
class Range:
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __iter__(self):                     # makes it iterable
        return RangeIter(self.start, self.stop)

class RangeIter:
    def __init__(self, cur, stop):
        self.cur = cur
        self.stop = stop

    def __iter__(self):                     # iterators must return self
        return self

    def __next__(self):                     # produces next value
        if self.cur >= self.stop:
            raise StopIteration
        v = self.cur
        self.cur += 1
        return v

r = Range(0, 3)
list(r)                                     # [0, 1, 2]
list(r)                                     # [0, 1, 2]   <-- reusable
```

The split into two classes is the **canonical pattern** when you want the container to be iterable multiple times in parallel.

### Self-iterator — when iterable IS its own iterator

If you only need single-use iteration, the same class can implement both:

```python
class Countdown:
    def __init__(self, n): self.n = n
    def __iter__(self): return self
    def __next__(self):
        if self.n <= 0: raise StopIteration
        self.n -= 1
        return self.n + 1

list(Countdown(3))     # [3, 2, 1]
# But: it's now exhausted — can't iterate again.
```

This is what generators do under the hood — `__iter__` returns `self`, `__next__` resumes the frame.

### Generators are iterators

Every generator function / generator expression produces an object that:
- Is its own iterator (`iter(g) is g`)
- Implements `__next__` (advances to the next `yield`)
- Raises `StopIteration` when the function returns

So the **easy way to write a custom iterator is to write a generator** — no class needed:

```python
class Range:
    def __init__(self, start, stop):
        self.start, self.stop = start, stop
    def __iter__(self):
        i = self.start
        while i < self.stop:
            yield i                          # generator handles state + StopIteration
            i += 1
```

Same behavior as the two-class version, half the code. Reach for the explicit `__next__` form only when you need fine control (e.g. peekable, resettable, or interop with non-Python code).

### `iter()` — three forms

```python
iter(iterable)                              # standard: calls __iter__
iter(callable, sentinel)                    # 2-arg: calls callable() until == sentinel
```

The 2-arg form is handy for streams:
```python
# read 4 KB chunks until the file returns b"" (EOF)
with open("big.bin", "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
        process(chunk)
```

### Useful iterator-shaped tools

```python
next(it)                  # advance once, raise if exhausted
next(it, default)         # advance once, return default if exhausted
iter([])                  # empty iterator

from itertools import islice, chain, tee
list(islice(it, 3))       # take first 3 (works on any iterator, no slicing required)
list(islice(it, 5, 10))   # skip 5, take 5
a, b = tee(it, 2)         # split one iterator into two independent ones
                          # WARNING: tee buffers — if a races far ahead of b, memory grows
```

### Peekable / lookahead — a common interview pattern

Iterators don't support peek. Wrap to add it:

```python
class Peekable:
    _SENTINEL = object()
    def __init__(self, iterable):
        self._it = iter(iterable)
        self._cache = self._SENTINEL
    def peek(self, default=_SENTINEL):
        if self._cache is self._SENTINEL:
            try: self._cache = next(self._it)
            except StopIteration:
                if default is self._SENTINEL: raise
                return default
        return self._cache
    def __iter__(self): return self
    def __next__(self):
        if self._cache is not self._SENTINEL:
            v, self._cache = self._cache, self._SENTINEL
            return v
        return next(self._it)

p = Peekable([1, 2, 3])
p.peek()       # 1
next(p)        # 1
next(p)        # 2
```

### Common pitfalls

- **`for x in it` exhausts an iterator.** A second `for` over the same iterator yields nothing. Iterate the **iterable** if you need multiple passes.
- **`iter(iterable)` returns a fresh cursor each time; `iter(iterator)` returns the same one.** That asymmetry is what enables nested loops over a list (`for a in xs: for b in xs: ...`) but breaks them over a generator.
- **`StopIteration` leaking from generator code.** Inside a generator, an unhandled `StopIteration` raised by `next(some_other_iter)` is converted to `RuntimeError` (PEP 479). Catch it explicitly or use a `for` loop.
- **Slicing doesn't work on iterators** — `it[2:5]` raises `TypeError`. Use `itertools.islice(it, 2, 5)`.
- **`len()` doesn't work on iterators.** Either materialize (`len(list(it))` — consumes!) or count via `sum(1 for _ in it)` (also consumes).
- **Forgetting `__iter__` on an iterator class.** Without it, `for x in my_iter:` fails because `for` calls `iter()` first. Always add `def __iter__(self): return self`.
- **`tee` memory blow-up.** If one branch consumes much faster than the other, the buffer between them grows unbounded. Don't use `tee` when branches diverge significantly.
- **Mutating an iterable while iterating it.** Modifying `list`/`dict`/`set` mid-iteration raises or silently misbehaves. Iterate over a copy (`list(d)`) if you need to mutate.

### `__iter__` vs `__getitem__` — the legacy fallback

Older Python (and still today, as a fallback) treats any class implementing `__getitem__(i)` as iterable: `for` calls `obj[0]`, `obj[1]`, … until `IndexError`.

```python
class Old:
    def __getitem__(self, i):
        if i >= 3: raise IndexError
        return i * 10

list(Old())     # [0, 10, 20]
```

Prefer `__iter__` for new code — it's the explicit protocol and supports non-integer keys without confusion.

### Generator vs iterator class — when to pick which

| Situation                                                | Pick                |
|----------------------------------------------------------|---------------------|
| Just want a sequence of values, possibly stateful        | **Generator function** |
| Need to attach methods (`peek`, `reset`, `close`) or extra state to the iterator object | **Iterator class** |
| Container is iterable multiple times in parallel         | **Container with `__iter__` returning a fresh iterator** (generator inside is fine) |
| Wrapping a callable / file / socket via sentinel         | `iter(callable, sentinel)` |
| Composing/transforming existing iterables                | **Generator expression / `itertools`** |

## 28. Metaclasses & `__init_subclass__`

### Metaclasses in one sentence

A class is an **object**, and its **type** is its metaclass. Just as `instance.__class__` is a class, `class.__class__` is a metaclass — and the default metaclass is `type`.

```python
class Foo: pass

type(Foo())     # <class 'Foo'>      — Foo is the type of its instances
type(Foo)       # <class 'type'>     — type is the metaclass of Foo
type(type)      # <class 'type'>     — type is its own type (the bedrock)
```

So `type` is overloaded: it both **inspects** an object's class and **creates** new classes. The 3-arg form is the class constructor:

```python
# These two are equivalent:
class Foo:
    x = 1
    def hi(self): return "hi"

Foo = type("Foo", (), {"x": 1, "hi": lambda self: "hi"})
```

### What a metaclass is for

A metaclass customizes **class creation** — code that runs when the `class` statement executes, not when an instance is created. Use cases:

- **Registries** — auto-collect every subclass into a registry.
- **Validation** — enforce that subclasses define required methods/attributes.
- **Auto-injection** — add methods, generated attributes, or `__slots__`.
- **Frameworks** — Django models, SQLAlchemy declarative base, ABCs, enum, dataclasses (sort of), Pydantic v1.

If you don't need to customize class creation itself, **don't reach for a metaclass.** Most of the time `__init_subclass__`, decorators, or descriptors are simpler and sufficient.

### Defining a metaclass

```python
class Meta(type):
    def __new__(mcls, name, bases, namespace, **kwargs):
        # runs BEFORE the class is created — can rewrite namespace
        namespace["created_by"] = "Meta"
        cls = super().__new__(mcls, name, bases, namespace)
        return cls

    def __init__(cls, name, bases, namespace, **kwargs):
        # runs AFTER the class is created
        super().__init__(name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        # runs when the CLASS is called (i.e. instance creation)
        print(f"creating instance of {cls.__name__}")
        return super().__call__(*args, **kwargs)

class Foo(metaclass=Meta):
    pass

Foo.created_by      # "Meta"
Foo()               # prints "creating instance of Foo"
```

The four hooks, in order of when they fire:

| Hook                  | Fires when                  | Receives                        |
|-----------------------|-----------------------------|---------------------------------|
| `Meta.__prepare__`    | Before class body runs      | Returns the namespace dict (e.g. `OrderedDict`) |
| `Meta.__new__`        | Class object is created     | `(mcls, name, bases, namespace)` |
| `Meta.__init__`       | Class object is initialized | Same as `__new__`               |
| `Meta.__call__`       | The class itself is called  | `(cls, *args, **kw)` — controls instance creation |

### `__init_subclass__` — the lightweight alternative (3.6+)

90% of metaclass use cases are subclass customization. PEP 487 added `__init_subclass__` as a **classmethod hook on the parent** that runs every time a subclass is created — no metaclass required.

```python
class Plugin:
    registry = {}

    def __init_subclass__(cls, *, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        key = name or cls.__name__.lower()
        Plugin.registry[key] = cls

class Csv(Plugin, name="csv"): ...        # auto-registered as "csv"
class Json(Plugin): ...                   # auto-registered as "json"

Plugin.registry      # {"csv": Csv, "json": Json}
```

Notes:
- It is **implicitly a classmethod**, even without the decorator.
- Keyword args in the `class` line (`class Csv(Plugin, name="csv")`) flow into `__init_subclass__` as kwargs.
- It does **not** fire for the base class itself, only for subclasses.
- Always call `super().__init_subclass__(**kwargs)` so cooperative chains work.

### Side-by-side: registry pattern

**Metaclass version**
```python
class RegistryMeta(type):
    registry = {}
    def __init__(cls, name, bases, ns):
        super().__init__(name, bases, ns)
        if bases:                         # skip the base
            RegistryMeta.registry[name] = cls

class Plugin(metaclass=RegistryMeta): pass
class A(Plugin): pass
class B(Plugin): pass
```

**`__init_subclass__` version** — same effect, no new metaclass:
```python
class Plugin:
    registry = {}
    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        Plugin.registry[cls.__name__] = cls
```

The second is shorter, easier to read, and composes better with other libraries.

### When you actually need a metaclass

Reach for a metaclass only when `__init_subclass__` and class decorators **can't** do it:

- You need to **change the namespace before the class body executes** — e.g. provide an `OrderedDict` so attribute order is preserved (pre-3.7), or restrict allowed names. Use `__prepare__`.
- You need to control **instance creation behavior across an entire hierarchy** by overriding `__call__` (singletons, caching, factory routing).
- You need **custom isinstance / issubclass behavior** — override `__instancecheck__` / `__subclasscheck__` (this is how `abc.ABCMeta` works).
- You're building a framework where users **shouldn't** have to inherit from a magic base class — but even then, prefer a class decorator first.

### Class decorator — usually the right answer

A class decorator runs **after** the class is built, can mutate it freely, and stacks cleanly with others:

```python
def register(cls):
    REGISTRY[cls.__name__] = cls
    return cls

@register
class A: ...
```

Decision order: **class decorator → `__init_subclass__` → metaclass**. Only escalate when the previous tier can't express what you need.

### `ABCMeta` — the canonical metaclass in the stdlib

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self): ...

class Dog(Animal):
    def speak(self): return "woof"

Animal()   # TypeError: Can't instantiate abstract class
Dog()      # ok
```

`ABC` is just `class ABC(metaclass=ABCMeta)`. The metaclass enforces that any class with unimplemented `@abstractmethod` cannot be instantiated, and adds `register()` for virtual subclasses.

### Common pitfalls

- **Metaclass conflicts in multiple inheritance.** If two bases have different metaclasses, Python raises `TypeError: metaclass conflict`. The combined metaclass must be a subclass of both. This is one of the strongest arguments for avoiding metaclasses in libraries.
- **Forgetting `super().__init_subclass__(**kwargs)`.** Breaks cooperative chains — sibling mixins silently don't run.
- **Putting validation in `__new__` when `__init_subclass__` would do** — the latter is simpler, doesn't poison the metaclass, and composes.
- **Trying to use `self` in a metaclass method.** Metaclass methods receive `cls` (the class being defined), not an instance — `cls` IS the "instance" from the metaclass's perspective.
- **Confusing `Meta.__call__` with `cls.__call__`.** `Meta.__call__(cls, ...)` runs when you write `Foo(...)` and orchestrates `__new__` + `__init__`. `Foo.__call__(self, ...)` runs when you call an *instance*.
- **Reading `__init_subclass__` as a regular method** — it's an implicit classmethod. Don't decorate it; don't call it on instances.
- **Assuming order of hooks.** For a class with a metaclass `Meta` and a base with `__init_subclass__`: order is `Meta.__prepare__` → body executes → `Meta.__new__` → `Meta.__init__` → `base.__init_subclass__`.

### TL;DR

> A metaclass is the class of a class. You almost never need one — `__init_subclass__` (subclass hook), class decorators (post-build mutation), and descriptors (per-attribute behavior) cover the vast majority of "I want something to happen when a class is defined" use cases. Reach for a metaclass only when you must intercept class **creation** itself.

## 29. Descriptors & `__get__` / `__set__`

### What a descriptor is

A **descriptor** is any object that defines at least one of `__get__`, `__set__`, `__delete__`. When such an object is stored as a **class attribute** (not an instance attribute), Python intercepts attribute access on instances and routes it through the descriptor's methods.

This is the underlying machinery behind `property`, `classmethod`, `staticmethod`, `super()`, and bound methods themselves.

```python
class Quiet:
    def __get__(self, instance, owner):
        return 42

class C:
    x = Quiet()                 # class attribute → descriptor

C().x      # 42  — Python called Quiet().__get__(c_instance, C)
```

### The descriptor protocol

| Method                                | Fires when                  | Signature                            |
|---------------------------------------|-----------------------------|--------------------------------------|
| `__get__(self, instance, owner)`      | `obj.attr` or `Cls.attr`    | `instance` is `None` for class access |
| `__set__(self, instance, value)`      | `obj.attr = value`          | —                                    |
| `__delete__(self, instance)`          | `del obj.attr`              | —                                    |
| `__set_name__(self, owner, name)`     | At class creation (3.6+)    | Tells the descriptor its attribute name |

### Data vs non-data descriptors — the lookup rule

This distinction governs **which wins** in attribute lookup, instance `__dict__` or class-level descriptor:

| Kind                | Defines                              | Priority                             |
|---------------------|--------------------------------------|--------------------------------------|
| **Data descriptor** | `__set__` (and/or `__delete__`)      | **Wins over** instance `__dict__`    |
| **Non-data**        | only `__get__`                       | **Loses to** instance `__dict__`     |

```python
obj.attr   →
    1. type(obj).__mro__ for a *data* descriptor named "attr"  → use it
    2. obj.__dict__["attr"]                                    → use it
    3. type(obj).__mro__ for a *non-data* descriptor or plain class attr  → use it
    4. AttributeError (or __getattr__)
```

This is why `property` (data descriptor) can't be shadowed by setting `obj.x = ...`, but methods (non-data) **can** be — assigning `obj.method = something` overrides the class method on that one instance.

### `__set_name__` — descriptor learns its attribute name

Without it, a descriptor doesn't know what name it was assigned to. With it (PEP 487, 3.6+), the descriptor can store data on the instance under a per-attribute key:

```python
class Typed:
    def __init__(self, expected):
        self.expected = expected

    def __set_name__(self, owner, name):
        self.name = name                                # remembers attribute name
        self.private = f"_{name}"

    def __get__(self, instance, owner):
        if instance is None:
            return self                                 # class access: return the descriptor itself
        return instance.__dict__[self.private]

    def __set__(self, instance, value):
        if not isinstance(value, self.expected):
            raise TypeError(f"{self.name} must be {self.expected.__name__}")
        instance.__dict__[self.private] = value

class Person:
    name = Typed(str)
    age  = Typed(int)

p = Person()
p.name = "alice"      # ok
p.age = "30"          # TypeError: age must be int
```

The `if instance is None: return self` branch is the convention for class-level access (`Person.name`) so introspection works.

### `property` is just a descriptor

`@property` is a built-in data descriptor. The hand-rolled equivalent:

```python
class Property:
    def __init__(self, fget=None, fset=None):
        self.fget, self.fset = fget, fset
    def __get__(self, instance, owner):
        if instance is None: return self
        return self.fget(instance)
    def __set__(self, instance, value):
        if self.fset is None: raise AttributeError("read-only")
        self.fset(instance, value)
    def setter(self, fset):
        return type(self)(self.fget, fset)

class C:
    @Property
    def x(self): return self._x
    @x.setter
    def x(self, v): self._x = v
```

Real `property` adds `__delete__`, docstring forwarding, and `fdel`, but the shape is identical.

### Methods are non-data descriptors

A function is a descriptor — its `__get__` returns a **bound method**:

```python
class C:
    def hi(self): return "hi"

C.hi              # <function C.hi>      — unbound, the function itself
C().hi            # <bound method C.hi>  — produced via function.__get__(instance, C)
C().hi()          # "hi"
```

`classmethod` and `staticmethod` are also descriptors — their `__get__` returns a method bound to the class, or the raw function, respectively.

### Common patterns

#### Validation / type-checked attributes
The `Typed` example above. Used by libraries like attrs, Pydantic v1, SQLAlchemy columns.

#### Lazy / cached attribute (compute once, store on instance)

```python
class lazy_property:
    def __init__(self, func):
        self.func = func
    def __set_name__(self, owner, name):
        self.name = name
    def __get__(self, instance, owner):
        if instance is None: return self
        value = self.func(instance)
        instance.__dict__[self.name] = value         # shadows descriptor next time
        return value

class Doc:
    @lazy_property
    def parsed(self):
        print("computing...")
        return expensive_parse(self.raw)

d = Doc()
d.parsed     # prints "computing...", caches result
d.parsed     # cache hit, no print
```

This works **only because `lazy_property` is a non-data descriptor** (no `__set__`) — the cached value in `instance.__dict__` shadows it on subsequent reads. The stdlib ships this as `functools.cached_property` (3.8+).

#### Read-only attribute
Define `__set__` to always raise:
```python
class ReadOnly:
    def __init__(self, value): self.value = value
    def __get__(self, inst, owner): return self.value
    def __set__(self, inst, value): raise AttributeError("read-only")
```

#### Per-instance storage without `__set_name__`
Pre-3.6 (or for compatibility), descriptors used a `WeakKeyDictionary` keyed by instance:
```python
import weakref
class Attr:
    def __init__(self): self._data = weakref.WeakKeyDictionary()
    def __get__(self, inst, owner):
        return self if inst is None else self._data.get(inst)
    def __set__(self, inst, value):
        self._data[inst] = value
```
Modern code prefers `__set_name__` + `instance.__dict__`.

### Where descriptors live

Descriptors **only fire when stored as class attributes**. Putting one in `instance.__dict__` does nothing special:

```python
class C: pass
c = C()
c.x = property(lambda self: 42)     # NOT a descriptor here
c.x                                  # <property object>, not 42
```

This is why properties must be defined on the class, not assigned per-instance.

### `__getattr__` / `__getattribute__` / descriptor — order of operations

`object.__getattribute__` is what implements the lookup rule above. Override it only if you really need to (rare). `__getattr__` is the **fallback** — it's called only when normal lookup (including descriptors) fails with `AttributeError`.

```
obj.attr
   ↓
__getattribute__  (handles descriptors, __dict__, MRO)
   ↓ raises AttributeError?
__getattr__       (fallback hook)
```

### Common pitfalls

- **Storing per-instance state on the descriptor itself.** A descriptor is a single object shared by all instances of the owner class — `self.value = v` inside `__set__` overwrites for everyone. Always store on `instance.__dict__` (use `__set_name__`) or a per-instance keyed structure.
- **Forgetting `if instance is None: return self`.** Without it, `Cls.attr` calls `__get__(None, Cls)` and behavior is undefined / crashes. The convention is to return the descriptor for class access so `inspect`/`dir` work.
- **Confusing data vs non-data priority.** Adding `__set__` flips the priority — your "lazy" cache that stored into `instance.__dict__` will silently stop caching the moment you add a `__set__`.
- **Defining descriptors as instance attributes.** They only work as **class** attributes. `self.attr = MyDescriptor()` in `__init__` does not engage the protocol.
- **Naming clashes between descriptor key and attribute.** If `__set_name__` stores under the same key (`instance.__dict__[self.name]`) AND the descriptor is non-data, the instance dict wins on subsequent reads and the descriptor never fires again. Use a different key (`f"_{name}"`) or make it a data descriptor.
- **Using a mutable default keyed by instance.** `WeakKeyDictionary` only works if the instance is weak-referenceable (most user classes are; some built-ins and `__slots__` classes without `__weakref__` are not).
- **Reading `obj.method` and storing it for later** — that's fine, but reassigning `obj.method = other` shadows it because methods are non-data descriptors. For a method that *cannot* be shadowed per-instance, define a data descriptor explicitly.

### When to reach for what

| Need                                                       | Use                              |
|------------------------------------------------------------|----------------------------------|
| Computed attribute, no per-instance setter logic           | `@property`                      |
| Compute once, then cache on instance                       | `functools.cached_property`      |
| Same validation/conversion across many attributes          | Custom data descriptor           |
| Read-only constant on class                                | Plain class attribute (or descriptor with `__set__` raising) |
| Replace **all** attribute access dynamically               | `__getattr__` (fallback) or `__getattribute__` (heavy hammer) |
| Cross-cutting attribute behavior in a framework            | Descriptor + `__set_name__`      |
