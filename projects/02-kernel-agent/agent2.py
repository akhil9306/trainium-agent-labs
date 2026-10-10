#!/usr/bin/env python3
"""
agent2.py — the kernel agent, rebuilt around what the AWS neuron-agentic-development harness does,
filtered through what this repo has already MEASURED about small models.

Same contract as agent.py: a model writes an NKI kernel, nkibench.py grades it on the CPU, the reason
goes back, it tries again. Same endpoint, same flags, same JSONL log, same reward -- so a run of this
file and a run of agent.py are directly comparable with --repeat. What changes is everything between
"the checker said no" and "the next prompt".

    python agent2.py --selftest                      # the parts that need no model and no device
    python agent2.py --level 3                       # one level
    nohup python agent2.py --all --links 5 --samples 4 --repeat 5 > run2.log 2>&1 < /dev/null &

THE ENSEMBLE CHAIN

Each run of a level is a chain of up to --links generations ("links"). Every link is an ensemble:
--samples candidate kernels written in parallel, and every one is compiled, simulated and checked.

  * The first VERIFIED candidate ends the chain: that run outputs its kernel and the next run starts.
  * Every candidate that fails leaves its error in a memory, and the next link's prompt opens with it:
    "these attempts failed, for these reasons -- do not do that again".
  * A candidate whose code is the same as one that already failed (comments and spacing ignored) is
    rejected without being checked, and the next prompt says it was a repeat. Each link therefore has
    to produce DIFFERENT code.
  * Half of a link's samples repair the best kernel so far, half start fresh with the memory, so the
    ensemble always holds both a fix and a new attempt.
  * Working comes first; speed second. Once a kernel is verified, --speed-links more links ask for a
    version that moves fewer HBM bytes in fewer DMA transfers. One is kept only if it is still verified
    AND moves less. "Faster" here is the simulator's traffic count, not a timing on the chip.

The memory belongs to one run by default, so the --repeat runs stay independent and the solve rate is
honest. --carry-errors lets later runs start from earlier runs' errors (and makes them dependent).

WHAT IS TAKEN FROM THE NEURON HARNESS, AND HOW IT HAD TO BEND

The neuron harness wraps a frontier model with a hard-rules file, tiered references, a debugger fix
ladder, multi-metric validation and stage bisection. This repo wraps an 8B model that measured
differently on two of those, so each idea is kept for what it buys and re-shaped where the measurement
says the original form hurts:

  C0 hard rules       The neuron harness loads a NEVER/ALWAYS table into every prompt. Measured here, a
                      rules list makes a small model audit itself and return nothing. So the table lives
                      in a LINT (stale_api_lint) and only the one row the kernel actually broke goes
                      back, as a FORBIDDEN -> CORRECT pair. Rules in the checker, one fix in the prompt.
  W2/W3 tiered refs   Short pattern cards (CARDS), chosen by the failure category and loaded only into
                      repair prompts, never the first one. A worked example in the first prompt was
                      measured to make every level worse, so the fresh prompt stays as agent.py's.
  X2/X3 fix loop      Replaced by the ensemble chain above: repair and fresh attempts side by side in
                      every link, an error memory, and a ban on resending failed code. The memory grows
                      and repeats are rejected, so no two links send the same prompt: that breaks the
                      greedy fixed point (an unchanged prompt returns an unchanged answer).
  V1 multi-metric     Cosine similarity and the fraction of elements in tolerance give a continuous
                      score INSIDE the 0.30 plateau where all of levels 1, 3 and 4 have sat. The official
                      reward is untouched (comparability); the shaped score only picks between samples.
  V2 bisection        Per-shape and per-tile correctness maps: "single-tile shape passes, 4-tile shape
                      fails, tiles (0,0) and (0,1) right" separates arithmetic bugs from tiling bugs, the
                      same split the neuron debugger gets by retesting on one 128x128 tile.
  Q2 budget, FAIL     A link budget (--links), and an honest end-of-level report: verified or not,
                      confidence, and the dominant failure modes. "Done" must mean verified.
  E1 isolation        Per-process temp files, so two agents in one pod do not overwrite each other.

TWO IDEAS THAT ARE NOT IN THE NEURON HARNESS

  1. Wrong-answer fingerprinting (diagnose_wrong_answer). When the numbers are wrong, the harness
     computes what a set of PLAUSIBLE WRONG kernels would have returned -- only the last K chunk
     accumulated, sum instead of mean, rows and columns swapped, softmax without 1/sqrt(d) -- and checks
     whether the output matches one. A match turns "NUMERICAL MISMATCH at (37, 211)" into "your output
     equals the product of the LAST K chunk only: the PSUM tile is being reset inside the K loop". That is
     a named change instead of a coordinate, which is the whole lesson of this repo.

  2. The ensemble chain itself (solve_chain): failure memory, repeat rejection, and a speed pass that
     keeps a faster kernel only if it is still verified.

Every kernel the agent verifies is also appended to a library file (--library), as a record of what it
produced. Only checker-verified code goes in, never the shipped reference kernels, and the file is never
shown to the model.

Plus held-out verification: a kernel that passes the visible shapes is re-run on fresh seeds and on
large-magnitude inputs before it is called solved, so "solved" means verified, not lucky.

HONESTY NOTES

  * The --repeat runs are independent: each run's error memory starts empty, and the library is only
    written. --carry-errors is the one switch that couples them (run 2 starts from run 1's errors).
  * Autofix (--no-autofix to disable) mechanically repairs things a model should not spend a round on:
    the pre-0.4 `neuronxcc.nki` import path, a missing @nki.jit, a misnamed entry point, missing imports,
    and test code left at module level. Every autofix is logged with the attempt.
  * Every number this prints still comes from the simulator. Nothing here measures latency.
"""

import argparse
import ast
import collections
import concurrent.futures as cf
import hashlib
import inspect
import json
import math
import os
import re
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field

import numpy as np

import nkibench
import agent as base   # reuse the measured pieces: enrich(), extract_code(), first_prompt(), WEIGHTS

MODEL = os.environ.get("KERNEL_AGENT_MODEL", "Qwen/Qwen3-8B")
WEIGHTS = base.WEIGHTS
FULL = sum(WEIGHTS.values())
TOL = 2e-2
REASONING_KEYS = ("reasoning", "reasoning_content")
CHARS_PER_TOKEN = 3.3   # code tokenizes denser than prose; err towards over-estimating the prompt


def est_tokens(text):
    return int(math.ceil(len(text or "") / CHARS_PER_TOKEN))


# ================================================================ C0: the hard rules, as a lint
#
# Each row: (id, how to detect, FORBIDDEN, CORRECT, blocking). Only `blocking` rows reject a kernel
# before it runs, and only one is blocking: PSUM -> HBM DMA, which a CPU simulator may tolerate and the
# device will not. Everything else is advisory -- shown only if the kernel then fails -- because
# rejecting a correct kernel is worse than missing a stale idiom (nkibench's rule, kept).

STALE_API = [
    ("nl_load", r"\bnl\.load\s*\(",
     "t = nl.load(x[...])",
     "t = nl.ndarray(shape, dtype=x.dtype, buffer=nl.sbuf); nisa.dma_copy(dst=t, src=x[...])", False),
    ("nl_store", r"\bnl\.store\s*\(",
     "nl.store(out[...], value=t)",
     "nisa.dma_copy(dst=out[...], src=t)", False),
    ("mask", r"\bmask\s*=",
     "f(..., mask=...)",
     "slice exactly the range you want, e.g. x[0:n, 0:m]; there is no masking", False),
    ("arange_index", r"\bnl\.arange\s*\(",
     "i_p = nl.arange(128)[:, None]; x[i_p, i_f]",
     "plain slices: x[0:128, 0:512], or nl.ds(start, size)", False),
    ("nl_matmul", r"\bnl\.matmul\s*\(",
     "res = nl.matmul(a, b, transpose_x=True)",
     "nisa.nc_matmul(dst=psum_tile, stationary=a_tile, moving=b_tile) with psum_tile in nl.psum", False),
    ("region_call", r"\bnl\.(sbuf|psum|shared_hbm|hbm)\s*\(",
     "t = nl.sbuf((128, 512), nl.float32)",
     "t = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.sbuf)", False),
]


def stale_api_lint(src):
    """Return [(id, forbidden, correct, blocking, line)] for every stale idiom the source uses."""
    hits = []
    for rid, pat, bad, good, blocking in STALE_API:
        m = re.search(pat, src)
        if m:
            hits.append((rid, bad, good, blocking, src[:m.start()].count("\n") + 1))
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return hits
    psum_names = set()
    for node in ast.walk(tree):
        # `x = nisa.f(...)`: ISA instructions write into dst= and return nothing.
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = nkibench._dotted(node.value.func)
            if callee.startswith("nisa."):
                hits.append(("isa_return", f"x = {callee}(...)",
                             f"allocate x with nl.ndarray first, then {callee}(dst=x, ...); "
                             f"nisa instructions write into dst= and return nothing", False,
                             node.lineno))
            if callee.endswith("ndarray") and any(
                    k.arg == "buffer" and nkibench._dotted(k.value).endswith("psum")
                    for k in node.value.keywords):
                psum_names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and nkibench._dotted(node.func).endswith("dma_copy"):
            src_kw = next((k.value for k in node.keywords if k.arg == "src"), None)
            base_name = src_kw
            while isinstance(base_name, ast.Subscript):
                base_name = base_name.value
            if isinstance(base_name, ast.Name) and base_name.id in psum_names:
                hits.append(("psum_dma", f"nisa.dma_copy(dst=<hbm>, src={base_name.id})  # a psum tile",
                             f"res_sb = nl.ndarray({base_name.id}.shape, dtype=<out dtype>, "
                             f"buffer=nl.sbuf); nisa.tensor_copy(dst=res_sb, src={base_name.id}); "
                             f"nisa.dma_copy(dst=<hbm>, src=res_sb)", True, node.lineno))
    return hits


def lint_instruction(hit):
    rid, bad, good, _, line = hit
    return (f"Line {line} uses an outdated or illegal NKI pattern. FORBIDDEN: `{bad}`. "
            f"CORRECT: `{good}`.")


# ================================================================ autofix: rounds a model should not spend

def autofix(src, entry):
    """Mechanical repairs, each logged. Returns (new_src, [what was done])."""
    done = []
    if "neuronxcc" in src:
        new = re.sub(r"^\s*from\s+neuronxcc\s+import\s+nki\b.*$", "import nki", src, flags=re.M)
        new = new.replace("neuronxcc.nki", "nki")
        if new != src:
            done.append("rewrote the pre-0.4 `neuronxcc.nki` import path to `nki`")
            src = new
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return src, done

    # Drop module-level test code. A test harness the model appends can call np.matmul, which the rule
    # checker then blames on the kernel, or run the kernel at import time.
    lines = src.splitlines()
    drop = []
    for node in tree.body:
        is_main = (isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
        is_call = isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        if is_main or is_call:
            drop.append((node.lineno, node.end_lineno))
    if drop:
        for lo, hi in sorted(drop, reverse=True):
            del lines[lo - 1:hi]
        src = "\n".join(lines) + "\n"
        done.append(f"removed {len(drop)} block(s) of module-level test code")
        tree = ast.parse(src)

    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    names = {f.name for f in funcs}
    if entry not in names:
        jitted = [f for f in funcs if any("jit" in ast.unparse(d) for d in f.decorator_list)]
        pick = jitted if len(jitted) == 1 else (funcs if len(funcs) == 1 else [])
        if pick:
            old = pick[0].name
            src = re.sub(rf"\b{re.escape(old)}\b", entry, src)
            done.append(f"renamed `{old}` to the entry point `{entry}`")
            tree = ast.parse(src)
            funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    fn = next((f for f in funcs if f.name == entry), None)
    if fn is not None and not any("jit" in ast.unparse(d) for d in fn.decorator_list):
        lines = src.splitlines()
        at = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
        lines.insert(at, "@nki.jit")
        src = "\n".join(lines) + "\n"
        done.append(f"added @nki.jit above `{entry}`")

    need = []
    if not re.search(r"^\s*import\s+nki\s*$", src, re.M):
        need.append("import nki")
    if re.search(r"\bnl\.", src) and "nki.language as nl" not in src:
        need.append("import nki.language as nl")
    if re.search(r"\bnisa\.", src) and "nki.isa as nisa" not in src:
        need.append("import nki.isa as nisa")
    if re.search(r"\bnp\.", src) and not re.search(r"import\s+numpy\s+as\s+np", src):
        need.append("import numpy as np")
    if need:
        src = "\n".join(need) + "\n" + src
        done.append("added missing import(s): " + "; ".join(need))
    return src, done


# ================================================================ idea 1: wrong-answer fingerprinting

def _rms(x):
    return float(np.sqrt((np.asarray(x, np.float64) ** 2).mean())) or 1.0


def _close(a, b, tol=TOL):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return a.shape == b.shape and float(np.abs(a - b).max()) / _rms(b) <= tol


def _softmax(s, axis=-1):
    s = s - s.max(axis=axis, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=axis, keepdims=True)


def mutants(level, args):
    """What plausible WRONG kernels would return for these inputs: [(name, array, instruction, category)].

    Each entry is a specific misunderstanding seen in, or predicted from, the kernels models write. A
    match is a diagnosis with a cause, which is worth far more than the coordinates of the worst element.
    """
    ref = nkibench.LEVELS[level]["ref"]
    out = []
    if ref is nkibench.ref_matmul:
        lhsT, rhs = (np.asarray(a, np.float64) for a in args)
        K, T = lhsT.shape[0], nkibench.PMAX
        chunks = [lhsT[k:k + T].T @ rhs[k:k + T] for k in range(0, K, T)]
        if len(chunks) > 1:
            out.append(("the product of the LAST K chunk only", chunks[-1],
                        "The PSUM accumulator is reset on every K step, so each chunk overwrites the "
                        "one before. Allocate the psum tile ONCE per output tile, BEFORE the k loop; call "
                        "nisa.nc_matmul into that same tile on every k step; tensor_copy it out AFTER "
                        "the k loop ends.", "k_accum_reset"))
            out.append(("the product of the FIRST K chunk only", chunks[0],
                        "Only rows 0:128 of K are contracted. The K loop runs once, or its slices do "
                        "not move: load lhsT[k*128:(k+1)*128, ...] and rhs[k*128:(k+1)*128, ...] "
                        "using the k loop variable, for k in nl.affine_range(K // 128).",
                        "k_slice_static"))
        full = lhsT.T @ rhs
        if full.shape[0] == full.shape[1]:
            out.append(("the correct product, TRANSPOSED", full.T,
                        "Rows and columns of the result are swapped. lhsT is [K, M] and is the "
                        "`stationary` operand; rhs is [K, N] and is `moving`; the result is [M, N].",
                        "operand_swap"))
    elif ref is nkibench.ref_avgpool2d:
        x, p = np.asarray(args[0], np.float64), args[1]
        C, H, W = x.shape
        win = x[:, :H // p * p, :W // p * p].reshape(C, H // p, p, W // p, p)
        r = win.mean(axis=(2, 4))
        out.append(("the window SUM, not the mean", r * p * p,
                    "Each window was summed but not divided. Scale the sum by 1/(pool_size*pool_size), "
                    "e.g. nisa.tensor_scalar(dst=out_tile, data=sum_tile, op0=nl.multiply, "
                    "operand0=1.0/(pool_size*pool_size)).", "wrong_scale"))
        out.append(("the window sum divided by pool_size once", r * p,
                    "Divide by pool_size*pool_size -- the window has pool_size**2 elements -- not by "
                    "pool_size.", "wrong_scale"))
        out.append(("the window MAXIMUM", win.max(axis=(2, 4)),
                    "This is max pooling. Average pooling sums each window with nl.sum and scales by "
                    "1/(pool_size*pool_size).", "wrong_op"))
        n = (H // p) * (W // p)
        flat = x.reshape(C, H * W)[:, :n * p * p].reshape(C, n, p * p).mean(-1).reshape(C, H // p, W // p)
        out.append(("means of p*p CONSECUTIVE elements of each flattened row", flat,
                    "Your windows are runs of p*p neighbours in memory, not p x p squares. In the .ap() "
                    "view, the two inner pairs must step down a ROW (step W, count p) and along a COLUMN "
                    "(step 1, count p); the two outer pairs step p*W (count H//p) and p (count W//p).",
                    "wrong_layout"))
        if H // p == W // p:
            out.append(("the right answer with rows and columns swapped", r.transpose(0, 2, 1),
                        "Output rows and columns are swapped: the first output axis must walk down the "
                        "rows (step p*W) and the second along the columns (step p).", "wrong_layout"))
    elif ref is nkibench.ref_transpose2d:
        x, (F1, F2) = np.asarray(args[0], np.float64), args[1]
        P, F = x.shape
        out.append(("the INPUT, unchanged", x,
                    "No element moved. Within each row, the element at position i1*F2 + i2 (i1 < F1, "
                    "i2 < F2) must go to position i2*F1 + i1.", "no_op"))
        if F1 != F2:
            out.append(("a transpose that read shape2D backwards",
                        x.reshape(P, F2, F1).transpose(0, 2, 1).reshape(P, F),
                        f"You treated each row as {F2}x{F1}; shape2D is (F1, F2) = ({F1}, {F2}). The "
                        f"source position is i1*F2 + i2 and the destination is i2*F1 + i1, with i1 in "
                        f"range(F1) and i2 in range(F2).", "wrong_layout"))
    elif ref is nkibench.ref_attention:
        q, k, v = (np.asarray(a, np.float64) for a in args)
        d = q.shape[1]
        s = q @ k.T
        out.append(("attention WITHOUT the 1/sqrt(dim) scale", _softmax(s) @ v,
                    "Multiply the scores Q.K^T by 1/sqrt(dim) before the softmax.", "wrong_scale"))
        out.append(("exp(scores) @ V without dividing by the row sum",
                    np.exp(s / np.sqrt(d) - (s / np.sqrt(d)).max(-1, keepdims=True)) @ v,
                    "The softmax is not normalised: divide each row of exp(scores - rowmax) by its "
                    "row sum before multiplying by V.", "missing_norm"))
        out.append(("softmax over the WRONG axis", _softmax(s / np.sqrt(d), axis=0) @ v,
                    "The softmax normalises columns. Take the max and the sum along each ROW of the "
                    "scores (the last axis).", "wrong_layout"))
    return out


def tile_map(got, want, tol=TOL):
    """Per-tile correctness for a 2-D output, tiles of 128 x 512: (text, category) or None.

    The neuron debugger isolates logic bugs from tiling bugs by retesting on one 128x128 tile. This is
    the same split without re-running anything: if some tiles are exactly right, the arithmetic is right.
    """
    if got.ndim != 2 or got.shape != want.shape:
        return None
    R, C = got.shape
    br, bc = min(nkibench.PMAX, R), min(nkibench.GEMM_MOVING_FMAX, C)
    blocks = [(i, j) for i in range(math.ceil(R / br)) for j in range(math.ceil(C / bc))]
    if len(blocks) < 2:
        return None
    scale = _rms(want)

    def blk(a, b):
        return a[b[0] * br:(b[0] + 1) * br, b[1] * bc:(b[1] + 1) * bc]

    ok = {b: float(np.abs(blk(got, b) - blk(want, b)).max()) / scale <= tol for b in blocks}
    wrong = [b for b in blocks if not ok[b]]
    if not wrong:
        return None
    moved = {}
    for b in wrong:
        g = blk(got, b)
        for b2 in blocks:
            w = blk(want, b2)
            if b2 != b and g.shape == w.shape and float(np.abs(g - w).max()) / scale <= tol:
                moved[b] = b2
                break
    if moved:
        eg = "; ".join(f"tile {b} holds the answer for tile {s}" for b, s in list(moved.items())[:3])
        return (f"OUTPUT TILES ARE IN THE WRONG PLACE: {eg}. The values are right, the addressing is "
                f"not. Every load and every store must use the loop variables of the tile being "
                f"computed -- rows m*128:(m+1)*128 and columns n*512:(n+1)*512 of the output, and the "
                f"matching slices of the inputs.", "tile_indexing")
    right = [b for b in blocks if ok[b]]
    if right:
        return (f"{len(right)} of {len(blocks)} output tiles are exactly right ({right[:4]}) and "
                f"{len(wrong)} are wrong ({wrong[:4]}). The arithmetic works, so the bug is in how the "
                f"loop reaches the other tiles: loop bounds, slice offsets, or a tile reused from the "
                f"previous iteration.", "partial_tiles")
    return None


def diagnose_wrong_answer(level, args, got, want, tol=TOL):
    """(instruction, category) naming what the kernel computed instead, or None to fall back to nkibench."""
    g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
    if g.shape != w.shape or not np.all(np.isfinite(g)):
        return None
    for name, cand, instr, cat in mutants(level, args):
        if cand.shape == g.shape and _close(g, cand, tol) and not _close(w, cand, tol):
            return f"YOUR OUTPUT EQUALS {name}. {instr}", cat
    t = tile_map(g, w, tol)
    if t:
        return t
    denom = float((w * w).sum())
    if denom > 0:
        alpha = float((g * w).sum()) / denom
        if abs(alpha - 1) > tol and _close(g, alpha * w, tol):
            return (f"EVERY ELEMENT IS {alpha:.4g} TIMES THE RIGHT VALUE. The structure is right; one "
                    f"constant is wrong -- look for a missing or extra scale or divide.", "wrong_scale")
    return None


def closeness(got, want, tol=TOL):
    """0..1, how near a wrong answer is: half fraction-in-tolerance, half cosine similarity."""
    try:
        g, w = np.asarray(got, np.float64), np.asarray(want, np.float64)
        if g.shape != w.shape or not np.all(np.isfinite(g)):
            return 0.0
        frac = float((np.abs(g - w) / _rms(w) <= tol).mean())
        den = float(np.linalg.norm(g) * np.linalg.norm(w))
        cos = float((g * w).sum()) / den if den else 0.0
        return 0.5 * frac + 0.5 * max(cos, 0.0)
    except Exception:
        return 0.0


# ================================================================ grading

@dataclass
class Verdict:
    reward: float = 0.0          # the official, agent.py-comparable reward
    shaping: float = 0.0         # reward + a little for being close; used ONLY to choose between attempts
    parts: dict = field(default_factory=lambda: dict(parses=False, rules=False, runs=False,
                                                     correct=False))
    feedback: str = ""
    category: str = ""
    source: str = ""
    autofixes: list = field(default_factory=list)
    verified: bool = False
    heldout: str = ""
    shapes_passed: int = 0
    hbm_bytes: int = 0           # summed over the shapes that passed; the speed proxy
    transfers: int = 0           # DMA transfers over the same shapes


def _so_far(parts):
    return sum(WEIGHTS[k] for k, v in parts.items() if v)


def _run_file(source, level):
    """Write the candidate to a per-process file (two agents in one pod must not collide) and load it."""
    fd, path = tempfile.mkstemp(prefix=f"kagent_L{level}_{os.getpid()}_", suffix=".py")
    with os.fdopen(fd, "w") as f:
        f.write(source)
    return path


def heldout_check(kernel, level, seeds):
    """Re-run a kernel that passed on inputs it has never seen: fresh seeds, and large magnitudes.

    Passing the visible shapes is evidence; passing these too is what earns the word "verified".
    """
    spec = nkibench.LEVELS[level]
    total, passed, first = 0, 0, ""
    for case in spec["shapes"]:
        variants = [(s, 1.0) for s in seeds] + [(seeds[0] if seeds else 7, 50.0)]
        for seed, mag in variants:
            args, _ = nkibench.make_inputs(case, level, seed)
            args = tuple(a * np.float32(mag) if isinstance(a, np.ndarray) else a for a in args)
            want = spec["ref"](*args)
            total += 1
            try:
                got, _ = nkibench.simulate_and_count(kernel, args)
                m = nkibench.describe_mismatch(got, want)
            except Exception as e:
                m = f"raised {type(e).__name__}: {e}"
            if m:
                first = first or (f"seed {seed}, inputs x{mag:g}, {nkibench.label(case, level)}: "
                                  f"{m.splitlines()[0]}")
            else:
                passed += 1
    return passed, total, first


def grade(source, level, a):
    v = Verdict(source=source)
    spec = nkibench.LEVELS[level]
    if not source.strip():
        v.feedback, v.category = ("No code came back. Reply with one python code block containing the "
                                  "kernel and nothing else."), "no_code"
        return v
    if a.autofix:
        source, v.autofixes = autofix(source, spec["entry"])
        v.source = source
    try:
        compile(source, "<candidate>", "exec")
        v.parts["parses"] = True
    except SyntaxError as e:
        v.feedback, v.category = (f"The code does not parse: {e.msg} on line {e.lineno}. Send one "
                                  f"complete python code block."), "parse"
        return v

    lint = stale_api_lint(source)
    violations = nkibench.check_rules(source, level)
    blocking = [h for h in lint if h[3]]
    if violations or blocking:
        v.reward = v.shaping = _so_far(v.parts)
        if blocking:
            v.feedback, v.category = lint_instruction(blocking[0]), "stale_api"
            return v
        extra = ""
        if any("no function named" in x for x in violations):
            params = ", ".join(inspect.signature(spec["ref"]).parameters)
            extra = (f" Start the function with exactly this line:  def {spec['entry']}({params}):  "
                     f"and put @nki.jit on the line above it.")
            v.category = "missing_entry"
        else:
            v.category = "rule_violation"
        v.feedback = ("Rule violations, which score zero however fast the kernel is. Fix exactly these: "
                      + " ".join(violations) + extra)
        return v
    v.parts["rules"] = True
    advisory = next((h for h in lint if not h[3]), None)

    path = _run_file(source, level)
    try:
        try:
            kernel = nkibench.load_kernel(path, spec["entry"])
        except ModuleNotFoundError as e:
            try:
                import nki  # noqa: F401
            except ImportError:
                raise SystemExit(f"cannot import {e.name!r}: the Neuron SDK is not here. Run inside a "
                                 f"seat pod, or use --selftest for the parts that need no device.") from e
            v.reward = v.shaping = _so_far(v.parts)
            v.feedback, v.category = (f"There is no module named {e.name!r}. The only imports that exist "
                                      f"are `import nki`, `import nki.language as nl` and "
                                      f"`import nki.isa as nisa`."), "import_error"
            return v
        except Exception as e:
            v.reward = v.shaping = _so_far(v.parts)
            v.feedback = f"The file imports but {spec['entry']} could not be loaded: {type(e).__name__}: {e}"
            v.category = "load_error"
            if advisory:
                v.feedback = lint_instruction(advisory) + " " + v.feedback
            return v

        results = []   # (label, ok, message, category, closeness)
        intensity = None
        for case in spec["shapes"]:
            lbl = nkibench.label(case, level)
            args, _ = nkibench.make_inputs(case, level)
            before = [x.copy() if isinstance(x, np.ndarray) else x for x in args]
            want = spec["ref"](*args)
            try:
                got, counted = nkibench.simulate_and_count(kernel, args)
            except nkibench.NkiMissing as e:
                v.reward = v.shaping = _so_far(v.parts)
                v.feedback, v.category = f"CANNOT SIMULATE: {e}", "environment"
                return v
            except Exception as e:
                msg = base.enrich(f"raised {type(e).__name__}: {e}")
                if advisory:
                    msg = lint_instruction(advisory) + " Separately, the simulator " + msg
                results.append((lbl, False, msg, classify_runtime(msg), 0.0))
                continue
            v.parts["runs"] = True
            m = nkibench.check_inputs_untouched(before, args)
            cat = "input_modified" if m else ""
            if not m:
                m = nkibench.describe_mismatch(got, want)
                if m:
                    d = diagnose_wrong_answer(level, args, got, want)
                    if d:
                        m, cat = d[0] + "\n  (detail: " + m.splitlines()[0] + ")", d[1]
                    else:
                        cat = classify_mismatch(m)
            if not m:
                m = nkibench.check_traffic_bar(level, counted, args, want)
                cat = "traffic" if m else ""
            hazards = [w for w in counted.get("warnings", []) if "incorrect results on hardware" in w]
            if hazards and not m:
                m, cat = ("CORRECT ON CPU BUT WRONG ON HARDWARE: " + hazards[0] + ". Fix that before "
                          "anything else."), "hardware_hazard"
            if m:
                results.append((lbl, False, m, cat, closeness(got, want)))
                continue
            results.append((lbl, True, "", "", 1.0))
            v.hbm_bytes += int(counted.get("bytes") or 0)
            v.transfers += int(counted.get("transfers") or 0)
            if "M" in case and counted.get("bytes"):
                intensity = nkibench.roofline(
                    nkibench.matmul_flops(case["M"], case["K"], case["N"]), counted["bytes"])

        n = len(results)
        passed = [r for r in results if r[1]]
        failed = [r for r in results if not r[1]]
        v.shapes_passed = len(passed)
        if failed:
            v.reward = _so_far(v.parts) + WEIGHTS["correct"] * len(passed) / n
            near = sum(r[4] for r in failed) / len(failed)
            v.shaping = v.reward + 0.09 * near
            # Prefer the failure we can NAME: a fingerprinted one beats a coordinate.
            named = {"k_accum_reset", "k_slice_static", "operand_swap", "wrong_scale", "wrong_op",
                     "wrong_layout", "no_op", "missing_norm", "tile_indexing", "partial_tiles"}
            lbl, _, msg, cat, _ = next((r for r in failed if r[3] in named), failed[0])
            head = f"{len(passed)} of {n} shapes passed."
            if passed:
                head += (f" PASSED: {', '.join(r[0] for r in passed)}. FAILED: "
                         f"{', '.join(r[0] for r in failed)}. Whatever differs between those shapes "
                         f"(more tiles, a K above 128, a size not a multiple of the tile) is where the bug "
                         f"is; the rest of the kernel already works.")
            v.feedback, v.category = f"{head} On {lbl}: {msg}", cat or "numerics"
            if advisory and v.category in ("runtime", "api_invented", "api_misuse"):
                v.feedback += " " + lint_instruction(advisory)
            return v

        v.parts["correct"] = True
        v.reward = v.shaping = FULL
        v.category = "solved"
        v.feedback = "Correct on every shape."
        if intensity:
            v.feedback += " " + nkibench.explain_roofline(intensity)
        if a.heldout_seeds:
            ok, tot, first = heldout_check(kernel, level, a.heldout_seeds)
            v.heldout = f"{ok}/{tot}"
            v.verified = ok == tot
            if not v.verified:
                v.shaping = FULL - 0.05
                v.category = "heldout_fail"
                v.feedback = (f"Correct on the visible shapes, but WRONG on {tot - ok} of {tot} unseen "
                              f"inputs -- the kernel depends on something that only happens to hold for "
                              f"the test data. First failure: {first}")
        else:
            v.verified = True
        return v
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ================================================================ failure taxonomy

def classify_runtime(msg):
    rules = [("MemoryRegion", "api_misuse"), ("unexpected keyword", "api_misuse"),
             ("multiple values", "api_misuse"), ("has no attribute", "api_invented"),
             ("same number of elements", "shape_mismatch"), ("could not be broadcast", "shape_mismatch"),
             ("partition dimension", "partition_limit"), ("Out-of-bound", "out_of_bounds"),
             ("contraction dimension", "k_too_big"), ("must be in [", "buffer_placement"),
             ("at least 2 dimensions", "shape_mismatch"), ("NkiTensor", "python_op_on_tile")]
    return next((c for k, c in rules if k in msg), "runtime")


def classify_mismatch(msg):
    if msg.startswith("WRONG SHAPE"):
        return "wrong_shape"
    if msg.startswith("NON-FINITE"):
        return "nonfinite"
    if "ZEROS" in msg or "zero, in the block" in msg:
        return "unwritten_output"
    if "ragged edge" in msg:
        return "ragged_edge"
    return "numerics"


# ================================================================ W3: pattern cards, loaded on demand
#
# Built only from calls that appear in the shipped tutorial kernels, so none of it is invented. They are
# idioms, never a level's answer, and they are NEVER in the first prompt: a worked example there was
# measured to make every level worse.

CARDS = {
    "psum_flow": """The only legal matmul data path:
  HBM --nisa.dma_copy--> SBUF --nisa.nc_matmul--> PSUM --nisa.tensor_copy--> SBUF --nisa.dma_copy--> HBM
  ps = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.psum)
  nisa.nc_matmul(dst=ps, stationary=a_sb, moving=b_sb)   # a_sb is [k, m], b_sb is [k, n], both nl.sbuf
  o_sb = nl.ndarray((m, n), dtype=out.dtype, buffer=nl.sbuf)
  nisa.tensor_copy(dst=o_sb, src=ps)
  nisa.dma_copy(dst=out[<rows>, <cols>], src=o_sb)""",
    "k_accum": """K above 128 is contracted in chunks that ADD UP in one PSUM tile:
  ps = nl.ndarray((m, n), dtype=nl.float32, buffer=nl.psum)   # once, BEFORE the k loop
  for k in nl.affine_range(K // 128):
      ... load the k-th 128-row slices of both operands into fresh sbuf tiles ...
      nisa.nc_matmul(dst=ps, stationary=a_sb, moving=b_sb)     # same ps every time
  ... tensor_copy ps to sbuf and dma_copy it out, AFTER the loop""",
    "tile_loop": """A first axis larger than 128 is processed in chunks, one tile per chunk:
  for i in nl.affine_range(R // 128):
      t = nl.ndarray((128, C), dtype=x.dtype, buffer=nl.sbuf)
      nisa.dma_copy(dst=t, src=x[i*128:(i+1)*128, 0:C])
      ... compute into an sbuf tile r of the same shape ...
      nisa.dma_copy(dst=out[i*128:(i+1)*128, 0:C], src=r)
Each tile has exactly the shape of the slice it is copied from. A dimension of 128 or less is used whole.""",
    "strided_view": """t.ap([[step, count], ...]) is a strided VIEW of an SBUF tile; nothing moves. Each pair is
(step in elements through the tile, how many). The first pair is the partition axis:
[elements per partition, number of partitions]. The view's shape is the list of counts.
  # a (P, 6) tile seen as P x 3 pairs, then summed within each pair:
  v = t.ap([[6, P], [2, 3], [1, 2]])    # shape (P, 3, 2)
  s = nl.sum(v, axis=[2])               # shape (P, 3)""",
    "column_copy": """Moving single columns inside SBUF: nl.ds(start, size) picks `size` elements from `start`
along the free axis, on every partition at once.
  nisa.tensor_copy(dst=out_t[:, nl.ds(dst_col, 1)], src=in_t[:, nl.ds(src_col, 1)])
Loop over source positions with nl.affine_range and compute dst_col from the loop variables.""",
    "output_path": """Every tile must reach the returned tensor: compute -> sbuf tile ->
nisa.dma_copy(dst=out[<this tile's rows>, <this tile's cols>], src=tile), inside the loop, with the
destination slice built from the loop variables. Return the tensor allocated with buffer=nl.shared_hbm.""",
    "scale": """nisa.tensor_scalar(dst=o, data=t, op0=nl.multiply, operand0=c) multiplies every element of
the sbuf tile t by the constant c. o and t have the same shape.""",
}

CATEGORY_CARD = {
    "k_accum_reset": "k_accum", "k_slice_static": "k_accum", "k_too_big": "k_accum",
    "buffer_placement": "psum_flow", "operand_swap": "psum_flow", "api_misuse": "psum_flow",
    "shape_mismatch": "tile_loop", "partition_limit": "tile_loop", "out_of_bounds": "tile_loop",
    "ragged_edge": "tile_loop", "tile_indexing": "tile_loop", "partial_tiles": "tile_loop",
    "unwritten_output": "output_path", "input_modified": "output_path", "wrong_scale": "scale",
}


def level_card(level):
    ref = nkibench.LEVELS[level]["ref"]
    if ref is nkibench.ref_avgpool2d:
        return "strided_view"
    if ref is nkibench.ref_transpose2d:
        return "column_copy"
    if ref is nkibench.ref_matmul:
        return "k_accum" if any(c.get("K", 0) > nkibench.PMAX for c in
                                nkibench.LEVELS[level]["shapes"]) else "psum_flow"
    return "psum_flow"


def card_for(level, category):
    return CARDS[CATEGORY_CARD.get(category) or level_card(level)]


# ================================================================ the verified-kernel record

class Library:
    """Kernels THIS agent wrote and the checker verified. Never the shipped reference kernels."""

    def __init__(self, path):
        self.path, self.items = path, []
        if path and os.path.exists(path):
            with open(path) as f:
                self.items = [json.loads(l) for l in f if l.strip()]

    def add(self, level, code, tag):
        item = dict(level=level, op=nkibench.LEVELS[level]["op"], code=code, tag=tag, t=time.time())
        self.items.append(item)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(item) + "\n")

# ================================================================ prompts, packed to a token budget

def pack(sections, budget):
    """sections: [(name, text, priority)], priority 0 = never dropped. Drops the least important first.

    Returns (prompt, {name: tokens}, [dropped]). The per-section token counts go into the log: where
    the budget went, attempt by attempt, is the instrumentation the challenge asks for.
    """
    keep = [s for s in sections if s[1]]
    dropped = []
    while sum(est_tokens(t) for _, t, _ in keep) > budget:
        droppable = [s for s in keep if s[2] > 0]
        if not droppable:
            break
        worst = max(droppable, key=lambda s: s[2])
        keep.remove(worst)
        dropped.append(worst[0])
    return ("\n\n".join(t for _, t, _ in keep), {n: est_tokens(t) for n, t, _ in keep}, dropped)


def signature_line(level):
    spec = nkibench.LEVELS[level]
    params = ", ".join(inspect.signature(spec["ref"]).parameters)
    return f"The kernel starts with exactly these two lines:\n@nki.jit\ndef {spec['entry']}({params}):"


def input_budget(a):
    return a.context - answer_reserve(a) - 64


def answer_reserve(a):
    return min(a.max_tokens, max(1200, a.context // 3))


# ================================================================ the model

def ask(a, prompt):
    import httpx
    budget = min(a.max_tokens, max(256, a.context - est_tokens(prompt) - 64))
    body = dict(model=a.model, messages=[{"role": "user", "content": prompt}],
                max_tokens=budget, temperature=0.6, top_p=0.95,
                chat_template_kwargs={"enable_thinking": a.think})
    try:
        r = httpx.post(f"{a.base.rstrip('/')}/chat/completions", json=body, timeout=900, verify=False)
    except httpx.HTTPError as e:
        print(f"    (request failed: {type(e).__name__}: {e})")
        return dict(content="", finish="error", usage={}, reasoning=0)
    if r.status_code != 200:
        raise SystemExit(f"the endpoint returned HTTP {r.status_code}:\n{r.text[:600]}")
    payload = r.json()
    ch = payload["choices"][0]
    msg = ch.get("message", {})
    reasoning = next((msg[k] for k in REASONING_KEYS if msg.get(k)), "")
    content = msg.get("content") or ""
    if ch.get("finish_reason") == "length":
        print(f"    (TRUNCATED at {len(content)} chars: any parse error is the budget, not the model)")
    if not content.strip() and reasoning:
        print(f"    (empty answer after {len(reasoning)} chars of hidden reasoning -- shorten the prompt)")
    return dict(content=content, finish=ch.get("finish_reason"), usage=payload.get("usage") or {},
                reasoning=len(reasoning))


def ask_all(a, prompts):
    with cf.ThreadPoolExecutor(max_workers=len(prompts)) as ex:
        return list(ex.map(lambda p: ask(a, p), prompts))


# ================================================================ the ensemble chain
#
# A run of a level is a chain of up to --links links. Each link is an ensemble of --samples candidates.
# The first verified candidate ends the chain. Each failed candidate's error goes into a memory that
# opens the next link's prompt; a candidate identical to one that already failed is rejected unchecked.
# After a verified kernel, --speed-links more links look for one that moves fewer HBM bytes.

@dataclass
class Failure:
    tag: str        # where it came from: "L2.s3" is link 2, sample 3; "r1/L2.s3" was carried from run 1
    key: str        # code_key of the code that failed ("" if there was no code)
    category: str
    feedback: str


def code_key(src):
    """Identity of a kernel for 'is this the same code again?'. Comments, blank lines, spacing and
    quoting do not count: two sources with the same syntax tree are the same kernel."""
    if not (src or "").strip():
        return ""
    try:
        canon = ast.dump(ast.parse(src))
    except SyntaxError:
        canon = "\n".join(l for l in (re.sub(r"\s+", " ", x.split("#")[0]).strip()
                                      for x in src.splitlines()) if l)
    return hashlib.sha1(canon.encode()).hexdigest()[:12]


def memory_text(fails, limit=6, head="These earlier attempts FAILED. Do not repeat any of them -- write "
                                     "something different that avoids each of these errors:"):
    """The 'do not do that again' block. The same error from several attempts is one line with a count,
    and only the `limit` most recent distinct errors are kept, so the block stays a few hundred tokens."""
    if not fails:
        return ""
    groups = collections.OrderedDict()
    for f in fails:
        k = (f.category, f.feedback[:160])
        g = groups.pop(k, None) or [f, 0, []]
        g[1] += 1
        g[2].append(f.tag)
        groups[k] = g                     # re-inserted at the end: most recent last
    lines = []
    for f, n, tags in list(groups.values())[-limit:]:
        seen = f" (seen {n}x: {', '.join(tags[-3:])})" if n > 1 else f" ({tags[0]})"
        lines.append(f"- {f.category}{seen}: {textwrap.shorten(f.feedback, 240)}")
    return head + "\n" + "\n".join(lines)


def traffic_floor(level):
    """Fewest HBM bytes the test shapes can need: each input read once, each output written once."""
    spec = nkibench.LEVELS[level]
    total = 0
    for case in spec["shapes"]:
        args, _ = nkibench.make_inputs(case, level)
        total += nkibench.minimum_hbm_bytes(args, spec["ref"](*args))
    return total


def chain_plan(link, n, have_code):
    """Link 0, or no code yet: every sample starts fresh. After that the ensemble alternates repair of
    the best kernel so far with a fresh attempt, so one link always holds both."""
    if link == 0 or not have_code:
        return ["fresh"] * n
    return ["repair" if i % 2 == 0 else "fresh" for i in range(n)]


def chain_prompt(kind, level, a, memory, best=None, terse=0):
    spec = nkibench.LEVELS[level]
    mem = memory_text(memory)
    S = []
    if kind == "fresh":
        S.append(("task", base.first_prompt(level, terse), 0))
        S.append(("memory", mem, 1))
        S.append(("signature", signature_line(level) + "\nReply with ONE python code block.", 0))
    elif kind == "repair":
        S.append(("task", f"This NKI kernel for {spec['op']} is not right yet.\n\n"
                          f"```python\n{best.source}\n```", 0))
        S.append(("feedback", f"A checker reports:\n{best.feedback}", 0))
        S.append(("memory", mem, 1))
        if a.cards:
            S.append(("card", "Reference pattern:\n" + card_for(level, best.category), 2))
        S.append(("ask", "Fix exactly what the checker names. Your code must differ from every failed "
                         "attempt above. Reply with ONE python code block.", 0))
    elif kind == "speed":
        S.append(("task", f"This NKI kernel for {spec['op']} is CORRECT on every test shape:\n\n"
                          f"```python\n{best.source}\n```", 0))
        S.append(("numbers", f"Across the test shapes it moves {best.hbm_bytes:,} bytes over HBM in "
                             f"{best.transfers:,} DMA transfers. The floor -- every input read once, the "
                             f"output written once -- is {traffic_floor(level):,} bytes. Make it FASTER: "
                             f"fewer HBM bytes, and fewer, larger DMA transfers (load each tile once and "
                             f"reuse it; copy whole tiles, not elements). It must stay correct on every "
                             f"shape: a faster wrong kernel is worth nothing.", 0))
        S.append(("memory", mem, 1))
        S.append(("signature", signature_line(level) + "\nReply with ONE python code block.", 0))
    else:
        raise ValueError(kind)
    return pack(S, input_budget(a))


def _ask_link(a, level, prompts, link):
    if a.offline:
        return [dict(content=r, finish="offline", usage={}, reasoning=0)
                for r in base.offline_answers(level, len(prompts), link)]
    return ask_all(a, prompts)


def _log_attempt(log, run_tag, level, link, i, kind, key, dup, v, secs, dropped, rep, phase):
    log.write(json.dumps(dict(
        mode="chain", phase=phase, run=run_tag, level=level, link=link, sample=i, kind=kind, key=key,
        duplicate=bool(dup), reward=v.reward, shaping=round(v.shaping, 4), parts=v.parts,
        category=v.category, verified=v.verified, heldout=v.heldout, autofixes=v.autofixes,
        hbm_bytes=v.hbm_bytes, transfers=v.transfers, prompt_tokens=secs, dropped=dropped,
        usage=rep["usage"], finish=rep["finish"], reasoning_chars=rep["reasoning"],
        code=v.source, feedback=v.feedback)) + "\n")


def speed_up(a, level, log, run_tag, start):
    """Links after a verified kernel, each asking for less HBM traffic. Returns the fastest verified
    kernel found, or None if none beat `start`. A candidate that is wrong, or right but no faster,
    goes into a memory of its own, so the next speed link is told what did not help."""
    cur, memory, keys = start, [], {code_key(start.source)}
    for s in range(a.speed_links):
        t0 = time.perf_counter()
        built = [chain_prompt("speed", level, a, memory, cur) for _ in range(a.samples)]
        replies = _ask_link(a, level, [b[0] for b in built], s)
        better = []
        for i, ((prompt, secs, dropped), rep) in enumerate(zip(built, replies)):
            tag = f"S{s}.s{i}"
            code = base.extract_code(rep["content"])
            key = code_key(code)
            if key in keys:
                v = Verdict(source=code, category="duplicate",
                            feedback="the same code as a kernel already tried; change the data movement")
            else:
                v = grade(code, level, a)
            keys.add(key)
            _log_attempt(log, run_tag, level, s, i, "speed", key, v.category == "duplicate", v, secs,
                         dropped, rep, "speed")
            if v.verified and (v.hbm_bytes, v.transfers) < (cur.hbm_bytes, cur.transfers):
                better.append(v)
            elif v.verified:
                memory.append(Failure(tag, key, "not_faster",
                                      f"correct, but {v.hbm_bytes:,} bytes in {v.transfers:,} transfers -- "
                                      f"no better than {cur.hbm_bytes:,} in {cur.transfers:,}"))
            elif v.category != "duplicate":
                memory.append(Failure(tag, key, v.category, v.feedback))
        log.flush()
        if better:
            new = min(better, key=lambda v: (v.hbm_bytes, v.transfers))
            print(f"  speed link {s}: FASTER and still verified: {cur.hbm_bytes:,} -> {new.hbm_bytes:,} bytes, "
                  f"{cur.transfers:,} -> {new.transfers:,} transfers  ({time.perf_counter() - t0:.1f}s)")
            cur = new
        else:
            print(f"  speed link {s}: nothing both correct and faster "
                  f"({', '.join(sorted({f.category for f in memory})) or 'no answers'})  "
                  f"({time.perf_counter() - t0:.1f}s)")
    return cur if cur is not start else None


def solve_chain(a, level, log, lib, run_tag, carry):
    """One run of one level as an ensemble chain. `carry` is this level's error memory shared across
    runs; it is read and extended only with --carry-errors."""
    spec = nkibench.LEVELS[level]
    print(f"\n=========== level {level}: {spec['op']}  (chain: up to {a.links} links x {a.samples} "
          f"samples) ===========")
    memory = list(carry) if a.carry_errors else []
    failed = {f.key: f.tag for f in memory if f.key}
    if memory:
        print(f"  starting from {len(memory)} error(s) carried over from earlier runs")
    cats = collections.Counter()
    best, solved, terse, links_used = None, None, a.terse, 0

    for link in range(a.links):
        links_used = link + 1
        t0 = time.perf_counter()
        kinds = chain_plan(link, a.samples, bool(best and best.source.strip()))
        built = [chain_prompt(k, level, a, memory, best, terse) for k in kinds]
        replies = _ask_link(a, level, [b[0] for b in built], link)

        graded, this_link = [], {}
        for i, (kind, (prompt, secs, dropped), rep) in enumerate(zip(kinds, built, replies)):
            tag = f"L{link}.s{i}"
            code = base.extract_code(rep["content"])
            key = code_key(code)
            dup = key and (failed.get(key) or this_link.get(key))
            if dup:
                v = Verdict(source=code, category="duplicate",
                            feedback=f"identical to attempt {dup}, which already failed. Write different "
                                     f"code that avoids that attempt's error.")
            else:
                v = grade(code, level, a)
                if rep["finish"] == "length" and v.category in ("parse", "no_code"):
                    v.category = "truncated"
            if key:
                this_link.setdefault(key, tag)
            cats[v.category] += 1
            _log_attempt(log, run_tag, level, link, i, kind, key, dup, v, secs, dropped, rep, "solve")
            graded.append((tag, kind, key, v))
        log.flush()

        tag, kind, _, top = max(graded, key=lambda t: (t[3].verified, t[3].shaping))
        if best is None or top.shaping > best.shaping:
            best = top
        n_dup = sum(v.category == "duplicate" for *_, v in graded)
        mix = collections.Counter(kinds)
        print(f"link {link}: {top.reward:.2f} ({top.category}, {kind} {tag})  best {best.reward:.2f}  "
              f"[{', '.join(f'{k} x{n}' for k, n in mix.items())}]  {n_dup} repeat(s) rejected  "
              f"{max(sum(b[1].values()) for b in built)} prompt tok  ({time.perf_counter() - t0:.1f}s)")
        if top.autofixes:
            print(f"  autofix: {'; '.join(top.autofixes)}")
        print(textwrap.indent(textwrap.shorten(top.feedback, 420), "  "))

        if top.verified:
            solved = top
            print(f"  SOLVED and VERIFIED on link {link}" + (f" (held-out {top.heldout})" if top.heldout else "")
                  + f"; {top.hbm_bytes:,} HBM bytes in {top.transfers:,} transfers")
            break

        for tag_i, _, key_i, v in graded:
            if v.category == "duplicate":
                continue
            memory.append(Failure(tag_i, key_i, v.category, v.feedback))
            if key_i:
                failed.setdefault(key_i, tag_i)
        if n_dup:
            memory.append(Failure(f"L{link}", "", "repeat",
                                  f"{n_dup} answer(s) were the same code as an attempt that had already "
                                  f"failed. Do not resend old code."))
        if not any(v.source.strip() for *_, v in graded):
            terse = min(terse + 1, 2)

    if a.carry_errors:
        carry[:] = [Failure(f"{run_tag.split('-')[-1]}/{f.tag}", f.key, f.category, f.feedback)
                    if "/" not in f.tag else f for f in memory][-24:]

    faster = None
    if solved:
        if lib is not None and not a.offline:
            lib.add(level, solved.source, run_tag)
        if a.speed_links:
            faster = speed_up(a, level, log, run_tag, solved)
        final = faster or solved
        print("  ---------------- the kernel ----------------")
        print(textwrap.indent(final.source, "  "))
        print("  -------------------------------------------")
        line = (f"VERIFIED on link {links_used - 1} ({links_used} of {a.links} used, held-out "
                f"{solved.heldout or 'n/a'}); "
                f"{final.hbm_bytes:,} HBM bytes"
                + (f", down from {solved.hbm_bytes:,} after the speed links" if faster else
                   (", no faster version found" if a.speed_links else "")))
        conf = 1.0
    else:
        b = best or Verdict()
        fails = [(c, n) for c, n in cats.most_common() if c != "solved"]
        line = (f"NOT SOLVED after {links_used} link(s): best reward {b.reward:.2f}, {b.shapes_passed} "
                f"shape(s) passing. Dominant failures: {', '.join(f'{c} x{n}' for c, n in fails[:3]) or 'none'}")
        conf = 0.0
    print(f"  {line}  confidence {conf:.2f}")
    b = solved or best or Verdict()
    return dict(reward=b.reward, verified=bool(solved), rounds=links_used, confidence=conf,
                taxonomy=dict(cats), line=line, faster=bool(faster),
                hbm_bytes=(solved.hbm_bytes if solved else None),
                hbm_bytes_final=((faster or solved).hbm_bytes if solved else None))


# ================================================================ selftest: everything that needs no device

def selftest():
    print("Proving the parts that need no model and no device.\n")
    rc = 0

    def check(what, ok, extra=""):
        nonlocal rc
        rc |= 0 if ok else 1
        print(f"  {what:<52} -> {'ok' if ok else 'FAIL'} {extra}")

    # autofix
    src = ("from neuronxcc import nki\nimport neuronxcc.nki.language as nl\n\n"
           "def my_kernel(lhsT, rhs):\n    return lhsT\n\nprint(my_kernel(1, 2))\n")
    fixed, done = autofix(src, "nki_matmul_tiled_")
    check("autofix: old import path rewritten", "neuronxcc" not in fixed)
    check("autofix: lone function renamed to the entry", "def nki_matmul_tiled_(" in fixed)
    check("autofix: @nki.jit added", "@nki.jit\ndef nki_matmul_tiled_" in fixed)
    check("autofix: module-level test call removed", "print(" not in fixed)
    check("autofix: result passes the rule check", not nkibench.check_rules(fixed, 4),
          str(nkibench.check_rules(fixed, 4)))
    ref4 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_level4.py")).read()
    check("autofix leaves a correct kernel alone", autofix(ref4, "nki_matmul_tiled_")[1] == [],
          str(autofix(ref4, "nki_matmul_tiled_")[1]))

    # C0 lint
    bad = ("import nki\nimport nki.isa as nisa\nimport nki.language as nl\n@nki.jit\n"
           "def f(a):\n    ps = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)\n"
           "    t = nl.load(a)\n    x = nisa.tensor_copy(src=ps)\n    nisa.dma_copy(dst=a, src=ps)\n")
    ids = {h[0] for h in stale_api_lint(bad)}
    check("lint: PSUM -> HBM dma is caught, and blocking", "psum_dma" in ids and any(
        h[3] for h in stale_api_lint(bad) if h[0] == "psum_dma"))
    check("lint: nl.load and `x = nisa.f()` are caught", {"nl_load", "isa_return"} <= ids, str(ids))
    for lv in (1, 2, 3, 4):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"reference_level{lv}.py")
        hits = stale_api_lint(open(p).read())
        check(f"lint: shipped reference_level{lv}.py is clean", not hits, str(hits))

    # fingerprinting
    print()
    case = nkibench.LEVELS[4]["shapes"][1]           # K=256: two K chunks
    args, _ = nkibench.make_inputs(case, 4)
    want = nkibench.ref_matmul(*args)
    last = args[0][128:].T.astype(np.float64) @ args[1][128:]
    d = diagnose_wrong_answer(4, args, last, want)
    check("fingerprint: PSUM reset inside the K loop", d and d[1] == "k_accum_reset", str(d and d[1]))
    first = args[0][:128].T.astype(np.float64) @ args[1][:128]
    d = diagnose_wrong_answer(4, args, first, want)
    check("fingerprint: K slice that never moves", d and d[1] == "k_slice_static", str(d and d[1]))
    moved = want.copy()
    moved[0:128, 0:512], moved[128:256, 0:512] = want[128:256, 0:512], want[0:128, 0:512]
    d = diagnose_wrong_answer(4, args, moved, want)
    check("fingerprint: tiles stored to the wrong place", d and d[1] == "tile_indexing", str(d and d[1]))
    part = want.copy()
    part[128:256, 512:1024] += 1.0
    d = diagnose_wrong_answer(4, args, part, want)
    check("fingerprint: some tiles right, some wrong", d and d[1] == "partial_tiles", str(d and d[1]))
    pargs, _ = nkibench.make_inputs(nkibench.LEVELS[1]["shapes"][0], 1)
    pw = nkibench.ref_avgpool2d(*pargs)
    d = diagnose_wrong_answer(1, pargs, pw * 4, pw)
    check("fingerprint: pooling sum instead of mean", d and "SUM" in d[0], str(d and d[0][:50]))
    targs, _ = nkibench.make_inputs(nkibench.LEVELS[2]["shapes"][0], 2)
    d = diagnose_wrong_answer(2, targs, targs[0], nkibench.ref_transpose2d(*targs))
    check("fingerprint: transpose that moved nothing", d and d[1] == "no_op", str(d and d[1]))
    aargs, _ = nkibench.make_inputs(nkibench.LEVELS[8]["shapes"][0], 8)
    q, k, v = (x.astype(np.float64) for x in aargs)
    d = diagnose_wrong_answer(8, aargs, _softmax(q @ k.T) @ v, nkibench.ref_attention(*aargs))
    check("fingerprint: attention missing 1/sqrt(d)", d and d[1] == "wrong_scale", str(d and d[1]))
    d = diagnose_wrong_answer(4, args, want * 0.5, want)
    check("fingerprint: constant-factor error", d and "0.5" in d[0], str(d and d[0][:60]))
    check("fingerprint: a correct output is not diagnosed", diagnose_wrong_answer(4, args, want, want) is None)
    check("closeness orders near misses above nonsense",
          closeness(part, want) > closeness(np.random.default_rng(0).standard_normal(want.shape), want))

    # prompts and budget
    print()

    class A:
        context, max_tokens, cards = 4096, 2500, True
    sections = [("task", "x" * 3000, 0), ("card", "y" * 3000, 2), ("ledger", "z" * 3000, 3)]
    _, secs, dropped = pack(sections, 2000)       # each section is ~910 tokens
    check("pack drops the least important first", dropped == ["ledger"] and "card" in secs,
          f"{dropped} {list(secs)}")
    _, secs, dropped = pack(sections, 500)
    check("pack never drops a priority-0 section", "task" in secs)
    lib = Library(None)
    lib.add(3, "def k(): pass", "t")
    check("library records a verified kernel", len(lib.items) == 1 and lib.items[0]["level"] == 3)

    # the ensemble chain
    print()
    k1 = code_key("import nki\n\n@nki.jit\ndef f(a):\n    return a\n")
    k2 = code_key("import nki  # the module\n@nki.jit\ndef f(a):   # entry\n\n    return a\n")
    k3 = code_key("import nki\n@nki.jit\ndef f(a):\n    return a + 1\n")
    check("chain: comments and spacing do not make code 'different'", k1 == k2)
    check("chain: a real change does", k1 != k3)
    check("chain: no code has no identity", code_key("  \n") == "")
    fails = [Failure("L0.s0", "a", "shape_mismatch", "dma_copy requires the same number of elements"),
             Failure("L0.s1", "b", "api_invented", "nki.isa has no multiply"),
             Failure("L1.s0", "c", "shape_mismatch", "dma_copy requires the same number of elements")]
    mt = memory_text(fails)
    check("chain: memory groups a repeated error with a count", "seen 2x" in mt and mt.count("shape_mismatch") == 1,
          mt.replace("\n", " | ")[:120])
    check("chain: memory is empty when nothing has failed", memory_text([]) == "")
    check("chain: link 0 is all fresh", chain_plan(0, 4, True) == ["fresh"] * 4)
    check("chain: later links mix repair and fresh", set(chain_plan(1, 4, True)) == {"repair", "fresh"})
    check("chain: no code yet stays fresh", chain_plan(3, 4, False) == ["fresh"] * 4)

    many =[Failure(f"L{i}.s{j}", str(i * 4 + j), "numerics", "x" * 400 + str(i * 4 + j))
            for i in range(5) for j in range(4)]
    good = Verdict(source=open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "reference_level3.py")).read(),
                   feedback="0 of 1 shapes passed", category="shape_mismatch", hbm_bytes=1234, transfers=3)
    for kind in ("fresh", "repair", "speed"):
        prompt, secs, _ = chain_prompt(kind, 3, A, many, good)
        check(f"chain: `{kind}` prompt with 20 failures fits the budget",
              sum(secs.values()) <= input_budget(A) and "memory" in secs, f"{sum(secs.values())} tok {list(secs)}")
    check("chain: the traffic floor of level 3 is positive", traffic_floor(3) > 0)

    print("\nSELFTEST " + ("PASSED" if rc == 0 else "FAILED"))
    return rc


# ================================================================ cli

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--level", type=int, choices=sorted(nkibench.LEVELS))
    ap.add_argument("--levels", help="comma-separated, e.g. 1,2,3,4,8")
    ap.add_argument("--all", action="store_true", help="levels 1 to 4")
    ap.add_argument("--links", type=int, default=5, help="generations (links) per run of a level")
    ap.add_argument("--speed-links", type=int, default=1,
                    help="extra links after a verified kernel, asking for less HBM traffic")
    ap.add_argument("--carry-errors", action="store_true",
                    help="later runs start from earlier runs' errors (runs become dependent)")
    ap.add_argument("--samples", type=int, default=4, help="candidates per link: the ensemble size")
    ap.add_argument("--max-tokens", type=int, default=base.MIN_ANSWER_TOKENS)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--base", default=os.environ.get("KERNEL_AGENT_BASE_URL")
                    or os.environ.get("GPTOSS_BASE_URL"))
    ap.add_argument("--path", default="", help="path to append, e.g. /agg/v1 for gpt-oss")
    ap.add_argument("--repeat", type=int, default=1, help="report a solve RATE over N runs")
    ap.add_argument("--log", default="attempts2.jsonl")
    ap.add_argument("--library", default="kernel_library.jsonl",
                    help="file that records every kernel this agent verifies; 'none' to skip")
    ap.add_argument("--terse", type=int, default=0, choices=(0, 1, 2))
    ap.add_argument("--context", type=int, default=8192,
                    help="the server's max-model-len (8192 in the seat pods)")
    ap.add_argument("--heldout-seeds", type=lambda s: [int(x) for x in s.split(",") if x], default=[101, 202],
                    help="extra seeds a passing kernel must also pass; '' disables held-out checks")
    ap.add_argument("--no-autofix", dest="autofix", action="store_false")
    ap.add_argument("--no-cards", dest="cards", action="store_false")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(selftest())

    if not a.offline:
        raw = (a.base or "").strip()
        if not raw:
            sys.exit("KERNEL_AGENT_BASE_URL is empty or unset. In a seat pod it is set for you "
                     "(http://localhost:8000/v1, after ./serve.sh). Or pass --offline or --selftest.")
        from urllib.parse import urlparse
        u = urlparse(raw)
        if u.scheme not in ("http", "https") or not u.netloc:
            sys.exit(f"base URL {raw!r} needs a scheme and a host, e.g. http://localhost:8000/v1.")
        a.base = raw.rstrip("/") + a.path
        print(f"endpoint {a.base}  model {a.model}")
    else:
        print("*** OFFLINE: replaying the reference kernel. Numbers are meaningless. ***")
    print(f"ensemble chain: up to {a.links} links x {a.samples} samples per run, {a.speed_links} speed "
          f"link(s) after a verified kernel, error memory "
          f"{'carried across runs (runs NOT independent)' if a.carry_errors else 'per run'}")
    print(f"context {a.context}: input budget {input_budget(a)} tok, answer reserve {answer_reserve(a)}; "
          f"autofix {'on' if a.autofix else 'off'}, cards {'on' if a.cards else 'off'}, "
          f"held-out seeds {a.heldout_seeds or 'off'}")

    lib = None if a.library.lower() == "none" else Library(a.library)
    if lib is not None:
        print(f"library {a.library}: {len(lib.items)} verified kernel(s) recorded so far; it is written, "
              f"never shown to the model, so it does not couple the runs")

    if a.levels:
        levels = [int(x) for x in a.levels.split(",")]
    else:
        levels = sorted(nkibench.LEVELS)[:4] if a.all else [a.level or 1]
    history = {lv: [] for lv in levels}
    carry = {lv: [] for lv in levels}
    stamp = time.strftime("%Y%m%d-%H%M%S")

    with open(a.log, "a") as log:
        for rep in range(a.repeat):
            if a.repeat > 1:
                print(f"\n################ run {rep + 1} of {a.repeat} ################")
            tag = f"{stamp}-r{rep + 1}"
            results = []
            for level in levels:
                r = solve_chain(a, level, log, lib, tag, carry[level])
                results.append((level, r))
                history[level].append(r)
            print("\n=========== summary ===========")
            for level, r in results:
                print(f"  level {level}  reward {r['reward']:.2f} after {r['rounds']} link(s)  {r['line']}")
            print(f"  verified {sum(r['verified'] for _, r in results)}/{len(results)}")
            log.write(json.dumps(dict(summary=True, run=tag, results={lv: r for lv, r in results})) + "\n")

    if a.repeat > 1:
        print(f"\n=========== over {a.repeat} runs ===========")
        for lv in levels:
            got = history[lv]
            rewards = [r["reward"] for r in got]
            print(f"  level {lv}: verified {sum(r['verified'] for r in got)}/{len(got)}  "
                  f"best {max(rewards):.2f}  worst {min(rewards):.2f}  mean {np.mean(rewards):.2f}  "
                  f"all={[round(x, 2) for x in rewards]}  links={[r['rounds'] for r in got]}")
            if any(r["verified"] for r in got):
                sped = [r for r in got if r.get("faster")]
                print(f"           speed links found a faster verified kernel in {len(sped)} of "
                      f"{sum(r['verified'] for r in got)} solved run(s)"
                      + (": HBM bytes " + ", ".join(f"{r['hbm_bytes']:,}->{r['hbm_bytes_final']:,}"
                                                     for r in sped) if sped else ""))
        print("\n  Report the rate, not your best run.")

    tax = collections.Counter()
    for got in history.values():
        for r in got:
            tax.update(r["taxonomy"])
    tax.pop("solved", None)
    if tax:
        print("\n=========== failure taxonomy (every graded attempt) ===========")
        for c, n in tax.most_common():
            print(f"  {n:5d}  {c}")
    print(f"\nattempts logged to {a.log}")


if __name__ == "__main__":
    main()
