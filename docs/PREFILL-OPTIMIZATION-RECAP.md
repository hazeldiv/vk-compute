# Prefill Optimization Recap (VK Engine)

Recap of the prefill (prompt-processing) optimization campaign on the RX 580 (36 CUs, wave64, ~250 GB/s). All measurements: **16k-token prompt**, `MODEL_PREFILL_CHUNK = 512`, logged per-op timings (`main.exe log` → `timing_log.txt`), acceptance = validators green + 128 all-zero tokens.

**Headline: prefill throughput 31.92 → 53.10 t/s (+66%), full 16k run 430 s → ~349 s wall (incl. 128 decode tokens).**

---

## 1. The tooling that made this possible

- **Per-op GPU timing log** (`dispatch.c`: `logLastFrame` / `timing_log.txt`): every op records `calls= total= avg=`, so each kernel can be measured independently instead of only wall time.
- **Chunked prefill** (`runPrefill`/`compilePrefill`/`executeChunked`, `generate.c`): the 16k prompt runs as 32 chunks of 512 tokens; each chunk is compiled with `m = cur` and offset-patched push constants, then submitted in ≤8-op slices with waits in between. This (a) fixed the OOM of a single 16k×M GEMM pass on the 8 GB card, and (b) fixed TDR (`VkResult -4 = VK_ERROR_DEVICE_LOST`) by keeping any single submission under the Windows 8 s TDR budget.
- **A/B discipline**: change one kernel → `make clean && make` from the repo root → `val` → 16k logged run → keep if faster, undo if not.

## 2. Phase summary

| Phase | Change | 16k prefill t/s | Notes |
|---|---|---|---|
| 0 | baseline + logging | **31.92** | original fused GEMM kernels (dual-matrix swiglu, fused RMSNorm per layer) |
| 1 | `RmsNorm-Prologue.comp` + swiglu GEMM2 | 35.72* | invRms computed once per chunk instead of per layer |
| 2 | register-blocked GEMM2 cores (TN=32/64), `GEMM-ADD2`, `LinearProj-GEMM2` | **39.60** | 4096-ctx runs reached 44.86–45.67 t/s |
| B | unfused attention: `Att-QK2-*` + `Att-Softmax` + `Att-PV2-*` | **47.96** | attention 48.7 s → 13.2 s; causal-mask fix (chunk-absolute limits) |
| 3a | swiglu-INT4 TN=32 probe | **52.66** | 159.2 → 115.6 ms/call |
| 3b | flattened swiglu kernel (INT4) | **52.82** | 111.8 ms + combine; silu out of inner loop |
| 3c | rollout: flat INT8 kept, flat FP16 **reverted** | **53.10** | INT8 113.3 → 110.3 ms; FP16 flat 103.8 vs dual 91.4 → undo |

*Phase 1/2 figures were measured at 4096-ctx first; the 16k series starts at Phase 2's 39.60.

## 3. The swiglu flatten (last round)

**Problem:** the dual-matrix swiglu kernel (gate + up side by side, 2× B tiles) at INT4/TN=64 took 159.2 ms/call — more than 3× the LP-GEMM2's 115.1 ms for 2× the work. INT8/FP16 at TN=32 were already at exact 2× parity.

- **Step 1 — TN=32 probe (kept):** 159.2 → 115.6 ms (−27%), exactly the predicted 2×LP parity.
- **Step 2 — flat kernel (kept for INT4/INT8):** `RmsNorm-swiglu-flat-GEMM2-{INT4,INT8}` flattens the grid over 2×FFN_N columns; each workgroup loads from **one** weight matrix (`upHalf = nBase >= off`) and does no silu in the inner loop; raw dots go to `gAct`/`uAct`; `Swiglu-combine.comp` (elementwise, ~0.6 ms/call, 0.6 s per 16k prompt) applies `silu(gAct)·uAct`.
- **Rollout:** INT8 flat 110.3 ms (kept, small win); FP16 flat **103.8 ms vs 91.4 ms dual → reverted** (a 13% regression — the unpackHalf2x16 dual loader with 2-block indexing wins for fp16). Final per-kernel state:

| swiglu variant | before | after |
|---|---|---|
| INT4 | 159.2 ms (dual, TN=64) | **111.8 ms** (flat) |
| INT8 | 113.3 ms (dual) | **110.3 ms** (flat) |
| FP16 | 91.4 ms (dual) | 91.4 ms (dual kept) |

## 4. The critical bug this round found (prologue binding)

Since the GEMM2 campaign started, both `RmsNorm-Prologue` ops were bound `{h, gammaF}` / `{h, gammaIn}` — i.e. the prologue **wrote invRms into the gamma weight** while `st->invRms` stayed zero. Every GEMM2-family projection (swiglu gate/up, LinearProj) therefore computed **zeros end-to-end** from Phase 1 until the fix.

Why it survived so long:
- Wall timings were unaffected (zeros cost the same FLOPs), so every "speed win" was real, just with dead math.
- The demo collapses to token 0 regardless (synthetic weights), so the token stream didn't flag it.
- The validators use their own wiring and never touched `st->invRms`.

Fix (one line per call site): `proBufs[] = {st->h, st->invRms}` in `buildFfn` and `addLinearProj` (`generate.c`). Lesson: a GEMM2 kernel's inputs must be traced back to their producers, not assumed — check the *binding* of a prologue whose output is a scratch buffer.

## 5. Phase B (unfused attention) — recap

Replaced `Att-full-GEMM-*` (fused online-softmax, K and V staged in the same shared array, per-element probabilities re-read from the clobbered reduction array) with three passes over an fp16 score buffer `attScores` (134 MB at M=512):

- `Att-QK2-*`: 16 q-tokens × 64 kv-tokens per workgroup, TN=64 K tiles, causal limit `qOff + mGlobal` — **chunk-absolute**, fixing the old kernel's chunk-local mask that under-attended history in chunked prefill.
- `Att-Softmax`: subgroup max → cross-subgroup via 4 shared slots, `exp(s − gmax)` in place, `× 1/total`.
- `Att-PV2-*`: padded LDS (`Vsub[16][17]`), 16 output dims/thread.

Result: attention 48.7 s → 13.2 s (QK2 ≈ 6.9 s, SM ≈ 3.2 s, PV2 ≈ 3.1 s); 16k prefill 39.69 → 47.96 t/s.

## 6. Current 16k prefill profile (post-campaign, ~300 s GPU time)

| kernel | time (16k prompt) | share |
|---|---|---|
| swiglu flat/dual + combine | ~71.9 s | ~24% |
| GEMM-ADD2-INT4 (TN=64) | ~67.6 s (60.4 ms/call) | ~22% |
| LinearProj-GEMM2 | ~41.5 s | ~14% |
| QKV-GEMM | ~24.6 s | ~8% |
| attention (QK2+SM+PV2) | ~13.2 s | ~4% |
| Embed-GEMM + GatedDeltaNet | ~14.3 s | ~5% |

## 7. Lessons learned

1. **Measure per-kernel, not wall time** — the timing log turned guesswork into targeted fixes.
2. **The dual-matrix penalty is real but not uniform**: at TN=64 (INT4) it cost +38%; at TN=32 (INT8/FP16) it vanished; flattening helped INT4/INT8 but **hurt** FP16. Always A/B per quant.
3. **Splitting attention into passes** (QK / softmax / PV) with a persistent score buffer beats a monolithic flash kernel on this GPU's shared-memory-limited CUs.
4. **Trace prologue outputs to their producers.** Scratch-buffer bindings are the easiest silent corruption vector; a zero-filled state buffer is indistinguishable from correct math at a distance.
5. **Keep the timing-log + val + token-signature triple** as the acceptance gate; it caught the flat-FP16 regression the wall time alone would have buried (±1 s noise).
6. **A `make` from the wrong directory silently no-ops** (see summary gotcha 12) — always rebuild from the repo root and verify `main.exe` mtime vs `generate.c`.

## 8. Where the remaining time sits (next candidates)

- **GEMM-ADD2-INT4 (22%)**: TN=64; register-blocking is maxed, so next levers are k-tile barrier reduction (KS=32) or s/z hoisting.
- **swiglu (24%)**: INT8/INT4 flat kernels are at 2×LP parity; further gains would need the same techniques as ADD2.
- **QKV-GEMM (8%)**: worst efficiency (0.15–0.29 TF/s, LDS-bound occupancy with the 256-wide B tiles).
- **Embed-GEMM (5%)**: column-major tied-embedding fetch (~150 ms at M=64 in the harness).
- Decode at 16k ctx (~16 t/s cumulative) is a separate frontier (attention decode split-K is the constraint).