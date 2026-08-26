# Prefill Optimization Recap (VK Engine)

Recap of the prefill (prompt-processing) optimization campaign on the RX 580 (36 CUs, wave64, ~250 GB/s). All measurements: **16k-token prompt**, `MODEL_PREFILL_CHUNK = 512`, logged per-op timings (`main.exe log` â†’ `timing_log.txt`), acceptance = validators green + 128 all-zero tokens.

**Headline: prefill throughput 31.92 â†’ 58.66 t/s (+84%), full 16k run 430 s â†’ ~313 s wall (incl. 128 decode tokens). GPU is ~90â€“95% busy (283 s of 313 s) â€” no idle gaps left.**

---

## 1. The tooling that made this possible

- **Per-op GPU timing log** (`dispatch.c`: `logLastFrame` / `timing_log.txt`): every op records `calls= total= avg=`, so each kernel can be measured independently instead of only wall time.
- **Chunked prefill** (`runPrefill`/`compilePrefill`/`executeChunked`, `generate.c`): the 16k prompt runs as 32 chunks of 512 tokens; each chunk is compiled with `m = cur` and offset-patched push constants, then submitted in â‰¤8-op slices with waits in between. This (a) fixed the OOM of a single 16kÃ—M GEMM pass on the 8 GB card, and (b) fixed TDR (`VkResult -4 = VK_ERROR_DEVICE_LOST`) by keeping any single submission under the Windows 8 s TDR budget.
- **A/B discipline**: change one kernel â†’ `make clean && make` from the repo root â†’ `val` â†’ 16k logged run â†’ keep if faster, undo if not.

## 2. Phase summary

| Phase | Change | 16k prefill t/s | Notes |
|---|---|---|---|
| 0 | baseline + logging | **31.92** | original fused GEMM kernels (dual-matrix swiglu, fused RMSNorm per layer) |
| 1 | `RmsNorm-Prologue.comp` + swiglu GEMM2 | 35.72* | invRms computed once per chunk instead of per layer |
| 2 | register-blocked GEMM2 cores (TN=32/64), `GEMM-ADD2`, `LinearProj-GEMM2` | **39.60** | 4096-ctx runs reached 44.86â€“45.67 t/s |
| B | unfused attention: `Att-QK2-*` + `Att-Softmax` + `Att-PV2-*` | **47.96** | attention 48.7 s â†’ 13.2 s; causal-mask fix (chunk-absolute limits) |
| 3a | swiglu-INT4 TN=32 probe | **52.66** | 159.2 â†’ 115.6 ms/call |
| 3b | flattened swiglu kernel (INT4) | **52.82** | 111.8 ms + combine; silu out of inner loop |
| 3c | rollout: flat INT8 kept, flat FP16 **reverted** | **53.10** | INT8 113.3 â†’ 110.3 ms; FP16 flat 103.8 vs dual 91.4 â†’ undo |
| 3d/3e | flat swiglu TN=64 probes (INT4 / INT8) | reverted | 168.6 / 167.8 ms vs 111.8 / 110.3 (+50-52%) - TN=64 toxic on this family |
| 4 | softmax normalize folded into PV2 (`smSum` reciprocal buffer) | **53.21** | softmax pool 3.2 s -> 0.78 s |
| 5 | `GEMM-ADD2-INT4` ping-pong k-tile double-buffer | **53.41** | ADD2-INT4 60.4 -> 58.9 ms/call |
| P1 | QKV split: `RmsNorm-QKV-GEMM2-*` + `Rope-GEMM-*`; fixed-stride KV scales | **56.73** | QKV chain 24.6 -> 8.7 s; latent KV-scale layout bug found & fixed (sec 7) |
| P2 | Embed pre-gather: `Embed-Gather.comp` + LP-GEMM2 over staged embeddings | **58.30** | embed path 8.5 -> ~2.1 s |
| L-A | flat swiglu TS=32 probe (1024-thread WG) | reverted | +63% - small WGs win (sec 8) |
| L-B | flat swiglu ping-pong (INT4+INT8) | **58.44** | INT4 -1.1%, INT8 -2.4% |
| L-C | `GEMM-ADD2-INT8/FP16` ping-pong | **58.46** | INT8 -1.4%, FP16 -0.2% |
| L-D | `LinearProj-GEMM2-*` ping-pong x3 quants | reverted | INT4 +1.5%; narrow-N grids can't hide prefetch |
| L-E | dual-FP16 swiglu ping-pong | **58.66** | 91.5 -> 89.7 ms/call |

*Phase 1/2 figures were measured at 4096-ctx first; the 16k series starts at Phase 2's 39.60.

## 3. The swiglu flatten

**Problem:** the dual-matrix swiglu kernel (gate + up side by side, 2Ã— B tiles) at INT4/TN=64 took 159.2 ms/call â€” more than 3Ã— the LP-GEMM2's 115.1 ms for 2Ã— the work. INT8/FP16 at TN=32 were already at exact 2Ã— parity.

- **Step 1 â€” TN=32 probe (kept):** 159.2 â†’ 115.6 ms (âˆ’27%), exactly the predicted 2Ã—LP parity.
- **Step 2 â€” flat kernel (kept for INT4/INT8):** `RmsNorm-swiglu-flat-GEMM2-{INT4,INT8}` flattens the grid over 2Ã—FFN_N columns; each workgroup loads from **one** weight matrix (`upHalf = nBase >= off`) and does no silu in the inner loop; raw dots go to `gAct`/`uAct`; `Swiglu-combine.comp` (elementwise, ~0.6 ms/call, 0.6 s per 16k prompt) applies `silu(gAct)Â·uAct`.
- **Rollout:** INT8 flat 110.3 ms (kept, small win); FP16 flat **103.8 ms vs 91.4 ms dual â†’ reverted** (a 13% regression â€” the unpackHalf2x16 dual loader with 2-block indexing wins for fp16). Final per-kernel state:

| swiglu variant | before | after |
|---|---|---|
| INT4 | 159.2 ms (dual, TN=64) | **111.8 ms** (flat) |
| INT8 | 113.3 ms (dual) | **110.3 ms** (flat) |
| FP16 | 91.4 ms (dual) | 91.4 ms (dual kept) |

## 4. The critical bug this round found (prologue binding)

Since the GEMM2 campaign started, both `RmsNorm-Prologue` ops were bound `{h, gammaF}` / `{h, gammaIn}` â€” i.e. the prologue **wrote invRms into the gamma weight** while `st->invRms` stayed zero. Every GEMM2-family projection (swiglu gate/up, LinearProj) therefore computed **zeros end-to-end** from Phase 1 until the fix.

Why it survived so long:
- Wall timings were unaffected (zeros cost the same FLOPs), so every "speed win" was real, just with dead math.
- The demo collapses to token 0 regardless (synthetic weights), so the token stream didn't flag it.
- The validators use their own wiring and never touched `st->invRms`.

Fix (one line per call site): `proBufs[] = {st->h, st->invRms}` in `buildFfn` and `addLinearProj` (`generate.c`). Lesson: a GEMM2 kernel's inputs must be traced back to their producers, not assumed â€” check the *binding* of a prologue whose output is a scratch buffer.

## 5. Phase B (unfused attention) â€” recap

Replaced `Att-full-GEMM-*` (fused online-softmax, K and V staged in the same shared array, per-element probabilities re-read from the clobbered reduction array) with three passes over an fp16 score buffer `attScores` (134 MB at M=512):

- `Att-QK2-*`: 16 q-tokens Ã— 64 kv-tokens per workgroup, TN=64 K tiles, causal limit `qOff + mGlobal` â€” **chunk-absolute**, fixing the old kernel's chunk-local mask that under-attended history in chunked prefill.
- `Att-Softmax`: subgroup max â†’ cross-subgroup via 4 shared slots, `exp(s âˆ’ gmax)` in place, `Ã— 1/total`.
- `Att-PV2-*`: padded LDS (`Vsub[16][17]`), 16 output dims/thread.

Result: attention 48.7 s â†’ 13.2 s (QK2 â‰ˆ 6.9 s, SM â‰ˆ 3.2 s, PV2 â‰ˆ 3.1 s); 16k prefill 39.69 â†’ 47.96 t/s.

## 6. Structural splits (rounds P1 / P2)

### P1 - QKV projection split

The fused prefill QKV-GEMM (head-wide workgroups, 256-wide B tiles, RoPE + cache writes inside) was the least efficient GEMM2 (~96 ms/call fused, 24.6 s total). Split it into two passes sharing a raw-projection scratch (`st->qkvRaw`):

- `RmsNorm-QKV-GEMM2-*`: register-blocked GEMM2 over all 12288 QKV columns into `qkvRaw` (INT4 TN=64/4acc, INT8/FP16 TN=32/2acc).
- `Rope-GEMM-*`: grid (24 heads x m rows); v-heads store raw to cache, q/k heads do per-head RMSNorm + RoPE and write q/cache.

QKV chain 24.6 -> 8.7 s; prefill 53.41 -> 56.73 t/s. Side effects: prefill shaders no longer touch the position buffer at all (host sets it once via `stateSetPosition` after the last chunk), and the chunk offset is threaded into builders instead of patched post-compile.

**P1 also uncovered a latent correctness bug** - see section 7.

### P2 - Embedding pre-gather

Layer-0's embed step used `Embed-RmsNorm-LinearProj-GEMM-FP16`, whose tied-lm-head fetch is column-major (`lm[(i/2)*V + tok]`) *inside* the GEMM's A-tile staging (~150 ms at M=64 in the harness; 8.5 s per 16k prompt). Replaced with:

1. `Embed-Gather.comp` (precision-agnostic): grid (m, 1); gathers each token's fp32 embedding row once into contiguous `st->embStaged` (+ copy to `st->embOut` for the residual) - ~0.42 ms/call.
2. Plain `RmsNorm-Prologue` + `RmsNorm-LinearProj-GEMM2-FP16` over `embStaged`.

Embed path 8.5 -> ~2.1 s; prefill 56.73 -> 58.30 t/s. Lesson: **hoist irregular memory patterns out of compute kernels** - a gather that is terrible inside a GEMM is nearly free as a standalone pass.

## 7. The latent KV-scale layout bug (found during P1)

INT4/INT8 KV-cache scale/zero writers used context-dependent strides:

- prefill writer: `kvh*(tokenIdx+M)+absTok`
- decode writer: `kvh*(pos+1)+pos`
- readers: assumed whatever the *current* chunk's stride was

So every token written by an earlier chunk had its scale read from the wrong slot once the cache grew past the writing chunk's stride - silent corruption affecting only quantized FULL-attention layers, masked by the all-zero demo signature. Fix: a single fixed stride `kvh * MODEL_MAX_CTX + token` in every writer (`Rope-GEMM`, decode `Reduce-Rope`) and reader (`Att-QK2/PV2/SplitK2/full` INT8+INT4).

Why the validators caught it: their fixtures still built scales at the old stride, so `val` went red immediately (`Attention INT8`, `SplitK2 INT8/INT4`, `QKV-Split-Rope-kv INT8/INT4`). Lesson: **GPU-side layout changes are contracts - grep every validator fixture and update it in the same change**, and never trust a green `val` that was run against stale fixtures. Also: one val run showed a transient NaN in an untouched shader - re-run twice before hunting ghosts.

## 8. Tile geometry & the capacity ceiling

Every configuration of the GEMM2 template has now been measured on this GCN4 target:

| probe | result |
|---|---|
| TN=32, TS=16 (256-thread WG) | **optimal everywhere** |
| TN=64 (flat swiglu INT4/INT8) | +50-52% regression |
| TS=32 (1024-thread WG) | +63% regression |
| dual-matrix vs flat | flat wins INT4/INT8, loses FP16 |
| ping-pong k-tile prefetch (wide-N kernels: FFN, ADD2) | -1..-2.4%, kept on all four |
| ping-pong on narrow-N kernels (LP N=12320) | INT4 +1.5%, rest flat - reverted |

Interpretation: occupancy and latency-hiding dominate barrier count on RX 580; bigger tiles trade waves for fewer syncs and lose decisively. Ping-pong only pays when grid.x is wide enough to keep the CU saturated while prefetching. With GPU busy at 90-95% of wall and every kernel within ~15% of weight-streaming bandwidth for its quant, further shader-level gains would require a different parallel decomposition (not template tweaks) for single-digit-second returns - declared at practical capacity at **58.66 t/s**.

## 9. Current 16k prefill profile (final, ~283 s GPU / 313 s wall)

| pool | time | share |
|---|---|---|
| swiglu flat INT4/INT8 + dual FP16 + combine | ~110.9 s | ~39% |
| GEMM-ADD2 (INT4 66.0 / INT8 20.3 / FP16 9.8) | ~96.1 s | ~34% |
| LinearProj-GEMM2 (INT4/INT8/FP16) | ~43.4 s | ~15% |
| QKV chain (GEMM2 8.5 s + Rope 0.13 s) | ~8.7 s | ~3% |
| attention (QK2 8.2 / SM 0.5 / PV2 1.7) | ~10.4 s | ~4% |
| GatedDeltaNet-GEMM + embed gather + long tail | ~13 s | ~5% |

Per-kernel averages (M=512 chunk): swiglu-flat INT4 110.55 ms / INT8 107.45 ms / FP16-dual 89.71 ms; ADD2-INT4 58.95 / INT8 33.36 / FP16 30.56 ms; LP-GEMM2 57.95 / 56.57 / 51.35 ms; QKV-GEMM2 42.24 / 29.55 / 25.62 ms.

## 10. Lessons learned

1. **Measure per-kernel, not wall time** - the timing log turned guesswork into targeted fixes.
2. **The dual-matrix penalty is real but not uniform**: at TN=64 (INT4) it cost +38%; at TN=32 (INT8/FP16) it vanished; flattening helped INT4/INT8 but **hurt** FP16. Always A/B per quant.
3. **Splitting attention into passes** (QK / softmax / PV) with a persistent score buffer beats a monolithic flash kernel on this GPU's shared-memory-limited CUs; folding the normalize into PV2 via a reciprocal side-buffer removed another full pass.
4. **Trace prologue outputs to their producers.** Scratch-buffer bindings are the easiest silent corruption vector; a zero-filled state buffer is indistinguishable from correct math at a distance.
5. **Keep the timing-log + val + token-signature triple** as the acceptance gate; it caught the flat-FP16 regression the wall time alone would have buried (plus-1 s noise).
6. **A `make` from the wrong directory silently no-ops** (see summary gotcha 12) - always rebuild from the repo root and verify `main.exe` mtime vs `generate.c`.
7. **timing_log.txt accumulates across runs** - clear it before every measured run or your pool totals double-count (this bit us once; call counts that don't match the expected per-run count are the tell).
8. **Layout changes are validator contracts**: when a GPU-side layout moves (e.g., KV scale stride), every validator fixture must move in the same commit.
9. **Small workgroups win on GCN4**: TN=64/TS=32 probes regressed 50%+ despite halving barriers; occupancy > synchronization. Ping-pong prefetch is worth exactly -1..-2% and only on wide grids.
10. **Hoist irregular access patterns out of GEMMs**: the tied-embedding column fetch cost 150 ms inside a GEMM but 0.4 ms as a standalone gather pass.

## 11. Closing status

Campaign complete at **58.66 t/s** (31.92 -> 58.66, +84%; wall 430 -> ~313 s incl. decode). Remaining pools are either at their measured optimum across all tested configurations (swiglu, ADD2) or structurally different kernels (Att-QK2 ~8 s, GDN-GEMM ~7 s) whose next gains need ground-up redesigns rather than template tuning. Decode (~17 t/s cumulative at 16k ctx) remains a separate frontier.