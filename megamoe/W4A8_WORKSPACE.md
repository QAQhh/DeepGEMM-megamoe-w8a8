# W4A8 routing workspace and logging

The W4A8 entry point keeps route counts on the GPU. Its workspace capacity is
computed from host metadata, with no `Tensor.item()` or routing-count D2H copy:

```
max_routes = ep_ranks * tokens_per_rank_capacity * topk
rows = align_up(max_routes + local_experts * (alignment - 1), alignment)
```

Every route can contribute at most one row; each expert's alignment padding
adds at most `alignment - 1` rows. The bound covers skewed, ragged and empty
ranks without assuming balanced routing. GPU offsets still delimit the actual
gather and clamped activation work. Workspace tensors are cached on SymmBuffer
and reused until the capacity or GEMM geometry changes.

This trades a larger stable allocation for avoiding a synchronous route-count
read and repeated allocation. For V4 Pro EP8, capacity 4096 and alignment 64,
the workspace has 199680 rows and occupies 7364198400 bytes (about 6.86 GiB)
per rank. SGLang reuses one symmetric buffer across compatible MoE layers.

Workspace messages are silent by default. Enable them before starting Python:

```bash
export MEGAMOE_ENABLE_LOGGING=1
```

Set Python's logging level to INFO to display the messages. Extension build
verbosity remains separately controlled by MEGAMOE_DCU_OPT_VERBOSE_BUILD,
which also defaults to 0.

## Precision checks

Run from outside the repository so imports use the installed package:

```bash
torchrun --standalone --nproc-per-node=8 /path/to/repo/tests/test_w4a8_mega_moe.py --correctness-only
torchrun --standalone --nproc-per-node=8 /path/to/repo/tests/test_w4a8_mega_moe.py --correctness-only --hidden 7168 --intermediate 3072 --tokens 8
python /path/to/repo/tests/test_w4a8_mega_moe.py --gemm --correctness-only --hidden 7168 --intermediate 3072 --tokens 65
```

The tests cover uniform/skewed routing, different token counts per rank, empty
input, canonical EPLB reduction, alignment boundaries and invalid expert IDs.
The installed DeepGEMM HIPC reference uses high-nibble INT4 values (q * 16);
the test adapter divides the true-channel weight scales by 16 for that interface.
The standalone GEMM test also checks a direct integer-matmul oracle. SGLang
converts loader scales to true-channel scales by multiplying by 16 at load time.
