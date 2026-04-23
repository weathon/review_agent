Now let me run calibration searches in parallel.Now I have enough information to write the final consolidated review.

---

## Summary

FTP (FFN Token Pruning) is a training-free method for accelerating the prefilling stage of long-context LLM inference. It prunes "unimportant" tokens before FFN computation in each layer, with token importance determined by aggregated attention scores from the last N queries, and crucially preserves pruned tokens' hidden states via the residual connection (zeroing out the FFN delta). The method achieves 1.2–1.45× TTFT speedup across multiple models evaluated on LongBench, with mostly modest accuracy degradation.

---

## Strengths

- **Novel and well-motivated FFN-specific targeting**: Figure 3 demonstrates concretely that the FFN accounts for 62.4% and 61.3% of per-layer walltime for Llama3-8B and Qwen2-7B respectively. Prior token pruning work (LazyLLM, PyramidInfer) targets attention, not FFN. This is a distinctive and previously underexplored angle.

- **Elegant residual-connection trick**: By setting FFN outputs to zero-vectors for pruned tokens (Section 3.2), the residual connection automatically "passes through" the pruned tokens unchanged, requiring no sparse kernel implementation or separate data structure for dropped tokens. This is a clean architectural observation.

- **Dynamic per-layer pruning via reserved ratio η (Eq. 3)**: Rather than a fixed pruning ratio, the method adapts the number of retained tokens per layer based on the actual attention concentration in each layer. This is motivated by Figure 5, which shows layer-to-layer variability in attention concentration, and is a well-grounded design choice.

- **Strong ablation showing selection quality matters (Table 3)**: When FTP uses random token selection (same token counts as the attention-based variant), performance collapses catastrophically across all tasks (e.g., Single-Doc QA: 37.20→11.14; Synthetic: 37.00→2.72 on Llama3-8B). The attention-based approach retains nearly full accuracy. This is compelling and cleanly establishes the importance of the attention-based selection strategy.

- **Scalability to larger models (Table 2)**: Qwen1.5-32B achieves 1.37–1.45× speedup and Qwen2-72B achieves 1.31–1.36×, showing the technique generalizes beyond 7B-scale models with enhanced speedup (deeper architectures mean more layers benefit from pruning).

- **Low overhead for importance scoring (Section 4.6.1)**: The importance computation adds only 7–15ms (0.8–3% of TTFT), confirming the residual attention recomputation is not a bottleneck at the tested context lengths.

---

## Weaknesses

### Fatal
None.

### Major

- **Unacknowledged catastrophic failure on Llama3-8B Code Completion**: Table 3 (and Table 1) shows FTP drops Code Completion from 55.17 to 35.91 on Llama3-8B — a **35% relative accuracy loss** — while achieving only 1.19× speedup. The paper's core claim in the abstract of "only a 1.30% performance drop" (which is averaged across tasks for Qwen2-7B) and the characterization of FTP as producing "negligible decrease in performance" (Section 4.2) are directly contradicted by this result. Notably, the same task on Qwen2-7B drops only from 58.43 to 56.74 (~3% relative). The large inter-model discrepancy on the same task is never analyzed or explained. A 35% relative drop is not covered by "negligible," and presenting an average that conceals this outlier without acknowledgment is a substantive misrepresentation of the method's reliability profile.

- **LazyLLM — the most directly comparable baseline — is absent from Table 1**: Section 2.1 explicitly describes LazyLLM (Fu et al., 2024) as a method that drops tokens from the prefilling stage, characterizes it as a close prior, and then excludes it from the main comparison. The paper justifies this only by saying LazyLLM yields "subtle speedup during prefilling," but this is exactly the claim FTP aims to beat. Without a direct comparison under an identical inference stack, the claim of outperforming "state-of-the-art prefilling acceleration methods" is not fully established.

- **Evaluated context lengths (5k–15k) do not match the stated motivation of 128k contexts**: The paper opens by citing GPT-4, Qwen2, and Claude-3 as motivation for handling 128k+ contexts, yet all experiments use LongBench datasets with average lengths of 5,000–15,000 tokens (Section 4.1). The key practical question — whether FTP's speedup holds at the context lengths that motivated the work — is unaddressed.

### Minor

- **Misleading accuracy claims for larger model experiments (Section 4.5, Table 2)**: On Qwen1.5-32B, Synthetic tasks drop from 52.67 to 46.25 (~12% relative) and Single-Doc QA drops from 40.68 to 37.16 (~8.6% relative). The paper's characterization of these as "subtle impact on accuracy" is inconsistent with the actual numbers. Authors should report relative drops explicitly rather than relying on absolute score differences in already-low-scoring tasks.

- **Hyperparameters (P, N, F, η) appear model-specific with no held-out validation**: Section 4.1 sets different η and F per model, apparently tuned on LongBench. There is no cross-validation protocol described, raising the concern that these choices are optimized for the evaluation benchmark. The robustness of these choices across diverse tasks and domains is unclear.

- **Re-implemented PyramidInfer as primary comparison baseline**: The authors acknowledge that the official PyramidInfer implementation fails to accelerate prefilling and re-implement it with flash attention (Section 4.3). This re-implementation may not faithfully preserve the accuracy characteristics of the original method. Including results from both PyramidInfer* and the re-implemented variant is appropriate transparency, but readers cannot determine whether the re-implementation is accurate relative to the original paper's reported quality metrics.

### Trivial

- **The "sometimes surpasses baseline" observation in Figure 7 is unexplained**: FTP achieves higher relative scores than the baseline on Single-Document QA and Synthetic Task in Figure 7. This could reflect noise in LongBench's reference-based metrics or a regularization-like effect; either way, it deserves a brief note rather than being passed off as evidence of strength.

---

## Nice-to-Haves

- A controlled experiment at 32k–64k context lengths (e.g., RULER or SCROLLS) would establish whether the speedup and accuracy hold in the "long-context" regime the paper uses as motivation.
- An analysis of *why* Code Completion on Llama3-8B fails so severely — e.g., visualizing which tokens are pruned in code inputs, or checking whether code tokens are attended to more uniformly — would help identify the failure mode and potentially fix it.
- An ablation comparing the "last N queries" scoring heuristic (borrowed from SnapKV) against full-sequence attention scoring or hidden-state norm alternatives would validate whether this design choice is optimal for the FFN pruning setting (it was designed for KV-cache compression, a different problem).
- Combination with KV-cache compression methods for the decoding stage: as Figure 2 shows, for code completion (RepoBench-P), decoding accounts for 76% of total inference time, so FTP's contribution to end-to-end latency for that task is minimal.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic's O(L²) attention recomputation overhead claim**: The critic argues the recomputation is O(L²) in memory at long sequences. This is factually incorrect. The paper explicitly uses only the last N=50 queries (Section 3.2.1, Algorithm 1 line 3: `M.sum(0)[P:-N]` with M of size (H, N, L)), so the recomputation cost is O(N×L) = O(L) with N fixed at 50. This scales linearly, not quadratically. The critic's concern about 128k sequences is absorbed into the "Minor" concern about context-length evaluation, but the specific O(L²) technical claim is wrong.

- **Reproducibility / hyperparameter tuning details**: The critic raises concerns about undisclosed hyperparameter choices and cross-validation. This is a legitimate concern (promoted to Minor above), but the aspects framed as a reproducibility issue (e.g., "no validation protocol to confirm not overfit") shade into nitpick territory; the hyperparameters (η=0.90, N=50, P=100, F=10) are reported and are coarse-grained enough to be plausible without LongBench-specific tuning.

- **Criticism of sort being O(L log L) per layer in Python**: The per-layer sort (Algorithm 1 line 4) is O(L log L), but the paper measures its cost at 7–15ms total (across all layers), which is empirically confirmed to be small. The theoretical concern about CPU-GPU synchronization does not appear to manifest in practice.

- **Strength Finder's "comparable with flash attention" as a standalone strength**: Being training-free and flash-attention compatible is an implementation property, not a scientific contribution. Removed as a generic strength.

---

## Novel Insights

The core novel observation — that prior token pruning work has ignored the FFN as the largest prefilling bottleneck (>60% walltime) while focusing on the attention module, and that the residual connection can be exploited to "absorb" pruned tokens without a separate data structure or sparse kernel — is the paper's genuine insight. The combination of FFN-specific pruning + residual passthrough is architecturally elegant and distinct from attention-centric approaches. The dynamic layer-adaptive threshold (Eq. 3) based on the entropy of per-layer attention concentration is a sensible extension that the paper motivates and validates. However, the failure to explain the code-completion collapse on Llama3-8B suggests the method's behavior on tasks requiring dense token attention is not yet understood.

---

## Evaluation on Key Axes

- **Originality**: Moderate-high. The FFN-targeting angle and residual passthrough trick are genuinely distinct from prior work.
- **Importance of research question**: High. TTFT is a real bottleneck for long-context LLMs.
- **Claims well-supported**: Partially. The Qwen2-7B results are fairly clean; the Llama3-8B code completion failure seriously undercuts the main "negligible drop" claim.
- **Soundness of experiments**: Moderate. LongBench is an appropriate benchmark, but the context lengths are short relative to the motivation, LazyLLM is missing, and one major failure case is unaddressed.
- **Clarity of writing**: Acceptable. Algorithm and method descriptions are clear; the discussion of results glosses over failure cases.
- **Value to research community**: Moderate. The residual-connection trick and FFN-targeting insight are useful; the current evaluation leaves open questions about reliability.

---

## Score and Decision

**Calibration anchors used:**

| Path | Avg Human Score | Comparison to FTP |
|---|---|---|
| `/human_reviews/uNrFpDPMyo.md` | 8.0 | FastGen (adaptive KV cache, oral): much stronger — richer experiments, custom CUDA kernel, broader model coverage, no major failure cases. FTP is clearly below this. |
| `/human_reviews/ALzTQUgW8a.md` | 7.2 | MagicPIG (LSH-based KV sampling, spotlight): better theoretical grounding, larger efficiency gains, no significant failure task. FTP is below this. |
| `/human_reviews/9iN8p1Xwtg.md` | 5.25 | GemFilter (prefilling via early-layer token filtering, rejected): very similar profile — training-free, LongBench evaluation, moderate speedup, performance drop on some model/task combinations, missing baselines. FTP is slightly stronger due to the FFN-specific insight and cleaner ablation. |
| `/human_reviews/SYv9b4juom.md` | 5.25 | OrthoRank (token selection for LLM efficiency, rejected): similar pattern of modest contribution, incomplete baselines, and limited scale experiments. Comparable to FTP. |
| `/human_reviews/2DD4AXOAZ8.md` | 2.0 | MixAttention (KV-cache reduction, rejected): truly weak — no novelty, no original contribution. FTP is clearly above this. |
| `/human_reviews/7DY2DFDT0T.md` | 2.5 | EfficientSkip (sparse LLM, withdrawn): limited validation, unclear advantages. FTP is clearly above this. |

**Positioning**: FTP falls squarely in the 5.0–5.5 range. It is comparable to GemFilter and OrthoRank (both scored 5.25, both rejected). FTP has a slightly stronger mechanistic insight (FFN-specific + residual trick) and a more thorough experimental setup than GemFilter, but has a more significant unexplained failure case (Llama3-8B code completion) and a missing critical baseline (LazyLLM). The aggregate evidence places FTP around **5.0** — a borderline-reject paper with a clean core idea but insufficient experimental rigor to support its central "negligible drop" claim.

**Score: 5.0**  
**Decision: Reject**

The paper proposes a clean and novel idea with genuine efficiency insights, but the 35% relative accuracy collapse on Llama3-8B Code Completion — presented as "negligible" — combined with the missing LazyLLM comparison and the short context lengths tested relative to the stated motivation are sufficient grounds for rejection at the current standard. The residual-connection trick and FFN-targeting insight have real value; the paper is revisable but not in its current form.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>