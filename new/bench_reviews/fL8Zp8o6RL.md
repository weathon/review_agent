## Summary
This paper proposes FTP, a training-free method to reduce time-to-first-token for long-context LLMs by pruning tokens only before the FFN sublayer during prefilling, while keeping attention intact and relying on the residual path for skipped tokens. The core empirical claim is reasonably supported: on LongBench and several Qwen/Llama models, FTP often yields roughly 1.2–1.4× TTFT speedups with modest average degradation, though the benefits and quality tradeoffs are clearly task-dependent rather than uniformly negligible.

## Strengths
- **Targets a real and underexplored bottleneck.** The paper’s profiling is concrete: Figure 3 shows FFN taking over 60% of decoder-layer walltime during prefilling on both Llama3-8B-Instruct and Qwen2-7B-Instruct, which justifies focusing on FFN rather than only KV-cache or attention sparsification.
- **Simple, training-free mechanism with a clean systems intuition.** FTP prunes tokens only for FFN, not attention, and uses the residual connection so skipped tokens keep their attention-updated representations. This is a neat intervention that is easy to understand and deploy.
- **Dynamic per-layer pruning is better motivated than a fixed-ratio heuristic.** The method uses cumulative attention mass to determine retained tokens per layer/sample, and Figures 5–6 provide evidence that attention mass is concentrated and varies by layer.
- **Experimental coverage across model sizes is meaningful.** Results are reported on Llama3-8B, Qwen2-7B, Qwen1.5-32B, and Qwen2-72B, which supports that the idea is not confined to a single model family or only small-scale settings.
- **The random-pruning ablation is informative.** Table 3 shows that matching the same pruning budget with random selection destroys accuracy, while FTP largely preserves it; this supports that the token selection rule is doing real work rather than speedup coming trivially from any FFN skipping.
- **The paper is generally clear about its mechanism and implementation.** The algorithm and the role of \(P, N, \eta, \mathcal{F}\) are laid out clearly enough to follow the method.

## Weaknesses
### Fatal
None.

### Major:
- **The paper overstates the breadth of its efficiency claim relative to what is measured.** The experimental efficiency metric is explicitly only TTFT: Section 4 says “The efficiency metric is the TTFT speedup.” That supports a claim about **prefilling acceleration**, but not a broad claim about overall “long-context LLM inference” efficiency. The paper’s own Figure 2 shows this limitation: on RepoBench-P, prefilling is only 23.71% of total inference time, so a 1.2–1.4× TTFT gain may translate to a much smaller end-to-end gain. This does not invalidate the paper, but the framing should be narrowed and total latency should be reported if broader practical claims are desired.
- **The “negligible decrease in performance” claim is too strong as written.** Most results are indeed fairly mild drops, but there is a clear counterexample: in Table 1, **Llama3-8B code completion drops from 55.17 to 35.91**, which is a very large degradation. Table 2 also shows several nontrivial drops on larger models (e.g., Qwen1.5-32B synthetic: 52.67 to 46.25). The method therefore has meaningful failure modes, and the current abstract/conclusion language overgeneralizes beyond the evidence.
- **The main superiority claim over PyramidInfer is weakened by the evaluation setup.** The official implementation (PyramidInfer*) is slower partly because it uses PyTorch attention rather than flash attention, which the paper itself acknowledges. The reimplemented PyramidInfer is more relevant, but the description is too thin to fully establish implementation parity beyond “re-calculate the necessary attention weights (i.e., 20% attention weights following the official setting).” Since the main comparative conclusion depends on this reimplementation, more detail is needed before treating the SOTA superiority claim as fully established.
- **The paper does not sufficiently characterize the actual operating behavior of FTP.** Although pruning is driven by dynamic cumulative attention thresholds, the main text never reports the realized retention/pruning rates by layer, model, or task. Since the headline contribution is efficient FFN pruning, readers should be able to see how aggressively FTP actually prunes in practice and how this correlates with speedup and failure cases.

### Minor
- **The quality metric is a heterogeneous aggregate that can obscure where quality is lost.** Section 4 averages F1, Rouge-L, accuracy, and edit similarity into a unified “score.” This is convenient for summarization, but it weakens blanket claims like “negligible decrease” because improvements/drops across disparate metrics are not equally interpretable. The task-level tables remain useful, but the headline summaries are somewhat smoothed by this aggregation.
- **Key hyperparameters are set empirically with limited justification in the main paper.** The method depends on \(P=100\), \(N=50\), \(\mathcal{F}=10\), and model-specific \(\eta\), but the main text provides limited sensitivity analysis for these choices. Since \(\mathcal{F}\) and \(\eta\) directly control the speed/accuracy tradeoff, more systematic reporting would improve confidence in robustness.
- **The support for attention-based importance as the right proxy for FFN importance is suggestive rather than definitive.** Figure 5 shows attention sparsity, and Table 3 shows random pruning is much worse, but that does not fully establish that attention is the best or most principled signal for deciding which tokens can safely skip FFN updates. The evidence is adequate for a practical paper, but mechanistically still somewhat indirect.
- **Timing methodology is underspecified for a systems-facing claim.** The paper reports TTFT improvements but does not clearly spell out measurement details such as warmup, synchronization, repeated runs, and batching policy. For moderate wall-clock gains around 1.2×, that additional rigor would help.

### Trivial
- **There is a likely notation typo in the FLOPs discussion.** After Equation (1), the text says the FFN FLOPs sum to \(6LCT\), but given the defined dimensions this appears intended to be \(6LCI\). This is minor and does not affect the main idea.

## Nice-to-Haves
- Report **end-to-end latency** (prefill + decoding) in addition to TTFT, especially for tasks where generation length is substantial.
- Add a concise analysis of the **code completion failure mode**, since it is the clearest counterexample to the paper’s generality claim.
- Include **realized per-layer pruning-rate statistics** or a heatmap across layers/tasks.
- Provide a more systematic sensitivity study for \( \mathcal{F}, \eta, P, N \).
- Clarify the **runtime benchmarking protocol** and, if feasible, include variability over repeated runs.

## Removed Points
These points are flagged to be removed, treat them with caution.

- **“Missing comparison with LazyLLM / other omitted related baselines.”** Removed under the instruction not to mention missing related works, since external completeness cannot be verified here. It is fair to say the comparative superiority claim over the included baseline is not fully established; it is not appropriate here to fault the paper for not comparing to specific uncited methods.
- **Concerns about release status / availability / verifiability of models or methods.** Removed per hard rule.
- **Formatting/style complaints.** Removed.
- **Strong reproducibility nitpicks about all implementation details or hyperparameters.** We keep only the substantive point that timing methodology is underspecified because the paper makes runtime claims; generic demands for exhaustive implementation detail are removed.
- **Claims that the paper is invalid because FlashAttention does not expose attention weights.** The paper explicitly addresses this in Section 4.1: it recalculates the necessary attention weights and gives overhead evidence in Table 3, so any objection that ignores this would be a misunderstanding.
- **Unfair-comparison complaints based solely on the official PyramidInfer implementation using a slower backend.** Removed in their stronger form, because the asymmetry there actually favors the baseline narrative less than the authors; the valid remaining issue is only that the authors’ reimplementation is underdescribed.

## Novel Insights
The strongest synthesis is that this paper is **better than its rhetoric but weaker than its headline framing**. The core technical contribution is real: FFN-only token pruning during prefilling is a clean and practically relevant design point that prior attention/KV-focused work can miss, and the empirical evidence is good enough to support TTFT reduction as a narrow claim. However, the paper’s own results suggest FTP is not uniformly safe across task types—especially for code completion—and that its practical value depends strongly on whether prefilling dominates runtime. In other words, the work is most compelling as a **specialized prefilling optimization for long-input, short-output regimes**, not yet as a general long-context inference accelerator.

## Suggestions
- Narrow the central claim throughout the paper to **prefilling / TTFT acceleration**, unless end-to-end latency evidence is added.
- Revise the abstract and conclusion to acknowledge that quality degradation is **usually modest but task-dependent**, with explicit mention of the code-completion exception.
- Add a table or figure of **realized retained-token ratios by layer/task/model**.
- Expand Section 4.3 with enough detail to assess whether the **PyramidInfer reimplementation** is faithful and runtime-comparable.
- Add a focused analysis of **why code completion degrades sharply** under FTP; this would substantially strengthen the paper.
- Improve the benchmarking section with clear timing protocol details and, ideally, repeated-run statistics.

## Score and Decision
**Originality:** moderate. The ingredients (attention-based token importance, pruning for efficiency) are not wholly new, but pruning specifically between attention and FFN using the residual path is a meaningful and well-motivated twist.  
**Importance of the research question:** high. TTFT for long-context models is an important practical bottleneck.  
**Whether the claims are well supported:** mixed. The narrow TTFT claim is supported; the broader “long-context inference acceleration” framing is overstated.  
**Soundness of experiments:** decent but incomplete. Multi-model evaluation and ablation are strengths, but end-to-end latency, detailed runtime methodology, and stronger comparative evidence are missing.  
**Clarity of writing:** generally good.  
**Value to the research community:** solid. Even with the caveats, the paper offers a simple, training-free idea that others can build on.

**Calibration against human-reviewed anchors:**
- Compared to **FlexPrefill** (Accept Oral, scores 8/8/8/8), this paper is clearly weaker: FlexPrefill appears to have stronger comparative evidence and more convincing breadth for long-context prefilling efficiency, while this paper has a narrower validation and a notable task failure.
- Compared to **D₂O** (Accept Poster, scores 6/6/6/5/6), this paper is in a similar band: both offer a real training-free efficiency contribution with decent experiments, but neither is flawless in comparative support and analysis depth.
- Compared to **LazyLLM** (Reject, scores 6/5/6/3) and **Recycled Attention** (Reject, scores around 5–6), this paper looks somewhat stronger because its core empirical story is cleaner and the mechanism is simpler, but it shares some of the same weaknesses around claim scope and evaluation completeness.

Overall, this paper lands **above borderline reject but below clear accept**: a real contribution with convincing narrow evidence, held back by overclaiming and incomplete practical evaluation. I would place it at **6.0**.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>