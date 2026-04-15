Now I have everything I need. Let me produce the final consolidated review.

---

## Summary

Cut Cross-Entropy (CCE) eliminates the O(N×|V|) logit materialization bottleneck in LLM cross-entropy training by decomposing the loss into an indexed true-class matrix multiplication and a linear-log-sum-exp operation, both implemented via SRAM-resident blockwise CUDA/Triton kernels. For Gemma 2 (2B), CCE reduces the loss-computation memory footprint from 24 GB to 1 MB and the full classifier-head training footprint from 28 GB to ~1 GB. The paper additionally proposes gradient filtering (exploiting softmax sparsity in bf16) for a 3.4× backward speedup, and vocabulary sorting for an additional 15% gain.

---

## Strengths

- **Quantifies a newly dominant, underappreciated bottleneck**: The paper empirically establishes that the cross-entropy layer consumes 40–89% of training memory across a broad set of frontier LLMs (Fig. 1, verified in Sec. 3.2). This is not obvious from prior work focused on attention and activation memory, and the concrete per-model breakdown (Phi 3.5: 40%, Llama 3 8B: 65%, Gemma 2 2B: 89%) is genuinely novel documentation.

- **Clean, immediately usable decomposition**: The reformulation of cross-entropy into indexed matrix multiplication plus linear-log-sum-exp (Eq. 4) is elegant and non-trivial to implement efficiently. The three algorithms (Alg. 1–3) are clearly specified with block sizes and access patterns, and the SRAM-resident strategy avoids the memory/latency tradeoff that plagued chunked methods like Torch Tune and Liger Kernels.

- **Achieves memory savings without trading away speed**: Unlike Liger Kernels (2× slower in Table 1) and Torch Tune chunked CE, CCE matches or beats torch.compile on wall-clock time (46ms vs 49ms for loss; 145ms vs 143ms for loss+gradient) while using 24× less memory. This is a qualitatively different regime from prior methods, not merely a better point on the speed/memory tradeoff curve.

- **Transparent and honest about pretraining failure modes**: The paper candidly reports that the default CCE variant degrades validation perplexity during pretraining due to two mechanisms (gradient filtering starving low-support tokens; global-memory bf16 summation precision loss), proposes CCE-Kahan-FullC to address both, and validates it independently on four models. This kind of upfront disclosure strengthens the paper's credibility.

- **Table 1 ablations are unusually clean and informative**: Isolating gradient filtering (3.4× speedup) and vocabulary sorting (15% speedup) in a single table with a clear lower bound row gives practitioners exact guidance on what to expect from each component.

---

## Weaknesses

### Fatal
*None.*

### Major

- **Gradient filtering safety is overstated in Section 4.3, internally inconsistent with Section 5.3.** Sec. 4.3 states: *"In practice, this leads to a 3.5x speedup without loss of precision in any gradient computation."* This is directly contradicted by Sec. 5.3: *"gradient filtering when applied to ∇C causes no gradient to be propagated to tokens that have little to no support in the training set. This does not cause issues when fine-tuning but does when pretraining."* The claim "without loss of precision in any gradient computation" is false for the pretraining setting—the paper's own experiments demonstrate this. The method is safe for fine-tuning and safe with CCE-Kahan-FullC for pretraining, but the unqualified language in Sec. 4.3 misrepresents the default variant's behavior. This discrepancy between the method-section claims and the training-stability section creates real confusion about what the core contribution guarantees.

### Minor

- **Abstract and introduction overstate the breadth of the "no sacrifice in speed or convergence" claim.** The headline claim ("without sacrificing training speed or convergence," abstract; "no sacrifice in speed or convergence," Fig. 1 caption) is supported by (a) a single Gemma-2B-shaped microbenchmark on one A100 variant and (b) training-loss curves on Alpaca fine-tuning for 4 models plus validation perplexity on 5% OpenWebText for the *modified* CCE-Kahan-FullC variant. The core CCE method is never shown to match torch.compile for pretraining; the pretraining-safe variant CCE-Kahan-FullC is 2× slower than CCE (Table 1, row 9 vs. 1). The paper implicitly attributes full pretraining safety to CCE but the supported claim is narrower: CCE-Kahan-FullC for pretraining, default CCE for fine-tuning.

- **Fig. 1 batch-size projections are model-derived estimates, not direct measurements.** The figure and its caption present max-attainable batch sizes for 11 frontier models on a 16-GPU setup, with exact values cited in Table A4. However, the main body never validates these projections against actual full-training runs on the listed models; only the cross-entropy kernel is directly benchmarked. The memory-model projections depend on assumed sharding strategy, checkpointing granularity, and optimizer state, none of which vary in an ablation. The caption should more explicitly flag which numbers are measured vs. modeled.

- **Runtime benchmarking is limited to a single hardware platform.** Table 1 covers one GPU generation (A100-SXM4, 80 GB) at one workload size. The paper's Appendix C extends to more (D, |V|) configurations but remains A100-only. Whether block sizes, SRAM utilization, and gradient filtering efficiency generalize to H100, A10G, or other architectures is not established. The paper acknowledges that CUDA may further improve performance (Sec. 6), but the cross-GPU portability of the Triton implementation is untested.

### Trivial

- **Vocabulary sorting analysis is minimal.** The heuristic of sorting by average logit is introduced without comparison to alternatives (frequency-based ordering, random, etc.), without reporting how quickly statistics go stale during training, or how often resorting is needed. Given that sorting accounts for 15% of the speedup, a brief ablation of the sorting criterion would be informative.

- **Section 5.2 (gradient filtering) is thin relative to the importance of the mechanism.** Fig. 3 provides only an aggregate sorted-probability curve. Actual block-skip rates, per-model or per-training-stage sparsity variation, and sensitivity of sparsity to model size are unanalyzed. This is adequate to support the empirical claim but falls short of the mechanistic justification framing in Sec. 4.3.

---

## Nice-to-Haves

- **Longer-horizon pretraining validation or downstream task metrics**: 5% of OpenWebText is a reasonable sanity check, but showing matching perplexity or standard benchmark performance (e.g., MMLU, HellaSwag) after a 1B+ token pretraining run would significantly raise confidence that CCE-Kahan-FullC is safe at production scales.
- **Gradient error characterization**: A direct quantitative comparison of filtered vs. full gradients (e.g., cosine similarity or relative L2 norm) under the ε-threshold, and how this changes with model, training stage, and vocabulary size, would make the filtering mechanism more credible beyond the empirical convergence curves.
- **Multi-GPU end-to-end throughput**: Table 1 benchmarks the loss kernel in isolation; a full training-step wall-clock comparison at maximum CCE batch size vs. maximum baseline batch size on the 16-GPU setup that motivates Fig. 1 would close the gap between microbenchmark and headline claims.
- **Vocabulary sorting sensitivity ablation**: Compare ordering strategies (random, frequency-based, logit-based) and report staleness effects to guide practitioners.

---

## Removed Points

*These points are flagged for removal; treat with caution.*

- **Liger Kernels comparison is unfair (Spark, Human Finder):** The paper explicitly explains the design tradeoff: Liger fuses forward+backward, making it faster for users who need no loss transformation, while CCE keeps them separate to allow user-defined transforms. This is a deliberate and clearly explained architectural choice, not an unfair benchmark setup. The paper never presents CCE as simply better than Liger; it explains the different operating points. Removed per the rule on asymmetric comparisons that favor the baseline.

- **Demands for formal gradient error bounds (Neutral, Spark):** Theoretical gradient error bounds with formal guarantees are not a standard expectation for ICLR systems papers; empirical convergence matching across 4 models with 5 seeds is the community norm for this type of contribution. Moved to Nice-to-Have.

- **Demands for full-scale pretraining over billions of tokens (multiple reviewers):** Providing a complete pretraining run to billions of tokens is not a standard requirement for a systems contribution paper. The 5% OpenWebText validation with the modified variant is reasonable evidence. The concern about long-horizon safety is legitimate but moves to Nice-to-Have rather than a blocking weakness.

- **Downstream task evaluation (Human Finder, Spark):** The paper is a systems contribution claiming memory savings with training-stability preservation. Matching training/validation loss curves for 4 models is the appropriate evaluation for such a claim. Demanding MMLU/HellaSwag evaluation is scope creep for a kernel-level systems paper; moved to Nice-to-Have.

- **Triton/CUDA gap as a weakness (Human Finder):** The paper transparently states in Sec. 6: *"We implemented CCE using Triton... Triton has some limitations in control flow... We expect that implementing CCE in CUDA may bring further performance gains."* This is honest disclosure of a limitation and an opportunity, not a methodological flaw. The existing Triton implementation already matches or exceeds torch.compile. Removed as the paper addresses this directly.

- **Liger Kernels convergence comparison missing (Spark):** Liger Kernels are a compute/memory implementation, not a modified training algorithm. Using torch.compile as the convergence baseline (which computes the mathematically identical loss) is the correct comparison. There is no reason to expect Liger to change convergence relative to torch.compile; demanding this experiment reflects a misunderstanding of what Liger does.

---

## Novel Insights

The most genuinely original observation in the paper—underemphasized in the reviews—is that gradient filtering and vocabulary sorting together achieve near-torch.compile speed at O(N+|V|) memory, which is qualitatively different from the trade-off curve on which all prior chunked methods (Torch Tune, Liger) operate. Prior methods treat the memory/speed tradeoff as a Pareto frontier; CCE effectively sidesteps that frontier by avoiding global materialization entirely. The empirical demonstration that 99.98%+ of softmax entries are below bf16 representability at steady-state in frontier models (Fig. 3) is a concrete and underappreciated property of modern LLM distributions that makes this implementation practically viable. The honest identification of when this breaks (pretraining with low-support tokens) and the targeted CCE-Kahan-FullC fix is a meaningful methodological contribution beyond the core kernel design.

---

## Suggestions

1. **Fix the Section 4.3 overclaim immediately**: Remove or qualify "without loss of precision in any gradient computation." Replace with language distinguishing fine-tuning (default CCE is safe) from pretraining (requires CCE-Kahan-FullC).
2. **Clarify Fig. 1 status**: Add a one-sentence note in the caption stating that batch-size projections are computed from a memory model under stated assumptions, not from direct end-to-end training measurements.
3. **Benchmark CCE-Kahan-FullC prominently in Table 1**: Currently rows 8–10 appear as ablation variants, but for pretraining users CCE-Kahan-FullC is the recommended method. Its 2× compute overhead relative to default CCE should be foregrounded in the main paper, not just visible in the ablation rows.
4. **Report multi-GPU end-to-end throughput for at least one model**: The Mistral NeMo 2-hour speedup anecdote in Sec. 5.3 is promising; formalizing this as a benchmark (tokens/sec at max batch size, with and without CCE) would be a strong practical datapoint.
5. **Add a one-paragraph characterization of softmax sparsity across training stages**: Fig. 3 shows a snapshot; whether sparsity is lower during early pretraining (where gradients matter most) is directly relevant to the safety of default CCE for pretraining.

---

## Evaluation

- **Novelty**: High for a systems paper. The decomposition and SRAM-resident blockwise computation are non-trivial, and the gradient filtering insight is practically valuable.
- **Technical soundness**: Good. The algorithms are correct, the implementation is carefully designed, and the ablations are rigorous. The main soundness issue is the overclaimed Sec. 4.3 language.
- **Empirical support**: Solid for a systems paper. Table 1 is directly measured; Fig. 1 projections are model-based estimates (a distinction that should be clearer). Fine-tuning and reduced-scale pretraining validations across 4 models with 5 seeds are convincing for the paper's scope.
- **Significance**: High. Eliminating the largest single memory bottleneck in modern LLM training, with open-source release, will likely see wide adoption.
- **Clarity**: Good overall; weakened by the internal inconsistency between Sec. 4.3 and Sec. 5.3 on gradient filtering safety.

---

## Score and Decision

**Calibration against past reviews in this run:**

- **mMPaQzgzAN (JumpReLU SAEs, 6.5 — Accept):** A solid contribution with principled KDE-STE theory, convincing Pareto frontier results, but limited model scope and a framing mismatch between architecture vs. training method novelty.
- **D0Cdljktp2 (Memformers, 4.0 — Reject):** Core proposition does not prove what it claims; experiments at toy scale; headline result uses training data.

CCE is clearly and substantially above JumpReLU SAEs (6.5): the problem is more practically important, the contribution more direct and immediately usable, the empirical validation broader (4 models × 5 seeds × two regimes), and the practical impact (enabling 1.5–10× larger batch sizes for frontier LLMs) considerably higher. The weaknesses are real (Sec. 4.3 inconsistency, narrow hardware benchmark, limited pretraining scale) but none approach fatal. Placing this above 6.5 by a meaningful margin lands at **7.5**.

**Score: 7.5 — Accept**

MY FINAL SCORE: <pineapple>7.5</pineapple>
MY FINAL DECISION: <orange>Accept</orange>