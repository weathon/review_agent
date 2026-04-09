##Summary

Thoughtbubbles introduces a transformer variant that enables unsupervised, adaptive parallel computation in latent space by learning to fork or delete residual streams mid-network. Tokens requiring more computation form "bubbles" of cloned residuals, with the forking behavior learned entirely from standard language modeling loss during pretraining. Experiments across 150M–772M parameter scales on OpenWebText and peS2o show consistent perplexity and zero-shot evaluation improvements over both parameter-matched and computation-matched baselines.

## Strengths

- **Genuinely unsupervised adaptive compute during pretraining**: Unlike CoT or pause-token methods that require fine-tuning, special prompts, or manually inserted tokens, Thoughtbubbles learns dynamic allocation of latent parallelism solely from the cross-entropy LM objective. This is a specific and non-trivial achievement—most prior work requires either architectural rigidity (fixed pause positions) or auxiliary training signals. The paper demonstrates this across multiple scales.

- **Careful baseline design isolating the value of adaptivity**: The inclusion of both parameter-matched (standard GPT-2) and computation-matched (Copy-3/Copy-5) baselines is methodologically sound. Copy-N baselines are the right control: they provide the same extra residual stream capacity but without dynamic allocation, directly testing whether *adaptivity* matters beyond raw compute. The consistent gap between Thoughtbubbles and Copy-N across scales (Table 1) is the paper's strongest empirical result.

- **Interpretable computation allocation without explicit supervision**: Figure 5 shows the learned fork allocation correlates with posterior entropy (measured both by the forking model and an independent baseline LM), and Figure 7 shows sensible allocation on the synthetic CLUTRR task. These analyses provide evidence that the model discovers meaningful computational heuristics from LM loss alone.

- **Thoughtful position encoding for variable fork counts**: The partial rotation RoPE variant (Appendix D, Eq. 13) that scales rotation proportionally to fork count is a specific and necessary technical contribution for making the architecture work, addressing a real challenge that would otherwise break positional semantics.

## Weaknesses

### Major:

- **Missing ablation on whether *learned* fork decisions matter**: The paper attributes its gains to adaptive forking, but never tests whether the *learned* allocation strategy is responsible vs. the mere *capacity* for extra residual streams. A critical missing ablation is: what happens with *random* forking at the same average rate? Or fixed forking at every token? Similarly, the score-attenuation mechanism (Eqs. 8–10) creates a strong inductive bias coupling forking scores to residual updates—without ablating it, we cannot determine whether the forking decisions are learning meaningful structure or whether the architectural bias alone drives the gains. These ablations are essential for validating the core claim that *adaptive, learned allocation* is the source of improvement.

- **KV cache memory overhead is unaddressed**: The paper proposes an inference-time architecture but does not discuss the memory implications of maintaining κ× longer sequences during autoregressive generation. With κ=4L, the KV cache is up to 4× larger than a standard model. For long-context inference, KV cache memory is often a harder constraint than FLOPs. The paper acknowledges wall-clock inefficiency (Section 8) but the memory constraint is a distinct and arguably more fundamental limitation for deployment that deserves explicit discussion and quantification.

- **Claim inconsistency about uncertainty allocation between Introduction and Analysis**: The Introduction states the method "allocates more computation at regions of higher uncertainty (i.e., posterior entropy)" (Section 1, Contribution 3). However, Section 5 reveals a "concave parabolic relationship" where computation *decreases* at the highest uncertainty tokens. The authors' explanation (highest uncertainty at clause boundaries where extra computation is unhelpful) is reasonable, but the Introduction's claim is misleading as written—it implies a monotonic relationship that the data do not support. The Introduction should be corrected to say the model allocates more computation at *moderate-to-high* uncertainty regions.

### Minor:

- **Gradient flow through the non-differentiable top-k is underspecified**: The forking judgment uses hard top-k selection (Section 2.3), which is non-differentiable. The paper does not explicitly state how gradients propagate through this operation. While the standard approach (straight-through: gradients flow through selected elements, dropped elements receive zero gradients) is likely used—and the authors acknowledge the resulting gradient bottleneck in Limitations—the method section should state this explicitly. The current description leaves the reader to infer the gradient mechanism.

- **Limited evaluation scope for the stated motivation**: The paper motivates the work as enabling "complex, multi-step problems" (Section 1), but evaluates only on perplexity and zero-shot NLU tasks (LAMBADA, HellaSwag, BLiMP, PIQA). The authors acknowledge in Limitations that scale constraints prevent evaluation on reasoning benchmarks like GSM8k. While understandable, this gap between motivation and evaluation leaves the central promise unverified—the tasks where adaptive computation should matter most (multi-step reasoning) are exactly those not tested.

- **Dual suppression mechanism rationale unclear**: The architecture applies both structural deletion (top-k removal) and attention masking (score attenuation, Eqs. 8–10) to low-scoring tokens. The paper does not analyze why both are needed or how they interact. If score attenuation can effectively zero out a token's contribution, why also delete it? Presumably top-k serves to reduce sequence length for computational efficiency, but this trade-off is never articulated.

### Trivial:

- The overforking ablation (Appendix B) is minimal (only 25K steps, ~0.8BT tokens) and does not conclusively establish that additional forking layers fail to help; the result (28.02 vs. 29.84 perplexity) is reported as "slightly worse" but the training budget is insufficient to draw firm conclusions.

## Nice-to-Haves

- Comparison against established adaptive computation methods (e.g., Universal Transformers, Mixture-of-Depths) at the same scale, to contextualize the gains against a broader landscape of adaptive architectures.
- Analysis of semantic divergence between parent and forked residual streams (e.g., cosine similarity across layers) to verify that forks compute genuinely different representations rather than redundant copies.
- Evaluation of whether pretrained forking behaviors are preserved after fine-tuning on downstream tasks.

## Removed Points

*These points are flagged to be removed, treat them with caution.*

- **"Unfair capacity comparisons" / "learned allocation vs. static allocation is an unfair comparison"**: The comparison between Thoughtbubbles (learned allocation) and Copy-N (static allocation) is not unfair—it is the core experimental design. Copy-N is the appropriate baseline to test whether *adaptivity* provides value beyond *more compute*. The asymmetry (learning vs. not learning) is the independent variable being tested.

- **"Parallel Thinking is misleading regarding latency"**: The computation IS architecturally parallel (multiple residual streams processed simultaneously in the same forward pass). Latency concerns are real but separate; the term is not misleading—it describes the structural property correctly.

- **"Equation 4 contradiction with 'ignore rightmost token'"**: On close reading, this is not a contradiction. Eq. 4 forces the keep score to 1 for the *top-k selection* (structural: the original token is never deleted), but the *cumulative scores* used for attention attenuation (Eqs. 8–10) are not forced to 1, allowing the original token to be functionally "ignored" via attention masking even while structurally present. The text's statement is correct.

- **"Missing FLOP-matched Dense Model baseline"**: The paper already shows a 319M Thoughtbubbles model beating a 772M baseline (Figure 3), which partially addresses whether adaptivity beats scale. While a wider/deeper model at matched FLOPs would be informative, the existing comparison provides relevant evidence. Demanding additional FLOP-matched dense baselines is a generic expansion of experimental scope.

- **"CoT can reduce performance on certain tasks"**: This is scope creep. The paper does not claim universal improvement across all task types and actually reports degraded BLiMP performance vs. computation-matched baselines. The finding that adaptive compute may not help for syntax is already present in the results.

- **"Extend scaling to 1B+"**: The paper already scales to 772M and shows consistent trends. Demanding larger scale is a generic weakness that does not engage with the specific contribution.

## Novel Insights

The concave relationship between uncertainty and computation allocation (Figure 5) is a genuinely interesting finding with potential broader implications. The model learns that moderate-uncertainty tokens (e.g., choosing between a few plausible continuations) benefit most from extra compute, while the highest-uncertainty tokens (e.g., clause boundaries, coreference edges) are inherently unresolvable by additional computation. This suggests a natural "computability frontier" where adaptive compute provides diminishing returns—a principle that could inform the design of future adaptive inference systems beyond this specific architecture.

## Suggestions

- Run a random-forking ablation (fork at the same average rate but with random allocation) and a fixed-forking ablation (fork every token equally) to establish that *learned* allocation is the source of improvement, not just extra capacity.
- Add explicit discussion of KV cache memory scaling with κ and quantify the memory-per-token cost relative to the baseline, as this is a deployment-critical metric.
- Correct the Introduction's claim about uncertainty allocation to reflect the concave relationship shown in the Analysis section.
- Explicitly state the gradient propagation strategy through top-k in Section 2.3 (e.g., "gradients propagate through selected elements via straight-through estimation; dropped elements receive zero gradients").

---

**Quality Assessment:**

- **Novelty**: High. The forking mechanism with cumulative scores learned unsupervised during pretraining is architecturally novel and distinct from prior pause-token or adaptive-depth approaches.
- **Technical soundness**: Moderate. The architecture is well-designed, but the lack of key ablations (random forking, score attenuation) leaves the causal mechanism behind the gains ambiguous. The claim inconsistency about uncertainty allocation is a clarity issue that could mislead readers.
- **Empirical support**: Moderate-to-good. Consistent improvements across scales and datasets against meaningful baselines, but missing ablations weaken the attribution of gains to the claimed mechanism. The evaluation tasks do not include the reasoning benchmarks most relevant to the stated motivation.
- **Significance**: Moderate-to-high. If the ablations confirm that learned allocation (not just capacity) drives the gains, this would be a significant contribution to adaptive computation. The unsupervised pretraining-time learning of adaptive compute is the key differentiator from prior work.
- **Clarity**: Moderate-to-good. The method is generally well-described, but the gradient flow through top-k is underspecified and the uncertainty allocation claim is inconsistent between sections.