## Summary

The paper proposes Augmented Intermediate Representations (AIR), a defense against indirect prompt injection attacks that injects layer-specific trainable embeddings encoding privilege levels into every decoder layer of an LLM, rather than only at the input layer. The core hypothesis is that input-level instruction hierarchy (IH) signals degrade as they propagate through the network; by recurrently injecting these signals, AIR achieves 1.6×–9.2× reduction in attack success rate (ASR) against gradient-based attacks compared to prior IH-based defenses (Delimiters, ISE), with minimal utility degradation across Llama-3.2-3B, Qwen-2.5-7B, and Llama-3.1-8B.

## Strengths

- **Strong mechanistic motivation with direct empirical validation.** The paper doesn't just hypothesize signal degradation—it demonstrates it via two complementary analyses: cosine similarity between privilege-level representations increases across layers for Delimiters/ISE (Figure 3), and linear probe accuracy for predicting privilege level drops from perfect to ~91% for ISE by the final layer, while AIR maintains near-perfect separability throughout (Figure 10, Appendix E). This dual evidence makes the motivation substantially more convincing than a purely intuitive argument.

- **Substantial robustness improvements on gradient-based attacks.** The ASR reductions are large and consistent: on GCG against Llama-3.2-3B with SFT, AIR achieves 4.1% ASR vs. 38% (Delim) and 48.1% (ISE); on Astra, 0.1% vs. 14.5% and 25.8%. These are not marginal improvements—they represent order-of-magnitude gains on the strongest attack category, which is where defensibility matters most.

- **Minimal architectural overhead.** The method adds only 0.4M parameters (0.005%) for Llama-3.1-8B and requires only a simple table lookup and vector addition per layer per token, making it straightforward to implement and deploy atop existing model stacks.

## Weaknesses

- **Insufficient attack optimization budget for gradient-based attacks.** The paper uses 50 steps (SFT models) or 200 steps (DPO models) for GCG/Astra optimization (Section 5.4). Standard GCG evaluations typically employ 500–1000+ steps. Fifty steps is exceptionally low and likely insufficient for attack convergence, especially against defended models where the loss landscape may be more complex. This risks substantially underestimating ASR and inflating the perceived robustness of all defenses, including AIR. The 1.6×–9.2× improvement claims rest on these under-powered attacks; their validity at higher budgets is unknown and critical to assess.

- **No adaptive attack evaluation.** All evaluated attacks treat the model as a fixed system. An attacker with white-box access who is aware of AIR's per-layer embedding tables could potentially optimize adversarial prefixes to counteract or cancel the injected IH signals (e.g., by learning perturbations that align with the embedding structure). This is a specific vulnerability of additive architectural defenses that the paper neither evaluates nor discusses, despite claiming robustness against "gradient-based (white-box)" attacks.

- **Missing ablation studies on AIR's own design.** The paper does not investigate key design choices: (a) whether injecting at every layer is necessary versus a sparse subset (e.g., every 4th layer), (b) whether layer-specific embeddings outperform a single shared embedding table across layers, or (c) the sensitivity of robustness to embedding dimensionality. Without these ablations, it is unclear whether the recurrent injection per se is the key factor, or whether simply adding more trainable parameters at any layer suffices.

- **Initialization sensitivity across architectures is a practical concern.** Appendix B.2 reveals that the default initialization ($\sigma=0.02$) failed for Qwen-2.5-7B, requiring a 5× larger $\sigma=0.1$. The paper attributes this to Qwen's larger activation magnitudes and provides heuristic guidelines, but no systematic ablation of $\sigma$ is performed. This sensitivity raises questions about out-of-the-box applicability to new architectures and whether the method requires per-model hyperparameter tuning that undermines its generality.

- **Residual attack success rates are unanalyzed.** Even with AIR, GCG achieves 22.6% ASR on Qwen-2.5-7B with SFT and 11.3% on Llama-3.1-8B with SFT. The paper does not analyze what characterizes the remaining successful attacks—whether they exploit specific token patterns, privilege conflicts, or architectural blind spots. Understanding these failure modes is essential for trusting the defense in high-stakes deployments.

- **No variance reporting across random seeds.** Given the acknowledged sensitivity to embedding initialization (Appendix B.2), the absence of multi-seed variance reporting for both robustness and utility metrics leaves the reliability of the reported numbers unverified. A single training run per configuration is insufficient to establish that the improvements are robust to initialization randomness.

## Nice-to-Haves

- Evaluation on reasoning benchmarks (e.g., GSM8K, MATH) to more sensitively detect capability degradation from modifying the residual stream at every layer; MMLU (Appendix G) is a start but tests factual knowledge rather than reasoning.
- Multi-turn conversational evaluation, which the paper explicitly scopes out but which is critical for agentic deployments—the primary motivating scenario.
- Comparison or combination analysis with detection-based defenses (guard models, perplexity filters) to clarify whether AIR is complementary to or redundant with that defense class.
- Sparse injection ablation (injecting at every $N$-th layer) to determine if full recurrent injection is necessary or if a more efficient variant suffices.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Inference latency concern** (harsh critic): Claimed that per-layer embedding lookups could disrupt kernel fusion and increase HBM bandwidth pressure. However, the operation is a single vector lookup from a tiny table (3 entries of dim 4096) plus an addition per layer—this is genuinely negligible compared to the attention and FFN computations. Without evidence of actual latency impact, this is speculative engineering nitpicking.

- **Architectural parity concern** (harsh critic): Whether the "None" baseline shares the same architectural modifications as AIR. The 0.005% parameter difference is too small to meaningfully conflate results with capacity differences, and the structural change (table lookup) does not alter the computational graph in a way that would independently improve performance.

- **Interference risk during Stage 1 training** (harsh critic): Speculation that AIR embeddings could "learn noise" during non-adversarial instruction tuning before adversarial training. This is purely hypothetical—the utility results show no degradation, which directly contradicts the concern. Without evidence of interference, this is unfounded.

- **Zero ASR on static attacks as evidence of memorization** (harsh critic): The paper explicitly acknowledges in Section 6.1 that Naive and Ignore attacks are "in-distribution as they are seen during adversarial training." The reviewer raised this as a concern, but the paper already addresses it.

- **Scalability to 70B+ models** (harsh critic): This is scope creep beyond the paper's stated experimental range. The limitation is acknowledged in Appendix A.

- **Missing related works / detection baseline comparison** (multiple reviewers): Hard rule—cannot flag missing related works. The comparison with detection methods is a nice-to-have, not a core flaw, since the paper explicitly positions itself within the IH defense framework.

- **Quantization compatibility** (spark finder): Not standard in this field for defense evaluation; scope creep.

- **Cross-architecture transfer of embeddings** (spark finder): Outside the paper's stated scope; the paper trains per-model.

## Novel Insights

The analogy between IH signal injection and positional encoding evolution (Section 4) is genuinely insightful. Just as the field moved from input-only sinusoidal/learned positional embeddings to RoPE's per-layer injection of relative position—and found this architectural choice critical for length generalization and performance—AIR applies the same principle to privilege information. This parallel suggests a broader meta-pattern: any signal that must persist through deep computation (position, privilege, task identity) may benefit from recurrent injection rather than input-only provision. The linear probing results in Appendix E provide causal-ish evidence that this is not merely an inductive bias but an observable phenomenon—ISE's probe accuracy degrades from 100% to 91% across layers while AIR stays near-perfect—making this one of the clearer demonstrations of "signal dilution" in intermediate representations that the community has produced.

## Suggestions

- **Re-run gradient-based attacks with substantially higher optimization budgets** (≥500 steps for all models) and report the resulting ASRs. This is the single most important action to validate the core claims. If AIR's advantage persists at higher budgets, the contribution is strong; if it shrinks significantly, the narrative needs revision.

- **Add a sparse injection ablation**: Test AIR with injection at every 2nd, 4th, and 8th layer. If every-other-layer injection matches full injection, the method becomes more efficient and the "recurrent" claim is refined; if robustness drops sharply, it validates the every-layer design.

- **Report results across at least 3 random seeds** for the primary configurations (especially AIR-DPO on Llama-3.1-8B and AIR-SFT on Qwen-2.5-7B) with standard deviations for both ASR and utility.

- **Discuss adaptive attacks explicitly** in the limitations section: acknowledge that an attacker aware of AIR could optimize against the embedding tables, and characterize what additional robustness (if any) the per-layer injection provides over input-only injection under such an adaptive threat model.

- **Analyze failure cases**: For configurations where ASR remains above 10% (e.g., AIR-SFT on Qwen-2.5-7B at 22.6%), examine the successful attack prefixes and model outputs to identify patterns, even qualitatively. This would strengthen the paper's contribution from "it works better" to "here is when and why it works, and here is where it doesn't."