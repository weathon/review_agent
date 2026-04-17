Now let me compose the final review, carefully verifying each claimed weakness against the paper text.

## Summary

This paper identifies that existing prompt injection defenses based on instruction hierarchy (IH) inject privilege signals only at the input layer, and hypothesizes that these signals degrade as they propagate through decoder layers. To address this, the authors propose Augmented Intermediate Representations (AIR), which adds layer-specific trainable embeddings keyed on privilege level to intermediate token representations at every decoder block. Evaluations across three model families (3B–8B) and two training methods (SFT, DPO) show AIR reduces gradient-based attack success rates by 1.6×–9.2× compared to prior IH injection methods (Delimiters, ISE) while largely preserving utility.

## Strengths

- **Clean, well-motivated architectural idea with negligible overhead**: The observation that input-layer IH signals may lose efficacy across layers (supported by the cosine similarity analysis in Figure 3) is a compelling motivation. AIR's design—adding a per-layer embedding table indexed by privilege level—is simple, adding only 0.005% parameters (0.4M for Llama-3.1-8B), making it practically deployable. The analogy to positional embeddings (input-only vs. RoPE-style injection throughout) provides intuitive grounding.

- **Systematic empirical comparison**: The paper evaluates 3 IH injection mechanisms (Delimiters, ISE, AIR) × 2 training methods (SFT, DPO) × 3 model families × multiple attack types, including both static and gradient-based attacks. This is above the standard for the field and enables direct, controlled comparisons. Prior work like the original Instruction Hierarchy paper evaluated on only one model with no baselines.

- **Strong and consistent improvements on gradient-based attacks**: AIR achieves dramatic ASR reductions on GCG and Astra (e.g., Qwen-2.5-7B DPO: GCG ASR drops from 7.7% to 1.6%; Astra ASR from 2.3% to 0.9%). These are the hardest attack types in the evaluation, and the improvements are consistent across all models and training regimes.

- **Best utility-separation tradeoff with DPO**: Figure 8 shows AIR+DPO achieves the highest separation scores on SEP while maintaining competitive utility, corroborating and extending prior findings about DPO's advantages for robustness.

## Weaknesses

### Major:

- **1 — Narrow robustness evaluation with a single adversarial target ("hacked!")**: All gradient-based and static attack evaluations on AlpacaFarm use the fixed adversarial instruction "print 'hacked!'" as the target. Robustness is measured as whether the model outputs this exact string (or its likelihood). This is an extremely narrow proxy for whether the model genuinely enforces IH under diverse adversarial goals. A defense that selectively suppresses this particular token sequence could score well while being vulnerable to semantically equivalent or rephrased adversarial instructions. While this is standard in some prior work (e.g., ISE uses similar setups), it limits the strength of the robustness claims. The SEP benchmark partially mitigates this with a witness-based metric, but SEP still uses a synthetic probe-witness structure that may not capture diverse real-world adversarial behaviors. The paper's abstract claims "1.6× to 9.2× reduction in attack success rate on gradient-based prompt injection attacks" without qualifying that this applies to a single fixed target string, which overstates the generality of the findings.

- **2 — No evaluation against adaptive attacks**: Since AIR introduces visible, architecturally-embedded signals at every layer, an attacker with white-box access could potentially exploit knowledge of the per-layer embedding tables. For example, a gradient-based attack could be modified to account for the additive privilege embeddings (e.g., by optimizing adversarial tokens whose intermediate representations align with high-privilege directions). The paper does not discuss this threat model or evaluate against any defense-aware adaptive attack, which is a significant gap for a security contribution. As noted in human reviews of related IH defense papers: "As only standard, off-the-shelf attacks were tested, it is still unknown whether adaptive attacks tailored against the instruction hierarchy could achieve higher success rates."

- **3 — No ablations on the core design choice**: The paper's central claim is that injecting IH signals at *every* layer is important. Yet there is no ablation testing whether injecting at fewer layers (e.g., every other layer, only the last N layers, or only the first N layers) achieves similar gains. Without this, it is unclear whether the benefit comes from recurrent reinforcement across layers specifically, or simply from having IH information available at deeper layers (which could be achieved with simpler modifications, e.g., re-applying the input-level embedding at intermediate layers). This is a critical missing piece for understanding whether the method's novelty—per-layer trainable embeddings—actually drives the improvement.

### Minor:

- **4 — Mechanistic claim about signal degradation is only suggestively supported**: Figure 3 shows that average cosine similarity between tokens of different privilege levels increases with depth for Delim and ISE but stays lower for AIR. The paper interprets this as IH signals "failing to adequately preserve." However, higher average similarity does not directly prove that privilege information is lost—it could reside in a subspace or be encoded differently at deeper layers. No probing experiment (e.g., training a linear classifier on intermediate representations to predict privilege level) is conducted to verify that IH information is actually less recoverable at deeper layers without AIR. This weakens the mechanistic narrative, though it does not undermine the empirical effectiveness.

- **5 — Training and evaluation distributions are tightly coupled**: Adversarial training uses only Naive and Ignore attacks on Alpaca data, and evaluation uses AlpacaFarm with the same data format and similar or identical attack types (plus gradient-based attacks). While GCG and Astra optimize novel prefixes not seen during training, the fixed "hacked!" target and Alpaca-style formatting remain constant. There is no evaluation on out-of-domain tasks, different data formats, or different adversarial objectives, making it unclear how well AIR generalizes beyond this setup.

- **6 — Utility degradation with SFT**: For Qwen-2.5-7B and Llama-3.1-8B trained with SFT, AIR shows lower utility than even the non-adversarially trained baseline (acknowledged in Section 6.2). This suggests AIR may interact poorly with full fine-tuning in some settings, and the paper does not analyze why.

### Trivial:

- The formalism in Section 2 introduces an alignment function A(O,I) that is never used in the evaluation; metrics instead rely on string matching or witness inclusion.

## Nice-to-Haves

- Evaluate with multiple diverse adversarial targets (e.g., exfiltration-style instructions, multi-sentence adversarial goals) to verify AIR's robustness is not target-specific.
- Conduct adaptive attack experiments where the attacker has knowledge of the AIR embedding tables and can optimize accordingly.
- Perform layer-wise ablations (inject at every 2nd layer, only last N layers, etc.) to validate that full per-layer injection is necessary.
- Analyze the learned embedding tables: are they similar or divergent across layers? This would provide mechanistic insight into whether AIR learns layer-specific privilege representations or redundant copies.
- Test on larger (70B+) and already-instruction-tuned models to assess practical deployability.
- Evaluate in multi-turn or agentic settings, as the authors themselves flag as a limitation.

## Removed Points

- **Reproducibility and hyperparameter concerns (Critic/Neutral/Spark)**: Concerns about undisclosed hyperparameters for baselines, initialization details, etc. The paper provides learning rates, training epochs, optimizer, batch sizes, LoRA configs, and embedding initialization strategy. These are standard for this venue.

- **Baseline fairness / parameter budget concerns (Critic)**: The concern that ISE and Delimiter baselines might not be implemented in their strongest configuration. The paper uses identical training procedures for all methods and directly mirrors prior papers' IH injection mechanisms (trainable delimiter tokens, trainable segment embeddings). This is a fair comparison setup.

- **No comparison with non-IH-based defenses (Spark)**: The paper explicitly scopes itself to improving IH injection mechanisms and provides a clear comparison framework within that family. Comparing to fundamentally different defense paradigms (perplexity filtering, detection-based methods) is outside the stated scope.

- **Limited utility evaluation scope (Human Finder)**: The paper uses AlpacaEval win rate and SEP utility, which are standard for the IH defense literature. Requesting additional benchmarks like reasoning or coding is beyond scope for a security-focused contribution.

- **Formatting nitpicks**: Removed.

## Novel Insights

The positional embedding analogy (Section 4) is genuinely insightful: just as positional encodings evolved from input-only (sinusoidal/learned) to per-layer mechanisms (RoPE), IH signals may benefit from the same progression. Figure 3's analysis showing that AIR maintains lower cross-privilege cosine similarity throughout the network provides initial evidence that per-layer injection changes representational dynamics, though the causal link to robustness remains unproven.

## Suggestions

1. **Add at least one layer ablation** (e.g., injecting IH embeddings at every 2nd layer) — this is the single most important experiment for validating the core claim.
2. **Evaluate against at least one adaptive attack** that incorporates knowledge of the AIR embeddings into the optimization objective.
3. **Add probing experiments** to verify whether privilege information is recoverable from intermediate representations with and without AIR, strengthening the mechanistic narrative.
4. **Soften claims in the abstract and conclusion** to specify that the 1.6×–9.2× ASR reduction applies to specific fixed-target gradient-based attacks, not to prompt injection attacks in general.

## Score and Decision

**Calibration:**
- ISE (same topic, accepted as poster, scores: 6,6,6,6): AIR is stronger in that it improves over ISE with clear empirical gains and systematic comparison, but weaker in that it doesn't evaluate on as many diverse benchmark scenarios.
- Instruction Hierarchy (rejected, scores: 3,6,8,3,8,3,6): AIR improves on IH's weaknesses (multiple models, proper baselines, gradient-based attack evaluation) but shares some evaluation narrowness.
- PFT (rejected, scores: 5,3,3,6): AIR is clearly stronger — better motivation, more thorough experiments, real improvements over prior methods.
- RobustKV (accepted as poster, scores: 6,8,6): AIR is comparable — both propose architectural defense modifications with good empirical results, both lack adaptive attack evaluation.

AIR makes a real contribution: the first systematic demonstration that injecting IH signals at intermediate layers meaningfully improves robustness over input-only approaches, with strong gains on gradient-based attacks. The weaknesses are significant but do not invalidate the core finding — they limit the generality of claims and leave open questions about mechanism and adaptive robustness. The paper is approximately at the same quality level as ISE (which was accepted as poster) but with a complementary contribution.

MY FINAL SCORE: <pineapple>6</pineapple>
MY FINAL DECISION: <orange>Accept</orange>