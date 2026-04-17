Now I have sufficient calibration context. Let me write the final review.

## Summary

This paper introduces a mechanistic interpretability method for gated neurons (SwiGLU/GEGLU) based on cosine similarities between their input, gate, and output weight vectors, defining a taxonomy of "read-write functionalities" (strengthening, weakening, conditional strengthening, etc.). Applied across 12 LLMs, the method reveals consistent patterns: early-middle layers are dominated by conditional strengthening neurons, while late layers shift toward weakening neurons. The authors further show that weakening neurons—though few—have outsized influence on model behavior (attribute rate and entropy), and introduce a conditional ablation method revealing that much of this influence comes from negative gate values of the Swish activation function.

## Strengths

- **Simple, well-motivated method with striking cross-model consistency.** The cosine similarity approach is straightforward to compute and apply. The consistent "strengthening-then-weakening" pattern across 12 different LLMs (Figure 1a) is a genuinely interesting empirical finding that others will want to investigate further. The paper demonstrates this with clear visualizations and reports it across a broad model zoo—this cross-model consistency is a genuine contribution.

- **Discovery of the negative gate value mechanism.** The finding that negative Swish gate values play a functional role in weakening neurons (Section 6.2) is novel and challenges the common assumption that negative pre-activations are negligible. The conditional ablation analysis (Figure 3b) showing that case (iii)—where both x_gate < 0 and x_in < 0—accounts for much of the entropy effect is a specific, testable finding that concurrent work (Kong et al., 2025) independently confirms for a different phenomenon.

- **Conditional ablation methodology.** The technique of ablating specific activation quadrants based on sign patterns of x_gate and x_in (Section 6.2) is a useful methodological contribution that enables more fine-grained causal analysis than standard ablation approaches. This is widely applicable beyond this specific paper.

- **Strong negative correlation between cos(w_in, w_out) and activation frequency.** The nearly linear relationship (correlations ≤ -0.71 in most layers, Figure 4) bridges the weight-based analysis with functional behavior and reveals a compensatory structure in the model—rare specialized strengtheners vs. frequent weakeners.

- **The taxonomy itself is a useful conceptual framework.** Table 1 provides a clear, interpretable classification of neuron behaviors that is immediately applicable to any model with gated activations.

## Weaknesses

### Major:

- **Ablation experiments limited to a single model.** While the weight-based analysis covers 12 models, all behavioral/ablation experiments (Sections 6–8) are conducted exclusively on OLMo-7B on a single dataset (Dolma subset). The paper's strongest claims—"weakening neurons have outsize influence" (Abstract), "a mechanism important for transformer functionality" (Abstract)—rest on behavioral evidence from one model. Given the known diversity of MLP behavior across architectures, sizes, and training corpora, this is a significant evidential gap. The weight patterns may be universal, but the *behavioral importance* of weakening neurons is untested beyond this single model. The paper acknowledges this ("we focus on a single model: We choose OLMo-7B," Section 6) but then does not moderate its claims accordingly. (Similarly, reviews of the Copy Suppression paper—which studied only GPT-2 Small—flagged this limitation, resulting in scores of 3–6 and a Reject decision.)

- **Activation frequency confound in ablation experiments.** The paper shows (Section 7, Figure 4) that weakening neurons activate far more frequently than other classes (negative correlation of ≤ -0.71 between cos(w_in, w_out) and activation frequency). The baseline in Section 6.1—random neurons "from the same layers"—does not match on activation frequency, so the observed ablation effects could be largely explained by weakening neurons simply being more active rather than their RW functionality being causally important. The paper does not control for this confound, which is significant because: (a) more frequently active neurons naturally contribute more to the residual stream; (b) zero-ablating high-activity neurons removes more total activation mass than zero-ablating low-activity ones. Without a frequency-matched control, the "outsize influence" claim is not convincingly attributed to the weakening functionality per se.

- **The construct validity of "weakening neurons" as a functional class needs more support.** The classification relies on a threshold τ = ±0.5 on cosine similarities (Section 4), dividing a continuous distribution into discrete categories. The paper provides no sensitivity analysis for this threshold. More fundamentally, in models where features are known to be superposed (Elhage et al., 2022, cited by the authors), a single neuron's weight vectors may not correspond to coherent "read" and "write" directions—the cosine similarity could reflect an entangled mixture of feature directions. The authors acknowledge this possibility in Section 4.1 and Appendix D but argue that "findings from neurons will, to some extent, carry over to linear combinations of neurons." This is reasonable as a scope decision, but the semantic labels ("weakening" = "removes a direction from the residual stream") oversimplify what cosine similarity measures in a non-orthogonal feature space. The case study (Section 8) shows this tension explicitly: the weakening neuron 31.9634 is "much harder to interpret" than the strengthening neuron, and the best interpretation relies on the x_gate < 0 case—suggesting the "weakening" label may not capture the actual functional behavior in a straightforward way.

### Minor:

- **The novelty of "negative gate values matter" may be overstated.** The paper claims "for the first time, we observe a mechanism important for transformer functionality that involves negative gate values." The concurrent work of Kong et al. (2025) is acknowledged, but the mathematical observation itself—that negative Swish outputs flip a weakening neuron's effective sign, making it behave like a strengthener—is a direct consequence of the SwiGLU formula. The paper's own explanation (Section 7, paragraph starting "This is also explanatory") confirms this: when x_gate < 0, Swish(x_gate) < 0, so a weakening neuron (negative cos(w_in, w_out)) gets a sign flip that makes it strengthen. This is algebraically expected; the empirical contribution is demonstrating this occurs in practice and affects model behavior, not the existence of the mechanism per se.

- **Single case study for weakening neurons.** Only one weakening neuron (31.9634) is examined in detail, and the analysis reveals it is "much harder to interpret" than the strengthening neuron. More case studies of weakening neurons with diverse behaviors would strengthen the claim that they form a coherent functional class.

- **Overgeneralization from median statistics.** Figure 1a shows that the median cos(w_in, w_out) becomes slightly negative in late layers, and the paper interprets this as "late layers tend more towards weakening." But a median slightly below zero could reflect many neurons with small negative cosines rather than a substantial weakening population. Figure 1b partially addresses this, but the threshold-based classification is sensitive to τ.

- **Limited metrics for behavioral importance.** The ablation experiments use attribute rate and entropy of the output distribution, but do not directly measure perplexity/loss—perhaps the most standard and interpretable metric. Whether weakening neuron ablation degrades language modeling capability on held-out text is not shown. While attribute rate and entropy are reasonable proxies, loss would make the "outsize influence" claim more directly convincing.

## Nice-to-Haves

- Ablation experiments on at least one additional model (e.g., Llama-3.2-3B or Gemma-2-2B) to test behavioral generality, even if on a smaller dataset.
- A frequency-matched control in ablation experiments (e.g., ablating non-weakening neurons with similar activation frequency/magnitude distributions) to isolate the effect of RW functionality from activation statistics.
- Sensitivity analysis on the threshold τ = 0.5, showing how the distribution of neurons across RW categories and the ablation effects change with different thresholds (e.g., 0.3, 0.4, 0.6, 0.7).
- Perplexity/loss as an ablation metric, in addition to attribute rate and entropy.
- A more rigorous analysis of how much of the ablation effect is explained by activation frequency vs. RW class (e.g., a variance decomposition or regression).

## Removed Points

- **Harsh Critic Point 1 (full construct validity challenge):** While the concern about superposition and whether cosine similarity captures "true" functionality is legitimate, the paper explicitly acknowledges this limitation (Section 4.1, Appendix D) and argues that neurons can still be a meaningful unit of analysis. The superposition debate is an active area of research, and requiring the paper to solve it before applying neuron-level analysis is too stringent. The partial validity is kept in the "Minor weaknesses" section above, but the claim that "the taxonomy remains an interpretive label, not an empirically established class" is overly dismissive—Figure 2 and the random baselines provide empirical grounding that the cosine patterns are non-random.

- **Harsh Critic Point 2 (preprocessing step concerns):** The concern that the sign-flipping preprocessing (Section 3.2) could reshape cosine similarity distributions is valid, but the paper provides an argument (Appendix C) for why it does not change model behavior. The question of whether it changes *interpretation* is reasonable, but this is presented as a deliberate preprocessing choice for interpretability, analogous to TransformerLens's own preprocessing. This is a matter of methodological choice rather than a fundamental flaw.

- **Harsh Critic Point 4 (overgeneralization from one model, full version):** While the single-model limitation is real and kept as a Major weakness, the harsh reviewer's claim that "behavioral claims are tested exclusively on OLMo-7B... generalization from a single 7B model... is not warranted" goes too far. The weight-based analysis *is* conducted across 12 models and shows strong consistency, and the behavioral analysis is appropriately scoped to OLMo-7B in the paper. The issue is that the abstract/conclusion overstate the behavioral claims, not that the analysis itself is invalid.

- **Spark Point about testing non-gated models:** This is scope creep. The paper explicitly focuses on gated activation functions (SwiGLU/GEGLU) and provides clear motivation for doing so. Requesting comparison with GPT-2 (which uses GELU) would be interesting but is not within the paper's stated scope.

- **Spark Point about SAE comparison:** The paper explicitly argues for the neuron-level approach (Section 4.1) and defers SAE investigation to future work. Demanding SAE comparison is reasonable as a nice-to-have but not as a core weakness.

## Novel Insights

The most genuinely novel finding is the specific, quantified demonstration that negative Swish gate values in weakening neurons have a functional role that cannot be reduced to ReLU-like behavior. The conditional ablation showing that case (iii) (x_gate < 0, x_in < 0) drives the entropy-sharpening effect is the paper's most striking empirical result—it transforms an algebraic curiosity (negative gates flip signs) into a concrete mechanistic finding. The nearly linear negative correlation between cos(w_in, w_out) and activation frequency (Figure 4) is also a genuinely new and non-obvious observation about the compensatory structure of gated MLPs.

## Suggestions

1. **Add a frequency-matched control experiment:** Ablate random neurons from the same layers matched on activation frequency and/or activation magnitude, and compare the effect sizes. This directly addresses the activation frequency confound and is the single most impactful experiment that could strengthen the paper.

2. **Moderate claims in the abstract and conclusion:** When making behavioral claims, explicitly qualify them as "in OLMo-7B" rather than universal statements. The weight-based patterns may be universal, but the behavioral importance findings are model-specific until demonstrated otherwise.

3. **Add at least one more model for behavioral experiments:** Even a smaller-scale ablation experiment on Llama-3.2-1B or Gemma-2-2B (e.g., just the entropy and basic loss effects) would substantially increase confidence that the finding is not OLMo-specific.

4. **Provide sensitivity analysis on τ:** Show how the proportion of neurons classified as "strengthening" vs. "weakening" changes with different thresholds, and ideally how ablation effects change.

5. **Report perplexity/loss from ablation:** Adding this standard metric would make the behavioral findings more immediately interpretable and comparable to other work.

## Score and Decision

**Calibration comparison:**
- Copy Suppression (Reject, scores 3–6): Detailed analysis of a single component in GPT-2 Small, limited generalizability. The current paper is stronger: it covers 12 models for its weight analysis and finds a genuinely novel phenomenon (negative gate mechanism).
- Knowledge Neuron Thesis (Accept spotlight, scores 6–8): Rigorous challenge to a widely-held thesis, broad evaluation. The current paper is weaker: its behavioral claims are less rigorously controlled.
- Capability Localization (Accept Poster, scores 5–6): Neuron localization with arbitrary thresholds, limited behavioral validation. Similar weaknesses to the current paper.
- Neuron Predictability Lens (Reject, scores 3–5): Simple framework with limited model scope and novelty concerns. The current paper is more novel and thorough.
- Safety Neurons (Reject, scores 3–6): Neuron-level analysis with limited model scope. The current paper has broader cross-model validation for its weight analysis.

The paper's weight-based analysis across 12 models is a genuine contribution, and the discovery that negative gate values play a functional role is novel and important. However, the behavioral claims rest on experiments from a single model without adequate controls for the activation frequency confound, and the taxonomy's construct validity is not fully established. These are significant but not fatal weaknesses—the findings are interesting enough to warrant dissemination, but the overclaiming in the abstract needs to be toned down.

Score: 5.5

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>