=== CALIBRATION EXAMPLE 5 ===

# Final Consolidated Review
---

## Summary
CodeUnlearn proposes a machine unlearning framework for language models that inserts a discrete codebook bottleneck (structured as a Sparse Autoencoder) into an intermediate transformer layer, jointly trains the full model with this bottleneck, and then unlearns specific topics at zero gradient-update cost by statistically identifying and removing codes enriched for the target topic via a chi-squared test. The method is evaluated on T5-small (60M parameters) for English-to-French translation on a literary text corpus. The paper claims this is the first method to successfully enable topic-level unlearning with contextual relevance in a language model.

---

## Strengths

- **Conceptually novel application of discrete bottlenecks to unlearning.** The idea of routing all downstream information through a codebook so that targeted knowledge can be surgically removed without gradient updates is a fresh and principled connection between mechanistic interpretability (superposition/disentanglement) and machine unlearning. This is not merely applying an existing framework—it identifies a structural property (discreteness + bottleneck) that intrinsically supports deletion.

- **Genuinely zero-gradient unlearning phase.** Once the codebook is trained, the actual unlearning step involves only a frequency-ratio statistic and code deletion—no backpropagation. This is a real computational advantage over gradient ascent, SCRUB, and influence-function methods that require optimization loops at unlearning time.

- **Qualitative evidence of semantic displacement (Table 1).** The progressive translation degradation ("l'aimer" → "le recevoir" → "le mettre en état") demonstrates that code deletion alters the underlying conceptual representation, not just surface probabilities. This is a more meaningful signal than a simple output suppression.

- **Amortized reusability across topics.** The same trained codebook can be used to unlearn multiple topics sequentially without retraining, which is a genuine scalability advantage over methods that require per-request model updates.

---

## Weaknesses

### Fatal
*(none that singularly invalidate the approach in principle, but the combination of Major issues constitutes a very weak empirical case)*

### Major

- **Absence of any unlearning baselines.** There is no comparison against even the simplest alternatives (gradient ascent on the forget set, random code deletion, zeroing attention heads). Without baselines, the paper cannot establish that CodeUnlearn offers a better forgetting–utility trade-off than trivial alternatives. This is the single largest experimental gap for an ICLR submission.

- **Evaluation metrics do not verify that unlearning has occurred.** BLEU, METEOR, BERTScore, and BARTScore measure translation quality degradation on target sentences—not whether specific knowledge has been forgotten. A model that degrades generally (e.g., from numerical instability or random code deletion) would produce the same signal. Standard unlearning evaluation requires showing the model's behavior on the forget set approaches that of a model retrained without that data, or verifying via membership inference attacks (MIA) or probing classifiers that the targeted knowledge is genuinely inaccessible. Neither is present.

- **Severe and inconsistent collateral damage on non-target data.** Table 2 shows that for several topics, normalized improvement on D_R falls *below* 0 (i.e., below even the zero-shot baseline without any codebook), indicating that unlearning these topics actively degrades general model capability beyond what the codebook insertion already costs. Specific cases: "Julien" BLEU −65.70%, BERTScore −94.63%; "Wish" BLEU −87.65%, METEOR −94.51%. These are not minor side effects — the model loses a large fraction of its translation ability on unrelated content. The paper's claim that "performance on non-target prompts remains relatively stable" is only plausible for a subset of topics (e.g., Captain, Poor at modest levels) and is contradicted for others. The reason — likely code polysemanticity — is not analyzed.

- **Non-standard "zero-shot" definition conflated with the paper's headline claim.** Section 3.6 explicitly redefines "zero-shot" to refer only to the unlearning phase, while the full pipeline requires joint re-optimization of all LM parameters (Eq. 8). The abstract and introduction repeatedly invoke "zero-shot" without this qualification, creating a misleading impression that CodeUnlearn can be applied to already-deployed models at no training cost. This matters practically: any deployment use-case that motivated this work (removing sensitive knowledge from an already-deployed model) requires a full re-train.

- **Experimental scope is too narrow to support claims about "LLMs."** All experiments use T5-small (60M parameters) on a single literary translation task. The paper's title, abstract, and conclusion use "LLM" and make claims about "real-world applications," yet 60M-parameter translation models represent a very small and atypical use case. It is unknown whether the codebook bottleneck can be inserted into billion-parameter autoregressive models without prohibitive performance costs, and whether the code statistics remain interpretable at that scale.

### Minor

- **Control dataset construction (D̄_T) is underspecified.** Section 3.5 says target-topic words are "replaced with unrelated terms," but does not specify: (a) how target words are identified (manual lexicon? automated?), (b) what "unrelated" means, or (c) whether this procedure preserves statistical properties that could confound the chi-squared test. The quality of D̄_T directly determines which codes are selected for deletion.

- **L1 penalty is applied to code vector entries, not code activation frequency.** Equation (7) penalizes the internal magnitudes of activated codes, which encourages sparse internal structure within individual code vectors—a different objective from SAE sparsity, which discourages too many codes from activating simultaneously for a given input. The paper conflates these two notions ("sparse, interpretable features") without distinguishing them. This may reduce monosemanticity in ways the paper does not account for.

- **Normalization scheme in Section 3.7 obscures absolute performance.** Setting 0 = zero-shot baseline and 1 = codebook model means a value of −0.66 indicates the unlearned model is worse than even the baseline that was never trained with the codebook. While this normalization helpfully shows relative degradation, reporting absolute scores alongside it (as partially done in Table 2) is necessary to let readers judge whether the starting point is acceptable. Absolute BLEU scores of 0.12–0.20 on the test set suggest the model quality is borderline even before unlearning.

- **No discussion of codebook overhead before unlearning.** There is no measurement of how much performance the model loses simply by inserting and jointly training with the codebook, independent of any unlearning. This baseline cost is necessary to interpret the unlearning results.

### Tiny

- **Ambiguous topic choices.** "White," "Black," "Captain," "Poor" are polysemous, high-frequency words whose codes likely encode many unrelated meanings. Their selection inflates collateral damage relative to more specific topics, but this is not discussed or controlled for.
- **Figure captions repeat verbatim (e.g., Figure 3 caption appears three times).** Minor editing issue.
- No explicit limitations section in the paper.

---

## Nice-to-Haves

- Validate unlearning with Membership Inference Attacks (MIA) or probing classifiers to confirm the targeted information is genuinely inaccessible, not just harder to surface in translations.
- Shift at least one evaluation to a knowledge-intensive generative task (e.g., TriviaQA, MMLU subset) where unlearning specific facts can be measured more directly.
- Ablation on codebook placement layer (why the third encoder layer?), codebook size, and S parameter to help future researchers configure the method.
- Code activation heatmaps showing that removed codes activate specifically on target-topic inputs and are largely silent on D_R, to directly support the monosemanticity/disentanglement claim.
- Investigate whether a smaller S' (fewer codes removed) can achieve acceptable target degradation with substantially lower D_R collateral damage—a frontier curve between forgetting efficacy and utility preservation would be highly informative.

---

## Removed Points
*These points were flagged for removal; treat with caution.*

- **[REMOVED — Architectural misread] Harsh critic's claim that "placing the codebook after the residual connection prevents information leakage via the residual connection" is architecturally confused.** The paper's reasoning is actually correct: if the codebook were placed *before* the residual merge, the original activations would bypass the bottleneck via the skip connection. By placing it *after* the merge, all information—skip path included—is forced through the discrete bottleneck. The critic inverted the logic.

- **[REMOVED — Related works] Criticism about omitting gradient ascent, SCRUB, LEACE, etc. from the related work section.** Per synthesis instructions, related-work omission criticism is excluded as external sources cannot be confirmed.

- **[REMOVED — Scope creep on theoretical guarantees] Demand for certified unlearning guarantees or formal privacy proofs.** This is an empirical systems paper; formal guarantees are not standard in this sub-field and are not claimed by the authors.

- **[REMOVED — Unfair asymmetric comparison] Criticism that the method's comparison against naive deletion or zeroing approaches should be included as a lower bound.** The paper explicitly frames itself as "a baseline," so demanding comparison to weaker baselines is less critical; the real gap is comparison to stronger methods.

- **[WEAKENED — Zero-shot terminology] The "zero-shot" label is non-standard but the paper transparently discloses the definition in Section 3.6.** The weakness is kept as Major because the disclosure is buried and the abstract/title create a misleading impression for practitioners, but it is not classified as outright deceptive.

- **[WEAKENED — Synonym leakage framing] Harsh critic frames Figure 5 (unlearning "love" degrades "like") purely as collateral damage.** The paper reasonably interprets semantic generalization as a designed feature for contextual unlearning. Whether this is desirable depends on the use case; it is not straightforwardly a flaw.

---

## Novel Insights

The spark finder identifies a genuinely underexplored diagnostic: **code polysemanticity as the root cause of collateral damage.** If the codes targeted for deletion represent multiple unrelated concepts (due to superposition in the codebook), then removing them will inevitably harm unrelated content. This is an architectural vulnerability that is distinct from the standard trade-off between forgetting efficacy and model utility—it predicts that collateral damage will be *especially severe for polysemous topic words* and *especially mild for rare proper nouns*. This prediction is qualitatively consistent with Table 2 (Captain and White show lower D_R damage than Julien and Wish in several metrics) and suggests a concrete diagnostic (probing code activation distributions for target vs. unrelated inputs) that could both explain the failures and guide architecture choices for future work.

---

## Suggestions

1. **Run MIA/probing before claiming unlearning.** Replace or supplement BLEU/METEOR on D_T with a classifier-based probe or MIA that directly tests whether target-topic representations persist. This is the minimum necessary to assert that information has been removed rather than degraded.
2. **Add at least one gradient-ascent baseline.** Even a simple gradient ascent on D_T with early stopping, reported on the same BLEU/BERTScore metrics, would let readers understand where CodeUnlearn sits on the forgetting–utility Pareto frontier.
3. **Measure and report codebook insertion cost separately.** Report: (a) zero-shot model (no codebook), (b) codebook model before unlearning, (c) codebook model after unlearning. Currently the normalization collapses (a) and (b) into a single "0" reference, hiding the performance cost of the architectural change itself.
4. **Address polysemantic codes explicitly.** Show activation patterns for at least one topic (e.g., "Love") to verify that removed codes are predominantly activated by target-topic inputs and not by unrelated ones. If they are not monosemantic, quantify how many codes are "shared" and discuss mitigation strategies (finer-grained codebooks, constrained code selection).
5. **Revise abstract and title framing.** "Zero-shot" in the title will be read by the community as applicable to existing deployed models. Retitling to something like "Training-Free Unlearning via Discrete Codebook Features" would be more accurate and avoid immediate credibility damage.
6. **Evaluate on a larger model (T5-base minimum) or provide a concrete argument** for why the codebook bottleneck scales without prohibitive performance overhead as model size increases.

---

**Axis evaluations:**
- **Novelty:** Moderately high — the use of a discrete bottleneck specifically for zero-gradient unlearning is a fresh framing, though the individual components (SAEs, VQ-codebooks, chi-squared enrichment tests) are not new.
- **Technical soundness:** Weak — the L1 penalty conflation, underspecified D̄_T construction, and lack of any analysis of codebook non-bypassability leave core mechanisms unverified.
- **Empirical support:** Poor — no baselines, inappropriate evaluation metrics for the unlearning claim, and severe collateral damage on multiple topics directly contradicting the paper's central claim.
- **Significance:** Currently low — the experimental scope (single small translation model, no proper unlearning verification) prevents the community from adopting or building on this work with confidence.
- **Clarity:** Moderate — the core idea is well-presented, but the normalization scheme, the "zero-shot" definition, and the figure captions are confusing.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0, 1.0, 3.0]
Average score: 3.8
Binary outcome: Reject
