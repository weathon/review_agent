Now I have all the information I need. Let me compose the final review.

## Summary

This paper identifies and formalizes the mismatch between class labels and target concepts in machine unlearning, introducing three new tasks beyond the conventional "all matched" setting: target mismatch, model mismatch, and data mismatch forgetting. The key insight is that practical unlearning requests may require forgetting concepts that are larger or smaller than a single training class. The authors propose TARF, a three-phase framework using annealed gradient ascent for forgetting and target-aware gradient descent on selected retaining data, motivated by a theoretical result linking representation distance to forgetting dynamics ("gravity effects"). TARF is evaluated across CIFAR-10/100, ImageNet-1k, and extended to LLM and diffusion model applications.

## Strengths

- **Novel and well-motivated problem formulation.** The four-task taxonomy (all matched, target mismatch, model mismatch, data mismatch) and the formal notation using label domain relationships (L₁ = L₂ vs. L₁ ≺ L₂) provide a clear, general vocabulary for discussing label-concept mismatches in unlearning (Section 2, Figure 1, Table 1). This conceptual contribution will likely shape how future work frames unlearning problems.

- **Compelling empirical demonstration of existing methods' failures on mismatched scenarios.** Figure 2 systematically shows that GA, FT, L1-sparse, and BS all break down in specific, analyzable ways under label domain mismatch, making a concrete case that the new settings address real gaps, not ornamental variations.

- **Strong empirical improvement on the newly introduced mismatch scenarios.** In Table 3, TARF achieves dramatically lower Gap to the retrained reference in the most challenging settings — target mismatch (0.96 on CIFAR-10, 0.21 on CIFAR-100) and data mismatch (0.96, 1.17) — where prior methods exhibit severe failures (e.g., SCRUB gaps of 25.53/29.90). In the conventional all matched setting, TARF remains competitive (Gap 1.01 vs. SCRUB's 1.03 on CIFAR-10).

- **Comprehensive evaluation across scales and modalities.** Beyond CIFAR-10/100, the paper evaluates on ImageNet-1k (Table 4), applies TARF to LLM unlearning with TOFU (Table 5), and demonstrates concept removal in stable diffusion (Figure 6). Ablations cover multiple architectures and hyperparameter choices.

- **Fine-grained evaluation that reveals method behavior.** Table 2 breaks down model mismatch into forgetting (UA-F) and affected-retaining (UA-R) components, showing TARF achieves the best Gap (3.42 on CIFAR-10, 1.36 on CIFAR-100) by effectively separating the target concept from entangled retaining data.

## Weaknesses

### Fatal
None.

### Major

- **The assumption that the number of target-concept classes is known limits practical applicability.** Section 2 states: "we assume that the number of classes in Dun belonging to the target concept is known in target mismatch forgetting." This assumption underlies how β is set for Phase I target identification. While the paper is transparent about this, it means the method's identification mechanism operates with oracle knowledge of the target concept's scope — the very information one would want to discover in practice. The paper mentions a quantile-based alternative ("setting the threshold β as the lowest value of top-10% data within a descending order") and studies robustness to varied quantile choices in Appendix E, but the main results rely on the oracle assumption. Knowing how many classes to select is substantially easier than knowing which classes, so this is not circular as some might claim, but it does restrict the method's applicability to scenarios where the scope of the target concept is at least approximately known.

### Minor

- **The theoretical contribution (Theorem 3.2) provides intuition but does not directly inform the algorithm's key design choices.** The theorem bounds the loss-change difference between two subsets as proportional to their representation distance, under a Lipschitz smoothness assumption. This motivates the idea of using representation proximity as a signal for identifying affected data (Phase I), which is reasonable. However, the specific algorithmic components — the annealing schedule k(t), the three-phase structure, the timing parameters t₀ and t₁, and the selection threshold β — are not derived from the theorem. The paper uses "based on the intuition" language (Section 3.3), which is somewhat honest about this gap, but the overall framing (Section 3.2 → 3.3) implies a tighter theory-method connection than exists. The theory is motivational rather than deductive, which is acceptable but should be more explicitly acknowledged.

- **The Gap metric aggregates metrics on different effective scales, which can obscure individual tradeoffs.** Gap = average of |R_{θ_un} − R_{θ_r}| across UA, RA, TA, MIA. While all metrics are on 0–100 scales, the differences from the retrained reference can vary enormously. For instance, SCRUB's data mismatch CIFAR-100 Gap of 45.54 includes an MIA contribution of |15.11 − 100.00| = 84.89 and a UA contribution of |95.50 − 0.00| = 95.50, both reflecting genuine failures, but their combined dominance means RA and TA deviations (1.29 and 0.47 respectively) are effectively invisible in the aggregate. The paper does report individual metrics, which mitigates this concern, but the headline Gap numbers should be interpreted with awareness of this composition effect.

- **Limited characterization of when representation gravity fails.** The paper acknowledges in the conclusion that the gravity signal becomes weaker when concepts are ambiguous or weakly clustered, but provides no empirical characterization of degradation boundaries. A controlled experiment varying the semantic distance between forgetting and false-retaining data would reveal the method's operating envelope, which is precisely where the method would be most needed.

### Trivial
None.

## Nice-to-Haves

- Comparison against a simple oracle baseline that directly uses ground-truth target concept labels for fine-tuning on true retaining data, to isolate TARF's contribution beyond the information advantage from knowing the target concept's scope.
- t-SNE/UMAP visualizations of representation dynamics across the three phases, to make the disentanglement mechanism more visually compelling.
- Automatic threshold selection for β (e.g., knee-point detection in ranked accuracy drops) to reduce reliance on oracle knowledge.

## Removed Points

These points are flagged to be removed, treat them with caution.

- **Harsh critic's claim that target identification is "circular."** The critic called the method's identification mechanism circular because it presupposes knowledge it claims to discover. However, knowing HOW MANY classes belong to the target concept (a count) is substantially different from knowing WHICH classes belong (the identity). The gravity signal does the identification; the count merely sets the selection threshold. The method uses the signal to answer the hard question (which classes?) while being given the easier one (how many?). This is a legitimate practical limitation, but calling it "circular" overstates the case.

- **Harsh critic's claim that Theorem 3.2 is "essentially a consequence of Lipschitz continuity" and therefore trivial.** While the theorem does follow from a Lipschitz smoothness assumption, it connects this to the specific dynamics of gradient ascent unlearning, yielding a bound that involves representation distance, the Jacobian eigenvalue, and gradient norms. The specific form provides the "gravity" interpretation that motivates Phase I. The result is not groundbreaking, but it is not trivially obvious either.

- **Strength finder's claim that Theorem 3.2 "directly motivates the target-identification and target-separation components of TARF."** This overstates the connection. The theorem motivates the general idea of using representation proximity as a signal, but does not directly motivate the specific algorithmic components (annealing, three phases, etc.).

- **Harsh critic's demand for missing related works.** Removed per rules — cannot verify existence of external references.

- **Formatting/presentation nitpicks.** Removed per rules.

- **Reproducibility concerns about hyperparameters.** The paper provides practical guidelines in Appendix E and a code repository. Removed per rules.

- **Strength finder's claim about "empirical validation in Figure 3 confirming that representation distance correlates with accuracy/loss change patterns" as a separate strength.** This is already captured as part of the broader empirical strengths and the theoretical motivation discussion.

- **Harsh critic's claim that "gradient cleaning may be better than gradient ascent for false retaining data, which somewhat contradicts the paper's emphasis on gradient ascent."** The ablation in the right of Figure 7 shows gradient cleaning performs better on RA for false retaining data, but the paper's main emphasis is on gradient ascent for the forgetting data (D_f), not for the false retaining data (D_fr). The gradient descent on D_fr is the retaining part. This is not a contradiction.

## Novel Insights

The most insightful observation across the reviews is that TARF occupies an interesting middle ground in the unlearning literature: it identifies a genuinely new problem space (label domain mismatch) and provides a reasonable engineering solution, but the practical value of the solution depends on how often real-world unlearning requests come with knowledge of the target concept's scope. If practitioners typically know approximately how many classes are involved (which seems plausible for IP/copyright/safety scenarios where the target concept is explicitly defined), the assumption is mild; if not, the method's practical value diminishes. This dependency on the strength of the assumption creates a spectrum of applicability that the paper could address more explicitly.

## Suggestions

- Provide an experiment in the main paper (not just appendix) showing TARF's performance when β is set via automatic threshold detection (e.g., knee-point or fixed quantile) rather than oracle knowledge, to directly demonstrate the method's practical viability without the assumption.
- More explicitly acknowledge the gap between Theorem 3.2's motivational role and the algorithm's empirical design, framing the theorem as providing "guiding intuition" rather than "analytical foundation."

## Score and Decision

**Calibration anchors:**

| Paper | Path | Avg Score | Comparison |
|-------|------|-----------|------------|
| SalUn (class-wise MU, gradient saliency) | gn0mIhQGNM.md | 7.50 | More mature method in established setting; TARF has a more novel problem formulation but less clean method with stronger assumptions |
| Probabilistic unlearning evaluation | 51WraMid8K.md | 8.00 | Much stronger theoretical framework; TARF is below this |
| Streaming forgetting (novel MU setting) | bIoWuzFm6r.md | 4.75 | Similar novelty in problem framing but heuristic approach; TARF has stronger empirical results and clearer formulation |
| UGradSL (flawed MU formulation) | hwXUmwJAq5.md | 3.00 | Fundamentally incorrect formulation and evaluation; TARF is clearly much better |
| H-CAT (novel taxonomy + benchmarking) | icTZCUbtD6.md | 6.20 | Both offer novel taxonomies; TARF additionally provides a method and comprehensive experiments |
| Label transfer (novel mismatch formulation) | ChHx5ORqF0.md | 6.00 | Similar pattern — new problem formulation with practical assumptions; comparable quality |

TARF is clearly above the streaming forgetting paper (4.75) and UGradSL (3.0) due to its stronger empirical results and correct formulation. It is below SalUn (7.50) and the probabilistic evaluation paper (8.0) due to its practical limitations and weaker theoretical grounding. It sits in the range of H-CAT (6.20) and the label transfer paper (6.0) — papers that introduce novel problem formulations with reasonable but imperfect solutions. The novel problem formulation is a genuine contribution, and the empirical results on the new scenarios are strong, but the oracle assumption on target concept scope and the theory-practice gap prevent a higher score.

MY FINAL SCORE: <pineapple>6.0</pineapple>
MY FINAL DECISION: <orange>Accept</orange>