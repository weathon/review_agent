Now I have sufficient calibration context. Let me synthesize the final review.

**Calibration Summary:**
- ARPrtuzAnQ (hardness of learning under symmetries): Scores 8,6,8, average ~7.3, spotlight. Clean SQ/CSQ hardness results, well-matched theory.
- HgOJlxzB16 (SGD for XOR): Scores 8,8,8,6, average ~7.5, spotlight. Tight, complete analysis for a specific problem.
- LEuuOaZNOT (Boolean functions, random features): Scores 3,3,3, rejected. Trains only last layer, poor presentation.
- QibPzdVrRu (early neuron alignment): Scores 5,8,5,8, average ~6.5, poster. Interesting alignment results but with restrictive assumptions.
- GqI4fTVUXC (theory-practice disconnect): Scores 5,5,8,6, average ~6, rejected. Relevant empirical contributions but overclaims.
- fAGEAEQvRr (GD for matrix factorization): Scores 6,5,5,6, average ~5.5, rejected. Novel SNR analysis but incomplete and confusing presentation.

The current paper is stronger than the rejected papers in the 3-5 range (LEuuOaZNOT, GqI4fTVUXC) because it has genuine technical contributions (GAL measure, exponential alignment bounds, clean positive construction). It's somewhat weaker than the 7-8 range papers (ARPrtuzAnQ, HgOJlxzB16) because those have more complete and self-consistent results. It's comparable to QibPzdVrRu (interesting alignment results, restrictive assumptions, but novel technique) but with larger gaps between claims and proofs. I'd place this at around 5.5-6.

Now let me write the review.</think>## Summary

This paper studies how the choice of initialization affects the learnability of high-degree parity functions (almost-full parities, including full parity) by gradient descent on two-layer ReLU networks. It shows that Rademacher (±1) initialization enables efficient strong learning, while Gaussian initialization or sufficiently perturbed Rademacher initialization (constant σ) prevents learning in polynomial time. The central technical contribution is the Gradient Alignment (GAL) measure, which quantifies the alignment between initial gradients and the target, and a coupling argument with a "junk flow" to prove that small GAL implies failure of noisy-GD with correlation loss.

## Strengths

- **Novel GAL measure and junk-flow coupling (Theorem 6).** The idea of coupling training dynamics on true labels with dynamics on random labels, using GAL as a control variable, is genuinely novel. This yields a general-purpose negative result applicable to any architecture with a linear output layer and any balanced target, beyond specific parity constructions and beyond SQ-style arguments. Notably, this circumvents the fundamental limitation of SQ arguments for singleton function classes like the full parity.

- **Exponentially small GAL bounds for Gaussian/perturbed initializations (Propositions 1–2).** These are clean, quantitative results showing that GAL under Gaussian initialization scales as exp(−Ω(d)) for high-degree parities. Combined with Theorem 6, they yield rigorous exponential-time hardness results (Theorems 7–8) that are specific to the initialization distribution rather than to the size of the function class.

- **Positive result extending weak to strong learning.** The paper extends the prior weak-learning result of Abbe & Boix-Adsera (2022) to strong (perfect accuracy) learning of almost-full parities under Rademacher initialization, with a clear linear-separability argument. The extension to perturbed Rademacher with σ = O(1/d) and hinge loss (Section 4.2) is a meaningful strengthening.

- **Conceptually important separation from SQ.** The full parity is trivially SQ-learnable (as a singleton class), yet the paper shows it is hard for noisy-GD with standard initializations. This is a clean conceptual distinction that cannot be obtained via cross-predictability or orbit-based arguments.

- **Well-designed and informative experiments.** The experiments (Section 6) systematically explore perturbed Rademacher, Gaussian, uniform mixture, sparse, and discrete initializations, with both online and offline settings. Figure 2's numerical GAL computations provide evidence that the GAL mechanism operates beyond the correlation-loss setting.

## Weaknesses

### Major:

- **Mismatch between positive and negative result settings undermines the "separation" narrative.** The positive result for correlation loss (Theorems 4–5) trains only the output layer with frozen Rademacher-hidden weights—essentially a random features model—while the negative results (Theorems 7–8) allow training all parameters under Gaussian initialization. The hinge-loss positive result (Section 4.2) does train both layers but is deferred to the appendix and restricted to full parity. There is no single (architecture, loss, training protocol) setting where both provable learning and provable non-learning are established. This means the central "Rademacher vs. Gaussian" separation is not demonstrated in an apples-to-apples comparison. It remains plausible that with Gaussian initialization and hinge loss (the most practically relevant setting), full parity could be learnable, which would contradict the paper's narrative.

- **Negative results are limited to correlation loss.** Theorem 6 and all its corollaries (including Theorems 7–8) apply only to the correlation loss L(y,ŷ) = −yŷ. This is a notably non-standard loss function; modern practice overwhelmingly uses cross-entropy or hinge loss. The paper acknowledges this (Remark 3) and provides only empirical GAL observations for hinge loss (Figure 2), but the conceptual framing of GAL as a "novel measure of potential broader relevance" (abstract) and the conclusion's broad claims about "noisy-GD on neural networks" significantly overstate what is proved. The gap between what the theory covers and what practitioners actually use is substantial.

- **Large gap between positive and negative σ regimes.** The positive result for perturbed Rademacher holds for σ = O(1/d) (Theorem 1), while the negative result requires σ ≥ σ₀ for some constant σ₀ (Theorem 8). The entire intermediate regime—including σ = Θ(1/√d), which would be natural for normalized weight initializations—remains completely uncharacterized. Given that the paper's title emphasizes the "crucial role of the initialization," leaving the key threshold question unresolved significantly limits the completeness of the picture. The paper acknowledges this but presents it as future work rather than a limitation on the current contribution.

### Minor:

- **Incomplete formal proof for the perturbed Rademacher negative result.** Section 5.2.2 explicitly states that the output-layer GAL bound for perturbed Rademacher is "omitted from this version of the paper." Without this, Theorem 8's claim is not fully proved, and the claimed extension to σ-perturbed Rademacher initialization remains conditional.

- **Informal Theorem 1 is not fully formalized in the main text.** The abstract and Theorem 1 claim a result for almost-full parities (|S| = d − O(1)) under σ-perturbed Rademacher with σ = O(1/d). The formal results in Section 4 cover either unperturbed Rademacher with correlation loss (Corollary 2) or full parity with hinge loss and σ-perturbation (Section 4.2, appendix). The precise formal statement for the intersection of these settings is not present in the main body.

- **Theory–experiment architecture mismatch.** All theoretical results are for two-layer networks, while experiments use a 4-layer MLP. All main theorems use correlation loss, while experiments primarily use hinge loss. These mismatches make it difficult to assess whether the proven mechanisms are actually responsible for the observed empirical behavior.

- **Width requirements are large.** Corollary 1 requires Ω(d⁴) hidden neurons for ReLU with correlation loss (though only Ω(d²) for clipped ReLU). Corollary 2 requires Ω(d^{4+2⌈a/2⌉}) neurons for almost-full parity. While still polynomial, these bounds are quite loose relative to the empirical regime (512–64 neurons for d=50).

### Trivial:

- The definition of "perturbed initialization" (Definition 1) normalizes by √Var(A_p), making its interpretation slightly non-intuitive when A has non-unit variance. This is a minor presentational issue.

## Nice-to-Haves

- Analyzing even a single point in the intermediate σ regime (e.g., σ = Θ(1/√d)) would dramatically strengthen the completeness of the story—even a partial result or a well-motivated conjecture with scaling experiments across dimensions.
- Extending Theorem 6 to hinge loss, even partially (e.g., for single-neuron networks or under additional structural assumptions), would significantly broaden the impact of the GAL framework.
- Running experiments with the specific theoretical setting (2-layer network, correlation loss, output-layer-only training) would allow direct empirical validation of Theorems 4–5.

## Removed Points

- **"Rademacher is not a standard initialization, so the practical relevance is limited."** The paper is explicitly a theoretical contribution studying the role of initialization structure. Rademacher initialization is a natural theoretical baseline, and the paper's value lies in understanding *why* discrete structure matters, not in proposing a practical recipe. The paper itself notes "the Rademacher initialization is in some sense a special case."

- **"The positive result only trains the output layer, so it's just a random features model."** While true for the correlation-loss result, this is exactly the point: the paper shows that the Rademacher embedding creates linear separability, which is a meaningful structural result. The hinge-loss result (Section 4.2) does train both layers. This is a valid observation but not a fatal flaw—the random-features mechanism is the *reason* training succeeds, not a weakness of the analysis.

- **"The noisy-GD model doesn't cover standard SGD."** The paper extends to large-batch SGD in Remark 1, and the noise level τ can be very small (exp(−o(d))). While the result doesn't cover noiseless GD, this is a standard modeling choice in the learning theory literature on parity (cf. Abbe & Sandon 2020), and the paper is transparent about the assumption.

- **"Specific bias choices are needed."** Corollary 1 uses specific bias values, but the paper explicitly notes "the precise values are not crucial except for 'unlucky' choices where Δ can become too small" and that "learning should hold for random biases for most reasonable distributions." This is a standard technical convenience, not a fundamental restriction.

## Novel Insights

The most interesting conceptual insight is that initialization can create a computational barrier *for a single fixed function*, bypassing SQ-type arguments that rely on function class cardinality. The GAL measure and its coupling with the "junk flow" provide a new theoretical handle for understanding when initializations enable or prevent learning, distinct from prior complexity measures (information exponent, NTK alignment). This measure is loss-dependent and directly tied to training dynamics, making it potentially more informative than class-level measures like cross-predictability. However, its current applicability is limited to correlation loss, and whether this conceptual framework genuinely extends beyond parities remains an open question.

## Suggestions

- Reduce the scope of claims in the abstract and conclusion to match what is actually proved. Specifically, qualify that the negative result is for correlation loss and that the "separation" involves different training protocols on the positive and negative sides.
- Complete the omitted output-layer GAL bound for perturbed Rademacher initialization. This is the most critical gap for making Theorem 8 fully rigorous.
- Add scaling experiments across dimensions (d ∈ {20, 50, 100, 200}) with σ = C/d for various C, to empirically test the positive prediction and characterize where the transition occurs. This would provide evidence for or against a sharp threshold.

## Score and Decision

**Calibration:** Compared to ARPrtuzAnQ (hardness of learning under symmetries, scores 8/6/8, accepted as spotlight): that paper proved clean, unified SQ/CSQ hardness results with less gap between claims and proofs. Our paper has genuine technical novelty (GAL, junk-flow coupling) but larger gaps between the narrative and what's rigorously established. Compared to HgOJlxzB16 (SGD for XOR, scores 8/8/8/6, spotlight): that paper gave a complete, tight analysis of a specific problem (XOR) with standard training; our paper aims at a broader scope but with more fragmented results. Compared to QibPzdVrRu (early neuron alignment, scores 5/8/5/8, poster): similar novelty in alignment-based measures, similarly restrictive assumptions; our paper has a stronger conceptual contribution but also larger overclaims. Compared to fAGEAEQvRr (GD for matrix factorization, scores 6/5/5/6, rejected): our paper is clearly stronger with more rigorous technical content and a clearer conceptual contribution.

The paper makes genuine contributions—the GAL framework, the junk-flow coupling, and the clean separation between initialization distributions for parity learning—but the mismatch between positive/negative settings and the correlation-loss limitation weaken the central narrative. The claims about "noisy-GD on standard neural networks" in the abstract and conclusion significantly overstate what is proved. This is a solid but incomplete contribution to learning theory.

MY FINAL SCORE: <pineapple>5.5</pineapple>
MY FINAL DECISION: <orange>Reject</orange>