Now I have sufficient calibration. Let me write the final review.

## Summary
This paper resolves the learnability of almost-full parity functions ($k = d - O_d(1)$) by establishing a sharp separation between Rademacher initialization (polynomial-time learnable) and Gaussian initialization (exponentially hard). The paper introduces the Initial Gradient Alignment (GAL) measure as a novel theoretical tool for analyzing initialization sensitivity and provides both theoretical proofs for 2-layer networks with correlation loss and empirical validation on 4-layer MLPs with hinge loss.

## Strengths
- **Resolution of an open case in parity learning**: The paper settles the previously unresolved case of almost-full parities ($k = d - O_d(1)$), proving efficient learnability with Rademacher initialization (Corollary 1 shows $\Omega(d^2)$ neurons suffice for clipped ReLU) versus exponential hardness with Gaussian initialization (Theorem 7 bounds accuracy by $\frac{1}{2} + \exp(-\Omega(d))$). This fills a gap left by prior work that only addressed sparse ($k=O(1)$) or dense (both $k, d-k \to \infty$) parities.

- **Introduction of the Initial Gradient Alignment (GAL) measure**: Definition 2 and Theorem 6 establish GAL as a loss-dependent complexity measure that provides a learnability bound: if GAL is exponentially small at initialization, the network cannot learn better than random guessing in polynomial time. This differs from prior measures like SQ dimension or cross-predictability that apply to function classes rather than single fixed target functions.

- **Empirical characterization of the perturbation threshold**: Figure 1 demonstrates a clear transition from learnable (~0.9 accuracy) to unlearnable (~0.5 accuracy) as the perturbation parameter $\sigma$ increases from 0.0 to 0.4, supporting the theoretical claim that Rademacher success is a special case sensitive to noise. Figure 3 further shows that other discrete initializations (sparse Rademacher, uniform over $\{-2,-1,1,2\}$) also fail, reinforcing the uniqueness of pure Rademacher initialization.

## Weaknesses

### Fatal
None

### Major
- **Incompatible scaling regimes between positive and negative theoretical results**: The positive result (Theorem 4, Section 4) initializes weights as i.i.d. Rademacher ($\pm 1$) without explicit $1/\sqrt{d}$ scaling, while the negative result (Theorem 7, Section 5.2.1) uses Gaussian initialization with variance $1/d$. This confounds the "discrete vs. continuous" distinction with a "large vs. small weight magnitude" distinction. The fair comparison would require proving the positive result for **scaled** Rademacher initialization ($\pm 1/\sqrt{d}$), which the paper does not do theoretically—it only addresses this empirically in Section 6 where variance normalization is applied. This undermines the core theoretical claim that the *distribution type* (discrete vs. continuous) is the causal factor rather than weight magnitude. While the empirical results suggest the phenomenon is real, the theoretical contribution does not currently establish this rigorously.

### Minor
- **Loss function mismatch between theory and experiments**: The theoretical negative results (Theorems 6, 7, 8) and GAL analysis are proven only for correlation loss ($L = -y\hat{y}$), while all empirical validations in Section 6 use hinge loss. The paper explicitly acknowledges this limitation (line 274: "A theoretical understanding of this observation would allow to extend our negative result to the hinge loss") and provides empirical GAL measurements for hinge loss in Figure 2. However, this disconnect means the theoretical mechanism (GAL) is not rigorously demonstrated to explain the empirical phenomenon in the loss function used in practice. This is a known limitation the authors are transparent about, but it weakens the claim that the theory explains the experiments.

- **Architecture depth discrepancy**: The theoretical analysis is restricted to 2-layer networks (Theorems 4, 7, 8), while experiments use 4-layer MLPs. The conclusion explicitly states that strengthening results to deeper architectures is future work (line 292). While 2-layer analysis is standard in learning theory and the empirical results are valuable, the gap between the theoretical setting and experimental validation means the experiments stand as isolated empirical observations rather than direct validations of the theory. Feature learning dynamics in deeper networks can differ significantly from 2-layer dynamics.

### Trivial
None

## Nice-to-Haves
- **Pre-activation distribution visualization**: Including a figure comparing pre-activation distributions for unscaled Rademacher, scaled Rademacher, and scaled Gaussian settings would help clarify whether the learning difference stems from activation distribution shape or variance magnitude.

- **Discussion of GAL behavior during training**: The paper shows GAL remains small along the "junk flow" (training on random labels) in Figure 2, but additional analysis of how GAL evolves during actual training on the parity task could provide deeper insight into why learning fails for Gaussian initialization.

## Removed Points
These points are flagged to be removed, treat them with caution:

- **Removed (Harsh Critic - Scaling regime concern)**: The critic's claim that "Theorem 4 initializes weights as i.i.d. Rademacher ($\pm 1$) without explicit $1/\sqrt{d}$ scaling, leading to pre-activations of magnitude $O(\sqrt{d})$" is **verified as accurate** from the paper text (Section 4, lines 134-135). However, the critic's framing that this "invalidates the claim" is too strong—the empirical results in Section 6 do normalize variance and still show the effect. The weakness is retained as **Major** but softened from "invalidates" to "undermines the core theoretical claim."

- **Removed (Harsh Critic - Loss function mismatch)**: The critic's claim about loss function mismatch is **verified as accurate** (theory uses correlation loss, experiments use hinge loss). However, the critic's statement that this "invalidates the claim that the provided theory explains the observed initialization sensitivity" is too strong since the paper explicitly acknowledges this as a limitation and doesn't claim full explanation. Downgraded to **Minor**.

- **Removed (Harsh Critic - Depth discrepancy)**: The critic's observation about 2-layer theory vs. 4-layer experiments is **verified as accurate**. However, the critic's claim this "creates a gap in reproducibility" is overstated—the paper is reproducible, and 2-layer theory with deeper experiments is common in learning theory. Downgraded to **Minor**.

- **Removed (Strength Finder - Generic strength)**: Any strength claiming "this paper addressed an important problem" or "this paper targeted an interesting question" without specific citation is removed as generic.

- **Removed (Strength Finder - Overclaimed strength)**: The strength claiming GAL "differs from prior complexity measures like Statistical Query dimension or Cross-Predictability... which typically apply to function classes rather than a single fixed target function" is **kept** as it is accurate and specific (SQ dimension does apply to function classes, and the full parity is trivially SQ-learnable as a singleton).

## Novel Insights
The paper's introduction of Initial Gradient Alignment (GAL) as a loss-dependent measure provides a genuinely novel perspective on initialization sensitivity. Unlike prior measures that focus on function classes (SQ dimension, cross-predictability) or input distribution properties (noise sensitivity, globality degree), GAL directly quantifies the alignment between the initialization and a specific target function under a given loss. The key insight—that exponentially small GAL at initialization implies hardness of learning even for singleton function classes that are trivially SQ-learnable—reveals a separation between SQ learning and gradient descent that depends critically on initialization choice. This suggests that for certain target functions, the "right" initialization can bypass SQ lower bounds, while "wrong" initializations impose exponential hardness even when information-theoretically feasible.

## Suggestions
- **Unify theoretical scaling regimes**: Provide a theoretical analysis of the positive result (Theorem 4) under scaled Rademacher initialization ($\pm 1/\sqrt{d}$) to establish that the learnability benefit comes from discreteness rather than weight magnitude. If the result fails under scaled initialization, the paper's core claim should be revised accordingly.

- **Extend GAL analysis to hinge loss**: Either provide theoretical justification for why GAL bounds should extend to hinge loss, or strengthen the empirical analysis showing GAL behavior under hinge loss during actual training (not just at initialization or on random labels).

- **Clarify the scope of claims**: The abstract and introduction should more carefully qualify that the theoretical separation is proven for specific scaling regimes and correlation loss, while the empirical validation uses normalized variance and hinge loss. This would prevent readers from over-interpreting the theoretical results as fully explaining the empirical phenomenon.

## Calibration and Score
I compared this paper against the following calibration anchors:

**High-scoring anchors (avg ≥ 6):**
- xDLE5n3x9Y (6.50, Oral): BBP transitions at initialization—rigorous theory with clear quantitative predictions, experiments support theory, acknowledged finite-N corrections but core claims are sound.
- BAQNrsr987 (6.67, Poster): Quantized NN training complexity—novel parameterized complexity analysis, strong hardness results, tractable cases identified, purely theoretical but contribution is clear.
- utSqpxQHXq (6.00, Poster): Transformer signal propagation theory—unified theory with quantitative predictions, experiments confirm theory, minor presentation issues but core contribution is strong.

**Medium-scoring anchors (avg 4.5-5.5):**
- 2Q0U2rV2Jz (5.50, Poster): Multi-index model learning—novel theoretical analysis with layer-wise training, but idealized setup, no experiments, acknowledged limitations.
- q7Nhu2Fw11 (5.33, Reject): Latent CoT theory—interesting theoretical framework but experiments don't fully validate theoretical claims, theory-experiment disconnect.
- dwNXKkiP51 (5.33, Poster): KAN initialization—empirical study with practical findings, but lacks theoretical justification for why methods work.
- 92d74WdgtG (4.50, Poster): Sinusoidal initialization—theory for specific architecture, questions about scope extension and causal claims.

**Low-scoring anchors (avg ≤ 4):**
- JNUS4L2Wlo (3.00, Reject): Scaling laws with parity—questions about practical relevance, outdated architecture, toy task limitations.
- 5WAGOydkNJ (3.00, Reject): Transformers with RL/SFT for parity—novelty concerns, setup similar to prior work, misleading claims about generality.
- I3spHvRHqo (4.00, Reject): Non-vacuous test error bounds—extensive experiments but clarity issues, novelty concerns, presentation problems.

**Positioning:**
This paper has genuine theoretical contributions (resolving almost-full parity case, introducing GAL measure) and real empirical validation, placing it above low-scoring papers. However, the scaling regime inconsistency between positive and negative theoretical results is a significant weakness that undermines the core claim about what causes the separation. This is similar to q7Nhu2Fw11 (5.33) where theory and experiments don't fully align, or 2Q0U2rV2Jz (5.50) where the theoretical setup is idealized. Unlike high-scoring papers like xDLE5n3x9Y or utSqpxQHXq where the theory is rigorous and experiments directly validate it, this paper's theoretical comparison is not perfectly fair.

The paper is stronger than purely empirical work (dwNXKkiP51) because it does provide theoretical analysis, and stronger than low-scoring papers because it resolves an open case and introduces a novel measure. However, the scaling issue prevents it from reaching the 6+ range where theory and experiments are well-aligned.

**Final Score: 5.0**

This paper is a borderline contribution with genuine theoretical insights but significant weaknesses in the theoretical setup that prevent it from being a clear accept. The empirical results are valuable and the GAL measure is promising, but the core theoretical claim about discrete vs. continuous initialization is not rigorously established due to the scaling regime mismatch.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>