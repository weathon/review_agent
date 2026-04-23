Now let me run calibration searches in parallel.Now I have enough information to write the final consolidated review.

---

## Summary

This paper proposes an identifiability theory for latent variables in nonlinear causal models using only single-domain data. The main contributions are: (1) a reduction procedure that transforms any SCM into an equivalent *powerset bipartite graph SCM* (PBG-SCM), and (2) identifiability theorems (Theorem 1 for the basis model, Theorem 2 for general PBG-SCMs) under three conditions—invertibility, independence, and a novel *minimality* condition on shared latent variables. Experiments on synthetic data ablate each condition using autoencoder variants.

---

## Strengths

- **Novel PBG reduction framework (Section 4, Fig. 1)**: The procedure of clustering exogenous variables by shared observed-descendant sets and forming a powerset bipartite graph is a clean and formally rigorous contribution. It bridges general SCMs and the structured models for which identifiability can be proven, and Proposition 4.1 gives a precise characterization.

- **Formal characterization of minimality and its connection to intrinsic dimension (Proposition 5.1, Corollary 5.1)**: The paper shows that a non-minimal z has "oversized intrinsic dimension" and plunders information from private variables. Corollary 5.1 concretely reveals that setting dim(z̃) = IDim(z) is a sufficient substitute, making an implicit standard assumption explicit and principled.

- **Clean ablation demonstrating necessity of each assumption (Table 1)**: The four conditions (AE vs. AE+CLUB, dz=5 vs. dz=7) systematically isolate the effect of each assumption. AE+CLUB with dz=5 consistently achieves R² > 0.93 across all three datasets, while violating either independence or minimality causes substantial drops. This is clear experimental support for the theory's necessity claims.

- **Constructive proof implemented and empirically validated (Fig. 4b)**: The iterative basis-model approach for Theorem 2's proof is not merely abstract—it is implemented and each intermediate identification step shows high R² scores, confirming the constructive proof is practically viable.

---

## Weaknesses

### Fatal
None.

### Major

- **No empirical comparison to any competing single-domain identifiability method.** The paper's primary competitive claim is that the minimality condition is "much easier to be satisfied in general scenarios" than subspace-span (Kong et al., 2024), additive structure (Lachapelle et al., 2024), or compositionality (Brady et al., 2023). Yet the experiments include *no baseline from these works*. Without comparison on the same datasets—especially in regimes where competing assumptions fail—the practical advantage of minimality over prior methods is entirely unsubstantiated. Kong et al. (2024) directly targets single-domain identifiability and is applicable to the same synthetic setup; the paper provides no evidence of when or why the proposed framework outperforms it.

- **Minimality condition, as implemented, reduces to the oracle assumption of correct latent dimension.** The paper acknowledges this explicitly: Corollary 5.1 shows that minimality is satisfied if and only if dim(z̃) = IDim(z), and Section 6.2 sets dz=5 specifically because it equals the ground truth. The paper also states in its limitations: "the succeeded algorithms in our experiments still need pre-known knowledge of the intrinsic dimension of latent variables." This means the "novel" minimality condition, in practice, is implemented the same way as the default assumption in virtually all prior experimental work (set the dimension to the known ground truth). While the theoretical formalization is genuine, the paper has no algorithm that enforces or verifies minimality without oracle knowledge—limiting the practical advance beyond prior work. The paper is honest about this, but it substantially weakens the claim that minimality is broadly easier to satisfy.

### Minor

- **R² metric measures linear predictability, not full nonlinear equivalence.** The theoretical guarantee (Definitions 3.1–3.2, Theorems 1–2) is equivalence up to an *invertible* (possibly nonlinear) transformation. R² measures linear predictability. The paper follows Kong et al. (2024) in adopting this metric, but provides no argument for why linear R² is an adequate proxy in this setting. In the specific experimental setup (Gaussian latents, MLP generation), variables may happen to be nearly linearly related, but this is not argued. There is a conceptual gap between the theoretical notion and the evaluation metric.

- **Global invertibility of the MLP generation functions is not rigorously established for the experimental datasets.** The paper writes (Section 6.1): "We checked the rank of weight matrices in each linear layer to ensure they are of full rank, therefore any fi is guaranteed to be invertible." Full-rank linear layers with Tanh activations do not guarantee that the *composed* nonlinear function is globally injective. This claim is insufficient, particularly for the Split and Fusion datasets which the paper itself notes are "globally invertible but not locally invertible." The underlying theory requires global invertibility and the experiments should verify it more rigorously (e.g., via empirical inversion tests or by using architectures with provable invertibility such as normalizing flows).

### Trivial
None beyond what the paper itself acknowledges in its limitations.

---

## Nice-to-Haves

- **Comparison to Kong et al. (2024) and Lachapelle et al. (2024) on existing synthetic benchmarks**, particularly in regimes where their assumptions are violated (e.g., non-additive mixing, non-compositional structure). Even a qualitative demonstration of where the competing methods fail while minimality succeeds would substantially strengthen the core claim.

- **An experiment with unknown latent dimension**: Designing a setting where the practitioner does not know the correct dz a priori, and demonstrating either a dimension-selection procedure or showing what goes wrong at varying dz values, would directly test the practical value of Corollary 5.1.

- **Failure mode analysis**: A mechanistic breakdown of *what* goes wrong when minimality is violated—which variables are misidentified and how—would give readers insight beyond the aggregate R² drop.

---

## Removed Points

*These points are flagged to be removed; treat them with caution.*

- **Harsh critic's framing of the minimality-oracle equivalence as a "structural conflation" that undermines novelty (Critical Issue #1, severity overstated)**: The paper explicitly acknowledges in Corollary 5.1 and in its Limitations section that minimality is satisfied by setting the correct dimension. This is not a hidden problem or a deceptive claim; the theoretical contribution of *formally* characterizing and naming this assumption (and showing what goes wrong when it fails) is genuine, even if its practical implementation requires oracle knowledge. The critic overstates this as a "deepest problem" when it is more accurately a limitation the authors themselves clearly articulate.

- **Claim that the "abstract's language about all latent variables is misleading"**: The abstract says "all latent variables in a powerset bipartite graph can be identified," and Theorem 2 precisely establishes this. The framing is accurate. Removed.

- **Request for standard deviation instead of min-max shading in figures**: The paper reports standard error of the mean for Table 1 and min-max range for figures, which is a reasonable choice. This is a minor presentation preference, not a substantive concern. Removed as formatting nitpick.

- **Criticism that the Strength Finder's generic strengths lack specificity**: The strengths as verified all cite specific sections, theorems, tables, and figures. No generic strengths were identified for removal.

---

## Novel Insights

The most conceptually interesting contribution of this paper is the *normalization of an implicit assumption*: virtually all prior identifiability experiments set the latent dimension to match the ground truth, which—as Corollary 5.1 reveals—is precisely what enforces minimality. The paper thus identifies that researchers have been unknowingly satisfying minimality by experimental convention, and that the condition only becomes visible (and potentially problematic) when the latent dimension is misspecified. The PBG reduction is also a clean structural insight: merging exogenous variables by their observed-descendant topology is a natural and formally grounded way to define the finest identifiable grain of any SCM. Together, these contributions offer a coherent new lens on single-domain identifiability, even if the practical impact is currently limited by the need for oracle dimension knowledge.

---

## Suggestions

1. **Add at least one competing baseline** (Kong et al. 2024 would be the most natural) to the existing synthetic experiments. Show a setting where their subspace-span condition is hard to verify or fails, while minimality succeeds.
2. **Conduct an experiment varying dz** from below to above the true dimension and plot how R² changes, to empirically characterize the sensitivity to dimension misspecification and motivate the need for a dimension-estimation procedure.
3. **Replace or supplement R²** with a metric that more directly tests nonlinear equivalence—e.g., mutual information normalized by entropy, or the MCC (mean correlation coefficient) used in ICA evaluation literature.
4. **Clarify the invertibility justification** for the MLP-based generation functions, either by proving the specific construction is globally injective with high probability, or replacing with architectures where this can be guaranteed.

---

## Score and Decision

**Calibration anchors used:**

| Paper | Path | Avg Score | Comparison |
|---|---|---|---|
| Cross-Entropy for ICA (high) | `/human_reviews/hrqNOxpItr.md` | 8.0 (Accept/Oral) | Much stronger: strong theory + real-world validation + clear advantage over prior work; this paper lacks all three. |
| Temporal Causal Repr. (IDOL) | `/human_reviews/2efNHgYRvM.md` | 8.0 (Accept/Oral) | Strong theory + practical algorithm + baselines; paper under review lacks algorithm and baselines. |
| Unification via Invariance | `/human_reviews/lk2Qk5xjeu.md` | 7.0 (Accept/Poster) | Similarly theoretical, but broader conceptual unification; paper under review is more limited in scope. |
| Identifiability with Task Structures | `/human_reviews/kkQSwtx0p3.md` | 5.25 (Reject) | Closest analog: theory-first identifiability paper with no clear proposed algorithm, similar gaps in practical demonstrability; paper under review has slightly more coherent theory and cleaner ablation. |
| Temporal Causal Repr. (non-invertible) | `/human_reviews/5tSLtvkHCh.md` | 5.5 (Reject) | Similar scope—new identifiability theory for causal models—rejected partly for unclear practical advantage over baselines. |
| Causal Flows VAE | `/human_reviews/etnG659OB9.md` | 3.0 (Reject) | Clearly weaker: insufficient empirical validation AND unclear theory; paper under review has cleaner theory. |
| Causal TS (non-novel constraint) | `/human_reviews/KpSNPeRuTf.md` | 2.5 (Reject) | Much weaker: missing related work and non-novel constraint; paper under review is more principled. |

**Positioning**: The paper is solidly above the low anchors (3.0, 2.5) — its theory is genuine and internally coherent. It sits below the high anchors (7.0–8.0) because those papers either connect theory to practical algorithms, compare against baselines, or demonstrate real-world applicability. The closest anchors are the medium-scoring rejected papers (5.25–5.5), which share the same pattern: valid theoretical contribution but insufficient practical demonstration. This paper's ablation is cleaner than kkQSwtx0p3, but its key practical limitation (oracle dimension = minimality) is more fundamental. I place it at **5.0**: the PBG reduction and minimality formalization are real contributions, but the paper cannot empirically establish its claimed advantage over prior single-domain methods, and the core novel condition reduces to an existing experimental convention in practice.

**Originality**: Moderate — PBG framework is novel; minimality formalization is an incremental but genuine step.  
**Importance**: Moderate — single-domain identifiability matters, but practical impact is unclear without working algorithms beyond oracle settings.  
**Claim support**: Partial — claims about necessity of each assumption are well-supported; claims of practical advantage over competing methods are not.  
**Experimental soundness**: Adequate for what is tested; seriously incomplete relative to the paper's competitive claims.  
**Clarity**: Good — the paper is clearly written and theorems are precisely stated.  
**Community value**: Limited in current form; the PBG framework could become valuable if paired with practical algorithms.

MY FINAL SCORE: <pineapple>5.0</pineapple>
MY FINAL DECISION: <orange>Reject</orange>