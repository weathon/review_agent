=== CALIBRATION EXAMPLE 14 ===

# Final Consolidated Review
## Summary
SiDyP introduces a two-stage denoising framework for learning from LLM-generated noisy labels. Stage I fine-tunes a pretrained language classifier (BERT) on the noisy dataset to extract training-dynamics-based embeddings, from which a KNN-based algorithm retrieves true label candidates. Stage II trains a simplex diffusion model to approximate the posterior p(y|ỹ, x), with a dynamic distillation mechanism that iteratively refines candidate weights using diffusion model feedback. The framework is evaluated across four NLP datasets, five LLMs, and three noise types, consistently outperforming prior noisy-label learning baselines.

---

## Strengths

- **Validated core thesis via comparative noise analysis.** The gap between SiDyP and DyGen on LLM noise (5.21%) vs. synthetic noise (3.26%) in Table 4 provides empirical support for the paper's central claim that LLM-generated noise is qualitatively harder — not just asserted, but evidenced by a controlled comparison.
- **Breadth of LLM coverage in robustness check.** Table 3 evaluates across Llama-3.1-70b, Llama-3.1-405b, GPT-4o, and Mixtral-8×22b on SemEval, yielding an average of 4.47% improvement over the second-best baseline (DyGen). This cross-family, cross-scale evaluation is more rigorous than single-LLM evaluation and strengthens the generalization claim.
- **Principled use of simplex space for discrete label diffusion.** Using simplex diffusion (Mahabadi et al., 2024) is a well-motivated architectural choice: diffusing in probability simplex space rather than Gaussian space preserves the categorical structure of labels. The ablation in Table 5 confirms the value of simplex vs. Gaussian diffusion (+8.58%), giving concrete support for this design choice.
- **Conservative experimental setup that avoids inflated results.** The paper uses noisy validation sets for model selection — explicitly noted as a harder evaluation regime compared to methods that use clean validation sets (Section 5.2). This design choice is self-limiting and increases credibility of the reported numbers.

---

## Weaknesses

### Fatal
None identified.

### Major

- **Core architectural components relegated to appendix without explanation in the main body.** Section 2 mentions "co-regularization mechanism (Appendix C)" as one of the five framework components, and Figure 1 explicitly shows M model branches and the final inference formula $p(y|x) = \frac{1}{M}\sum_m \sum_c F_\psi^m(\hat{y}=c|x) \cdot p_\theta^m(y|\hat{y}=c, W)$. However, neither the multi-branch structure, nor the co-regularization objective, nor this inference formula is explained anywhere in the main text. For an ICLR methods paper, leaving architecturally pivotal components unexplained in the main body is a reproducibility gap and undermines full understanding of the method.

- **Potential mathematical inconsistency in Equations (3) and (7) regarding α_t vs ᾱ_t.** In standard DDPM, the closed-form sampling at arbitrary timestep t uses the cumulative product $\bar{\alpha}_t = \prod_{j=1}^t \alpha_j$, yielding $s_t = \sqrt{\bar{\alpha}_t} s_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$. Equation (3) of the paper correctly defines $\bar{\alpha}_t$ just above, but then writes $s_t^y = \sqrt{\alpha_t} s_0^y + \sqrt{1-\alpha_t}\epsilon_t$ using the single-step $\alpha_t$ rather than $\bar{\alpha}_t$. The same pattern appears in Equation (7). If this reflects the actual implementation rather than a PDF rendering artifact inherited from Mahabadi et al. (2024), the forward process would be incorrect. Authors should explicitly confirm whether this follows Mahabadi et al.'s specific formulation or correct the equations.

- **Narrow ablation scope.** Table 5 ablates solely on SemEval with Llama-3-70b zero/few-shot. The dynamic prior's contribution of 1.53% is modest and evaluated only in this single setting. Neither the sensitivity of the certain/uncertain thresholds (λ, γ) nor the noise rate splitting parameter σ is analyzed in any ablation. Given that the clean/noisy split (σ), candidate certainty (λ), and candidate narrowing (γ) all directly determine what the diffusion model trains on, their sensitivity is high-stakes and must be characterized.

### Minor

- **σ estimation undescribed in main text.** Section 3.1 states the dataset is split "by cutting off the top σ percent of training trajectories, where σ is the estimated error rate," but no description of *how* σ is estimated appears in the main body. For LLM-generated labels the noise rate is unknown a priori. If the estimation procedure is only in Appendix D, it should at minimum be summarized in the main text, as it is load-bearing for Stage I.

- **KNN embedding assumption not empirically verified at high noise rates.** The candidate retrieval in Stage I relies on the assumption that textual embeddings "are robust enough to discriminate between clean and corrupted data samples" (Section 3, citing Ortego et al., 2021). However, BERT is fine-tuned on noisy labels in Stage I, potentially corrupting the embedding space. Since the paper uses ~50% noise on SemEval, this assumption requires some empirical validation (e.g., nearest-neighbor purity as a function of noise rate) rather than being taken on faith from a cited reference.

- **Circular dependency risk in warm-up generalization.** Algorithm 2 warms up the diffusion model on "certain" samples only (epochs ≤ α), then evaluates it on "uncertain" samples to update candidate weights. If the certain and uncertain subsets have systematically different data characteristics (e.g., uncertain samples are harder or from underrepresented classes), the feedback signal used for distillation could be biased without characterization.

- **Abstract headline number measured against PLC, not strongest baseline.** The abstract reports "an average of 5.33% and 7.69%" improvement, which compares against the PLC baseline. This is stated correctly ("increase the performance of the BERT classifier"), but readers typically interpret headline gains as margins over the strongest competitor. The paper does report the ~2.05% gain over DyGen in Section 5.3, but the abstract leads with the more impressive PLC-relative figure. The abstract should prominently report the margin over DyGen as the scientifically most informative comparison.

- **Equation (3) noise scaling ε_t ~ N(0, k²I) not justified.** The paper states this distribution "as we convert data into simplex space," but provides no argument for why simplex conversion changes the noise variance to k² rather than simply scaling the signal. If this follows directly from Mahabadi et al. (2024), a citation-based justification would suffice.

### Tiny

- The typo "calibrate classifier's predication" (for "prediction") appears in both the abstract and Section 2 (line 42) and should be corrected.
- The term "False Labels" in Figure 1's description is non-standard; "Noisy Labels" is the established term in the learning-from-noisy-labels literature and should be used consistently.

---

## Nice-to-Haves

- **Oracle upper bound.** Reporting accuracy when training directly on ground-truth labels would contextualize the results — e.g., on SemEval where SiDyP achieves ~64%, it is unclear whether the ceiling is ~75% or ~95%.
- **Computational cost comparison.** The two-stage pipeline (PLC fine-tuning → KNN retrieval → diffusion training) adds overhead relative to PLC or DyGen. A rough training-time comparison would help practitioners assess the cost-benefit tradeoff.
- **LLM self-correction as a reference baseline.** Comparing against simple prompting strategies (e.g., LLM re-labeling with chain-of-thought, or self-consistency voting) would help position SiDyP relative to the most direct and computationally cheap alternative.
- **Dynamic prior convergence visualization.** A plot of candidate weight evolution for mislabeled samples over training epochs (feasible on synthetic noise where ground truth is known) would concretely demonstrate that Algorithm 2 genuinely shifts confidence toward correct labels rather than reinforcing errors.
- **Transition matrix accuracy analysis.** Comparing the learned posterior against the ground-truth transition matrix on synthetic noise would directly verify that the diffusion model is learning noise structure rather than prior fitting.

---

## Removed Points
*These points are flagged as removed; treat with caution.*

- **Critic: Title is too narrow.** The paper's primary contribution is framed around Llama-generated labels; extensions to other LLMs are robustness checks. The title accurately reflects the primary focus. *Removed: scope creep.*
- **Critic: Contributions 1 and 3 are redundant.** Evaluating baselines and conducting experiments are described at different levels of specificity. This is a stylistic nitpick. *Removed: pure formatting/style.*
- **Critic: The LLM noise motivation is appendix-bound.** Section 2 provides substantive motivation: "LLM-generated label noise is more intricate, contextually influenced, and reflective of real-world class relationships," with further detail in Appendix G. The claim is stated and supported; Appendix G elaboration is reasonable. The paper's Table 4 also provides empirical support. *Removed: misreads the paper.*
- **Critic: Statistical significance tests required.** Single-run evaluation is the norm in large-scale NLP benchmarking; five-run mean/std reporting is standard practice at ICLR. Formal t-tests are not an expected requirement in this field. *Removed: demands non-standard rigor.*
- **Critic: 20News few-shot omission is a gap.** The paper explicitly states the context length limitation of Llama-3-70b (8192 tokens) makes few-shot prompting infeasible for document-level 20News. This is a system constraint, not an omission. *Removed: paper addresses this.*
- **Critic: Co-Teaching/JoCoR unfair implementation.** Co-Teaching and JoCoR were designed for image tasks and naturally underperform on NLP. Their underperformance weakens the case for *those* baselines, not the case for SiDyP — any asymmetry favors the baselines. *Removed: unfairness benefits baseline.*
- **Critic: Chen et al. (2023a) missing as baseline.** The paper cites Chen et al. (2023a) in related work as using Gaussian diffusion for noisy label learning. Not including it as a baseline is a choice, but given missing references cannot be confirmed from external sources, this criticism is removed per review policy. *Removed: cannot confirm external claim.*
- **Neutral: "First study" overstatement.** The paper's claim is specifically about *enhancing learning* (i.e., training robust classifiers) under LLM-generated noise — not generating LLM labels or improving LLM annotation. This is meaningfully distinct from X-MLClass (label discovery) or EASE (in-context learning). The "first study" claim appears defensible within the scope. *Removed: mischaracterizes scope.*
- **Critic: Embedding space collapse at >60% noise.** The paper's evaluated noise rates are around 50%; claims of failure above 60% are speculative extrapolation. *Removed: speculative, outside paper's scope.*
- **Critic: English-only limitation.** This is a scope observation, not a flaw. The paper never claims cross-lingual applicability. *Removed: scope creep.*

---

## Novel Insights

The most illuminating finding is methodological: the paper provides indirect but compelling empirical evidence that LLM-generated noise functions qualitatively differently from synthetic noise by showing the SiDyP advantage over DyGen enlarges on LLM noise (5.21%) relative to synthetic noise (3.26%) in controlled conditions (Table 4, same dataset and method). This cross-validates the theoretical motivation of Section 2 without requiring ground-truth noise structure analysis. Additionally, the weight update rule in Algorithm 2 — increasing $w_i^*$ by $\frac{1-w_i^*}{\beta}$ — has an implicit convergence property: if the diffusion model consistently agrees with a candidate, weights asymptotically approach 1 because increments shrink as $w \rightarrow 1$. This informal convergence behavior is worth making explicit, as it partially addresses the confirmation bias concern raised by reviewers.

---

## Suggestions

1. **Add a brief, self-contained description of co-regularization and the multi-branch inference formula** (Section 2 or a new subsection) — even one paragraph — so the method is reproducible without the appendix.
2. **Clarify Equations (3) and (7):** explicitly state whether the use of $\alpha_t$ (rather than $\bar{\alpha}_t$) follows Mahabadi et al. (2024) directly, or correct the equations if this is an error. A one-sentence note resolves this.
3. **Describe σ estimation in the main text**, even briefly, and add a sensitivity analysis (Appendix) showing how performance degrades when the assumed σ deviates from the true noise rate by ±10-20%.
4. **Expand ablations to two or three datasets** (e.g., SemEval + TREC) and include at least one sensitivity plot over λ or γ to demonstrate robustness of the clean/uncertain split to threshold choice.
5. **Revise the abstract's headline figure** to lead with the improvement over DyGen (the strongest baseline) alongside the PLC-relative figure, to present the clearest and most honest characterization of SiDyP's practical advantage.

---

**Axis evaluations:**
- **Novelty:** Moderate-to-high. Individual components (simplex diffusion, KNN label retrieval, training dynamics) exist prior to this work, but the specific orchestration — dynamic candidate distillation via diffusion feedback — and the application to LLM-generated noise is a fresh contribution.
- **Technical soundness:** Adequate, with concerns. The core framework is coherent, but the $\alpha_t$ vs $\bar{\alpha}_t$ ambiguity and the undescribed co-regularization mechanism need resolution.
- **Empirical support:** Strong. Results are consistent across 4 datasets, 5 LLMs, 3 noise types, and 5 seeds. The cross-noise comparison (Table 4) is particularly compelling.
- **Significance:** High for the subfield. LLM-generated label noise is an increasingly common real-world scenario, and a principled denoising approach outperforming specialized baselines by meaningful margins is practically relevant.
- **Clarity:** Adequate but uneven. Main method sections are reasonably clear; however, deferring co-regularization, inference formula, and σ estimation to appendices creates gaps in the main-body presentation that weaken the paper's self-containedness.

# Actual Human Scores
Individual reviewer scores: [5.0, 5.0, 5.0, 5.0]
Average score: 5.0
Binary outcome: Reject
