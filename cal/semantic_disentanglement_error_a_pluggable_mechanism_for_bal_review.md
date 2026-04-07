=== CALIBRATION EXAMPLE 4 ===

# Harsh Critic Review
## Section-by-Section Critical Review

---

### Title & Abstract

The title refers to "Semantic Disentanglement Error," but within the body of the paper this quantity is called by at least **three different names**: "Semantic Separability Error" (§3.2), "Semantic Decomposition Error" (§4.3 heading), and "Semantic Disentanglement Error" (title only). This is not a minor terminological slip—SDE is the paper's central concept, and the inconsistency makes it genuinely difficult to follow the technical thread. The abstract claims "consistent gains in forecasting accuracy and representational robustness," but the actual experimental results (discussed below) show the proposed method is **substantially worse** than baselines on several datasets at longer horizons. The abstract is therefore misleading.

---

### Introduction & Motivation

The motivation for addressing semantic imbalance in contrastive time-series learning is reasonable, and the three stated limitations (no inductive bias for decomposition, time-domain-only objectives, isotropic embedding collapse) are cogent. However, the second and third limitations are already explicitly addressed by CoST—the paper the authors build directly on—without adequate acknowledgment. The framing gives the impression these are novel discoveries when they are partly the premise of prior work.

The three stated contributions lack precision. Contribution 2 ("systematic ablations") is overstated; what is presented is a single failed experiment (SDE regularization on TS2Vec) and then a switch to a different framework (CoST + APW). Contribution 3 promises a "semantic rebalancing framework" but delivers a scalar re-weighting of two existing loss terms.

---

### Related Work

The related work is thin and largely a literature summary. Critically, the discussion of CoST (§2.3) does not clearly articulate what this paper adds *beyond* CoST—a fundamental requirement given that the proposed method is literally a plugin for CoST. There is no discussion of dynamic or adaptive loss weighting methods in self-supervised learning, which is the closest prior art to the APW mechanism. References such as GradNorm or uncertainty-based multi-task loss balancing are conspicuously absent.

---

### Method (§3)

**SDE Definition (§3.2):** The formula

> SDE_{a,b} = 1 − cos(v(a+b) − v(b), v(a))

measures whether component **a** is linearly recoverable from the composite embedding via subtraction of the component-**b** embedding. This implicitly assumes that the encoder is *approximately linear* with respect to signal superposition—a strong and unjustified assumption for deep nonlinear encoders. No theoretical motivation or empirical validation of this linearity assumption is provided. The analogy to word2vec vector arithmetic is invoked (citing Baevski et al. 2020—an odd citation choice; this is usually attributed to Mikolov et al. 2013, which is also in the references), but word embeddings are trained with objectives that specifically encourage linear arithmetic; contrastive time-series encoders are not.

**Asymmetry Factor (§3.3):** The asymmetry factor Δ = SDE_{period,trend} − SDE_{trend,period} is reasonable as a diagnostic. However, the paper does not discuss whether Δ can be negative in practice for real (non-synthetic) datasets, or how the weighting behaves when Δ fluctuates during training—a critical omission for understanding stability.

**Loss Reweighting (§3.4):** The adaptive objective

> L = (1 + γ·Δ) L_season + (1 + γ'·(−Δ)) L_trend

is presented cleanly, but several issues arise:
1. **Sign behavior**: If Δ < 0 (trend underrepresented), then (1 + γ·Δ) < 1, potentially making the seasonal loss weight *negative* for large |Δ| and small γ. No clipping or constraint is mentioned.
2. **Hyperparameter sensitivity**: γ and γ' are introduced but never ablated. Their values in experiments are not reported anywhere in the paper.
3. **Computational cost**: Computing SDE at each training step requires three encoder forward passes (for x, a, and b separately). This is a 3× overhead on the encoding step, which is not acknowledged or measured.
4. **Gradient issues**: When SDE is used as a *weighting scalar* rather than as a direct loss, its gradient does not flow through the contrastive losses (only through the weighting coefficients). The paper does not clarify whether Δ is treated as a stop-gradient constant or as a differentiable scalar, which changes the optimization semantics substantially.

**MLP Fusion Layer (§4.4.2):** A composite embedding MLP g_ϕ is introduced in Section 4.4.2 without any appearance in Section 3 (the method section). This is a structural addition to CoST that is not part of the paper's stated contributions and is not ablated independently. It is unclear whether performance gains come from APW or from the additional MLP capacity.

---

### Experiments & Results

**Critically missing results (Table 3):** Table 3 is presented with a header but its body is empty in the manuscript. The numerical results appear only as an orphaned data block following the Conclusion section, formatting-separated from the table. While this may be a PDF extraction artifact, even reading the detached numbers reveals that:

- On **Electricity** at horizons 168, 336, and 720, CoST+APW has MSE of 0.566, 0.677, and 1.010—substantially *worse* than vanilla CoST (0.425, 0.576, 0.911) and TS2Vec (0.429, 0.565, 0.863). TNC outperforms the proposed method on this dataset at all three longer horizons.
- On **Weather** at horizons 336 and 720, CoST+APW (0.266, 0.299) is again substantially worse than all three baselines (TS2Vec: 0.231, 0.233; TNC: 0.215, 0.219).

The conclusion that the method provides "consistently lower SDE values and superior forecasting accuracy compared with both TS2Vec and vanilla CoST baselines" is **directly contradicted** by these numbers. The authors do not discuss these failures at all.

**SDE values not reported:** Table 3 is described as reporting "the SDE metrics and forecasting performance," but the table contains only MSE and MAE. The paper's diagnostic metric—the one central to its story—is never reported at test time on real datasets.

**Synthetic analysis (Table 1):** The SDE analysis in §4.2 is conducted only on TS2Vec, not on CoST (the framework being modified). We do not know whether CoST already mitigates the imbalance that SDE regularization would address, which raises the question of whether the problem is still present in the CoST baseline at all.

**No statistical significance testing:** Results are reported as single numbers across 9 dataset-horizon combinations per dataset. No confidence intervals, variance across seeds, or significance tests are provided.

**No ablation of the MLP fusion layer:** The method has two new components (APW and the MLP g_ϕ). Neither is ablated independently against the vanilla CoST baseline.

**Forecasting-only evaluation:** Despite claiming broad applicability to "anomaly detection and classification," all experiments are forecasting tasks only.

---

### Writing & Clarity

Beyond the triple naming of SDE, the narrative structure is confusing. The paper presents SDE regularization (§4.3) as a failed attempt, then pivots to APW (§4.4), but does not clearly separate these as two distinct experimental phases. A reader could reasonably wonder whether the final system still includes SDE regularization or only APW. The method section (§3) describes the overall approach while Section 4 re-introduces components (including the MLP) not mentioned in §3, creating a disjoint structure.

---

### Limitations & Broader Impact

The paper has no dedicated limitations section. The acknowledged limitations in the Conclusion are framed as future work directions rather than honest acknowledgments of failure. The complete degradation on Electricity and Weather at longer horizons is not discussed at all. The assumption of encoder linearity underpinning SDE is never flagged. The paper also does not discuss whether the approach generalizes beyond the two-component (trend + periodicity) decomposition setting.

---

## Overall Assessment

This paper addresses a legitimate problem—semantic imbalance in contrastive time-series learning—but the execution falls substantially short of ICLR acceptance standards. The central contribution, SDE-based adaptive loss weighting, rests on an unjustified linearity assumption about nonlinear encoders. The experimental results do not support the paper's conclusion of "consistent gains": the proposed method is markedly worse than baselines on Electricity and Weather at longer horizons, and this failure is neither discussed nor acknowledged. The paper's core metric (SDE) is never reported on real datasets. Critical ablations (MLP vs. APW, hyperparameter sensitivity) are missing. The terminological inconsistency in naming the central concept is symptomatic of a paper that needs substantially more development. The idea of using representational asymmetry as a dynamic loss-weighting signal has intuitive appeal, and the framing as a plug-in module is practically useful in principle, but the contribution as presented is too narrow (a scalar reweighting of two existing CoST losses), theoretically undergrounded, and empirically unpersuasive. A major revision addressing the theoretical justification of SDE, the experimental failures, and the missing ablations would be required before this work could be considered for acceptance.

# Neutral Reviewer
## Balanced Review

### Summary
This paper addresses the problem of semantic imbalance in contrastive time-series representation learning, where dominant components (e.g., trends) overshadow minor components (e.g., seasonality). The authors propose a metric called Semantic Separability Error (SDE) to quantify this imbalance and integrate it into an Asymmetric Perceptual Weighting (APW) strategy applied to the CoST framework. The method aims to dynamically reweight contrastive losses to ensure balanced representation of trend and periodic components, demonstrating empirical improvements on benchmark forecasting tasks.

### Strengths
1.  **Clear Problem Diagnosis:** The paper effectively identifies and quantifies a valid and often overlooked issue: the suppression of weak semantic signals (like periodicity) by dominant ones (like trends) in models like TS2Vec. Table 1 provides compelling evidence of this asymmetry in the baseline model.
2.  **Simple and Pluggable Mechanism:** The proposed Asymmetric Perceptual Weighting (APW) is lightweight and does not require re-architecting the encoder. It can be integrated into existing frameworks (specifically CoST), which is valuable for community adoption.
3.  **Rigorous Ablation Studies:** The authors conduct useful ablations, such as testing direct SDE regularization (Section 4.3), and correctly identify that a naive regularization approach fails while the weighting mechanism succeeds. This demonstrates thoughtful experimental design.
4.  **Interpretable Metric:** SDE offers a quantitative lens to evaluate "semantic" quality in embeddings, moving beyond simple reconstruction or contrastive loss values to measure component recoverability.

### Weaknesses
1.  **Inconsistent Empirical Performance:** The Abstract claims "consistent gains... in forecasting accuracy," but the experimental results in Table 3 show mixed performance. Specifically, on the Electricity dataset at longer horizons (e.g., 168, 336), the baseline CoST outperforms the proposed CoST+APW. This lack of robustness across all horizons weakens the claim of a universal improvement.
2.  **Limited Baseline Comparison:** The evaluation compares primarily against older or foundational contrastive methods (TS2Vec, TNC) and CoST. It lacks comparison with more recent state-of-the-art forecasting models (e.g., PatchTST, iTransformer, DLinear) which often outperform purely contrastive baselines on time-series tasks.
3.  **Terminology Inconsistency:** There are naming inconsistencies throughout the manuscript that reduce clarity. The title and Abstract use "Semantic Disentanglement Error," while Section 3.2 defines "Semantic Separability Error," and Section 4.2 refers to "Semantic Decomposition Error." This confusion suggests a lack of polish in the paper's final draft.
4.  **Dependency on CoST Architecture:** The method relies heavily on CoST's specific decomposition into separate trend and seasonal encoders. It is not clear how the APW mechanism performs if the decomposition is less explicit or learned implicitly, limiting the generalizability of the contribution.

### Novelty & Significance
**Novelty:** The paper offers moderate novelty. While the observation of semantic imbalance is insightful, the solution—adaptive loss weighting based on an imbalance metric—is conceptually similar to techniques used in class imbalance (e.g., Focal Loss). The primary novelty lies in defining the SDE metric specifically for time-series components and applying it to contrastive objectives, rather than a fundamentally new learning paradigm.

**Significance:** Addressing semantic imbalance is significant for the time-series community, as robust representations require capturing both trend and seasonality. However, the practical significance is tempered by the marginal performance gains in many settings and the reliance on heavy decomposition. For ICLR acceptance, the method would need to demonstrate clearer theoretical grounding or more consistent SOTA-level performance gains to justify the incremental nature of the improvement.

### Suggestions for Improvement
1.  **Clarify Terminology:** Standardize the terminology in the title, abstract, and main text. Select one term (e.g., Semantic Disentanglement Error) and use it consistently.
2.  **Analyze Inconsistencies:** Provide a deeper analysis of why CoST+APW underperforms CoST on specific horizons in the Electricity dataset. Is it overfitting the imbalance metric? Is the gamma hyperparameter sensitive? A sensitivity analysis plot for $\gamma$ and $\gamma'$ is necessary.
3.  **Expand Baselines:** To position the work correctly within the ICLR scope, compare against more recent strong baselines (e.g., Transformer-based forecasting models) to clarify if the benefit is specific to self-supervised learning or general representation quality.
4.  **Strengthen Theoretical Motivation:** The connection between minimizing SDE asymmetry and minimizing forecasting error is currently empirical. A brief discussion or proof sketch on why balancing component recoverability leads to generalization would strengthen the paper's theoretical contribution.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. Demonstrate semantic imbalance specifically in the **CoST baseline** before applying APW. Table 1 only analyzes TS2Vec, so the motivation for modifying CoST remains unsupported without evidence that CoST suffers from the same suppression issue.
2. Compare against recent SOTA time-series foundation models (e.g., Moment, TimesNet, or 2023-2024 SSL methods). Relying solely on TS2Vec and CoST (2021-2022) is insufficient for ICLR's current acceptance bar regarding empirical competitiveness.
3. Ablate the dynamic weighting mechanism against a **static weighting** scheme. It is unclear if the computational overhead of calculating $\Delta$ online is necessary versus simply upweighting the seasonal loss by a fixed hyperparameter.

### Deeper Analysis Needed (top 3-5 only)
1. Correlate SDE scores with downstream forecasting error across multiple seeds. Without showing that lower SDE consistently predicts better MSE/MAE, the metric remains an unverified proxy rather than a meaningful diagnostic.
2. Analyze the stability of the feedback loop where $\Delta$ (derived from encoder outputs) weights the encoder's own loss. Provide training curves of the weighting factors to rule out oscillation, divergence, or gradient instability.
3. Validate the linearity assumption ($v(a+b) - v(b) \approx v(a)$) in the latent space. Deep non-linear encoders rarely satisfy vector arithmetic properties without explicit constraints, yet this assumption underpins the entire SDE definition.

### Visualizations & Case Studies
1. Provide t-SNE projections of trend versus seasonal embeddings to visually confirm increased disentanglement in CoST+APW compared to vanilla CoST.
2. Show case studies of reconstructed signals where vanilla CoST misses periodic peaks that CoST+APW successfully captures to illustrate the "semantic rebalancing" claim.
3. Plot the evolution of the asymmetry factor $\Delta$ over training epochs to demonstrate convergence rather than noisy fluctuation during optimization.

### Obvious Next Steps
1. Verify the "plug-and-play" claim by applying APW to a different architecture (e.g., TS2Vec or a Transformer-based encoder) beyond CoST.
2. Include standard deviations over multiple random seeds for all table results, as single-run metrics are unreliable for claiming consistent gains.
3. Evaluate on a domain with weak or no seasonality to ensure the method does not degrade performance when the semantic imbalance problem is not present.

# Final Consolidated Review
## Summary

This paper identifies semantic imbalance in contrastive time-series representation learning, where dominant components (trend) suppress weaker ones (seasonality). The authors propose Semantic Disentanglement Error (SDE), a metric quantifying component recoverability from composite embeddings, and introduce Asymmetric Perceptual Weighting (APW) to dynamically reweight seasonal and trend contrastive losses within the CoST framework.

## Strengths

- **Clear problem diagnosis with empirical evidence:** Table 1 provides compelling synthetic experiments demonstrating that TS2Vec exhibits systematic semantic imbalance—when trend dominates, periodicity is underrepresented (Δ approaches +1.0), and vice versa. This quantifies an important phenomenon that was previously only anecdotally observed.
- **Honest reporting of failed approach:** The authors explicitly show that direct SDE regularization (Table 2) does not improve performance, demonstrating scientific integrity by documenting negative results rather than hiding them. The explanation that SDE as a diagnostic metric fails to provide constructive optimization gradients is reasonable.
- **Lightweight, pluggable design:** The APW mechanism adds only a scalar reweighting to existing CoST losses, requiring no architectural changes. This design choice makes the method immediately adoptable by practitioners using CoST.
- **Conceptually interpretable metric:** SDE offers a quantitative lens for evaluating semantic quality beyond reconstruction losses, measuring whether component embeddings satisfy linear arithmetic properties analogous to word2vec.

## Weaknesses

- **Terminology inconsistency undermines clarity:** The paper's central concept is called "Semantic Disentanglement Error" in the title/abstract, "Semantic Separability Error" in §3.2, and "Semantic Decomposition Error" in §4.3. This is not a minor polish issue—it creates genuine confusion about whether these are different quantities or the same concept. Readers must infer they are identical.

- **Empirical claims exceed actual results:** The abstract states the method provides "consistent gains in forecasting accuracy," but the results contradict this on Electricity and Weather at longer horizons. Specifically:
  - Electricity horizon 720: CoST+APW MSE = 1.010 vs CoST = 0.911 vs TS2Vec = 0.863
  - Weather horizon 720: CoST+APW MSE = 0.299 vs TS2Vec = 0.233
  The method is substantially *worse* than baselines in these settings, yet the conclusion claims "consistently lower SDE values and superior forecasting accuracy."

- **Unjustified linearity assumption:** SDE's definition (SDE_{a,b} = 1 − cos(v(a+b) − v(b), v(a))) assumes embeddings approximately satisfy v(a+b) − v(b) ≈ v(a). Deep nonlinear encoders do not inherently preserve such linearity. The word2vec analogy is invoked, but word embeddings are explicitly trained with objectives encouraging linear structure—contrastive time-series encoders are not. No validation of this assumption is provided.

- **MLP fusion layer introduced post-hoc without ablation:** Section 4.4.2 introduces an MLP g_ϕ for computing composite embeddings, but this component never appears in Section 3 (Methods). The method section describes only APW, yet experiments include both APW and this MLP. There is no independent ablation separating the contribution of APW from the added MLP capacity.

- **Missing SDE evaluation on real datasets:** The paper's diagnostic metric—the core conceptual contribution—is computed only on synthetic data (Table 1). Table 3 is described as reporting "SDE metrics and forecasting performance" but contains only MSE/MAE. The reader cannot verify whether APW actually improves semantic balance on real-world time series.

- **Hyperparameters and implementation details absent:** The scaling parameters γ and γ' are introduced in §3.4 but their values are never reported. The critic's concern about potential negative weights when Δ < 0 and |γ·Δ| > 1 is unaddressed. No clipping mechanism or constraints are mentioned.

- **No demonstration that CoST suffers from semantic imbalance:** Table 1 analyzes only TS2Vec. Since the proposed method modifies CoST, demonstrating that CoST also exhibits semantic imbalance would strengthen motivation. Without this, the paper shows TS2Vec has a problem, proposes a fix for CoST, but never proves CoST has the same problem.

## Nice-to-Haves

- **Compare static vs. dynamic weighting:** A simple ablation comparing APW's online Δ computation against fixed seasonal-trend loss weights would clarify whether the computational overhead of computing SDE at each training step is necessary.

- **Training dynamics visualization:** Plots of Δ evolution across training epochs would demonstrate whether the feedback loop converges stably or exhibits oscillation.

- **Validation of linearity assumption:** Even a simple empirical test—showing that v(a+b) − v(b) correlates with v(a) for learned embeddings—would strengthen the theoretical foundation.

- **Apply APW to additional architectures:** Testing on TS2Vec or Transformer-based encoders would validate the "pluggable" claim beyond CoST's specific decomposition architecture.

## Removed Points

These points are flagged to be removed, treat them with caution:

- **Comparison with recent SOTA forecasting models (e.g., PatchTST, iTransformer):** The paper focuses on improving contrastive self-supervised representation learning, not on beating all supervised forecasting models. Evaluating against other SSL methods (TS2Vec, TNC, CoST) is appropriate for the paper's stated scope.

- **No statistical significance testing / confidence intervals:** Single-run evaluation is standard practice in time-series representation learning benchmarks. While variance reporting would strengthen the paper, its absence is not a fatal flaw.

- **Forecasting-only evaluation despite claiming broader applicability:** The paper's core contribution is a method for contrastive learning; forecasting is a reasonable downstream task to demonstrate effectiveness. Expanding to anomaly detection or classification would be beneficial but is not required for a methods paper.

- **No discussion of negative results on Electricity/Weather:** This is already covered above under "Empirical claims exceed actual results." The concern is valid and kept, but the characterization that authors "do not discuss these failures at all" is an overstatement—the paper structure simply buries this in results rather than discussing it explicitly.

## Novel Insights

The observation that semantic imbalance can be *directional*—that SDE_{trend,period} ≠ SDE_{period,trend} systematically—reveals something important about representation learning: encoders may preserve dominant components linearly while non-dominant ones become entangled. This asymmetry is more subtle than standard "imbalance" and suggests that representation quality depends not just on what information is captured, but on *how* it is geometrically structured in latent space. The failed SDE regularization experiment further suggests that diagnostic metrics do not automatically yield constructive training signals—a distinction between what we can measure and what we can optimize.

## Suggestions

1. **Standardize terminology:** Choose one name (suggest "Semantic Separability Error" for precision about the specific formula, or "Semantic Disentanglement Error" for broader concept) and use it consistently throughout.

2. **Add a CoST baseline SDE analysis:** Extend Table 1 to show that CoST also exhibits semantic imbalance, or explain why TS2Vec findings transfer to CoST.

3. **Ablate the MLP independently:** Report results for: (a) CoST + APW only, no MLP; (b) CoST + MLP only, no APW; (c) CoST + APW + MLP. This isolates the contribution of each component.

4. **Report SDE values on real datasets:** Add a column to Table 3 (or a separate table) showing SDE scores for each method on the benchmark datasets, demonstrating that APW improves the metric it was designed to optimize.

5. **Discuss failure cases explicitly:** Acknowledge that APW degrades performance on Electricity/Weather at longer horizons and hypothesize why (e.g., the semantic imbalance assumption may not hold for these datasets, or the MLP may overfit to synthetic decomposition patterns).

# Actual Human Scores
Individual reviewer scores: [0.0, 2.0, 0.0, 0.0]
Average score: 0.5
Binary outcome: Reject
