=== CALIBRATION EXAMPLE 49 ===

# Harsh Critic Review
## Section-by-Section Critical Review

### Title & Abstract
The title accurately reflects the paper's contribution: estimating causal concept effects under visual latent confounding. The abstract clearly states the problem (unobserved concepts as confounders), the method (partitioned latent representations via a VAE), and the key results (less biased estimates on MorphoMNIST). The claims are specific and appear supported by the experiments. No major issues.

### Introduction & Motivation
The motivation is strong and well-articulated, using compelling examples from medical imaging and fairness to illustrate the real-world consequences of unobserved visual confounders. The introduction effectively situates the work at the intersection of concept-based explanations and proximal causal inference. The contributions (formalizing the problem, partitioned latent bias, confounding detection, empirical validation) are clearly listed at the end.

### Preliminaries & Problem Setup
The setup is clear: images \(X\), binary concepts \(C\), binary outcome \(Y\), with the goal of estimating ATE for concepts. The causal graph (Figure 2) is well-defined and central to the method. The identification discussion (Section 3, "Identification requirements and limitations") is appropriately cautious, noting the necessity of the confounder leaving a visual trace and the need for overlap. This manages expectations.

**Concern:** The assumption that \(Y\) is independent of \(X\) given \(C\) and \(Z_C\) (footnote 1 and generative model) is a strong modeling choice. While it follows from the assumed graph, its validity in real data is not trivial. The paper should discuss the implications of this conditional independence assumption, especially if the image contains direct causes of \(Y\) not mediated by the observed concepts or the learned confounder proxy \(Z_C\).

### Method (UnCoVAEr)
The method is a structured extension of CEVAE with a partitioned latent space (\(Z_C\) for confounding, \(Z_S\) for residual variation). The ELBO derivation and inclusion of auxiliary losses and a mutual information regularizer are sound. The use of a binary \(Z_C\) for interpretability is justified given the binary concept/label setup.

**Major Concerns:**
1.  **Identifiability & Theoretical Grounding:** The method relies heavily on the ability of the VAE to learn a proxy \(Z_C\) sufficient for backdoor adjustment. While the paper references proximal causal inference literature, it does not provide a formal identifiability argument for *its specific model and training objective*. This is a significant gap for an ICLR submission. The critique of CEVAE by Rissanen & Marttinen (2021) is cited, but the response is essentially empirical ("our results show that UnCoVAEr recovers unbiased ATEs whenever these identification assumptions hold"). A more rigorous discussion or proof sketch under what conditions the learned \(Z_C\) satisfies the adjustment sufficiency property (P1) is needed.
2.  **Mutual Information Regularizer:** The use of the CLUB estimator to minimize \(I(Z_C; Z_S)\) is motivated as reducing "information leakage." However, the factorization \(q(Z_C, Z_S|X,C,Y) = q(Z_C|...)q(Z_S|...)\) already assumes conditional independence. The regularizer seems to enforce *marginal* independence. The necessity and effect of this choice are not validated (e.g., via an ablation study on \(\lambda_{MI}\)). Its impact on the disentanglement of confounder vs. residual information is unclear.
3.  **ATE Estimation Procedure (Eq. 7):** The procedure samples \(c \sim q_{\xi_C}(C|X)\), \(z \sim q_{\phi_c}(Z_C|X,c,y)\), then evaluates \(p_{\theta_y}(Y|C_i=c, C_{-i}=c^{(-m)}, Z_C=z)\). This mixes the auxiliary concept predictor \(q_{\xi_C}\) with the generative concept decoder \(p_{\theta_c}\) and outcome decoder \(p_{\theta_y}\). The justification for this hybrid sampling scheme is not fully explained. Why not sample \(c\) from the (trained) prior \(p(C|Z_C)\)? A clearer derivation of the estimator from Eq. 3 would help.

### Experiments & Results
The controlled MorphoMNIST benchmark is appropriate for initial validation. The three confounding scenarios (single, common, multiple) test different aspects of the method. The baselines are well-chosen and representative.

**Major Concerns:**
1.  **Performance in "Multiple Confounders" Scenario:** The results show that Naive and CBM baselines achieve near-oracle error (~0.01), while UnCoVAEr's error is higher (~0.07). The authors note that naive methods "directly learn and exploit the *intensity-Y* relation" because the confounder acts via XOR. This is a critical observation: **in this specific setup, the unobserved confounder does *not* bias the observational ATE estimate.** This suggests the "problem" the method aims to solve is absent here, making it an unfair comparison. The paper should either: (a) modify the data-generating process so that confounding *does* bias the naive ATE, or (b) clearly frame this scenario as a stress test for confounding *detection* (where UnCoVAEr should ideally report no confounding for *intensity*), not for ATE estimation improvement. Figure 3 shows UnCoVAEr's detection rate for *intensity* is near 1.0, which is a *false positive* if the true ATE is unbiased. This undermines the claim of reliable confounding detection.
2.  **Confounding Detection Evaluation:** The detection method (bootstrapped CI overlap between ATE and ATE_naive) is heuristic. Its accuracy is not quantitatively evaluated against the ground truth (which is known in this synthetic benchmark). Figure 3 reports "detection rates" but without a clear definition of ground truth positives/negatives. The paper should report standard detection metrics (precision, recall) for each scenario.
3.  **Ablation Study Incomplete:** The ablation in Table 1 is useful but omits key components. Most importantly, it does **not** ablate the mutual information regularizer (\(L_{MI}\)) or the auxiliary losses (\(L_{aux,C}, L_{aux,Y}\)). The contribution and sensitivity of these terms are therefore unknown.
4.  **Limited Scope of OOD Evaluation:** The "OOD" test only varies the confounding strength (\(\alpha\)). This is a very limited distribution shift. A more convincing robustness test would involve changes to the image distribution itself (e.g., different MorphoMNIST parameters) while keeping the causal structure fixed.

### Writing & Clarity
The paper is generally well-written and structured. The causal graph figures are helpful. Some parts of the method section (e.g., the sampling procedure for ATE estimation) could be explained more step-by-step for clarity. No major impediments to understanding.

### Limitations & Broader Impact
Section 7 appropriately notes key limitations: reliance on visual manifestation of confounders, difficulty with non-linear interactions (XOR), and the critical need for validation on real-world data. A broader impact statement is missing but could easily note positive applications in auditing models for spurious correlations and potential misuse if the method fails silently.

### Overall Assessment
This paper tackles an important and well-motivated problem at the nexus of interpretability and causality. The core idea—using a partitioned VAE to learn confounder proxies from images—is novel and sensible. However, for an ICLR submission, the **lack of theoretical grounding** (identifiability arguments) and **incomplete empirical validation** (especially the problematic "multiple confounders" scenario, unvalidated confounding detection, and missing ablations) are significant weaknesses. The contribution is promising but currently rests on empirical results from a single, controlled benchmark where the method's advantage is not fully demonstrated across all scenarios. To be acceptable, the paper needs to address the identifiability gap, provide a more rigorous evaluation of its confounding detection, and re-analyze or reconfigure the multiple confounder experiment.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces UnCoVAEr, a latent-variable model designed to estimate unbiased causal effects of human-interpretable visual concepts on model predictions when some concepts are unobserved and act as confounders. The method partitions the latent representation of an image into a confounder-related component and a non-confounding residual, enabling backdoor adjustment for causal effect estimation and identification of which observed concepts are confounded. Evaluation on a semi-synthetic MorphoMNIST benchmark shows improved performance over several baselines.

### Strengths
1. **Clear Problem Formulation**: The paper clearly motivates the practical issue of latent visual confounding in concept-based explanations, providing compelling examples from medical imaging and fairness (e.g., scanner artifacts confounding atrophy measurements, lighting confounding facial attributes).
2. **Novel Methodological Integration**: UnCoVAEr thoughtfully integrates ideas from concept-based models (e.g., CBMs) and latent-variable proximal causal inference (e.g., CEVAE). The structured latent partition (confounder vs. residual) is a sensible inductive bias for the problem.
3. **Rigorous and Controlled Empirical Evaluation**: The authors design a detailed semi-synthetic benchmark on MorphoMNIST with multiple confounding scenarios (single, common, multiple confounders) and both in-distribution and out-of-distribution test regimes. The results demonstrate that UnCoVAEr consistently reduces ATE estimation error compared to strong baselines, including CEVAE, CaCE, and image-adjustment methods.
4. **Thorough Ablation Studies**: Ablations validate key design choices (e.g., the necessity of the image reconstruction term, the partitioned latent space, and the mutual information regularizer), providing evidence for the model's internal mechanics.
5. **Good Reproducibility**: The paper includes a detailed appendix with code availability, hyperparameters, and a reproducibility checklist, meeting ICLR's standards for replicability.

### Weaknesses
1. **Limited Real-World Validation**: The evaluation is confined to a controlled, semi-synthetic dataset (MorphoMNIST). While this is appropriate for initial validation, the paper lacks demonstration on a complex, real-world image dataset where the true confounding structure is unknown but the problem is most pressing (e.g., medical imaging or fairness audits). This limits the claim of providing a "practical tool."
2. **Simplified Data Generation**: The MorphoMNIST concepts are binary and generated via simple conditional sampling (Gaussian means). The confounding mechanisms are also simple (e.g., direct copying with probability α). The method's performance on more complex, continuous, or non-linear concept relationships is untested.
3. **Incomplete Discussion of Identification**: The identifiability discussion (Section 3) is somewhat informal. While it correctly notes that identification requires the confounder to leave a trace in the image, it does not deeply engage with the formal completeness/rank conditions from the proximal inference literature or discuss the implications of model misspecification.
4. **Weaker Performance in Complex Settings**: As noted in the results, UnCoVAEr's performance degrades in the "multiple confounders" scenario with a non-linear (XOR) mechanism, and the per-concept proxy variant becomes unstable. This indicates a limitation in handling interacting or complex confounder structures.

### Novelty & Significance
The paper's novelty lies in its specific integration of partitioned latent representations into a VAE framework to address *visual* latent confounding for *concept-based* causal effect estimation. This is a meaningful advancement over prior work that either handles latent confounding in generic tabular settings (CEVAE) or estimates concept effects without addressing incomplete concept sets. The significance is potentially high for trustworthy AI, as unbiased concept-effect estimates are crucial for model auditing and explanation in high-stakes domains. However, the impact is currently tempered by the lack of validation on real-world data.

### Suggestions for Improvement
1. **Include a Real-World Case Study**: Add an experiment on a real dataset (e.g., a medical imaging dataset with suspected site/scanner confounding or a facial attribute dataset with potential lighting/demographic confounders). Even without ground-truth ATE, qualitative analysis of which concepts are flagged as confounded and how effect estimates change after adjustment would greatly strengthen the practical contribution.
2. **Explore More Complex Concept Relationships**: Test the method on benchmarks with continuous concepts or more complex, non-linear generative relationships between confounders and observed concepts to better understand its limitations.
3. **Deeper Theoretical Grounding**: Briefly formalize the identification assumptions using the language of proximal causal inference (e.g., discussing relevance and exclusion conditions for the image as a proxy) and discuss how the model architecture aims to satisfy them.
4. **Address Computational Complexity**: Discuss the training and inference cost relative to baselines, especially if scaling to higher-resolution images or larger concept sets is anticipated.
5. **Clarify the Confidence Interval Test**: The bootstrap test for flagging confounded concepts is mentioned but not detailed. Providing the specific procedure (e.g., number of bootstrap samples, how the intervals are constructed) in the main text or appendix would aid reproducibility.

**Overall Recommendation for ICLR:** This is a well-executed paper that addresses an important problem with a novel method and thorough synthetic experiments. For ICLR, which values both technical rigor and real-world relevance, the major gap is the lack of validation on real data. A *borderline accept* could be justified if the reviewers value the clear problem formulation, methodological innovation, and strong controlled evaluation. A rejection might occur if reviewers deem the real-world applicability unproven. The authors should strongly address this in a revision.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validation on a real-world dataset with known or suspected confounding.** The paper only uses a controlled synthetic benchmark (MorphoMNIST). Without demonstration on a real dataset (e.g., medical imaging with scanner effects or facial attributes with lighting/skin tone confounds), the claim of providing a "practical tool" is unsupported and the method's utility in practice is unproven.
2. **Comparison with state-of-the-art image-based proximal causal inference methods.** Baselines omit recent methods like Kompa et al. (2022) or Schulte et al. (2025) that directly adjust for confounding using image features. Their exclusion leaves it unclear whether the proposed partitioned latent space offers any advantage over these existing high-dimensional proxy approaches.
3. **Ablation on the binary/discrete latent assumption.** The method assumes binary confounder proxies and concepts. An experiment with continuous concepts/confounders (common in real data, e.g., age, severity) is missing. Without this, the generality of the approach is questionable.
4. **Sensitivity analysis to violations of the assumed causal graph.** The method relies on the graph in Figure 2. Experiments where key assumptions are violated (e.g., presence of unobserved colliders, or when the image does not contain sufficient proxy signal) are absent. This is critical to understand the method's robustness and failure modes.

### Deeper Analysis Needed (top 3-5 only)
1. **Interpretability and validation of the learned confounder proxy ZC.** The paper claims ZC aligns with underlying latent factors, but provides no analysis (e.g., correlation with true confounders in synthetic data, or visualization of what ZC captures). Without this, it's unclear whether ZC captures meaningful confounding or just arbitrary variation.
2. **Detailed performance analysis of the confounding detection criterion.** The method flags a concept as confounded based on bootstrap confidence intervals. A systematic analysis of false positive/negative rates across confounding strengths and types is missing. This is necessary to trust the detection claim.
3. **Investigation of failure in the XOR (multiple confounders) scenario.** Performance degrades in this case, but no analysis is given to explain why (e.g., inability of the variational approximation to capture complex interactions, or insufficient capacity). Understanding this failure is key for limitations.
4. **Hyperparameter sensitivity analysis.** The paper uses fixed weights (λC, λY, λMI) and annealing schedules. No analysis shows how sensitive the results are to these choices, which is important for reproducibility and practical use.

### Visualizations & Case Studies
1. **Counterfactual image generations by intervening on ZC.** The decoder can generate images given (C, ZC, ZS). Visualizing counterfactuals by intervening on ZC would show what visual factors the learned proxy captures, directly validating whether it represents a plausible confounder.
2. **Case studies on a real dataset (e.g., CelebA).** Even without ground truth, qualitative analysis showing how ATE estimates change for attributes like "smiling" vs. "attractiveness" after adjustment could illustrate the method's potential to correct for suspected confounders like lighting or demographics.
3. **Visualization of the latent spaces (ZC and ZS).** Using t-SNE/PCA plots colored by true confounders (in synthetic data) or by observed concepts would help assess if ZC captures confounding variation and if ZS is indeed residual.

### Obvious Next Steps
1. **Apply the method to at least one real-world dataset with plausible confounding.** This is the most critical next step; without it, the paper remains a proof-of-concept on synthetic data and lacks convincing evidence for practical impact.
2. **Extend the model to handle continuous concepts and confounders.** Many real-world concepts are continuous; the binary assumption severely limits applicability. This extension should have been explored.
3. **Compare with additional baselines from the proximal causal inference literature.** As noted, methods like Kompa et al. (2022) are highly relevant and their omission weakens the empirical comparison.
4. **Provide a more thorough analysis of identifiability and estimation consistency.** The paper briefly mentions identification assumptions but does not empirically test how violations affect estimates. A simulation study varying proxy strength and overlap would strengthen the theoretical claims.

# Final Consolidated Review
## Summary
This paper introduces UnCoVAEr, a variational autoencoder with a partitioned latent space designed to estimate causal effects of human-interpretable visual concepts when unobserved confounders are present. By separating latent representations into confounder-related and residual components, the method enables backdoor adjustment for unbiased effect estimation and identifies which observed concepts are confounded. Evaluation on a controlled MorphoMNIST benchmark shows improved performance over several baselines in reducing estimation bias.

## Strengths
- **Clear and well-motivated problem formulation:** The paper effectively articulates the practical issue of latent visual confounding in concept-based explanations, using compelling examples from medical imaging and fairness to underscore the need for robust causal inference.
- **Novel methodological integration:** UnCoVAEr thoughtfully combines ideas from concept-based models and latent-variable proximal causal inference, introducing a structured latent partition as a sensible inductive bias for isolating confounder proxies.
- **Thorough controlled evaluation:** The design of multiple semi-synthetic confounding scenarios (single, common, multiple) on MorphoMNIST, along with in-distribution and out-of-distribution tests, provides rigorous evidence for the method's effectiveness in reducing ATE estimation error compared to relevant baselines.

## Weaknesses
- **Lack of real-world validation:** The evaluation is confined to a controlled synthetic benchmark. Without demonstration on a complex real-world dataset (e.g., medical imaging or facial attributes with plausible confounding), the claim of providing a "practical tool" is unsupported, significantly limiting the practical impact.
- **Incomplete evaluation of confounding detection:** The method for flagging confounded concepts relies on a bootstrap confidence interval test, but its accuracy is not quantitatively assessed against ground truth (e.g., via precision/recall metrics). This undermines the contribution of reliable confounding identification.
- **Performance degradation in complex confounding settings:** In the multiple confounders scenario with a non-linear (XOR) mechanism, UnCoVAEr shows higher error than naive baselines and may produce false positives in detection, indicating a limitation in handling interacting or complex confounder structures.
- **Missing ablations for key components:** The necessity of the mutual information regularizer and auxiliary losses is not validated through ablation studies, leaving their contribution and impact on performance unclear.
- **Restrictive binary and discrete assumptions:** The method is designed for binary concepts and confounder proxies, with no exploration of continuous settings common in real data (e.g., age, severity), which limits generality and applicability.
- **Insufficient interpretability analysis:** No evidence is provided to validate that the learned proxy Z_C meaningfully aligns with underlying confounding factors beyond improving ATE estimates, such as correlation analysis with true confounders in synthetic data or visualizations.

## Nice-to-Haves
- Comparison with additional proximal causal inference methods that adjust using image features, such as Kompa et al. (2022) or Schulte et al. (2025), to better contextualize the advantages of the partitioned latent space.
- Visualizations of counterfactual images by intervening on Z_C or latent space plots to enhance interpretability of the learned proxies.
- Hyperparameter sensitivity analysis, especially for the weights of the mutual information regularizer and auxiliary losses, to aid reproducibility.

## Removed Points
*These points are flagged to be removed, treat them with caution*
- **Criticism that the ATE estimation procedure is unclear or unjustified:** The paper describes the sampling scheme in Section 4 and Eq. 7, and while it could be explained more step-by-step, it is not factually incorrect or fundamentally flawed.
- **Demand for extensive theoretical identifiability proofs:** The paper acknowledges identification assumptions in Section 3 and cites proximal inference literature; while deeper theoretical grounding could strengthen the work, its absence does not invalidate the empirical contributions given the community standards for such methods.
- **Claim that the OOD evaluation is overly limited:** The paper defines out-of-distribution as a shift in confounding strength, which is a reasonable and focused test for robustness within the study's scope.
- **Suggestion to add a broader impact statement:** This is a formatting or stylistic note not central to the technical evaluation.

## Novel Insights
None beyond the paper's own contributions.

## Suggestions
- Conduct experiments on at least one real-world dataset with suspected confounding (e.g., medical imaging with scanner effects or CelebA with lighting/demographic confounds) to demonstrate practical utility, even if ground-truth effects are unknown.
- Perform a quantitative evaluation of the confounding detection mechanism using standard metrics like precision and recall based on the known ground truth in synthetic settings.
- Ablate the mutual information regularizer and auxiliary losses to clearly establish their contribution to the model's performance.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 6.0, 2.0]
Average score: 3.5
Binary outcome: Reject
