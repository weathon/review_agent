=== CALIBRATION EXAMPLE 42 ===

# Harsh Critic Review
## Section-by-Section Critical Review

**Title & Abstract**  
The title accurately reflects the core contribution: estimating causal concept effects under latent visual confounding. The abstract clearly states the problem (unobserved concepts as confounders), the proposed method (UnCoVAEr, a partitioned VAE), and the key results (less biased estimates on MorphoMNIST). However, the abstract’s claim of providing a “practical tool for trustworthy concept-level causal inference in partially annotated image datasets” is overstated given the evaluation is entirely on a controlled semi-synthetic benchmark, not real-world data. This limitation should be acknowledged more clearly in the abstract.

**Introduction & Motivation**  
The introduction effectively motivates the problem with compelling examples (medical imaging, facial attributes) and situates the work at the intersection of concept-based explanations and latent-variable causal inference. The contributions are clearly stated. However, the paper could more explicitly differentiate itself from prior latent-variable methods like CEVAE by emphasizing the **partitioned latent space** as a key inductive bias for visual confounding. The related work section is comprehensive but could be better structured to highlight the specific gap (concept incompleteness + visual confounding) that UnCoVAEr addresses.

**Method / Approach**  
The causal graph (Figure 2) is clearly presented, and the problem setup is well-defined. The decomposition into \(Z_C\) (confounder-related) and \(Z_S\) (residual) is a sensible inductive bias. However, several aspects need clarification or raise concerns:

1. **Identifiability and Assumptions**: Section 3 states identification relies on standard proxy assumptions, but the conditions under which the partitioned latent representation actually recovers a sufficient adjustment set are not formally established. The claim that “our adjusted estimator using \(Z_C\) is consistent in principle” is too vague for a causal inference paper. More rigorous discussion of identifiability given the model’s assumptions (e.g., no unobserved colliders, sufficiency of \(X\) as a proxy) is needed.

2. **Model Design Choices**: The use of a binary \(Z_C\) is motivated by interpretability, but no ablation or justification is provided for this choice versus a continuous latent. Similarly, the mutual information regularizer (CLUB) is used to encourage independence between \(Z_C\) and \(Z_S\), but its effectiveness and necessity are not empirically validated. The auxiliary losses are adopted from CEVAE, but their role in improving the quality of \(Z_C\) as a confounder proxy is not clearly demonstrated.

3. **ATE Estimation Procedure**: The sampling procedure for ATE (Equation 7) is clear, but it relies on the auxiliary predictors \(q_{\xi_C}\) and \(q_{\xi_Y}\). The impact of approximation errors in these predictors on the final ATE estimate is not analyzed. The confounding detection criterion (bootstrap test comparing ATE vs. ATE_naive) is reasonable but may be sensitive to sample size; its power/false-positive rate is not evaluated.

4. **Scalability to Many Concepts**: The method conditions on all concepts \(C\) in the decoders. With many concepts, the model may become unwieldy, and the assumption that \(Y\) is independent of \(X\) given \(C\) and \(Z_C\) (footnote 1) may break if the image contains additional predictive information not captured by \(C\) and \(Z_C\). This should be discussed as a limitation.

**Experiments & Results**  
The experimental design on MorphoMNIST is appropriate for a proof-of-concept, with controlled confounding patterns and clear ground-truth ATE. The baselines are well-chosen. However, several issues weaken the empirical validation:

1. **Lack of Real-World Data**: The entire evaluation is on a semi-synthetic dataset where the confounding structure is perfectly known and the image generation process aligns neatly with the model’s assumptions. This severely limits the claim of practical utility. At minimum, a qualitative analysis on a real dataset (e.g., showing learned \(Z_C\) aligns with plausible confounders) should be included.

2. **Incomplete Analysis of Learned Proxies**: The paper claims UnCoVAEr learns confounder proxies that align with underlying latent factors, but no quantitative or qualitative evidence is provided to support this. For MorphoMNIST, one could measure correlation between \(Z_C\) and the true confounder (e.g., thickness). Without this, it’s unclear if the improvement stems from meaningful confounding adjustment or simply better regularization.

3. **Ablation Study Gaps**: The ablation in Table 1 is useful, but key components lack systematic analysis: the effect of the MI regularizer (\(\lambda_{MI}\)), the choice of discrete vs. continuous \(Z_C\), and the sensitivity to the dimensionality of \(Z_C\) and \(Z_S\). The per-concept proxy variant (\(Z_{C_i}\)) is unstable in the multiple confounders case, but no diagnosis is offered.

4. **Confounding Detection Evaluation**: Figure 3 shows detection rates, but the evaluation is incomplete. The method should be assessed in terms of precision and recall (or similar) against ground-truth confounded concepts, not just detection rates. In the multiple confounders case, the detection rate drops significantly; this warrants deeper investigation.

5. **Statistical Significance**: Results are reported with standard deviations over 5 seeds, but no statistical significance tests are performed to confirm that UnCoVAEr’s improvements are meaningful.

**Writing & Clarity**  
The paper is generally well-written and logically structured. However, some sections are dense and could benefit from clearer exposition. For instance, the derivation of Equation 3 and the sampling procedure for ATE could be explained step-by-step. The figures are helpful, but Figure 1’s caption is somewhat confusing (panel (b) is described but not visually distinct). The paper also contains minor formatting artifacts (e.g., misplaced underscores in equations) but these do not impede understanding.

**Limitations & Broader Impact**  
The limitations section (Section 7) appropriately notes the reliance on visual manifestation of confounders and difficulty with complex interactions (e.g., XOR). However, it misses several key limitations: (1) the assumption that the image \(X\) is a sufficient proxy for the confounder, which may fail if confounding arises from non-visual factors (e.g., demographic metadata); (2) the sensitivity to hyperparameters (e.g., \(\lambda_{MI}\), KL annealing schedules); (3) the computational cost of sampling-based ATE estimation; and (4) the lack of validation on real data. The broader impact statement is positive but generic; a more concrete discussion of potential misuse (e.g., overreliance on the method’s confounding detection in high-stakes settings) would be beneficial.

### Overall Assessment
This paper proposes a novel method, UnCoVAEr, for estimating causal concept effects in the presence of unobserved visual confounders. The core idea—partitioning the latent space into confounder-related and residual components—is intuitive and well-motivated. The empirical results on MorphoMNIST demonstrate improved ATE estimation over several baselines in controlled confounding scenarios. However, the paper has significant weaknesses: (1) the identifiability and theoretical grounding are not sufficiently rigorous; (2) the evaluation is entirely on a semi-synthetic benchmark, with no evidence that the method works on real-world data; and (3) key ablations and analyses (e.g., quality of learned proxies, hyperparameter sensitivity) are missing. For ICLR, where novelty, technical soundness, and empirical thoroughness are paramount, the paper in its current form is likely below the acceptance bar. The contribution is promising but requires stronger theoretical justification and more compelling empirical validation, ideally on real datasets, to be considered for publication.

# Neutral Reviewer
## Balanced Review

### Summary
This paper introduces UnCoVAEr, a latent-variable model that addresses the problem of estimating causal effects of human-interpretable concepts on model predictions when some confounding concepts are unobserved. The method partitions the latent representation of an image into a confounder-related component and a non-confounding residual, enabling backdoor adjustment to debias effect estimates. The authors demonstrate on a semi-synthetic MorphoMNIST benchmark that their approach reduces bias compared to existing concept-based and causal baselines.

### Strengths
1. **Clear Problem Formulation and Motivation**: The paper effectively motivates the critical issue of latent visual confounding in concept-based explanations, using compelling examples from medical imaging and fairness (e.g., scanner artifacts confounding atrophy diagnosis, lighting confounding attractiveness predictions). This addresses a significant gap in the interpretability literature.
2. **Well-Structured Methodological Contribution**: UnCoVAEr provides a principled integration of ideas from proximal causal inference (proxy learning) and concept-based models. The explicit partitioning of the latent space (into discrete Z_C and continuous Z_S) with a mutual information regularizer is a clear and justified inductive bias for isolating confounding information.
3. **Rigorous and Controlled Experimental Validation**: The use of MorphoMNIST allows for the creation of precise, semi-synthetic ground-truth causal graphs (single, common, and multiple confounders) with known ATEs. This enables a clean, quantitative comparison against strong baselines (CEVAE, CaCE, CBM variants) and an oracle, demonstrating UnCoVAEr's superior performance in most confounding scenarios, particularly under distribution shift (OOD test).

### Weaknesses
1. **Limited Evidence of Real-World Applicability**: The evaluation is confined to a controlled, synthetic dataset (MorphoMNIST). While this is a valid first step, ICLR typically expects strong evidence that a method can scale to and be effective on complex, real-world image data (e.g., medical scans or facial attribute datasets mentioned in the motivation). The paper acknowledges this as a major limitation, but its absence weakens the claim of providing a "practical tool."
2. **Strong and Unverified Identification Assumptions**: The method's validity hinges on standard but strong proxy causal assumptions (e.g., that the image X contains sufficient proxy signal for the unobserved confounder, no unobserved colliders). The paper states these but does not empirically test their robustness when violated or provide sensitivity analyses. The failure mode in the "multiple confounders (XOR)" scenario hints at these challenges.
3. **Incremental Novelty and Baselines Performance**: The core architectural idea—partitioning a VAE latent space—builds directly upon CEVAE. While the application to concepts and the structured partition are novel, CEVAE itself is a very close baseline and performs competitively in some tests. Furthermore, in the "multiple confounders" setup, simple baselines (Naive, CBM) surprisingly match the oracle, complicating the narrative about the problem's difficulty and the method's necessity.

### Novelty & Significance
**Novelty**: Moderate. The paper adeptly combines and adapts ideas from the causal latent variable (CEVAE) and concept-based explanation literatures. The key novelty is the explicit, regularized partitioning of the image latent space specifically for the task of debiasing concept effects, along with a criterion for confounding detection. However, the individual components (VAEs for causal estimation, concept bottlenecks) are well-established.
**Significance**: Potentially high. If the method proves robust on real-world data, it would provide a crucial tool for improving the trustworthiness of concept-based explanations, which are increasingly used for model auditing in high-stakes domains. The work highlights a critical, often-overlooked assumption (concept completeness) in the interpretability field.

### Suggestions for Improvement
1. **Demonstrate on a Real-World Dataset**: To meet ICLR's bar for impact, a minimum requirement is an experiment on a non-synthetic dataset with plausible latent confounding, even if the ground-truth ATE is unknown. Qualitative analysis (e.g., visualizing what the learned Z_C captures, showing corrected vs. naive explanations on real images) would greatly strengthen the case for practical utility.
2. **Deeper Analysis of Limitations and Assumptions**: Conduct sensitivity analyses on the MorphoMNIST benchmark. For example, systematically degrade the "proxy strength" of X for Z_C or introduce simulated colliders to show how the method's performance degrades. This would provide practitioners with a better understanding of its operating boundaries.
3. **Strengthen the Comparison and Discussion**: Provide a more detailed discussion explaining *why* simple methods perform well in the XOR confounding case and what this implies about the problem structure. Furthermore, compare more directly to recent methods like MCCE (Gao & Chen, 2024, cited) that also address concept incompleteness but with different assumptions, to better situate UnCoVAEr's contributions.

# Spark Finder Review
## How to Improve This Paper

### Missing Experiments (top 3-5 only)
1. **Validation on a real-world dataset with plausible latent confounding.** The paper only evaluates on a fully controlled semi-synthetic benchmark (MorphoMNIST). Without evidence on a realistic dataset (e.g., medical imaging with scanner effects or facial attributes with lighting/skin tone confounds), the claim of a "practical tool" is unsupported.
2. **Ablation on the mutual information regularizer and auxiliary losses.** The paper does not show how much the MI regularizer and auxiliary classification heads contribute to performance. Without this, it's unclear if the partitioned latent design is the key factor or if the gains come from these auxiliary components.
3. **Comparison to state-of-the-art proximal causal inference methods for images.** Baselines like CEVAE are from 2017. More recent methods (e.g., Kompa et al. 2022, Schulte et al. 2025) specifically address image-based confounding and should be compared to demonstrate relative advancement.
4. **Sensitivity analysis to the number of latent confounder dimensions (K).** The paper fixes K (likely to 1 or 2) but does not test sensitivity. If K is misspecified (e.g., true confounders are more complex), the method may fail; this is critical for real-world use where K is unknown.
5. **Experiment where the image does not contain a perfect proxy for the confounder.** The method assumes the confounder manifests visually. A test where the confounder's signal in the image is weak or noisy would reveal the method's breaking point and the practical validity of the identification assumption.

### Deeper Analysis Needed (top 3-5 only)
1. **Quantitative analysis of how well the learned proxy ZC aligns with the true confounder(s).** The paper claims ZC captures confounder-related variation but only shows ATE error reduction. For the semi-synthetic benchmark, report correlation or mutual information between ZC and the ground-truth confounder to validate the proxy quality directly.
2. **Analysis of the confounding detection criterion's false positive/negative rates.** The bootstrap test for flagging confounded concepts is not evaluated against ground truth. Without reporting precision/recall, it's unclear if the detection is reliable or if the reported rates in Figure 3 are meaningful.
3. **Investigation of why the method degrades in the "multiple confounders" (XOR) setting.** The paper notes performance drop but does not diagnose the cause. Is it due to the discrete latent representation, the VAE's inability to capture nonlinear interactions, or the failure of the MI regularizer? This is critical for understanding limitations.
4. **Analysis of the latent disentanglement between ZC and ZS.** The paper uses an MI regularizer but does not provide evidence that ZC and ZS are actually independent or that ZS does not contain confounding information. Metrics like MIG or simple correlation tests are needed to trust the partition.

### Visualizations & Case Studies
1. **Visual counterfactuals showing intervention on ZC.** To demonstrate that ZC captures a meaningful confounder, show generated images where ZC is intervened upon while concepts are fixed. For MorphoMNIST, this should visually alter the confounder attribute (e.g., thickness) without changing other concepts.
2. **t-SNE plots of ZC colored by true confounder states.** This would visually confirm whether the learned latent clusters according to the true unobserved confounder, providing direct evidence for claim (iii) about learning aligned proxies.
3. **Case study on a subset of a real dataset (e.g., CelebA) with simulated confounding.** Even if full real-world validation is missing, a small-scale study where a known attribute is artificially hidden and treated as a latent confounder would show if the method can recover it and correct biases.

### Obvious Next Steps
1. **Apply the method to at least one real-world dataset with suspected latent confounding** (e.g., medical imaging with site effects or facial attributes with lighting). The paper's contribution is severely limited without this step.
2. **Provide theoretical discussion or empirical evidence on identifiability under the model assumptions.** The paper briefly mentions identification requirements but does not connect them to the model design. A discussion of how the partitioned VAE satisfies (or approximates) proxy completeness conditions is necessary for causal credibility.
3. **Explore continuous or higher-dimensional confounder representations.** The paper uses binary ZC for simplicity, but real confounders may be continuous or multi-faceted. An ablation with continuous ZC would test the generality of the approach.
4. **Include a baseline that uses a standard VAE (no partition) with the same adjustment procedure** to isolate the benefit of the explicit partition versus simply using a latent representation from a standard VAE for backdoor adjustment.

# Final Consolidated Review
## Summary
This paper introduces UnCoVAEr, a latent-variable model that estimates the causal effect of human-interpretable concepts on a model's prediction when some confounding concepts are unobserved. The method partitions an image's latent representation into a confounder-related component and a non-confounding residual, enabling bias correction via backdoor adjustment. Evaluation on a controlled semi-synthetic MorphoMNIST benchmark shows improved estimation accuracy over several baselines.

## Strengths
- **Clear and well-motivated problem formulation.** The paper effectively identifies a critical gap in concept-based explanation methods: the bias introduced by unobserved visual confounders. It provides compelling examples from medical imaging and fairness to ground the work.
- **Principled methodological integration and a sensible inductive bias.** The model cleanly integrates ideas from proximal causal inference and concept-based models. The explicit partitioning of the latent space into discrete confounder-related (`Z_C`) and continuous residual (`Z_S`) components, regularized for independence, is a direct and appropriate structural bias for the stated problem.
- **Rigorous and controlled experimental validation.** The use of MorphoMNIST allows for the construction of precise, semi-synthetic ground-truth causal graphs (single, common, and multiple confounders). This enables a clear, quantitative comparison against strong and relevant baselines (CEVAE, CaCE, CBM variants), demonstrating the method's superior performance in reducing ATE estimation bias, particularly under distribution shift.

## Weaknesses
- **Evaluation is confined to a semi-synthetic benchmark.** The entire empirical validation is performed on MorphoMNIST, a controlled dataset where the confounding structure is perfectly known and aligns with the model's assumptions. This severely limits the claim of providing a "practical tool" and offers no evidence that the method would work on the complex, real-world datasets (e.g., medical imaging, facial attributes) used to motivate the problem.
- **Incomplete analysis of the learned confounder proxies.** The paper claims UnCoVAEr learns proxy variables that "align with underlying latent factors," but provides no quantitative evidence (e.g., correlation between learned `Z_C` and the true confounder) to support this. The improvement in ATE error, while clear, could stem from better regularization rather than meaningful confounding adjustment.
- **Limited investigation of a key failure mode.** In the "multiple confounders (XOR)" scenario, performance degrades and confounding detection fails. The paper notes this but does not diagnose the cause, leaving an important limitation unexplained. This is a substantive weakness as it reveals a boundary condition for the method's applicability.

## Nice-to-Haves
- A sensitivity analysis of key hyperparameters (e.g., the mutual information regularizer strength, latent dimension `K`) and model design choices (e.g., binary vs. continuous `Z_C`).
- A more detailed comparison with a broader set of recent image-based proximal causal inference methods (beyond CEVAE from 2017).
- Visualizations (e.g., t-SNE plots of `Z_C`, counterfactual images from `Z_C` interventions) to qualitatively demonstrate what the learned proxy captures.

## Novel Insights
The paper's core novel insight is the explicit framing of concept incompleteness in interpretability as a visual latent confounding problem, addressed via a partitioned latent-variable model. By structurally separating confounder-related from residual image variation within a VAE framework and using this partition for backdoor adjustment, it provides a concrete, learnable mechanism to debias concept effect estimates—a specific need at the intersection of causal inference and concept-based explanation that prior works had not directly solved.

## Suggestions
- To substantiate claims of practical utility, conduct an experiment on a real-world dataset with plausible latent confounding (e.g., a medical imaging dataset with site/scanner metadata or a facial attribute dataset). Even without ground-truth ATE, a qualitative analysis showing what the learned `Z_C` captures and how it changes concept effect estimates would be invaluable.

# Actual Human Scores
Individual reviewer scores: [2.0, 4.0, 6.0, 2.0]
Average score: 3.5
Binary outcome: Reject
