# Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Accurate prediction of the need for invasive mechanical ventilation (IMV) in intensive care units (ICUs) patients is crucial for timely interventions and resource allocation. However, variability in patient populations, clinical practices, and electronic health record (EHR) systems across institutions introduces domain shifts that degrade the generalization performance of predictive models during deployment. Test-Time Training (TTT) has emerged as a promising approach to mitigate such shifts by adapting models dynamically during inference without requiring labeled target-domain data. In this work, we introduce Adaptive Test-Time Training (AdaTTT), an enhanced TTT framework tailored for EHR-based IMV prediction in ICU settings. We begin by deriving information-theoretic bounds on the test-time prediction error and demonstrate that it is constrained by the uncertainty between the main and auxiliary tasks. To enhance their alignment, we introduce a self-supervised learning framework with pretext tasks: reconstruction and masked feature modeling optimized through a dynamic masking strategy that emphasizes features critical to the main task. Additionally, to improve robustness against domain shifts, we incorporate prototype learning and employ Partial Optimal Transport (POT) for flexible, partial feature alignment while maintaining clinically meaningful patient representations. Experiments across multi-center ICU cohorts demonstrate competitive classification performance on different test-time adaptation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
The paper proposes a variant of test-time training for tabular data in which they predict invasive mechanical ventilation in the intensive care.  They derive an information-theoretic bound on the prediction error, which suggests that the prediction error is bounded by the entropy between the main task labels and the auxiliary task labels at test time. The authors use this insight to propose an adaptive self-supervised task that masks features with probabilities depending on their importance for the main task. The authors additionally a variant of prototype-guided adaptation. The method and baselines from the image domain are trained on data from Site A and evaluate on a temporal holdout (Site A) and two geographical hold-outs (Site B and MIMIC).

### Strengths
- The paper derives bounds on the prediction error that provide an information-theoretic perspective on how to choose a suitable SSL objective, which they use to propose an SSL task suitable for longitudinal, tabular ICU data. 
- The authors perform several ablation studies showing the contributions of individual components.

### Weaknesses
- The theoretical insight that the SSL task should align with the main task is of limited novelty and already discussed, albeit from another angle, in the original work on test-time training by Sun et al. 
- I understand how the theoretical insights motivate using an adaptive masking strategy for SSL. The motivation for using both SSL *and* prototypes is less clear to me, though. The result is a complicated method with six different loss terms. Despite this complexity, the method barely outperforms doing no test-time training. 
- The experiments are limited to a single disease area in a single setting. It would be relatively easy to include other outcomes in the ICU (e.g., acute kidney injury or the classic sepsis example from the PhysioNet Challenge 2019).

### Questions
- Could the authors comment on how realistic their assumption of $Y'_s \rightarrow Z' \rightarrow Y'_m$ is for the main and auxiliary task they are investigating? What is the implication if it doesn't hold? 
- I was unable to find details on how cluster assignment $\mathcal{A}(z_i)$ was performed. Please add details on this in the manuscript.   
- Could the authors comment on why they used the specific version of OT rather than more conventional POT like the referenced version by Chapel? 
- All baselines are NN-based and were adapted from the image domain. There seem to be recent works on test-time adaptation that were specifically created for tabular data, such as TabLog presented at ICML by Ren et al. (2024). Could the authors comment on why no dedicated methods for tabular data were included as baselines?
- The conclusions seem too strong. While the method proposed by the authors did outperform the baselines (most of which performed worse than doing nothing), its improvement over doing nothing was marginal and seems to highlight a limited benefit of test-time training over leaving the model unchanged at test-time. If the authors think their method does provide a meaningful practical benefit for their use case, it would be good to demonstrate this with additional evaluations.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes Adaptive Test-Time Training (AdaTTT) for predicting 24-hour invasive mechanical ventilation (IMV) need from multi-center EHR data under distribution shift. AdaTTT adapts an IMV predictor at inference time on multi-site EHR data using  
(i) an info-theoretic bound to guide auxiliary SSL,  
(ii) dynamic feature-aware masking, and  
(iii) prototype alignment via partial optimal transport (POT).  
Across three cohorts (incl. MIMIC-IV), it shows consistent but modest AUROC gains (~1–2%) better calibration (Brier, reliability plots). Authors added stronger baselines (CoTTA, SAR, T3A), runtime analysis, and ablations; some concerns remain.

### Strengths
- Info-theoretic error bound motivates SSL + prototype/POT.  
- Source-free, per-patient test-time adaptation with bounded updates and reset-to-clean safeguard.  
-  Consistent multi-site improvements; improved calibration; expanded baselines and ablations.

### Weaknesses
-  Gains (~1%) may fall within noise; lacks decision-curve or threshold-level analysis.  
-  Vision-based baselines may not adapt fairly to tabular EHR; no EHR-transformer baseline.  
-  No subgroup audit; only tested on IMV/EHR; reproducibility details partly scattered.

### Questions
1. What exactly is novel about AdaTTT compared to existing TTT, TTT++, ClusT3, or contrastive/temporal SSL frameworks?

2. Are the vision-based TTT baselines fairly adapted to structured EHR data?

3. Why choose reconstruction and masked-feature SSL tasks?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Adaptive Test-Time Training (AdaTTT), a novel framework designed to get rid of large-scale labeling and retraining, and enhance model robustness for predicting the need for invasive mechanical ventilation (IMV) in ICU patients across multi-center EHR datasets. It builds upon traditional test-time training (TTT) by addressing two main challenges: (1) ensuring stronger alignment between main and auxiliary tasks, and (2) maintaining robustness under domain shifts in clinical data.

### Strengths
* This paper introduces a theoretically principled and practically motivated adaptation framework combining information theory, SSL, and transport alignment.
* This paper includes experiments across real EHR datasets with fair baseline comparisons and ablation studies.
* This paper is structured and well-written, with figures showing dynamic risk evolution and feature importance shifts.
* The experiment result demonstrates consistent improvement in both predictive accuracy and calibration under domain shifts — a critical problem for clinical ML deployment.

### Weaknesses
* The testing windows for Site A (Jan–Jun 2024) and Site B (Jan 2023–Aug 2024) partially overlap, which raises the possibility that similar patient populations, care protocols, or even duplicated encounters could appear in both test sets. Such overlap could inflate generalization performance by reducing the effective distribution shift the model faces.
* Although Appendix B.1 outlines inclusion criteria (≥ 5 hours ICU stay, no prior IMV, etc.), it does not explain how cohorts were sampled from each institution, whether patient IDs were cross-checked for duplication across sites, or how temporal leakage was prevented between development and test periods. It would be helpful if the author could provide a clear population selection process figure for further reference.
* While the author provides plots for feature importance through multiple iterations and risk trajectories, the paper does not include any clinical human evaluation to verify that the learned feature relevance aligns with actual clinical reasoning.

### Questions
* The paper mentions that Site B includes ICU admissions between January 2023–August 2024. Does this dataset originate from a completely independent institution or another ICU within the same healthcare network as Site A?
* The paper currently does not include the summary statistics. Could you include a demographics table to help assess whether differences in population structure might explain performance gaps across sites?
* For the prototype learning, in the paper, it states that "The prototype set size is k = 4". Could you provide more details of how this value is chosen and if there's any exploration of how sensitive AdaTTT is to larger k?
* Did domain experts review these feature importance trends, or do they only reflect internal gradient magnitudes?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces AdaTTT (Adaptive Test-Time Training), a framework for predicting the need for invasive mechanical ventilation (IMV)  across multiple ICU centers. The method addresses domain shift challenges in EHR data through two key innovations: (1) dynamic self-supervised learning with feature-aware masking that prioritizes clinically relevant features, and (2) prototype-guided adaptation using Partial Optimal Transport (POT) for flexible feature alignment. The authors provide information-theoretic bounds showing test-time prediction error is constrained by uncertainty between main and auxiliary tasks. Experiments across multi-center ICU cohorts (Sites A, B, and MIMIC-IV) demonstrate consistent improvements over existing test-time adaptation baselines.

### Strengths
Following are the strengths of the paper:

1. The paper addresses an important problem—predicting IMV need across hospitals with different EHR systems, patient populations, and clinical practices.  

2. The novel dynamic masking strategy that adapts based on feature importance is an important contribution. Also, POT-based alignment is more flexible than rigid full-transport approaches and better suited to partial distribution shifts.

3. The information-theoretic bounds provide valuable insights into why auxiliary task alignment matters for test-time adaptation.

### Weaknesses
Following are the main weakenesses of the paper:

1. Paper presentation and writing needs improvement. Specifically:
   - The prototype-guided adaptation component (Section 3.3.2) is not well-motivated. The rationale behind its inclusion is unclear, and the implementation details are insufficiently explained.
   - The methodology as a whole lacks clarity, largely due to the number of interconnected components. It would be helpful to include a comprehensive diagram or a formal algorithm to clearly illustrate the roles of each component and how they interact within the overall framework.

2. Although the proposed approach consistently outperforms existing baselines, the improvements are relatively modest. For instance, Site A achieves an AUC of 85.02% compared to SAR's 84.30%, and Site B achieves 84.10% versus CoTTA's 83.81%. While these results are favorable, the clinical relevance of a $\approx$1% AUC improvement remains uncertain. A discussion on the practical impact of such performance gains in real-world clinical settings would add value to the paper.

3. Figure 5 shows the sequential update mechanism degrades over time, but the paper doesn't investigate why or propose solutions. This also limits applicability in streaming scenarios where resetting to pretrained weights may not be desirable.

### Questions
1. Does the framework also address conditional distribution shifts (i.e., shifts in ( P(Y|X) ))? Under what assumptions does the framework operate effectively, and in which scenarios might those assumptions break down?

2. How strong is the assumption that auxiliary task labels $ Y_s' $ influence the main task labels $Y_m'$ only through the shared representation $Z'$? What are the implications if this assumption does not hold fully in practice?

3. The framework introduces several hyperparameters — such as  $\lambda_{\text{recon}}$, $ \lambda_{\text{proto}} $, $\lambda_{\text{reg}} $, $ \lambda_{\text{ot}} $, masking warm-up epochs, and prototype size ( k ).  How should these hyperparameters be selected or tuned?

4. Why were only five gradient steps used during test-time adaptation in the experiments? What is the rationale behind this choice, and what are the effects of using more than five steps?

### Soundness
3

### Presentation
2

### Contribution
2
