# CASE: Coupled Adaptive Feature–Target Smoothing with Density-Gated Mixture-of-Experts for Robust Imbalanced Tabular Regression

- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Although tabular data are central to many real-world applications, their target distributions are often imbalanced, as the majority of samples correspond to a narrow range of values. This imbalance severely degrades performance in sparse, few-shot regions. Prior work on imbalanced regression has typically relied on coarse binning of the target space, discarding fine-grained information, or on spatial-locality assumptions inherited from image domains. We introduce CASE (Coupled Adaptive Feature–Target Smoothing with Density-Gated Mixture-of-Experts), a framework tailored to deep imbalanced tabular regression. CASE combines two complementary mechanisms. (i) Coupled Adaptive Smoothing first identifies “true” neighbors by jointly considering similarities in both the feature and target spaces. Based on these neighbors, it then calibrates representations in sparse regions by scaling the smoothing strength according to each sample’s continuous density. (ii) A Density-Gated Mixture-of-Experts (MoE) weights the contributions of specialized experts via a gate that predicts a density range from the original features. During training, experts take the calibrated features as input; at inference, the learned experts operate on the original features, yielding both higher accuracy and faster inference. Across 40 tabular benchmarks, CASE establishes state-of-the-art performance on balanced test sets by attaining the top average rank. Notably, it demonstrates exceptionally robust by minimizing performance loss on the original imbalanced test sets, consistently delivering balanced predictions that enhance few-shot accuracy without significantly sacrificing many-shot performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the challenge of imbalanced tabular regression. The authors propose CASE, a novel framework with two main components. First, Coupled Adaptive Smoothing. Instead of just finding neighbors based on features, CAS identifies "true" neighbors by jointly considering similarity in both the feature and target spaces. It then smooths samples from sparse regions by blending them with these neighbors. Second, Density-Gated Mixture-of-Experts. A "gating network" learns to identify the data's density region (e.g., few-shot, medium-shot, or many-shot) from the original features. It then routes the data to specialized "expert" models. The model uses an asymmetric training process: the experts are trained on the "smoothed" features from CAS to help them learn, but at inference time, they run on the original features.

### Strengths
1.	The paper tackles the critical and relevant challenge of imbalanced regression, a problem that is far less explored than its classification counterpart 
2.	The authors validate their method on a diverse benchmark of 40 public tabular datasets.

### Weaknesses
1.	A notable concern is the inconsistency regarding inference speed. The abstract and main text describe the model as achieving "faster inference" and being "fast and efficient." However, the empirical results in Appendix A.4 (Table 13) show an inference latency is 4 times higher than the single MLP baselines, as ($K+1$) forward passes are needed due to MoE.
2.	The model's asymmetric architecture introduces a significant theoretical gap. The expert networks are trained on CAS-calibrated (smoothed) features, but are evaluated on the original, uncalibrated features during inference. The paper does not provide a theoretical justification or empirical about this.
3.	The CASE framework is highly complex, combining several novel mechanisms (Coupled Smoothing, a Learnable Metric, Adaptive Density, and an Asymmetric MoE). The provided ablation study is insufficient to disentangle the individual contributions of these sub-components.
- It is unclear if the performance gain originates from the complex density-gating or simply from ensembling three models. A baseline comparing CASE to a simple, non-gated ensemble of three MLPs or other baseline is critically missing.
- It is not possible to determine which sub-component of the CAS module (e.g., the learnable metric vs. the coupled smoothing) is responsible for its benefit.
4.	The paper's clarity could be improved.
- The justification for some design choices relies on qualitative descriptions rather than empirical proof. For instance, the claim that the smoothing strength $s_i$ "allows the model to intelligently navigate the bias-variance trade-off based on data uncertainty" is presented without supporting evidence. The proof does have any impact of the way the model is used during inference.
- Core components of the methodology, such as the auxiliary and load-balancing losses, are relegated to the appendix. I would suggest to the authors to put it in the main paper. 
- Figure 1 explicitly labels the experts as “Many,” “Medium,” and “Few” shots, suggesting a strict specialization of each expert to a particular density region. However, the training mechanism does not appear to enforce or guarantee this one-to-one correspondence. Moreover, the paper does not provide any empirical evidence that such specialization actually emerges in practice. This raises a key question: if the “few-shot” expert is indeed exposed to only a limited number of samples, how can it learn a meaningful and generalizable representation?
5.	The related work section does not reflect the current state of the field. The majority of the cited works on imbalanced regression appear to be from before 2023. 
6.	I would like to see newer baselines but also CatBoost results. 
7.	I would suggest that authors reconsider the name of their paper as ‘Case’ will be hard to find online.

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the problem of "deep imbalanced tabular regression" and proposes a framework named CASE to mitigate it. The method consists of two main components: (1) a Coupled Adaptive Smoothing (CAS) module that performs input-space smoothing, and (2) a Mixture-of-Experts (MoE) module. The authors evaluate the method on 40 public tabular regression datasets from UCI, OpenML, Scikit-Learn, and REBAGG benchmarks.

### Strengths
- The paper provide a good and technically sound integration of many existing ideas, and form a practical design for real-world problems. 
- The paper is clear and easy to follow

### Weaknesses
### 1. Problem setup
I hesitate to frame “deep imbalanced tabular regression” as a brand-new research direction. (1) Imbalanced regression focuses on label distribution, where prior work aims to mitigate the mismatch between imbalanced training data and balanced test performance; (2) Tabular data introduces input heterogeneity, and improving it with deep networks still remains an open problem. As the authors' experiments also suggest, the main challenge still lies in the tree-based vs. deep-based performance gap. These two aspects (deep imbalanced regression vs. deep tabular regression) are inherently orthogonal, and the methods addressing them are largely independent (so is the proposed CAS module). Combining them as a single research direction feels more like overlapping two separate challenges than defining a genuinely new problem.

### 2. Method
The proposed framework contains a variety of components. Even though the integration sounds reasonable, none of them (weighted metric, input smoothing, or MoE) are technically novel, and the paper lacks sufficient ablation to clarify individual effect of each module or how they interact. Overall, the method feels more like a good engineering combination of existing ideas rather than novel contribution typically expected at a top-tier deep learning venue.

- It is unclear why smoothing the input in CAS is effective for imbalanced tabular regression. The mechanism is similar to MixUp [1], where mixing neighboring samples improves generalization by encouraging local smoothness. The paper does not convincingly explain why input-space smoothing would specifically benefit imbalance + tabular scenarios beyond the general regularization effect already provided by MixUp-style interpolation.
- The design of the auxiliary and load-balancing losses is conceptually unclear. My understanding is that the auxiliary loss uses the ground-truth frequency region as supervision, but the load-balancing loss, as defined in the manuscript, does not reflect frequency or density at all. It only enforces equal utilization across the three experts. If the authors believe that the load-balancing loss can automatically route low-frequency samples to a “low-shot” expert, this should be discussed and empirically verified, since the auxiliary loss is inherently biased toward majority regions. 

### 3. Experiment
The experimental comparison does not feel entirely fair. The authors mainly compare against deep imbalanced regression baselines, while the key challenge in deep imbalanced tabular regression still remains the handling of input heterogeneity. As a result, tree-based methods still show highly competitive performance. Including comparisons with recent deep tabular networks would make the experimental results more solid.

[1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and David Lopez-Paz. MixUp: Beyond Empirical Risk Minimization. ICLR, 2018.

### Questions
- I find Eq. (1) and Eq. (2) interesting, and the authors could strengthen the paper by providing more ablation studies to evaluate and showcase the effect of $\omega$. This would help solidify the technical contribution.
- The authors could also show how accurately the MoE router assigns samples to the correct regions.

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
3

### Summary
This paper focuses on tabular data, and introduces the Coupled Adaptive Feature–Target Smoothing with Density-Gated Mixture-of-Experts (CASE) method for imbalanced tabular regression. It integrates Coupled Adaptive Smoothing (CAS) and Density-Gated Mixture-of-Experts (MoE) modules. CAS adaptively calibrates features by coupling similarity in both feature and target spaces, while MoE promotes expert specialization over regions. During training, experts take the calibrated features as input; at inference, the learned experts operate on the original features for fast and accurate inference. The architecture uses an asymmetric dual-path design, where the gating network processes raw features and the experts operate on density-calibrated representations. Experiments across 40 tabular benchmarks show that CASE can provide the best average rank on balanced and imbalanced test sets, and improve few-shot accuracy without sacrificing many-shot performance.

### Strengths
+ Overall, the paper is clearly written, well organized, and easy to follow.  
+ This paper focuses on a less explored problem of imbalanced regression on tabular data. CASE is a method specialized for this problem that combines a novel adaptive smoothing that considers feature and target spaces with a density-gated MoE that weights the contributions of specialized experts.   
+ The CAS module is a novel smoothing mechanism that adaptively calibrates the feature space according to data density to stabilize feature representations in sparse, low-density regions. Calibrated features from CAS are utilized by expert networks within the density-​ated MoE.  The authors provide a connection to Vicinal Risk Minimization, where CAS is interpreted as an adaptive, density-aware extension. 
+The empirical evaluation involves 40 tabular benchmarks datasets, multiple baselines, ablations, and efficiency analyses. CASE can outperform SOTA methods on balanced test sets by attaining the top average rank. It can provide balanced predictions that enhance few-shot accuracy without significantly sacrificing many-shot performance. 
+ The Appendix has additional information on datasets, hyper parameter settings, ablation studies, analysis of computational complexity, feature visualizations, description of CASE components, analysis of novelty, and experimental results that help support the paper.

### Weaknesses
- CASE is based on known approaches, like VRM, MoE, density smoothing. The paper combines adaptive feature-target smoothing and a density-gated MOE. The motivation for certain architectural choices (e.g., fixed three experts, specific gating losses) could be justified more rigorously. The math formulation in Section 3 could benefit from an improved explanation of the intuition.
- In the experimental validation, the authors should provide a better interpretation of results on cross-dataset generalization and diversity.
- The CASE architecture can significantly increase computational cost. For instance, the training overhead is about 4 times that of baselines.This paper should contain a more detailed analysis and comparison  to SOTA methods of time and memory complexity (at training and test times).
- Their code is not made available, so there is a concern that the results in this paper would be difficult for a reader to reproduce.

### Questions
See my comments in weaknesses.

### Soundness
2

### Presentation
2

### Contribution
3
