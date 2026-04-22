# Adjusting Prediction Model Through Wasserstein Geodesic for Causal Inference

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 8

## Abstract
Causal inference estimates the treatment effect by comparing the potential outcomes of the treated and control groups. Due to the existence of confounders, the distributions of treated and control groups are imbalanced, resulting in limited generalization ability of the outcome prediction model, \ie, the prediction model trained on one group cannot perform well on the other group. To tackle this, existing methods usually adjust confounders to learn balanced representations for aligning the distributions. However, these methods could suffer from the over-balancing issue that predictive information about outcomes is removed during adjustment. In this paper, we propose to adjust the outcome prediction model to improve its generalization ability on both groups simultaneously, so that the over-balancing issue caused by confounder adjustment can be avoided. To address the challenge of large distribution discrepancy between groups during model adjustment, we propose to generate intermediate groups through the Wasserstein geodesic, which smoothly connects the control and treated groups. Based on this, we gradually adjust the outcome prediction model between consecutive groups by a self-training paradigm. To further enhance the performance of the model, we filter the generated samples to select high-quality samples for learning. We provide the theoretical analysis regarding our method, and demonstrate the effectiveness of our method on several benchmark datasets in terms of multiple evaluation metrics.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper addresses covariate shift between treated and control groups in observational causal inference. Instead of balancing covariates via reweighting/representation learning (which can cause over-balancing and remove predictive information), the authors propose to adjust the outcome prediction models themselves. They:

- Build an optimal transport plan between the empirical control and treated distributions and use the Wasserstein geodesic (displacement interpolation) to generate K intermediate “groups” µκ that smoothly connect the two.
- Perform gradual self-training along this path: starting from a supervised model trained on one endpoint (e.g., h0,0 on control), iteratively pseudo-label the next intermediate group and fit a new model h0,κ, until reaching the other endpoint (yielding h0,1). A symmetric process is used for h1.
- Use MC-Dropout uncertainty to filter the generated samples (keep the r fraction with lowest predictive std) to mitigate pseudo-label noise.
- Provide generalization bounds that adapt gradual domain adaptation theory (He et al., 2024) to their regression setting, bounding E(h0,1) and PEHE in terms of endpoint errors, the average inter-step Wasserstein distance Δ and finite-sample/model-complexity terms.
- Empirically evaluate on News, Twins, Jobs, and a synthetic suite with increasing group shift, showing competitive or state-of-the-art performance on most metrics, especially as confounding grows.

### Strengths
- **Originality and framing**
  Conceptual shift from "adjust covariates" to "adjust the prediction model" via a smooth geodesic path is a clear and appealing alternative to conventional balancing. This directly targets the over-balancing concern. Using OT-induced displacement interpolation to construct intermediate "domains" is principled and leverages known geometry of probability measures.

- **Methodological quality**
  The algorithmic pipeline is simple, modular, and practical: compute OT; generate intermediate groups; gradual self-training with uncertainty filtering. It can wrap around standard regressors. The uncertainty-based filtering is a reasonable, low-overhead safeguard against error amplification in self-training.

- **Theoretical and Empirical Support**
  The claims are well-supported by both theoretical analysis and extensive empirical evidence. The authors provide a theoretical upper bound on the treatment effect estimation error (Theorem 1), which solidifies the method's foundations. The experimental evaluation is comprehensive, covering multiple datasets, evaluation metrics, and a large number of strong baselines. The consistently superior performance of G-learner across different scenarios robustly demonstrates its effectiveness.

### Weaknesses
- **Scope and assumptions**
  The approach still relies on ignorability/overlap and does not address unobserved confounding. While this is standard for meta-learners, the paper’s emphasis on avoiding balancing could be read as implying robustness to confounding; a clearer articulation of identifiability limits would help.
  
- **Discrete vs. Continuous Covariates**
  The use of Wasserstein distance with an L2 cost function seems most natural for continuous covariates. It is not immediately clear how the method would perform with datasets dominated by discrete or categorical features. While the method appears to work well on the presented real-world datasets which likely contain mixed data types, a brief discussion on the applicability or potential adaptations for high-dimensional discrete covariate spaces would be beneficial.

- **Experimental gaps**
  - On News, G-learner's PEHE is worse than TARNet; the paper states best or competitive, but this discrepancy merits discussion and an ablation (e.g., how K, r, or representation choice affects PEHE on News).
  - Dataset breadth: adding IHDP/ACIC or a modern semi-synthetic benchmark would strengthen generality claims.
  - Computational cost: solving the OT problem can be heavy for large n.

### Questions
- **Necessity of OT geodesics:**
  How much of the gain comes from using the OT coupling versus any random or nearest-neighbor pairing for interpolation? Would performing OT in a learned latent space (e.g., via an autoencoder or TARNet trunk) further improve the realism of intermediate groups? Any preliminary results?

- **Computational Cost:**
  Could the authors please comment on the computational complexity of the proposed G-learner, particularly the optimal transport step?

- **Visualization of Model Adjustment:**
  The visualization of the generated intermediate data in Figure 2 is very effective. It would be highly insightful to also visualize the function learned by the outcome models (e.g., h_0,k) at each step k. For a simple 1D or 2D covariate space, plotting how the prediction surface smoothly deforms from the control group towards the treated group would provide a powerful visual confirmation of the "gradual model adjustment" concept. Is this something the authors have considered or could provide?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel method, `G-learner`, to address the over-balancing issue during alignment of treated and control groups in causal inference. The authors leverage the Wasserstein space and the concept of the Wasserstein barycenter to generate intermediate domains. Models are then progressively fine-tuned on these intermediate domains. Additionally, an uncertainty quantification technique is introduced to filter data during the fine-tuning stage. Theoretical analyses are provided, and extensive experiments are conducted to demonstrate the efficacy of the proposed approach.

### Strengths
1. The use of intermediate domains to generate auxiliary samples for alleviating the over-balancing issue is both creative and effective.
2. The research topic—improving causal inference—is highly relevant to the ICLR community.
3. The experimental results are clearly quantified and support the claims of the proposed approach.

### Weaknesses
1. To the best of the reviewer’s knowledge, some related works have explored using the Gromov-Wasserstein distance and unbalanced Wasserstein distance to account for local similarity and mini-batch sampling effect during alignment. Including such baselines (e.g., ESCFR-Pro [1] and CE-RCFR [2]) would strengthen the comparison and evaluation.

2. The paper states that the tables underline the second-best results, but these underlined results are not clearly visible.

3. Figure 3 should include uncertainty/error bars to better demonstrate the robustness of the proposed approach.

4. An ablation study evaluating the effectiveness of the data filtering component is recommended.

5. Is Eq. (9) a variant of barycentric projection? Please clarify how it is derived from Eq. (5).

6. In Figure 3, why does performance decrease as the number of intermediate groups increases? 

--- 
References:  
[1]. Proximity Matters: Local Proximity Enhanced Balancing for Treatment Effect Estimation  
[2]. CE-RCFR: Robust Counterfactual Regression for Consensus-Enabled Treatment Effect Estimation

### Questions
Please see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
To balance the distributions between treated and control groups while reducing the loss of predictive information, the authors propose to generate intermediate groups through the Wasserstein geodesic, which smoothly connects the control and treated groups. The authors present a theoretical analysis of this method and demonstrate its effectiveness across several benchmark datasets.

### Strengths
1. The authors provide a theoretical analysis of the estimation error for their proposed method.

2. The authors focus an important issue in causal learning, treatment effect, and provide a potential solution to address confounding bias.

### Weaknesses
1. The language quality of the submission is not very good, especially in the abstract, which fails to clearly explain the motivation behind the proposed method.

2. Lacking ablation experiments to verify the effectiveness of each module.

3. The authors should compare one or some relatively new algorithms.

### Questions
1. The paper's writing style needs improvement, especially in the abstract and introduction, which fail to clearly explain the research motivation. 

2. The novelty of this paper should be more clearly highlighted, enabling readers to easily understand the authors' contributions.

3. The authors should compare their work with some relatively new algorithms and conduct experiments on benchmark datasets, such as IHDP, for a more comprehensive evaluation.

4. This paper lacks a sensitivity analysis of the parameters to demonstrate the robustness of the proposed method.

5. The authors discuss the Optimal Transport in the related work section but fail to compare with it in the experiments.

6. Lacking ablation experiments to verify the effectiveness of each module.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
In this paper, the authors address an important limitation of existing causal inference methods that rely on balanced representation learning: the over-balancing issue that arises in outcome prediction models due to imbalances in the treatment and control groups. They propose generating intermediate groups through the Wasserstein geodesic to adjust the outcome prediction model between consecutive groups via a self-training paradigm. This approach is novel and well-motivated, and supported by both theoretical and experimental evidence.

### Strengths
1. The problem addressed in this paper, the over-balancing issue that arises in outcome prediction models due to imbalances in the treatment and control groups, is indeed a significant and common challenge in the real-world application of causal inference methods based on balanced representation learning. It has not been extensively studied in existing studies, and the authors' focus on this gap is highly relevant.

2. The authors' idea of incorporating optimal transport theory to address this challenge is both novel and well-motivated. By leveraging the elegant properties of the Wasserstein geodesic, the approach enables the learning of intermediate distributions along the optimal transport paths between treatment and control groups. This, in turn, allows the outcome prediction model to be adjusted between consecutive groups via a self-training paradigm. Its ability to incorporate the control group distribution into the treatment group model (and vice versa) is an excellent solution to the imbalance problem.

3. The theoretical proof of the proposed method is quite thorough, providing a solid foundation for the approach.

4. The authors have conducted experiments on many benchmarks and included hyperparameter analysis, effectively demonstrating the effectiveness of the proposed method.

### Weaknesses
1. The current assumption that $p_0$ and $p_1$ follow uniform (or other prior) distributions seems challenging to satisfy in real-world scenarios, where such prior knowledge may not be available. I am curious whether it would be possible to directly estimate $p_0$ and $p_1$ from the data, thereby relaxing this assumption.

2. In the generated data filtering process, the authors select only the samples with high confidence. However, low-confidence samples might not only arise from model errors, but could also be inherently more difficult to learn due to their scarcity, such as in the case of long-tail distributions. I wonder whether the cumulative removal of these samples in each step could lead to the situation where such samples are never adequately learned by the model.

3. The paper aims to address the imbalances in the treatment and control groups, but the baselines compared do not specifically address this issue. I suggest comparing the proposed method with other baselines that also aim to resolve such imbalances, such as DESCN [1], to provide a more relevant evaluation.

[1] Zhong, K., Xiao, F., Ren, Y., Liang, Y., Yao, W., Yang, X., & Cen, L. (2022, August). Descn: Deep entire space cross networks for individual treatment effect estimation. In Proceedings of the 28th ACM SIGKDD conference on knowledge discovery and data mining (pp. 4612-4620).

### Questions
Please see Weaknesses. It is a very interesting work, and I look forward to the authors' responses and further discussions.

### Soundness
3

### Presentation
4

### Contribution
4
