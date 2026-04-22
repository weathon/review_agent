# Fisher Divergence for Attribution

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 6, 6

## Abstract
Feature attribution methods aim to explain model predictions by identifying input features that are most influential to the output. While perturbation-based methods are intuitive and widely used, they often rely on restricted or discretized perturbation spaces, limiting their ability to capture the complex dependencies in high-dimensional data. In this work, we propose a novel framework that defines the perturbation space as a continuous-time stochastic differential equation (SDE), enabling a general, unbounded formulation of the perturbation space. This formulation significantly increases the expressiveness of the perturbation space while introducing new optimization challenges. To address this, we derive a theoretical connection between the KL divergence and the Fisher divergence under general SDEs, and further establish that the time derivative of mutual information between perturbed and original inputs is governed by the Fisher divergence. These results allow us to simplify the attribution objective and compute pointwise information as feature importance scores. Empirical results on large-scale image classification tasks show that our method produces sharper, more coherent, and better localized attribution maps compared to existing approaches, as demonstrated by both qualitative visualizations and quantitative evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper models attribution as exploration along a continuous-time SDE and develops a theoretical link that ties the time-varying mutual information to the Fisher divergence. The method defines a pixel-wise integral objective and adapts the perturbation depth per pixel via a learned stopping time. Experiments cover standard image-classification benchmarks with comparisons to gradient-based and information-bottleneck baselines. Results are reported on insertion/deletion DAUC, EHR, and Box-Ratio. Consistent gains are reported across these benchmarks.

### Strengths
1. The paper is easy to follow, and the motivation is clear.

2. Rather than fixing a hand-crafted perturbation set, the method defines an unconstrained perturbation space via an SDE under a constraint, which is a new angle for perturbation-based attributions.

3. The core objective is theoretically supported, connecting the time derivative of the KL divergence to Fisher divergence.

### Weaknesses
1. The presentation quality of this paper is limited, and it is suggested to clearly define all notions and settings.

2. The experiment part lacks the necessary detail and analysis. Most experiments provide only one or two paragraphs without a clear setup or further discussion. For example, for the results in Table 1, the authors should specify the fraction of pixels modified per step for DAUC and include at least one full insertion/deletion curve rather than only AUC/DAUC scalars. For Table 2, the authors should explain how they use segmentation masks to evaluate attribution, including thresholding and region selection, and they should state whether this evaluation aligns with or differs from the Box-Ratio setting in Table 3.

3. The metric selection and baseline coverage do not provide sufficient justification and therefore weaken empirical solidity.

- Because the proposed method is perturbation-based, the authors should include standard perturbation explainers such as Occlusion/Mask variants, RISE, or model-agnostic LIME/SHAP for images, or justify their omission.

- The sanity check in Figure 1 shows only the proposed method, which limits interpretability; the authors should compare other methods under the same parameter-randomization setting.

- The authors should explain their choice of ground-truth localization metrics. Why do the authors not consider existing attribution metrics that also use ground truth annotations such as Pointing Game or DiFull/DiPart?

- Since the insertion/deletion benchmark can cause input distribution shift, the authors are expected to show how this problem is mitigated or include results using ROAR/ROAD.

- Since this work claims an efficient framework, the authors are expected to report computational efficiency.

### Questions
1. What is the exact DAUC schedule?
2. For EHR/Box-ratio, what thresholds are used? How do the authors explain that the results based on bounding box are better than those based on segmentation?
3. Why omit Occlusion/RISE and model-agnostic perturbation baselines (LIME/SHAP)? 
4. Why does the sanity check (Fig. 1) not compare other methods? 
5. Why is the efficiency comparison missing?

### Soundness
1

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
3

### Summary
This paper introduces a novel framework for feature attribution, leveraging the continuous-time stochastic differential equations (SDEs) to define a perturbation space. Unlike traditional attribution methods that use fixed or restricted perturbation spaces, the authors propose using SDEs to capture more realistic and complex feature variations. By using Fisher divergence and mutual information, the framework can be efficiently optimized and provide principled attribution scoring. The experiments show that this approach gives better attribution maps both visually and quantitatively on standard image classification tasks.

### Strengths
- This paper is well-written and well-organized.
- This paper proposes to define an unconstrained perturbation space using continuous-time Stochastic Differential Equations (SDEs), which is a more general and principled modeling framework than prior discretized or fixed-noise approaches
- The work has extensive qualitative and quantitative experiments demonstrating it outperforms the baseline methods. The empirical results on large-scale image classification tasks show that the method produces sharper and better localized attribution maps.

### Weaknesses
- The paper claims that the proposed method requires non-trivial computational resources for training, but does not mention or evaluate the computations required to compute the attributions. From my point of view, the method involves simulating continuous-time diffusion processes for each input, along with the computation of Fisher divergence across time steps, which seems to require multiple forward and backward passes, and the paper does not provide the efficiency comparisons.
- The paper does not explicitly state what *pretrained* diffusion model they used. Does one pretrained model generalize to all the tasks? More information or experiments should be included.
- Is it DeepSHAP or DeepLIFT in Sec 5.2 (L392)? It seems DeepSHAP is not included in the baselines. How does the proposed method compare to the more commonly used Shapley-based methods?
- There are some missing information regarding the experiments. For example, what are the reason to choose the baselines in the paper? What is the trade-off hyperparameters $\beta$, and how does it affect the results?

### Questions
See Weaknesses

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a novel information-theoretic framework for feature attribution based on modeling the perturbation process as a stochastic differential equation (SDE). Instead of using discrete or fixed perturbations. The key insight is that the time derivative of mutual information (MI) between the noisy input and the original input can be expressed using the Fisher divergence.
Extensive experiments on ImageNet show that the proposed method produces sharper and more coherent saliency maps, outperforming existing attribution methods (e.g., IBA, IG, DeepLIFT) across multiple metrics (DAUC, EHR, Box-Ratio), while passing sanity checks for robustness and consistency.

### Strengths
1. The paper establishes a solid connection between diffusion dynamics and information theory. By showing that the time derivative of MI equals a Fisher divergence term, it unifies several previous heuristics under a principled formulation.

2. Modeling perturbations as an SDE allows the method to capture a continuum of noise intensities, offering a richer and more stable attribution mechanism than discrete perturbation sampling.

### Weaknesses
1. The quality of the score approximation depends heavily on the pretrained diffusion model, which may not always align with the classifier’s domain distribution.

2. Experiments focus solely on image classification. The generalization to other modalities is not validated, especially those with discrete input space such as text.

3. Integrating score differences along a diffusion trajectory is still computationally expensive at high resolution.

4. While the method’s improvements are consistent and theoretically grounded, the paper does not include ablation studies isolating the contribution of the SDE-based perturbation.

### Questions
1. How robust is the method to the choice of the pretrained diffusion (score) model? Have the authors tested different score models to assess distributional sensitivity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In feature attributions, input features are perturbed to assess their importance to a classifier output. This paper proposes to model perturbed features with a stochastic differential equation, motivating an optimization problem based on the mutual information. Implementation details are derived with theoretical justifications. Empirically, the proposed method outperforms baseline attribution methods in terms of the deletion/insertion metric and alignment with image bounding boxes.

### Strengths
- This paper proposes a novel approach for perturbing input features for feature attribution. The proposed approach is theoretically grounded in stochastic differential equation, while the empirical implementation is connected to recent advances in diffusion models.
- Although multiple implementation details deviate from the original optimization problem in Equation (3), all of them are well justified.

### Weaknesses
- The proposed method relies on hyperparameters such as $\beta$ in Equation (16) and $K$ when discretizing the Fisher-MI lower bound. It is unclear how one can tune these hyperparameters given an unseen sample to be explained.
- Also, it is unclear how sensitive the proposed method is to changes in $\beta$ and $K$.
- The proposed method assumes that the pretrained diffusion model $s_{\theta}$ can approximate the ground-truth score well. This can limit the scope of the proposed method to domains where powerful diffusion models have been pretrained.

### Questions
- How does one tune $\beta$ and $K$ for an unseen sample to explain?
- How sensitive is the proposed method to changes in $\beta$ and $K$?
- How sensitive is the proposed method to the usage of different score-based diffusion models $s_{\theta}$?
- Can you comment on how well the proposed method is expected to work when there is gap between $s_{\theta}$ and $s_t$?

### Soundness
3

### Presentation
3

### Contribution
3
