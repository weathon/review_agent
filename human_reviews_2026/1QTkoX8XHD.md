# Efficient Facial Landmark Detection via Prior Knowledge-Guided Agents

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
We present a highly efficient, agent-based framework for facial landmark detection that prioritizes model compactness and computational efficiency over maximum accuracy. Unlike conventional approaches that rely on large, fully supervised models, our method assigns each agent to a specific landmark, enabling it to infer its position solely from local observations and prior knowledge without explicit location awareness or inter-agent communication. Prior knowledge is modeled in two embedding spaces—feature and coordinate—using class-conditional Gaussian distributions. Agents navigate by minimizing deviations from these priors via a lightweight policy network. To enhance representation learning, we introduce a proximity-weighted contrastive learning strategy that incorporates spatial proximity into the training objective. A multi-stage detection strategy further reduces redundant computation by detecting sub-landmarks relative to core landmarks. While our method produces slightly higher normalized mean error than state-of-the-art (SoTA) methods, it achieves over $16\times$ and $41\times$ improvements in space and time complexities, respectively, compared to the SoTA lightweight model, running at $4.19$ and $1.29$ frames per second on an i5 CPU (2.5 GHz) for the COFW and 300W datasets, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel, agent-based framework for facial landmark detection that departs from conventional global regression approaches. Each agent is assigned to a specific landmark and navigates the image using only local observations and prior knowledge modeled in dual (feature and coordinate) embedding spaces. The method employs several strategies for efficiency, including a proximity-weighted contrastive loss, a multi-stage detection scheme, and a delayed decision algorithm. While the method's accuracy, measured by NME, is substantially lower than state-of-the-art models, it achieves a dramatic reduction in computational and spatial complexity, making a case for its use in resource-constrained, real-time applications.

### Strengths
1. The core paradigm is innovative and thought-provoking. Reformulating landmark detection as a distributed navigation task for independent agents is a non-trivial and intellectually interesting contribution. It opens up a new and potentially fruitful research direction beyond dense prediction with large, monolithic models.
2. The efficiency gains are compelling and quantitatively significant. The paper's most convincing argument is its extreme lightweight nature. The reported >16x and >41x reductions in space and time complexity compared to a modern lightweight baseline are hard, undeniable metrics that strongly support its potential for embedded deployment.

### Weaknesses
Prohibitive degradation in accuracy severely limits practical utility. The compromise on precision is arguably too great. The NME on 300W (9.36) is approximately three times higher than current SOTA methods (~2.8-3.3). This level of inaccuracy renders the method unsuitable for most real-world applications requiring precision (e.g., high-fidelity face recognition, medical analysis), significantly diminishing the value of its efficiency.

### Questions
1. Fundamental Trade-off: Is the significant accuracy loss an inherent limitation of the agent-based paradigm itself, or can it be substantially mitigated with more sophisticated network designs, training strategies, or prior knowledge modeling? What is your estimate of the method's ultimate accuracy ceiling?
2. Generalization of Priors: Your prior knowledge is modeled as class-conditional Gaussians derived from the training set. Does this mean the method will fundamentally fail on faces with atypical morphology (e.g., extreme expressions, facial anomalies) not well-represented in the training data? This poses a serious limitation to its generalizability.
3. Convergence and Failure Modes: Does the hopping policy guarantee convergence? Please provide an analysis of failure cases, such as agents getting stuck in loops, oscillating between points, or failing to locate occluded landmarks. The reliability of the search process is a critical concern that is not adequately addressed.

### Soundness
2

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
4

### Summary
The paper proposes an **agent-based framework** for facial landmark detection where **each agent localizes a single landmark** using only **local multi-scale observations** and **prior knowledge** modeled as class-conditional Gaussians in **feature** and **coordinate** embedding spaces. Agents iteratively “hop” using a lightweight policy (dirPolNet + distance-quantized step size) until a deviation criterion from priors is satisfied. Training leverages a **proximity-weighted contrastive loss** to encode spatial proximity, and runtime efficiency is boosted via a **two-stage** and **cascaded** detection strategy (core landmark → sub-landmarks with RelCoordNet) plus a **delayed decision** mechanism to avoid redundant recomputation. Experiments on **COFW** and **300W** show substantially lower compute/params (≈ **577k** params; **21–29 MFLOPs**) and CPU FPS up to **4.19** (COFW) / **1.29** (300W), at the cost of higher NME versus SOTA.

### Strengths
* **Efficiency first:** ~**577k** parameters and **21–29 MFLOPs** with concrete CPU FPS; strong engineering story for edge inference.
    
* **Clear mechanism:** Dual priors in feature/coordinate spaces with a principled deviation criterion and a small policy network.
    
* **Training insight:** **Proximity-weighted** contrastive objective aligns embeddings with spatial closeness, improving robustness to occlusion/noise.
    
* **System tricks that matter:** Two-stage refinement, cascaded (core→sub) detection, and delayed decision to prune redundant compute.

### Weaknesses
* **Accuracy trade-off:** NME notably higher than SOTA (esp. 300W, aggravated on jawline); causes and bounds could be probed more.
   
* **Stats & variance:** No CIs or multi-seed std; several deltas vs. strong baselines might be within noise without variance reporting.
    
* **Limited stress testing:** Few controlled perturbations (Gaussian blur, JPEG, motion blur, occlusion masks) to validate the “local-observation + priors” claim under adverse conditions.
    
* **Ablation scope:** Good focus on λ_ft schedule; less on **policy design** (dirPolNet architecture/targets), **threshold schedule**, and **choice of Gaussian priors vs. richer models**.
    
* **Interpretability:** Nice qualitative plots, but no **quantitative** measure that agents consistently specialize by region/frequency or that cascades reduce token/patch evaluations by X%.

* **Missing References:** Several works should be included. Particularly, heatmap-based landmark detection approaches. E.g.,

- Robinson, J. P., Li, Y., Zhang, N., Fu, Y., & Tulyakov, S. (2019). Laplace landmark localization. In Proceedings of the IEEE/CVF international conference on computer vision (pp. 10103-10112).

Though several others.

### Questions
1. **Statistical robustness.** Please report **mean±std over ≥3 seeds** (or bootstrap CIs) for NME and duration on COFW/300W; do the efficiency–accuracy trade-offs hold?
    
    16871_Efficient_Facial_Landmar
    
2. **Controlled degradations.** How does NME change under blur (σ grid), JPEG (Q grid), motion blur, and synthetic occlusion masks? Does proximity-weighted contrastive training confer robustness?
    
    16871_Efficient_Facial_Landmar
    
3. **Policy ablations.** What is the impact of: (i) different direction sets (e.g., 16-way), (ii) continuous step sizes vs. quantized HDst, and (iii) removing λ_ft/θ_d scheduling?
    
    16871_Efficient_Facial_Landmar
    
4. **Priors.** Why diagonal Gaussians? Would **mixture** priors or **low-rank covariances** help multi-modal landmark appearance/pose? Any cost increase?
    
    16871_Efficient_Facial_Landmar
    
5. **Cascade efficiency.** Can you quantify wall-clock and FLOP savings from the cascaded detection (core→sub) relative to parallel independent agents?
    
    16871_Efficient_Facial_Landmar
    
6. **Generalization.** Any results on non-frontal or in-the-wild video frames (e.g., low-light, motion) to demonstrate edge robustness?
    
    16871_Efficient_Facial_Landmar

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel, agent-based framework for facial landmark detection, prioritizing computational efficiency and model compactness for edge devices. The framework involves independent agents localizing landmarks, guided by learned prior knowledge modeled as Gaussian distributions in both feature and coordinate spaces. While explicitly trading accuracy for efficiency, the experiments show significantly higher NME than existing state-of-the-art methods.

### Strengths
1. This paper novelly proposes dual feature and coordinate priors and a proximity-weighted contrastive loss (PWConLoss).
2. The framework shows exceptional model compactness and low computational cost, achieving 16.8× lower space complexity and 41.1× lower time complexity than the best lightweight baseline.
3. The concept of using completely independent agents reframes the landmark detection problem from a single large inference to a collection of lightweight, parallelizable searches.

### Weaknesses
1. The claim of "real-time CPU inference" is misleading. According to Table 2, the highest FPS on CPU is only 4.19, far from the common "real-time" definition (>20 FPS).
2. Severe accuracy sacrifice. According to Table 1, the proposed framework's NME is almost three times larger than that of the state-of-the-art baseline.
3. The experiments lack comparison with other real-time methods, e.g., MediaPipe Face Mesh.

### Questions
Please refer to the weakness part.
1. Please clarify the "real-time CPU inference" claim.
2. Is there any idea or plan to improve the accuracy?
3. Could you include some comparison with other real-time methods?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an agent-based framework for facial landmark detection that emphasizes computational efficiency and model compactness over state-of-the-art accuracy. Each landmark is assigned to an independent agent that localizes its target through local observations and learned priors, without inter-agent communication. The prior knowledge is modeled as class-conditional Gaussian distributions in both latent feature and coordinate embedding spaces. A proximity-weighted contrastive loss (PWConLoss) is further introduced to improve the robustness of feature embeddings by weighting positive samples according to their spatial proximity. The framework also employs multi-stage and cascaded detection strategies to enhance efficiency. Experiments on COFW and 300W demonstrate notable reductions in parameters (16.8×) and FLOPs (41.1×) compared with lightweight baselines, highlighting a strong trade-off between accuracy and efficiency.

### Strengths
Exceptional Efficiency: The most notable strength lies in the substantial improvement in computational and memory efficiency. Achieving over a 40× reduction in FLOPs and a 16× reduction in parameters compared with a lightweight state-of-the-art model (PoPos) is highly impressive, making the method exceptionally practical for real-world deployment on edge devices.

Clarity and Presentation: The paper is written with excellent clarity. The authors effectively present a complex system with well-structured explanations and thorough mathematical derivations.

### Weaknesses
Significant Accuracy Drop: Despite the strong efficiency gains, the accuracy falls far below the competitive range, making the method impractical for accuracy-sensitive applications. Efficiency improvements lose significance when core task performance is not maintained.

Unclear Accuracy–Efficiency Trade-off: The trade-off between accuracy and efficiency is not convincingly analyzed. Comparisons should control for a single variable—e.g., reducing parameters in existing models or increasing those of the proposed one—to fairly assess the balance.

High Complexity and Limited Ablation: The framework integrates multiple heuristic components (three networks, two-stage and cascaded strategies, delayed-decision buffer, and several hyperparameters), making it difficult to analyze, tune, and reproduce. Its performance likely depends heavily on these design choices. Only λft is ablated, with no analysis of other major modules.

Unexplained Parameters: Certain parameters, such as the constant 0.025 in Equation (4), are introduced without justification.

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2
