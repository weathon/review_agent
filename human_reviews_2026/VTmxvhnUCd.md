# Online Versatile Incremental Learning: Towards Class and Domain-Agnostic Adaptation at Any Time

- Decision: Reject
- Scores: 4, 8, 4, 2

## Abstract
Continual learning enables vision systems to adapt to ever-changing data distributions. Despite significant advances, existing approaches fail to capture the seamless, concurrent transitions, a critical capability for real-world deployment. This work introduces $\textbf{Online VIL (Online Versatile Incremental Learning)}$, a novel scenario where class concepts and visual domains evolve simultaneously online without explicit boundaries. To better adapt to the challenges of such dynamic environments that more closely resemble real-world conditions, we propose a novel framework $\textbf{TopFlow}$, $\textbf{Top}$ology preservation with $\textbf{Flow}$ matching representation that contains two complementary mechanisms: $\textbf{Domain-agnostic Flow Matching (DFM)}$ and $\textbf{Global Topology Preservation (GTP)}$. DFM guides the model to have domain-agnostic representations by integrating the geodesic flow kernel into contrastive learning. In contrast, GTP maintains the global structure of the feature space without explicitly storing past examples. Our extensive experiments demonstrate that TopFlow effectively addresses the limitations of existing methods within the Online VIL scenario, achieving state-of-the-art performance in challenging Online VIL. The proposed methods suggest potential directions for building continual learning systems in realistic dynamic environment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a new continual learning scenario called Online Versatile Incremental Learning (Online VIL), designed to better reflect real-world settings where both semantic classes and visual domains shift unpredictably and without clear task boundaries. The authors introduce TopFlow, an approach combining Domain-agnostic Flow Matching and Global Topology Preservation. Extensive experiments on several benchmarks show TopFlow achieving state-of-the-art performance.

### Strengths
1. The paper tackles a widely acknowledged limitation in existing continual learning setups, that the unrealistic isolation of either class increments or domain increments. This motivates the definition of the Online VIL scenario.
2. The mathematical formulations of DFM and GTP are well provided, with notation, equations, and pseudo code.
3. The proposed method achieves SOTA performance against different online continual learning and domain incremental learning methods.

### Weaknesses
1. The review of online CL is broad, but the paper omits explicit discussion, comparison, or empirical benchmarking against Versatile Incremental Learning approaches.
2. The experimental setup appears to be insufficiently rigorous. Specifically: The results for iDigits with varying buffer sizes are not reported. Some commonly used datasets, such as DomainNet, are not included in the experiments, limiting the comprehensiveness of the evaluation. The comparison methods differ between Table 1 and Table 2, making the evaluation inconsistent.
3. TopFlow is implemented on ViT, while several comparison methods (e.g., CBA and OnPro) are based on ResNet. Since ViT architectures generally outperform ResNet under similar conditions, the comparisons in Table 1 and Table 2 may be biased or unfair. A fairer comparison would require either architecture alignment or additional analysis to account for architectural differences.
4. This work[1] proposes methods specifically for online DIL, very closely related to the core challenge addressed by TopFlow. It should be cited and discussed in Related Work.
[1] Gunasekara N, Gomes H, Bifet A, et al. Adaptive online domain incremental continual learning

### Questions
Please refer to Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Online Versatile Incremental Learning (Online VIL), a practical setting where both domains and classes change in a chaotic manner. Under this challenging scenario, the authors introduce two approaches to achieve efficient continual learning: (1) Domain-Agnostic Flow Matching (DFM), which extends contrastive learning to incorporate not only last-layer but also intermediate-layer features, and (2) Global Topology Preservation (GTP), which maintains the feature space through prototypes updated via online clustering. The authors evaluate their method under the designed setup on iDigits, CORe50, and CLEAR100, showing that their approach outperforms existing baselines.

### Strengths
1. The proposed Online VIL setting is convincing. It assumes continuous enviroment changs, unpredictable domain shifts, and the emergence of new classes, which are conditions that closely reflect real-world scenarios.
2. The proposed methodology is novel and well motivated. In particular, the contrastive operation between $h_{n,(i)}^{*}$ and $h_{n,(i)}$ in DFM is very interesting.
3. The authors provide extensive experimental evidence supporting the effectiveness of their approach.

### Weaknesses
1. The connection between the proposed method and the emphasized setup should be strengthened.
   1. The proposed approach does not appear to specifically target the Online VIL setting; TopFlow could also be effective under standard VIL conditions.
   2. Therefore, the authors should demonstrate that TopFlow remains effective and superior to prior methods when evaluated under the same VIL settings as previous works.
2. While the authors provide sufficient explanation that Online VIL is closer to real-world conditions, the motivation for why such a setting is necessary within this research field requires further clarification.
   1. Although the proposed setup is realistic, it would be valuable to show how and why existing methods struggle under this scenario. In the current version, it is not evident that previous methods suffer significantly in Online VIL.
   2. Without such clarification, Online VIL may appear to be an excessively challenging task at the research level. From a reviewer’s perspective, standard VIL already presents substantial difficulty, and VIL works that performs well in that setting could likely extend to real-world applications with additional engineering.
3. Including a discussion on test-time adaptation (TTA) studies would further improve the paper. The reviewer suggests that the following works be included in the 'Related Work' section.
   1. Continual learning and test-time adaptation differ primarily in whether label information is available.
   2. In the TTA field, several studies have already explored settings and methodologies similar to those addressed in this paper, including:
      1. Prototype-based adaptation [1,3]
      2. Continually changing environments [2]
      3. Sudden domain shifts and online clustering [3]
      4. Contrastive learning approaches [4,5]
   3. Additionally, the insight in Figure 2 reflects observations that have been recognized in prior literature [6,7,8]. However, the reviewer acknowledges that these findings are revisited to motivate the proposed method. Citing the following works would be beneficial.



[1] Test-Time Classifier Adjustment Module for Model-Agnostic Domain Generalization NIPS 2021

[2] NOTE: Robust Continual Test-time Adaptation Against Temporal Correlation NIPS 2022

[3] Test-time Adaptation in the Dynamic World with Compound Domain Knowledge Management RA-L 2023

[4]  Contrastive Test-Time Adaptation CVPR 2022

[5] Robust Mean Teacher for Continual and Gradual Test-Time Adaptation CVPR 2023

[6] Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net ECCV 2018

[7] Revisiting Batch Normalization For Practical Domain Adaptation arXiv 2016

[8] Learning Transferable Features with Deep Adaptation Networks ICML 2015

### Questions
1. In Equation (6), $h$ in $h(x)$ appears to refer to the intermediate feature described in Section 3.2. Could the authors clarify this point?
2. Can the authors confirm that the features from the selected layers in Table 5 are used as $H_{n}$? Adding an explanation around Line 457 would be beneficial.
3. Including $h_{l,(j)}$ in $H_{(i)}^{-}$  is intuitively justified. However, the contribution of $h_{n,(i)}$ is less straightforward. Could the authors provide an additional ablation study?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Online Versatile Incremental Learning (Online VIL), which simulates real-world online adaptation where both class and domain distributions shift unpredictably. The paper also introduces TopFlow, a framework that incorporates two principal mechanisms: Domain-agnostic Flow Matching (DFM), designed to induce domain-invariant representations through geometry-guided contrastive loss, and Global Topology Preservation (GTP), which preserves the global structure of the demand space online without requiring explicit data replay. Experiments demonstrate that the proposed method outperforms previous ones.

### Strengths
1. The introduction of Online VIL represents a step towards more realistic continual learning, simulating previously neglected, unpredictable class and domain shifts.
2. The experimental results demonstrate clear and consistent improvements across the iDigits, CORe50, and CLEAR100 datasets, with additional experiments on the challenging real-world datasets TinyImageNet and UCF-101.

### Weaknesses
1. The paper lacks discussion and comparison with the most recent relevant work, and contains scarcely any references from 2025/2024. The SOTA in online incremental learning algorithms should be compared to highlight the advancement of the proposed TopFlow.
2. Figure 2 is core to illustrating the justification for layering (domain vs. class focus), but the effect of GTP on feature topology is described in text only. Explicit t-SNE or clustering visualizations showing the before-and-after topology with and without GTP would better clarify this critical component and its actual effect on catastrophic forgetting and representational drift.
3. While the mathematical derivation of the DFM loss is grounded in geodesic flow kernel theory, key aspects remain underspecified for reproducibility and theoretical clarity. The normalization and batchwise projection steps in DFM are described at a high level, but implementation subtleties are omitted. For instance, how exactly the projection onto the common singular space U is computed from intermediary/final layer features and maintained for perturbed samples (particularly with only local observations) is not concretely specified.

### Questions
1. The paper seems to be an extension of VIL [1], expanding upon VLL to form an online VIL. What technical challenges does this extension to online learning present? How does it differ from other online learning methods? How does the paper demonstrate that its contribution is more than merely incremental?
[1] Versatile Incremental Learning: Towards Class and Domain-Agnostic Incremental Learning. ECCV 24.
2. How would TopFlow adapt to settings where domain or class axes are not explicitly separable, e.g., datasets with only a single domain (CIFAR-100) or severe class imbalance (ImageNet-R)? Is any protocol or modification suggested, or is this scenario intentionally outside scope?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper addresses the challenge of continual learning under realistic, dynamic conditions where both class concepts and visual domains evolve concurrently. The authors introduce the Online Versatile Incremental Learning (Online VIL) setting to capture this scenario and propose TopFlow, a framework designed to handle such seamless transitions. TopFlow integrates two components: Domain-agnostic Flow Matching (DFM) — promotes domain-invariant representations using a geodesic flow kernel within contrastive learning. Global Topology Preservation (GTP) — maintains the global structure of the feature space without storing past samples. Experiments show that TopFlow achieves state-of-the-art results under the Online VIL setting, suggesting its potential for real-world continual learning applications.

### Strengths
1. This paper introduces an important problem — Online Versatile Incremental Learning (Online VIL) — a new continual learning scenario that models unpredictable and heterogeneous shifts in an online setting.

2. Extensive experiments demonstrate that TopFlow achieves state-of-the-art performance on multiple challenging Online VIL benchmarks, outperforming existing methods.

3. The work provides meaningful insights and promising directions for designing robust continual learning systems that can adapt effectively to complex, heterogeneous, and unpredictable environment shifts.

### Weaknesses
1. The authors propose to address the Online VIL (Online Versatile Incremental Learning) problem, but the experiments do not demonstrate the method’s effectiveness in a truly real-world online setting. Validation on a real-world application, such as autonomous driving, would make the work more compelling and impactful.

2. Most of the baselines compared in the paper, including the latest ones, are from 2024. It is recommended that the authors include more recent 2025 baselines to provide a stronger and up-to-date comparative evaluation.

3. According to ICLR guidelines, the paper should include an Ethics Statement, a Reproducibility Statement, and a declaration regarding the Use of Large Language Models (LLMs), if applicable. Adding these sections would improve the completeness and transparency of the submission.

4. It is recommended that the authors include more visualization analyses to clearly illustrate why the proposed DFM and GTP components effectively address the two main challenges of Online Versatile Incremental Learning. Such visualizations would enhance the interpretability and intuitiveness of the method.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
