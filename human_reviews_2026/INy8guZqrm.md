# HUMOF: Human Motion Forecasting in Interactive Social Scenes

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Complex dynamic scenes present significant challenges for predicting human behavior due to the abundance of interaction information, such as human-human and human-environment interactions. These factors complicate the analysis and understanding of human behavior, thereby increasing the uncertainty in forecasting human motions. Existing motion prediction methods thus struggle in these complex scenarios. In this paper, we propose an effective method for human motion forecasting in dynamic scenes. To achieve a comprehensive representation of  interactions, we design a hierarchical interaction feature representation so that high-level features capture the overall context of the interactions, while low-level features focus on fine-grained details. Besides, we propose a coarse-to-fine interaction reasoning module that leverages both spatial and frequency perspectives to efficiently utilize hierarchical features, thereby enhancing the accuracy of motion predictions. Our method achieves state-of-the-art performance across four public datasets. The source code will be available at https://github.com/scy639/HUMOF.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a novel approach to address the problem of human motion forecasting in interactive environments.  It builds a hierarchical interaction representation, modeling interactions at multiple levels by combining explicit cues (e.g. inter-object distances) with learned features. Specifically, HUMOF captures high-level context and low-level details for both social interactions and scene contacts. A key innovation is the coarse-to-fine reasoning module, a multi-layer Transformer that progressively incorporates interaction features: early layers use high-level features, while later layers integrate fine-grained features. An adaptive DCT‐based rescaling further suppresses high-frequency components in early layers, encouraging a coarse-to-fine refinement of the motion prediction.

HUMOF obtained state-of-the-art results on two datasets with human-human and human-scene interactions (HIK. HOI-M3) as well as two datasets with human-scene interactions (GTA-IM, HUMANISE).

### Strengths
1. The paper is well written and explains deeply the architecture and how it works.
2. The paper models both human-human and human-scene interactions in a single framework, addressing a realistic scenario of interactive environments.
3. The use of multi-level representations (body-level vs joint-level for social cues, and multi-scale point clouds for scene context) is a strong idea. It balances global context and local detail effectively.
4. The injection of high-level features in early Transformer layers and finer details later (along with the DCT rescaling mechanism) is novel and well motivated by the ablation studies.
5. The authors compare their approach to a large set of baselines and include visualizations to support their claims.

### Weaknesses
1. HUMOF’s architecture is quite elaborate (multiple modules, Transformer layers, DCT processing). This complexity might make it hard to reproduce or tune and the paper should give more details in the appendices.
2. The authors note that existing datasets have few moving scene objects. Thus, HUMOF’s performance on truly dynamic environments (e.g. moving furniture or vehicles) is untested.
3. The baselines weren’t supposed to work with all the tested datasets, hence they had to be adapted. This raises a question of fairness in comparisons: it’s possible some methods could be improved with similar context. However, the large gaps suggest HUMOF’s advantage is likely genuine.
4. The runtime analysis in the Appendix was performed on only one dataset (HOI-M3).
5. The supplementary video lacks failure cases. Although some failure scenarios are described in the Appendix, it would be helpful to include them in the video as well.

### Questions
1. The paper uses a function $\phi (\cdot)$ to map distances to higher values for closer points. How sensitive is the performance to the choice of this mapping? Could a learned function improve results?
2. HUMOF 3D uses poses for all humans and a 3D point cloud of the scene. In real-world settings, these could be noisy or incomplete. Would the approach still work with noisy or incomplete inputs?
3. The runtime is around 43ms per inference for HOI-M3. Does this runtime change on other datasets? Have the authors tested HUMOF in an online setting? Is the latency competitive for real-time applications?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces HUMOF, a novel framework for human motion forecasting in complex social environments. HUMOF effectively integrates human kinematics, dynamics, spatial–temporal context, and interaction cues into a unified predictive model.

A key contribution is the Hierarchical Interaction Representation, which jointly captures human–human and human–scene interactions using both explicit distance features and learned semantic–geometric representations. To exploit this representation, the authors design a Coarse-to-Fine Interaction Reasoning Module, which improves motion prediction through two mechanisms:

Spatial hierarchy: High-level semantic features are introduced in early Transformer layers, while fine-grained geometric cues are progressively refined in later layers.

Frequency control: A DCT-based frequency modulation strategy prioritizes high-frequency motion components early in training and focuses on low-frequency refinements in later stages.

### Strengths
1. The paper is logically organized and easy to follow, with a coherent narrative from motivation to methodology and results.

2. The evaluation covers both multi-person interactive and single-person forecasting scenarios, including tests on unseen environments. The proposed method consistently outperforms baselines across all benchmarks.

3. The visualizations and supplementary videos are convincing — showing smooth, stable, and realistic motions with minimal jitter and drift compared to prior works.

### Weaknesses
No major weakness

### Questions
N/A

### Soundness
3

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
HUMOF presents a human motion forecasting method in social environments that takes into account both human-human as well as human-scene interactions. A DCT rescaling allows for controllability of the coarseness of the signal processing. The authors compare on two social scene datasets and on two human-scene interaction datasets and consistently outperform SOTA.

### Strengths
The problem of social human motion forecasting is highly relevant but relevantly under-explored. HUMOF is a complex method with many moving parts. However, the authors ablate the model relatively well. Utilizing DCT is a commonly used technique in motion forecasting. Using the rescaling for coarse-to-fine prediction in the context of human motion fc is novel and clever. The method description is mostly clear experiments have been conducted on two social scene datasets and two human-scene interaction datasets, demonstrating the methods effectiveness, while being parameter and inference time efficient.

### Weaknesses
The method is complex (as the task itself is complex) but I have some concerns about some of the model specifics:
(1) The utilized relative encoding is overly simplistic: I wonder if instead of just utilizing the point distance the method could utilize the geometric transformations in SE(3), for example as has been utilized in [1]. 
(2) While DCT works well for-single person motion, I wonder if it is limiting the human-object and human-human interaction quality. In the transformer, the tokens do not directly correspond to frames anymore, so temporal alignment over longer time frames might be hindered. The authors should evaluate this by showing either closer human-human (or human-object) interaction.
If the current datasets do not contain sufficient close person-to-person data, the authors should utilize other dyadic datasets, i.e Inter-X [2].

My second concern is with regards to the evaluation: the authors only evaluate directly comparing to GT (path error, pose error) - however, due to the complexity of the scene, multiple “answers” could be correct - I wonder if the authors have considered utilizing methods to compare the generated sequences on a distribution level, i.e. by utilizing FID, i.e. the authors could use the combined input-output sequence ($X^{1:H} \oplus \hat{X}^{H+1:H+T}$) to compare the distributions. 

[1] GTA: Geometric Transform Attention (ICLR 2024)

[2] Inter-X: Towards Versatile Human-Human Interaction Analysis (CVPR 2024)

### Questions
Did the authors measure human-object and human-human penetration? This feels like a natural form of evaluating this task.

### Soundness
2

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
This paper proposes HUMOF, a method for human motion forecasting in complex dynamic scenes that involve both human–human and human–scene interactions.
The approach introduces a hierarchical interaction representation that separately encodes high-level contextual and low-level geometric information, and a coarse-to-fine interaction reasoning module that injects these features into Transformer layers from semantic to geometric levels.
The framework combines Discrete Cosine Transform, Graph Convolutional Networks, and Transformer architectures.
Experiments on four public datasets (HIK, HOI-M3, GTA-IM, and HUMANISE) show that HUMOF achieves state-of-the-art performance.

### Strengths
1.	Comprehensive interaction modeling.

- Unlike prior works that focus on either human–scene or human–human interactions, this paper successfully integrates both within a unified framework. This joint modeling is meaningful and leads to noticeable performance improvements on dynamic scenes.

2.	Strong empirical performance.

- The method achieves consistent improvements across multiple datasets and evaluation metrics. The results demonstrate that the hierarchical representation and coarse-to-fine reasoning work effectively together.

3.	Practical contribution as a baseline.

- If the authors release the code as promised in the abstract, HUMOF can become a valuable benchmark for future research on interactive human motion prediction.

### Weaknesses
1.	Limited novelty in components.

Most elements of the architecture come from existing works (coarse-to-fine approach, distance-based interaction modeling, abstraction, ...). The contribution lies primarily in how these components are integrated, rather than in introducing a fundamentally new modeling paradigm.

2.	Lack of quantitative evaluation for multi-person inference.

The paper briefly demonstrates qualitative results for joint multi-person inference in Figure 4 but does not provide quantitative metrics or runtime analysis. Since multi-person prediction is highly relevant for real-world applications such as social robotics or crowd simulation, the absence of measurable evaluation limits the practical significance of this claim.

3.	High computational complexity for multi-agent prediction.

The proposed framework models each target person independently and computes pairwise interactions with every surrounding person. 
This design implies that when there are K individuals, the method requires roughly K² human–human interaction computations. 
Furthermore, because the scene abstraction (HSI) is recomputed for each target, the total cost scales linearly with K, resulting in O(K³) level complexity when both factors are considered. 
This raises concerns about the scalability and feasibility of real-time multi-agent forecasting.

4.	Issue in references.

The paper repeatedly cites prior works using informal author patterns such as “Jeong & etc., 2024” or “Xing & etc., 2025.” This format is inappropriate for a scientific paper and must be corrected to proper citation styles.

### Questions
Please see the weakness section

### Soundness
3

### Presentation
2

### Contribution
3
