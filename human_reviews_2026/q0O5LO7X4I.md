# Adaptive Mixture of Disentangled Experts for Dynamic Graph Out-of-Distribution Generalization

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 4

## Abstract
Dynamic graph out-of-distribution (OOD) generalization has drawn an increasing amount of attention in the research community, given its wide applicability in real-world scenarios. Existing methods typically employ a fixed-architecture design to extract invariant patterns. However, there may exist evolving distribution shifts in dynamic graphs, leading to suboptimal performance of fixed-architecture designs. To address this issue, we propose a novel adaptive-architecture design to handle evolving distribution shifts over time, to the best of our knowledge, for the first time. The proposed adaptive-architecture design introduces an adaptive mixture of architecture experts to capture invariant patterns under evolving distribution shifts, which imposes three challenges: 1) How to detect and characterize evolving distribution shifts to inform architectural decisions; 2) How to dynamically route different expert architectures to handle varying distribution characteristics; 3) How to ensure that the adaptive mixture of experts effectively discovers invariant patterns. To solve these challenges, we propose a novel **Ada**ptive **Mix**ture of Disentangled Experts (**AdaMix**) model to adaptively route architecture experts to varying distribution shifts and jointly learn spatio-temporal invariant patterns. Specifically, we propose a spatio-temporal distribution detector to infer evolving distribution shifts by jointly leveraging historical and current information. Building upon this, we develop a prototype-guided mixture of disentangled experts that adaptively routes experts with disentangled factors to different distribution shifts. Finally, we design a distribution-aware intervention mechanism that discovers invariant patterns based on expert selection of nodes. Extensive experiments on both synthetic and real-world datasets demonstrate that our proposed **AdaMix** model significantly outperforms state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenge of dynamic graph learning under evolving distribution shifts. The authors propose AdaMix, an adaptive MoE framework that dynamically routes nodes to different GNN experts based on spatio-temporal distribution characteristics. The model includes a memory-augmented distribution detector, prototype-guided disentangled experts, and a distribution-aware intervention mechanism. Experiments on several real-world and synthetic dynamic graph datasets are reported to show improved OOD generalization compared to existing methods.

### Strengths
1. The proposed adaptive mixture formulation is interesting and aligns with the idea of conditional computation in dynamic systems.

2. Experimental results show consistent improvements over several baselines, including recent OOD graph learning models.

3. The inclusion of ablation studies and visualizations helps demonstrate the contribution of individual components.

### Weaknesses
1. Novelty is limited. The framework closely follows the structure of prior dynamic OOD works with mostly terminological changes (more like a combination of DIDA, SILD and EAGLE).

2. The “adaptive architecture” claim is somewhat overstated. The architecture is fixed after training, and only expert weights are adaptively combined during inference.

3. The motivation for modeling distribution shifts via expert routing, rather than latent environments or causal factors, is underdeveloped and lacks theoretical grounding.

4. The mathematical formulation remains descriptive rather than rigorous. The objective functions and routing updates are introduced heuristically without clear derivation.

5. The experiments do not provide sufficient diagnostic evaluation (e.g., temporal shift severity, expert specialization visualization) to validate the “adaptive” behavior claimed.

6. Writing style is occasionally redundant and derivative, with repeated phrases and limited conceptual clarity in the theoretical section.

### Questions
1. How is the proposed adaptive routing different in principle from a standard MoE gating mechanism conditioned on node embeddings?

2. During inference, does the model truly change its computation path, or merely reweight pre-trained experts?

3. What ensures that each expert captures a distinct distributional pattern rather than overlapping ones?

4. How sensitive is AdaMix to the number of experts? Were there cases where performance degraded with more experts?

5. Could the observed performance gains stem from increased model capacity rather than intrinsic adaptability to distribution shifts?

6. How does the proposed intervention mechanism differ from data augmentation or cross-domain mixing used in prior OOD graph works?

### Soundness
2

### Presentation
3

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
This paper investigates dynamic graph representation learning under evolving distribution shifts and introduces an adaptive-architecture framework, AdaMix, designed to handle such changes over time. The proposed approach employs a mixture of architecture experts, guided by a spatio-temporal distribution detector, a prototype-based expert routing mechanism, and a distribution-aware intervention module to capture invariant spatio-temporal patterns. The authors identify and address three main challenges: detecting evolving shifts, dynamically routing experts, and ensuring effective invariant learning. Experimental results on synthetic and real-world datasets are reported to demonstrate the method’s performance compared with existing approaches.

### Strengths
1. The problem studied in this paper is important.

2. The experimental datasets are sufficient.

3. The work has a certain degree of theoretical support.

### Weaknesses
1. Some techniques are applied too directly without sufficient explanation. For example, why is invariant pattern modeling performed in the spectral domain?

2. The comparison methods in the experiments are not up to date (latest from 2023), lacking evaluations against GraphMoE-type approaches such as GMoE and GraphMETRO.

3. Considering that the proposed method follows an ensemble learning paradigm and incorporates more GNN encoders than the baselines, the performance gains on real-world datasets are relatively small—mostly within a 1% range. This raises the question of whether such limited improvement justifies the increased model complexity.

4. The method performs better on synthetic datasets but only moderately on real-world ones. This discrepancy suggests that the paper’s assumptions might be overly idealized and not well aligned with the characteristics of real-world data distributions.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper aims to address the challenge of dynamic graph representation learning under evolving distribution shifts, a limitation of existing fixed-architecture methods. It proposes an adaptive-architecture framework for this task, comprising three core components: a spatio-temporal distribution detector, prototype-guided disentangled experts, and a distribution-aware intervention mechanism. Extensive experiments show the proposed model outperforms state-of-the-art baselines (e.g., SILD, EAGLE) in link prediction and node classification.

### Strengths
1. Using MoE technology to solve the problem of distribution shifts on dynamic graphs is interesting and effective.

2. The distribution-aware mechanism avoids inefficient random interventions by leveraging dominant experts, enhancing invariant pattern extraction.

3. Experiment results on diverse synthetic datasets are good.

### Weaknesses
1. Apart from the MoE part, there is existing work on both disentangling (e.g., DIDA, SILD) and intervention (e.g., SILD, EAGLE) on dynamic graphs. Therefore, the overall architecture has a piecemeal feel. It is impossible to discern any fundamental changes in these sections compared to existing work.

2. The MoE section (section 4.1) has relatively low innovation.

3. The performance improvement on real datasets is very limited (less than 1%).

### Questions
Please refer to the weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses dynamic graph representation learning under evolving distribution shifts, a setting where both graph topology and features change over time and the nature of distribution shifts itself evolves. The authors propose AdaMix, an adaptive mixture-of-experts (MoE) framework that dynamically adjusts model architectures to capture invariant patterns across time. Extensive experiments on real-world (Collab, Yelp, Aminer) and synthetic datasets show that AdaMix consistently outperforms both standard dynamic GNNs (e.g., DySAT, EGCN) and prior OOD methods (DIDA, EAGLE, SILD).

### Strengths
1. The paper argues that evolving shifts make fixed architectures sub-optimal, which is well-supported.
2. The paper proposes a three component framework to dynamically adjust the architecture

### Weaknesses
1. Some baselines show very high variance (e.g., Aminer with SILD has huge std), while AdaMix gains on real data are sometimes modest. For examples, on Aminer15, Aminer16 and Aminer17, Adamix does not have evident improvements. The reported mean ± std of baselines often overlap or even exceed AdaMix in mean. 
2. Adaptive MoE at node-time granularity plus FFT-domain masking and memory updates may increase training/inference cost.
3. Although an ablation study is reported, the performance drop after removing individual components (e.g., memory module, prototype disentanglement, distribution-aware intervention) is not quantitatively large or clearly demonstrated in the main paper. The results do not convincingly show that each component is essential to the final performance.

### Questions
1. What are the training/inference time compared with baselines?
2. Could you add a “no-FFT” variant (time-domain only) and a “no-memory but deeper router” variant to disentangle spectral vs. temporal-memory contributions.

### Soundness
3

### Presentation
2

### Contribution
2
