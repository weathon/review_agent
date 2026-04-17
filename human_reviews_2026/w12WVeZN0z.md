# FlowDrive: moderated flow matching with data balancing for trajectory planning

- Decision: Reject
- Scores: 6, 4, 4, 4

## Abstract
Learning-based planners are sensitive to the long-tailed distribution of driving data. Common maneuvers dominate datasets, while dangerous or rare scenarios are sparse. This imbalance can bias models toward the frequent cases and degrade performance on critical scenarios. To tackle this problem, we compare balancing strategies for sampling training data and find reweighting by trajectory pattern an effective approach. We then present FlowDrive, a flow-matching trajectory planner that learns a conditional rectified flow to map noise directly to trajectory distributions with few flow-matching steps. We further introduce moderated, in-the-loop guidance that injects small perturbation between flow steps to systematically increase trajectory diversity while remaining scene-consistent. On nuPlan and the interaction-focused interPlan benchmarks, FlowDrive achieves state-of-the-art results among learning-based planners and approaches methods with rule-based refinements. After adding moderated guidance and light post-processing (FlowDrive*), it achieves overall state-of-the-art performance across nearly all benchmark splits.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces FlowDrive, a trajectory planning framework for autonomous driving. The framework leverages flow matching, data sampling, and a novel guidance technique to improve trajectory generation. FlowDrive overcomes limitations in previous approaches by generating diverse driving trajectories without the need for hard-coded offsets. It achieves enhanced diversity in trajectory outputs while ensuring feasibility by introducing a guidance technique that injects small perturbations during flow integration. The framework also integrates a data balancing strategy to mitigate dataset bias, ensuring better generalization to rare but critical driving scenarios. Empirical results on benchmarks like nuPlan and InterPlan show that FlowDrive significantly outperforms existing rule-based and diffusion-based planners.

### Strengths
1. The authors introduce a unique guidance mechanism, which helps in generating diverse trajectories by injecting small, structured perturbations. This technique improves the lateral diversity of trajectories and avoids post-processing or hard-coded adjustments, which is a key contribution of the work.

2. FlowDrive demonstrates superior closed-loop performance, outperforming rule-based methods as well as learning-based planners across both nuPlan and InterPlan benchmarks. It achieves state-of-the-art results.

3. FlowDrive is more efficient than diffusion-based planners. The paper provides detailed experiments showing that FlowDrive maintains competitive performance while requiring fewer flow steps, thus reducing computation time and increasing scalability.

### Weaknesses
1. Minor Issues: On page 3, line 122, the term “a diverse trajectory” could be more appropriately referred to as “diverse trajectories”. Similarly, on line 112, the term “a trajectory score model” should perhaps be “trajectory scoring model”.

2. Lack of Ablation Studies on Hyper-parameters: While the paper addresses the importance of the guidance mechanism, there is a need for more quantitative ablation studies on the various hyper-parameters used in the guidance process. Based on Figure 4, the choice of parameters like discrete flow times in the guidance mechanism seems to have a significant impact on the trajectory generation. This process also involves other hyper-parameters like the horizon weight and different magnitudes. A deeper analysis on the hyper-parameters and such designs could provide more insight into this technique.

3. The authors primarily evaluate FlowDrive on nuPlan-based datasets, which are highly representative but may not fully cover the variability found in real-world driving environments. The framework’s ability to generalize across a wider range of datasets (such as Waymo and CARLA) remains unclear. Providing more diversity in the evaluation datasets would strengthen the claims of FlowDrive’s real-world applicability.

### Questions
Please refer to the weaknesses section. I do not have other questions.

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
This paper proposes FlowDrive, a flow-matching-based trajectory planner designed to address the long-tailed distribution problem in driving data. By introducing a trajectory-pattern-based reweighting strategy, the method mitigates bias toward frequent maneuvers. FlowDrive learns a conditional rectified flow that maps noise directly to trajectory distributions with few steps, enabling fast and diverse trajectory generation. A moderated, in-the-loop guidance mechanism further enhances diversity while maintaining scene consistency. Experiments on nuPlan and interPlan show that FlowDrive achieves state-of-the-art performance among learning-based planners and approaches hybrid rule-based systems in both accuracy and efficiency.

### Strengths
The method demonstrates fast inference and strong overall performance, outperforming its baselines. The results indicate that the approach is effective in practice.

### Weaknesses
1. The technical novelty is somewhat limited, as most of the key ideas—such as data balancing, constrained sampling, and flow-based formulations—have already been explored in related works.
 
2. The experimental section lacks deeper insights. Most results focus on baseline comparisons and ablations, making it difficult to understand why the flow-based planner is a better choice. Moreover, the paper does not provide an inference speed comparison, despite claiming that flow-based methods require fewer forward passes.

3. A comparison between the proposed flow-based planner and a diffusion-based planner with balanced data sampling is also missing, which would strengthen the empirical justification of the approach.

### Questions
Questions are included in the weakness section.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces FlowDrive, a flow-matching-based planner that maps noise to trajectories in just a few denoising steps. Experiments and ablation studies on the nuPlan benchmark demonstrate the effectiveness of the proposed flow-matching decoder and data reweighting strategy.

### Strengths
Pros
- The proposed method is clearly presented and easy to follow.
- Experiments show that FlowDrive achieves state-of-the-art (SOTA) performance on the nuPlan benchmark, outperforming DiffusionPlanner in all three evaluated scenarios.
- Clear planning-oriented flow formulation with few-step sampling; strong wall-clock efficiency vs diffusion.

### Weaknesses
Cons
- Limited Novelty. The method's novelty appears somewhat limited. The core contribution seems to be a straightforward application of flow-matching to the autonomous driving planning task, supplemented by a simple resampling technique. Furthermore, the results from the main experiment show only a marginal improvement over the DiffusionPlanner baseline. No obvious advantages or novelty part over other Flow-Based methods.
- In lines 46-48, the authors claim their main motivation is that flow-matching can directly transform random noise into trajectories, yielding faster sampling than diffusion planners. However, in Sec. 4.4 and Tab. 2(b), experiments show that FlowDrive still requires about 8 inference steps and exceeds DiffusionPlanner by only 20%. This improvement seems marginal, especially considering that rectified flow methods can perform well using only a single step [1][2].
- The main experiment lacks a comparison with GoalFlow. Although it is an end-to-end method, a comparison against its planner module seems necessary to fully demonstrate the contribution of the proposed planner.
- Hybrid dependence blurs novelty in final SOTA: FlowDrive* relies on post-processing (smoothing, speed-limit enforcement) and rule-based scoring (PDM) to pick the executed plan; this makes comparisons with purely learning-based systems less clean.
- Clustering target space: k-means on flattened  [𝑥,𝑦] ignores yaw/speed and scene semantics; choice  𝐾= 20 is ad-hoc and may underspecify rare but safety-critical interactions.

[1] Xing, Zebin, et al. "Goalflow: Goal-driven flow matching for multimodal trajectories generation in end-to-end autonomous driving." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.
[2] Liu, Xingchao, Chengyue Gong, and Qiang Liu. "Flow straight and fast: Learning to generate and transfer data with rectified flow." arXiv preprint arXiv:2209.03003 (2022).

### Questions
1. I am curious that your cluster-based balancing uses only [x,y] sequences—would including yaw/speed (or map-relative features) alter the rare-mode coverage and the gains in Table 2a?
2. Moderated guidance is injected once at t=1/2, did you test multi-step schedules, or learned schedules conditioned on scene risk (e.g., near merges)?
3. The offset magnitudes are sampled in normalized coordinates—what’s the corresponding meter-scale across city types, and how do you prevent over-nudging on narrow lanes? 
4. I am curious that FlowDrive* relies on rule-based scoring (PDM) for selection—do your learning-only variants reach the same diversity/safety if you replace the hand-crafted scorer with a learned critic? 
5. The jerk is handled post-hoc—have you tried training-time smoothness constraints or second-order flow fields (as in “second-order planning” work you cite)?

### Soundness
3

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
This paper introduces FlowDrive, a trajectory planner for autonomous driving that addresses the long-tailed distribution of driving data and the need for efficient, diverse trajectory generation. The authors propose two main contributions: first, a cluster-based data balancing strategy that reweights training samples to improve performance on rare maneuvers, and second, a flow-matching generative model that maps noise to trajectories in very few steps for fast inference. To further improve results, they add a "moderated guidance" technique, which injects small lateral perturbations during generation to systematically increase trajectory diversity. The final model, combines these contributions with light post-processing to achieve state-of-the-art performance on the nuPlan and interPlan benchmarks.

### Strengths
1. Strong Empirical Performance: The paper presents very strong results on the nuPlan and interPlan benchmarks. The final model, FlowDrive*, achieves state-of-the-art (or very near-SOTA) performance, outperforming previous learning-based and hybrid planners on almost all metrics (Table 1).

2. Effective Data Balancing: The paper clearly shows the benefit of its cluster-based sampling (Table 2a). This method directly addresses the long-tail problem by balancing the motion patterns in the training data, leading to a significant performance boost (e.g., from 81.91 to 85.37 on Val14) compared to no balancing or scenario-based balancing.

### Weaknesses
1. Limited Methodological Novelty: The core idea is to apply flow-matching (specifically rectified flow) to trajectory planning. While this is a valid contribution, it feels like an incremental step given that other generative models, including diffusion planners (which are related to flows) and even other flow-matching planners (like GoalFlow, as cited), already exist in the literature. The model architecture itself is a solid but standard encoder-decoder design.

2. "Moderated Guidance" Seems Ad-hoc: The excellent SOTA results for "FlowDrive*" rely heavily on a "moderated guidance" technique (Sec 3.4) and a rule-based post-processor (Sec 4.1). This guidance, which involves injecting small lateral perturbations during the flow steps, feels like a practical heuristic to force diversity rather than a fundamental improvement. If the flow model truly learned the multi-modal trajectory distribution, it's not clear why this manual guidance should be necessary.

3. Unclear Ablation of Contributions: The final SOTA model (FlowDrive*) combines the base flow model, the moderated guidance, and a rule-based scorer from the PDM planner. The ablations don't clearly separate the impact of these last two components. It's hard to tell if the performance gain comes from the novel guidance or just from applying a strong, existing rule-based filter to an ensemble of 30 candidates.

4. Motivation for using flow matching: the main motivation mentioned in the introduction section for using flow models instead of diffusion ones is the speed and the number of denoising steps required. While efficiency and higher speed are always a positive feature, in the context and size of the models used in planning, the difference seems to be of lower importance. These models are small and fast anyway, so even reducing the inference time by an order of magnitude does not seem very important alone.

### Questions
1. Regarding the moderated guidance: Why is this in-loop perturbation needed? If the flow model has correctly learned the multi-modal data distribution, shouldn't sampling from the model (perhaps with low temperature) be enough to generate diverse and feasible trajectories? Does the fact that this guidance is so effective suggest that the base FlowDrive model is failing to capture the necessary diversity (e.g., for overtaking)?
2. Could the guidance be made learned (e.g., classifier-free style or value-guided) rather than hand-tuned δ_lat and a single mid-flow injection? Any early experiments on learning δ_lat or learning where to inject?
3. Could you provide an ablation that separates the effect of the guidance from the effect of the rule-based scorer? For example, what is the performance of:
   * Base FlowDrive (no guidance) + PDM scorer (with 30 samples)?

   * FlowDrive + moderated guidance, but without the final PDM scorer (e.g., just taking the best-of-30 based on a simpler metric, or the 0-offset trajectory)?
 This would help clarify where the performance gain is really coming from.

### Soundness
3

### Presentation
3

### Contribution
2
