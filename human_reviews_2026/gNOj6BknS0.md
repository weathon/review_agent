# Two-Stage Coverage Expansion for Cross-Domain Offline Reinforcement Learning via Score-Based Generative Modeling

- Avg Score: 3.60
- Decision: Reject
- Scores: 2, 4, 6, 4, 2

## Abstract
Cross-domain reinforcement learning (RL) aims to transfer knowledge from a source domain to a target domain with different dynamics, but existing approaches often directly reuse source transitions, which can lead to severe distributional mismatch and performance degradation when the domain gap is large or target data is scarce. We propose Two-stage Coverage Expansion (TCE), a dual score-based generative framework that first expands state coverage through a mixture-based state score network and then aligns transitions with target-domain dynamics using a target-transition score network. This two-stage design broadens the effective support of the target dataset while mitigating harmful distributional shift, enabling more improved policy learning under limited target data. Extensive experiments on diverse cross-domain benchmarks demonstrate that TCE consistently outperforms state-of-the-art cross-domain RL baselines, achieving substantial gains even under large domain gaps and extremely small target datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents a data augmentation method to solve the cross-domain adaptation RL problem. Specifically, the two-stage method comprises a state expansion module and a transition module, which generate new target-like data. Further, they provide a data filtering method to filter the generated data.

### Strengths
The paper is well written and easy to follow. The motivation is clear to me, and the solution seems novel and fits the problem. Figure 1 and the visualization appear clear to me, which effectively states the problem.  The experimental results appear promising under specific conditions.

### Weaknesses
1. The experimental setting is not well stated (maybe I am missing something, or at least you should state that in the table caption?). What is the shift level the paper are using in the experiment section? The ODRL benchmark has easy/medium/hard for kinematics and morphology shift and 0.1/0.5/2.0/5.0 for gravity? For gravity shift, the paper seems to report only a 0.5 shift level, and friction shift is not included (I found it very far away from the table in the XML code). I suggest that more shift levels should be preferred, and a large shift case should be included, as the author claims that they can improve in a large shift case such as gravity/friction 5.0/0.5 or morphology/kinematics shift. 

2. Though the motivation of the paper is clear to me, I am not very clear about the motivation of the method itself. For eaxmple, why do you need to expand the state instead of only using the source/target state to do the second stage? Regarding the transition, why do you only learn the (s_t, s_{t+1}) instead of including the action, (though it might be hard as I can imagine)? If you ignore the action here, it seems that you are only generating the some "state pair", regardless the action. 

3. Figure 3 (a) looks good to me, and it is good to have such a visualization. It seems that the generated state avoids all the source states, showing that it learns something. However, I wonder whether the generated states are actually the target states or just something in between the source and target states? Also, is it possible that you have some figure like Fig. (a) to show what the generated transition looks like, whether it is really some new transition instead of only mimicking/overfitting the target data?

4. Another question is regarding the method itself.  As stated in (2) you ignore the action here, it seems that you are only generating the some "state pair", regardless the action. And it seems to me that you are given some state pair and do the inverse state to get the action. Then, why do you still need such complicated generation process, instead of directly inverse the source state pair (or some trivial solution to get some state pair) to get the target action and filter out data. It seems to me that the inverse action part is something that really make this method works.  Because given any state pair, you can always use the inverse action to make the transition look like target data and then filter out those bad data, it is just a matter of computational efficiency. So, I suggest that more experiments are needed.

### Questions
See weakness.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper propose a novel two-stage network approach for cross-domain RL. In stage 1, train a mixture SGM to expand state-space coverage by leveraging rich source data and in stage 2 uses a conditional SGM to align the generated transitions with the target-domain dynamics. Followed by auxiliary model to create the corresponding action and reward, the paper are able to curate an augmented dataset.

### Strengths
1. Novel and principled methodological design: the two-stage design is a novel and elegant solution to the data transfer problem in cross-domain RL. By separating the learning of state coverage from the learning of target dynamics, the work offers a clean, architectural approach for data augmentation in RL that moves beyond simple distance-metric-based source-data selection.

2. Strong empirical performance: The experiments result strongly suggests that the method achieves "substantial gains" and "consistently outperforms state-of-the-art cross-domain RL baselines" even under large domain gaps and extremely small target datasets. This empirical strength on a critical, real-world problem warrants serious consideration.

### Weaknesses
1. Lack of rigorous mathematical proof: The paper does not provide a formal boundedness analysis on the resulting distribution of the augmented data which is mandatory for any method based on synthesis in offline RL. It does not introduce any theoretical modification to the policy constraint or pessimism. The policy still potentially risks catastrophic Extrapolation Error when acting greedily on the augmented data

2. No guarantees of filtering: The reliance on "Z-score-based filtering" can not guarantee that the distance like KL divergence between the augmented dataset and the true target dynamics are provably bounded. The filtering itself could inadvertently prune high-value, but naturally low-likelihood, target transitions, thereby introducing a new bias.

3. Lake of enough sensitivity analysis: Score-based models are intrinsically sensitive to discretization steps, noise schedules, and weighting functions. By chaining two score-matching procedures, the overall framework's sensitivity is compounded, making the method a hyperparameter tuning nightmare. The lack of comprehensive sensitivity analysis severely limits the reproducibility and trust in the empirical results.

4. Potential computional burden: Training two score networks + the auxiliary models adds significant computional burden.

### Questions
1. Provide a theoretical result that demonstrates that the combination of the two-stage SGM and the Z-score filtering step provides a bound on the KL divergence between $D_{aug}, D_{tar}$ or similar distance. What is the theoretical justification for the Z-score threshold choice?

2. Present an ablation study comparing the full TCE against a simpler, single conditional SGM, $p(s, s' | a)$, that models the entire transition directly, or a non-mixture-based state SGM. Is the significant computational overhead strictly necessary for the claimed performance gains?

3. Provide a comprehensive sensitivity analysis for the SGM's key hyperparameters like the noise schedule $\sigma(\tau)$ and the number of discretization steps $K$. How robust is the final performance to these choices?

### Soundness
1

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TCE to resolve the distribution mismatch in resuing source transitions. TCE expands state coverage first and aligns transitions with the target domain. Experiments show that TCE outperforms cross-domain baselines on several benchmarks.

### Strengths
- The paper introduces a clear and well-motivated two-stage generative design, where state and transition modeling are decoupled, leveraging conditional score-based diffusion modeling to manage both state support expansion and careful transition dynamics alignment.

- The results section offers extensive benchmarking across 36 cross-domain morphology, kinematic, and gravity shift tasks in MuJoCo, comparing TCE to numerous strong baselines (IQL*, DARA, BOSA, SRPO, IGDF, OTDF). TCE solidly outperforms these across most scenarios, especially where the target data is scarce or the domain gap is high.

### Weaknesses
- The quality of synthesized transitions is at the mercy of the inverse dynamics and reward prediction models, both trained only on small target datasets. Given the known overfitting risks and distributional shifts, further justification or diagnostic evaluation of the labeling process (as the final step before RL training) would be helpful. There is, for instance, no explicit quantitative evaluation of the action/reward labeling accuracy on held-out real transitions.

- The selection of baselines is solid for cross-domain settings. However, the comparison omits a few important general offline RL methods (e.g., MOReL, which also emphasizes pessimism and support constraints) that could provide a useful baseline reference, helping to ground the observed performance gains.

- Several directly relevant recent works are omitted from the related works. Notably, recent papers on nearest-neighbor-guided diffusion, reverse dynamics-based cross-domain offline RL, and hybrid robust RL are not discussed or empirically compared. This limits the ability of the reader to precisely situate TCE’s advances and also misses opportunities to critique or learn from closely related design choices.

### Questions
1. Can the authors provide more systematic evidence (e.g., statistical plots or empirical tabulations) for the robustness of TCE to $y_{\max}$ and $z_{\mathrm{th}}$ across all domains/agents and more than just a handful? Are there settings where improper tuning of these parameters substantially reduces policy performance?
2. Can the authors demonstrate qualitative or even hand-checked inspection of generated transitions for high-dimensional states (esp. Ant), to substantiate that coverage expansion does not admit unrealistic/unfeasible samples (beyond Z-score filtering)?
3. How scalable is the approach in both data generation (time/memory) and RL training if (a) state/action spaces are significantly higher-dimensional, or (b) there are “many” source domains? What are the concrete bottlenecks and trade-offs?
4. Could the authors explicitly discuss and, if possible, empirically compare to the directly relevant recent diffusion-based, reverse dynamics, and hybrid robust cross-domain RL works listed above?
5. What's the performance of the proposed method without the KL regularization term? Is this term added to other baselines?

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
In this paper, the authors study cross-domain offline reinforcement learning where the source and target MDPs share \((\mathcal S, \mathcal A, R, \gamma)\) but differ in transition dynamics \(P_{src} \neq P_{tar}\), and the target dataset is small. They propose Two-stage Coverage Expansion, a score-based generative framework with SDEs: a mixture-based state score network first expands target state coverage via controllable source–target interpolation, and a target-transition score network then generates next states aligned with target dynamics. After generation, Z-score filtering removes outliers, inverse dynamics and reward models reconstruct actions and rewards, and the resulting synthetic dataset augments for offline policy learning with IQL plus KL regularization to stay close to target behavior. Experiments on MuJoCo tasks under morphology, kinematic, and gravity shifts show that naive source data reuse can harm performance when domain gaps are large, while TCE improves returns by balancing coverage and dynamics consistency. Overall, TCE provides a  pipeline that decouples coverage expansion from transition alignment to enhance data efficiency in cross-domain offline RL.

### Strengths
1. The experimental evaluation is extensive and well-executed: the authors test TCE across multiple MuJoCo domains , and provide detailed ablations on key hyperparameters.  
2.    The theoretical part is well-grounded, the paper carefully defines builds the method on established SDE-based score generation theory, and shows step by step how the two score networks work together to expand coverage and match target dynamics.

### Weaknesses
1.   The TCE framework relies on two separately trained score-based generative models, which increases implementation complexity and training cost. The coupling between these two stages is not formally analyzed, and potential error propagation from Stage 1 to Stage 2 is not well quantified. Moreover, the subsequent construction of \(D_{gen}\) through simple inverse dynamics and reward prediction appears somewhat ad hoc, lacking a more principled treatment of uncertainty or consistency. Overall, while the data generation pipeline is elaborate, the offline RL component itself remains relatively standard, making the overall architecture feel unbalanced—most design novelty lies in data synthesis rather than policy learning.
2.   The mixture-based state sampling in Stage 1 depends heavily on the hyperparameter \(y_{\max}\), yet no principled way to set or adapt this parameter is provided. The balance between “coverage expansion” and “dynamics consistency” is treated empirically rather than theoretically.

3.   The coverage expansion in Stage-1 focuses on enlarging the statistical support of the target state distribution, but it does not guarantee physical reachability. Even if the generated \(\hat{s}_t\) lies within a statistically plausible region after Z-score filtering, it may still be dynamically infeasible under the target environment’s transition dynamics. Since Z-score is purely a statistical criterion, it cannot ensure that generated states belong to the reachable set of the underlying MDP. A more principled approach would explicitly incorporate reachability constraints or learned feasibility critics to maintain physical consistency.

### Questions
Q1: The paper defines the cross-domain setting as sharing the same state, action, and reward spaces while differing only in transition dynamics. Although this assumption can be satisfied in MuJoCo-style simulations, real-world cross-domain scenarios often involve variations in observation or action spaces. Would this simplification make the problem definition too narrow or less representative of practical settings?

Q2: If there exist slight mismatches in state spaces (e.g., limited joint angles or missing body parts in Ant variants), can the proposed TCE framework still function effectively, given that it assumes perfectly aligned state–action representations across domains?

Q3: The baselines do not include decision-transformer-based offline RL methods such as Meta Decision Transformer or Prompt Decision Transformer, which explicitly handle domain variations in both reward functions and dynamics. Could the authors clarify why such methods were not compared, and whether TCE could be integrated with or outperform them under the same problem setting?

Q4: In Section 4.3, the inverse dynamics and reward models are trained using only \((s, s')\) pairs. What is the rationale behind this design choice, and is the assumption that the reward can be accurately inferred from \((s, s')\) alone theoretically sound in tasks where

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents two-stage coverage expansion (TCE) for cross-domain offline RL via score-based generative modeling. TCE trains a mixture-based state score network to expand the state space, and a target-transition score network to align the dynamics. Then the generated samples are filtered based on Z-score and mixed with the target dataset for data augmentation. Experiments on MuJoCo-like environments with dynamics shifts indicate TCE outperforms previous baselines.

### Strengths
- The paper is easy to understand and follow.
- TCE seems effective on cross-domain offline setting compared with previous baselines.

### Weaknesses
- The novelty is limited. It seems that TCE merely utilizes the generation capabilities of the score-based generative model in cross-domain offline RL setting, without more in-depth analysis or investigations. Therefore, the inspiration of this work is limited. I think an ICLR paper should include more inspirations and analysis.

- The source dataset is not fully utilized. It seems that only the state score network training uses the states in source dataset, while other information in the source dataset such as source dynamics is disgarded.

- The reward model and inverse dynamics model are trained on target datasets. Then how can they be reliable for the generated samples which may contain OOD data? No more analysis or error bound are provided. 

- The figures in this paper are blurry. Please provide vector graphics (such as pdf, svg) instead of png or jpg.

### Questions
Please see the weaknesses for the concerns. I think this paper lacks of novelty and depth for an ICLR paper and recommend for rejection.

### Soundness
3

### Presentation
2

### Contribution
2
