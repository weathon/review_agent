# Expertise need not monopolize: Action-Specialized Mixture of Experts for Vision-Language-Action Learning

- Decision: Reject
- Scores: 2, 6, 2, 2

## Abstract
Vision-Language-Action (VLA) models are experiencing rapid development and demonstrating promising capabilities in robotic manipulation tasks.
However, scaling up VLA models presents several critical challenges:
(1) Training new VLA models from scratch demands substantial computational resources and extensive datasets. 
Given the current scarcity of robot data, it becomes particularly valuable to fully leverage well-pretrained VLA model weights during the scaling process.   (2) Real-time control requires carefully balancing model capacity with computational efficiency.  To address these challenges, we present a Mixture-of-Experts (MoE) architecture that naturally scales the VLA model's action expert by replacing dense feedforward layers with sparsely activated MoE layers. However, the conventional MoE framework is hampered by a critical drawback: the auxiliary loss for load balancing generates interfering gradients that misalign with the primary optimization trajectory. Therefore, we propose AdaMoE, a MoE architecture that employs a decoupling technique that decouples expert selection from expert weighting through an independent scale adapter working alongside the traditional router. This decoupling mechanism alleviates the gradient conflict between the primary and load-balancing objectives during the training process, leading to models with enhanced performance. AdaMoE consistently outperforms the baseline model across key benchmarks, delivering performance gains of 1.8\% on LIBERO and 9.3\% on RoboTwin. Most importantly, a substantial 21.5\% improvement in real-world experiments validates its practical effectiveness for robotic manipulation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper is about building mixture-of-expert models from vision-language-action models for scaling up. The focus is on separating the selection of experts in the mixture from the weight for their contribution. This is achieved by a specific architecture proposed in the paper. This is supposed to allow experts to collaborate in a more flexible way and improves the overall performance and load balancing.

### Strengths
- The results show superior performance numbers in experiments. 

- The paper contains a hyper-parameter ablation study. 

- The paper addresses a relevant problem.

### Weaknesses
- The authors write that their experts specialize in distinct aspects of the manipulation tasks and that their balancing loss supports this or makes it possible. However, it is unclear why this is happening and what makes the different experts specialize in different things. Shouldn't there be some loss that ensures that experts are different? 

- The layout makes the text quite hard to follow with figures often breaking text with equations. 

-  Looking at figure 2, we can see how the experts different in their probabilities and we can be convinced that they are actually learning something different. However, we do not see that the experts have weights that are not corresponding to their probabilities, which is the main reason that the separate scale adapter was needed. It seems demonstrating this explicitly (not just the overall performance numbers in Table 3), would be very suitable to support the claim.

### Questions
- Why are only the top-k experts used? What is the benefit?

- One of the main contributions is the introduction of a separate scale adapter. The authors argue for the necessity of this part (lines 205 to 211). However, the arguments seem vague and are based on that the selection probability for an expert and its contribution to the result are not aligned. Why should this happen? Would it not be better to correct this misalignment instead of separating the weights and the probabilities. As the weights are computed separately from the probabilities in the proposed approach, it is not guaranteed that a very important expert (with high weight) is also selected due to hight probability. Making something that is important also have high probability seems like the most fundamental goal of a mixture model.

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
2

### Summary
This paper proposes AdaMoE, a Mixture-of-Experts architecture for Vision-Language-Action (VLA) models that (i) inherits weights from a strong dense backbone (π0 / flow-matching action expert), (ii) adds shared and routed experts to the action expert, and (iii) decouples expert selection from contribution weighting via an additional scale adapter so that routing (top-k) and final weights are controlled independently (“expertise need not monopolize”). The method includes a load-balancing loss and retains π0’s chunked action decoding. Experiments show gains on LIBERO (avg 94.2→96.0%; +1.8pp) and RoboTwin-2.0 across 19 tasks (40.4→49.7%; +9.3pp), plus real-robot improvements on four tasks (avg +21.5pp, e.g.). Ablations examine router variants (vanilla / concatenated / additive) and hyper-parameters (top-k, #experts, λ).

### Strengths
1. Clear, targeted scaling for VLAs: converts a pretrained flow-matching VLA into an MoE specifically in the action expert, preserving I/O while increasing capacity. The architectural description (shared + routed experts, load balance, equations) is concrete.

2. Gains across two standard suites: +9.3pp on RoboTwin-2.0 (19 tasks) and +21.5pp on real world tasks

3. Router design ablations show the additive (decoupled) variant best; also reports the interesting observation that even “router collapse” (single expert) can still beat the dense model, hinting the router acts as adaptive scaling.

### Weaknesses
1. 1.8% Gain in LIBERO is not very substantial.

2. Novelty vs prior MoE decoupling: The paper acknowledges related decoupling ideas in MoE (e.g., DeepSeekMoE’s independent biases). A crisper positioning—what is architecturally new beyond adding a second head and summation—would strengthen the contribution.

### Questions
1. please see weaknesses

2. it is a bit strange that AdaMoE got 1.8% and 9.3% gains in LIBERO and RoboTwin, but got 21.5% gain in real world experiment. Could you give us deeper discussion of this result?

### Soundness
3

### Presentation
2

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
This paper proposes a MoE design that, the authors claim, improves expert weighting in VLAs by decoupling the selection criteria from the output criteria. The hypothesis is that there are some situations in which an expert may be highly relevant to the task, but not actually require a high contribution in the output and thus the model should have the flexibility in this decision.

The authors test their hypothesis on LIBERO and RoboTwin, as well as in several real-world experiments, showing somewhat marginal LIBERO gains (~2%) but larger RoboTwin gains (~9%).

Overall, the paper is reasonably written but lacks a strong motivation and a clear connection between the claimed contribution and the results.

### Strengths
The paper is relatively clear to understand and contains a reasonable set of evaluation benchmarks (LIBERO/RoboTwin/4 real world exps). Moreover, the general direction of MoE for VLAs is reasonable as these models get larger and are applied to more domains, yet latency remains an issue.

### Weaknesses
Overall, I don't find the motivation for this paper compelling and the experiments do not appear to directly prove the author's claims. The central claim is that experts may benefit from high likelihood but low contribution, but this claim is (1) only supported by a single experiment with a 1% gain on LIBERO avg results, with no detailed analysis and (2) not in any way self-evident or backed up by cited prior literature. Particularly given the results that topk=1 generally performed better than k=2, it is not clear that this decoupling is at all necessary in this setting.

It is also helpful to clarify that the dense -> sparse MoE conversion is common, even in VLA MoE papers (https://arxiv.org/abs/2505.21906), so this element is not unique (The authors do not claim this, although it is prominently mentioned in the abstract/contribution sections).

### Questions
- What is the activation flops and inference latency between the dense π0 baseline and the proposed MoE model?
- To clarify "We compare our AdaMoE against the π0 baseline using the same transfer learning protocol" - Was the π0 baseline and the proposed model trained with the same data/steps for the RoboTwin experiment as well? Is this the case for the LIBERO experiments as well?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces an architecture named AdaMoE, which aims to apply Mixture-of-Experts (MoE) to an existing flow matching-based Vision-Language-Action (VLA) model (π₀). The core modification is the addition of a "scale adapter" to the standard MoE routing mechanism. This adapter is designed to decouple the experts' selection probabilities from the contribution weights of their final outputs. The authors claim this decoupled design is better suited for robotics manipulation tasks. The paper presents experiments on both simulated and real robots, reporting improved performance over the baseline, π₀.

### Strengths
The direction is novel, and transferring MoE, a paradigm that has been successfully proven in the field of NLP/Vision, to VLA is a logical and worthy direction to explore.

### Weaknesses
1. Lack of detailed description of the real machine experiment, Adjust Bottle, Click bell What are these tasks? In addition. Baseline performance is too low, and the improvement content is questionable: the huge improvement of 21.5% in real-world experiments is based on the pi0 model with only a 50% average success rate. The performance of pi0 seems to be too low.
2. Lack of critical efficiency analysis: One of the core motivations of the paper is the computational efficiency of MoE. However, the full text does not provide any quantitative data on inference latency, FLOPs, or memory footprint. AdaMoE introduces additional scale adapters and routing computations, which all incur overhead. In the absence of this data, claims that it "maintains computational efficiency" are groundless empty talk.
3. The analysis of expert specialization is superficial: Figure 2 only shows the correlation of expert activation, without proving its causality. The authors observe that Expert 3 activates more at certain stages, but does this mean that it really "learns" to localize and release? The lack of more interventional experiments makes the conclusion that "experts achieve meaningful specialization" very weak.
4. The conclusion that "fewer experts are more effective" is debatable. The data show that the success rate of 8 experts is only 0.4% lower than that of 4 experts, the difference is not significant, and the experiment is carried out when the overall success rate is already high. Therefore, it is necessary to add more ablation experiments to further verify the reliability of this conclusion

### Questions
Please see the section on weaknesses for details.

### Soundness
2

### Presentation
3

### Contribution
2
