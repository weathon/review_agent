# Combination-of-Experts with Knowledge Sharing for Cross-Task Vehicle Routing Problems

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 2, 6, 6, 4

## Abstract
Recent neural methods have shown promise in generalizing across various vehicle routing problems (VRPs). These methods adopt either a fully-shared dense model across all VRP tasks (i.e., variants) or a mixture-of-experts model that assigns node embeddings within each task instance to different experts. However, they both struggle to generalize from training tasks with basic constraints to out-of-distribution (OOD) tasks involving unseen constraint combinations and new basic constraints, as they overlook the fact that each VRP task is defined by a combination of multiple basic constraints. To address this, this paper proposes a novel model, combination-of-experts with knowledge sharing (CoEKS), which leverages the structural characteristic of VRP tasks. CoEKS enhances generalization to constraint combinations via two complementary components: a combination-of-experts architecture enabling flexible combinations via prior assignment of constraint-specific experts, and a knowledge sharing strategy strengthening generalization via automatic learning of transferable general knowledge across constraints. Moreover, CoEKS allows new experts to be plugged into the trained model for rapid adaptation to new constraints. Experiments demonstrate that CoEKS outperforms state-of-the-art methods on in-distribution tasks and delivers greater gains on OOD tasks, including unseen constraint combinations (relative improvement of 12\% over SOTA) and new constraints (25\% improvement).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes CoEKS, a novel neural routing solver for cross-task VRPs. The core idea is to leverage the structural property that VRP variants are defined by combinations of basic constraints. The model consists of two main components: 1) a Combination-of-Experts (CoE) architecture, where individual constraint-specific FFNs are activated via a prior assignment based on known constraints, and 2) a multi-view knowledge sharing strategy to enhance the learning of transferable knowledge. The authors claim this method achieves state-of-the-art performance on ID tasks and demonstrates superior generalization on OOD tasks, including unseen constraint combinations and rapid adaptation to new constraints by plugging in new experts.

### Strengths
1. The paper is well-written and easy to follow.

2. The experimental design is rigorous and comprehensive, the inclusion of strong SOTA baselines provides a high-quality empirical evaluation.

3. The implementation on both POMO and ReLD backbone networks demonstrates the universality of the CoEKS architecture.

### Weaknesses
My main concerns stem from the model's core assumption, CoEKS must rely on **prior assignment of constraint-specific experts**, which limits its technical novelty and the general applicability of its claims. 

1. The contribution is relatively incremental. The model essentially replaces a learned gating mechanism in standard MoE-based multi-task solvers (ReLD-MoEL[1], MVMoE[2]) with a handcrafted external gating mechanism. While the idea of constraint-specific experts is effective, the implementation offloads the most complex gating selection logic to the user (or what we might call the real expert with domain knowledge). 

2. For the scalability for adaptation to unseen constraints, Appendix C.2 mentions the spatio-temporal conflict between MB and TW. Although the model's overall performance is promising, the process requires the inclusion of a TW  constraint to mitigate performance degradation.  I also notice that CoEKS is less effective than ReLD[1] on some tasks when only the MB constraint is applied.

3. The claim of robust scalability to new constraints is supported by a weak experiment. Mixed Backhauls (MB) is merely a relaxation of the Backhaul constraint, which is already part of the training set. Both constraints deal with the same problem type (linehaul and backhaul demands). This is an easy test case and does not sufficiently support the claim that the method can adapt to any new constraint.

4. The authors acknowledge that handling more constraints increases parameters. This one-to-one mapping of constraints-to-experts seems inefficient.

### Questions
1. It would greatly enhance the paper's contribution to evaluate the zero-shot generalization capabilities on unseen constraints(e.g., Mixed Backhauls and Multi-Depot) without applying additional expert networks and fine-tuning operations.

2. For the scalability for adaptation to unseen constraints, the authors conducted fine-tuning experiments on unseen constraints with 50-node VRP variants. I am curious about how the performance of CoEKS would compare with existing methods if the problem size increases to 100?

References:

[1] Huang, Z., Zhou, J., Cao, Z., & XU, Y. Rethinking Light Decoder-based Solvers for Vehicle Routing Problems. ICLR, 2025.

[2] Zhou, J., Cao, Z., Wu, Y., Song, W., Ma, Y., Zhang, J., & Xu, C. MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts. ICML, 2024.

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
A neural "cross-task" vehicle routing problem (VRP) solver is presented. The main insight is that model components are introduced for each constraint type, e.g., time windows or backhauls. These are called experts, and the authors define a strategy for combining their outputs to deal with problems with combinations of multiple outputs. The approach is compared against other multi-task approaches for the VRP on a part of a standard benchmark of VRP tasks. It outperforms the current state-of-the-art approaches for this multi-task setting.

### Strengths
1. The idea is generally simple and well-founded, I think it basically makes sense even if it has scaling issues in the number of constraints supported.
2. The performance is good, but I do have a very important question regarding the benchmark (see below)
3. The paper is generally easy to understand.

### Weaknesses
1. The experimental section is written very short, and important details are pushed to the appendix multiple times. This is unfortunate, as the paper wastes space on page 6 providing what I can only describe as a marketing rehash of the introduction. That entire paragraph ought to go towards describing the results. Perhaps the authors can address this.
2. The results are only incrementally better than previous work; this paper is a small step forward, but certainly not a breakthrough.
3. I do not think the idea is one that will scale in general; the concept of adding a new expert for each type of constraint will eventually just overload the model with parameters that are likely unnecessary. The approach reminds me of the Poppy approach from Grinsztajn et al. (2023) in NeurIPS in which multiple decoders are used. The extra decoders in Poppy were shown in the Polynet paper (Hottung et al. (2025) in ICLR) to not be necessary, a vector input can be used instead. The same likely holds here, so I feel like this paper is not quite finished.

Some other comments that the authors should consider addressing:
1. The contributions in the introduction are vague, it would be better to be more precise about what is offered in this work.
2. The first paragraph of the related work has some grammar issues, let an LLM fix it.
3. The research questions offered in the experimental section are well done.
4. I would like to see more about the ablation and sensitivity analysis in the main text.

### Questions
1. RouteFinder has 48 problem variants, this paper only has 24. What happened to the other variants and why are they not used?

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
This paper proposes CoEKS, a new method to encode problems in multi-task learning for VRPs. Instead of employing MoE in encoder side, the authors propose to assign an expert to each constraint. For problems with multiple constraints, all the respective experts of each constraint will be activated at the same time, and a sharing mechanism enhances performance. The model can also be adapted for new problems by adding a new expert for each constraint. CoEKS demonstrates SOTA results among neural methods.

### Strengths
1. The paper is well-written, easy to follow and with good motivation
    
2. CoEKS is novel to my knowledge and tackles the problem in an interesting manner
    
3. Good performance

### Weaknesses
1. My major concern is the following: given that we have multiple experts, why not simply train on all of them at the same time and always activate them? In other words, why bother with specialization when one can have multiple MLPs at the same time (or a larger MLP with the same number of parameters with the sum of experts)? I believe the bump in performance might be simply due to having more parameters
    
2. The method will need to activate all the parameters for problems, including all of the constraints, which is however acknowledged by the authors
    
3. Overall, the work is a bit incremental, in the sense that it just adds a specific mechanism for the encoder layers with a relatively small improvement in performance
    
4. Code is not provided

### Questions
1. What if instead of experts we just had all MLPs work at the same time, or a larger MLP with the same number of parameters?
    
2. I am not too clear about the relevance of OOD generalization to “unseen” tasks. As the tasks in Table 2 are merely a combination of constraints, in practice, why not simply train a method that incorporates them already?
    
3. Why do you finetune only the added experts? Wouldn’t the performance improve if you finetune all of the models as well?
    
4. It would be interesting to see what the learned representations of experts are in the latent space. Do they align with the specified constraints, or do experts learn generic representations? It would be interesting to see more about this in say a t-SNE analysis similar to POMO-MTL or RF

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses the problem of vehicle routing problem in a multi task setting.
This work modifies the Mixture of Experts (MoE) model into a Combination of Experts (CoE) framework, where experts are constraint-specific and are combined later. Knowledge sharing with MSE loss is applied between the experts. A combiner is introduced to regularize the model. Experiments are conducted to show the validitty of the method.

### Strengths
1. The idea of combining experts is intuitive and well-aligned with the CVRP multi-task setting.
2. The use of separate heads for each constraint is straightforward and easy to understand.

### Weaknesses
1. **Writing needs improvement:**  
   The description of the combiner is presented in two separate sections (lines 203–211 and lines 263–275), with no clear connection between them. It is unclear whether these sections refer to the same concept or different ones. If they are the same, the repetition needs to be justified. If they are different, the components should be clearly defined.

2. **MD loss for \( K = 2 \) vs \( K > 2 \):**  
   The rationale behind the different MD loss for \( K = 2 \) and \( K > 2 \) is not provided. There is no explanation for this distinction.

3. **Distillation loss (MD) adoption from (Xie et al., 2024): [1]**  
   The MD loss is adopted from (Xie et al., 2024), but it is unclear whether this approach has been extensively studied in other works besides Xie et al. (2024). Regardless, the reasoning behind using distillation when both heads are learning simultaneously is not clear. It appears to be primarily a regularization technique, restricting the output heads to remain similar. Even with this interpretation, it is not clear why this regularization is combined with the Mixture of Experts model (or the Combination of Experts model, in this case), which involves a weighted sum of outputs. The main motivation behind MoE (or CoE) is to capture distinct modes across different network heads and combine them. This regularization approach would work against that motivation, as it encourages all experts to output similar embeddings.

4. **Comparison with SHIELD:**  
   A relevant work, **SHIELD**[2], presented at ICML, shares a similar motivation of generalizing across different tasks and distributions, but incorporates tasks as part of the input features. This work is neither compared nor discussed in the paper.

## Overall:
The idea of combining different heads for different tasks is reasonable. However, the framework appears somewhat incremental, with the primary modification being the adaptation of the Mixture of Experts model to fit experts to different subtasks, along with the addition of regularization techniques.

## References:
1. Zhitian Xie, Yinger Zhang, Chenyi Zhuang, Qitao Shi, Zhining Liu, Jinjie Gu, and Guannan Zhang. Mode: A mixture-of-experts model with mutual distillation among the experts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 16067–16075, 2024
1. Goh, Yong Liang, et al. "SHIELD: Multi-task Multi-distribution Vehicle Routing Solver with Sparsity and Hierarchy." Forty-second International Conference on Machine Learning.

### Questions
Please see weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
