# On the Design of One-step Diffusion via Shortcutting Flow Paths

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 8, 8, 4, 4, 8

## Abstract
Recent advances in few-step diffusion models have demonstrated their efficiency and effectiveness by shortcutting the probabilistic paths of diffusion models, especially in training one-step diffusion models from scratch (\emph{a.k.a.} shortcut models). However, their theoretical derivation and practical implementation are often closely coupled, which obscures the design space.
To address this, we propose a common design framework for representative shortcut models. This framework provides theoretical justification for their validity and disentangles concrete component-level choices, thereby enabling systematic identification of improvements. With our proposed improvements, the resulting one-step model achieves a new state-of-the-art FID50k of 2.85 on ImageNet-256×256 under the classifier-free guidance setting with one step generation, and further reaches FID50k of 2.53 with 2× training steps. Remarkably, the model requires no pre-training, distillation, or curriculum learning.
We believe our work lowers the barrier to component-level innovation in shortcut models and facilitates principled exploration of their design space.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the problem of designing one-step diffusion models, often called shortcut models, which can be trained efficiently from scratch. The authors argue that existing works are complex, with theoretical derivations and practical implementations being "closely coupled." This makes it difficult to understand the core design principles, compare methods, or innovate on individual components. To solve this, this paper proposes a unifying framework that reframes existing shortcut models under a single principle: approximating a two-step flow map target. They empirically and theoretically demonstrate that continuous-time shortcut models (CTSC) using a linear flow path consistently outperform discrete-time (DTSC) or cosine-path-based models. Building on this analysis, the authors propose an improved CTSC model called ESC with three key designs: plug-in velocity, class-consistent batching, and a gradual time sampler. Using these improvements, the authors train a scaled-up (SiT-XL/2) ESC model from scratch, achieving a new state-of-the-art FID of 2.85 on ImageNet-256x256 with a single function evaluation (1-NFE), without requiring any pre-training, distillation, or curriculum learning.

### Strengths
1. The paper's primary strength is its common design framework. The "one-step prediction vs. two-step target" (Eq. 5) is a powerful and intuitive abstraction that successfully unifies a complex and dense field of research. Table 1, which decomposes prominent models into their constituent parts, is an excellent contribution to the community.
2. The motivations are clearly stated and verified. The analysis in Section 3, which ablates flow paths (linear vs. cosine) and time sampling (discrete vs. continuous), is thorough and convincing, backed by both empirical results (Fig. 2) and theoretical analysis.
3. The final result, an FID of 2.85 (1-NFE) on ImageNet-256x256, is a significant achievement, especially given that the model is trained entirely from scratch. This result challenges the long-held belief that one-step models require costly distillation from a pre-trained, multi-step teacher.

### Weaknesses
This is a strong paper without major technical weaknesses. Please see my questions below.

### Questions
1. Does the trained model actually satisfy the flow consistency property $X_{0.5,0}^{\theta}(X_{1,0.5}^{\theta}(x_1)) \approx X_{1,0}^{\theta}(x_1)$? The 2-NFE failure suggests it may not (Figure 6). Could this imply the model has simply memorized a single shortcut rather than learning the underlying flow field?
2. Following the previous question, I'd like to know if ESC can be used in a multi-step fashion. If yes, are there any results or visualizations? 
2. Is ESC applicable to video DiT models? Are there any results?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This submission summarizes and organizing algorithmic design space of one-step diffusion models including consistency training, shortcut diffusion models, and mean flows.
Although differences in derivation and wordings of the methods, they share computational framework of training by aligning one-step predictions toward targets that are acquired by two-step computation, which can be viewed as learning "shortcut".
The manuscript includes theoretical analyses claim that 1) both two-step DTSC and CTSC have bounded errors w.r.t. Lipschitz constants of the velocity field, and 2) inference error of  shortcut models measured Wasserstein-2 distance are bounded using bias and variance defined using targets and losses.
Following the insights,  improved training method Explicit&easier Shortcut Model (ESC) is proposed. It uses techniques from the summarized literatures and new plug-in velocity, which aggregates in-minibatch velocity.
Experiments using CIFAR and ImageNet shows that ESC can make improvements over MeanFlow slightly (-0.1 -- -0.3 in FID).

### Strengths
- One-step diffusion models are a hot topic in 2025 after Shortcut diffusion and Mean flow. This submission is a timely and nice follow-up to help understanding.
- Theoretical analyses are deep and well motivated for understanding shortcut behavior of continuous and discrete models. (although I could not check all of the proofs in the near-30-page appendix.)
- In the method part, plug-in velocity (Algorithm 1) is novel and shown to be useful in the experiments.
- The overall paper are well organized.

### Weaknesses
- Practical impact: surely the SOTA results in FID are achieved but the FID gains provided by ESC may be marginal. I think that the differences of FID ranging within 0.1 -- -0.3 hardly impacts on human opinions on the quality of images.

- Experimental/theoretical supports of design choices are relatively weak: Table 2 shows the design choice picked from the design space, but I think these selections looks empirical and I could not grasp how the theoretical results are exploited. Please point if I missed some parts.

### Questions
- In algorithm 1, some lines are too much pythonic to interrupt non-programmer readers. for example, x[:,None,:] and logp_fn = Normal(0, 1).log_prob (function treated as an object) may be replaced if possible.

### Soundness
4

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
4

### Summary
This paper proposes a unified framework for ODE based one-step diffusion models trained from scratch. The authors systematically analyze existing methods (CT, SCD, IMM, sCT, MeanFlow) by decomposing them into modular components and studying their design choices. 
Building on this analysis, they introduce ESC (Explicit & Easier Shortcut model), which incorporates several technical improvements including plug-in velocity, gradual time sampling, and adaptive loss weighting. ESC achieves improved FID50k of 2.85 on ImageNet-256×256 with one-step generation, without requiring pre-training, distillation, or curriculum learning.

### Strengths
- The paper is clearly written and easy to follow.
- While the work does not introduce a fundamentally new method and offers only moderate novelty, it contributes a valuable unifying perspective on ODE-based one-step diffusion models.
- The theoretical contributions are strong: Theorem 2.2 establishes error bounds for both DTSC and CTSC; Proposition 3.1 presents an insightful bias–variance analysis clarifying when CTSC outperforms DTSC; and Theorem C.7 theoretically justifies why linear paths are optimal for shortcut models under Fisher information metrics.
- It provides comprehensive ablations and analyses that thoroughly examine the effectiveness of the proposed techniques, demonstrating overall high quality.

### Weaknesses
- Limited Novelty of Core Framework: While the unified view is valuable for understanding, this is already well established in previous work Flow Map Matching[1,2], which further hurt the contribution of "propose a common design framework for representative shortcut models".
    -from discrete to continuous, one can either set $s= t−dt$ to get backward formula such as MeanFlow and sCM; or set $s=r+dt$ to obtain the forward formula such as AlighYourFlow [3].
- Empirical Gaps:
    - Slow convergence in 2-NFE generation (Fig. 6, Section E) suggests the improvements may be overfitted to 1-NFE
    - Limited comparison with recent distillation-based methods that achieve better FID scores [4,5,6]
    - The improvement compared to the baseline is nevertheless incremental.
- Technical Concerns: The time sampler design (gradual transition from sCT to MeanFlow) and the choice of $p_{plugin}$ appears heuristic without principled justification

[1] Nicholas Matthew Boffi, Michael Samuel Albergo, and Eric Vanden-Eijnden. Flow map matching with stochastic interpolants: A mathematical framework for consistency models. Transactions on Machine Learning Research, 2025. 

[2] Qiang Liu. Icml tutorial on the blessing of flow. International conference on machine learning,
2025.

[3] Amirmojtaba Sabour, Sanja Fidler, and Karsten Kreis. Align your flow: Scaling continuous-time flow map distillation. arXiv preprint arXiv:2506.14603, 2025.

[4] Tianwei Yin, Micha¨ el Gharbi, Richard Zhang, Eli Shechtman, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6613–6623, 2024.

[5] Mingyuan Zhou, Huangjie Zheng, Zhendong Wang, Mingzhang Yin, and Hai Huang. Score identity distillation: Exponentially fast distillation of pretrained diffusion models for one-step generation. In Forty-first International Conference on Machine Learning, 2024.

[6] Weijian Luo, Tianyang Hu, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, and Zhihua Zhang. Diff-instruct: A universal approach for transferring knowledge from pre-trained diffusion models. Advances in Neural Information Processing Systems, 36:76525–76546, 2023.

### Questions
- how to interpret IMM as a special case in the proposed framework? e.g. build connection between eq(5) and eq(50)
- In fig.1, it shows that practical prediction matches practical target, which is unrealistic. In addition, in fig 1 (c)(e), we have vector addition $v_{t|0}+u_{t|r}^\theta=(r-t) d u_{t|r}^\theta / dt$, which is inconsistent with eq(8).
- In sec 2.4 Q3, the authors claim that "(s)CM benefits from distillation by learning from a pretrained velocity field", however, in sCM, "We always initialize the CM from the EMA parameters of the teacher diffusion model. For sCD, we always use the $F_{pretrain}$ of the teacher diffusion model with its EMA parameters during distillation." That's to say, for sCT the teacher model is used as initialization for the one-step generator, which is critical.
    - Given that the authors admit that at least a pre-trained model is important for better few-step performance as answer to this Q3, and that it's impractical to train large-scale text-to-image model from scratch, the authors failed to justify why in this work they insist working on training from scratch.

-------- below could be biased (does not affect the rating) -------
- focus of the paper. while aims to "disentangles concrete component-level choices, thereby enabling systematic identification of improvements", the empirical results are mainly focus on improved training techniques for MeanFlow baseline.
- the authors show in Fig 1 visually and with prop 3.1 that the challenge of constructing flow map targets and the consequence during inference, however, this challenge is not directly addressed in the paper. 
The training technique/design are proposed to enhance training stability, while no empirical results can validate the training instability of the baseline.

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
4

### Summary
This paper proposes a unified framework for various shortcut model methods and provides both theoretical and empirical analyses of the advantages and disadvantages of different designs. Based on these analyses, several improvements are introduced, such as plug-in velocity, gradual time samplers, and class-consistent mini-batching, collectively referred to as the explicit&easier shortcut modeling method, which achieves state-of-the-art performance in one-step ImageNet-256×256 generation.

### Strengths
- This paper provides a comprehensive framework for a series of shortcut model methods, offering effective tools for analyzing this family of approaches.
- The analysis of each component is detailed and well-structured. By disentangling the individual components, the paper makes the design space considerably more transparent. Theoretical analyses are extensive and appear to be well-structured.  
- The proposed method achieves SOTA results on one-step ImageNet-256×256 generation, demonstrating the effectiveness of the improvements.
- This paper releases detailed code, ensuring reproducibility.

### Weaknesses
- According to Figure 3, it is difficult to claim that “the convergence of FID50k during training is substantially faster with the class-consistent mini-batching technique.” More evidence is needed to support this statement.
- As shown in Figure 6, at the XL scale, the proposed method achieves worse FID under 2-NFE compared to 1-NFE. This might indicate a potential scalability issue of the proposed approach.
- There are a few minor typo errors. For example, in line 198, $X^\theta_{t,r}(xt)$ should be $X^\theta_{t,r}(x_t)$; in line 263, $l_{scm}$ seems to refer to $l_{sct}$ mentioned earlier.

### Questions
- The results in Tables 2 and 3 suggest that the proposed method brings more improvements with the large-scale network architecture than with the basic one. What causes this difference?
- In Figure 3(b), several comparison curves included in Figure 3(a) are missing. What is the reason for this omission? The same question applies to Figure 3(c).

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper first summarizes and presents the design space of shortcut flow paths in diffusion models for one-step generation, such as consistency model, shortcut diffusion, MeanFlow, etc. The authors further propose plug-in velocity along with multiple pratical techniques to improve training of continuous-time shortcut models (SC), and enhance generation performance of shortchut models in their experiments.

### Strengths
+ The paper presents a formalization of shortcut models within a unified framework. This can provide a valuable foundation for subsequent work.
+ The paper elucidates the design space of SC models. Both the mathematical formulation and the overall writing quality are presented with clarity.
+ The paper makes several theoretical contributions that will likely benefit future research. These include: i) A Wasserstein distance bound for the objectives of discrete-time (DT-SC) and continuous-time (CT-SC) models. ii) An inference error bound for both CT-SC and DT-SC in terms of the variance of the average velocity. iii) The optimality of linear paths for SC models under Fisher information metrics.

### Weaknesses
- The empirical results indicate that the proposed plug-in velocity yields marginal performance gains, suggesting that its practical benefits over existing methods may be limited.

- The experimental comparison would be strengthened by the inclusion of other state-of-the-art baselines, such as rectified flow and reflow, for a more comprehensive evaluation.

- The improvement techniques presented appear to be specific to the MeanFlow architecture. The paper's impact could be broadened by exploring the generalizability of these techniques to other SC models. For example, the authors might include a discussion or experimental analysis on the selection of loss metrics for different SC model variants.

### Questions
1. Regarding the multi-GPU training implementation, are the batches gathered across all devices to compute a plug-in velocity, or is the plug-in velocity computed locally on each device's batch?

2. Do the authors investigate the impacts of the number of samples used for calculating the plug-in velocity? For instance, setting computational efficiency aside, does increasing the sample size lead to further performance gains?

3. Table 3 indicates a significant performance gap between the MeanFlow-based approaches and the other methods benchmarked. Could the authors offer an explanation or formalize the reasons for this observation?

### Soundness
4

### Presentation
4

### Contribution
3
