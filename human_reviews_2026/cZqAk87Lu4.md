# Shortcut Diffusion Training with Cumulative Consistency Loss: An Optimal Control View

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Although iterative denoising (i.e., diffusion/flow) methods offer strong generative performance, they suffer from low generation efficiency, requiring hundreds of steps of network forward passes to simulate a single sample. Mitigating this requires taking larger step-sizes during simulation, thereby allowing one- or few-step generation. Recently proposed shortcut model learns larger step-sizes by enforcing alignment between its direction and the path defined by a base many-step flow-matching model through a self-consistency loss. However, its generation quality is significantly lower than the base model. In this paper, we formulate few-step generation as a controlled base generative process, and show that self-consistency loss can be understood through the lens of optimal control. This perspective naturally motivates its generalization to the proposed cumulative self-consistency loss that cumulatively penalizes misalignment along the entire trajectory. This encourages larger step-sizes that not only align with the base model at the current time step but also support alignment in the subsequent steps, facilitating high-quality generation. Furthermore, we draw a connection between our approach and reinforcement learning, potentially opening the door to a new set of approaches for few-step generation. Experiments show that we significantly improve one- and few-step generation quality under the same training budget. Implementation is available at: [https://github.com/paribeshregmi/Shortcut-CSL](https://github.com/paribeshregmi/Shortcut-CSL)

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the low sampling efficiency of diffusion/flow models by enabling one- or few-step generation. Building on “shortcut” samplers that learn large step sizes via a self-consistency loss against a many-step base flow-matching model, the authors reinterpret this loss through optimal control, treating few-step sampling as a controlled base process. From this view, they propose a cumulative self-consistency loss (CSL) that penalizes misalignment not only at the current step but also along future steps of the trajectory, encouraging large yet reliable steps that maintain downstream sample quality. They further draw connections to reinforcement learning. Experiments indicate improved one-/few-step generation quality under the same training budget, and the proposed shortcut-CSL variant outperforms baselines on certain tasks.

### Strengths
• The paper is clearly written with precise notation; figures and tables are informative and easy to follow.

• The proposed shortcut-CSL shows better performance than prior shortcut variants on some tasks, supporting the value of the cumulative alignment objective.

### Weaknesses
1. Metrics are too narrow. Evaluating primarily with FID is insufficient, and the observed FID gains are minor in some cases. Please broaden the evaluation metrics like F1, CLIP-score, and qualitative comparisons, and, if feasible, a small human preference study. This multifaceted assessment would make the empirical case more solid.

2. The teaser (Fig. 1) visualizations look only slightly improved over baselines. 

3. The dataset scope is limited. Results are reported only on CelebA and CIFAR-10. While broader evaluation is costly, for fair comparisons with other methods, you should include additional datasets—e.g., ImageNet (64×64 and/or 256×256), FFHQ, or another modern benchmark—while keeping ablations on a single dataset to manage workload. This would better test generality and strengthen claims about scalability and robustness.

### Questions
My primary concern is the limited practical performance gains demonstrated (see Weaknesses).

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
3

### Summary
This paper improves upon the original shortcut model paper by introducing a cumulative self-consistency loss. In the paper, the original shortcut model loss is first framed as an optimal control problem, where the objective is to minimize *total* accumulated error. However, the standard loss only penalizes error at the *current* timestep, and thus does not account for future error. This paper derives the optimal objective that maximizes for future error minimization, analogous to maximizing future return rather than reward in an RL framework. Practically, it is found that using a discrete number of future steps (R=2 or R=4) already leads to sizable improvement. Experiments show that the proposed cumulative SL reliably improves upon the base SL, and ablations show that the improvement can be steadily improved with additional compute allocated to R.

### Strengths
This paper highlights a well-posed analysis of the shortcut modelling loss, showing that is an instantaneous approximation to an objective that should be expanded to account for future errors. Using this viewpoint, the paper derives a clear true objective, and shows that even a discrete approximation to this true objective results in performance improvement. In this way, the paper clearly ties together empirical improvement towards a theoretically-motivated insight.

The clarity in the paper is passable. Figure 2 provides a solid intuition as to why the cumulative objective is a more precise target. (See below for weaknesses).

The significance of this paper is solid, as it provides a new viewpoint which can be applied to distillation methods in general. The experiments are conducted on standard benchmarks at a reasonable network size, and wall-clock experiments are presented to take computational requirements into account. Error bars are included.

### Weaknesses
The empirical performance of this method could be better framed by comparing to recent prior works, specifically Meanflow. 

In terms of clarity, the notation introduced in Section 3/4/5 can be refined. A full explanation of optimal control as given in section 3 may not be necessary, especially terms such as b, g, and h which are not used in the final relation to shortcut model loss. In equation 4, it will help to make it clear that u(X) is the *error* term which has an optimality at zero, but the network itself does not predict u(X) but rather a velocity (that must be compared to the teacher velocity). 

For equation 14, is the gradient with relation to future errors taken through multiple evaluations of the network? If so, is the error term for k>1 backpropgated to all previous steps including the current step? An algorithmic explanation would strengthen this section.

While the text argues that equations 11 and 12 should be different, the resulting integrals appear identical. What is the precise difference between these two objectives? Commentary would strengthen this section.

### Questions
See section above for further questions associated with each topic.

As a side note, the discussion in the Appendix on the connection to TD-based value learning methods is quite interesting, and could be a worthy direction for future work.

In general, the ideas in this paper can be strengthened by a more thorough clarity in writing. The introduction and background sections can be condensed, and the relation between optimal control and the shortcut loss can benefit from additional textual commentary explaining the intuition behind the equations. With clarifications to the questions in the weaknesses section, I would consider a raise to the review score.

Can the cumulative self-consistency objective be applied to other few-step techniques such as consistency models or meanflow?

### Soundness
3

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
3

### Summary
This paper proposes Cumulative Self-Consistency Loss (CSL) as an extension of the shortcut training loss (SL) used in one-step and few-step diffusion models. The authors reinterpret shortcut training through an optimal-control lens. CSL generalizes this by integrating consistency along the remaining trajectory, encouraging alignment not just locally but cumulatively over time. The paper derives a continuous formulation of this loss, motivates it with a “cumulative gradient” analysis, and implements a discrete estimator involving R rollouts (typically R=2). Empirically, CSL yields improved FID scores on CIFAR-10 and CelebA-256 for 1-, 2-, and 4-step sampling, with minimal additional compute cost.

### Strengths
- Conceptually clear reinterpretation of shortcut loss using an optimal-control framework, identifying why standard self-consistency only optimizes instantaneous alignment.  
- The cumulative loss is simple to implement and shows consistent, measurable FID improvements with small computational overhead (~5–10%).  
- Thorough experimental evaluation with ablations (number of rollout terms, backbone scale, training cost) and comparisons across DiT backbones.

### Weaknesses
- The “cumulative gradient” claimed in Eq. 13 does not seem to match the implementation. Algorithm 1 detaches rollout targets with `stopgrad` and does not backpropagate through future steps, so it seems that the optimization reduces to multiple local MSEs rather than true cumulative assignment. 
- The midpoint update rule used during 2d rollouts is unconventional and unexplained. 
- Evaluation scope is a bit narrow, authos only evaluate on CIFAR-10 and CelebA-256, with large baselines borrowed from prior work under different setups.

### Questions
1. How is the ∇ₓJ term in Eq. 13 realized in practice? Are gradients ever propagated through rollout steps, or is CSL purely a multi-point supervision objective?  
2. Why does Algorithm 1 advance the 2d step from the midpoint state rather than from the starting point? Was this empirically motivated?

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
2

### Summary
The paper proposes an improvement over shortcut models to distill multi-step diffusion pipelines, thereby improving inference speed. The central idea of the paper is that the shortcut models only consider immediate errors, which can lead to misalignment with the model's true trajectory in severe step compression. The authors propose to penalise not just the immediate misalignment but also further misalignments down the trajectory of the teacher model.

### Strengths
- By framing the shortcut problem as an optimal control one, the authors provide mathematical grounding for their work. They model self consistency as an optimal control objective and then relax constraints to incorporate errors further along the ODE trajectory

- The authors include fair comparisons that ensures equal training budgets with baselines

- Training algorithm is included to aid reproducibility

### Weaknesses
- The paper lacks sufficient clarity, specifically in the connection of the work to Reinforcement Learning. Linking J_csl as a value function can be better explained, given the authors have mentioned this in the abstract as an important section of the paper.

### Questions
The authors discuss the increasing training cost with increasing R. Would be good to know if the authors experimented with R>4 or if the authors have an idea of how training scales with R.

### Soundness
3

### Presentation
2

### Contribution
2
