# Causal Explanations for Human Understanding in Deep Neural Policies

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Explainable deep learning models are important for the development, certification, and adoption of autonomous systems. Yet, without methods to quantify causal relationships between explanations and actions, interpretability remains correlational. Furthermore, explanations typically address lower-level actions. This poorly serves human understanding, which benefits from higher-level abstractions, and underactuated robotics, whose behaviors often require richer descriptions. To address these gaps, we introduce Causal Concept-Wrapper Network (CCW-Net), a general training method across differentiable architectures that adapts mediation analysis from fields such as economics, medicine, and epidemiology to align the causal effects of abstract, information-rich explanations with policy actions. CCW-Net expands the expressiveness of prior work in both explainable deep learning and mediation analysis allowing each explanation to serve as a mediator encoding both its presence and context-based expression. In a high-fidelity, underactuated aircraft formation task, CCW-Net produces high-level explanations that are both interpretable and quantifiably causal without degrading task performance. We demonstrate CCW-Net across diverse architectures including capsule networks with dynamic routing, modified concept bottleneck models, and cross-attention mechanisms. Notably, we present the first adaptation of capsule networks to sequential decision-making in robotics. This breadth shows that CCW-Net applies broadly across neural network architectures, offering a general path toward transparent and trustworthy autonomy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces CCW-Net, a method that aims to enforce causality in concept-based explanations for deep neural policies. To do so, it aligns each concept’s influence on actions with causal effect estimates from data. The method is tested in a realistic aircraft formation task, where it is shown that CCW-Net improves concept-to-action causal alignment while maintaining accuracy.

### Strengths
- enforcing causally-aligned explanations into RL is an interesting research direction
    
- this research direction seems novel

### Weaknesses
Major:

- **W1** - The evaluation is minimal:
    - Although the paper is presented as methodological and makes general claims, the evaluation is limited to a single application (aircraft formation). Furthermore, from the paper, it was not clear to me how causality is key for this application.
        
    - No comparisons against alternative approaches are presented. It seems to me that the proposed method is only compared to an ablation of itself by removing the alignment loss.
        
    - Broad benefits of the proposed methodology are not studied, nor are metrics specific to causality.
    

- **W2** - Assuming no unmeasured confounding seems a very strong assumption, which could not apply to practical real-world settings. It seems to me the validity of this assumption is not discussed in the paper, neither theoretically nor empirically.
    
- **W3** - Notation is very confusing and inconsistent, which makes it hard to evaluate the soundness of the approach. Find below a non-exhaustive list:
    
    - It is not clear what uppercase and lowercase symbols indicate, e.g., “we extract the latent vector Z”, but then “vector concepts $s_j$”.
       
    - undefined symbols: e.g., $\tilde{C}$ and $p$ in l.272 are not defined.
       
    - In the main text, I could not find a formal definition for two of the three employed losses, i.e., $L_{cls}$ and $L_{imit}$.
        
    - The same symbol $\theta$ is used for both the encoder $h$ and the head $g$.
        
    - In l. 306, $A$ is used as the dimensionality of a vector space, while it was previously defined as the variable denoting binary concept activation.
        
    - Concept labels were indicated with $c^{ell}$, but later, in l. 376, “Concept labels $A = (A_L, A_C)$”.
        
    - The alignment loss is sometimes referred to as ‘causal loss‘, also in Algorithm 1.

     - The chosen naming for variables is unusual:  $C$ is used to represent the raw input $X$, $A$ to represent concept activation, and $M$ for the set of concepts’ representations, for which the notation goes back to $c_j$.
        
Effort should go in making the notation clear and consistent. To start, I would recommend using uppercase symbols to indicate random variables, lower case (with indices) for realizations and bold to distinguish scalars from vectors.
        
    

Minor:

- typo: “with concept labels a tuple” (l.205).
- ll.260-264 repeat the lines above.

### Questions
- **Q1** (Related to W1) - What makes the investigated application (aircraft formation) best-suited to test causal explanations?
    
- **Q2** - I am not sure I understand the part where the authors claim to represent concepts as vectors. Concepts need to be aligned with human semantics; how are these interpretable? Where are these vectors computed? 
    

While I remain open to a constructive discussion, I believe the paper requires substantial improvement, especially in clarity and in the evaluation section. At this stage, I'm not inclined to raise my score, as the gap between the current submission and a version that would meet the bar for acceptance still feels too wide.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Causal Concept-Wrapper Network (CCW-Net), a method for training neural policies that provide human-interpretable explanations grounded in causal relationships. The approach wraps a pre-trained black-box policy with a concept module and policy head, using supervised learning from expert trajectories labeled with human-defined concepts. The method uses mediation analysis techniques to align gradient-based concept attributions with interventional indirect effects estimated from observational data. The authors evaluate CCW-Net on an aircraft formation control task using three different architectures: capsule networks, vector concept bottleneck models, and cross-attention mechanisms. Results demonstrate that the causal alignment loss improves the correspondence between concept sensitivities and causal effects while maintaining task performance and concept classification accuracy.

### Strengths
* Using a black-box policy is a practical advantage: you can apply this to existing trained policies without starting over. This is important since retraining can be expensive or impossible in many real systems.

* The experiments are well-organized with clear hypotheses (H1-H4) that are tested one by one. This makes it easy to see what the authors claim and whether the data supports it.

* The method works across three different neural architectures (capsule networks, concept bottleneck models, and cross-attention). This flexibility matters for real-world deployment.

* The aircraft formation task is a good test case because pilots actually use these concepts (lead/lag pursuit, climb/dive) in training and operations. This grounds the evaluation in real interpretability needs rather than made-up concepts.

### Weaknesses
My main concern is that the causal framework isn't clearly defined. The paper introduces a structural causal model but it's hard to tell exactly what the variables are and how they relate to each other. The notation switches between (X, Z, M, Y) and (C, A, M, Y) without clear explanation, which makes it difficult to evaluate if the causal claims are valid. The paper mentions standard assumptions for causal identification but doesn't explain how to check if these hold in practice or what happens when they don't.

Minor points:
* Some relevant related work is missing:
  - Work on using causal models for understanding language models [1,2]
  - Work on causal explanations of RL policies [3,4]
  - Explanations of sequential decision making [5,6]

References:

[1] Geiger, Atticus, Chris Potts, and Thomas Icard. "Causal abstraction for faithful model interpretation." arXiv preprint arXiv:2301.04709 (2023)

[2] Atticus Geiger, Hanson Lu, Thomas Icard, and Christopher Potts. Causal abstractions of neural networks. NeurIPS, 2021

[3] Madumal, Prashan, et al. "Explainable reinforcement learning through a causal lens." 2020.

[4] Kekic et al. "Learning Nonlinear Causal Reductions to Explain Reinforcement Learning Policies." (2025).

[5] Bica et al. Learning "what-if" explanations for sequential decision-making. ICLR, 2021.

[6] Nashed, Samer B., et al. "Causal Explanations for Sequential Decision Making." Journal of Artificial Intelligence Research 83 (2025).

### Questions
* Definition of the SCM: in L214 it seems like the set of variables (or space of the variables) is given by $\mathcal{X}, \mathcal{Z}, M, \hat{\mathcal{Y}}$, but the samples of the variables are $C, A, M, Y$ and the causal relations are given in terms of $C, A, M, Y$. What are the variables in the SCM? What are the assumed causal relationships?
* What are the assumptions made on the SCM? How did you check that these assumptions are valid in the RL case? In cases where those assumptions do not hold or only hold approximately what are the consequences for the method?
* Assumptions in L144: for all 3 assumptions, given a concrete policy/dataset, how would you check if those assumptions hold? Is there a way to measure this in order to determine if the method is appropriate for a given policy?
* L184 "Furthermore, following Loh et al. (2022), we relax assumptions on the causal graph by introducing an interaction term": which assumptions and why can they be relaxed?
* What is the conceptual difference between $A$ ("which correspond to a binary concept activation, representing if a concept is active or not" L216) and the mediators $M$ ("to represent vectors cj corresponding to each concept" L218)?
* How is this causal formulation different from a purely correlational model? Wouldn't a non-causal model that tries to predict a concept given the current state and action lead to the same results? If not, can you describe a situation in which such two models (mediation analysis model and correlational) would not yield the same concepts?

I'm happy to increase my score if we can resolve these points of confusion during rebuttal.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces the CCW-Net, which explains the actions taken by a pre-trained reinforcement algorithm with human interpretable concepts. The paper builds on concept explanations by extending them from scalar valued to vector valued. Furthermore, they add causality based explanations using mediator interactions. This paper has interesting contributions to explainability via concepts, such as moving away from correlation based interactions to causality based interactions. They also demonstrate that, on the aircraft formation task, the casual alignment loss improves the concept-to-action causal alignment across three architectures while not harming the concept accuracy.

### Strengths
1.	This paper lays the ground for producing casual explanations in reinforcement learning algorithms using human interpretable concepts.
2.	Demonstrating the increased casual alignment without degradation in performance 
3.	Interesting approach on aligning concept representations with casual effect.
4.	The methodology is head-agnostic as long as everything remains differentiable (which is usually the case, and it is an relatively easy constraint to satisfy)
5.	Clearly written paper, easy to follow.

Overall, the paper is well written an easy to follow and could be a nice contribution to the field. Therefore, I am leaning more to acceptance and the score could be increased if further ablation studies are done, limitations are clearly stated, anddifferent tasks could be added.

### Weaknesses
1.	Limitations are not stated in the paper as well as the computational requirements for training vector valued concept explanations.
2.	Lack of ablation studies on the loss lambda terms, and the fraction of replacement of reference sets. 
3.	Tested on only one task, thus would be nice to see the application of this methodology on a different task.

### Questions
For the further ablation studies, it’d be interesting to see the affect the lambda parameter on the alignment. For example, how would the alignment score change with lambda = 0.01, and lambda = 1.0 is used? Are there any drawbacks to using higher lambda? Also I am curious what would the effect of replacing a higher fraction of the Reference sets (p) would be on the alignment scores.

Furthermore, it would be nice to see a different task so that the claims can be further supported and validated.

Further clarifications questions I have:
1.	What is the reason behind excluding the Direct Effects on the Total Effects? 
2.	What is the computational complexity of adding more concepts to the pipeline in terms of computational resources and training time?

Remarks:  
- Typo in the last line 13 of the algorithm section, should be L_align instead of L_casual
- opening quotes in last line of discussion ``

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper wraps a frozen, high-performing policy with a concept module and lightweight head so that the model’s concept→action sensitivities (local Jacobians) are explicitly aligned to interventional, data-estimated causal effects (via mediator-swap–style estimators). The wrapper is head-agnostic (vector CBM, capsules, cross-attention shown). On an aviation control task, the method improves causal alignment of explanations while preserving imitation error and concept accuracy.

### Strengths
* **Originality.** Trains attributions to match interventional effects rather than reporting correlational saliency; vector concepts enrich semantics; wrapper applies across heads.
* **Quality.** Clear objectives, reasonable identifiability story, overlap diagnostics, and consistent gains in causal alignment without degrading task performance.
* **Clarity.** Clean separation of (a) interventional effect estimation, (b) local sensitivity computation, and (c) alignment.
* **Significance.** In safety-critical control, auditable concept-level attributions checked against interventional baselines are valuable for validation and governance.

### Weaknesses
1. **Strong supervision requirement (trajectories + expert concepts).**
   The approach requires expert-defined concept ontologies and per-trajectory concept labels. Performance and portability depend on the quality, granularity, and coverage of these concepts; annotation cost and domain expertise create a practical bottleneck. 

2. **Sensitivity to data scale & overfitting (imitation learning).**
   The wrapper is trained by supervised imitation. With fewer labeled trajectories and concepts, the head can overfit, yielding optimistic in-sample alignment but poor generalization. Results are reported at a single (large) data scale; learning curves vs data size and train–test gaps are missing.

3. **Assumption sensitivity.**
   The method assumes that the declared concepts capture all important pathways from inputs to actions. If any meaningful pathway bypasses these concepts, the interventional effect estimates used for alignment can be systematically wrong. The paper does not include stress tests—such as removing a concept, merging concepts, or injecting synthetic confounders—to evaluate how sensitive results are to such violations.

### Questions
1. **Labels & efficiency.** Briefly describe the labeling pipeline and show learning curves vs labeled fraction of trajectories (e.g., 10/25/50/100%).
2. **Small-data generalization.** Report train vs test imitation error and causal alignment at reduced data sizes.
3. **Concept robustness.** Run a compact stress test: remove/merge one concept, add label noise (≈10–20%), and a placebo concept; summarize impact on alignment.

### Soundness
3

### Presentation
3

### Contribution
3
