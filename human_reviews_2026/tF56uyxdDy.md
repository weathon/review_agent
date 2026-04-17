# Test-Time Scaling with Reflective Generative Model

- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4, 8, 2

## Abstract
We introduce a new Reflective Generative Model (RGM), which obtains OpenAI o3-mini's performance via a novel Reflective Generative Form. This form focuses on high-quality reasoning trajectory selection and contains two novelties: 1) A unified interface for policy and process reward model: we share the backbone network and use task-specific heads for reasoning trajectory predicting and scoring respectively, introducing only 50M extra parameters for trajectory scoring. 2) Eliminating the reliance on process-level annotation: we provide a self-supervised process reward model (SPRM), which can directly learn the high-quality reasoning trajectory selection from the outcome reward. Equipped with the reflective generative form, RGM is naturally suitable for test-time scaling based on the controllable thinking length. Experiments show that our RGM, equipped with only 50M additional parameters in SPRM, outperforms policy models with 72B extra reward models, thereby enabling 32B model to outperform OpenAI o3-mini on AIME24 (84.2 vs. 79.6) and HMMT25 (53.1 vs. 53.0).
Code is available at https://github.com/MetaStone-AI/XBai-o4.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on parallel TTS techniques to enhance the reasoning capabilities of large models. The authors propose Reflective Generative Model (RGM), where the core idea is to have the policy model and the PRM share the same backbone network. This reduces the parameter and computational overhead associated with deploying a separate PRM. Experiments demonstrate that a 32B model equipped with just a 50M-parameter SPRM can outperform OpenAI's o3-mini on challenging reasoning benchmarks.

### Strengths
1.  clear performance improvement is observed as the number of candidates increases, proving the method's effectiveness. 
2. The shared-backbone design adds minimal parameter overhead, significantly reducing computational costs.
3. As claimed by the authors, RGM enables efficient reasoning selection while eliminating the need for expensive process-level annotation.

### Weaknesses
1. Scalability Concerns: Like many PRM methods, this approach requires manually decomposing responses into "Steps." This process relies heavily on human priors and empirical design, raising doubts about its scalability to new tasks.
2. A core claimed contribution of the paper is removing reliance on process-level supervision. However, ImplicitPRM already claims they can fulfill this (learn PRM from ORM ). How do you position your work relative to this line of work, which suggests that a PRM can function without any extra parameters?

### Questions
Could you elaborate on how the "Step-tokens" are identified and inserted during the reasoning process?

Is it possible to compare with the ImplicitPRM methods?

### Soundness
3

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
The paper introduces Reflective Generative Models to enable test time scaling. Authors propose a unified interface where the policy model and PRM share the same backbone network, which saves parameter; they also introduce a Self-supervised PRM that learns to evaluate reasoning quality from only outcome-level supervision. Experiment results show that RGM with SPRM achieves performance comparable to OpenAI o3-mini on math benchmarks.

### Strengths
* The unified interface design is elegant and addresses a real deployment challenge in test-time scaling. Achieving comparable performance to 72B parameter reward models with only 50M parameters is a solid contribution
* The evaluation is thorough and well-designed with multiple base models and diverse architectures
* Paper writing is clear and readable. The formalization of different inference paradigms provides a clean framework for understanding the contribution
* The "aha moment" analysis is very interesting and provides valuable insights for future works

### Weaknesses
* The SPR loss introduces a bootstrapping process where the current model quality decides the psudolabel quality. I am curious if there is any formal analysis on the convergence properties. 
* Since the authors are proposing a novel architecture, I would expect more ablation studies on model design, for example the SPRM head MLP structure, and the geometric mean aggregation. 
* The method relies on '.\n\n' tokens as semantic boundaries, which assumes the policy model naturally produces well-segmented reasoning. This heuristic may not transfer to models with different output conventions, non-English languages, or domains where reasoning doesn't follow paragraph-like structure.

### Questions
* Can the authors provide some analysis on the model's robustness to initialization? 
* Is there a way to detect the aha moment online during training, or can you predict when it will occur based on model size or other factors?
* Additional ablation studies or explanations of design choices regarding the model architecture would strengthen the paper. See weakness.
* Given the dependency on training SPRM jointly with each policy model, have you explored whether SPRM trained on one model can transfer to similar architectures with fine-tuning? This would make the approach more practical for people who want to apply it to other base models without access to full retraining.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes RGM (Reflective Generative Model), a test-time scaling approach that shares the backbone network between the policy model and process reward model (PRM), adding only 50M parameters for trajectory scoring. The key innovation is a Self-supervised Process Reward Model (SPRM) that learns to evaluate reasoning steps using only outcome-level supervision. Experiments show strong empirical results across models of different sizes.

### Strengths
- Model overlap in the policy and reward model means that the inference overhead is minimal.
- SPRM can be trained with just the outcome labels, and from the ablation experiments seems to correspond to process-level correctness. For example, score of the last step performs worse than the score of the entire sequence. 
- Good empirical results
- Generalization beyond math to coding 
- The experimental setup is thorough and tested across multiple model sizes.

### Weaknesses
- Important baselines such as majority voting is missing. Moreover, there have been recent work such as "GenSelect: A Generative Approach to Best-of-N" (Toshniwal, 2025) and "Learning to Reason Across Parallel Samples for LLM Reasoning" (Qi, 2025) which demonstrate strong parallel reasoning performance. 
- The claims about beating 72B RM is somewhat misleading. The Qwen2.5-RM models were trained on short, non-reasoning CoT solutions, and are not suitable for scoring the long reasoning traces generated by models evaluated in this paper. The parameter count claim of 50M beating 72B RM in the abstract is also somewhat misleading because the RM is sharing the backbone.
- MCTS results being slightly worse than Best-of-N suggests the RM is still not good enough to conduct search which begs the question if a process reward model is really buying any performance in this setup.  Ideally, an ORM trained on the same data as SPRM would be a more fair comparison.

### Questions
- Is the "Aha moment" surprising? Isn't it just training dynamics?
- Why did you not train a new ORM on the training data for SPRM?
- Why are obvious baselines like majority voting/self-consistency missing from the paper?

### Soundness
2

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
This work aims to improve the test-time performance of policy models. Specifically, the focus is on allowing the policy to generate multiple candidates for a given query and then using a process reward model to select the best candidate from the pool. The key innovation of this paper is sharing the backbone network between the policy and the process reward model to reduce parameter overhead. Additionally, the authors propose a method that leverages the consistency between final answer correctness and the scores generated by the process reward model, in order to mitigate the negative effects caused by false positive and false negative samples during the training of the process reward model.

### Strengths
- By sharing the backbone network, the proposed method reduces the inference cost of using the PRM to evaluate policy rollouts.

- Experimental results show that the proposed SPRM achieves superior performance with the addition of fewer parameters.

- The proposed method is simple and seems to be effective.

### Weaknesses
- Missing related work on process reward models. Several studies [1-4] also incorporate outcome labels to train a process reward model, which is highly relevant to this paper.

- Other work [5] has introduced '\n' as a step token. What is the rationale and benefit behind selecting '\n\n' instead? A concern is that if the policy model does not generate '\n\n', how would this method remain applicable?

- Regarding line 219, I have a concern about the clarification: "Since the representation in the last layer mainly captures the logits prediction for a single token, we use the hidden representations from the second-to-last layer of the policy model to provide richer contextual information." Why does the second-to-last layer provide richer contextual information than the last layer? More theoretical or empirical justification is needed for this assertion.

[1] From r to Q: Your Language Model is Secretly a Q-Function, COLM 2024.

[2] Discriminative Policy Optimization for Token-Level Reward Models, ICML 2025.

[3] DPO Meets PPO: Reinforced Token Optimization for RLHF, ICML 2025.

[4] Free Process Rewards without Process Labels, ICML 2025.

[5] Let's Verify Step by Step, Arxiv 2023.

### Questions
- What does the variable c represent in the Linear(c, 2c) at Line 206?

- In Equation 6, what do the terms $y_i$ and $Score_i$ with the index $i$ represent?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper introduces reflective generative models for test-time scaling, which use a shared network for policy and process reward models. This dramatically decreases the number of extra parameters to 50M. Additionally, the paper introduces a self-supervised loss (SPR Loss), which directly learns the quality of the reasoning trajectory from the outcome reward. The core idea is to have the same model both generate reasoning trajectories and score them with minimal extra parameters. The authors conducted a wide range of experiments across baseline models and demonstrated high performance (OpenAI o3-mini level), outperforming billion-scale reward models.

### Strengths
- Originality: The paper proposes a highly original idea of using the same backbone for policy and reward models. This idea is a novel and exciting extension of prior reward models, which are typically large and separately trained. This idea opens up a lot of exciting directions for enabling richer interactions between reasoning trajectory generation and evaluation. 
- Quality: The proposed framework is clearly defined and well-motivated. The experimental evaluation is comprehensive, including multiple models and benchmarks. 
- Clarity: The text and figures are clear and easy to understand. 
- Significance: The paper makes a highly significant contribution to test-time scaling.

### Weaknesses
1. The design of the self-supervised process reward loss (SPR loss) could benefit from additional motivation and clarification. Specifically, the binary weight w_n only includes a step in the loss when the predicted step score aligns with the final outcome. Why choose a hard threshold (0.5) vs other alternatives? Could such a hard cutoff potentially discard a large fraction of training samples, particularly early in training? And could this selective inclusion behavior relate to later observations, such as the “aha” moment? Also, a minor point: y_n is the correctness of the final answer, which shouldn’t depend on n? Why use a subscript (which could be misleading)? 
2. The paper’s discussion of the “aha moment” (Sec. 5.4; Fig. 5) is vague and under-analyzed. The authors highlight a green dashed line to indicate where correct and incorrect trajectory scores begin to diverge, yet provide no quantitative criterion for identifying this point. Visually, the gap between curves appears to increase gradually rather than showing a discrete transition. If there is indeed a transition, could it be simply explained by the use of a hard 0.5 threshold in the SPR loss? The authors could add more quantitative analysis on the representation or gradient through learning if this “aha moment” is indeed an important finding.  
3. The paper claims that SPRM generalizes across domains, but this claim is only supported by results on LiveCodeBench. Given that mathematics and coding reasoning tasks share very similar structures, this claim of generalization currently lacks sufficient evidence. This claim could be substantially strengthened by either including evaluation on more diverse tasks or adding a discussion on what types of domains the current approach is expected to generalize well to and where its limitations might lie. For example, the segmentation choice of using ‘\n’ might not be as suitable for tasks involving natural language?
4. The explanation for why MCTS underperforms relative to Best-of-N (BoN) is speculative and unsupported by evidence.

### Questions
1. The final trajectory score is defined as the geometric mean of the single-step scores (Eq. 5). It might be helpful to add a sentence on the motivation/justification for this choice. I assume the geometric mean is chosen to push all step scores to be decent, since a single low score can substantially reduce the overall score. It might be interesting to add what scenarios this choice of final score does not yet capture. For example, right now, the final score treats each single step score equally and independently. In the case where the reasoning trajectory later corrects for its earlier mistakes, the earlier bad step might undesirably penalize the entire trajectory. There is an interesting question of how to incorporate the trajectory structure into the score. Have you tried any alternative form? 
2. Minor type: Eq. 6 (line 245), score_i and y_i should be score_n, y_n.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a Reflective Generative Model that unifies a policy model and a process reward model through a shared backbone. It proposes a filtering mechanism for learning a process reward module (SPRM) based on outcome rewards, aiming to eliminate the need for process-level annotation

### Strengths
Strong empirical validation: The experiments are extensive and include multiple benchmarks and LLMs.

### Weaknesses
- SPRM is a filtering mechanism, not self-supervised: The model filters step-level data via a binary weight that retains only steps consistent with the outcome.
- Limited novelty: The shared backbone between policy and reward model is an engineering optimization rather than a conceptual advance in test-time scaling. Moreover, the paper does not study whether shared parameters introduce bias.
- Terminology and clarity issues: The formulation of LLMs lacks rigor  (e.g., “basic LLM”)

### Questions
- What is the reasoning behind using the same backbone for the reward model and policy? Could this introduce bias or reward hacking effects?
- What is the multi-agent data cleaning framework used to obtain high-quality samples for the dataset?
- Why is SPRM described as a self-supervised method, given that it relies on filtered outcome correctness?

### Soundness
1

### Presentation
1

### Contribution
1
