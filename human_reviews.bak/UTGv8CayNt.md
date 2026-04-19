# Chain-of-Thought Predictive Control

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
We study generalizable policy learning from demonstrations for complex low-level control tasks (e.g., contact-rich object manipulations). We propose a novel hierarchical imitation learning method that utilizes scalable, albeit sub-optimal, demonstrations. Firstly, we propose an observation space-agnostic approach that efficiently discovers the multi-step subgoal decomposition (sequences of key observations) of the demos in an unsupervised manner. By grouping temporarily close and functionally similar actions into subskill-level segments, the discovered breakpoints (the segment boundaries) constitute a chain of planning steps (i.e., the chain-of-thought) to complete the task. Next, we propose a Transformer-based design that effectively learns to predict the chain-of-thought (CoT) as the high-level guidance for low-level action. We couple action and CoT predictions via prompt tokens and a hybrid masking strategy, which enable dynamically updated CoT guidance at test time and improve feature representation of the trajectory for generalizable policy learning. Our method, named Chain-of-Thought Predictive Control (CoTPC), consistently surpasses existing strong baselines on a wide range of challenging low-level manipulation tasks with scalable yet sub-optimal demos.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a policy learning from the demonstration method. The authors propose a novel hierarchical imitation learning that utilizes scalable demonstrations. The demonstration is decomposed into a sequence of key observations, and then CoT is leveraged to generalize policy learning.

### Strengths
1. I think the author's environment looks cool and real. This is very important in the policy learning domain and in the robotic domain. I see the supplementary video and I believe that this environment provides a good test environment for policy learning methods. The authors are encouraged by the reviewer to continue their study in this domain, even though this paper may be rejected.

2. The general direction of learning policy with demonstrations is good, and the hierarchical RL formulation for this task is also sound and important, although those two ideas are not very novel.

3. The numerical results are good and the improvement looks significant.

4. I like the supplementary video as well, although it can be improved (see below).

### Weaknesses
# 1. About the novelty

(1) I think CoT is not very novel in this context and I don't think the proposed approach can be regarded as CoT. The proposed approach refers to a demonstration split methodology (or a subgoal discovery mechanism in hierarchical RL). Since no language is involved, I don't think this method can be related to CoT.

(2) This paper is not positioned well in the context of hierarchical RL + demonstration, so the novelty is not well-stated. The authors should mention, discuss, or compare the following works [1-5]. Disclaim: I am not an author of any of these works. The idea of learning subgoals by similarity or diversity is not novel.

(3) Using transformers in policy learning is not novel as well, and I don't think the major goal of this paper is related to architecture design. If the goal is to claim the novelty of CoT control, the authors should try other architectures as well. Since nowadays transformers are very common architectures, I don't think this can be claimed as a major novelty.

(4) It's unclear why the authors use a single model to learn CoT and learn the action. The authors should try to compare to architectures like [6].

# 2. About the experiments

(1) The authors should discuss their results and summarize the conclusions in the main text. I'm confused about the main result in Tab. 1, and this is because the results are not discussed (the authors only say results in Tab.1 without further explanations). It seems that BC, DT, and BeT are not hierarchical RL methods. So it is very unfair. The authors should not compare to methods that do not use subgoals. Instead, the authors should mainly compare to hierarchical RL methods that leverage demonstrations. The authors should search for [1-5] as long as their follow-up works to get the most related hierarchical RL methods to compare to. 

(2) The variance should be shown for each method, and the learning curves are required.

(3) The supplementary videos should be combined with a presentation video, which can reveal the comparisons between the proposed approach and previous works.

# 3. About the writing

(1) The function of the model is not discussed and section 4.2 looks confusing as a result. I think the function and signature of the model should be discussed prior to the model details.

(2) The details of the CoT algorithm are unclear. Particularly, these two sentences look very confusing to me.

"therefore, propose to group contiguous actions into segments, using a similarity-based heuristic to find these subskills." How does the group algorithm work? How to discretize continuous actions?

"We then utilize the Pruned Exact Linear Time (PELT) method [38] with cosine similarity as the cost metric to generate the changepoints in a per-trajectory manner." How does this work? I'm not familiar with the PELT method and this should be discussed in detail.

(3) The authors assume the readers know the decision transformer and detection transformer in advance, which is not very good.

[1] Jiang, Yiding, et al. "Learning Options via Compression." Advances in Neural Information Processing Systems 35 (2022): 21184-21199.

[2] Eysenbach, Benjamin, et al. "Diversity is all you need: Learning skills without a reward function." arXiv preprint arXiv:1802.06070 (2018).

[3] Konidaris, George, et al. "Robot learning from demonstration by constructing skill trees." The International Journal of Robotics Research 31.3 (2012): 360-375.

[4] Pickett, Marc, and Andrew G. Barto. "Policyblocks: An algorithm for creating useful macro-actions in reinforcement learning." ICML. Vol. 19. 2002.

[5] Kipf, Thomas, et al. "Compile: Compositional imitation learning and execution." International Conference on Machine Learning. PMLR, 2019.

[6] Target-driven Visual Navigation in Indoor Scenes using Deep Reinforcement Learning

### Questions
1. How does the number of CoT affect the policy learning results?

2. If the transformer's weights are tuned, why the CoT can accurately serve as a target signal?

3. How efficient is the transformer architecture?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present Chain-of-Thought (CoT) Predictive control, a transformer based approach to policy learning, trained via sequence modeling. The transformer is augmented with learnable CoT prompt tokens that guide low-level action learning. In addition, the transformer is trained to predict the next and last high-level prompt, further encouraging abstractions that capture higher level semantic information. The high-level prompts are discovered in an unsupervised manner, as changepoints in time, discovered with Pruned Exact Linear Time methods, using cosine similarity as a cost metric. The model is trained on suboptimal demos and surpasses other transformer based methods on held-out tasks.

### Strengths
The paper has several strengths:


1) Reasonably well written
2) Simplistic and effective approach that outperforms similar methods
3) Nice to see their approach applied to complex dynamic settings, and generalizing favorably

### Weaknesses
My main concern with the method is their motivation for how they perform sub-task decomposition. Cosine similarity metric seems heuristical, not well motivated, and anecdotal. Whilst the results are promising, it is unclear whether more principled decompositions  would lead to better results: e.g. obtained via bottleneck options [1] or gaussian processes [2]. The paper would benefit from a greater discussion/comparison on this front. It is unclear to me whether their decomposition approach would favor different tasks, with distinct action-space statistics.  

In addition, there are a couple of presentation limitations:

1) Citations are not in the correct ICLR format (surname and year)
2) Results are lacking confidence intervals (how many runs/model seeds)

[1] - Salter, Sasha, et al. "Mo2: Model-based offline options." Conference on Lifelong Learning Agents. PMLR, 2022.

[2] - Saatçi, Yunus, Ryan D. Turner, and Carl E. Rasmussen. "Gaussian process change point models." Proceedings of the 27th International Conference on Machine Learning (ICML-10). 2010.

### Questions
1) How well does this approach to sub-task decomposition scale to larger action-spaces?
2) How sensitive is their approach to the beta parameter that controls the number of detected changepoints?
3) Fig 2 - Can the author's comment on what the action groupings correspond to intuitively for these examples?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents CoTPC, a behavior cloning method that predicts simultaneously multiple future sub-goals (Chain-of-thoughts), as well as low-level actions.  It also presents a method for discovering brakpoints and a chain of planning steps. It's evaluated on state-based tasks across various settings, from 2D moving maze, to franka kitchen, then to several tasks in maniskill2. The experiments show CoTPC outperforms other baselines as well as other ablation choices. It also show some preliminary results on 2 real world tasks.

### Strengths
I like the general direction this paper is pursuing. Addressing suboptimality in demonstrations by finding shared hierarchical patterns and key states makes a lot of sense. Predicting a sequence of subgoals simultaneously, as opposed to auto-regressively one-by-one, is also reasonable in terms of better guiding low-level actions prediction.
The paper has a set of extensive experiments, as well as some preliminary study using realistic visual inputs and real-world experiments.
In addition, i was a reviewer reviewing this paper during its previous round of submission, back then one of my major concern is lacking of a automated machanisim for extracting key states from the demo. This has been addressed to some extent in this version.

### Weaknesses
* I am still not fully convinced by using the term 'Chain-of-Thought'...
*  Real world evaluation is a bit too simple

### Questions
I have no further questions since I reviewed this before.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes CoTPC, a Transformer-based architecture that performs hierarchical planning.
An important part of this method is the unsupervised discovery of subgoals, which assume that temporally close and similar actions belong to the same subskills.
The overall architecture uses learned subgoal embeddings and uses the goals discovered by the unsupervised algorithm to train these learned embeddings, as an auxiliary loss.
The remainder of the architecture fits into the family of the Behaviour transformer, with some design differences that impact the performance, and utilize the CoT learned embeddings.
Experiments on the Moving Maze, Franka Kitchen and ManiSkill2 environments show the effectiveness of CoTPC over baselines.
An ablation study is also presented to showcase the importance of different design choices in the architecture.

### Strengths
- The dataset chosen for experiments are relevant, and the results are quite convincing, with the CoT model clearly performing better than the baselines. I thought the exposition of the experiments section clear and easy to follow. The choice made on experiments were clear and, to my knowledge, the choice of baselines are fair.

- Although the architecture is limited in novelty, the method as a whole is novel and the specific design decisions are novel. I particularly liked the use of learned embeddings that are trained using an auxiliary loss with targets generated using an unsupervised subgoal discovery process.

- The proposed work is clearly motivated and properly positioned in the literature.

### Weaknesses
- The writing of section 4 needs improvements. Specifically, I found Section 4.2.2 quite hard to parse and easy to get lost. I would encourage the authors to re-write this section and redo Figure 1 so that things are clearer. 
    - Specifically, I'm confused by what the inputs of $g_\text{CoT}$. 
    - What is the CoT predictor? It appears once in the last paragraph of section 4.2.2. 
    - Why aren't actions used as inputs to $g^x(.)$ functions?
    - What exactly are the contents of $\{ \mathbf{S}^\text{CoT}_{...}\}$? Do they change with time? What is the (...) subscript? 
    - There seems to be $T$ CoT features, but that seems confusing as these are suppose to represent subgoals and are trained using the auxiliary $L_\text{CoT}$. I thought PELT was minimizing the number of goals. How do you ensure alignment between the learned tokens and the output of PELT?

- I found the ablation study to have limited value. I think it should aim to provide the reader with more intuition on what exactly is learned by CoT embeddings. This could possibly be shown on the maze environment, and would show clearly the discovery of the subgoals. I see Figure 5 in the appendix and it is a step in the right direction, but in my opinion, it would be be easier and more informative to show in the maze environment.

- As a minor point, I would ask the authors to express the limitations of the approach. For instance, is it always possible to assume that actions that are similar or close temporally belong to the same subskill?

### Questions
- It seems like BeT should have been explained in section 3 as the method seems to be heavily based on it. I leave that up to the authors to decide, but it could potentially ease the exposition. An alternative could be to present it in the appendix, similar to the Rt-1 paragraph.

- In the last paragraph of section 5.4 named "Shared tokens for CoT and action predictions", I think the different variants should be shown as a three part diagram two help the reader understand the differences in inputs. As of right now, I am not entirely sure what the differences are. 

- I would be curious to see what the effect of setting the component of the auxiliary loss term to 0, but keeping learnable prompt tokens. I think this is basically equivalent to BC but with added learnable tokens , maintain the same capacity as the CoTPC architecture.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
