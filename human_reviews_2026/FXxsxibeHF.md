# Prompting Decision Transformers for Zero-Shot Reach-Avoid Policies

- Decision: Reject
- Scores: 6, 2, 6

## Abstract
Offline goal-conditioned reinforcement learning methods have shown promise for reach-avoid tasks, where an agent must reach a target state while avoiding undesirable regions of the state space. Existing approaches typically encode avoid-region information into an augmented state space and cost function, which prevents flexible, dynamic specification of novel avoid-region information at evaluation time. They also rely heavily on well-designed reward and cost functions, limiting scalability to complex or poorly structured environments.
We introduce RADT, a decision transformer model for offline, reward-free, goal-conditioned, avoid region-conditioned RL. RADT encodes goals and avoid regions directly as prompt tokens, allowing any number of avoid regions of arbitrary size to be specified at evaluation time. Using only suboptimal offline trajectories from a random policy, RADT learns reach-avoid behavior through a novel combination of goal and avoid-region hindsight relabeling.
We benchmark RADT against 3 existing offline goal-conditioned RL models across 17 tasks, environments, and experimental settings. RADT generalizes in a zero-shot manner to out-of-distribution avoid region sizes and counts, outperforming baselines that require retraining. In one such zero-shot setting, RADT achieves 35.7% improvement in normalized cost over the best retrained baseline while maintaining high goal-reaching success.
We also apply RADT to cell reprogramming in biology, demonstrating its versatility.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper focuses on the reach-avoid problem under the offline goal-conditioned setting, and proposes a decision transformer model to achieve zero-shot generalization to varying numbers and sizes of avoid regions. Specifically, it incorporates the avoid-region information into prompt tokens, whose number can be changed during inference. In addition, the paper introduces an avoid-region hindsight relabeling method, which generates safe and unsafe trajectory pair by modifying the avoid region for each trajectory. Experimental results show that the proposed method performs competitively on in-distribution tasks and achieves better overall performance on out-of-distributiontasks without retraining.

### Strengths
Strengths

- The paper is intuitive and well-motivated. Incorporating avoid-region information into prompts to adapt to different numbers and sizes of constraints is reasonable and addresses some limitations of prior work.

- The experimental results are convincing, demonstrating the effectiveness of the proposed approach.

- The paper is clearly written, and the implementation details are easy to understand.

- The experiments on extreme OOD generalization are impressive, showing strong generalization capabilities.

### Weaknesses
Weaknesses

- The maze tasks used in experiments are relatively simple. It would be better to include more complex control tasks besides fetchreach, such as those from safe RL benchmarks, to further validate the method.

- I did not find ablation studies on the relabeling component, which is an important part of the contribution and should be discussed in the main text.

- Some prior works on offline safe RL have explored dynamic constraints, such as [1,2], which adjust the constraint threshold dynamically. It would be better to discussed and compared.

- While the motivations and problem setting are novel, similar ideas of using prompt-based models for dynamic task adaptation have been explored in prior works [1,3], which may somewhat limit the perceived technical novelty.

[1] Liu, Zuxin, et al. "Constrained decision transformer for offline safe reinforcement learning." International conference on machine learning. PMLR, 2023.

[2] Lin, Qian, et al. "Safe offline reinforcement learning with real-time budget constraints." International Conference on Machine Learning. PMLR, 2023.

[3] Xu, Mengdi, et al. "Prompting decision transformer for few-shot policy generalization." international conference on machine learning. PMLR, 2022.

### Questions
Questions and Comments

- In Table 1, RADT does not dominate all baselines simultaneously on both MNC and SR metrics. While I agree that the smaller difference in SR and the significant improvement in MNC demonstrate the advantage of RADT, I wonder whether there is a quantitative metric that could better summarize the overall performance. For example, in classic constraint RL, the best algorithm is typically defined as the one that achieves the highest reward among all algorithms that satisfy the constraint. However, in the avoid-region setting, since no algorithm achieves MNC=0, it is unclear how to fairly evaluate the tradeoff.

- Figure 3 (c, e) seems to duplicate results from Table 1, and could be streamlined.

- I am not sure whether the terms “reward-based” and “reward-free” are used precisely. The referenced work [1] still relies on reward signals to compute return-to-go. When the goal is known, designing sparse rewards is not difficult, so I am not entirely clear about the essential difference and advantage between these two categories. Does it mean Q-learning–based methods might be more sensitive to reward sparsity than supervised RL–based ones?

- The paper mentions hard constraints in the maze environment. Does this mean that agents are physically prevented from crossing walls? If so, the constraint is already enforced by the environment dynamics—why does it still need to be considered in decision-making?

[1] Janner, Michael, Qiyang Li, and Sergey Levine. "Offline reinforcement learning as one big sequence modeling problem." Advances in neural information processing systems 34 (2021): 1273-1286.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
**Summary.**
This work introduces RADT, Reach-Avoid Decision Transformer, a novel safe and goal conditioned RL framework via a tokenized conditioning. The RADT is built upon a decision transformer and encodes both goals and avoid regions as prompt tokens. The model is trained on suboptimal, random-policy offline data using hindsight avoid-region relabeling to enable reward-free learning. The method is benchmarked on robotics and biology domains and claims robust zero-shot generalization to out-of-distribution avoid region configurations.

**Review summary.** This paper presents a sound but incremental approach to safe goal-conditioned offline RL through prompt-based decision transformers. While the motivation and the overall structure is clear, the proposed method relies heavily on established hindsight relabeling techniques and assumes access to precise geometric knowledge of unsafe regions, which limits its practical realism. The evaluation design, i.e., allowing violations without terminatio, and the use of only three random seeds, reduce confidence in the reported performance. Methodologically, the success-indicator token is non-deployable in real-time inference, and several modeling choices are insufficiently analyzed. Besides, the reviewer have a suspicion that some of the entire paragraph level was written using LLM at the moment. Therefore, the reviewer assigns a preliminary score of 2 and would reconsider this rating if the authors provide deeper theoretical justification, stronger baselines, and more rigorous empirical validation.

### Strengths
**Writing**
- The paper is well-structured.
- Fig. 2 well illustrates the prompt pipeline. 
- The authors clearly develop the motivation behind the necessity of each technology and the problem setting.
    - Prompting as decoupling mechanism. prompt tokens decouple task spec from the state and enable test-time conditioning.
    - Zero-shot matters. The narrative ties zero-shot generalization to realistic deployments where avoidance constraints vary.

**Method**
- The authors explains embeddings and ordering of prompt tokenization clearly.
- $k_t$ prediction and loss encourage explicit awareness of being inside/outside avoid boxes.

**Experiments**
- Table 1 and 2. RADT generalizes to unseen avoid sizes, and unseen counts, often matching or beating retrained baselines. In addition, they shows the performance when extrems up to $20$ avoid regions in Table 7.
- Table 5 and 6. This work includes enough ablations, removing SI, AG, GS degrades MNC. 
- Table 9. Alternative spherical encoding shows similar trends
- It is interesting that both the biological domain and the robotics domain are different.

### Weaknesses
**LLM**
- I found three times **\*something\*** letters. Although using LLM to correct the grammar and rephrase some sentences, regarding the appendix, the reviewer is concerned that some of the entire paragraphs may have been written by LLM.
    - Line 422: We then run a second set of 200 episodes, this time providing the most visited intermediate state as an **\*avoid token\*** in the prompt.
    - Line 1229 and 1236: **\*larger, OOD\***  and **\*adversely\***

---
**Writing**
- "Avoid" is a verb. Something is not "avoid-region," it is "unsafe/hazardous/dangerous/etc region or condition," which made it a bit clunky to read when this is done so frequently.
- Notation for prompts and embeddings could be formalized more cleanly to aid readers unfamiliar with tokenized sequential modeling.
- Some figures are redundant, for example, Figures 5 and 7 are overlapped, and could be consolidated for clarity. 
    - Make sure the text within the figure is at a similar level to the text itself.
    - Make sure the mathematical notation in the figure is the same as in the main text.
    - In Figure 3 and 4, there are cases where there is no error bar or the error bar is not distinct. 
    - In Appendix C.2, please make a table for organizing binary state representation, e.g., initial state - 000000000000000. 
- The reviewer thinks that **Properties** described in Section 2 are more accurately assumptions and problem formulation rather than properties.
    - The authors would benefit from an explicit discussion of when these assumptions (properties) might fail.

---
**Methodology**
- The method presumes precise and vectorized knowledge of avoid-regions at inference time. The reviewer thinks that this is a strong perception-free assumption, limiting applicability in realistic sensory environments.
- The reviewer thinks that hindsight avoid-region relabeling is incremental, and it seems like engineering improvement, a expansion of goal relabelling into avoid-region.
- The SI is offline/hindsight information. At evaluation, authors always condition on $z=1$ or $z=0$, which cannot be known online and therefore is not a deployable input.
- They set the joint loss coefficient as a fixed $\alpha=1$ with the claim that effects vanish with enough training. However, the impact of tuning this hyperparameter is not deeply evaluated and provided.
- RADT allows prompts of effectively unbounded length, but the impact of prompt length or order on policy inference is not discussed or ablated.

---
**Experiments**
- While RADT maintains strong goal-reaching SR for up to 20 avoid regions, MNC begins to degrade notably past the regime seen in training. The reivewer thinks that this results suggest the zero-shot generalization is limited and could be sensitive to further task scaling.
- Environments allow passing through avoid regions and do not terminate episodes on violation at evaluation. The authors mention that SR can remain high while MNC is non-trivial, decoupling safety from success. The reviewer thinks that this is non-fair evaluation.
- It is hard to strongly trust empirical results due to the low number of seed (3 seeds).
- Baselines are retrained for each configuration, which structurally disadvantages them against RADT’s flexible prompting. More recent constrained- or prompt-DT baselines would strengthen the case [1, 2, 3].
- No wall-clock/params/memory comparison vs. AM-Lag/RbSL/WGCSL, making deployability hard to assess.

---
**Suggested references**

[1] M. Xu, et al. Prompting Decision Transformer for Few-Shot Policy Generalization. ICML 2022.

[2] C. Cao, et al. Offline Goal-Conditioned Reinforcement Learning for Safety-Critical Tasks with Recovery Policy. ICRA 2024.

[3] H. Lin, et al. Safety-aware Causal Representation for Trustworthy Offline Reinforcement Learning in Autonomous Driving. RA-L 2024.

### Questions
* Can the authors provide quantitative comparisons of resource footprint between RADT and baselines?
* How does the model handle scaling to situations where avoid regions have complex geometry or there is significant spatial/temporal overlap between many avoid regions?
* What is the effect of prompt length on inference time, training convergence, and attention allocation?
* How sensitive is the model to the weighting parameter $\alpha$ between the action and avoid-awareness losses?
* Could dynamic weighting or alternative normalization improve outcomes, particularly in OOD settings?
* Can the relational inductive bias for structured avoidance be encoded into the prompt or architecture, and does it help?

### Soundness
2

### Presentation
1

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
This paper introduces RADT (Reach-Avoid Decision Transformer), an offline RL framework for goal-conditioned reach-avoid tasks that addresses flexible specification of avoid regions at evaluation time. The method uses a prompting-based approach where goals and avoid regions are encoded as discrete tokens in the input sequence to a causal transformer.

### Strengths
- Encodes both goals and avoid regions as prompt tokens, decoupling reach-avoid specifications from state representation and enabling zero-shot generalization to arbitrary avoid region counts, locations, and sizes
- Creative strategy that generates trajectory pairs with opposite avoid success labels, allowing the model to learn from both successful and unsuccessful demonstrations without requiring expert data
- Eliminates brittle reward/cost function design by learning directly from hindsight-relabeled trajectories, addressing practical challenges of balancing conflicting objectives

### Weaknesses
- Only 2 robotics environments with relatively simple geometric constraints; no high-dimensional state spaces (e.g., pixel observations) or complex avoid region shapes beyond boxes/spheres
- Training time (72 GPU hours mentioned in appendix), memory overhead, and model size (GPT-2 architecture) not compared against baselines; acknowledged in limitations but critical for practical deployment
- RbSL/AM-Lag baselines use impassable obstacles (Figure 3b) rather than passable avoid regions, making direct comparison questionable
- Table 8 shows introducing expert trajectories improves avoid performance on OOD box sizes but degrades goal-reaching SR, suggesting reliance on random-policy data may limit ceiling for certain configurations

### Questions
1. What is the wall-clock training time, memory usage, and inference latency compared to RbSL/AM-Lag? Is the zero-shot capability worth the computational overhead?

2. How would RADT handle complex, non-convex avoid regions (e.g., L-shaped obstacles, multiple disconnected regions)? Could you use learned embeddings instead of hand-crafted box/sphere representations?

3. Since RbSL/AM-Lag use impassable obstacles (no training data violates avoid regions), can you evaluate all methods on the same passable-obstacle setup to ensure fair comparison?

4. How does RADT perform with pixel observations or point cloud inputs? Would you need to learn avoid region embeddings from visual features?

5. What is the maximum number of avoid regions RADT can handle before prompt length becomes prohibitive? Does attention mechanism degrade with very long prompts?

### Soundness
3

### Presentation
3

### Contribution
3
