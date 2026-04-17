# Vintix II: Decision Pre-Trained Transformer is a Scalable In-Context Reinforcement Learner

- Decision: Accept (Poster)
- Scores: 6, 4, 2, 4

## Abstract
Recent progress in in-context reinforcement learning (ICRL) has demonstrated its potential for training generalist agents that can acquire new tasks directly at inference. Algorithm Distillation (AD) pioneered this paradigm and was subsequently scaled to multi-domain settings, although its ability to generalize to unseen tasks remained limited. The Decision Pre-Trained Transformer (DPT) was introduced as an alternative, showing stronger in-context reinforcement learning abilities in simplified domains, but its scalability had not been established. In this work, we extend DPT to diverse multi-domain environments, applying Flow Matching as a natural training choice that preserves its interpretation as Bayesian posterior sampling. As a result, we obtain an agent trained across hundreds of diverse tasks that achieves clear gains in generalization to the held-out test set. This agent improves upon prior AD scaling and demonstrates stronger performance in both online and offline inference, reinforcing ICRL as a viable alternative to expert distillation for training generalist agents.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper extends the Decision Pre-Trained Transformer (DPT), a large-scale transformer model, to diverse reinforcement learning domains using a decision-transformer-style objective. DPT integrates a flow-matching policy head to better model multi-modal continuous-action distributions and is trained on a large multi-domain dataset. The authors evaluate DPT both in offline settings (fixed demonstration contexts) and online settings (autoregressive conditioning on recent transitions). The authors publish the used large-scale dataset with over 700m transitions. The paper also draws a connection, empirically, to the posterior-sampling interpretation of in-context learning, suggesting that DPT behaves as a Bayesian in-context learner following the reasoning of Lee et al. (2023). Empirically, DPT achieves strong performance across most domains and outperforms existing generalist RL models such as Vintix and REGENT on several benchmarks.

### Strengths
- The paper is well-written, well-structured, and easy to follow. The dataset composition, architecture, and training setup are described clearly, and the figures are of high quality.
- The scale of the dataset and experiments is high. The construction of the model, as well as the dataset, covering 10 domains is a significant engineering effort. The resulting benchmark could be a valuable community resource.
- Replacing the Gaussian policy head with a flow-matching head for continuous actions is a natural and elegant design choice. 
- DPT performs consistently well on both seen and unseen tasks, with significant gains in the more complex domains.

### Weaknesses
**Novelty**:

The paper would benefit from a clearer articulation of what exactly differentiates DPT from prior large-scale decision-transformer architectures such as REGENT and Vintix. Currently, the novelty seems to lie mainly in the use of the flow-matching policy head and the dataset scale. A comparison table summarizing architectural and algorithmic differences (policy head, encoding choice, inference procedure, etc.) would make the contributions clearer.

**Ablation studies**:

I would appreciate more detailed ablations that would allow a more finegrained assessment of what architectural choices contribute most to performance. Primarily the choice of using a flow-matching algorithm for policy heads comes to mind here, but also the choice of embeddings and context ordering, or context length.

**Experiments**:

- it seems that the selection of experiments makes it somewhat hard to draw quantitative conclusions from the chosen implementation choices, because most of the included domains appear to be solves by all tested models. Only few (Meta-World and Bi-DexHands) provide sufficient difficulty to distinguish algorithms. 

- Although the results are strong, I could not infer whether all baselines were re-tuned on the same dataset and training budget. In particular, Vintix and REGENT may have been evaluated under different protocols. Clarifying this would improve the fairness of the comparison.

### Questions
- Could you provide ablations isolating the contribution of the flow-matching policy head relative to standard Gaussian heads?
- Could you clarify the experimental protocol for the baselines? Were baselines retrained on your dataset or reproduced from prior work?
- Do you plan to release the code publicly to support reproducibility and benchmarking?

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
This work presents an in-context reinforcement learning (ICRL) method that extends the Decision Pre-trained Transformer (DPT) architecture with flow matching to model complex output distributions. An empirical study on a large cross-domain dataset demonstrates the improved performance of this design compared to baselines.

### Strengths
- The writing is clear and detailed.
- The empirical study is comprehensive and fair.
- The method improves upon the original DPT without too much overhead or modification.

### Weaknesses
- The paper lacks a background section to formalize RL, ICRL, and flow matching, which are the focus of this work. 
- I don't think the removal of the next observation $o'$ is well-justified, as it is crucial for capturing the dynamics of the environment, especially when the context is randomly permutated. Without $o'$, RL algorithms that optimize the policy based on the reward signal would not work unless it's a bandit setting. In this case, the DPT is likely merely learning by imitation and using the contextual information for task identification. Even in imitation learning, I doubt removing the next observation will leave the learner intact because, as I pointed out previously, $o'$ is crucial for capturing the environment dynamics. I suspect that the authors' claim that removing it will not impact the performance is due to that the dynamics of the testbeds are somewhat consistent across tasks.

### Questions
- What is the source of the optimal actions used for training?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper presents the Decision Pre-Trained Transformer (DPT) with a flow-matching generative policy head as a scalable agent for cross-domain, in-context reinforcement learning. The authors create a new large, cross-domain dataset featuring over 700 million transitions across 10 domains, and successfully train the DPT agent to achieve high performance on unseen tasks in several domains. 

**Recommendation:**\
I recommend to reject. The paper is lacking in several fundamental areas, including the following: comparison with baselines, clarity of research methodology, and unconvincing results.

### Strengths
- The analysis in Section 4.4 is potentially interesting. 
- A sizeable new cross-domain benchmark and dataset is created. 
- High performance on unseen tasks in 6 out of 10 domains.

### Weaknesses
- Very little comparison with baselines
- No mention of the significance of the results. Not mentioning number of seeds, the standard deviation or confidence intervals for any methods, makes it impossible to judge any level of significance. 
- Unclear research methodology. There is no mention of hyperparameter tuning or strict validation - test set splits.
- Reproducability is lacking. The experimental details are very lacklustre. 
- Results not that strong. Only on a single domain (Meta-World) out of 10 does the performance on unseen tasks appear to be better than one of the baselines (Vintix), whilst appearing to be the same as the other baseline (REGENT). Furthermore, performance in the online and offline setting are overlapping in most domains, suggesting it doesn't actually learn in-context at all. This is corroborated by the mostly flat curves in Figure 5. 
- Overclaiming in the introduction. 
- Lack of background section makes it difficult to judge novelty.

### Questions
- Which contributions are novel, and in what way exactly?
- How was hyperparameter tuning performed? And how was contamination between validation and test sets avoided? 
- Several potential baselines were mentioned in the related works section, why did you not compare to them? Why are the existing baselines only evaluated on very few of the domains?
- How did you test for significance of the results? 
- Figure 3 shows mostly overlapping performance between the online and offline setting, and Figure 4 shows mostly no change in performance as context size increases. Do these two things suggest your agent is not actually learning in-context? 
- Figure 4 shows concentration of the action distribution for longer contexts, but how does this actually relate to performance? 


**Things to improve that did not impact decision:**
- Figure 2: What do the red lines and shaded regions indicate?
- Figure 3: How is the performance for the online setting defined? 
- Line 437: Performance on the Industrial-Benchmark domain does not improve significantly with prompt size as the shaded regions are completely overlapping. 
- If each domain requires its own pre-trained encoder, is it really a cross-domain setting?

### Soundness
1

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
5

### Summary
This paper extends pretrained Decision Transformer models to diverse multi-domain environments and integrates a flow-matching objective for action generation. The authors evaluate on 209 tasks across 10 domains and report that their approach (DPT with flow matching) outperforms baseline methods.

### Strengths
- The paper is clearly written and easy to follow.
 - The topic of leveraging past interaction data for generalizable policy learning is important and relevant to the community.

### Weaknesses
- The primary contribution appears to be the incorporation of a flow-matching objective into a Decision Transformer framework. However, flow-matching and related diffusion-style generation objectives have been extensively explored in prior work, which makes the contribution feel incremental.
 - The paper does not clearly specify the types of observations used in each domain. Since the tasks span diverse settings, it is important to clarify whether inputs are proprioceptive states, images, or mixed modalities. If the experiments rely solely on low-dimensional proprioceptive inputs, the significance and applicability to high-dimensional tasks may be limited.
 - Although the motivation emphasizes multi-domain generalization, the architecture groups tasks and encodes them using group-specific encoders. This raises the question of how much knowledge is actually shared across domains. A comparison with models trained on single-domain data would help clarify whether multi-domain training provides measurable benefit.

### Questions
- What observation modalities are used in each domain? Are the inputs entirely proprioceptive, or do any domains include image-based observations?
 - How does the performance of the multi-domain model compare to models trained separately for each domain? Is there measurable improvement from cross-domain data sharing?
 - Could the authors provide experiments or discussion on applying the model to high-dimensional visual inputs?

### Soundness
3

### Presentation
3

### Contribution
2
