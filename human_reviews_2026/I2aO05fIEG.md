# REX-RAG: Reasoning Exploration with Policy Correction in Retrieval-Augmented Generation

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
Reinforcement learning (RL) is emerging as a powerful paradigm for enabling large language models (LLMs) to perform complex reasoning tasks. Recent advances indicate that integrating RL with retrieval-augmented generation (RAG) allows LLMs to dynamically incorporate external knowledge, leading to more informed and robust decision making. However, we identify a critical challenge during policy-driven trajectory sampling: LLMs are frequently trapped in unproductive reasoning paths, which we refer to as "dead ends", committing to overconfident yet incorrect conclusions. This severely hampers exploration and undermines effective policy optimization. To address this challenge, we propose **REX-RAG** (**R**easoning **EX**ploration with Policy Realignment in **R**etrieval-**A**ugmented **G**eneration), a novel framework that explores alternative reasoning paths while maintaining rigorous policy learning through principled distributional corrections. Our approach introduces two symbiotic innovations: **(1) Mixed Sampling Strategy**, which combines a novel probe sampling method with exploratory prompts to escape dead ends; and **(2) Policy Correction Mechanism**, which is essential for correcting the distributional shifts introduced by exploration. REX-RAG demonstrates that effective exploration is only viable when paired with such a rigorous correction. We evaluate it on seven question-answering benchmarks, and the experimental results show that REX-RAG achieves average performance gains of **5.1%** on Qwen2.5-3B and **3.6%** on Qwen2.5-7B over strong baselines, demonstrating competitive results across multiple datasets. Anonymous repository is provided on https://anonymous.4open.science/r/REX-RAG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
During reinforcement learning (RL) training, “dead ends” — consistently failing trajectories — often occur. Simple self-reflection does not effectively address this issue, as it tends to only slightly perturb the original path without resolving the underlying problem. To mitigate this, the proposed method introduces several design considerations. First, it employs a mixed sampling strategy, which samples from both the current policy and a probe policy. Second, it includes a policy correction mechanism to account for the distribution shift introduced by exploration.

### Strengths
1. The paper is well-motivated, clearly identifying a key limitation of existing self-reflection methods and proposing a principled approach to address it.

2. The paper is clearly written and easy to follow, with well-structured explanations of both intuition and methodology.

### Weaknesses
1. It is unclear whether the prompts used in “Construction of the Probe Policy” were also used as the prompts for the self-reflection baseline shown in Figure 1. Additionally, does the self-reflection baseline involve training with self-reflected responses, or is it only evaluated with self-reflected responses without training? This distinction is important because the main difference between the proposed mixed sampling strategy and prior self-reflection approaches is not entirely clear. Some previous works also correct answers using self-reflection and incorporate these corrected responses during training. Alternatively, is this work applying self-reflection in parallel to the sampled generations within GRPO? This needs to be clarified in the paper.

2. The experimental section lacks a comparison with existing self-reflection baselines, which are necessary to contextualize the improvements.

3. The ablation study only investigates components within the policy correction mechanism. It should also include an ablation study evaluating the contribution of the mixed sampling strategy.

### Questions
1. Why does this work focus solely on the RAG problem? The overall pipeline appears to be general and potentially applicable to a broader range of tasks.

2. Please provide the clear distinction with existing self-reflection works (especially those using self-reflection in training).

3. Please provide the ablation study results of the full pipeline.

### Soundness
3

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
5

### Summary
Overall, the paper is interesting and presents a novel idea. It proposes an agentic RL framework that teaches an LLM how to reflect and improve upon its own reasoning. The method leverages a probe policy that, given a trajectory leading to a dead end, generates a new trajectory with reflective reasoning to potentially correct the original mistake. The idea is original, and the overall modeling and writing are clear and well-structured.

However, the main issue is that the authors appear to have omitted the appendix from the submission, which is a serious oversight. This omission makes it difficult to fully understand the paper, particularly regarding several important implementation details.

In summary, I find the paper promising and would be inclined to raise my score to acceptance if the authors can provide the missing appendix and clarify the implementation details during the rebuttal period.

### Strengths
- The paper presents an interesting and well-written idea.

- It addresses a compelling question in agentic reinforcement learning: how can we enable LLMs to learn to reflect, or more generally, when introducing an external policy (e.g., for reflection), how can we ensure the policy we want to learn remains on-policy? The paper provides reasonable and well-motivated solutions, including (1) filtering and (2) distribution realignment.

- The experimental evaluation is comprehensive and thoughtfully designed, effectively exploring several important questions related to the framework’s design choices.

### Weaknesses
- The appendix pages are missing, which makes it difficult to fully understand several key parts of the work.

- The training process is somewhat complicated and not clearly explained, leading to confusion. For example, it is unclear what the complete training pipeline looks like — whether the two policies are trained jointly or sequentially, and whether they share parameters. 

- The ambiguity in describing the training procedure, along with the complexity of the overall training pipeline, makes it difficult to reproduce or fully evaluate the proposed method.

### Questions
### Questions for the Authors

- Regarding the definition of *dead ends*: besides trajectories that end with `<answer> ... </answer>`, are there other types of dead-end trajectories considered?  

- There is confusion between the *current policy* and the *probe policy*. Does the probe policy $\pi_\epsilon$ share the same parameters as $\pi_\theta$ (only using different prompts), or are they two separate models with distinct parameters?  

- For the probe policy, after identifying a dead end, are the subsequent rollouts — including reasoning, search, and answer generation — produced by $\pi_\epsilon$?  

- What is the exact training procedure? Are $\pi_\theta$ and $\pi_\epsilon$ trained jointly, or is $\pi_\theta$ first warmed up and then used to train the probe policy?  

- Equation (5) defining the probe policy is difficult to interpret. Could the authors provide more explanation for this formulation, particularly why the denominator involves $z^{\frac{1}{|o'_{\text{origin}}|}}$ when $o'_{i,t} \in o'_{\text{origin}}$? What is the intuition behind this design?  

- In the ablation study, what exactly is *coarse-PPD*? A one-sentence description is insufficient to understand the difference — please clarify what it looks like in detail.  

- It would be helpful to include more **case studies** comparing the behaviors of Search-R1 and the proposed model, to better illustrate their differences.  

- During the reflection process, how many new search actions are typically performed?

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
This paper investigates the dead-end problem in RAG systems trained with RL, where the model often gets stuck in incorrect reasoning paths and fails to explore new directions. To overcome this, the authors propose REX-RAG, a framework that introduces a Mixed Sampling Strategy to inject exploratory prompts and guide the model toward more diverse reasoning trajectories. However, such exploration may cause distributional shifts that make RL training unstable, so a Policy Correction Mechanism is further designed to re-weight the exploratory data using trajectory filtering and multiple importance sampling, keeping the optimization process stable and unbiased. Experiments on several question-answering benchmarks show that this approach brings consistent performance gains, and ablation results confirm the importance of the correction mechanism for effective exploration.

### Strengths
The proposed method seems sound and effective based on the reported results, which makes it convincing that this approach is useful for training policies that better interact with search engines (or tools). Additionally, the experiment setup is very comprehensive and is accompanied by a good set of ablations that clarify the effectiveness of each component in the system.

### Weaknesses
The proposed method is significantly more expensive than the baselines, specifically the most similar baseline, search-r1. It is not clear for me the improvements observed here are from the increased number of sampling during training or because of the sampling strategy. We know that number of rollouts in the training can significantly increase the compute budget of training and improving performance. I am curios to  see if this method still performs better than search-r1 if it uses the same number of rollouts (including initial and exploratory rollouts).

### Questions
What happens if you assign the same exploration budget (n) to the policy model of search-r1? Would it still downperform your model? Or in other words, I would like to see how your model performs if it can only sample (including exploratory sampling) the same as search-r1? Would that affect your findings?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces REX-RAG, a retrieval augmented generation framework to address the dead end problem in RL-based RAG training. That is, when RL rollouts results in incorrect reasoning paths but the policy is unable to self-correct. The proposed REX-RAG consists of (1) a sixed sampling strategy that uses a so-called "probe sampling" to help the model increase rollout sample size to avoid dead ends; and (2) policy correction learning that introduces techniques such as trajectory filtering and multiple importance sampling to stabilize RL training with the introduced rollouts & additional corrections introduced by REX-RAG. The authors evaluate the proposed method on several QA benchmarks, where REX-RAG achieved consistent improvements over RL-based RAG baselines.

### Strengths
1. The authors study an important problem of dead end in RL-based RAG settings, where the policy is often unable to generate correct reasoning paths for complex input queries.

2. The authors introduces effective techniques to improve the training dynamics on the rollouts and additional correction continuations by the probe policy.

3. REX-RAG shows strong performance on open-domain QA datasets, suggesting the model learns improved reasoning patterns for multi-turn search LLMs.

### Weaknesses
1. The entire exploration and correction mechanism is training-only, as it relies on ground truth labels (rather than learning a verifier) to identify dead ends and trigger exploration. Therefore in inference, these mechanisms are deactivated and the model cannot really correct itself if it heads down to incorrect reasoning paths.

2. The exploration & additional sampling introduces extra computation overhead during the training phase, which may be potentially unfair to baselines like Search-R1 which adopts fixed group size in training. Although the authors provide additional results with over-sampling using DAPO, these results are inconsistent (Search-R1 outperforms REX-RAG) and does have dataset-specific results.

3. Some technical details are missing in writing. For example, although I can image how these are computed, the authors should provide a formula to show how PMF is computed in Eq. 5.

### Questions
1. For the exploration prompt sampled from a curated prompt pool, are these tokens masked out in policy update or are they also included in training as in Eq. 5-7?

2. Can you provide more dataset-specific results with DAPO on Search-R1 and REX-RAG?

### Soundness
3

### Presentation
2

### Contribution
2
