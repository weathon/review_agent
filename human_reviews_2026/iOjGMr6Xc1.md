# Learning Instruction-Following Policies through Open-Ended Instruction Relabeling with Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4

## Abstract
Developing effective instruction-following policies in reinforcement learning remains challenging due to the reliance on extensive human-labeled instruction datasets and the difficulty of learning from sparse rewards. In this paper, we propose a novel approach, Open-ended Instruction Relabeling (OIR), that leverages the capabilities of large language models (LLMs) to automatically generate open-ended instructions retrospectively from previously collected agent trajectories. Our core idea is to employ LLMs to relabel unsuccessful trajectories by identifying meaningful subtasks the agent has implicitly accomplished, thereby enriching the agent's training data and substantially alleviating reliance on human annotations. Through this open-ended instruction relabeling, we efficiently learn a unified instruction-following policy capable of handling diverse tasks within a single policy. We empirically evaluate our proposed method in the challenging Craftax environments, demonstrating clear improvements in sample efficiency, instruction coverage, and overall policy performance compared to state-of-the-art baselines. Our results highlight the effectiveness of utilizing LLM-guided open-ended instruction relabeling to enhance the instruction-following abilities through reinforcement learning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Open-Ended Instruction Relabeling (OIR), a framework that leverages large language models (LLMs) to automatically generate new natural language instructions from agent trajectories in reinforcement learning (RL). The key idea is to use LLMs to retroactively relabel both successful and failed trajectories with meaningful instructions that reflect the subtasks the agent implicitly accomplished. This creates a richer and more diverse dataset of instruction–trajectory pairs without human annotations, mitigating sparse reward issues.
The method integrates LLM-based relabelling with a prioritised instruction replay buffer and an embedding-based reward function based on cosine similarity. Experiments on the Craftax benchmark demonstrate improved sample efficiency, task coverage, and generalisation to unseen instructions compared to several baselines.

### Strengths
* **Clear motivation and presentation:** The paper is well written and logically structured, with clear explanations of the method and experimental setup.
* **Strong empirical results:** OIR achieves notable improvements in sample efficiency, number of tasks completed, and generalisation performance compared to strong baselines.
* **Comprehensive evaluation:** The authors include detailed analyses across three axes—efficiency, generalisation, and diversity—and conduct ablations on threshold sensitivity, LLM backbone, and buffer sampling strategies.
* **Technical soundness:** The formalisation of the instruction relabelling process and the integration with prioritised replay seem technically sound. The inclusion of algorithm pseudocode, hyperparameters, and open-source code improves reproducibility.
* **Practical impact:** The approach offers a scalable way to bootstrap instruction-following agents without human labels, which could be useful for developing autonomous open-ended RL systems.

### Weaknesses
* **Limited conceptual novelty:** While the combination of hindsight relabelling and LLM-based generation is interesting, the approach mainly extends existing ideas (HER, LLM-guided labelling, and semantic reward shaping) rather than introducing a fundamentally new principle. The paper’s main contribution lies in integration of existing ideas and empirical demonstration rather than theoretical innovation.
* **Dependence on LLM quality:** The success of the approach depends heavily on the quality of LLM-generated instructions, as demonstrated in Figure 4. 
* **Single-domain evaluation:** All experiments are limited to the Craftax environment. While it serves as a good benchmark, evaluating OIR on a another domain would strengthen the generality claims.
* **Auxilary formalism:** Equations (5,6) formalise the notion of a “good” relabelling but are not directly used in the algorithm or experiments; they serve more as conceptual motivation than operational definitions.

### Questions
1. What are the reasons for Gemma3-1B-IT relabelling to work significantly worse than the Qwen models? Have you compared qualitatively the generated instructions by the two models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes Open-ended Instruction Relabeling (OIR), which uses LLM-based semantic reasoning to improve training of instruction-following RL policies. The method converts trajectories into textual observations, prompts an LLM to propose instructions implicitly achieved in those trajectories, and then relabels data with these instructions. Rewards for generated instructions are computed via cosine similarity between instruction and state embeddings, with episode termination when the similarity crosses a threshold. Experiments on Craftax-Classic show improved efficiency and generalization over baselines built on PQN and ELLM.

### Strengths
- The paper is clearly written, well structured, and easy to follow.

- The evaluation explicitly tests generalization to paraphrased and compositional instructions (simple and complex variants), not just the original instruction set.

### Weaknesses
- The approach presumes environments that can provide or be mapped to textual observations to prompt the LLM. A clearer statement of the environment class (symbolic / text-describable state, discrete action space, sparse achievements) would help understand the limit of the contribution.
- Results are reported on only one environment (Craftax-Classic), which limits claims of generality and leaves open whether gains depend on environment-specific engineering.
- The comparison to only ELLM and PQN is insufficient to measure the method performance. Two categories of baseline are missing: (i) state-of-the-art Craftax agents (see [the leaderboard](https://github.com/MichaelTMatthews/Craftax)), and (ii) LLM/VLM-based methods (both pretrained and RL fine-tuned) such as :
    - (2024) Tan, Weihao, et al. "True knowledge comes from practice: Aligning llms with embodied environments via reinforcement learning.”
    - (2023) Zitkovich, Brianna, et al. "Rt-2: Vision-language-action models transfer web knowledge to robotic control.”
    - (2022) Yao, Shunyu, et al. "React: Synergizing reasoning and acting in language models.”
- The related work should include prior efforts that use LLMs to *generate new instructions for RL training, which are closely related to this paper’s approach. Such as :
    - (2024) Qi, Zehan, et al. "Webrl: Training llm web agents via self-evolving online curriculum reinforcement learning.
    - (2023) Xu, Can, et al. "Wizardlm: Empowering large language models to follow complex instructions.”

### Questions
- In §4.1 you note LLMs may propose inaccurate/misleading instructions. What fraction of generated instructions are flawed in practice, and how sensitive is training to this rate? Please detail the rule-based instruction filters you add and any safety checks to avoid harmful/detrimental instructions.
- How many semantically unique instructions are generated during training, and does this number plateau? Can you report a diversity curve (e.g., unique instructions vs. steps) and relate it to performance?
- What additional compute overhead does your method introduce compared to vanilla PQN?
- Are the state/instruction embedding functions frozen or trained for Craftax? How reliably does cosine similarity track ground-truth success across varied generated instructions?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Open-ended Instruction Relabeling (OIR), which leverages the capabilities of large language models (LLMs) to automatically generate open-ended instructions retrospectively from previously collected agent trajectories. Based on the general idea of hindsight relabeling, unsuccessful trajectories are relabeled using LLMs by identifying meaningful subtasks the agent has implicitly accomplished. This LLM-based labeling procedure enriches the agent's training data and substantially alleviate the reliance on human annotation. Experiments on Craftax environments demonstrate that OIR improves in sample efficiency, instruction coverage, and success rate compared to some baselines.

### Strengths
- The paper is well written and well-motivated. 

- The idea of using LLM to label trajectory in RL is well implemented. 

- The experimental results are well presented.

### Weaknesses
- Leveraging Large Language Models to label data is a very general idea that has been investigated in various domains. This diminishes the novelty of the paper.

- The benchmark is limited to Craftax. Therefore, it is hard to tell the generalization ability of OIR to other environments (especially larger game environments, such as Minecraft.)

- The overall method of OIR looks ad hoc: the prompt, the relabeling of Failed Trajectories, reward definition, etc. Hence, it is necessary to test it on more benchmarks. (the previous weakness)

- the performance of OIR is very sensitive to the cosine-similarity threshold $\delta$ (Figure 4)

- the effect of the instruction-buffer sampling strategy, i.e., prioritized instruction replay, seem very marginal compared to uniform sampling. (Figure 4)

### Questions
- Is OIR able to generalize to other game environments without much redesign of the different components in OIR? Can you provide some evidence?

- Is there some guideline on how to set the cosine-similarity threshold $\delta$?

### Soundness
3

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
5

### Summary
This paper proposes Open-ended Instruction Relabeling (OIR), a new method to improve instruction-following policies in reinforcement learning by reducing the need for human-labeled data. OIR uses Large Language Models (LLMs) to retrospectively analyze agent trajectories, including unsuccessful ones, and automatically generate open-ended instructions for subtasks the agent implicitly completed. Evaluated in Craftax environments, this approach enhances training data, improves sample efficiency, and results in a more capable, unified instruction-following policy compared to baseline methods.

### Strengths
1: The paper presents a novel paradigm for training open-ended instruction-following agents in a clear and logically coherent way. 

2: The experimental results clearly demonstrate that the proposed method outperforms the baselines on the majority of tasks. 

3: This method significantly improves the model's generalization capability.

### Weaknesses
1: The experimental evaluation is conducted in only one environment. It is recommended to further validate the method in more environments, such as the vanilla MineCraft or Robotics.

2: The paper lacks a detailed discussion of the observed performance degradation in the experiments. Is the trade-off between this decline in performance and the improvement in semantic representation truly justified?

3: The scalability of the proposed method has not been discussed.	

4: Could the authors provide a more detailed analysis of the semantics generated by OIR for instructions that do not correspond to explicit environment achievements? Specifically, what behaviors do these semantics represent, and how does learning such semantics contribute to the performance or generalization ability of a multi-task agent?

### Questions
See in weakness.

### Soundness
2

### Presentation
2

### Contribution
2
