# SEAgent: Self-Evolving Computer Use Agent with Autonomous Learning from Experience

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 6, 4

## Abstract
Repurposing large vision-language models (LVLMs) as computer use agents (CUAs) has led to substantial breakthroughs, primarily driven by human-labeled data. However, these models often struggle with novel and specialized software, particularly in scenarios lacking human annotations. To address this challenge, we propose SEAgent, an agentic self-evolving framework enabling CUAs to autonomously evolve through interactions with unfamiliar software. Specifically, SEAgent empowers computer-use agents to autonomously master novel software environments via experiential learning, where agents explore new software, learn through iterative trial-and-error, and progressively tackle auto-generated tasks organized from simple to complex. To achieve this goal, we design a World State Model for step-wise trajectory assessment, along with a Curriculum Generator that generates increasingly diverse and challenging tasks. The agent's policy is updated through experiential learning, comprised of adversarial imitation of failure actions and Group Relative Policy Optimization (GRPO) on successful ones. Furthermore, we introduce a specialist-to-generalist training strategy that integrates individual experiential insights from specialist agents, facilitating the development of a stronger generalist CUA capable of continuous autonomous evolution. This unified agent ultimately achieves performance surpassing ensembles of individual specialist agents on their specialized software. We validate the effectiveness of SEAgent across five professional software of OSWorld, ScienceBoard and AndroidWorld. Our approach achieves a significant improvement  over a competitive open-source CUA, UI-TARS. All the code and models will be made publicly available to foster further research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an agentic self-evloving pipeline for learning autonomous computer use agent in software environments without human-labeled data.

The pipeline consists of several key components: (1) a World State Model to provide step-level reward signals for RL training and state change captioning as well as the trajectory-level judgement for maintaining a software guidebook for a Curriculum Generator; (2) a Curriculum Generator that automatically generates new task instructions for curriculum learning of the agent; (3) a training process involving adversarial imitation of failure actions and GRPO based on successful actions; (4) a specialist-to-generalist training strategy that  involves SFT on successful trajectories from specialist agents first and then RL across multiple softwares.

Experiments on OSWorld, ScienceBoard and AndroidWorld show the effectiveness of proposed pipeline.

### Strengths
1. This work identifies several key elements contributing to the development of autonumous agents for computer use without relying on annotated data, providing insights for practical implementation of such agents. The key elements include:

(1) a new task generation strategy using a Curriculum Generator;

(2) a new reward model to assess both the step-level actions and trajectory-level success using a World State Model;

(3) a new training objective combining both adversarial imitation of failure actions and RL (using GRPO).

(4) a specialist to generalist strategy that outperforms both direct RL in multiple softwares and RL in each environment.

2. Extensive experiments are conducted to justify different aspects, including the effectiveness on different benchmarks and using different actor models, the performance of World State Model compared to other models as judges, the comparison between Specialist RL, General RL and Specialist-to-Generalist, the performance gains during different training phases and ablation studies to show the usefulness of different components in the poposed pipeline.

### Weaknesses
1. The writing of the paper is too poorly, requiring many efforts to revise and polish.

(1) Too many grammatical errors, making it hard to read. 

(2) Captions of most Tables and Figures are unclear, lacking explanations to details in the Table or Figure. For example, the meaning of different colors of the curves in Figure 4. 

(3) The experimental setup is missing in the main paper, including benchmarks, baselines, evaluation metrics, and implementation details, making it hard to make a fair comparison between methods (although I can find some in the appendix).

(4) Some notions or details are not explained. For example, the explanation of *t* in Eq. 5, and how to construct the training data for GRPO? For other issues, please see questions below.

2. The introduction and comparison with related work is insufficient. 

(1) The paper claims that "SEAgent achieves superior performance, which can be attributed to World State Model providing fine-grained, step-level rewards from the full history". However, similar ideas has been justified in [1] but not referred to, where a process reward model is prompted by the full interaction history.

(2) Some related work on GUI agents are missing and not discussed, e,g., [2][3]

3. It seems that the training sample for GRPO is a one-step data from a trajectory instead of a whole trajectory. Why use the single-turn RL in an agentic setting? What if the proposed training method is compared to multi-turn RL approaches?

References:

[1] [EPO: Explicit Policy Optimization for Strategic Reasoning in LLMs via Reinforcement Learning](https://aclanthology.org/2025.acl-long.747/) (Liu et al., ACL 2025)

[2] Luo, Run, et al. "Gui-r1: A generalist r1-style vision-language action model for gui agents." arXiv preprint arXiv:2504.10458 (2025).

[3] Chen, Kevin, et al. "Reinforcement learning for long-horizon interactive llm agents." arXiv preprint arXiv:2502.01600 (2025).

### Questions
1. Why do performance gains saturate beyond the thrid training phase?
2. What do the ensembled results of specialized RL mean in Table 2? How to understand the "individual specialist ensemble" in line 429?
3. Does **General RL** refer to training a single agent in all five environments simultaneously? If so, how to train a single agent in different environments through RL? And it seems that the only difference between **General RL** and **Specialist-to-Generalist** is that the latter involves SFT initialization before RL.
4. In all the experimental results, does SEAgent refer to SEAgent (Specialist-to-Generalist) by default?
5. There is an severe error in the paragraph of **Ablation Study of Specialist Training**, of which the content is not the analysis of ablation results. Also, how to understand Table 4? For example, Does Qwen2.5VL-72B refer to Curriculum Generator? What does the result in the first row mean?
6. The result in the last row of Table 4 is 46.1, which equals to that of SEAgent (Specialized RL) in Table 2. What are the differences between the two experiments in terms of its configuration?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces SEAgent, an agentic self-evolving framework designed to enhance the adaptability of Computer Use Agents (CUAs) operating in previously unfamiliar software environments. The framework enables agents to engage in autonomous exploration and experiential learning, supported by a world state model and a curriculum generator. SEAgent optimizes its policy by using adversarial imitation and GRPO methods. With a specialist-to-generalist training strategy, SEAgent shows significant performance improvements across five professional software applications from OSWorld.

### Strengths
### 1. **Innovation**
   - **Self-evolving framework**:  **SEAgent**, a **self-evolving framework** for autonomous exploration and experiential learning, is innovative in CUAs fields. It allows agents to autonomously generate tasks and assess their success/failure in previously unfamiliar software environments without human intervention, advancing the capabilities of autonomous systems.

### 2. **Task Generation and Evaluation Precision**
   - **World State Model**: The paper introduces the **World State Model**, which enables precise environmental state captioning and step-wise trajectory assessment. This model provides fine-grained reward signals, significantly improving the precision of task evaluation.

### 3. **Experimental Results and Performance Improvements**
   - **Multi-software environment adaptability**: SEAgent demonstrates excellent adaptability, not only excelling in single-software environments but also in **multi-software scenarios**. Its validation across five professional software applications from **OSWorld** showcases its generalization.

---

### Weaknesses
1. **Inconsistency in the Paper's Claims**  
   The paper's claims are not fully self-consistent. Although the paper's title suggests **self-evolving** agents and emphasizes the exploration of LVLMs for autonomous exploration, the **World State Model** used for exploration in this study still relies on **human-annotated high-quality datasets**. This contradicts the starting point outlined in **line 53**, which advocates for agents to evolve without such dependencies. The paper should clarify this contradiction between the claimed self-evolving nature of the agent and its reliance on human-annotated data.

2. **Lack of Novelty in Contributions**  
   The contributions of the paper lack sufficient innovation. The **self-taught**[1] research has already demonstrated that additional training using high-quality critic modules can guide the improvement of LLMs through **Supervised Fine-Tuning (SFT)**, but with limited gains from multiple rounds of training. The paper uses **Group Relative Policy Optimization (GRPO)**, a reinforcement learning method, for enhancement, but no additional conclusions are provided. One important question that remains unaddressed is whether more rounds of interaction could yield better results, as reinforcement learning theory suggests that continued interaction can lead to further improvements.  
   On the other hand, the **specialist-to-generalist distillation method** may not be necessary in this context. In traditional RL, such a method is useful due to the small size of the network, where **Catastrophic Forgetting** may occur. However, for large models, multi-domain data mixing can be achieved during pretraining, and there is no clear need for separate training followed by distillation during the fine-tuning phase. The paper lacks ablation studies to analyze this point of innovation.

3. **Experimental Setup Issues**  
   The experimental setup is not entirely reasonable.  
   - **Table 1** does not provide results for the World State Model under the **LS setting**, leaving this aspect of the model's performance unexplored.  
   - **Table 2** raises a concern: is the training data size consistent across **general RL**, **SFT**, and **specialist-to-generalist** methods? From the descriptions in the paper, it seems that the **specialist-to-generalist** approach would likely involve more training data, but this is not explicitly clarified.  
   - **Table 4**: The ablation study does not fully incorporate the innovative aspects discussed in **point 2**, specifically regarding the necessity and impact of the specialist-to-generalist strategy.

4. **Writing Problems**  
   The writing lacks rigor. For example, in **line 131**, the phrase should be "Recently, ... has achieved". Similarly, in **line 235**  "in Section [???]". Authors should check their writing more carefully.

[1] Wang, Tianlu, Ilia Kulikov, Olga Golovneva, Ping Yu, Weizhe Yuan, Jane Dwivedi-Yu, Richard Yuanzhe Pang, Maryam Fazel-Zarandi, Jason Weston, and Xian Li. "Self-taught evaluators." arXiv preprint arXiv:2408.02666 (2024).

### Questions
See weekness above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
To address the issue that CUAs often struggle with novel and specialized software, particularly in scenarios lacking human annotations, the authors propose SEAgent, an agentic self-evolving framework enabling CUAs to autonomously evolve through interactions with unfamiliar software. In addition, they introduce an effective specialist-to-generalist strategy for shaping a versatile generalist agent. The authors validate the effectiveness of their method across five professional software of OSWorld, ScienceBoard, and AndroidWorld.

### Strengths
- The lower part of Figure 1 presents the specialist-to-generalist training strategy with great clarity, and this strategy also provides valuable inspiration for progress in other domains. In general, each domain requires models to possess multi-dimensional capabilities, and the visualization and the demonstrated effectiveness of the proposed strategy in addressing this issue are particularly insightful and valuable.
- The paper provides a very clear introduction to the background of the task. Although I had not previously studied this field, I was able to gain a basic understanding of the task through reading the paper.
- Overall, the paper presents a carefully specific design of RL (such as reward signal), and the experimental section demonstrates a high degree of completeness.

### Weaknesses
- The difference between the specialist-to-generalist training strategy proposed in this paper and previous similar strategies requires further clarification.
- The readability of Section 3.1 and Figure 2 is not very good, and Figure 2 is not cited anywhere in the paper.
- There is a citation error at line 235.

### Questions
- Could the authors illustrate the workflow in Section 3 with a concrete example—specifically, how the evolution process unfolds and how each component functions?
- Is the term 'world state world' something you proposed? Or does the concept of world models also exist here?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes SEAgent, a self-evolving framework that enables computer-use agents to autonomously master novel software through curriculum task generation,  World State Model, and experiential learning. But why can a Large Vision-Language Model as a World State Model?

### Strengths
The method effectively reduces reliance on human-labeled data by allowing agents to learn from trial-and-error interactions in unfamiliar software environments.

### Weaknesses
1. If Large Vision-Language Models (LVLMs) can serve as World State Models, it means that LVLMs contain all the information for the application. So why not directly use LVLMs for decision-making? This completely contradicts the "without human intervention" description in Figure 1's caption and the motivation of this paper.
2. The World State Model relies on GPT-4o annotations, which may introduce bias and affect reproducibility.

### Questions
1. If Large Vision-Language Models (LVLMs) can serve as World State Models, it means that LVLMs contain all the information for the application. So why not directly use LVLMs for decision-making? This completely contradicts the "without human intervention" description in Figure 1's caption and the motivation of this paper.
2. The World State Model relies on GPT-4o annotations, which may introduce bias and affect reproducibility.
3. How does the World State Model ensure interpretability in its step-wise trajectory judgments, especially for ambiguous GUI states?
4. Why were certain software applications excluded from ScienceBoard, and how might this affect the validity of OOD performance claims?
5. How does the Curriculum Generator avoid generating repetitive or infeasible tasks as task complexity increases?
6. Can the reward model generalize to software with highly dynamic or non-standard GUI elements not seen during training?

### Soundness
2

### Presentation
2

### Contribution
2
