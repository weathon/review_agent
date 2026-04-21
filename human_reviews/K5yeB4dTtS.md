# MLLM as Retriever: Interactively Learning Multimodal Retrieval for Embodied Agents

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
MLLM agents demonstrate potential for complex embodied tasks by retrieving multimodal task-relevant trajectory data. However, current retrieval methods primarily focus on surface-level similarities of textual or visual cues in trajectories, neglecting their effectiveness for the specific task at hand. To address this issue, we propose a novel method, MART, which enhances the performance of embodied agents by utilizing interaction data to fine-tune an MLLM retriever based on preference learning, such that the retriever fully considers the effectiveness of trajectories and prioritize them for unseen tasks. We also introduce Trajectory Abstraction, a mechanism that leverages MLLMs' summarization capabilities to represent trajectories with fewer tokens while preserving key information, enabling agents to better comprehend milestones in the trajectory. Experimental results across various environments demonstrate our method significantly improves task success rates in unseen scenes compared to baseline methods. This work presents a new paradigm for multimodal retrieval in embodied agents, by fine-tuning a general-purpose MLLM as the retriever to assess trajectory effectiveness. All the code for benchmark tasks, simulator modifications and the MLLM retriever is available at https://github.com/PKU-RL/MART.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper introduces MLLM As Retriever (MART), a novel approach designed to enhance the performance of embodied agents in complex tasks by improving multimodal retrieval of task-relevant trajectory data. Traditional retrieval methods often rely on surface-level similarities of textual or visual cues, which may not effectively capture the nuances required for specific tasks. MART addresses this limitation by fine-tuning a Multimodal Large Language Model (MLLM) retriever through interactive preference learning, allowing it to evaluate and prioritize trajectories based on their effectiveness for unseen tasks. Additionally, the paper presents Trajectory Abstraction, a mechanism that leverages MLLMs’ summarization capabilities to represent trajectories with fewer tokens while preserving essential information, enabling agents to better understand key milestones. Experimental results across various environments demonstrate that MART significantly improves task success rates on unseen tasks compared to baseline methods, consistently surpassing them by over 10%. The authors commit to releasing all benchmark task sets and simulator code modifications to facilitate further research.

### Strengths
1. Innovative Retrieval Method: The paper introduces a novel method by fine-tuning an MLLM as a retriever using interactive preference learning, moving beyond traditional similarity-based retrieval methods. This allows the retriever to assess trajectory effectiveness more holistically.
2. Trajectory Abstraction Mechanism: The introduction of Trajectory Abstraction is a significant contribution. By condensing trajectories into shorter representations while retaining crucial information, agents can process and utilize past experiences more efficiently.
3. Demonstrated Performance Improvements: The method substantially improves task success rates across various environments and unseen tasks, indicating its effectiveness and potential for broader applications in embodied AI.
4. Advancement in Embodied Agent Capabilities: By enhancing how agents retrieve and utilize past trajectories, MART contributes to developing more intelligent and capable embodied agents capable of handling complex, long-horizon tasks.
5. Commitment to Open Science: The release of benchmark task sets and simulator code modifications promotes transparency and allows other researchers to build upon this work, fostering collaboration in the community.

### Weaknesses
1. Efficiency Issues: The proposed method of using a Multimodal Large Language Model (MLLM) as a retriever may not be as efficient as conventional similarity retrieval methods, known for being fast and low-cost. Given that complex language model reasoning can already achieve high-quality planning decomposition and performance, the added cost of training and deploying an additional MLLM may not be justified. There is a need for verification to determine if the high cost is worth the potential benefits, including considerations of additional inference time and required training resources.

2. Potential Stability and Hallucination Problems: Replacing simple similarity calculations with judgments based on a language model introduces potential instability and hallucination issues inherent to language models. The MLLM’s retrieval process might be inconsistent, leading to unreliable action selection. It’s unclear how these issues are overcome in the system and whether effectiveness has been verified across multiple evaluation methods. Since MLLM retrieval effectively acts as a terminal evaluator to judge the best action, stability is critical.

3. Experimental Confusion: In Table 4, the ablation study comparing MLLM Retrieval and similarity retrieval methods shows minimal differences in Average Steps (AS) but noticeable differences in Success Rate (SR). Questions arise about the correlation between these two evaluation metrics. Specifically:

    a) Clarification of AS: What does AS (Average Steps) represent? Is it the number of iterations the agent system takes to succeed once?

    b) Difference in Metrics: Why is there a significant difference in SR but not AS between the two methods?

    c) Fairness of AS: Given the introduction of an additional language model, using AS may not be fair. It might be more reasonable to replace AS with the average inference time of the entire system to reflect the true cost and efficiency.

4. Unfair Comparison in Case Study: The case study presents success cases of the proposed MLLM Retrieval method alongside failure cases of the similarity retrieval method. This comparison may not be fair or sufficient to demonstrate the advantages of the proposed method. Providing success and failure cases for each method would offer a more balanced and comprehensive evaluation.

### Questions
1. Is the High Cost of Training an Additional MLLM Justified?
Could you verify whether the benefits of training and using an additional MLLM outweigh the high costs in terms of additional inference time and required training resources? Are there specific scenarios where the MLLM retriever significantly outperforms conventional similarity retrieval methods to justify its use?

2. How Did You Overcome Stability and Hallucination Issues?
Since language models can be unstable and prone to hallucinations, what methods did you employ to mitigate these issues in your MLLM retrieval system? Have you verified the effectiveness of your approach across multiple evaluation methods to ensure reliable performance?

3. Clarification on Evaluation Metrics in Table 4:

	a) What Exactly Does AS Represent? Could you clarify the meaning of AS (Average Steps) in your evaluation? Does it represent the number of iterations required for the agent to succeed once?

	b) Correlation Between AS and SR: Does Average Steps and Success Rate correlate? How do you interpret the results where AS shows minimal difference but SR differs more significantly?

	c) Fairness of Using AS as a Metric:
Considering the additional computational cost of incorporating an MLLM, would it be more appropriate to use the average inference time of the entire system instead of AS to ensure a fair comparison between methods?

4. Can You Provide Balanced Case Studies?
To ensure a fair comparison, could you provide both the success and failure cases for each method (MLLM Retrieval and similarity retrieval) in your case studies? This would help understand each approach's strengths and weaknesses more comprehensively.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces MART, a novel trajectory retrieval paradigm leveraging interactive learning to enhance embodied agents’ task performance. MART utilizes interaction-based feedback to identify the most effective trajectories and constructs preference pairs based on trajectory comparisons. These preference pairs are used to fine-tune a Multimodal Large Language Model (MLLM) retriever, prioritizing trajectories that improve task outcomes. Additionally, MART introduces Trajectory Abstraction, a mechanism using MLLMs’ summarization capabilities to condense trajectory representations by reducing token counts while preserving essential information, enabling agents to better understand task-relevant details. Experimental results across diverse environments demonstrate that MART significantly improves success rates in unseen tasks compared to various baselines. This work bridges the gap between general-purpose MLLMs and embodied task requirements, offering a new paradigm for multimodal trajectory retrieval for embodied agents.

### Strengths
1. The problem addressed in this paper—decision-making in embodied tasks—is a crucial area.
2. The writing in this paper is clear and well-structured, with each paragraph presenting ideas in a logical, cohesive manner. The authors effectively communicate complex concepts, making the paper accessible and easy to follow.
3. This paper introduces an innovative multimodal retrieval tool and selects appropriate baselines to validate the effectiveness of the proposed method.

### Weaknesses
1. The methodology in this paper shows some similarities with previous works, such as LLM-Planner (arXiv:2212.04088 ) and P-RAG(	arXiv:2409.11279). While there are innovations in the retriever design, the overall novelty of the approach is somewhat reduced.
2. The paper lacks some detailed analysis and deeper insights. For instance, the ablation study does not include experiments explaining HOW Trajectory Abstraction contributes to performance improvement

### Questions
1. It would be better if the authors could further explain the specific innovations in the pipeline compared to prior works LLM-Planner and P-RAG, highlighting how the proposed approach diverges or improves upon these method.
2. Could the authors provide an explanation for HOW Trajectory Abstraction contributes to performance improvements? 
3. If incorrect trajectories are retrieved, could this lead to catastrophic performance loss?

### Soundness
3

### Presentation
3

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
The paper presents a novel method, MLLM As ReTriever (MART), which aims to enhance the performance of embodied agents in complex environments by improving the retrieval of multimodal, task-relevant trajectory data. It uses expert trajectories as prompts for an MLLM agent, collects interactive feedback in the form of success rates, and organizes this data into preference pairs. These pairs are then used to fine-tune the MLLM retriever, enabling it to prioritize more effective trajectories for unseen tasks. The authors conducted experiments across two environments, demonstrating MART's success rates in unseen scenes.

### Strengths
1. This paper proposes a new retrieval-augmented MLLM agent, which finds the most matched expert reference trajectory to enhance current trajectory planning. To compress these multimodal trajectories with lots of images, actions, and feedback, Trajectory Abstraction (GPT-4o) is introduced to identify important trajectory milestones.

2. The experimental results demonstrate the effectiveness of MART in improving task success rates across different environments.

### Weaknesses
I am not an expert in this field, but I have several problems:

---

1. This work aims to tackle the challenge of grounding in environments. However, the literature review focuses more on embodied agents, memory retrieval in agents, and MLLM. These areas encompass a wide range of topics and, as a result, **overlook numerous specific and significant studies concerning how past embodied grounding approaches have tackled this issue**. In summary, I am unable to identify key references within this manuscript, and the innovation it presents remains obscure to me.

--- 
2. The proposed model is evaluated on AI2-THOR and LEGENT, with performance comparisons against Plain-Agent, LLaVA-Plain, Similarity+LLaVA, and RAP. However, it's worth noting that these comparative methods are predominantly variations of Multimodal Language Models (MLLM) or retrieval-based approaches, effectively serving as ablation studies. **This suggests that the absence of key comparisons with prior grounding techniques raises concerns regarding the efficacy of the proposed method's performance.**


---
3. What are the implementation details including model and training details?

### Questions
You can find all my inquiries detailed in the Weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a novel method, MLLM As ReTriever (MART), which enhances the performance of embodied agents by utilizing interaction data to fine-tune a multimodal language model (MLLM) retriever. The method aims to improve the effectiveness of retrieved trajectories for specific tasks, and it introduces Trajectory Abstraction to summarize trajectories while preserving key information. The experimental results show significant improvements in task success rates in unseen scenes compared to baseline methods.

### Strengths
1. The paper proposes a unique method for enhancing multimodal retrieval in embodied agents by leveraging preference learning and trajectory abstraction. This approach addresses a critical gap in current retrieval methods that often focus on surface-level similarities.
2.  The introduction of Trajectory Abstraction is a valuable contribution. It effectively reduces the complexity of trajectories while maintaining essential information
3. The experimental results are comprehensive and demonstrate the effectiveness of the proposed method across various environments. The improvements in task success rates in unseen scenes are particularly noteworthy.

### Weaknesses
1. While the paper demonstrates the effectiveness of MART in several environments, the scope of experiments could be expanded to include more diverse and challenging scenarios to further validate the robustness of the method.
2. The paper does not provide a detailed analysis of the computational complexity and resource requirements of the proposed method. This information is crucial for practical implementation and comparison with existing methods.
3. The paper mentions the use of interaction data but does not delve into the specifics of how user interactions are collected and processed. More details on this aspect would enhance the clarity and reproducibility of the method.
4. Provide more detailed implementation guidelines and discuss potential limitations and solutions. This would make the method more accessible to other researchers and practitioners.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3
