# R4: Nested Reasoning-Retrieval for Reward Modeling in Role-Playing Agents

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Role-playing dialogue presents unique challenges for large language models (LLMs): beyond producing coherent text, models must sustain character persona, integrate contextual knowledge, and convey emotional nuance. Despite strong reasoning abilities, current LLMs often generate dialogue that is literal, stylistically bland, and misaligned with character-specific traits. Existing approaches such as retrieval-augmented generation (RAG) or reinforcement learning (RL) with scalar rewards are insufficient, as they cannot capture nuanced preferences or adapt reliably to diverse character contexts.
In this work, we introduce R4, a unified framework that equips both the reward model and the role-playing agent with reasoning and retrieval capabilities. Our reward model reformulates evaluation as structured reasoning: it integrates multi-step deliberation and retrieved knowledge to assess responses along multiple dimensions. This reward supervision is then used within reinforcement learning to train a dialogue agent with the same dual capabilities, enabling contextually grounded and persona-consistent generation.
Experiments demonstrate that R4 substantially improves dialogue quality, particularly in persona fidelity, narrative coherence, and emotional expressiveness. Analysis of training dynamics and case studies further shows that R4 agents employ retrieval more effectively, engage in retrieval-informed self-reflection, and achieve emergent role-playing behaviors unattainable by prior methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper considers training methods to improve role-playing agents. For role-playing agents, the authors show that it is insufficient to just return factually correct responses, as users also look for dimensions such as character coherence, factual consistency, and emotional engagement. To this end, the authors proposes R4, a method that first trains a generative reward model that performs both reasoning and retrieval from a character database (also constructed by the authors) using GRPO. Then, R4 uses the trained reward model to train the role-playing agent which also performs both reasoning and retrieval using GRPO. Results on the CharacterEval benchmark shows that the proposed method outperforms both current strongest reasoning models as well as some existing role-playing finetuned models such as CharacterGLM.

### Strengths
- I believe the proposed method for constructing a character-specific knowledge database is sound. The database contains both diverse information (e.g., personal traits, emotional states, contextual knowledge, and narrative goals) and good quality checks (using both LLM and expert reviews).

- R4-trained models showed better performance than all existing reasoning models/role-playing models on CharacterEval.

- I believe some of the analysis provided in this work is sound and insightful. For example, findings such as "reward model capabilities establish the foundation, while agent capabilities set the upper bound" (L451-453) could generalize beyond role-playing tasks.

### Weaknesses
I believe the some of the contribution mentioned in the introduction is over-claimed and does not align well with the novelties I perceived in the method section. I detail them below.

1. In L100-102 the paper claims "we reformulate reward modeling as a structured reasoning task.." and  "novel reward model architecture that integrates ...reasoning and retrieval". However, equipping generative reward models with retrieval (or more generally, tools) is not new, and has already been proposed in existing work such as [1-2].

2. in L103-105 the paper claims "we propose end-to-end training that unifies reasoning and retrieval...", but the general method of performing retrieval (e.g., as a tool) during RL is has already been proposed by prior work such as Search-R1 and ReSearch. The main training difference is the reward model used during training (labeled reward for RLVR or a trained reward model), which I believe is too small of a change to be claimed as a novelty.

However, I do believe the paper is novel in its focus on character-specific knowledge base construction and the importance of good reward models for role-playing (e.g., via RL training). I suggest the authors to perhaps adjust the focus in introduction towards section 2.1 and 2.2, instead of section 2.3.


---

References

[1] Zhu, Jiachen, et al. "Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning." arXiv preprint arXiv:2502.14361 (2025).

[2] Gou, Zhibin, et al. "Critic: Large language models can self-correct with tool-interactive critiquing." arXiv preprint arXiv:2305.11738 (2023).

### Questions
- how many training and test runs were performed to obtain results in Table 1? Can the authors also report standard deviations, as the performance gap on some metrics are small.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes R4, a unified framework designed to enhance reward modeling and response generation for role-playing dialogue agents by integrating reasoning and retrieval capabilities.  The authors address the key limitations of existing methods (such as literal, stylistically bland responses, and poor alignment with character personas) by equipping both the reward model and the role-playing agent with structured reasoning and access to character-specific knowledge.

The key contribution of this work includes:

- It proposes a novel reward model that integrates multi-dimensional evaluation and refines reward modeling as structured reasoning to mitigate role/reference biases.

- Unifies reasoning/retrieval across reward modeling and response generation for role-playing.

- Extensive experiments are conducted to show the advantages of the proposed R4.

### Strengths
1. Role-playing dialogue is an interesting topic.
2. This work refines reward modeling for role-playing dialogue as a structured reasoning task,  which can effectively address  role and reference biases in existing scalar/generative reward models.
3. Unifies reasoning and retrieval across both reward models and dialogue agents, creating a mutually reinforcing framework for more authentic role-playing.
4. Enough empirical study to show the shortages of prior reward models.
5. The experiments are detailed and comprehensive. And the corresponding experimental results are good.

### Weaknesses
1. It is not very clear that how R4 performs with non-literary role-playing characters instead of only novel-derived ones.
2. It would be better to compare more specialized baselines.
3. The computational efficiency detail of this work is not very clear.

### Questions
1. Did you calculate inter-annotator agreement for the 2K human data? 

2. How about the performance of R4 on long multi-turn dialogues (5+) ?

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
The paper presents R4, a unified reasoning–retrieval framework for reliable reward modeling in role-playing dialogue. It identifies two key sources of bias—role bias and reference bias—and proposes a reasoning-augmented reward model combined with a retrieval-enhanced dialogue agent. The work is timely and addresses a genuine weakness in current LLM-based dialogue systems.

### Strengths
- The paper tackles an important and underexplored problem — reliable reward modeling for role-playing dialogue — and proposes a reasonable unified reasoning-retrieval framework (R4) as a solution.

- The identification of role bias and reference bias is insightful, and the experimental results demonstrate consistent and interpretable improvements across multiple dimensions.

### Weaknesses
- Methodological clarity is limited. The description of the structured / character-aligned knowledge base synthesis is vague. Specifically, the four agents involved in this process (knowledge extraction, perspective transformation, mind modeling, dialogue extraction) are not clearly defined in terms of their inputs, outputs, and organizational relationships. Furthermore, the final hierarchical tree structure of the character-aligned knowledge base is not explicitly illustrated or formalized.

- Figure 2 lacks essential experimental details. It is unclear how “main” versus “minor” characters were defined and selected for evaluation. The selection criteria may directly affect the observed role bias.

- Quality of narrative segmentation is not analyzed. Since the segmentation quality likely influences downstream character-specific knowledge construction, some form of segmentation-level evaluation (e.g., coherence metrics or human judgment) would strengthen the paper.

- Missing comparison with related reasoning-augmented RL methods. The paper does not discuss or compare against recent works such as Process Reinforcement through Implicit Rewards (Cui et al., 2025), which is already cited. This omission weakens the claim of methodological novelty.

### Questions
- In Equation (1), the reward function includes rfmt1 with coefficient λfmt1, but it is unclear how this differs from λfmt2 used in Equation (4). Are these two separate formatting rewards, or do they share the same objective?

- What is the architecture and training procedure of the consistency verifier used in the consistency reward? How is it supervised and integrated into the optimization pipeline?

- How are retrieval queries generated during training? Are they manually templated, LLM-generated, or learned through gradient feedback?

- How many characters are included in each category (main / minor)? Is the dataset balanced in terms of narrative diversity?

- How sensitive is the overall system to errors or inconsistencies in the extracted knowledge base? Has any ablation been performed to measure this impact?

- What is the inference latency relative to baseline models? How does computational cost scale with the number of retrieval operations?
Is the requirement of 64 × H100 GPUs essential, or can the framework be trained with smaller resource budgets?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This research describes R4 (Reward, Role-play, Reason and Retrieve), a unified framework and system designed and developed to improve the performance of LLMs in role-playing dialogue. The authors identify a key problem: standard LLMs produce dialogue that is often bland and inconsistent with a character's persona. They argue and show that existing methods like simple Retrieval-Augmented Generation (RAG) or Reinforcement Learning (RL) with scalar rewards are insufficient and not optimal for maintaining the fidelity of the role playing dialogue agents.

The main contribution of this research is to equip both the reward model (RM) and the dialogue agent with interleaved strong reasoning and SOTA retrieval capabilities. The framework has three main parts:
- A character-specific knowledge base is constructed by segmenting narrative sources (similar to lumbarchunking pipeline) and extracting persona traits, emotional states, and goals.
- A reasoning-augmented reward model is trained using reinforcement learning (GRPO) to evaluate dialogue responses. Instead of just giving a score, this RM generates a structured reasoning chain, justifying its preference by retrieving and analyzing character knowledge. This process is designed to mitigate observed "role bias" (poor performance on lesser-known characters via use of balanced training datasets) and “(scarce-)reference bias" (by providing relevant context).
- A role-playing agent is then trained using the output of this RM as its reward signal. Because the agent shares the same reasoning-retrieval architecture, it learns to generate responses that are persona-consistent, emotionally expressive, and grounded in the narrative context.

Experiments show that R4 significantly outperforms a wide range of baseline models—including instruction-tuned, reasoning-specialized, and dedicated role-playing models—particularly on metrics of character consistency and role-playing attractiveness.

### Strengths
- Making the reward model and the agent symmetrical—both using the same reasoning and retrieval mechanisms—is compelling. It addresses the problem of misaligned objectives by ensuring the agent is optimized by a reward signal that understands and values the tasks (reasoning and retrieval) the agent must perform.
- The experiments are comprehensive (showing results across many SOTA baselines) leveraging multiple SOTA metrics and methods from the role playing dialog literature (CharacterEval, FlashRAG, LumbarChunker)
- The ablations in Table 3 provide insightful findings. The finding that reward model and supervision signal quality (reward formulation via structured reasoning over semantically retrieved content) is very critical to achieve best performance. Furthermore, demonstrating that generic retrieval is worse than no retrieval for the RM highlights the importance of knowledge specificity. -> this is the most critical result most suitable for this conference -> trying to understand this better will add a lot of value to this work (how well does this finding generalized across other domains?, more insights here)

### Weaknesses
- There’s no evidence that R4 transfers to other domains where reward + agent reasoning & retrieval might matter (e.g., tool-using agents, search-based assistants, task-oriented dialogue, or multimodal agents). Any reinforcement and validation of these findings from other existing literature will be very helpful here to make this work more relevant for ICLR venue (imo, this work seems more relevant for ACL/EMNLP venues or more focused tracks/workshops at ML conferences) 
- The paper makes a claim about using a "multi-faceted hierarchical knowledge organization" to support efficient and multi-hop reasoning. However, the description in Section 2.1 is vague and lacks implementation details. It is unclear how this structure is technically different from a standard vector database with metadata filtering. The paper does not explain the data structure, the clustering method, or provide a concrete example of how this hierarchy enables multi-hop reasoning in a way that a flat document index cannot. This part feels over-claimed and under-explained.
- The paper does not discuss the practical, real-world applications for such a complex/speacialized  system.
- The system is quite complex comprising of advanced (and costly, how costly?) segmentation, extraction, transformation, etc - these complex interaction may result in cascading errors, trying to understand what types of errors are more critical along with their relative impact will be insightful.

### Questions
- Can you provide more details about multi-faceted hierarchical indexing scheme? How is the hierarchical index built (data structures, algorithms, etc) 
- Role-play can easily drift into unsafe behaviors; any studies on this topic would have made this paper more insightful and interesting from ML system’s safety perspective (have the authors thought of using safety aware rewards). 
- Beyond role-play and dialog systems implications, what broader ML insight should readers take from R4? Why is this role-play a good testbed for RLHF research? 
- There is a brief mention of dynamic expansion of the KB during construction - how are knowledge gaps detected and what if there are contradictions between synthesized content and original text?

### Soundness
3

### Presentation
3

### Contribution
3
