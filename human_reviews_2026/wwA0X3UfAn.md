# Attend to the Active: Structure-Aware Dynamic Attention in LLMs for Compositional Instruction Following

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 8

## Abstract
Large language models (LLMs) have exhibited strong instruction-following capabilities; however, they often struggle with compositional instructions involving multiple interleaved yet logically independent sub-tasks. These sub-tasks are typically organized in mutually exclusive structures, such as branching, chaining, or paralleling, where only one sub-task should be active at each generation step, while the others remain dormant.  Despite their inactivity, dormant sub-tasks can inadvertently attract the model's attention due to structural entanglement within the input context or intermediate representations, leading to interference that compromises output fidelity. To address this challenge, we propose ATA, a structure-aware dynamic attention mechanism grounded in compositional structures, which dynamically identifies the active sub-task during generation while suppressing attention to inactive ones. By precisely steering the model’s focus, ATA mitigates interference and explicitly enhances model adherence to the active sub-task.  Importantly, ATA operates within a single forward pass without requiring parameter updates. Extensive experiments show that ATA consistently enhances LLMs' instruction-following ability across various compositional structures, effectively mitigating attention distraction and demonstrating a strong generalization ability.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This research studies the complex instruction-following problem. The authors consider that there are three structures of sub-tasks from the user instructions, and propose to manipulate the attention-patterns for different task structures such that the model can attend to the correct contents for each sub-task. The results of the control experiments support the effectiveness of the proposed method.

### Strengths
1. The overall idea of manipulating the attention mask to enforce the model to attend to certain sub-tasks for complex instruction following is reasonable.
2. The comprehensive ablation study provides evidence for the effectiveness of each proposed component. 
3. For most of the manuscript, the writing is clear.

### Weaknesses
1. The authors decompose sub-tasks by leveraging an LLM, which can be a significant bottleneck to the framework when it tries to handle more challenging tasks. Although the performance of structure identification (Appendix B.1) looks good on current datasets, I am not sure whether this reliability can extend to more challenging tasks/datasets, such as AIME 24/25 for mathematical solving. The reason I raise this concern is that, for many hard questions, the sub-tasks are not explicitly presented in the user instructions, so the model needs to plan them by itself at first. 

2. Figure 3 has not been rendered properly. 

3. How is the model's performance on general (simple) instructions, such as email drafting and poem writing? In other words, whether the system is robust to the instructions that do not hold the hypothesis that they have multiple sub-tasks? 

4. How much is the overall overhead for model inference, including the time of annotating sub-task structures?

### Questions
See Weakness.

### Soundness
2

### Presentation
2

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
This paper proposes an attention steering technique for improving instruction following by LLMs for compositional instructions. Three compositions are considered: chaining (sequential), branching (conditional), and parallel execution of sub-tasks. The key claim is that at any generation step, only one sub-task should be active irrespective of the type of composition. The steering technique, Attend To the Active (ATA), first requires the composite instruction to be distilled to the composition type and list of sub-tasks. It then first constructs a mutual attention masking  matrix that downweighs attention between independent tasks, and a dynamic attention masking matrix that downweighs attention to tasks independent of the current "active task" (the active task is identified as the one receiving the most attention at the current step of generation). These masks are applied at every step of generation to a selected subset of attention heads. The experiments show that ATA outperforms other instruction following techniques and attention steering methods on instructions from the three composition types. An analysis of components of ATA demonstrates that it is robust to how the structure information is presented and the masking degree, but the structure information along with the masking modules and the control strategy (identification of the active task) are essential -- the selection of attention heads is crucial to the success of ATA.

### Strengths
- The paper is well written and the key ideas are fairly easy to understand.

- The problem of better instruction-following with compositional instructions for LLMs is important and the proposed approach, ATA, of masking attention on sub-tasks based on structural exclusiveness and the current active task seems novel.

- The experiments include a good variety of baselines (other instruction following prompting strategies and attention steering methods) and provide insights into how various components of ATA are useful.

Overall, I think that the paper studies an important problem and the experiments demonstrate effectiveness over other techniques.

### Weaknesses
- ATA essentially ensures that only one sub-task is attended to at any point but does not focus on whether all sub-tasks are eventually attended to (for chaining / parallel compositions). Improved instruction following would require that the order of the tasks is also taken into account (for chaining), and that all sub-tasks are "accomplished" (for parallel). At the least, I would expect this to be explicity mentioned in the text and some analysis from the experiments on whether this happens with ATA.
- For branch compositions, I would also expect the condition to be a sub-task that is attended to first. But the prompt template in Section 4.1 does not seem to include the condition during structure identification. Moreover, as I mention in the previous point, ATA does not take into account that only one of the sub-tasks should be accomplished (at least in the way it is defined and discussed in Section 4.2).
- I find it difficult to understand which tokens are being included in the identification of the current active sub-task. For Equation 5, it seems like it computes the average attention between the tokens of a task instruction T_i (k) and all tokens after the instruction (q >= k)? Would this not include other tokens of T_i, othe following sub-tasks, and the entire generation sequence so far? The preceeding description of the score seems to convey that the score is computed with respect to the next token at a given step (lines 283-286).
- The identification of the active task also mentions the entropy threshold. How this threshold is computed and how the value affects the performance is not discussed in the main text / experiments. Moreover, I do not understand what happens is the entropy is over the threshold.

Overall, I think this work would benefit from providing more details on the technique (the active sub-task score, the entropy threshold, the inclusion of the condition for branching tasks) and the contribution can be improved by taking the overall structure into account (sequential tasks should happen in order, parallel tasks should all eventually happen, etc.).

### Questions
I will summarize my questions from the weaknesses section above (please refer to that section for more details):
1. Which tokens are included as "next tokens" in the score computation for the active task?
2. How is the entropy threshold computed and how sensitive is ATA to this choice?
3. How is the condition for branching tasks factored in during structure identification?
4. Does ATA promote global structure following by restricting attention to the current active task -- do sequential / parallel sub-tasks all eventually happen?

Additionally:

5. How does the structure identification work for the nesting experiments (Table 4)? What does the prompt look like here? 
6. In the ablation on structure identification (Figure 4(b)), what do the variants partial, original and human-revised mean?
7. For active sub-task identification, what happens when the entropy is over the threshold?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper improves the instruction following behavior of LLMs by identifying three different compositional structures of instructions which LLMs struggle to follow. They propose an attention modification method which ensures that the LLM only attends to the relevant sub-instruction at any time during generation, and they show that this improves instruction following.

### Strengths
* The paper is very well written and easy to follow.
* The approach is novel and addressed an important problem with a relatively lightweight method.
* A large number of baselines and models are used in the evaluation which shows that the proposed approach is most effective.
* The attention head selection strategy proposed in Appendix A.3 is shown to be effective while using substantially less compute than baselines which makes these methods easier to use.
* Results on nested composition also show that the method remains effective.

### Weaknesses
* Missing standard deviations for results.
* The datasets used for evaluation are designed to have the three types of composition, but a traditional instruction following dataset with more “real” tasks is not used for evaluation.

### Questions
* Figure 4a shows that ATA reduces the identified generation errors, but it doesn’t go to 0. Why do you think some of these errors remain? Are the wrong subtasks selected by the model’s attention?
* How often is active subtask selection incorrect?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Attend to the Active (ATA), an inference-time method that steers a model’s attention toward the currently “active” sub-task within a complex instruction. ATA masks attention between mutually exclusive sub-tasks and then boosts attention to the detected active sub-task. The authors evaluate on chain, branch, and a synthetic parallel setup and report consistent gains over prompting and planning baselines, with ablations on masking strength and head selection. They also include robustness checks for structure identification quality and some results on nested compositions.

### Strengths
1. Good idea and implementation. The focus on compositional instruction following is timely. Doing it with attention masking at inference is interesting because it avoids retraining and aims to localize interference between sub-tasks.  

2. Clear writing and organization. The paper explains the three structure types, gives concrete examples, and walks through the masking and steering pipeline in a way that is easy to follow.   

3. Strong experiments and analysis. The experiments cover two base models, three composition types, and include ablations for masking degree and number of steered heads, as well as component removals. The figures show sensitivity to α and head counts, and tables report the impact of removing structure info, mutual masks, dynamic masks, and active control.

### Weaknesses
1. Parallel instruction data is simplistic - The “parallel” benchmark is built by concatenating independent GSM8K problems. This construction makes sub-tasks independent by design, which favors attention masking because there is little need for cross-task sharing or entity linking. A harder parallel set, closer to the realistic prompts illustrated in figure 2, would be better test.  

2. General capability preservation is under-tested - Because ATA modifies attention patterns, it is important to show that language abilities are not degraded. Following evaluations in [1] it would be good to measure side effects on general text quality.  


References: 
[1] Stolfo et al., “Improving Instruction-Following in Language Models through Activation Steering,” ICLR 2025.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
