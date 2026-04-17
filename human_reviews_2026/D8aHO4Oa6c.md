# S2J: Bridging the Gap Between Solving and Judging Ability in Generative Reward Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 4, 2

## Abstract
With the rapid development of large language models (LLMs), generative reward models (GRMs) have been widely adopted for reward modeling and evaluation. Previous studies have primarily focused on training specialized GRMs by optimizing them on preference datasets with the judgment correctness as supervision. While it's widely accepted that GRMs with stronger problem-solving capabilities typically exhibit superior judgment abilities, we first identify a significant solve-to-judge gap when examining individual queries. Specifically, the solve-to-judge gap refers to the phenomenon where GRMs struggle to make correct judgments on some queries (14\%-37\%), despite being fully capable of solving them. In this paper, we propose the Solve-to-Judge (S2J) approach to address this problem. Specifically, S2J simultaneously leverages both the solving and judging capabilities on a single GRM’s output for supervision, explicitly linking the GRM’s problem‑solving and evaluation abilities during model optimization, thereby narrowing the gap. Our comprehensive experiments demonstrate that S2J effectively reduces the solve-to-judge gap by 16.2\%, thereby enhancing the model's judgment performance by 5.8\%. Notably, S2J achieves state-of-the-art (SOTA) performance among GRMs built on the same base model while utilizing a significantly smaller training dataset. Moreover, S2J accomplishes this through self-evolution without relying on more powerful external models for distillation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the challenge of improving generative reward models by targeting the gap between judging and solving. The authors empirically observe a gap between a model's ability to solve and judge. The authors propose integrating a traditional judging-based reward with a solving-based reward, thereby aligning evaluation performance with problem-solving ability. The goal is to reduce the solve-to-judge discrepancy while enhancing evaluation through the model’s solving capabilities. Extensive experiments demonstrate that the approach improves judging accuracy, significantly narrows the solve-to-judge gap, and achieves state-of-the-art results with fewer training samples.

### Strengths
1. **Clear presentation**: The paper is well-written, with a clearly stated problem and a simple yet effective proposed method. Diagrams and figures are used effectively to support understanding.  
2. **Strong empirical motivation**: The authors convincingly demonstrate the existence of the solve-to-judge gap, providing a solid foundation for the necessity of their approach.  
3. **Builds on prior work**: The method extends existing approaches with novel observations and solutions that advance the field.  
4. **Well-designed experiments**: The experimental results cover multiple benchmarks and highlight the effectiveness of S2J.

### Weaknesses
1. **Limited model diversity**: Experiments are conducted only with Qwen models as the base. Including additional model families would strengthen claims of generalizability.  
2. **Lack of theoretical discussion**: While the solve-to-judge gap is clearly shown empirically, the paper provides little discussion on why it arises. Is it inherent to LLMs, a consequence of training objectives, or another factor? A deeper analysis would add valuable insight.  
3. The main algorithmic contribution is incremental and requires an external expert model for training

### Questions
1. What exactly is meant by self-evolving in this context?  
2. The paper claims the method does not require powerful external models for distillation, yet in the subjective setting, an external model is used to provide scores. Does this imply that a judge is still required to train the new S2J judge? If so, how can both statements be true a the same time?
3. Is there any discussion or hypothesis on the root cause of the solve-to-judge gap? Could it be addressed earlier during pretraining or supervised fine-tuning?  


**Questions that did not impact the rating**
- For subjective tasks, your method relies on an auxiliary scalar model. Could the framework be extended into a two-step or iterative scheme, where earlier versions of the learned model serve as the auxiliary model?  
- In Table 1, the best scores are not highlighted in bold. Is there a reason for this choice?  
- In Figure 1, when ground truth is provided, error is ~5%. When models solve the task, the error rises to 15–35%. Does the ground truth include explanations? If not, why is the error higher when the model correctly solves the task?  
- Regarding the S2J prompt (Figure 3), how was this prompt designed? Was there any ablation to measure how prompt choice affects S2J effectiveness?

### Soundness
4

### Presentation
4

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
Generative reward models (GRMs) are widely used with large language models (LLMs) for evaluation, but they often show a solve-to-judge gap—failing to judge 14–37% of queries they can solve. The proposed Solve-to-Judge (S2J) method links a GRM’s solving and judging abilities during training, reducing this gap by 16.2% and improving judgment accuracy by 5.8%. S2J achieves state-of-the-art results with a smaller dataset and without relying on external model distillation.

### Strengths
1. The authors reveal the solve-to-judge gap through extensive experiments, showing that current GRMs incorrectly evaluate 14–37% of problems they can solve, thereby establishing a comprehensive understanding of this limitation.

2. The authors introduce Solve-to-Judge (S2J), a method that jointly optimizes solving and judging capabilities, effectively narrowing the solve-to-judge gap and enhancing the judgment performance of GRMs.

3. The authors demonstrate that S2J reduces the solve-to-judge gap by 16.2% and improves judgment performance by 5.8% across multiple benchmarks.

### Weaknesses
1. I am concerned about the existence and validity of the “solve-to-judge gap,” as the experimental design does not sufficiently rule out other possible factors that might explain the observed phenomenon.
2. There is a lack of clear explanation of the link between LLMs’ problem-solving abilities and their judgment capabilities, and a stronger theoretical or empirical justification for this claim is needed.
3. The authors’ claim is unclear, and the experimental design requires a more detailed and transparent explanation.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper identifies a "solve-to-judge gap" in generative reward models (GRMs), where models fail to correctly judge queries they can solve, and proposes Solve-to-Judge (S2J), a RLVR method that jointly optimizes solving and judging on preference data. Using Qwen2.5-7B-Instruct, S2J reduces the gap by 16.2% on average.

### Strengths
Clearly defines and quantifies the solve-to-judge gap problem, providing empirical evidence across multiple datasets.

### Weaknesses
1. The training approach for subjective tasks could benefit from further exploration; at present, it resembles distillation from one RM to another, though I recognize that the core objective here is indeed to train an RM from scratch.

2. It would be valuable to include experiments assessing generalization to other model architectures and larger models (e.g., beyond 14B parameters). For a methods-focused paper, evaluating a single model may limit the demonstration of broader applicability. 

3. The reward design appears somewhat heuristic; incorporating sensitivity analyses (e.g., examining the method's robustness to variations in reward formulation) could strengthen this aspect.

4. A comparison with other reward models, such as scalar reward models (e.g., Bradley-Terry reward models) [1], would be a helpful addition.

5. The work could be further enriched with additional analytical experiments, for instance:

   - Evaluating the probability that S2J correctly judges responses when the solving step is incorrect;

   - Investigating whether S2J's gains stem from longer chains of thought rather than the mechanism itself, perhaps by conducting experiments with constraints on generation token limits and their impact on judgment accuracy.



[1] Rethinking Bradley-Terry Models in Preference-Based Reward Modeling: Foundations, Theory, and Alternatives

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2
