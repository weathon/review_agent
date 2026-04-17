# Dancing in Chains: Strategic Persuasion in Academic Rebuttal via Theory of Mind

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 4

## Abstract
Although artificial intelligence (AI) has become deeply integrated into various stages of the research workflow and achieved remarkable advancements, academic rebuttal remains a significant and underexplored challenge. This is because rebuttal is a complex process of strategic communication under severe information asymmetry rather than a simple technical debate. Consequently, current approaches struggle as they largely imitate surface-level linguistics, missing the essential element of perspective-taking required for effective persuasion. In this paper, we introduce RebuttalAgent, the first framework to ground academic rebuttal in Theory of Mind (ToM), operationalized through a ToM-Strategy-Response (TSR) framework that models reviewer mental state, formulates persuasion strategy, and generates evidence-based response. To train our agent, we construct RebuttalBench, a large-scale dataset synthesized via a novel critique-and-refine approach. Our training process consists of two stages, beginning with a supervised fine-tuning phase to equip the agent with ToM-based analysis and strategic planning capabilities, followed by a reinforcement learning phase leveraging the self-reward mechanism for scalable self-improvement. For reliable and efficient automated evaluation, we further develop Rebuttal-RM, a specialized evaluator trained on over 100K samples of multi-source rebuttal data, which achieves scoring consistency with human preferences surpassing powerful judge GPT-4.1. Extensive experiments show RebuttalAgent significantly outperforms the base model by an average of 18.3% on automated metrics, while also outperforming advanced proprietary models across both automated and human evaluations.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work introduces an LLM-based agentic scaffold for building rebuttal agents. The scaffold is inspired by Theory of Mind (ToM) from cognitive science and tries to construct reviewers' mental model (e.g., intent and stance) before generating response strategies and contents. The authors also propose a data curation pipeline for converting an existing rebuttal dataset ($\mathrm{Re}^2$) into a synthetic dataset for training the rebuttal agent. Finally, they also trained a reward model (i.e., Rebuttal-RM) for scoring rebuttals.

### Strengths
1. The motivation that we need to consider reviewers' mental model and stance during rebuttal is intuitive and reasonable, and I do think this direction is worth exploring in general.

2. Both automatic evaluation (reward model based) and human evaluation show that the proposed RebuttalAgent achieve better empirical performance on average.

3. Experiments and ablations are extensive. The proposed method is compared against a variety of baselines, including different LLMs, different training data, and training methods.

4. It is an interesting and useful finding that the LLM's self-generated reward can be used to improve its rebuttal quality, as it reduces the reliance on an explicitly trained reward model.

### Weaknesses
1. The generalizability of the proposed method is dependent on the alignment between the dataset and the current rebuttal practice. For example, currently many conferences have a "discussion period", during which the authors will engage in back-and-forth discussion with the reviewers, and the final reward score is highly dependent on this process. In this regard, the applicability of this work is somewhat limited.

2. The current method focuses on the "writing" component of rebuttal. However, rebuttal sometimes requires the authors to come up with supporting experiment results that clarify reviewers' questions. Therefore, except for "writing" itself, another crucial aspect is to think about what additional experiments are most useful for addressing reviewers' concerns.

### Questions
1. There are duplicated contents from line 316 to line 323.

2. If the review comment requires additional experiment results, how would RebuttalAgent handle this case?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces RebuttalAgent, a framework designed to enhance academic rebuttal writing by incorporating Theory of Mind (ToM) reasoning.
The model follows a three-stage TSR pipeline that first infers the reviewer's mental state, then formulates a persuasion strategy, and finally generates a context-aware response.
To train the agent, the authors construct RebuttalBench, a synthetic dataset of over 70K samples generated via a multi-teacher critique-and-refine process.
Training involves supervised fine-tuning followed by reinforcement learning with a self-reward mechanism, enabling scalable self-improvement without external reward models.
For evaluation, they develop Rebuttal-RM, an automatic evaluator trained on over 100K examples, showing higher agreement with human judgments than GPT-4.1.
Experiments demonstrate that RebuttalAgent outperforms its base model and achieves performance comparable to advanced models like o3 in both automated and human evaluations.

### Strengths
- The paper introduces an innovative perspective by framing academic rebuttal as a ToM reasoning task.

- The integration of supervised fine-tuning and reinforcement learning creates a self-improving training pipeline.

- The model demonstrates consistent improvements over its base model across multiple evaluation metrics.

### Weaknesses
- RebuttalBench contains over 70K samples generated via a multi-teacher critique-and-refine pipeline. Although some reviewer comments may originate from real OpenReview data, the dataset remains largely synthetic. This raises concerns that the model may internalize the linguistic and strategic biases of the teacher models rather than authentic author–reviewer interactions. The paper does not report any validation showing distributional or pragmatic similarity between the synthetic data and real human-written rebuttals. As a result, the claimed benefit of ToM reasoning is only demonstrated in a simulated environment, without confirmed real-world effectiveness.


- It is unclear whether GPT-4.1, o3, DeepSeek, and other baselines were evaluated under identical prompting and retrieval settings. It seems that only the Strategy-Prompt (GPT-4.1) baseline mimics the proposed strategic planning stage; other foundation models were not prompted to perform ToM inference. Thus, the reported gains may reflect differences in pipeline design or fine-tuning rather than inherent reasoning ability.


- All model outputs are required to include the explicit <analysis><strategy><response> structure, and the reward function also checks format adherence. This raises the question of whether the model mainly learns to reproduce the expected format rather than to perform genuine strategic reasoning. 

- Since the reinforcement learning stage uses a self-reward mechanism derived from the same model, it is possible that the agent learns to optimize for its own scoring heuristics instead of true persuasiveness.

### Questions
- The work does not show that generated rebuttals can actually influence reviewer judgments or acceptance outcomes. The "persuasiveness score" used by Rebuttal-RM serves only as a proxy, without evidence that it correlates with real reviewer behavior. While a full-scale deployment would be difficult, would it be possible to conduct a small-scale testing to verify whether the generated rebuttals are genuinely persuasive to expert reviewers

- The proposed approach is trained on Qwen3-8B and Llama-3.1-8B.
Can the TSR pipeline and self-reward mechanism transfer effectively to other backbones, such as smaller models (<8B)?

### Soundness
2

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
This work focuses on the academic rebuttal task — how authors can comprehensively address reviewers’ comments, resolve their concerns, and persuade them through well-reasoned responses. This is an interesting and meaningful direction, representing an important step for LLMs toward academic reasoning. The paper introduces a ToM-based reasoning framework, a rebuttal benchmark for instruction tuning, a RebuttalJudge model for response evaluation, and further integrates reinforcement learning (RL) to optimize the RebuttalAgent. However, from a technical perspective, the paper does not present novel or compelling innovations; rather, it seems to combine existing techniques to tackle the defined task.

### Strengths
1. The task itself is very interesting, the motivation is clear, and the paper is well written.

2. The study provides several valuable resources, including the RebuttalBench benchmark and the RebuttalJudge evaluation model.

3. The paper offers a complete framework for developing a RebuttalAgent and presents comprehensive experiments that demonstrate its effectiveness.

### Weaknesses
1. The main concern is the lack of technical innovation.  I am curious whether using RebuttalJudge as a reward function to guide RL training would lead to improved performance under human evaluation. This would be an insightful experiment, showing that both the benchmark and the judge model can effectively guide the optimization of the RebuttalAgent.

2. Another concern lies in the baseline comparisons, which appear somewhat weak. While I am not deeply familiar with prior work specifically on academic rebuttal, the baselines used here seem to rely on relatively simple or earlier reasoning frameworks. Moreover, since this work involves a significant training cost, it would be helpful to include stronger baselines such as o3 or GPT-4 evaluated under the proposed ToM framework — these would serve as powerful non-trained baselines for comparison.

3. The ablation studies could be more detailed and targeted. They should aim to answer key questions — for example, how well does a model perform when trained solely on RebuttalBench? It would also be beneficial to conduct ablations separately under the SFT and RL settings rather than only presenting overall results.

### Questions
Refer to our proposed weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper is on using Theory of Mind (ToM) to help with effective persuasion in the academic rebuttal scenario. Such goal is to be achieved through three stages. First, motivated by ToM, the paper proposes a 'RebuttalAgent' framework using a process of reviewer perspective inference, strategy and persuasive responses (TSR). Then the 'RebuttalBench' is formed as the dataset for training, with which both  supervised fine-tuning and reinforcement learning are utilized. Finally, a 'Rebuttal-RM' is also trained as the model to measure the responses generated by the above agent. The advantage of such agent is shown, compared to baseline models, across several dimensions of the responses.

### Strengths
> **Originality**
- The framework is motivated by ToM.

> **Significance**
- The improvement, compared to the list of selected baseline models, is significant.

### Weaknesses
> **Originality**
- A discussion on the difference in methodology, compared to those mentioned as Agent-based Methods, could be helpful.

> **Quality**
- Section 3.1 Comment Extraction: As the authors suggest, the raw review comments can be unstructured with varying noises, styles or formats, then the proposed LLM-as-Extractor may need some systematic verification on its performance.
- Section 3.1 Comment Extraction (continued): According to the prompt in Figure 6, there could be several concerns. First, the justification of the categories listed in Macro/Micro-Analysis. Second, why never split a comment item? What if multiple directions are covered? Overall the concern is the fixed framework of dimensions and criteria that may not always guarantee a satisfying comprehension.
- Such comprehension issue can also exist in Section 4.1 Reviewer Profile Modeling.
- According to the reviewer's understanding (feel free to correct), the response is generated comment-wise. Then how to guarantee the consistency and avoid redundancy when integrating all responses together?
- Whether it is justified to use SFT-tuned model in its own evaluation/training in Section 5.3, especially how to make sure the evaluation aligns with the expected direction of improvement?


> **Clarity**
- A more detailed review of the Theory-of-Mind is necessary before introducing the proposed methods.
- It is a bit unclear where the original response, $r_\textrm{orig}$, is from. If it is given as part of the dataset, then in the final implementation of trained model, will this part be excluded?
- Line 316-323: The paragraph contains two copies of the same sentences.
- Section 7.1: The notations of metrics used should be explained in the main paper.

### Questions
- Could the authors elaborate how ToM is explicitly used? According to the reviewer's understanding, such details are rather limited in the main paper.
- When evaluating the responses in the main experiment, is the score an average of comment-wise responses or considers the quality of the concatenated responses against each complete review?

### Soundness
2

### Presentation
2

### Contribution
3
