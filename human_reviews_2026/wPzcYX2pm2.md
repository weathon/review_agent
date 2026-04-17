# FronTalk: Benchmarking Front-End Development as Conversational Code Generation with Multi-Modal Feedback

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
We present **FronTalk**, a benchmark for front-end code generation that pioneers the study of a unique interaction dynamic: **conversational code generation with multi-modal feedback**. In front-end development, visual artifacts such as sketches, mockups and annotated screenshots are essential for conveying design intent, yet their role in multi-turn code generation remains largely unexplored. To address this gap, we focus on the front-end development task and curate FronTalk, a collection of 100 multi-turn dialogues derived from real-world websites across diverse domains such as news, finance, and art. Each turn features both a textual instruction and an equivalent visual instruction, each representing the same user intent. To comprehensively evaluate model performance, we propose a novel *agent-based evaluation framework* leveraging a web agent to simulate users and explore the website, and thus measuring both implementation correctness and user experience. Evaluation of 14 models reveals two key challenges underexplored in the literature: (1) a significant *forgetting issue* where models overwrite previously implemented features, resulting in task failures, and (2) a persistent challenge in *interpreting visual feedback*, especially for open-source vision-language models (VLMs). We propose a strong baseline to tackle the forgetting issue with ACECoder, a method that critiques the implementation of every past instruction using an autonomous web agent. This approach significantly reduces forgetting to **nearly zero** and improves the performance by up to **9.3\%** (56.0\%$\rightarrow$65.3\%). Overall, we aim to provide a solid foundation for future research in front-end development and the general interaction dynamics of multi-turn, multi-modal code generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a benchmark that evaluates multi-turn and multi-modal code generation for front-end code generation. As  multi-modal feedback (textual, visual, and annotated screenshots) is critical for this target domain, the proposed benchmark is unique and useful.

### Strengths
* Multi-Modality: Assess textual and visual feedback, crucial for front-end development. 
* Interesting analysis: Reporting analysis on implementation accuracy and user experience
* Baseline model: Discussing benchmarks with baseline model helps appreciating the valuae of benchmarks. e.g.,  "Forgetting Issue" in multi-turn generation.

### Weaknesses
*Time/infra cost: Sharing time/infra cost for benchmarks will be useful
*Public access: Any public link to reach benchmarks and report model performance would be useful

### Questions
Please answer questions in Weakness

### Soundness
3

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
3

### Summary
This paper introduces FronTalk, a benchmark for conversational code generation with multi-modal feedback, targeting front-end development (i.e., website generation). 
The dataset is derived from the C4 corpus, containing: 100 multi-turn dialogues with 1K turns and 3.6K manually refined test cases. 
An agent-based evaluation framework is proposed, where a web agent simulates users to explore generated websites and measures the functional correctness and the user experience (by pairwise way). 
The authors evaluate 14 models and identify two key challenges: (i) a forgetting issue and (ii) a difficulty in interpreting visual feedback. 
As a baseline, the authors propose AceCoder, an autonomous web agent that critiques the implementation of every past instruction, reduces the forgetting rate to nearly zero and improves performance by up to 9.3\% in pass rate.

### Strengths
- The paper presents a new benchmark focusing on multi-turn, multi-modal front-end development, accompanied by broad model evaluations and several meaningful analyses.

- The proposed AceCoder baseline effectively mitigates the forgetting issue.

### Weaknesses
- The main distinction from WebGen-Bench lies in the multi-turn setup. However, recent works such as WebGen-Agent [1] have already extended single-turn benchmarks into multi-turn scenarios, which suggests that this contribution may not be as technically challenging as implied.

- In Section 3.2, while Cohen's kappa is informative for evaluator reliability, it would be more meaningful to analyze intra-model (across different configurations) and inter-model (across models under the same configuration) ranking consistency. This aspect is missing but critical for validating benchmark stability.

- The AceCoder mechanism seems conceptually similar to running a single-turn setting in the main table--a concatenation of all user intents, effectively simplifying multi-turn reasoning into a static aggregation problem. 
    - Notably, only 5 of the 14 evaluated settings (e.g., GPT-4o, Qwen2.5-VL-72B, Qwen2.5-VL-7B) show multi-turn (T) outperforming single-turn in pass rate.
    - For models where single-turn performance exceeds multi-turn (e.g., Claude-4-Sonnet, Llama-3.3-70B, GLM-4.1V-Thinking-9B, Ovis2.5-9B), it remains unclear whether AceCoder still provides statistically meaningful improvements.
    - If not, reverting to a single-turn setup might actually be the optimal strategy—rendering AceCoder unnecessary.

[1] Lu, Zimu, et al. "WebGen-Agent: Enhancing Interactive Website Generation with Multi-Level Feedback and Step-Level Reinforcement Learning." arXiv preprint arXiv:2509.22644 (2025).

### Questions
- The description of the single-turn baseline (concatenating all user feedback) only appears around line 154, which is quite late. This should be introduced earlier, ideally in Section 2.1. Additionally, including a variant that uses only the initial user intent (without concatenated feedback) would allow for an ablation study on the utility of simulated user feedback.

- In lines 459–464, the distinction between FronTalk and Sketch2Code should be elaborated more concretely. For example, show comparative examples illustrating how FronTalk involves more complex, realistic, and context-dependent interactions.

- Since both PR and UX automatic evaluations only achieve around 80% agreement with human decisions, consider reporting confidence intervals or error bars to indicate statistical significance when claiming improvements.

- Claude-4-Sonnet performs remarkably well in single-turn but poorly in multi-turn settings. Could this discrepancy relate to max token length (as suggested in Table 6, where Claude uses 30K tokens)? Re-evaluating it under shorter context windows (e.g., 10K–20K) could clarify this effect.

- Since 9 out of 14 models achieve higher pass rates in the single-turn setting, the results may reflect degradation from long iterative contexts. It would be valuable to discuss this phenomenon—similar findings were reported in recent studies [2]—as it might indicate a broader limitation of long-horizon LLM reasoning.

- (Not a requirement) I wonder if there are plans to release the training set (e.g., gym-like setup) to facilitate reproducibility and further research.


[2] https://alexzhang13.github.io/blog/2025/rlm/

### Soundness
2

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
3

### Summary
This paper presents FronTalk, a conversational benchmark for frontend coding. At each turn, an LLM-based user simulator converts static intents into context-aware instructions that prompt the agent to edit the codebase. A web agent evaluates the developed website on two axes: instruction following, measured by test cases, and usability, measured by simulating a first-time user. Experiments on the new dataset reveal a common issue of forgetting: repeated edits to the same code region across turns often introduce regressions. To address this, the authors use an external web agent as a critic model that verifies compliance with instructions on the rendered site and produces textual critiques of failures. The approach substantially reduces forgetting.

### Strengths
1. The proposed task of conversational frontend coding is both novel and realistic. It is potentially important for real-world human-machine collaborative web development.

2. It is an interesting and valuable finding that existing LLM agents often break functionality introduced in earlier turns.

3. The proposed method, ACECODER, is simple yet effective.

### Weaknesses
1. The evaluation does not penalize the agent for adding unrequested functions or layout. Such additions may reduce usability, but the score does not reliably reflect that.

2. The evaluation relies mainly on an automatic agent, which is itself a complex problem, so the approach needs rigorous justification that is missing. The authors report a human study with 82.0 accuracy and a Cohen’s kappa of 62.7. It is unclear whether these are sufficient to show that the proposed metrics are a reliable proxy. The paper also omits key details of the human study, including the number of annotators, how they were recruited, any qualification screening, and the instructions they received. Similar issues exist for the LLM-based user simulator.

3. The method introduces an additional critic model that increases inference latency, yet the evaluation does not account for this cost.

4. In real-world web development, users often provide ambiguous requests and iterate. For example, they may ask to add a button and later decide to remove it. FronTalk does not consider these scenarios.

### Questions
Questions:

See weaknesses.

Others:

* Fig. 4: Please adjust the margin between the labels on the x-axis.

### Soundness
2

### Presentation
3

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
This paper presents FronTalk, a new benchmark designed to evaluate code generation models in the domain of front-end web development across multiple conversational turns, incorporating both text and image feedback, and simulating realistic user interactions during website building.
The benchmark comprises a dataset of 100 dialogues (approximately 1,000 turns and 3.6k test cases) derived from real websites, paired with a multi-modal user simulator that provides both textual and visual cues. The paper introduces an agent-based evaluation framework to assess both code correctness and usability.
To address the "forgetting issue" where previously implemented features are overwritten during subsequent turns, they propose AceCoder, a baseline method employing agent-based critique as a mitigation strategy.

### Strengths
- This work addresses incorporating multi-modal (text & image) feedback into multi-turn code generation, specifically targeting front-end development, which is underexplored in current benchmarks.
- The dataset contains 1,000 conversational turns across 100 dialogues and 3,676 manually refined test cases for robust model evaluation. The data is grounded in real-world websites from diverse domains, thereby increasing the benchmark's practicality and relevance.
- Evaluations are comprehensive and cover a wide range of 14 different models.

### Weaknesses
- The dataset is generated using an LLM-based user simulator that generates context-aware instructions conditioned on prior dialogue. For evaluation, first-time users simulated by LLMs interact with each website, and a secondary LLM then compares the resulting trajectories and judges which interface is more usable. In this work, both dataset generation and evaluation rely heavily on LLMs, which can cause multiple problems: the evaluation is not reliable, the user simulator might not generate the follow-ups that are representative of actual multi-turn conversations, and the model is more likely to ask questions that are already in the model's distribution.

- Proposed Acecoder is inefficient, has high cost and latency, and uses the same LLM for evaluation and generation, which can cause bias.  
The paper would be more academically sound if it were to present the dataset itself.

### Questions
Lots of hyperparameters have not been mentioned in the paper; please consider adding a section to discuss them in more detail.

### Soundness
3

### Presentation
4

### Contribution
3
