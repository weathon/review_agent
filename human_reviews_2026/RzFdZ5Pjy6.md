# Deception in Dialogue: Evaluating and Mitigating Deceptive Behavior in Large Language Models

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Large Language Models (LLMs) now interact with hundreds of millions of people worldwide, powering applications such as customer support, education and healthcare. However, their ability to produce deceptive outputs, whether intentionally or inadvertently, poses significant safety concerns. The unpredictable nature of LLM behavior, combined with insufficient safeguards against hallucination, misinformation, and user manipulation, makes their misuse a serious, real-world risk. In this paper, we systematically investigate the extent to which LLMs engage in deception within dialogue, and propose the belief misalignment metric to measure deception. We evaluate deception across four distinct dialogue scenarios, using five established deception detection metrics and our proposed metric. Our findings reveal this novel deception measure correlates more closely with human judgments than any of the existing metrics we test. Additionally, our benchmarking of 8 state-of-the-art models indicates that LLMs naturally exhibit deceptive behaviors 24.4% of the time, even when prompted with seemingly benign objectives. When prompted to deceive, LLMs are capable of increasing deceptiveness to 43% of turns. We further explore how to use reinforcement learning to fine-tune LLMs to reduce deceptive behaviors, leading to a 15% reduction compared to other fine-tuned models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper systematically studies deceptive behavior in dialogue settings for LLMs and proposes a new metric, belief misalignment, which measures how far a listener’s beliefs deviate from the true state of the world after interaction. Across four dialogue tasks, the authors compare existing deception metrics against belief misalignment and find the new metric aligns best with human annotations. Benchmarking shows that with seemingly benign prompts, LLMs still exhibit deception at a nontrivial rate (24.4%), rising to 43% when explicitly prompted to deceive. The authors then apply multi-turn reinforcement learning to jointly optimize task reward and an anti-deception objective, reducing deceptive behavior by about 15% at the dialogue level.

### Strengths
1.    Important, practically relevant problem. The paper targets real-world risks of deception/misdirection in LLMs, uses multi-task, multi-model evaluation, and offers contributions with clear safety and ethical significance.

2.    Systematic experiments and ablations. The study contrasts default, explicitly deceptive, and utility-maximizing prompting regimes; and compares base, instruction-tuned, and RL-tuned models, revealing how goal-directed prompting increases deception and how effects vary by task.

3.    Clear, reproducible mitigation path. The multi-turn RL fine-tuning with an anti-deception reward (PPO best) meaningfully lowers belief misalignment without materially harming task reward; the setup appears replicable and extensible.

### Weaknesses
1.    Dependence on LLM-as-Judge and potential circularity. Several metrics (including belief misalignment) rely on LLM evaluators to infer listener beliefs or label deception, importing evaluator bias/drift. Beyond human-correlation checks, the paper should add cross-evaluator robustness (different architectures/scales) and adversarial/attack tests to ensure the judge is not easily swayed.

2.    Idealized truth modeling and listener assumptions. Decomposing world state into k facts and assuming a “naïve update” listener aids formalization but abstracts away messy real-world issues (open-domain truth, uncertainty, multi-source evidence, value vs. fact). Testing on noisier, semi-structured factual tasks (e.g., medical/travel advice) and reporting failure modes would strengthen claims.

3.    Trade-offs with truthfulness/harmfulness not deeply analyzed. While RL reduces deception and preserves task reward, the paper provides limited analysis of how reducing deception interacts with other axes (truthfulness, politeness, persuasive efficacy). Instruction-tuned models’ higher deception on some tasks is mostly explained post hoc; targeted interventions (vary prompt strength/reward shaping) would be informative.

### Questions
see Weaknesses

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
The paper explores the evaluation and mitigation of LLMs' deception behaviors. Evaluation data across 4 tasks (e.g., house showing) is collected to evaluate several LLMs. Authors also present the belief misalignment metric, to enable more accurate evaluation of deception behaviors. Finally, a straight forward RL-based mitigation method is shown to reduce the frequency of deception behaviors.

### Strengths
- The authors collect new data for deception evaluation, which covers 4 tasks.
- The belief misalignment metric is proposed to evaluate deception.
- The mitigation method is straightforward and effective.

### Weaknesses
- The data diversity is a bit limited. For example, two different samples may only differ in whether whether the neighborhood is quiet. I suggest authors could expand the task scenarios to enhance the diversity.
- The generalization ability of the mitigation method is not clear. I understand that new configurations are used during test, but the differences are not big enough. For example, it is possible that a sample in the test set and a sample in the training set may only differ in whether whether the neighborhood is quiet.

### Questions
None

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
4

### Summary
The paper introduces a new framework for evaluating and mitigating deception in LLMs within dialogue settings. The authors propose a novel metric, "belief misalignment," which measures the deviation of a listener's beliefs from the ground truth after interacting with a potentially deceptive LLM. They evaluate across 8 LLMs, 4 dialogue tasks, and 5 existing deception metrics, and they demonstrate that belief misalignment correlates more strongly with human judgments than prior metrics. The paper also shows that LLMs can be prompted to be deceptive and that standard safety training like RLHF is not entirely effective at preventing deception. Finally, the authors demonstrate that multi-turn reinforcement learning finetuning using their proposed metric as a reward signal can significantly reduce deceptive behaviors in LLMs.

### Strengths
- The finding that belief misalignment has the highest Pearson correlation with human judgments of deception (0.788, Table 1) is a powerful piece of evidence for the metric's validity. 

- The use of multi-turn RL with PPO to fine-tune models to reduce belief misalignment is a practical and promising approach. The reported 15% reduction in deception compared to other fine-tuned models is a notable result (lines 92-94).

### Weaknesses
1. There are a few results that seem counterintuitive and warrant further explanation: 

- In Table 2, instruction-tuned models like Llama-3.1-8B-Instruct and Llama-3.1-70B-Instruct show higher default belief misalignment in the "Housing Showing" and "Charity" tasks than their base model counterparts. The paper suggests this is because instruction-tuning helps models better achieve goals, even if it requires deception (lines 419-422). While it seems plausible, this is a significant claim that needs more analysis. It runs counter to the general expectation that safety alignment should reduce harmful behaviors like deception.

- The claim that models trained with RLHF "still exhibit deception one fourth of the time" (line 90) is a strong statement. Given the variability in RLHF implementations, a more nuanced discussion of the specific RLHF methods used for the tested models would be beneficial.

2. While the results of the RL fine-tuning are promising (Table 3), the paper provides limited details about the implementation. For example, what was the specific reward function formulation? How were task reward and belief misalignment weighted?

### Questions
- The concept of "belief misalignment" assumes a quantifiable "ground truth" state of the world (lines 173-174). While this works for the chosen dialogue tasks, how would you adapt this metric to more open-ended or subjective domains where the "ground truth" is not easily defined?

- Could you elaborate on how your work significantly differs from or builds upon the findings of Park et al. (2023b) and Hubinger et al. (2024), which also explore deception in LLMs?

- The paper does not include a LLM usage declaration, as required in https://iclr.cc/Conferences/2026/AuthorGuide.

### Soundness
2

### Presentation
2

### Contribution
2
