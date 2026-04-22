# Self-Anchor: Large Language Model Reasoning via Step-by-step Attention Alignment

- Avg Score: 3.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 4, 2

## Abstract
To solve complex reasoning tasks for Large Language Models (LLMs), prompting-based methods offer a lightweight alternative to fine-tuning and reinforcement learning.
However, as reasoning chains extend, critical intermediate steps and the original prompt will be buried in the context, receiving insufficient attention and leading to errors. 
In this paper, we propose Self-Anchor, a novel pipeline that leverages the inherent structure of reasoning to steer LLM attention. Self-Anchor decomposes reasoning trajectories into structured plans and automatically aligns the model's attention to the most relevant inference steps, allowing the model to maintain focus throughout generation. 
Our experiment shows that Self-Anchor outperforms SOTA prompting methods across six benchmarks. 
Notably, Self-Anchor significantly reduces the performance gap between ``non-reasoning'' models and specialized reasoning models, with the potential to enable most LLMs to tackle complex reasoning tasks without retraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Self-anchor, a training-free method designed to mitigate attention misalignment in large language models (LLMs) during long answer generation. The core idea is to decompose reasoning tasks into structured plans and automatically steer the model’s attention toward the previous plan steps and the original question during generation. This uses existing attention steering methods on top of a prompt that encourages structured reasoning.

Overall, the work is interesting and promising, but it feels incremental and somewhat ad hoc, lacking rigorous justification and analysis of what specifically Self-anchor (and attention steering) solves. I would suggest to do a proper analysis of attentions before and after Self-anchor steering, showing that: (i) attention misalignement is the problem (as argued in some previous work) and (ii) self-anchor does indeed target the correct tokens and solve the attention misalignment. Also, because the design choices seem ad-hoc, one possible improvement is to just learn which tokens to steer which would have a slightly larger computational cost but would be compatible with any prompting scheme.

### Strengths
The approach combines planning-based prompting with attention steering, an intuitive integration.
The results seem to demonstrates robust improvements across reasoning benchmarks (math, commonsense, symbolic reasoning) and model sizes. Achieves competitive performance without expensive reinforcement learning or model fine-tuning.

### Weaknesses
Unverified justification: The paper attributes improvements to fixing attention misalignment but provides no direct evidence.
There is no visualization or quantitative comparison of attention weights before and after applying SPA.
There is no analysis contrasting attention patterns in successful vs. failed cases. I think that these analyses would add value to the paper

Design choices: The confidence-based scaling in Equation (4) appears ad hoc and poorly motivated. It is unclear why low token-level confidence should trigger stronger step-level adjustments. It was not clear to me how the confidence score p_avg influences the attention steering strength w_i. There may be notation overload in Equation (4), where i indexes both steps and tokens.

Tuning on test data:
Table 7 (Appendix) suggests design choices may have been tuned based on partial test-set performance (on two of the test sets).
This questions whether other choices were made by looking at test results.

Missing Statistical significance:
Tables 1 and 2 report no variance or statistical significance, making it unclear when and whether improvements are significant.
Yet, Figure 2 does report variance, and we can see that it quite high possibly overwhelming the reported gains. It is unclear that the improvements over RE2 are significant for any of the complexity bracket.

### Questions
Have you compared attention patterns in successful vs. failed cases, before and after applying SPA?
What is the theoretical or empirical motivation for using token-level confidence to modulate step-level attention in Equation (4)?
Were any hyperparameters or design choices (as shown in Table 7) tuned using test data, or was a held-out validation set used?
Can you report variance or statistical significance of the results in Tables 1 and 2? 
Are improvements over RE2 statistically significant given the observed variance in Figure 2?

### Soundness
2

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
This paper targets the problem of "attention misalignment" in Large Language Models (LLMs) during complex, multi-step reasoning tasks. To address this, the authors employ an attention steering mechanism proposed in previous work, Selective Prompt Anchoring (SPA).

The authors proposed SELF-ANCHOR. Unlike previous work that require manual specification of what to focus on, SELF-ANCHOR cleverly uses the model's own generated $plan_i$ as the dynamic, automatic anchor for generating $reason_i$.

This work is based on two key insights: (1) complex reasoning problems can be decomposed into structured
plans, and (2) each decomposed plan can naturally serve as a component for attention alignment, which they term the "self-anchor".

The authors claim this method significantly outperforms baselines and enables non-reasoning models to achieve performance "on par" with specialized, RL-enhanced thinking models.

### Strengths
1. The authors conducted comprehensive experiments on various tasks and models.

2. The ablations are comprehensive and some of them are solid.

### Weaknesses
1. This work lacks enough novelty, since it is only an application of previous SPA method on a plan and solve workflows. This raises the question: **can this philosophy be readily transferred to other reasoning workflows, or is its utility limited to this specific pipeline?**

2. Typo in Line 30, "Howevere, "

3. The claim in Line 266 (that the method achieves "competitive or superior performance" compared to RL-enhanced models) appears to be a significant overstatement. The authors' own data in Table 2 contradicts this claim. For example, on the MATH benchmark, Llama3.1-8B with SELF-ANCHOR scores 52.50, while its "thinking model" counterpart scores 72.50. This 20-point gap cannot be described as "competitive." A similar, significant gap is also present for the Phi-4-mini-4B model on the same benchmark.

4. As describle in Appendix E.2, the evaluation on the MATH benchmark was conducted on a **randomly sample 200 test instances**. This is not a reliable sample size. The authors should use the full test set (5k) or, at minimum, a well-established, representative subset (e.g., MATH-500).

5. The paper claims "minimal computational overhead" (Sec 3.6) based on the Token/sec metric (Table 3). This is highly misleading. The underlying SPA mechanism requires contrasting $logits^{original}$ with $logits^{mask}$. This necessitates a second, masked forward pass for every generated token, more than doubling the true computational cost. The Token/sec metric, which is bottlenecked by memory bandwidth, completely obscures this massive increase in compute.

### Questions
1. Line 261: "To investigate this question, we compare our method applied to non-reasoning LLMs against corresponding thinking models." What is the detail of these "corresponding thinking models", have you trained them using RL algorithms?

2. Appendix F: case study of CoT failures, it doesn't reveal the true problem is "attention misalignment", but, instead, the LLM you chose lacks common knowledge about 7 is a prime number.

3. Line 156: "This confidence score serves as additional factor to scale the attention steering strength $w_i$ in Equation 1. We discuss detailed design choices and experiments in Appendix B." I cannot find any details about how the additional factor is used in appendiex B.

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Self-Anchor, a prompting-based test-time method for improving multi-step reasoning in LLMs. The core idea is to first generate a structured plan and then steer attention toward (a) the original question and (b) the corresponding plan step during reasoning. The authors leverage a logit-level attention steering mechanism (SPA) to amplify the influence of selected tokens. Additionally, a confidence-based dynamic modulation adjusts the steering strength. Experiments on math and commonsense reasoning benchmarks suggest improvements over baseline prompting methods without model training.

### Strengths
1. Clear motivation: long reasoning suffers from attention misalignment.
2. Bridges planning-based prompting and logit attention steering in a unified pipeline.
3. Does not require model fine-tuning; test-time only approach is practical.
4. Evaluations span multiple reasoning benchmarks and LLM backbones.

### Weaknesses
1. Plan quality sensitivity not deeply studied. The method assumes that plans are reliable anchors. If plans are suboptimal or incorrect, the anchor may reinforce wrong reasoning. 
2. Fixed steering structure is rigid. The current design always attends to the question during planning and attends to {question, current plan} during reasoning. This hard-coded scheme may not generalize to tasks without clear hierarchical structure.
3. Limited ablations. Ablations exist but do not sufficiently isolate contributions of: 1) plan as anchor vs question anchor; 2_ dynamic steering vs fixed steering; 3) number of plans or plan granularity; 4) different attention steering positions.
4. Missing comparison with strong planning baselines. Particularly Tree-of-Thought (ToT) and variants.

### Questions
1. Why should plan tokens necessarily serve as good attention anchors?
2. Do wrong plans degrade attention by reinforcing incorrect contexts?
3. Can you provide experiments that vary anchor positions (e.g., question-only, plan-only, plan history, or dynamic learned anchors)?
4. Could you include comparison with ToT or other multi-path reasoning prompting?
5. How sensitive is the method to plan length and number of steps?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Self-Anchor, a prompting-based pipeline that improves LLM reasoning by dynamically aligning model attention with the reasoning structure. Self-anchor decomposes a problem into plan steps with corresponding reasoning steps; the model’s attention is steered toward the original question and the corresponding plan step using a modified version of Selective Prompt Anchoring (SPA). The core idea is to apply a form of logit blending (linear mix of logits) that steers attention. The model’s confidence dynamically modulates the steering strength. 

The method is evaluated on six benchmarks (GSM8K, AQuA, MATH, StrategyQA, T4D, and BIG-Bench Hard) using six base LLMs (LLaMA-3, Phi-4, and Qwen3 families). The authors report consistent accuracy gains over standard prompting baselines (CoT, Plan-and-Solve+, and Re-Reading), closing much of the gap between standard “non-reasoning” models and RL-enhanced reasoning models.

### Strengths
- The paper addresses an important limitation in LLM reasoning — the degradation of attention over long reasoning chains (“attention drift”). The motivation is well-grounded in prior empirical observations. It is generally well written. 
 
- The experimental section is extensive, covering six benchmarks and multiple model scales. The gains are consistent and meaningful (often 5–15%), and efficiency overheads are minor. Ablation studies are also compelling, especially showing degradation without attention steering.

- Self-anchor is lightweight and plug-and-play: it operates purely at the inference level and is compatible with off-the-shelf LLMs.

### Weaknesses
- Although the paper frames SELF-ANCHOR as an “attention steering” mechanism, the actual implementation does not modify the model’s attention computations. Instead, it performs a linear interpolation between logits from the full and masked prompts, which biases generation toward tokens associated with anchor regions. This effectively functions as a soft repetition rather than genuine control over attention flow. 

- SELF-ANCHOR seems to be a structured application of an existing steering technique (SPA). The paper’s novelty lies more in how SPA is applied (automated anchor selection + confidence scaling) than in the attention steering mechanism itself. 

- While accuracy improves, it is unclear whether self-anchor leads to more faithful reasoning (i.e., correct intermediate steps) or simply produces longer reasoning chains which is helping the accuracy. 

- The baselines focus primarily on older prompting methods (CoT, Plan-and-Solve, Re-Reading). It is unclear how SELF-ANCHOR compares to newer inference time scaling frameworks such as ToT (Tree-of-Thoughts), Graph-of-Thought, Process Reward Modeling with MCTS, or Stepwise Judges that also mitigate reasoning drift.

### Questions
Please check the above reviews.

### Soundness
3

### Presentation
3

### Contribution
2
