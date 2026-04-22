# TACLer: Tailored Curriculum Reinforcement Learning for Efficient Reasoning

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
Large Language Models (LLMs) have recently demonstrated remarkable performance on complex reasoning tasks, especially when equipped with long chain-of-thought (CoT) reasoning. However, eliciting long CoT reasoning  typically requires large-scale reinforcement learning (RL) training, while often leading to overthinking with redundant reasoning steps. To improve learning and reasoning efficiency, while preserving or even enhancing performance, we propose TACLer, a tailored curriculum reinforcement learning framework that gradually increases the complexity of the data based on the model's proficiency in multi-stage RL training. Our framework features two core components: (i) tailored curriculum learning that determines what knowledge the model lacks and needs to learn in progressive stages; (ii) a hybrid Thinking/NoThinking reasoning paradigm that balances accuracy and efficiency by enabling or disabling the Thinking mode. Our experiments show that TACLer yields a twofold advantage in learning and reasoning: (i) it reduces computing effort by cutting training compute by over 50\% compared to long thinking models and by reducing inference token usage by over 42\% relative to the base model; and (ii) it improves accuracy by over 9\% on the base model, consistently outperforming  state-of-the-art Nothinking and Thinking baselines across four math datasets with complex problems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Reinforcement Learning (RL) models trained with long Chain of Thought (CoT) reasoning often struggle with costly and excessive "thinking" due to redundant tokens. To mitigate this, the authors propose TACLer, a novel approach that categorizes data by difficulty and employs a hybrid strategy, dynamically deciding whether to engage in complex thought processes or solve problems directly. The model was trained using a modified GRPO in a continual learning framework. When applied to mathematical datasets, this method demonstrated improved accuracy while significantly reducing the length of CoT reasoning.

### Strengths
- The paper presents interesting, although already well-known, findings. The use of a hybrid thinking model, which employs a long chain of thought for complex cases and direct answers for simpler ones, is particularly interesting. Ablation studies demonstrate the benefits of this strategy compared to either extreme.

- The paper is well-written and effectively highlights the desired accuracy-efficiency Pareto gains.

### Weaknesses
- The paper's main idea that a hybrid model is better than "thinking" or "non-thinking" ones doesn't have enough experimental proof. The experiments only used one small model (1.5B, which isn't big enough to say for sure) and one domain (math). We don't know if this approach would work for bigger models or other areas like coding. Moreover, using the DeepScaleR dataset is a bit iffy because it might have some of the same data as the AMC and AIME datasets. It's not clear if the authors checked for duplicates or overlaps.

- A significant concern with the current setup is during the inference process. There's no automatic method to determine whether a query requires "thinking mode" or not. If a user runs it twice, which output should be selected is not clear. It would be valuable to establish an upper bound for direct thinking and non-thinking models (by taking a union of their accuracies) and compare it against TACLer. This would help in assessing the limitations of hybrid models. Furthermore, it appears that TACLer's "no-thinking mode" often incurs higher token costs than comparable modes in other models, such as Qwen. The reason for this discrepancy is not clear.

- It's tough to argue that how hard a problem is directly relates to how much it costs in tokens. For example, a simple multi-step math word problem would use more tokens than just multiplying two three-digit numbers. But TACLer would say the word problem needs "thought" and the multiplication doesn't, which feels a bit off.

- The paper really talks up the cost savings of curriculum learning, but it doesn't get into the training costs (like FLOPS, GPU hours, etc.). Every step of this process needs a full dataset inference and then difficulty labeling. This seems like a very expensive process. Also, Table 6 brings up using R1 Qwen for difficulty assessment. The paper should explain what this model is, why it was picked, and why they didn't just use the generation length method they mentioned earlier instead.

- [MINOR] Why was KL and clipping removed from GRPO? Was the approach tested with KL and clipping and was that not working well?

### Questions
1. Checking duplicates in DeepScalerR dataset and making sure there are no train test overlaps.

2. How TACLer works during inference is not clear? If the user needs to decide which mode to choose, it seems like the best bet is to always choose the thinking mode and then this hybrid approach converges to a thinking model. 
3. More discussion on the training cost is needed to understand the train costs compared to other baselines. 
4. Can the approach be generalized beyond math? And can this be scaled to larger models or to other model families?
5. Table 6 can also be reported in terms of absolute token costs instead of percentage. That would be more clear. 
6. Why were tweaks made to the GRPO algorithm? What was the reasoning behind it?
7. How are the token costs calculated? Which tokenizer was used?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
# Summary

This paper introduces **TACLer** (Tailored Curriculum Reinforcement Learning), a framework designed to improve both training efficiency and reasoning efficiency for large language models on mathematical reasoning tasks. The key contributions are:

1. **Tailored Curriculum Learning**: Unlike conventional curriculum learning that uses fixed difficulty metrics, TACLer dynamically assesses problem difficulty based on the model's current proficiency (measured by pass rate), organizing training data into progressive stages that adapt to what the model specifically needs to learn.

2. **Hybrid Reasoning Mode**: The framework incorporates both Thinking mode (with explicit `<thinking>` tags for long chain-of-thought reasoning) and NoThinking mode (concise responses without explicit thinking markers), allowing flexible balance between accuracy and computational efficiency.

3. **Training Efficiency**: By maintaining a consistent 8K context window across all training stages (vs. 8K→24K in baselines), TACLer reduces training compute by over 50% compared to models like DeepScaleR-1.5B.

4. **Strong Empirical Results**: On four math benchmarks (MATH500, AMC, AIME 2024/2025), TACLer achieves +9-11% accuracy improvement over the base model while reducing average token usage by 42-51%, outperforming both long-thinking baselines and efficient reasoning methods.

The paper demonstrates that curriculum learning tailored to model proficiency, combined with hybrid reasoning modes, can simultaneously improve learning efficiency (faster training), reasoning efficiency (shorter responses), and accuracy on complex mathematical reasoning tasks.

### Strengths
# Strengths

The paper makes meaningful contributions in both methodology and practical impact. **Originality**: TACLer introduces a novel model-adaptive curriculum learning approach that dynamically adjusts training difficulty based on the model's actual pass rate rather than arbitrary metrics (e.g., input length), effectively addressing the high truncation rates (>40%) observed in prior work. **Quality and Significance**: The framework achieves substantial practical improvements—reducing training compute by over 50% while improving accuracy by 9-11% and reducing inference tokens by 42-51% across four challenging math benchmarks. The experimental evaluation is comprehensive, comparing against 10+ recent baselines covering both long-thinking and efficient reasoning methods.

### Weaknesses
## 1. Writing Quality Issue (Minor)
**Line 48**: Repetitive phrasing - "applying techniques such as such as length-based rewards" contains duplicate "such as". This should be corrected to "applying techniques such as length-based rewards".

## 2. Insufficient Explanation of Efficiency Gains in Thinking Mode
**Section 4.2 & 3.3**: The paper reports significant response length reduction (-42.7%) in Thinking mode (Table 1), but fails to explain the mechanism behind this improvement. The training methodology (Section 3.3) only describes using GRPO with two tricks (removing KL loss and increasing upper clip bounds), neither of which explicitly targets response length control. Since GRPO itself suffers from overthinking issues (as acknowledged in Section 2), the source of efficiency gains remains unclear. The paper needs to:
- Explicitly explain which component(s) of the framework lead to shorter responses
- Provide ablation studies isolating the effects of curriculum learning, hybrid mode, and GRPO modifications
- Discuss the causal relationship between the proposed methods and efficiency improvements

## 3. Unclear Training Procedure for Hybrid Reasoning Modes
**Section 3.2**: The hybrid reasoning mode description lacks critical implementation details. It remains ambiguous whether:
- Both Thinking and NoThinking modes are **explicitly trained** during RL, or
- Only Thinking mode is trained while NoThinking is achieved solely through prompting (e.g., "<thinking> Okay, I think I can solve it directly. </thinking>")

This ambiguity undermines the theoretical analysis (lines 195-206) about mutual benefits between modes. Claims like "a model trained solely in the NoThinking mode may compensate..." and "when trained jointly with the Thinking mode, the model can leverage a compression effect..." assume joint training, but if NoThinking is merely prompt-based inference, these hypotheses lack empirical foundation. 

## 4. Inconsistent Explanation of Response Length Trends
**Section 4.3 & Figure 4**: The paper observes opposite trends in response length across training stages - NoThinking responses **increase** while Thinking responses **decrease** - but provides no mechanistic explanation. This observation appears to contradict the hypothesis in Section 3.2 that "NoThinking mode promotes conciseness by reducing verbosity." While the paper acknowledges "NoThinking does not simply mean 'short'", readers need to understand:
- Why NoThinking responses grow longer during training (is this the "compensation effect" mentioned in 3.2?)
- Why Thinking responses become more concise (learned efficiency? curriculum effect?)
- What training dynamics or architectural properties drive this divergence

Without this analysis, it's unclear whether the observed phenomena align with the paper's design intentions or represent unexpected emergent behaviors.

I am happy to raise my score if your response is reasonable.

### Questions
## Q1: Clarification on Pure-NoThinking Model in Table 3
**Section 4.3, Table 3**: The paper states "We compare TACLer with the pure NoThinking model trained in the first stage." However, based on the methodology description (Sections 3.1 and 3.3), the first stage only differs in data difficulty (easier problems via curriculum learning), not in the reasoning mode. Could you clarify:
- How is the "Pure-NoThinking" model produced in the first stage? 

This clarification is essential to understand the experimental design and validate the claims about hybrid mode benefits.

## Q2: Mechanism Behind Efficiency Improvements
**Section 4.2 & 3.3**: Given that:
1. Your RL objective (GRPO + two tricks) doesn't include explicit length penalties
2. GRPO alone doesn't solve overthinking (as noted in Section 2)
3. Curriculum learning primarily addresses data difficulty, not response verbosity

Could you explain the specific mechanism(s) responsible for the 42.7% reduction in response length? Possible hypotheses to address:
- Do the two GRPO modifications (removing KL, increasing ϵ_high) implicitly encourage conciseness? If so, how?
- Does curriculum learning lead to more efficient reasoning patterns? Why and What evidence supports this?

Clarifying this would strengthen the paper's contribution and help readers understand which aspects of your framework are essential for efficiency.

## Q3: Theoretical Justification for Opposite Length Trends
**Section 4.3, Figure 4(b)**: The observation that NoThinking responses grow longer while Thinking responses grow shorter is intriguing but unexplained. Could you:
- Provide theoretical analysis or intuition for why these opposite trends emerge?
- Verify whether this aligns with your hypothesis in Section 3.2 about "compensation" in NoThinking mode?

Understanding this phenomenon would help assess whether your framework achieves the intended balance between accuracy and efficiency or whether the behaviors emerge for other reasons.

I am happy to raise my score if your response is reasonable.

### Soundness
2

### Presentation
3

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
This paper proposes TACLer, a reinforcement learning framework that trains models to reason efficiently without losing accuracy. It builds a tailored curriculum that adapts to what the model actually knows, instead of feeding random easy-to-hard data, and trains it progressively through RL. The model also learns to switch between Thinking (long reasoning) and NoThinking (concise reasoning) modes, allowing efficient inference without overthinking.

### Strengths
The idea of tailoring the curriculum to the model’s own proficiency is a clean and original way to make RL-based reasoning training more efficient. The hybrid Thinking/NoThinking setup is also elegant. The experiments are well-executed and comprehensive, showing consistent gains across multiple reasoning benchmarks. Overall, it’s a thoughtful and well-structured paper that pushes the discussion on efficient reasoning.

### Weaknesses
There seems to be multiple related works that aren't referenced, could the authors comment on the difference to these listed works?

https://arxiv.org/pdf/2505.14970
https://arxiv.org/pdf/2506.06632

### Questions
Could the authors do a more recent related work comparison.

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
The authors present a simple curriculum strategy for training for reasoning with GRPO. They propose to assess the model's accuracy on the questions first to go from easy to hard questions, which in combination with the hybrid reasoning idea, seem to improve the model's reasoning further than others. But, there are a few works [1, 2] that explored exactly this idea and am skeptical of the novelty of the idea here despite strong results.

[1] Parashar, Shubham, et al. "Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning." arXiv preprint arXiv:2506.06632 (2025).
[2] Ji, Yunjie, et al. "How Difficulty-Aware Staged Reinforcement Learning Enhances LLMs' Reasoning Capabilities: A Preliminary Experimental Study." arXiv preprint arXiv:2504.00829 (2025).

### Strengths
- The paper is clearly written and the proposed contribution of designing a curriculum based on the current model's success rate is warranted. They also show strong results on the math datasets, and it interesting how the hybrid reasoning mode seems to yield concise reasoning.

### Weaknesses
- The main limitation that I see with the paper is novelty. There are mainly two works cited in the summary [1, 2] that seem to propose the same idea and they do not compare against these baselines. If the authors tackle this point effectively, I would be willing to revise my score.

### Questions
- What does it mean to have complete solution but wrong final answer? Does that just mean formatting error for the answer?
- How is the hybrid mode actually used? Which questions are trained with the thinking and no-thinking?

### Soundness
2

### Presentation
3

### Contribution
1
