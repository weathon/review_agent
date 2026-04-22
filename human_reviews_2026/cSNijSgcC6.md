# Enhancing Reliability across Short and Long-Form Question Answering via Reinforcement Learning

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4

## Abstract
While reinforcement learning has unlocked unprecedented complex reasoning in large language models, it has also amplified their propensity for hallucination, creating a critical trade-off between capability and reliability. This work confronts this challenge by introducing a targeted RL framework designed to mitigate both intrinsic and extrinsic hallucinations across short and long-form question answering. We address extrinsic hallucinations (flawed internal knowledge) by creating a novel training set from open-ended conversions of TriviaQA. Concurrently, we tackle intrinsic hallucinations (unfaithfulness to context) by leveraging long-form texts from FineWeb in a fact-grounding reward scheme. To further bolster reliability, our framework explicitly rewards the model for refusing to answer unanswerable questions, thereby cultivating crucial cautiousness. Extensive experiments demonstrate that our methodology yields significant performance gains across a diverse suite of benchmarks, substantially reducing both hallucination types. Ultimately, this research contributes a practical framework for resolving the critical tension between advanced reasoning and factual trustworthiness, paving the way for more capable and reliable large language models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a targeted RL framework to mitigate both intrinsic and extrinsic hallucinations in LLMs for short and long-form question-answering tasks. The authors make several contributions, including the development of novel training datasets and a thorough empirical analysis of various training strategies. Despite the novelty of the application scenario, the work is not solid enough, especially the presence of several serious typology issues which leads me to believe this work requires significant further effort.

### Strengths
1. The paper primarily trains the LLM's refusal capability in short-form QA to enhance reliability and successfully generalizes this to long-form QA scenarios, which is a novel and valuable exploration of the setting.
2. The paper presents some interesting insights through its experiments, such as the asymmetry in learning dynamics (the model learns to refuse much faster than it learns to be correct).

### Weaknesses
1. Lack of baselines. Comparing only against the base model makes it difficult to verify that the proposed method is indeed effective compared to existing methods. There are many existing alignment works in the direction of LLM reliability, such as [1][2].
2. Insufficient analysis of the post-trained model. The analysis in this work focuses on experimental settings and additional insights, while the advantages of the method are only shown in the main table and are not concretely demonstrated. For example, there is no specific analysis on whether the model selectively refuses to answer difficult questions, the distribution of refusal behavior across datasets of varying difficulty, or the generalization of refusal behavior trained on short-form QA to long-form QA.
3. Reliance on an LLM Judge with limited capabilities. You mentioned the potential negative impact of the LLM judge's performance on the results in several places. For instance, the paper mentions that "the evaluator may provide a misleading training signal." It also notes that "the LLM judge consistently prefer the RL-tuned model’s responses" over the baseline model, but in the Informative Win-Rate setup, the focus should have been on information density while ignoring accuracy. These errors raise concerns about the validity of the training results and the analytical conclusions.
4. Under-investigated Catastrophic Forgetting. The RL-tuned Qwen3-4B model showed a notable performance decline on the AIME mathematical reasoning benchmark. The analysis of this phenomenon is insufficient. Exploring why this occurred and potential mitigation strategies would have strengthened the paper.
5. There are several serious and non-negligible typographical errors. 
    1. The observations and conclusions in Figure 3 are inconsistent with the text in section 4.2. The text states that full CoT improves performance on Facts Grounding but degrades performance on LongFact. However, the chart shows that on Facts Grounding, the accuracy of full CoT (orange line) is lower than that of without CoT (green line), while on LongFact, full CoT is significantly better than the other methods. Figure 3 is in complete conflict with the observations and conclusions.
    2. The content on line 283 is incomplete. The sentence "To improve the model’s ability to recognize unsolvable queries, 25" is cut off.
    3. The paper alternates between "Mimo" and "MiMo". It is recommended to maintain consistent capitalization.

References
[1] https://arxiv.org/abs/2311.09677

[2] https://arxiv.org/abs/2403.18349

### Questions
1. In the CoT experiments in section 4.2, you mentioned that the poor performance of Full CoT might be because "the evaluator misinterprets tentative or self-corrected steps within the reasoning chain as final, incorrect claims." However, this could be avoided by simply using an LLM to extract the final answer before evaluation. I suggest you could try this approach.
2. In section 5.2, you demonstrate that the model fails to generalize its refusal capability without explicit instructions in the prompt. Have you tried to guide the model to generate refusal responses without including any explicit refusal instructions in the training data? This approach was used in many papers like [1] and was shown to work. Maybe you could give it a try.
3. In your long-form QA reward function, the hyperparameters alpha (format penalty) and beta (information density penalty) were both set to 0.2. How were these values determined? Was any sensitivity analysis performed to understand their impact on the training trade-offs?

References

[1] https://arxiv.org/abs/2311.09677

### Soundness
2

### Presentation
2

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
This paper proposes a reinforcement learning (RL) framework aimed at mitigating both intrinsic and extrinsic hallucinations in Large Language Models (LLMs) across short and long-form Question Answering (QA) tasks. The core contributions are threefold:

1. A Novel RL Framework: An application of a GRPO-variant RL algorithm to jointly improve factual accuracy and encourage refusal on unanswerable questions.

2. New Training Datasets: The construction of two long-form QA datasets: one for intrinsic hallucination (using FineWeb-derived, reference-grounded Q&A) and one for extrinsic hallucination (using open-ended questions converted from TriviaQA).

3. Empirical Analysis: A series of experiments analyzing factors like Chain-of-Thought (CoT) supervision and the trade-off between factual accuracy and information density.

The authors demonstrate performance improvements on a suite of standard benchmarks (e.g., TriviaQA, Facts Grounding, LongFact) and provide insights into the learning dynamics of cautiousness versus correctness.

### Strengths
Comprehensive Scope: The paper tackles a broad and critical problem—mitigating both intrinsic and extrinsic hallucinations—across a wide range of QA formats. This holistic approach is more ambitious than prior work focused on a single facet.

High-Quality Empirical Evaluation: The experimental section is thorough, using multiple models (MiMo, Qwen) and a diverse set of well-chosen benchmarks to validate the claims from different angles.

Valuable Analysis: The investigation into CoT supervision (Section 4.2) yields a clear and practical finding: that full CoT supervision is not cost-effective for this task. The analysis of asymmetric learning dynamics (Section 5.1) and the accuracy-density trade-off (Section 5.3) provides nuanced insights beyond mere performance numbers.

Clarity of Core Method: The overall framework and the design of the reward functions for short and long-form QA are explained with sufficient clarity for the reader to understand the high-level approach.

### Weaknesses
Reproducibility Crisis: The heavy, non-ablatable dependence on multiple proprietary, massive LLMs (GPT-OSS-120B, Gemini-2.5-Pro) for both training and evaluation is a major weakness. It places the work outside the reach of most academic labs for reproduction or direct building upon, which is a significant concern for a scientific conference.

Lack of Ablation Studies: The paper presents a complex pipeline but does not isolate the effects of its key components. How much do the novel datasets contribute versus the reward design? What is the individual impact of removing the KL penalty? Without ablations, the source of the improvements remains ambiguous.

Insufficient Algorithmic Detail: As noted, Section 3.2 is unacceptably vague for a technical conference. The "variant of GRPO" must be described in precise detail, or the code must be released for the review to properly assess the algorithmic contribution.

Superficial Treatment of a Key Finding: The failure of instruction generalization (Section 5.2) is a critical and disappointing result, but it is not explored in depth. Why does this happen? Is it a limitation of the model size, the training data distribution, or the RL objective? A deeper discussion is needed.

### Questions
1. Reproducibility & Cost: Given the reliance on GPT-OSS-120B for reward modeling and evaluation, what was the total computational and financial cost of this project? Will the authors commit to releasing their synthesized training datasets and the exact prompts used for the LLM judge to ensure a degree of reproducibility for the community?

2. Ablation Study: Can the authors provide an ablation study to quantify the contribution of (a) the new long-form QA datasets, and (b) the individual terms in the long-form reward function (e.g., the information density penalty)?

3. Algorithmic Detail: Could the authors provide a detailed, formal description of their GRPO variant, specifically clarifying the "Dynamic Sampling" and "Clip-Higher" mechanisms, and justifying the removal of the KL divergence penalty?

4. Generalization of Refusal: The finding in Section 5.2 that refusal does not generalize without explicit instructions is significant. What are the hypothesized reasons for this failure? Did the authors experiment with alternative training strategies (e.g., gradually phasing out the instruction) to encourage generalization?

5. Statistical Significance: The results in Tables 1 and 2 represent a single run. Can the authors report results with standard deviations over multiple random seeds, or perform statistical significance tests to bolster the claims of improvement?

### Soundness
3

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
4

### Summary
This work considers three types of QA tasks: short-form QA, and long-form QA with and without references. To ensure good performance across all three tasks, the authors synthesized training data and conducted reinforcement learning training, ultimately providing several experimental insights.

### Strengths
* The paper is clearly written and easy to follow. Although the work leans toward an engineering contribution rather than deep theoretical novelty, it provides ample implementation details and experimental transparency, making it highly reproducible and accessible to practitioners.
* The authors carefully construct a new training corpus integrating both short- and long-form QA data (from FineWeb and TriviaQA), explicitly designed to address intrinsic and extrinsic hallucinations.

### Weaknesses
* Although the paper’s data construction process is well-executed and methodologically solid, all datasets are either synthetic or repurposed from existing sources (TriviaQA and FineWeb). As a result, the work does not introduce any genuinely new data or annotations, limiting its originality as a dataset contribution.
* The reinforcement learning experiments are relatively superficial. Key ablations, such as varying the reward weighting coefficients, analyzing the effect of removing or modifying specific reward components, or exploring different KL/divergence regularization schemes, are missing. These are essential to understand the robustness of the proposed reward formulation.
* The experiments primarily compare RL-trained models against their untuned base versions on the same dataset. This setup is insufficient to isolate the contribution of the newly constructed dataset or the RL pipeline itself. Comparisons against (1) SFT-only models trained on the same data, and (2) RL models trained on alternative open-source datasets, are necessary to demonstrate that the proposed dataset and framework provide unique benefits.
* 
* typo: It seems that something is missing on line 151.

### Questions
* Why did you choose MiMo-7B and Qwen3-4B for their experiments?
In other words, why didn’t you conduct experiments using Qwen3-8B instead?

### Soundness
2

### Presentation
2

### Contribution
2
