# Benchmarking and Rethinking Knowledge Editing for Large Language Models

- Decision: Reject
- Scores: 4, 4, 2, 4

## Abstract
Knowledge editing aims to update the embedded knowledge within large language models (LLMs). However, existing approaches, whether through parameter modification or external memory integration, often suffer from impractical evaluation objectives and inconsistent experimental setups. To address this gap, we conduct a comprehensive and practically oriented benchmarking study. Particularly, we introduce more complex event-based datasets and general-purpose datasets drawn from other tasks, in addition to fact-level datasets. Our evaluation covers both instruction-tuned and reasoning-oriented LLMs, and we adopt a realistic autoregressive inference setting rather than teacher-forced decoding. Beyond single-edit assessments, we also consider multi-edit scenarios to better capture real-world requirements. We employ four evaluation metrics, with particular emphasis on portability. We compare all recent methods against a simple baseline named Selective Contextual Reasoning (SCR). Empirical results show that parameter-based editing methods perform poorly under realistic conditions, while SCR consistently outperforms them across all settings. Our findings suggest that when knowledge updates are minimal, parameter adjustments can sometimes yield higher reasoning efficiency; however, in most cases, selectively injecting external knowledge into the context proves to be the more robust strategy. Overall, this study delivers a comprehensive evaluation framework for future research and offers fresh perspectives for rethinking knowledge editing methods. The implementation is provided in the Supplementary Materials.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a comprehensive benchmarking study of knowledge editing methods for large language models (LLMs), evaluating both parameter-modifying and non-parameter-modifying approaches under realistic settings. The authors introduce a simple yet strong baseline—Selective Contextual Reasoning (SCR)—and compare it against 12 recent methods across multiple datasets, model types, and editing scenarios. The results show that parameter-based editing methods often fail under sequential or complex editing settings, while context-based methods like SCR perform more robustly.

### Strengths
(1) The benchmarking is extensive and practical, covering autoregressive inference, sequential editing, event-level knowledge, and reasoning-oriented LLMs.
(2) The introduction of SCR as a strong and simple baseline is valuable and highlights the effectiveness of retrieval-augmented in-context learning.
(3) The evaluation includes multiple dimensions (reliability, generalization, locality, portability) and considers both general and reasoning-specific LLMs.
(4) The study raises important questions about the practicality of parameter-based editing methods in real-world scenarios.

### Weaknesses
(1) The contribution is somewhat incremental, as the benchmarking is largely conducted on existing datasets and follows established evaluation protocols without introducing new datasets or evaluation paradigms.
(2) While the extension to reasoning-oriented LLMs is a plus, the evaluation directly reuses existing tests rather than developing specialized metrics or settings tailored to reasoning capabilities.
(3) The study primarily experiments with smaller-scale models (e.g., 7B–14B parameters), which may limit the generalizability of findings to larger, state-of-the-art LLMs.

### Questions
What is the difference between SCR and In-Context Learning-based approaches？

### Soundness
3

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
This paper presents a comprehensive benchmark of knowledge editing methods for LLMs across five families. The authors argue that prior evaluations over-emphasize on fact-level edits and overlook portability/generalization. They propose a unified, more realistic setup, e.g., including event-level and general-purpose datasets beyond fact triples, evaluating instruct and reasoning-oriented LLMs, and using autoregressive inference instead of teacher-forcing. They also introduce a simple baseline, i.e., selective contextual reasoning (SCR), which retrieves relevant knowledge text and injects it into the prompt. Empirically, parameter-modification methods underperform or collapse under sequential edits, while context-based approaches (IKE/ICE) and SCR are substantially more stable.

### Strengths
+ I agree that autoregressive decoding, sequential edits, and portability are all important and often ignored, and it's good to see that the paper productively consolidates them in one framework.
+ The author conducted extensive experiments that evaluates 12 recent methods across multiple LLMs (instruct + reasoning) and dataset types (facts, event-level; plus general reasoning benchmarks).
+ It's also interesting to see that that editing can harm reasoning capabilities.

### Weaknesses
- Although the paper provides some empirical insights, the core advance is an evaluation protocol + dataset selection and a simple SCR baseline. There is no new algorithmic insight for knowledge editing itself, which leads to limited novelty.

- The scale of edits seem to be limited. The experiments are conducted on 1 or 100 edits, but it seems that mass editing method such as Memit claims that they can scale up to thousands of edits. This raises concerns on whether the conclusions hold for larger-scale editing tasks.

- The conclusion suggests relying on in-context learning for small updates or re-training for large volumes, but it would be great if the author can have more in-depth discussions on how knowledge editing methods can evolve for more effective small-scale update.

### Questions
Please refer to my summary of weaknesses.

### Soundness
3

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
3

### Summary
This paper presents a comprehensive benchmarking study of knowledge editing methods for LLMs. The authors evaluate recent methods across on various LLMs, datasets, and under practical settings like autoregressive inference and sequential editing. They also compare those methods against Selective Contextual Reasoning (SCR). The paper's key findings suggest that parameter-modification-based editing methods perform poorly in realistic scenarios, often degrading the model's reasoning abilities and failing under sequential edits. In contrast, the SCR baseline consistently outperforms these methods, demonstrating robustness and effectiveness. The authors conclude that for knowledge updates, selectively injecting external knowledge into the context is a more robust strategy than modifying model parameters, especially when dealing with a limited number of updates.

### Strengths
1. This paper is clearly written, well organized, and generally easy to understand.

2. This paper addresses the important problem of knowledge editing in LLMs.

3. The authors' effort in conducting a large-scale, comprehensive benchmark is commendable.

### Weaknesses
The primary weakness of this paper is its limited novelty. While the benchmarking effort is extensive, the core research questions and many of the conclusions align closely with existing knowledge and prior work, making the contribution more of a confirmation than a new discovery.

1. RQ1 investigates editing under autoregressive inference and sequential editing scenarios. The challenges of sequential editing are well-investigated problems in the field. Similarly, previous works such as [1], [2] and [3] have already adopted autoregressive decoding in their evaluations, moving away from the less practical teacher-forcing paradigm. Therefore, the findings in this section, which show that parameter-modification methods struggle under these conditions, are largely confirmatory.

2. RQ3 explores editing more complex, event-level knowledge. However, the conclusion that parameter-modification methods struggle to handle the complex semantic relationships inherent in events is an expected extension of findings from fact-based editing. The results are similar to those observed in [1] and [2].

3. RQ4 investigates the trade-off between edit time and inference latency. It is intuitive that methods involving parameter updates would have higher edit-time costs but no additional inference overhead. Conversely, context-based methods that augment the input prompt naturally increase inference latency due to longer sequence processing. This conclusion does not offer new insights.

[1] Everything is Editable: Extend Knowledge Editing to Unstructured Data in Large Language Models

[2] AnyEdit: Edit Any Knowledge Encoded in Language Models

[3] MQuAKE: Assessing Knowledge Editing in Language Models via Multi-Hop

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper conducts a comprehensive and unified benchmark of knowledge editing methods for large language models (LLMs), addressing the inconsistency and unrealistic assumptions in prior work. It evaluates twelve representative approaches across five paradigms—parameter-based, meta-learning, adapter-based, in-context, and external-memory—using diverse datasets spanning factual, event-level, and reasoning tasks. The authors introduce four evaluation dimensions (reliability, generalization, locality, and portability) and propose a simple yet strong baseline, Selective Contextual Reasoning (SCR), which retrieves and injects relevant knowledge into prompts. Experiments on multiple LLMs (Llama2/3, Mistral, DeepSeek-R1) reveal that parameter-modification methods collapse under sequential edits and harm reasoning ability, while SCR and other context-based methods remain stable and robust. The study concludes that context-driven knowledge integration is a more practical and reliable approach than parameter editing for real-world LLM updates.

### Strengths
1. This paper is good writing and easy to follow. The idea and findings are presented very clear.
2. This paper conduct comprehensive experiments covering different types of base models, editing methods and benchmarks.
3. The findings are promising and reasonable.

### Weaknesses
1. The technical and even the practical contribution is limited. This paper didn't provide theoretical analysis why knowledge editing models hurt the model's capability or explain why sequential editing will fail. Second, this paper didn't give any technical contribution in how to mitigating the side effects of knowledge editing. Third, although the paper claims that they "aim to conduct a comprehensive benchmark study", they only use existing datasets for evaluation, instead of making any contribution for building a benchmark. Although the paper introduce the SRC, the main idea of SRC is too simple which is not technical enough for a published paper, or the improvement is not significant based on Table 1/2/3 results. To summary, this paper only conduct evaluation experiments and summarizes some findings, lacking of deep analysis and significant contribution.
2. In line 95, this paper claims that "Regarding the editing scenarios, many studies do not test in sequential / continuous editing scenario", however, there are many knowledge editing work that conduct sequential editing experiments: (1) Navigating the Dual Facets: A Comprehensive Evaluation of Sequential Memory Editing in Large Language Models (https://arxiv.org/abs/2402.11122); (2) Model Editing at Scale leads to Gradual and Catastrophic Forgetting (https://arxiv.org/pdf/2401.07453). Both paper not only conduct comprehensive analysis, but also provides analysis and explanation of why the side effects happens and how to solve it (or provide some potential direction). It would be better to incorporate those papers into related work, and clarify the differences between your work and those related work.

### Questions
I have a question about SCR: how sensitive is SCR’s performance to retrieval quality (e.g., noisy or incomplete knowledge bases)? How do you make sure the retrieval is correct to make sure the promising of this method?

### Soundness
2

### Presentation
3

### Contribution
2
