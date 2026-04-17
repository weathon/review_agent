# Know When to Fold 'Em: Predicting an LLM-Judge for Efficient but Performant Inference

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
Large language models (LLMs) face a fundamental trade-off between computational efficiency (e.g., number of parameters) and output quality, especially when deployed on computationally limited devices such as phones or laptops. One way to address this challenge is by following the example of humans and have models ask for help when they believe they are incapable of solving a problem on their own; we can overcome this trade-off by allowing smaller models to respond to queries when they believe they can provide good responses, and deferring to larger models when they do not believe they can. To this end, in this paper, we investigate whether models can predict---prior to responding---how an LLM judge would score their output. We evaluate three approaches: zero-shot prediction, prediction using an in-context report card, and supervised fine-tuning. Our results show that larger models (particularly reasoning models) demonstrate good zero-shot prediction abilities, while smaller models require in-context report cards or fine-tuning for reliable predictions. While the effectiveness varies across datasets, both approaches can substantially improve smaller models' prediction accuracy, with fine-tuning achieving mean improvements up to 52\% across datasets. These findings suggest that models can learn to predict their own performance limitations, paving the way for more efficient and self-aware AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This is a well-executed paper that tackles the practical challenge of balancing an LLM's computational efficiency with its output quality. The core idea is to empower smaller, faster models to "know what they don't know" by having them predict, before generating a full response, how an LLM-judge would score their potential answer. If the predicted score is poor, the query can be deferred to a larger, more capable model. The paper compellingly evaluates three methods for this pre-hoc prediction: zero-shot, an in-context "report card," and supervised fine-tuning. The authors demonstrate that while large models have some innate self-assessment ability, smaller models do not. However, both the report card and fine-tuning methods dramatically improve this capability, with fine-tuning showing the most promise.

### Strengths
- The paper addresses a core, real-world problem. The trade-off between cost/latency and quality is a primary concern for anyone deploying LLMs, and the proposed "deferral" system is an elegant solution.

- The concept of an in-context "report card" is a particularly clever, training-free approach. It’s a great interim solution for closed-weight models where fine-tuning isn't an option.

### Weaknesses
- The study relies on a single LLM-judge (Llama 3.3 70B). While the judge's evaluations were confirmed to be stable, it would be great to see how the prediction models hold up against different judges (e.g., GPT-4o, Claude 3.5 Sonnet, or even a human panel). Would a model fine-tuned to predict a Llama-judge also be able to predict a GPT-judge?

- The SFT approach is clear, but it would be helpful to have a more detailed discussion on the cost of creating this fine-tuning dataset. It requires generating responses from all models and then running the expensive judge model on them. A brief analysis of this "setup cost" would make the SFT method's practicality even clearer.

- The full, practical implementation of the deferral system isn't explored. It would be fantastic if you could add experiments showing the actual end-to-end performance. For example, a "small model + SFT predictor + large model" system vs. just using the large model, showing the blended cost-per-query and overall quality score.

- The system seems to imply a binary "answer or defer" choice based on "great/ok/bad." It would be interesting to explore a more granular system. For instance, could a predicted "ok" score trigger a simpler, cheaper intervention (like a RAG query) rather than a full deferral to the most expensive model?

-  For the in-context report card method, it would be great to see a more direct analysis of the token overhead vs. the accuracy gain. How many tokens does the report card add to the context, what's the added latency from that, and what is the "break-even" point where the time saved by not generating a bad answer equals the time spent processing the report card?

- There are red "REDACTED" comments at the end of sections like acknowledgement or authors contribution

### Questions
- Your judge rubric is quite general. Did you experiment with how prediction accuracy changes if the rubric is made more specific or complex? For instance, if the judge was asked to only score for "factual accuracy" and ignore tone.

- Beyond just prediction accuracy (correctly guessing "great," "ok," or "bad"), did you look at the model's calibration? For example, when the SFT model predicts "bad" with high confidence, is it almost always correct?

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
This paper explores whether LLMs can anticipate how an external LLM-based judge would evaluate their answers before actually generating them. The authors investigate three strategies: (1) zero-shot prediction, (2) in-context “report card” prompting that summarizes past performance, and (3) supervised fine-tuning using the hindsight trick. Results show that larger reasoning models display reasonable self-assessment ability even in zero-shot settings, while smaller models benefit significantly from contextual report cards or fine-tuning.

### Strengths
1. The paper addresses how to make LLMs self-aware enough to know when to ask for help. This is both conceptually interesting and relevant for efficient LLM deployment.

2. The proposed approaches are well-motivated and systematically compared across diverse datasets and model sizes.

3. The authors provide some detailed empirical results and ablations (e.g., per-category analysis on MMLU-Pro) that reinforce their claims.

### Weaknesses
1. All experiments rely on a single LLM judge (Llama-70B). This raises questions about generalization to different evaluators or judging paradigms.

2. Since both judge and fine-tuning signals ultimately depend on LLM-generated labels, there is no ground truth accuracy. Also, the number of classification types is small (only 3), making the classification problem seem easy.

3. The report cards are generated on a training set, and then given to the testing set as part of the prompt. As this paper considers the i.i.d. case testing, this may give too much shortcut for the problem, making it hard to tell how much improvement is coming aside from just aligning the distribution of the test and training set.

### Questions
See Weaknesses.

### Soundness
3

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
The paper introduces a pre-self-assessment mechanism that enables large language models (LLMs) to predict, prior to generation, how an LLM-based judge would evaluate their responses. It explores three strategies—zero-shot, in-context report cards, and supervised fine-tuning—and demonstrates that smaller models can be effectively calibrated to route queries to larger models only when needed, achieving efficient inference under resource constraints.

### Strengths
1. Methodological novelty

The report card approach is innovative. It consolidates joint judge evaluations across multiple models and datasets into concise textual performance summaries, eliminating the need for expensive per-query judge calls. By combining hindsight relabeling with supervised fine-tuning (SFT), the method effectively repurposes existing judge scores as supervision signals. An ablation study (Appendix E) comparing joint versus isolated judging further shows that joint evaluation enhances score diversity.

2. Broad & reproducible evaluation

The study presents extensive experiments across five diverse datasets (MedQA, LongFact, AIME’24, SciCode, MMLU-Pro) and eleven models ranging from 0.9B to 120B parameters, including recent reasoning architectures such as Llama-4 Scout and DeepSeek-R1. The release of complete prompts (Appendix C) and the detailed judge rubric reinforces the work’s reproducibility and transparency.

3. Strong empirical findings

Fine-tuning yields substantial improvements, with the some model achieving a +52 percentage point gain in prediction accuracy. Even large non-reasoning models benefit from contextual report cards. Notably, prediction accuracy increases with query difficulty, suggesting that task complexity itself can serve as an informative signal for adaptive model routing.

### Weaknesses
1. Reliance on a Single Judge Model

The study depends exclusively on a single LLM judge (Llama 3.3 70B) for all evaluations, raising concerns about evaluation bias and potential overfitting to one model’s judgment criteria. Without comparisons across multiple judges or human evaluations, the generality and robustness of the proposed approach remain uncertain.

2. Methodological Simplicity and Incomplete Framework

The three proposed methods—zero-shot probability prediction, contextual report card prompting, and supervised fine-tuning—are conceptually straightforward, relying on techniques such as prompt engineering and fine-tuning that have been widely explored in prior work. Moreover, the paper does not clearly specify how queries predicted as “bad” are routed to larger models, leaving the proposed self-assessment-based routing framework incomplete for real-world deployment.

3. Unrealistic Assumption of Report Card Availability

The report card method assumes access to detailed historical performance summaries for each model across multiple datasets. In practical settings, such comprehensive records are rarely available, particularly for unseen data. This assumption limits the method’s applicability and generalization potential.

4. Lack of Baselines and Comparative Analysis

The paper omits key baselines such as uncertainty modeling and self-evaluation approaches. Without these comparisons, it is difficult to assess the relative improvement, effectiveness, or novelty of the proposed techniques.

### Questions
1. The paper discusses agent in both PRELIMINARIES and RELATED WORK, but the main methods and contents of the paper do not seem to involve agent?

2. The PRELIMINARIES section contains some redundant details. The description of LLM architectures is not directly relevant to the paper’s central research question—predicting LLM judge scoring. It may be beneficial to simplify this section and focus only on the components essential to understanding the proposed methods.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates whether language models can predict LLM judge scores before generating responses, enabling efficient routing where small models handle easy queries and defer hard ones to larger models. This is practically relevant for 2025's on-device AI deployment trends.
The study tests three approaches across 11 models and 5 datasets: zero-shot prediction, report cards with historical performance, and fine-tuning. Key finding: reasoning models show inherent self-awareness while small models achieve up to 52% improvement with fine-tuning.

### Strengths
1. Timely practical problem addressing real deployment challenges with comprehensive experiments across medical, mathematical, coding, and factual domains.
2. Strong empirical findings demonstrating that small models can learn their limitations, providing actionable baselines for production systems.
3. Well-documented reproducible methodology with extensive ablations.

### Weaknesses
1. Zero technical innovation: zero-shot is basic prompting, report cards are standard in-context learning, fine-tuning is vanilla supervised learning from 2022.
2. All tasks have objective answers (multiple choice, math solutions, code correctness). No subjective tasks like creative writing or advice-giving where evaluation is ambiguous.
3. Single judge (Llama 3.3 70B) creates uncertainty whether models learn true self-awareness or just memorize one judge's preferences.

Critical Questions

- For LongFact success: what drives correct predictions? Is it response length estimation, keyword matching, or topic familiarity? Need 10-20 concrete examples showing why model correctly predicted "great" versus "bad" with feature analysis.
- For AIME math problems: how do models assess difficulty? Is prediction based on problem length, mathematical terminology, or actual computational complexity? Requires stratified analysis by difficulty level showing models correctly identify when they fail on olympiad problems but succeed on algebra.
- Which evaluation criteria drive predictions? The rubric includes accuracy, relevance, clarity, formatting. Do models fail when answers are correct but poorly formatted? When verbose but accurate? Need ablation isolating each criterion's influence.

### Questions
- RAG integration: does retrieval-augmented generation improve prediction accuracy on knowledge-intensive queries by providing reference context?
- Federated learning: can distributed edge devices collaboratively build report cards without sharing raw data, learning when to defer to cloud models based on collective experience?
- Cross-judge generalization: train on Judge A, test on Judge B and human ratings to distinguish true self-awareness from judge-specific overfitting.

### Soundness
3

### Presentation
3

### Contribution
3
