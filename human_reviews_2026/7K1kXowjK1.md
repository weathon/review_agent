# Self-Correction Bench: Uncovering and Addressing the Self-Correction Blind Spot in Large Language Models

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 4, 2, 2

## Abstract
Although large language models (LLMs) have transformed AI, they still make errors and follow unproductive reasoning paths. Self-correction is vital for safety-critical applications. However, we uncover a failure: LLMs can correct errors (by fixing external ones) but fail to activate this capability for identical internal errors - a limitation we term the Self-Correction Blind Spot. To study this, we introduce Self-Correction Bench, an evaluation framework that isolates self-correction behavior from knowledge limitations through controlled error injection. Testing 14 open-source non-reasoning models shows a 64.5% average blind spot rate. We show robustness in mathematical reasoning across complexities, and extend to closed-source models, non-mathematical domains, and on-policy errors. Causal evidence links this to training data: human demonstrations lack error-correction sequences, but fine-tuning with them reduces the blind spot. Appending a simple "Wait" prompt cuts blind spots by 89.3%, revealing latent capabilities. Our work exposes a training-induced limitation and provides practical fixes to boost LLM reliability in critical domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors target a known limitation in LLMs, self-correction, wherein it is harder to find errors in their own outputs, compared to finding similar / identical errors from external sources. They term this limitation Self-Correction Blind Spot, and introduce Self-Correction Bench. They argue that since human demonstrations rarely include self-correction, models only learn error-correction through outcome feedback during the RL phase. They also mention that self-correction is dormant in language models, and appending “Wait” improves the models ability to overcome blind spots.

### Strengths
- Studying self reflection is a very relevant problem for LLMs.
- The experiment of injecting incorrect responses in the output of the model and removing the stop token, and comparing it with the injection in the prompt is interesting.
- The results in Appendix C, indicating that the results don’t change with temperature is intriguing.
- Reducing the possibility for knowledge gaps by using easy, medium, and hard datasets is a good progressive difficulty metric. This isolates the self-correction capability from confounding factors.
- The author’s conclusion that reasoning models are better than non reasoning models on self correction is intuitive. Also the fact that correction markers can reduce the gap.
- Self-Correction Bench can be a useful benchmarks for identifying reflection capabilities in models.

### Weaknesses
- Even though the conclusions about corrective markers is interesting, the paper does not introduce any theoretical / technical contributions.
- The benchmarks are very limited, to SCLI5, GSM8K and MATH. It would be interesting to see this across different tasks like code, logic, and multimodal reasoning, etc.
- Overall, while this dataset has potential to be a useful benchmark for studying LLM reflection capabilities, the paper lacks strong technical backing and explanations, and is also very limited in the domains it is targeting.

### Questions
- It is unclear how authors are measuring Self Correction scores before and after committing to an answer. Is it using an LLM as a judge? More explanation is required here.

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
4

### Summary
The paper studies the self-correction capability of large-language models (LLMs) by: 1) creating a dataset of self-correction prompts, 2) showing empirically that some LLMs do poorly on self-correction prompts (referred to as a self-correction "blind spot" in the paper), and 3) showing empirically that appending the self-correction prompts with certain conditioning tokens (such as the word "wait") can improve the self-correction performance of models that did poorly initially. A variety of non-reasoning and reasoning models were studied, across a number of model families (such as Qwen, Gemma, and DeepSeek). The self-correction prompts are all based on math problems, ranging from the trivial (SCLI5) to quite difficult.

### Strengths
1. Dataset contribution specifically focused on self-correction (or "correction", see weakness 1) by asking models to determine whether the given answer or reasoning in the prompt needs to be corrected, and indeed make the correction. This allows for the analysis in the paper to examine the correction ability of both reasoning and non-reasoning LLMs.

2. Empirical comparative analysis of the correction capability of reasoning and non-reasoning LLMs reveals that one strong differentiating factor that specifically makes reasoning models better in performance is their ability to recognize when an error is present (in a given answer or reasoning trace), and make the necessary correction. Whereas, as this paper shows, non-reasoning models cannot.

3. Empirical evidence for inserting conditioning tokens such as "Wait" that alerts a non-reasoning model to the fact that the given answer may not be correct can improve their correction performance, making up the gap to or even exceeding the performance of reasoning models. It is quite clear that even non-reasoning models can give the correct answer when they are made aware that correction is required.

### Weaknesses
1. The dataset does not strictly study self-correction. By the construction description, all 3 sub-datasets were generated by inserting wrong answers or reasoning traces into a given prompt from standard datasets (with possibly a short sequence of model output) using off-the-self closed-source models (such as GPT 4.1). Since the incorrect answers were not generated by the models-under-test (open-source models with fewer parameters), it is unknown how likely they are under each model's own sampling distributions, this is not self-correction (i.e., the model may not even sample the incorrect solution). A proper self-correction dataset would sample strictly on-policy from each model and filter for incorrect answers. 

The datasets are still valuable as benchmarks for correcting an off-policy incorrect answer or reasoning trace, but they do not study "self" correction.

2. No actionable conclusions for reasoning models. The main result and contribution of the paper is that non-reasoning models have a self-correction blindspot, which could be corrected by conditioning on certain tokens. Moreover, reasoning models have small or no self-correction blindspot. As reasoning models are the stronger of these two types at solving math problems, this weakens the significance of the key findings of the work (i.e., Why not just use reasoning models?).

### Questions
Please show more examples of the self-correction dataset, e.g., 1 for each subset, and for when there are reasoning steps.

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
4

### Summary
This paper studies LLMs' self-correction blind spot and introduces Self-Correction Bench, an evaluation framework to measure this phenomenon through controlled error injection.

### Strengths
1. For originality and significance, I think the newly introduced evaluation framework is a nice addition to existing works.

2. The paper is generally understandable, but I need more explanation/analysis described in the weaknesses and questions sections below.

### Weaknesses
1. The paper discusses self-correction blind spot on LLMs, but closed-source LLMs are unfortunately not studied. Although it is explained as "close-source models lack support for fine-grained control of prefix inject critical for our methodology" in line 236, I do not think this would stop you from studying closed-source models. You may want to analyze their reasoning chains directly and compare model outputs side-by-side. Otherwise, the findings of this paper is very limited.

2. The analysis offered appears to be coarse and is not very comprehensive. You mentioned that "this limitation may be influenced by training data", but I feel there can be much more reasons. For example, an LLM may tend to have relatively lower uncertainty on its own tokens than on tokens generated by another model.

3. I do not see any quality check on the benchmark introduced by the authors in Section 4, so I am not very confident with its overall quality. A common practice of similar papers is to leverage human evaluation to do some manual verification.

4.  A “Wait” prompt is not a novel idea. For example, [1] explores an ensemble of critics and the model's own feedback. This is a 2-year-old work. You may find more papers talking about something similar to your prompt.

[1] "N-Critics: Self-Refinement of Large Language Models with Ensemble of Critics"

### Questions
1. Is it possible to design some customized methods to analyze closed-source LLMs?

2. Do you think there are any other reasons of self-correction blind spot besides training data?

### Soundness
1

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
The paper focuses on isolating a failure in modern LLMs called "Self-correction Blind Spot", where the model can correct an error if attributed to an external factor (e.g. a user or a tool), but fails to do so for itself. It does so by quantifying the phenomena on 3 benchmarks of increasing complexity (SCLI5, GSM8K-SC, and PRM800K-SC).

It then points out that the lack of correction is due to a lack of activation (or intent to fix), rather than the inability to fix (e.g. not having the knowledge). Minimal intervention is design succesfully, namely by using the word "Wait."

### Strengths
There are several strengths to the manuscript:
1. Rather clear isolation of the problem and extensive empirical analysis of multiple models on the behavior. The experiment design seems reasonable, where exact same sequence of tokens is presented but with different attribution to observe differing model behavior.
2. The introduction of a toy benchmark (SCLI5) to explain clearly the phenomena, followed by studying on real-world benchmarks.
3. Identifying an intervention, e.g. the word "wait".

### Weaknesses
I believe the overall contribution is somewhat naive and does not go sufficiently in-depth in understanding the mechanics of the model behavior. The weaknesses I would like to hear the author's opinion are:

1. The entire work is prompt engineering - from the detection to the solution. The authors acknowledge this, but I find it necessary to go a step further and point out some training recipe changes that mitigates this to some degree. Naive fine-tuning with the word 'wait' may not work, while asking all researchers / developers to start using 'wait' seems also non-ideal.

2. An off-policy (in the RL sense) / biased setup by design. The generation of the high-quality incorrect reasoning traces by another model and then feeding them into a current model making them looking like it's own output is somewhat flawed by design. The generated tokens would have to be from the model itself (e.g. it's weights) or otherwise the sequence would always be off-policy of the model state (and hence can't be used for claiming the blind spot).

### Questions
Please refer to weaknesses. 

Furthermore, can the code be released? Unfortunately the repository at 4openscience is returning an error for all files, despite some directory structure existing.

### Soundness
3

### Presentation
3

### Contribution
2
