# LogiNumSynth: Synthesizing Joint Logical-Numerical Reasoning Problems for Language Models

- Decision: Reject
- Scores: 0, 6, 2

## Abstract
Joint logical-numerical reasoning remains a major challenge for language models, yet existing datasets rely on fixed rule sets and offer limited control over task complexity, constraining their generalizability for evaluation and training. We present LogiNumSynth, a flexible natural language problem synthesizer that synthesizes tasks requiring proficiency in joint logical reasoning (e.g., rule-based reasoning) and numerical reasoning (e.g., arithmetic computation). LogiNumSynth supports fine-grained control over reasoning world richness, logical reasoning depth, and the complexity of numerical computations, enabling flexible data synthesis across difficulty levels. We demonstrate three key contributions: (1) Synthesizer---synthesizing fully controllable joint reasoning tasks over natural language; (2) Evaluation & Process Analysis---evaluating both process accuracy and answer accuracy; (3) Targeted Training---using synthesized data to enhance LLMs' reasoning performance. Experiments with multiple LLMs highlight persistent weaknesses in logical-numerical reasoning, showing that LogiNumSynth can serve as both a diagnostic tool and a source of targeted supervision for advancing integrated reasoning skills.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper introduces LogiNumSynth, a synthetic data generator designed to create logical and numerical reasoning tasks for language models. Its main goal is to bridge the gap between these two types of reasoning. The framework offers fine-grained control over logical depth and numerical complexity, allowing the generation of tasks across multiple difficulty levels.

The authors present a dataset composed of 4+1 subsets covering easy/hard logical and easy/hard numerical reasoning, along with an additional “extra hard” subset that combines both. They evaluate language models trained on LogiNumSynth to measure their logical and numerical reasoning capabilities. Furthermore, they train models on both downstream benchmarks and their own dataset, demonstrating improved performance on the downstream benchmarks they have trained on.

### Strengths
* **Relevance and Motivation:** The focus on combining logical and numerical reasoning is well-motivated. It tackles a structured synthesis approach that could help evaluate or improve models’ reasoning abilities.
* **Control over Task Complexity:** Allowing adjustable difficulty and reasoning depth is a valuable feature for both evaluation and curriculum-based training.
* **Process and Answer Evaluation:** Assessing reasoning steps, not just final outputs, aligns with current trends in verifiable reasoning.

### Weaknesses
* **Lack of Clarity in Training Setup:**
  The description of the training and evaluation procedure is inconsistent. While the paper claims *out-of-domain evaluation* after fine-tuning on LogiNumSynth, Appendix F.3 suggests that the models were directly trained on downstream benchmarks (e.g., GSM8K, LogiQA). This critical detail is omitted from the main text, giving the impression that improvements stem from LogiNumSynth training, when in fact they may result from training on the downstream datasets themselves. Table 5 reinforces this concern. The current wording misleads readers about the dataset’s generalization effects and must be corrected. 

If this interpretation is incorrect, I invite the authors to clarify. However, if my understanding is accurate, the current presentation seriously misleads readers. This is a major concern and I therefore must strongly recommend rejection unless this issue is convincingly addressed.

* **Unclear Novelty Claim:**
  The claim of being the *first dataset* to integrate logical and numerical reasoning is questionable. Numerical operators are already intrinsic to many logical reasoning tasks [1, 2, 3]. The authors should explicitly compare LogiNumSynth to prior datasets that also integrate logical and numerical reasoning.

* **Limited Training Scope:**
  The paper does not evaluate whether fine-tuning on LogiNumSynth improves or maintains performance on downstream reasoning benchmarks not included at training timesuch as ProntoQA, CLUTRR, or SLR. Such experiments would be essential to demonstrate the dataset’s real-world usefulness beyond its synthetic scope.

* **Missing Rationale for Separate Training:**
  The authors train logical and numerical subsets separately, despite positioning their work as an integration of the two. A joint training setup would be a more natural test of the dataset’s stated purpose and could reveal cross-domain reasoning benefits.

* **Insufficient Explanation of Data Quality:**
  The paper provides little evidence regarding the semantic coherence or plausibility of the synthesized samples. If tasks are generated purely through random sampling, they risk being logically inconsistent or meaningless. Including representative examples and a qualitative analysis of data quality would strengthen the paper’s credibility.

---

**References**

[1] Helff et al., *SLR: Automated Synthesis for Scalable Logical Reasoning*

[2] Ma et al., *KOR-Bench: Benchmarking Language Models on Knowledge-Orthogonal Reasoning Tasks*

[3] Liu et al., *SynLogic: Synthesizing Verifiable Reasoning Data at Scale for Learning Logical Reasoning and Beyond*

### Questions
## Revise Empirical results
* Clarify in the main body that the model is also fine-tuned on the downstream dataset.
* Just train your model only on the LogiNumSynth train set (EN-train+EL-train) and check for downstream improvements on the out-of-domain datasets
* If your domain scope is to narrow and the model overfits, you can compare your extensive training run with a model trained only on the downstream benchmarks (excluding LogiNumSynth).
* Also evaluate on MMLU and see whether you can maintain performance or the degradation is not too large, and on inductive logical reasoning to see whether you generalize OOD (e.g., SLR-Bench or CLUTRR)?
* You might also want to compare your tuned model to other logic-tuned models (e.g. 'AIML-TUDA/Llama-3.1-8B-SLR'). If you do so, remeber to also evaluate on OOD dataset, both models have not been trained on.
* Also, add how well the tuned model performs on the test set of LogiNumSynth.

## Add clarifications
* How is the overlap between train/test and within the dataset prevented?
* Specify number of samples of LogiNumSynth datasets in the main body of the paper?
* How is correctness of a solution verified? exact match?
* Also report the selected “reasoning effort” parameter for GPT-5-mini?
* Have you considered using RL-based reasoning training (e.g., GRPO, GSPO, ...) on LogiNumSynth?

While I remain unconvinced by some technical aspects, I find the general idea promising and the problem important. If the authors can address my concerns, I would be open to reconsidering my score.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces LogiNumSynth, a synthesizer for creating natural language tasks that require combined logical and numerical reasoning, with fine-grained control over difficulty. The tool is used to evaluate and diagnose specific weaknesses in large language models' joint reasoning capabilities. Furthermore, the synthesized data is shown to be effective for targeted training, improving the models' performance on these challenging reasoning tasks.

### Strengths
It introduces LogiNumSynth, a novel tool that can generate a wide range of joint logical-numerical reasoning problems with fine-grained control over complexity, moving beyond fixed and limited existing datasets.

The framework is designed not just for evaluation but also for targeted training, demonstrating its value as both a diagnostic tool to identify model weaknesses and a means to improve them.

This paper presents a large-scale evaluation of 29 models, demonstrating that training with synthetically generated data enhances model performance on external reasoning benchmarks.

### Weaknesses
The performance improvement from SFT on the synthetic data appears to be marginal (approximately 1%).

Experimental results for SFT on larger-scale models (>=7B) are lacking.

How was the correctness and quality of the synthetic data validated? It is recommended to conduct a small-scale human evaluation.

How diverse is the synthetic data? Is there a trade-off between the diversity of the data and its correctness?

### Questions
See the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces LogiNumSynth, a natural language problem synthesizer for logical and numerical reasoning tasks. It demonstrates its usage through the creation of four datasets, based on different combinations of both easy and hard numerical and logical tasks, which are then evaluated on several LLM. It is shown that fine-tuning LLM on the synthetic data can improve its performance on several reasoning benchmarks.

### Strengths
Overall, the paper is clearly written and easy to follow. The experiments and analyses appear to be fully documented.

### Weaknesses
Unfortunately, the paper lacks innovative ideas, impactful contributions, and analytical depth. It is unclear what purpose the proposed method and analysis can serve meaningfully.

1.  The abstract states that LogiNumSynth should serve as a diagnostic tool and a source of datasets for fine-tuning, but it provides no explanation or insight into why this should significantly improve LLM performance or reduce systematic wrong behavior. The slight to moderate performance gains could be attributed to fine-tuning on a somewhat related domain with respect to the reasoning benchmarks.
2.  The diagnostic possibilities are limited to CoT (process) and answer evaluation on logical and numerical tasks of varying complexity, with no theory on how to explain particular good or bad performance or an indication of how to categorize or assess different failure modes. 
3.  There is also no theoretical discussion on why current LLM struggle with numerical and logical reasoning or on the possible limitations of fine-tuning for solving these problems. This makes it difficult to place this work in a scientific context, which is necessary for an impactful publication.

### Questions
While this paper, with its proposed methods, is clearly scientifically inspired, it at least needs a discussion of its implicit assumptions, expectations, and potential limitations to function as a stand-alone conference publication.

### Soundness
2

### Presentation
2

### Contribution
1
