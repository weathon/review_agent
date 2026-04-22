# Benchmarking Uncertainty Estimation in Large Language Model Replies for Natural Science Question Answering

- Avg Score: 3.00
- Decision: Reject
- Scores: 4, 2, 2, 4

## Abstract
Large Language Models (LLMs) are commonly used in Question Answering (QA) settings, increasingly in the natural sciences if not science at large. Reliable Uncertainty Quantification (UQ) is critical for the trustworthy uptake of generated answers. Existing UQ approaches remain weakly validated in scientific QA, a domain relying on  fact-retrieval and reasoning capabilities. We introduce the first large-scale benchmark for evaluating UQ metrics in scientific QA studying calibration of UQ methods, providing an extensible open-source framework to reproducibly assess calibration. Our study spans up to 20 large language models of base, instruction-tuned and reasoning variants. Our analysis covers seven scientific QA datasets, including both multiple-choice and arithmetic question answering tasks. We evaluate and compare methods representative of prominent approaches on a total of 685,000 long-form responses, spanning different reasoning complexities representative of domain-specific tasks.
At the token level, we find that instruction tuning induces strong probability mass polarization, reducing the reliability of token-level confidences as estimates of uncertainty. Models further fine-tuned for reasoning are exposed to the same effect, but the reasoning process appears to mitigate it  depending on the provider. At the sequence level, we show that verbalized approaches are systematically biased and poorly correlated with correctness, while answer frequency (consistency across samples) yields the most reliable calibration. In the wake of our analysis, we study and report the misleading effect of relying exclusively on ECE as a sole measure for judging performance of UQ methods on benchmark datasets.
Our findings expose critical limitations of current UQ methods for LLMs and standard practices in benchmarking thereof.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a large-scale benchmark framework for evaluating uncertainty quantification (UQ) methods in scientific question answering (QA) with large language models (LLMs). The authors compare multiple token-level and sequence-level UQ methods (e.g., Verbalized Uncertainty, P(True), Frequency-of-Answer, and Claim-Conditioned Probability) across various open-source models and datasets. They also discuss the calibration behavior and reveal how instruction tuning or reasoning-style prompting may lead to probability polarization, thus weakening token-level confidence as a reliable UQ signal. Overall, the topic is important and timely, and the paper provides a broad and systematic experimental analysis.

### Strengths
1. Uncertainty quantification for LLM-based QA systems is highly relevant, particularly for scientific and high-stakes domains.
2. The benchmark spans multiple datasets and model families, including base, instruct, and reasoning variants, which allows a broad comparison of model calibration behaviors.
3. The identification of probability polarization after instruction-tuning and the empirical comparison of sequence-level methods (e.g., Frequency-of-Answer vs. CCP) provide useful insights for practitioners.
4. The paper emphasizes open-sourcing the benchmark and results, which can facilitate follow-up research and reproducibility within the community.

### Weaknesses
1.	Some core observations (e.g., why Mistral models behave differently or why GPT-based verbalized uncertainty performs better) are descriptive but not analytically explained. Additional ablation or controlled studies are needed to support these interpretations.
2.	The benchmark relies on prompts that maximize label-token probability, which introduces potential bias. The stability of results across different prompts, temperatures, or decoding settings is not sufficiently reported.
3.	The paper mentions that CCP scores often vanish due to sequence aggregation and NLI errors, but provides no mitigation or alternative aggregation strategies, limiting the method’s generalizability.
4.	Frequency-of-Answer is reported as the most reliable but computationally expensive method. The paper would benefit from a concrete compute/time/cost table to assess real-world feasibility.
5.	Many plots and tables lack statistical testing (e.g., bootstrapped CIs or significance analysis), making it unclear whether observed performance differences are robust.
6.	As a benchmark-oriented paper, it is important to clarify what the research community can concretely gain from this work beyond the empirical comparison. For instance, what types of new studies, analyses, or extensions could future researchers perform using the provided data or benchmark framework? Expanding this discussion would better articulate the lasting value of the benchmark to the community.

### Questions
1.	Could you include a cost analysis for the Frequency-of-Answer method (e.g., average sampling count, GPU-hours, runtime per instance)?
2.	Beyond performance comparison, what future research directions or applications does this benchmark enable for the community (e.g., evaluating new UQ techniques, cross-modal extensions, safety assessments)?
3.	How stable are the results under different prompts, temperatures, or decoding parameters? Would the key findings (e.g., performance ranking, polarization) remain consistent?
4.	Ensure that all abbreviations use capital initials and are defined upon first use, e.g., Large Language Models (LLMs), Question Answering (QA), etc.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents an empirical benchmark evaluating various uncertainty quantification methods for LLMs' responses in natural science question answering. The authors conduct experiments across 11 LLM models and 8 scientific QA datasets. They identify several issues, including pronounced polarization of token probabilities and systematic biases in verbalized uncertainty, and find that answer frequency offers the most reliable calibration, despite its high computational cost.

### Strengths
- The paper's motivation to investigate uncertainty quantification specifically for scientific question answering is relevant.
- The authors attempt to evaluate a range of intrinsic and extrinsic uncertainty quantification methods, providing an empirical comparison across different approaches.
- This paper evaluates 11 models across 8 datasets spanning multiple task formats (MCQA and arithmetic QA), providing an empirical investigation that includes base, instruction-tuned, and reasoning model variants.

### Weaknesses
- While the authors claim this is the "first large-scale benchmark" for UQ in scientific QA, LM-Polygraph [1] already provides a more comprehensive evaluation framework with 28 UQ methods. Many datasets used here (MMLU, ARC, GSM8K, SVAMP) are already standard in existing UQ literature, undermining the claimed distinctiveness of the scientific QA focus.
- Despite emphasizing "Natural Science Question Answering" in the title, the paper's main findings, "probability polarization and verbalized uncertainty bias," can apply equally to general QA settings, with no meaningful analysis of what makes scientific reasoning unique.
- I found some critical issues were ignored in the main text. For instance, Table A.5 reveals that numerous models produce massive invalid answers under specific prompts (e.g., Magistral-Small-Reasoning has 18,716/25,316 invalid responses for Prompt 2). However, the main paper fails to acknowledge or discuss this severe reliability issue that fundamentally undermines the evaluation.
- The exclusion logic for some specific UQ methods appears arbitrary. Semantic Entropy is rejected for being "claim-level" despite its proven effectiveness across settings. At the same time, all ensemble/density-based methods are dismissed for "computational cost," even though Frequency of Answer (requiring 10 samples per prompt) is included, revealing a lack of principled selection criteria.
- About the probability polarization phenomenon, while the paper documents that instruction tuning induces strong probability polarization (evident in Figure 1), it remains a purely descriptive observation without investigating the underlying causes (training objectives? data distributions?), potential mitigation strategies (temperature scaling? calibration methods?), or whether the effect varies systematically across question types.
- I think the most intriguing finding of the paper is "extreme polarization in reasoning models (Magistral, Qwen-Thinking, DeepSeek-R1), with Qwen3-30B-A3B-Thinking placing nearly all predictions in the highest confidence bin". However, it receives only surface-level discussion, with no analysis of how reasoning chains affect uncertainty or whether better uncertainty estimates could be extracted from the reasoning process itself, as discussed in a set of recent works [2-6].
- Questionable Dataset Selection. The claim that physics serves as a "representative natural science domain" is unsubstantiated since "chemistry/biology" may exhibit different uncertainty patterns. Most datasets actually contain multi-disciplinary content rather than pure physics. GPQA's 448 samples are insufficient for a "large-scale" benchmark, and the exclusive focus on verifiable formats (MCQA/arithmetic) excludes open-ended scientific QA and genuine proof-based reasoning tasks, severely limiting generalizability.

[1] Fadeeva et.al, LM-Polygraph: Uncertainty Estimation for Language Models, EMNLP 2023.

[2] Mei et.al, Reasoning about Uncertainty: Do Reasoning Models Know When They Don't Know, arXiv 2025.

[3] Zhang et.al, CoT-UQ: Improving Response-wise Uncertainty Quantification in LLMs with Chain-of-Thought, ACL 2025.

[4] Becker et.al, Cycles of thought: Measuring llm confidence through stable explanations, arXiv 2024.

[5] Da et.al, Understanding the uncertainty of llm explanations: A perspective based on reasoning topology, COLM 2025.

[6] Mo et.al, Tree of uncertain thoughts reasoning for large language models, ICASSP 2024.

### Questions
Please see the weaknesses I've outlined.

### Soundness
2

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
The authors construct a benchmark of science QA datasets, and evaluate several uncertainty quantification methods with large models. They analyze confidence for various types of models (such as reasoning or instruction-tuned models) and find polarization in token probabilities in reasoning and instruction-tuned models.

### Strengths
1. I have not previously seen an analysis of polarization for reasoning models separately from instruction-tuned models.
2. The paper is well-written and organized.

### Weaknesses
1. This work’s main weakness is its lack of novelty. Fundamentally, the benchmark consists of already-published datasets, there are no new methods examined, and most of the conclusions drawn by the authors have already been mentioned in previous literature—for instance, as they point out, there have been previous papers which find that instruction-tuned models are poorly calibrated.
2. I don’t agree with the assertion that “The standard evaluation technique for UQ methods are calibration plots”—there are many methods of evaluating UQ methods, including calibration plots, but also including ECE, AUROC, PRR, etc. Calibration plots additionally have several limitations—binning strategies, number of bins, etc all can majorly affect the way they look, and their interpretation is inherently qualitative which can be difficult to compare with many different configurations. This can be seen in figure A.1, where in order to fit all configurations on the screen the text and labels are illegibly tiny. With this in mind, I also disagree with listing use of summary statistics as a limitation of previous benchmarks, as these are widely used in the field for evaluation.
3. This benchmark examines many fewer UQ methods than previous work, and many are discarded because they produce unnormalized scores. In these, metrics such as AUROC could still be used for evaluation. 
4. One limitation the paper claims to be addressing is narrowness of domain, but scientific QA is arguably equally narrow to most of the factual QA datasets that these UQ methods have been evaluated on already.

### Questions
I’m curious how increasing temperature would change the distribution for the reasoning models—is there any setting at which the polarization begins to approach that of instruction-tuned models, or does increasing temperature degenerate performance too fast?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces an open-source benchmark to evaluate UQ for LLMs on QA tasks, spanning 11 base, instruction-tuned, and reasoning models across 8 datasets. It finds that instruction tuning polarizes token probabilities, making token-level confidences poorly calibrated, while at the sequence level verbalized self-reported confidence is biased and weakly correlated with correctness. In contrast, self consistency across sampled generations is the more reliable but computationally expensive signal.

### Strengths
- Establishes a large-scale benchmark focused specifically on uncertainty estimation for QA, spanning 11 LLM variants and 8 datasets. 
- Provides clear empirical insights: instruction tuning polarizes token probabilities and consistency via answer frequency is a stronger sequence-level signal.
- Covers diverse difficulty levels and domains to stress-test UQ under both factual recall and multi-step reasoning.
- Open-sources a repo for reproducibility.

### Weaknesses
- I personally find the focus on multi-choice QA unreasonable and unrealistic. Most of the time in practice LLMs are hardly applied to some questions with a pre-defined answer set. Therefore evaluating the model uncertainties under this limited scenario does not reflect their behavior on open-ended QA and bears minimal practical impact.

- **Most of the observations, unfortunately, are defined and discussed (and addressed to some extend) more rigorously in a not-so-recent paper [1]**. The previous work also addressed the accuracy issue while calculating ECE by balancing the correct/incorrect answers. The contribution and novelty of this paper is greatly undermined given this previous work.

- Missing related works, e.g., [2].

- This paper focuses on scientific QA, but none of the discussion is specific the the special characteristics of the scientific domain (e.g., how the questions and answers differ from the general domain?)

[1] Li Y, Qiang R, Moukheiber L, Zhang C. Language model uncertainty quantification with attention chain. arXiv preprint arXiv:2503.19168. 2025 Mar 24.  
[2] Wang X, Zhang Z, Chen G, Li Q, Luo B, Han Z, Wang H, Li Z, Gao H, Hu M. Ubench: Benchmarking uncertainty in large language models with multiple choice questions. InFindings of the Association for Computational Linguistics: ACL 2025 2025 Jul (pp. 8076-8107).

### Questions
What findings are specific to the scientific domain?

### Soundness
2

### Presentation
3

### Contribution
2
