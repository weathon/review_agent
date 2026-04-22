# Analyzing and Evaluating Unbiased Language Model Watermark

- Avg Score: 4.00
- Decision: Accept (Poster)
- Scores: 4, 2, 6, 4

## Abstract
Verifying the authenticity of AI-generated text has become increasingly important with the rapid advancement of large language models, and unbiased watermarking has emerged as a promising approach due to its ability to preserve output distribution without degrading quality. However, recent work reveals that unbiased watermarks can accumulate distributional bias over multiple generations and that existing robustness evaluations are inconsistent across studies. To address these issues, we introduce UWBench, the first open-source benchmark dedicated to the principled evaluation of unbiased watermarking methods. Our framework combines theoretical and empirical contributions: we propose a statistical metric to quantify multi-batch distribution shift, prove an impossibility result showing that no unbiased watermark can perfectly preserve the distribution under infinite queries, and develop a formal analysis of robustness against token-level modification attacks. Complementing this theory, we establish a three-axis evaluation protocol—unbiasedness, detectability, and robustness—and show that token modification attacks provide more stable robustness assessments than paraphrasing-based methods. Together, UWBench offers the community a standardized and reproducible platform for advancing the design and evaluation of unbiased watermarking algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This is a benchmark paper that evaluates different unbiased watermarking methods in large language models. Primally, the authors evaluate the detectability, unbiasedness, and robustness of different unbiased watermarking methods.

### Strengths
1. This paper introduces UWBench, a systematic evaluation of unbiased watermarking methods over the three-axis protocol: unbiasedness, detectability, and robustness.
2. The authors provide a better procedure to evaluate unbiasedness and robustness for unbiased watermarking methods, which I agree with.

### Weaknesses
1. The authors may miss some unbiased watermark baselines in the evaluation, e.g., [1], although this work is mentioned in the introduction section.  
2. I believe the main issue with this paper is the lack of motivation to provide a benchmark work for unbiased watermarking methods. The authors should better justify why such a benchmark is necessary and what specific gaps it aims to fill in the existing literature. Without a clear motivation, the contribution appears limited.
3. A conceptual evaluation of different baselines could also be provided. For example, what are the drawbacks of each method? The authors could also offer guidelines for designing future unbiased watermarks. 
4. The authors claim that UWBench is an open-source benchmark. I would appreciate it if the authors could provide the anonymized codes in their supplementary material.  

[1] Christ, M., Gunn, S., & Zamir, O. (2024, June). Undetectable watermarks for language models. In The Thirty Seventh Annual Conference on Learning Theory (pp. 1125-1139). PMLR.

### Questions
Please see weaknesses.

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
The paper proposed an evaluation framework for watermarking methods. The paper focuses on unbiasedness, detectability, and robustness. For unbiasedness, the paper proposed the Single-prompt multi-generation (SPMG) unbiasedness metric, which measures the difference of some metrics of generations from two models averaged across prompts and multiple generations of the same prompts. For robustness, the paper discussed the token effect region and the expected score decrease under one edit. In experiments, the weighted average of the true positive rate under different false positive rates and different perturbations is used. The proposed metrics are evaluated on several unbiased and a few biased watermarking methods.

### Strengths
Evaluation of watermarking methods is mostly ad-hoc, developing standard benchmarks is a timely topic. The discussion on the unbiasedness and robustness of watermarking methods makes sense.

### Weaknesses
1. The paper claims to introduce an open-source benchmark, but no source code is provided.

2. The contributions are not clear. In Section 5.1, there are a lot hyperparameters in the scores that seem to come from nowhere, e.g., the weights in the weighted average for the robustness score and detectability score. Why are these choices or these scores good? I would expect much more experiments to demonstrate that these scores are robust to the choice of hyperparameters, or give consistent evaluations for watermarks across different data sets, or number of prompts and length of generations.

3. The organization of the paper is confusing in some places, e.g., in section 4.2 token effect region and expected score decrease under one edit are discussed, but in the experiments, it seems that these concepts are not used, the score used for robustness are TPR under different FPR and perturbations. And the paragraphs "Green-count detectors" and "SynthID-style bit tests" seem to be examples for "expected score decrease under one edit". But they are all listed as if they were parallel concepts.

Some minor issues:

There are some typos, e.g., in line 465, I think it is Figure (1). In line 229, the form of $t_{\alpha}$ is inconsistent with the inequality above

Some citations are missing, e.g., for the DIPPER method in line 416, there are no citations.

### Questions
See weakness above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces UWBENCH, a benchmark and theoretical framework for evaluating unbiased watermarking methods in large language models.
The authors propose a single-prompt multi-generation (SPMG) metric to quantify distributional drift under repeated queries and define a variance-controlled detection statistic using McDiarmid’s inequality to derive finite-sample guarantees.
On the empirical side, they benchmark several watermarking schemes across three axes — unbiasedness, detectability, and robustness — and argue that token-modification attacks offer more stable robustness evaluation than paraphrasing.
Overall, the work aims to provide a reproducible, theoretically grounded evaluation suite for future watermark research.

### Strengths
### Strengths

1. Clear motivation and structure. The paper identifies a real gap — the lack of principled evaluation for unbiased watermarking — and builds both theoretical and empirical tools to address it.
2. Balanced theoretical and empirical contributions. The impossibility result (Theorem 4.1) and the finite-sample McDiarmid bound (Theorem 4.2) are neat results that add formal grounding to an otherwise empirical area.
3. Reproducible benchmark. Making UWBENCH open-source with standardized metrics is a strong community contribution.
4. Thoughtful experimental framing. The move away from paraphrasing attacks toward token-modification attacks as robustness tests is well-justified and supported by variance analysis.
5. Readable figures and tables. The scatterplot (Fig. 1) effectively summarizes the trade-offs across unbiasedness, detectability, and robustness.

### Weaknesses
### Weakness

- While I found the theoretical framing around McDiarmid’s inequality and the single-prompt multi-generation (SPMG) detection metric interesting, the use of the concentration bound feels a bit “black-boxed.” The paper could be clearer about the assumptions that allow McDiarmid to hold — namely independence across samples and bounded differences — rather than invoking it as a matter of fact. Clarifying these assumptions would help readers trust the finite-sample guarantees and better understand what breaks when they are violated.
  
- Beyond that, I think the work would benefit from being placed more explicitly in the growing literature studying watermark–utility trade-offs. For example, Downstream Trade-offs of a Family of Text Watermarks (Ajith et al., EMNLP Findings 2024) shows that even unbiased watermarks can hurt downstream task accuracy by 10–20 %, and WaterJudge: Quality-Detection Trade-off when Watermarking Large Language Models (Molenda et al., NAACL Findings 2024) quantifies how stronger watermarks improve detectability but reduce generation quality. Most relevant, Watermarking Degrades Alignment in Language Models (Verma et al., ICLR 2025) demonstrates that watermarking shifts alignment behaviors such as truthfulness, safety, and helpfulness, and mitigates this via an external reward model. Verma et al. (2025) analyze how Gumbel watermarking affects model alignment. They note that Gumbel watermarking produces the same output deterministically for a fixed seed, meaning that in your repeated-query setup the SPMG drift would be trivially zero by design rather than informative about unbiasedness. It might help to briefly acknowledge this point when interpreting your results on Gumbel watermarking. Verma et al. also show that watermarking tends to decrease reward-model scores, which could, in principle, serve as another lens for detection. If your benchmark already defines a bounded per-generation metric Met(⋅), it might be interesting to mention that a reward-based version of this metric could empirically capture the same drift your McDiarmid analysis predicts. Even a short comment or future-work note on whether these two detection views—statistical vs. reward-based—might align would make the discussion more rounded without expanding scope.

### Questions
- **Notation check:** Equation (2) likely intends  
  $$t_\alpha = A\sqrt{\tfrac{12\ln(1/\alpha)}{mn}},$$  
  not $A^2\sqrt{\cdot}$.  
- **Terminology:** Define “SPMG” and “Met” explicitly on first mention; readers outside the immediate subfield may not recognize these acronyms.  
- **Writing:** The proofs in Appendix B are clear and worth highlighting — consider moving a short intuitive summary of Theorem 4.2’s logic into the main text.  
- **Question:** Have the authors considered testing whether UWBENCH metrics correlate with alignment or reward-model score improvement as in Verma et al.? Even a small pilot experiment could reveal whether detectability improvements coincide with improved alignment or if they diverge.  


This is a well-executed and timely paper that provides both theoretical insights and practical tools for unbiased watermark evaluation. My main feedback is to clarify the assumptions behind the theoretical bound and to connect the benchmark more explicitly to alignment and downstream-utility studies. Adding or even briefly discussing a reward-model-based alignment ablation would substantially strengthen the paper’s impact.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces UWBENCH, an open-source benchmark for unbiased language‑model watermarks. The authors contribute (i) a single‑prompt multi‑generation (SPMG) metric and a calibrated statistic DetWmk to quantify drift that emerges when the same prompt is queried many times, (ii) an impossibility theorem showing that any detectable unbiased watermark cannot preserve the LM distribution under repeated queries with a fixed key, and (iii) a token‑level robustness analysis with a simple certificate based on effect regions. Empirically, the benchmark evaluates methods along unbiasedness, detectability, robustness across several LMs and tasks, and argues that there is a trade-off between (i) optimizing detectability and unbiasedness, (ii) robustness.

### Strengths
1. Theorems 4.1 is clean and establishes a fundamental limitation of detectable watermarks.
2. The evaluation covers multiple models/datasets and reports matched‑FPR operating points.
3. The three‑axis protocol is well‑specified for unbiased watermarking.

### Weaknesses
1. The proposed unbiasedness metric uses performance metrics as surrogate and might not guarantee Eq.(1). Thus it is misleading to call it unbiasedness. The benchmark lacks reasoning tasks where the accuracy is an important performance indicator.
2. The robustness framework is neat but may not capture common edits such as typo, translation attack, synonym substitution. Also Random token replacement lacks practicality and distort the output distribution.
3. Novelty is limited compared with existing results: similar arguments to Theorems 4.1 is already established in Christ and Gunn (2024) and the three-axis evaluation is common in MarkMyWords.

### Questions
1. If the watermark key is re‑sampled per generation (or per prompt), does SPMG drift vanish in expectation?
2. In practice, is the same key used multiple times for the same prompt? 
3. SPMG requires ∣Met(g)∣≤A; yet the text cites perplexity as an example, which is unbounded. Clarify how you bound or clip PPL (or switch to bounded NLL)?

### Soundness
2

### Presentation
3

### Contribution
2
