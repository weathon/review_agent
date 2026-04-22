# MetaLint: Generalizable Idiomatic Code Quality Analysis Through Instruction-Following and Easy-to-Hard Generalization

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Large Language Models, though successful in code generation, struggle with code quality analysis because they are limited by static training data and can’t easily adapt to evolving best practices. We introduce MetaLint, an instruction-following framework that formulates code quality analysis as the task of detecting and fixing problematic semantic code fragments or code idioms based on high-level specifications. Unlike conventional approaches that train models on static code quality conventions, MetaLint employs instruction tuning on synthetic linter-generated data with dynamic conventions to support easy-to-hard generalization, enabling models to adapt to novel or complex code patterns without retraining. 
To evaluate this, we construct a benchmark of challenging idioms inspired by real-world coding standards such as Python Enhancement Proposals (PEPs) 
and assess whether MetaLint-trained models reason adaptively or simply memorize. 
Our results show that MetaLint training improves generalization to unseen idioms. Qwen3-4B attains a 70.37% F-score on a manually curated and challenging PEP idiom detection benchmark, achieving the highest recall (70.43%) among all evaluated models. For localization, it reaches 26.73%, which is a strong outcome for its 4B parameter size and comparable to larger state-of-the-art models such as o3-mini, highlighting its potential for future-proof code quality analysis. Furthermore, MetaLint training enables generalization in idiom detection across model families, model scales, synthetic data from diverse linters, and Java idioms, demonstrating the general applicability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes METALINT, a training framework for generalizable code quality analysis. The core problem it addresses is the issue of outdated rules that arises when using large models for code quality checks, as these models cannot adapt to new rules. Inspired by the generalization capabilities of instruction tuning, the paper constructs data pairs consisting of rule descriptions, code, and the locations of rule violations. This approach guides the model to learn rules from their descriptions rather than relying on memory, thereby enabling it to  generalize to new rules.
The training process employs a two-stage SFT+DPO method, which standardizes the model's output while enhancing its generalization ability. Finally, the framework was tested on models of various scales, yielding mixed results. On the positive side, a small model achieved a violation detection recall that surpassed even top-tier large models. On the negative side, SFT training alone led to a performance decline, and all models performed very poorly in accurately localizing the line numbers of violations.

### Strengths
1. The paper convincingly articulates the limitations of current linter/static analysis tools and LLM-based code quality checkers, emphasizing the challenge of evolving/crossing idiom boundaries and the pitfalls of memorization.
2. The experiments confirm the method's effectiveness, though a gap remains when compared to top-tier general-purpose models.
3. The work is expected to have a notable impact in the field of code quality analysis.

### Weaknesses
1. The paper is missing the section “The Use of Large Language Models (LLMs)”.

2.While METALINT leverages existing linters (e.g., Ruff, PMD) as verifiers in preference optimization, this design partially contradicts its core motivation—moving beyond static rule-based systems. The reliance on linters for feedback might limit the semantic depth of preference learning, especially for complex idioms where linters are inherently insufficient. A stronger or human-aligned verification mechanism would strengthen the claim of adaptive, semantics-aware generalization.

3.While the paper claims to achieve “easy-to-hard generalization”, the proposed framework does not include any mechanism explicitly designed to enhance generalization. The observed transfer ability largely stems from the intrinsic properties of DPO, which is already known to improve alignment and robustness in instruction-tuned LLMs. In other words, the reported generalization reflects DPO’s inherent strengths rather than any algorithmic innovation introduced by METALINT itself.

4.The figures are too rough, posing obstacles to reading the paper. For example, the flow arrows in the verifiable reward diagram in Figure 1 are confusing. The rows and columns in the result tables are also misaligned, for example, Detection and Localization in Table 1.

5. $\mathcal{L}$ is usually regarded as a loss function and is not recommended for use in linters.

### Questions
1.Why does using CoT on the base model not cause a drop in recall, while using CoT on the model after the two-stage SFT-DPO training causes a sharp drop in recall?

2.What is the purpose of SFT? The experimental results show that adding SFT caused a decline in all three metrics. For a model that has already been instruction-tuned, is this kind of SFT redundant and does it damage the model's generalization ability?

### Soundness
3

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
4

### Summary
This paper presents an approach to fine-tuning LLMs to detect and localize programming idioms in programs. The basic idea is to construct a dataset using rule-based techniques to detect programming idioms in existing source code. Based on the dataset, SFT and RL are used to fine-tune LLMs to acquire the capability of detecting and localizing programming idioms in programs. The fine-tuned LLM (especially the LLM fine-tuned with RL) is superior to the base LLM, and even for harder benchmark, the superiority of the fine-tuned LLM over the base LLM exists.

### Strengths
1. The idea of obtaining a strong code quality analyzer via fine-tuning on a dataset of easier instances is interesting.
2. The empirical results indicate that SFT and RL are effective for training a code quality analyzer.

### Weaknesses
1. The current evidence may not support that the obtained LLM-based code quality analyzer is a stronger one. Since the base LLM is not specific to code quality analysis, the empirical superiority over the base LLM does not suffice to claim a stronger code quality analyzer. At least, the fine-tuned LLM-based code quality analyzer should be compared with the rule-based approaches that generate the data for fine-tuning.
2. There is lack of empirical comparison with existing SOTA LLM-based (or even deep-learning-based) code quality analyzer. Without such comparison, it is unclear whether the proposed scheme is a better one to obtain a strong code quality analyzer.
3. The novelty seems to be limited. Although the idea of training a analyzer to tackle harder idioms with simple idioms is interesting, this paper seems to rely solely on existing techniques to achieve this purpose.

### Questions
1. Can the trained code quality analyzer outperform the rule-based analyzers that generating the the training dataset?
2. How does the trained code quality analyzer compare with SOTA LLM-based code quality analyzers?

### Soundness
2

### Presentation
3

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
The paper’s core idea is to move “code quality” away from memorizing fixed lint rules and toward instruction-guided detection of code idioms: give the model a natural-language spec plus a few examples, and it flags good/bad idioms in source code and pinpoints the lines. To teach small models to generalize “from easy to hard”, the authors first use real linters (Python Ruff, Java PMD, with a dash of Tree-Sitter) to mass-produce easy idioms for SFT, then apply verifiable preference optimization to boost recall and localization; the reward is simply overlap with the linter’s labeled line set. In experiments on their Hard PEP Idioms benchmark, a Qwen3-4B METALINT model delivers strong detection/localization, especially recall, and even generalizes across model families and over to Java. In short, the contributions are: (1) METALINT, a framework that treats code quality as instruction-following over updatable idioms; (2) a tougher, more realistic evaluation set that goes beyond typical linter coverage; (3) a systematic study showing plain SFT tends to “memorize rules”, while adding DPO meaningfully improves recall and localization; and (4) robust generalization across model sizes, optional CoT, multiple languages, and multiple tools.

### Strengths
The paper’s most original contribution is to reframe code quality from fixed rule matching into instruction-conditioned “code idiom” detection, coupled with preference optimization using a verifiable line-level F1 reward. This design reduces the introduction of “new rules or best practices” to supplying a natural-language description plus a few examples, thereby breaking free from the rule engineering and coverage limitations of traditional linters. The authors also build cross-language (Python/Java), cross-model-family, and cross-scale training and evaluation setups to systematically validate the framework’s transfer and extensibility on both detection and localization; in particular, under easy to hard transfer to PEP-level fine-grained idioms, small models still achieve high recall and competitive localization, demonstrating practical potential and scientific value.

The paper’s quality and clarity are also strong: the sources and synthesis process of training data are reproducible; the evaluation metrics distinguish detection from localization and specify formatting constraints and the “no violations” case; and the ablations span SFT vs. DPO, with/without chain-of-thought, and different data compositions. The conclusions consistently indicate that verifiable rewards + preference optimization encourage generalization rather than memorization.

### Weaknesses
1. The benchmark feels small and narrowly built. The PEP “hard” set isn’t that big, and the initial collection relies on high-recall heuristics followed by manual cleanup, easy to introduce retrieval bias and hurt external validity. Also, there’s no mention of inter-annotator agreement.

2. New application not equals to new method. Porting instruction tuning, synthetic data, and RS-DPO to this setting is neat, but these are mature techniques. It reads more like a solid deployment to a new domain than a fundamental methodological breakthrough.

3. Missing key baselines. There’s no few-shot training directly on the hard PEP data, no RAG comparison, and no multi-task setup that mixes easy/hard idioms—all of which are natural baselines to include.

4. Localization is weak and under-analyzed. Line-level localization F1 is only 26.73%, and there’s no real error analysis explaining why localization lags detection. It also doesn’t discuss whether coarser localization (e.g., function-level instead of line-level) might be sufficient or more reliable.

### Questions
Could you clarify the distinction between “hard idioms” and “unseen idioms”? Is “hard” about intrinsic complexity semantics or context or multi-span localization, or does it simply mean the idiom wasn’t present in training and is therefore “unseen”? Have you tried idioms that are fundamentally different from both Ruff and PEP (e.g., cross-file semantics, resource lifecycles, concurrency patterns) to test robust generalization?

Regarding the ablation on the proportion of “NO VIOLATIONS FOUND” cases((k%)): you report that 5% yields the best F1. Does this hold across different models and datasets, or does it require per-model tuning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents METALINT, a framework for teaching LLMs to perform code quality analysis by following natural language instructions. Its core "easy-to-hard" generalization idea involves training (via SFT and DPO) on synthetic data from simple linter rules. The authors show this enables a small 4B model to generalize to complex semantic PEP idioms, achieving SOTA detection recall on a custom benchmark.

### Strengths
1.  The paper is well-written and addresses the critical problem of LLM-based code quality analysis, rightly identifying that models struggle to adapt to evolving best practices.
2.  The proposed "easy-to-hard" generalization framework is novel and clever. It leverages instruction tuning on linter-generated synthetic data, avoiding the need for expensive manual annotation.
3.  Using the linter itself as a "verifiable reward model" for preference optimization (RS-DPO) is a strong methodological contribution that provides a data-efficient path to generalization.

### Weaknesses
1.  The "meta-linting" formulation appears to evaluate only one idiom specification at a time. This is a potential departure from real-world linters, which must check hundreds of rules simultaneously. It would be beneficial to explore how METALINT's performance scales when many idiom specifications are provided in a single prompt.
2.  The paper convincingly shows that DPO enables generalization from "easy" to "hard" idioms, but the underlying why remains an interesting open question. It would be valuable to further investigate what the model is learning: a generalized concept of "code quality," or a more general "instruction-following" capability. Further analysis here could strengthen this compelling hypothesis.
3.  The SOTA claims on the "hard" PEP benchmark are very promising. However, the benchmark's current size (536 examples) is relatively modest. Expanding this benchmark in future work could help further solidify the robustness of these strong results.
4.  The finding that the CoT model underperforms (lower recall) is counter-intuitive. The paper attributes this to "overthinking." An alternative hypothesis worth exploring is that the RS-SFT data collection (filtering for perfect rewards) may have inadvertently trained the model to be overly conservative when facing ambiguity.

### Questions
1.  The non-CoT model achieved the highest recall, while the CoT model had much higher precision. Does this suggest a fundamental precision/recall trade-off? Was an ensemble of the two models considered to potentially achieve the best of both?
2.  Table 10 shows a trade-off based on the fraction of "NO VIOLATIONS" (NV) data in DPO training. The 0% NV model had the highest recall on the Ruff test set. What was the performance (P/R/F1) of this 0% NV model on the "hard" PEP benchmark? Is it possible the main SOTA recall claim is actually an underestimate?
3.  The meta-task definition includes both a description ($D_I$) and examples ($E_I$). How sensitive is the model's performance to the quality and quantity of these examples? An ablation on few-shot vs. zero-shot (description only) instructions would be insightful.

### Soundness
3

### Presentation
3

### Contribution
3
