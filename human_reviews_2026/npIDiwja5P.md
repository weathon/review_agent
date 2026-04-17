# SciDA: Scientific Dynamic Assessor of LLMs

- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
Advancement in Large Language Models (LLMs)  reasoning capabilities enables them to solve scientific problems with enhanced efficacy. Thereby, a high-quality benchmark for comprehensive and appropriate assessment holds significance, while existing ones either confront the risk of data contamination or lack involved disciplines. To be specific, due to the data source overlap of LLMs training and static benchmark, the keys or number pattern of answers inadvertently memorized (i.e. data contamination), leading to systematic overestimation of their reasoning capabilities, especially numerical reasoning. 

We propose **SciDA**, a multidisciplinary benchmark that consists exclusively of over 1k Olympic-level numerical computation problems, allowing randomized numerical initializations for each inference round to avoid reliance on fixed numerical patterns. We conduct a series of experiments with both closed-source and open-source top-performing LLMs, and it is observed that the performance of LLMs drop significantly under random numerical initialization. Thus, we provide truthful and unbiased assessments of the numerical reasoning capabilities of LLMs. The evaluation framework has been anonymized and is publicly available at **https://anonymous.4open.science/r/SciDA-0184**

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces SciDA, a dynamic, multidisciplinary benchmark designed to assess the numerical reasoning abilities of large language models (LLMs) while mitigating data contamination. SciDA consists of over 1,000 expert-curated, Olympic-level numerical computation problems covering mathematics, physics, chemistry, and biology, with dynamic variable initialization to systematically eliminate answer memorization and pattern-based shortcuts. The authors evaluate a wide suite of contemporary LLMs in both fixed and randomized settings and find significant performance drops under randomization, suggesting overestimation of existing models’ reasoning abilities on static benchmarks.

### Strengths
1. Dynamic Contamination-Proof Design: SciDA leverages randomized parameter initialization, which directly tackles data contamination and memorization issues commonly plaguing static benchmarks.
2. Multidisciplinary Coverage: By including computation-heavy problems from mathematics, physics, chemistry, and biology, SciDA offers broader evaluative reach than most prior benchmarks limited to single domains.
3. Robust Experimental Evaluation: The authors present comprehensive experiments involving 14 LLMs, reporting per-model results in both fixed and random settings.
4. Insightful Error Analysis: The paper provides a nuanced breakdown of error types and links performance degradation to the lack of true numerical/generalization skills.

### Weaknesses
1. Limited Discussion of Theoretical Underpinnings of Randomization Strategy: While Equations (3.1)–(3.5) define the sampling and evaluation framework, the theoretical justification for why uniform randomization over the prescribed intervals is sufficient to prevent contamination (vs. adversarial or structured randomization), or how initialization ranges are determined for each problem, remains underexplored.
2. Potential Annotation and Validation Quality Risks: The annotation/validation process relies on teams of medalists and students, but the degree of independent double-checking, systematic error checking, or inter-annotator agreement is not quantified.
3. Insufficient Mathematical Formalization of Correctness/Tolerances: While the paper specifies that model predictions are deemed correct if "within a prescribed tolerance," the exact metrics are not made explicit, nor are the thresholds justified for each domain/problem type.

### Questions
1. Unclear Definition and Justification of Correctness Tolerances Across Domains: How are correctness tolerances defined and justified across the different scientific domains? For example, do relative/absolute error thresholds vary by subject (e.g., chemistry vs. mathematics), and what is the rationale for those choices?
2. Absence of Quantitative Uncertainty and Error Bar Reporting: Are there quantitative uncertainty/error bars (e.g., variance across runs) that could be reported for the benchmark’s headline metrics?

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
4

### Summary
This paper introduces SciDA, a new dynamic benchmark with over 1,000 scientific problems designed to address performance overestimation in LLMs caused by data contamination in static benchmarks. Its core feature is the random initialization of numerical parameters for each problem, preventing models from relying on memorization. The authors' key finding is that this randomization causes a significant drop in accuracy for all tested LLMs, revealing a vulnerability in their reasoning capabilities.

### Strengths
1. The problem this paper addresses is a well-recognized and critical issue in the field of LLM evaluation.
2. The dynamic "functionalization" and random initialization of problems is a sound and necessary methodology for testing true generalization over memorized patterns.
3. The rigorous, expert-led data collection and annotation pipeline ensures a high-quality, difficult set of problems that yield valuable insights into model capabilities.

### Weaknesses
1. The dataset's scale (1,000 problems) is limited when distributed across four distinct scientific disciplines and multiple difficulty levels, which may affect the statistical reliability of results in specific sub-domains.

2. The paper's main conclusion is ambiguous. The performance drop is attributed to a failure in "genuine problem-solving" , yet the paper's own error analysis (Figure 4) shows "Calculation Errors" are far more common than "Logical Errors" in most subjects . This suggests models might be generalizing the logic correctly but failing at the computation. If the logic is correct, do the models just need a calculator or code interpreter tool to solve the problems?

3. Key methodological details are missing, specifically how the "scientifically valid ranges" for parameter randomization were defined and validated.

### Questions
1.  Could the authors please provide a more detailed statistical breakdown of the 1000 problems in an appendix? Specifically, a table showing the exact distribution across the four disciplines and three difficulty levels (as referenced in Figure 6a 35) would be very helpful.

2.  I suggest the authors refine their claim of being the "first" dynamic benchmark. Please clarify the distinction from KORgym and Math-perturb more explicitly in the Related Work section and adjust the contribution statement to be more precise.

3. Please elaborate on the implications of the findings in Figure 3,4. If models are primarily making "Calculation Errors," does this not imply that their logical generalization is (at least partially) successful, but their internal arithmetic/symbolic manipulation capabilities are brittle? This distinction is crucial for understanding what "reasoning" means for LLMs and where the true bottleneck lies.

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
4

### Summary
This paper presents SciDA (Scientific Dynamic Assessor of LLMs), a multidisciplinary dynamic benchmarking framework designed to comprehensively and authentically evaluate the scientific reasoning capabilities of large language models (LLMs). SciDA comprises over 1,000 olympiad-level numerical computation problems, each of which can be randomly initialized with different numerical values at every inference attempt, thereby preventing models from relying on fixed numerical patterns. Experiments conducted on SciDA with multiple top-tier open-source and closed-source LLMs reveal a significant performance drop under randomly initialized numerical values, demonstrating that SciDA provides a realistic and unbiased assessment of numerical reasoning abilities.

### Strengths
SciDA effectively prevents models from relying on fixed numerical patterns by randomly initializing the variables in each problem. Moreover, SciDA spans multiple disciplines, including mathematics, physics, chemistry, and biology, and all its problems are drawn from Olympiad-level competitions, ensuring high quality and complexity. This comprehensive evaluation approach provides researchers with a more realistic and holistic assessment tool for evaluating the scientific reasoning capabilities of large language models (LLMs).

### Weaknesses
- The core approach of SciDA, randomly initializing variables within problems, effectively mitigates model reliance on fixed numerical patterns. While this strategy reduces the risk of data contamination to some extent, it remains relatively simplistic and lacks deeper analysis or evaluation of the model’s actual reasoning process.
- Although SciDA’s dynamic initialization strategy addresses data contamination to a degree, similar techniques have already been employed in other domains, such as dynamic benchmarking. Consequently, SciDA does not represent a significant innovation in methodology.
- The approach may also lack flexibility: the range and manner of random initialization are fixed and not adaptively tuned based on problem difficulty or model capability. This rigidity could result in some problems becoming either too easy or excessively hard after randomization, thereby failing to accurately reflect the model’s true reasoning capacity.

### Questions
1. In the experiments involving random initialization, did the authors consider potential differences in how sensitive various models are to such randomization? For instance, some models might be more robust and adaptable to randomized inputs, while others could be heavily reliant on fixed numerical patterns.

2. Within the dynamic initialization strategy, did the authors explore the possibility of adaptively adjusting the range and distribution of variables based on problem difficulty and model capability? For example, for particularly challenging problems, could narrowing the variable range help moderate difficulty and thereby yield a more precise assessment of a model’s reasoning ability?

3. In the error analysis section, did the authors consider performing a finer-grained categorization of model errors? Beyond logical and computational errors, could additional error types—such as misinterpretation (understanding errors) or formulation/representation errors (expression errors)—be introduced to provide deeper insights into model failure modes?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SciDA, a multidisciplinary benchmark designed to assess the scientific reasoning ability of LLMs under dynamic randomized conditions. The authors identify a critical issue about data contamination in the current benchmarks for evaluating LLMs. To address this, SciDA collects over 1K Olympiad-level scientific computation problems covering math, physics, chemistry, and biology, each parameterized with random variable ranges to eliminate memorization effects. The benchmarking approach involves expert curation and dynamic random initialization. Experiments on 14 mainstream LLMs reveal that accuracy drops by up to 60% under randomization, exposing an overestimation of current LLMs' reasoning ability. The paper concludes that SciDA offers a contamination-free approach for evaluating genuine reasoning performance, with plans for future expansion into more disciplines.

### Strengths
- The data curation pipeline with Olympiad-level problems and expert annotation ensures quality and complexity.

- The empirical findings of systematic performance drop show the effect of randomized conditions and reveal concerns of data contamination.

- Code is provided for reproducibility.

### Weaknesses
- Some statements are a bit overclaimed.
	- The claim of being "contamination-proof" is not fully substantiated. This is actually a very hard problem to completely solve.
	- In the abstract, there is "we provide truthful and unbiased assessments of the numerical reasoning capabilities of LLMs", but there are actually no guarantees.

- Writing can be refined for conciseness and professional tone.

### Questions
- How to ensure that the dynamic randomization does not inadvertently generate unsolvable problems beyond the predefined ranges?

- How to verify the "contamination-proof" claim? How to verify empirically that SciDA problems are unseen in major training corpora like Common Crawl?

### Soundness
3

### Presentation
2

### Contribution
3
