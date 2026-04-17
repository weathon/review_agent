# SoSBench: Benchmarking Safety Alignment on Six Scientific Domains

- Decision: Accept (Poster)
- Scores: 2, 4, 4, 4

## Abstract
Large language models (LLMs) exhibit advancing capabilities in complex tasks, such as reasoning and graduate-level question answering, yet their resilience against misuse, particularly involving scientifically sophisticated risks, remains underexplored. Existing safety benchmarks typically focus either on instructions requiring minimal knowledge comprehension (e.g., ``tell me how to build a bomb") or utilize prompts that are relatively low-risk (e.g., multiple-choice or classification tasks about hazardous content). Consequently, they fail to adequately assess model safety when handling knowledge-intensive, hazardous scenarios.

To address this critical gap, we introduce SOSBench, a regulation-grounded, hazard-focused benchmark encompassing six high-risk scientific domains: chemistry, biology, medicine, pharmacology, physics, and psychology. The benchmark comprises 3,000 prompts derived from real-world regulations and laws, systematically expanded via an LLM-assisted evolutionary pipeline that introduces diverse, realistic misuse scenarios (e.g., detailed explosive synthesis instructions involving advanced chemical formulas). We evaluate frontier models within a unified evaluation framework using our SOSBench. Despite their alignment claims, advanced models consistently disclose policy-violating content across all domains, demonstrating alarmingly high rates of harmful responses (e.g., 84.9% for Deepseek-R1 and 50.3% for GPT-4.1). These results highlight significant safety alignment deficiencies and underscore urgent concerns regarding the responsible deployment of powerful LLMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
LLMs are advancing in complex tasks but their resilience against misuse, especially scientifically sophisticated risks, is underexplored. Existing safety benchmarks focus on low-risk instructions or minimal knowledge comprehension, failing to assess model safety in knowledge-intensive scenarios. SOSBENCH, a regulation-grounded benchmark, aims to address this gap by incorporating six high-risk scientific domains. However, advanced models consistently disclose disallowed content, highlighting safety alignment deficiencies.

### Strengths
+ The authors introduce the first regulation-grounded, hazard-focused benchmark spanning six scientific disciplines.

### Weaknesses
- The scope of the benchmark is too large (scientific knowledge), and the many domain knowledge is not involved.
- Only a few jailbreaking methods are involved, which hinder the completeness of the safety analysis.
- The benchmark only contain a few models in different domains, and even some of the representative models are missing.

### Questions
1. The benchmark data is generated intrinsically by LLMs, which means the data could be too synthetic. The reliability should be examined by the experts.

2. The scope of the benchmark is too large (scientific knowledge), and many domain knowledge areas are not involved. For example, toxicity in chemistry and illegal drugs in medicine. A lot of knowledge in different domains is missing. I would recommend that the authors narrow the scope.

3. The experimental analysis only involves a limited number of LLMs. Some representative domain-specific models, e.g., BioGPT and MedPaLM, are not included. This hinders the reliability of the conclusion.

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
This paper introduces SoSBench, a regulation-grounded benchmark designed to evaluate the safety alignment of large language models (LLMs) across six high-risk scientific domains, including chemistry, biology, medicine, pharmacology, physics, and psychology. The benchmark is constructed through a three-stage process: expert seed term extraction from regulations, template-based prompt generation, and LLM-assisted evolution using coverage-driven sampling. Evaluation across multiple LLMs under varying reasoning budgets and jailbreak settings reveals that advanced models still exhibit unsafe behaviors in domain-specific high-risk contexts.

### Strengths
1. The task is significant: existing safety benchmarks mainly focus on generic or shallow risks, leaving gaps in evaluating scientifically grounded hazards defined by formal regulations.

2. The data construction pipeline is novel and structured, and the inclusion of a Lite version lowers the barrier to reproduction and large-scale testing.

3. The experiments are comprehensive, covering diverse LLMs and testing configurations, and yield meaningful insights into safety behavior patterns.

### Weaknesses
1. Quality verification only checks whether responses are harmful, not whether the questions themselves are rigorous or unambiguous. There is no reported validation of problem correctness or semantic clarity.

2. Hill notation (mentioned in Section 3.2.1) and similar representations may not be unique identifiers; multiple structural isomers can share the same formula, leading to ambiguous or misleading prompts.

3. The paper does not convincingly demonstrate that SoSBench-Lite results are consistent with the full benchmark; no statistical correlation or ranking consistency is reported.

4. The paper provides limited constructive guidance on improving safety alignment—no concrete methods are proposed to mitigate the identified vulnerabilities.

5. Although the appendix discusses the use of an LLM judge, relying on a single model as evaluator could introduce bias; stronger or more diverse judges might produce different results.

The paper tackles a highly relevant and underexplored problem and contributes a potentially valuable benchmark. However, key methodological and validation issues remain unresolved—particularly question quality assurance, data transparency, and the lack of quantitative verification for SoSBench-Lite’s representativeness. Addressing these could elevate the work to acceptance level.

### Questions
1. How can the authors verify that the generated questions are of high quality—for example, logically sound and factually valid?

2. In Section 3.2.1, the paper mentions using alternative forms to replace core terms; how do the authors ensure one-to-one correspondence (e.g., molecular Hill formulas not mapping to multiple isomers)?

3. What empirical evidence supports that the Lite benchmark can reliably approximate results from the full dataset?

4. Could the authors disclose more details about the Manual Seed Collection phase—specifically, the number and background of experts, term-pool size, and review/adjudication workflow?

5. Regarding Finding 5, why would a model with visible CoT become more unsafe simply by increasing reasoning budget? If the CoT were hidden and only final answers shown, would the same issue persist?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript presents SOSBENCH, a 3,000-prompt, regulation-grounded benchmark for measuring LLM safety in six domains (chemistry, biology, medicine, pharmacology, physics, psychology). Prompts are seeded from regulatory terminology, expanded via an LLM-assisted evolutionary pipeline, and validated with surrogate models. The authors evaluate a wide range of frontier models and report Policy Violation Rate (PVR) per model/domain using an LLM-as-judge pipeline. Results indicate high rates of policy-violating output across many models and domains.

### Strengths
1. The paper addresses a timely and important problem by evaluating model safety in scientific domains where domain-specific knowledge can lead to misuse. This fills a clear gap in existing safety benchmarks and has high potential impact for both research and deployment.
2. The work is well grounded in authoritative regulatory sources. This regulatory foundation enhances real-world relevance and helps ensure that the benchmark reflects authentic, high-risk scenarios rather than artificial examples.
3. The experimental evaluation is extensive and thorough. The authors test a wide range of both open and closed models, report per-domain Policy Violation Rates (PVRs), and include adversarial multi-turn experiments that effectively demonstrate model brittleness.

### Weaknesses
1. Given that all safety scores rely entirely on GPT-5 as the evaluator, how reliable is this judgment process in practice? Have the authors provided quantitative validation against human annotations, for example, agreement metrics such as Cohen’s κ or F1 on a human-labeled subset, and discussed potential biases or failure cases of the judge?
2. The multi-turn jailbreak experiments are intriguing, but can the authors clarify the exact setup? What specific attack algorithms, interaction lengths, and numbers of trials were used, and how can these results be reproduced independently?
3. Several baselines in Table 1 seem to be taken from prior papers that used different evaluation settings. Have any baselines been re-evaluated under the same GPT-5 judging and decoding setup, or are the incomparable results clearly marked as such?
4. Line 229 states “LlamaGuard 2” while footnote links to Llama-Guard-3-8B, please clarify exact versions.

### Questions
1. Beyond binary PVR, do you report a severity-weighted metric (mapped to specific clauses/levels in the cited standards) so minor slips don’t count the same as high-risk procedural guidance?
2. Did the author check for overlap between SOSBENCH prompts and public training/eval corpora, and quantify any leakage that could inflate or skew results?

### Soundness
2

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
The paper highlights safety alignment deficiencies in a wide range of advanced LLMs. This is demonstrated on a collection of scientific prompts that elicit unsafe behaviour from the models. The set of prompts is collected by conditioning on regulatory text and it is constructed in a manner to increase coverage of scenarios.

### Strengths
- The paper highlights an important issue raised in other forms by various other works, that general safety mechanisms have blind spots on domain-specific knowledge. 
- The provided benchmark and empirical analysis highlight this shortcoming well. The empirical analysis in particular is very comprehensive, evaluating a wide range of model classes and scales.

### Weaknesses
- The claimed difference in hazard level from SciSafeEval [1] is not as pronounced as suggested. Examining the example prompts from either benchmark in Appendix B.2, a person intending to act on the model’s output is equally likely to pose either question. What, then, is the precise aspect of hazard level that this benchmark seeks to distinguish? Clarifying this distinction would help future practitioners determine which benchmark is most appropriate for their specific use cases. A related concern is raised below.
- The overall motivation for evaluating on this benchmark would also benefit from additional clarity. It appears to target safety vulnerabilities in scenarios where the model’s scientific knowledge is being applied: yet, as stated in L49–50, not in cases involving “advanced” scientific knowledge. This distinction is somewhat ambiguous and should be elaborated.
- *Missing standard error in the empirical analysis:* The results in Tables 2 and 3 and Figure 3 are averaged over multiple data points and categories. The underlying variance or standard error of the PVR should be reported to assess statistical significance. For instance, the difference in PVR for Finding 5 (as supported by Figure 5) is quite small—on the order of 0.1 across budgets. Reporting the standard error in such cases is necessary to gauge the statistical strength of the results.

---
[1] Tianhao Li, Jingyu Lu, Chuangxin Chu, Tianyu Zeng, Yujia Zheng, Mei Li, Haotian Huang, Bin
Wu, Zuoxian Liu, Kai Ma, et al. Scisafeeval: a comprehensive benchmark for safety alignment
of large language models in scientific tasks. arXiv preprint arXiv:2410.03769, 2024b.

### Questions
What does Figure 3 look like if SciSafeEval is also included? It would be insightful to show a comparison to the most similar existing benchmark.

### Soundness
3

### Presentation
4

### Contribution
2
