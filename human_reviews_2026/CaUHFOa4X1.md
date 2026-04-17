# Scam2Prompt: A Scalable Framework for Auditing Malicious Scam Endpoints in Production LLMs

- Decision: Reject
- Scores: 4, 4, 6, 6, 6

## Abstract
Large Language Models (LLMs) have become critical to modern software development, but their reliance on uncurated web-scale datasets for training introduces a significant security risk: the absorption and reproduction of malicious content. This threat is not merely theoretical, as demonstrated by a real-world incident in this paper where developers lost thousands of dollars executing LLM-generated code containing scam API endpoints. To systematically evaluate this risk, we introduce Scam2Prompt, a scalable automated auditing framework that identifies the underlying intent of a scam site and then synthesizes innocuous, developer-style prompts that mirror this intent, allowing us to test whether an LLM will generate malicious code in response to these innocuous prompts. In a large-scale study of four production LLMs (GPT-4o, GPT-4o-mini, Llama-4-Scout, and DeepSeek-V3), we found that 4.24\% of code snippets generated from these prompts contained malicious URLs. Our framework also unexpectedly discovered 62 active scam sites missed by scam databases, now confirmed and added to industry blocklists. To test the persistence of this security risk, we constructed Innoc2Scam-bench, a benchmark of 1,559 innocuous prompts that consistently elicited malicious code from all four initial LLMs. When applied to seven additional production LLMs released in 2025, we found the vulnerability is not only present but severe, with malicious code generation rates ranging from 12.7\% to 43.8\%. Furthermore, existing safety measures like state-of-the-art guardrails proved insufficient to prevent this behavior. Our findings offer conclusive evidence of large-scale data poisoning in the training pipelines of production LLMs, highlighting a fundamental security gap that requires urgent attention from the research community.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies a concerning security issue in modern LLMs: even with normal-looking programming prompts, models sometimes generate code that contains real scam URLs. The authors propose Scam2Prompt, an automated pipeline to detect such cases, and release Innoc2Scam-bench with validated prompts. Experiments show malicious code generation exists across models released in 2024–2025. The paper suggests this is due to poisoned web data absorbed during training.

### Strengths
1. Addresses a practical and under-explored LLM safety issue.
2. Real-world incident strengthens motivation.
3. Automated and scalable auditing pipeline.
4. Benchmark release provides community value.
5. Able to discover new scam domains later confirmed by industry.

### Weaknesses
1. The prompt generation process does not enforce non-malicious intent strictly. Although manual filtering is claimed, generated prompts may still contain unusual scam-related cues. More rigorous methodology is needed.
2. Most data comes from Web3 scam lists. Generalization to other domains (e.g., cloud SDKs, finance APIs, SaaS) is not shown.
3. URL oracle approach is reasonable, but there is limited discussion of false positives, false negatives, and disagreement resolution.
4. Missing ablations on prompt patterns, model size, training cutoff, and alignment methods. These could help understand root causes better.
5. Problem statement is strong, but discussion on defenses (data curation, unlearning, runtime filtering) is very brief.

### Questions
1. How many prompts were manually rejected as adversarial? Do you have any inter-annotator agreement metrics?
2. Could the prompt-generating LLM unintentionally leak malicious context into "innocent" prompts?
3. Do you expect same behavior outside crypto domain? Any preliminary evidence?
4. Are original scam-site contents archived for reproducibility?
5. Have you tried filtering URLs in generated code or unlearning malicious URLs from models?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Scam2Prompt, a framework for systematically evaluating whether large language models (LLMs) generate malicious code containing scam API endpoints. The authors created Innoc2Scam-bench, a benchmark of 1,559 innocuous prompts designed to test if LLMs would generate malicious code in response. Their study found that 4.24% of code snippets generated from these prompts contained malicious URLs across four initial LLMs (GPT-4o, GPT-4o-mini, Llama-4-Scout, and DeepSeek-V3). When tested on seven additional LLMs released in 2025, the malicious code generation rates ranged from 12.7% to 43.8%. The framework also discovered 62 active scam sites missed by existing scam databases. The authors conclude that existing safety measures like state-of-the-art guardrails are insufficient to prevent this behavior, indicating a fundamental security gap in LLM training pipelines.

### Strengths
* Comprehensive Empirical Evaluation
  - The paper evaluates four 2024 LLMs and seven 2025 LLMs, totaling over 265k prompts and 1,559 benchmark cases (Sec. 6.1–6.2; Table 1–2).
  - Cross-model comparisons expose persistent vulnerabilities (malicious rates 3–6% in 2024 models; 12–44% in 2025 models), providing strong evidence of systemic data contamination.

* Clear and Modular Framework Design
  - Figure 3 and Section 4 describe a four-stage pipeline (malicious URL collection, prompt synthesis, code generation, oracle verification).
  - The use of independent “Prompt LLM” and “Codegen LLM” modules isolates contamination sources, improving causal interpretability.
  - Clean modularity allows easy adaptation to other malicious behaviors beyond URLs.

* Novel methodology and framework
  - The introduction of Scam2Prompt and Innoc2Scam-bench represents a significant contribution to the systematic evaluation of LLM security.

* Strong empirical evidence 
  - The paper provides concrete data showing the prevalence of the vulnerability across multiple LLMs, with rates ranging from 4.24% to 43.8%.

### Weaknesses
- Questionable prompt validity: The study provides limited evidence that its generated prompts genuinely represent realistic developer queries. Many appear to be synthetic or overly engineered, raising concerns about the ecological validity of the evaluation setup.

- Lack of statistical rigor: The analysis does not report statistical significance or confidence intervals for the observed differences between models, making it difficult to assess whether the findings are robust or simply due to random variation.

- Unsubstantiated causal claims: The paper asserts that “large-scale data poisoning” explains the presence of scam URLs in model outputs, but it fails to provide direct empirical evidence tracing these URLs to specific training data sources or contamination pathways.

- Superficial mitigation discussion: While the paper convincingly identifies a potential vulnerability, its discussion of mitigation strategies remains cursory and speculative. It offers few concrete or actionable recommendations for improving model robustness or dataset hygiene.

- Opaque methodology on prompt innocuousness: The authors provide insufficient transparency regarding how they validated that their prompts were “innocuous.” Beyond brief mentions of automated filtering, the methodology lacks qualitative justification or human evaluation to substantiate this claim.

### Questions
The paper states that "existing safety measures like state-of-the-art guardrails proved insufficient to prevent this behavior". However, the paper doesn't provide specific details about these guardrails. Could the authors clarify:

1. Which specific safety mechanisms were tested (e.g., content filters, policy-based guardrails, etc.)?
2. How were these guardrails evaluated against the Innoc2Scam-bench prompts?
3. What specific aspects of the scam code were not detected by these guardrails (e.g., URL patterns, semantic intent, etc.)?
4. How does the vulnerability identified in this paper compare to other known security vulnerabilities in LLMs (e.g., prompt injection attacks, data leakage, etc.) in terms of severity and prevalence?

### Soundness
3

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
4

### Summary
This paper proposes an automated audit framework and benchmark for evaluating whether production LLMs generate code that embeds malicious URLs in response to innocuous prompts. The pipeline contains several main steps from malicious collection, prompt synthesis, code generation and URL extraction, and validation for the results. They collected 1559 innocuous prompts that are validated for LLM to generate malicious code and finally showed that many SOTA models are facing such a challenge.

### Strengths
1. A good aspect concerning LLM security from malicious code generation rather than general prompt injection, jailbreaking or other techniques. Malicious coding is highly related to real-world development with LLM and both realistic and understudied, which is a promising direction.
2. The framework design is scalable and reasonable, considering both utilizing LLM for data synthesis and human-in-the-loop for verification. Also, the framework is kind of automatic, which leaves the buffer for further data updates.
3. The data after filtering still remains a good amount, the evaluation is also sufficient(evaluated a lot of SOTA models, considered temperature or other parameters), and final results show the value of the work.

### Weaknesses
1. Only focus on the evaluation of the model itself, rather than consider such threats from an agentic angle. It is understandable that a large language model trained by a lot of online data, which already contains malicious data and is uncontrolled. However, for the usage of LLM, we should still consider if such kind of problem can be resolved at the agent level (for example, by integrating detection tools or doing web search). 
2. Only focus on URL malicious code in this work, but it can be explored a lot for the whole malware aspect; the author can consider research on other malware and add static analysis or sandbox execution for a new signature in future work.
3. Currently, the work is focusing on web3, which may bias the malicious URL landscape. It might still have limitations for other domain URLs.

### Questions
1. It might be good to cite some secure code generation work as a shared related interest for LLM security.

[1] Seccodeplt: A unified platform for evaluating the security of code genai

[2] SafeGenBench: A Benchmark Framework for Security Vulnerability Detection in LLM-Generated Code

[3] CodeLMSec benchmark: Systematically evaluating and finding security vulnerabilities in black-box code language models

2. Do you have demo testing or a rough impression about whether it will go worse or get better when the model is allowed to use search tools?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Scam2Prompt, an automated system that audits production LLMs to detect if they unintentionally generate malicious code containing scam URLs when given normal-looking developer prompts. The authors also built a benchmark called Innoc2Scam-bench with 1,559 innocuous prompts that consistently trigger malicious code generation, revealing that even the latest 2025 models (e.g., GPT-5 and Gemini-2.5) still suffer from data poisoning. The study highlights how contaminated training data has made scam URL generation a persistent, industry-wide vulnerability

### Strengths
+ The study starts with a genuine scam incident, making the threat concrete and relatable.

+ Scam2Prompt offers a reusable, automated way to uncover hidden malicious behavior across multiple LLMs.

+ The authors back claims with large-scale tests (265K+ prompts, 11 models), new dataset releases, and reproducibility through public benchmarks

### Weaknesses
- The work focuses only on malicious URLs and overlooks other forms of malicious or insecure code generation.

- Its reliance on external oracles such as Safe Browsing limits the detection scope to known URL-based threats.

- The root causes of model poisoning and how data contamination propagates are not analyzed in depth.

- The evaluation centers mostly on code models, without examining text or multimodal LLMs.

- The assessment of safety tools like NeMo Guardrails is brief and lacks deeper diagnostics or alternative baselines.

### Questions
How exactly the system differentiates between “innocuous” and borderline “semi-adversarial” prompts; human validation is mentioned, but criteria aren’t well-defined.

The causal chain between dataset contamination and model outputs is assumed, not proven.

The reproducibility setup (e.g., randomness in hosted APIs, temperature differences) could use more clarity on how they ensured consistency across LLM APIs

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors introduce Scam2Prompt, an automated framework that audits production LLMs to determine whether they generate code containing malicious URLs in response to benign, developer-style prompts.
They construct Innoc2Scam-bench, a benchmark of over 1500 innocuous prompts that consistently trigger malicious code generation from four LLMs (GPT-4o, GPT-4o-mini, Llama-4-Scout, and DeepSeek-V3). They then evaluate seven newer production models released in 2025 and find that all of them still produce malicious code at significant rates, demonstrating that current training practices, dataset sanitisation, and safety guardrails remain insufficient to mitigate this threat.

### Strengths
* High impact topic. Paper well-written.
* Provides empirical evidence that the threat of malicious code from production LLMs is systemic.
* Provides a benchmark of over 1500 manually validated dangerous prompts.
* Large-scale evaluation across numerous production LLMs.

### Weaknesses
* The framework detects malicious behaviour only if the generated URL is recognised as malicious by one of the URL oracles. It may fail if the URL is new and not yet flagged anywhere, or the attack doesn’t use URLs.
* their results imply a safety–usability trade-off, but this aspect is not explored/discussed in depth. Eg. the fact that gemini-pro's high safety comes at the cost of a high filtering rate.
* no mitigations or defences are proposed.

### Questions
* How would your framework perform if a malicious domain is new and not yet flagged by any oracle?
* Are certain programming topics (eg. Solana) more vulnerable? Do you have any insights on prompt characteristics that lead models to retrieve malicious endpoints?
* Do you think malicious URL generation should be handled at generation time or post-processing?

### Soundness
3

### Presentation
3

### Contribution
3
