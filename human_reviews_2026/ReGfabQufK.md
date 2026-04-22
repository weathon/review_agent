# A New Perspective on Large Language Model Safety: From Alignment to Information Control

- Avg Score: 2.67
- Decision: Reject
- Scores: 4, 2, 2

## Abstract
Large Language Models (LLMs) have demonstrated remarkable capabilities
across a wide range of domains, yet their increasing deployment in sensitive
and high-stakes environments exposes profound safety risks—most notably, the
uncontrolled generation of inappropriate content and the inadvertent leakage of
confidential information. Traditionally, such risks have been approached through
the lens of alignment, focusing narrowly on ensuring outputs conform to general
notions of helpfulness, honesty, and harmlessness. In this work, we argue that
such alignment-centric perspectives are fundamentally limited: information itself
is not inherently harmful, but its appropriateness is deeply context-dependent.
We therefore propose a paradigm shift in LLM safety—from alignment to information control. Rather than merely shaping model behavior through the existing
practice of alignment, we advocate for the principled regulation of who can access
what information under which circumstances. We introduce a novel framework
for context-sensitive information governance in LLMs, grounded in classical secu-
rity principles such as authentication, role-based access control, and contextual
authorization. Our approach leverages both the internal knowledge represen-
tations of LLMs and external identity infrastructure to enable fine-grained,
dynamic control over information exposure.

We systematically evaluate our framework using recent models and a suite of
benchmark datasets spanning multiple application domains. Our results demon-
strate the feasibility and effectiveness of information-centric control in mitigating
inappropriate disclosure, providing a robust foundation for safer and more
accountable language model deployment. This work opens a new frontier in LLM safety, one rooted not in abstract alignment ideals, but in enforceable,
context-aware control of information flow.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposed an information control framework for large language models, shifting from alignment-based safety to context-aware information flow management. To enhance security, the framework integrates user identification, policy alignment, role-based access control, and post-processing modules. A benchmark dataset was constructed based on real enterprise scenarios, and experiments demonstrated that the proposed method significantly improves defense against unauthorized information leakage while maintaining high correctness and acceptable latency.

### Strengths
Unlike existing approaches that primarily focus on content filtering or adversarial training, this research reframes LLM security at the architectural level by proposing "information control" as a new paradigm that offers superior engineering feasibility and organizational adaptability. The technical solution demonstrates rigorous design with clear module segmentation and logical consistency. The inclusion of Shapley value analysis and module combination testing further strengthens the credibility of the conclusions. As LLMs become increasingly integrated into core enterprise workflows, the need for fine-grained, controllable access to sensitive information has become increasingly urgent. This solution that offers significant reference value for both subsequent research and real-world deployment.

### Weaknesses
1. Compared to classical rule-based access control systems, the semantic approach of the PA module may introduce new uncertainties and attack surfaces.

2. While the PP module helps prevent information leakage, it may lead to erroneously removing non-sensitive but semantically similar content,  compromising output usability. Although the paper mentions "configurable sensitivity," it does not quantify the usability-security trade-off under different configurations.

### Questions
1. The PA module is described as using "LLM-based reasoning to semantically interpret prompts." Is this module itself also an LLM? How does it ensure robustness against the same jailbreaking attacks it aims to prevent? Is its decision-making process auditable?

2. When the PP module incorrectly redacts critical business terminology, how can users obtain actionable feedback?

### Soundness
2

### Presentation
2

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
The paper proposed a framework that shifts the paradigm of LLM safety mechanisms from reshaping alignment to information/access control. Rather than merely shaping model behavior through the existing practice of alignment, they examined the traditional access control-based technique in LLMs to prevent information leakage.

### Strengths
1.	The paper introduces a clear and practical framework that controls who can access what information in an LLM, making it much safer for real-world use in companies handling sensitive data.
2.	The proposed system improves protection against data leaks and harmful outputs while still keeping good accuracy and only a small delay in response time.

### Weaknesses
1.	The description of the modules of Figure 1 lacks a complete picture of the methods on how each component works. For example, the user identification module checks if the user is authenticated to have such information. How does this module perform this operation? A detailed methodology is highly required to make it clear. 
2.	How would the authors justify the use of GPT-4o as a judge since the same model is used as the target model for evaluation?
3.	Why did the authors only evaluate on those specific three models in the paper, where numerous open-sourced and closed-sourced models are available? What justifies the selection of the specifically 2 OpenAI models and one Google model?
4.	Also, what's the performance of the proposed defense on reasoning-focused models and mixture-of-expert models?
5.	The paper should also test the defense performance against SOTA attacks to understand the proposed defense method’s utility and compare the performance with the other defense techniques under the same attacks.
6.	What are potential failure cases of the proposed method?
7.	This paper severely lacks a performance comparison with the baseline techniques for LLM defense.

### Questions
Please follow the Weaknesses.

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
3

### Summary
This paper argues that traditional alignment-based safety frameworks (e.g., RLHF, Constitutional AI) are insufficient because they treat all content generation as equally harmful or safe, ignoring who requests the information and in what context.
The authors propose a paradigm shift from alignment to information control, introducing a modular framework inspired by classical access control principles (authentication, RBAC, and contextual authorization).
The framework integrates four modules (user identification, policy alignment, role checking,  and post processing) to regulate LLM outputs based on identity, role, and context.
Experiments on GPT-4o and open-source models show that the combined modules can reach 97.7% defense rate while maintaining high correctness and acceptable latency, outperforming baseline alignment-only and static-control setups.

### Strengths
1. The paper provides an original and well-argued conceptual shift: viewing LLM safety as information governance rather than behavioral alignment. This perspective could influence future safety frameworks.
2. The empirical results include comparisons across different module combinations, ablation studies, latency analysis, and both closed- and open-source models.

### Weaknesses
1. Although framed as a new paradigm, many components (policy filtering, role-based rules, post-filtering) resemble structured prompt-engineering pipelines rather than a fundamentally new safety algorithm, which raises concerns about how different this really is from prompt conditioning or retrieval gating.
2. While latency and correctness are measured, the paper does not analyze usability degradation, false denials, or context misclassification, which are crucial for deployability.
3. The comparison with alignment-based safety methods is shallow. The paper positions itself as a paradigm shift but doesn't empirically demonstrate where alignment fails and information control succeeds. More comprehensive comparison should be made (such as comparison with other alignment techniques, and representative jailbreak defense methods).

### Questions
1. Can this framework be combined with alignment methods to get better results?

### Soundness
2

### Presentation
2

### Contribution
2
