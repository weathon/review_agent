# Robust LLM Unlearning via Post Judgment and Multi-round Thinking

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
The unlearning capability of LLMs is vital for ensuring compliance and safety, especially when removing sensitive knowledge from deployed models. Pre-filtering methods, enabling rapid deployment without parameter changes, are a prominent unlearning approach. However, they exhibit significant robustness deficiencies against adversarial attacks: in the worst case, simple prefix attacks can induce up to a 1,150-fold surge in information leakage for fictitious entity knowledge, while composite question attacks can cause accuracy on hazardous knowledge to rebound from the 24.9% random-guess baseline to as high as 67.0%. To address this, we propose a new unlearning framework via post judgment and multi-round thinking (PoRT), which consists of three key modules. First, a data cleaning module compiles a dynamic few-shot prompt that instructs the LLM to simultaneously generate both a cleaned version of the user's query and a corresponding initial response, supported by an extensible demonstration library for adaptive defense. Second, unlike existing pre-filtering methods that typically judge based solely on prompts, our post-judgment module jointly evaluates cleaned prompts and their corresponding responses to better detect non-compliant outputs. Finally, a selective multi-round thinking process is employed to trigger LLM's self-correction for low-confidence outputs, enhancing reliability and result quality. Extensive experiments on benchmarks demonstrate PoRT's superior robustness against adversarial attacks and strong unlearning effectiveness without compromising general model utility. Code is available at https://github.com/ChnIRuI/PoRT_LLM_Unlearning

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper highlights that existing pre-filtering unlearning methods fail under adversarial attacks, causing severe leakage and utility loss. To address this, the authors propose PoRT, a framework combining dynamic data cleaning, post-judgment evaluation, and multi-round self-correction. Experiments show PoRT is more robust against attacks and has strong unlearning effectiveness while preserving model utility.

### Strengths
- PoRT introduces a post-judgment mechanism that jointly analyzes prompts and responses, effectively leveraging LLMs’ reasoning capabilities for more reliable unlearning.

- The paper provides extensive empirical evaluation, demonstrating robustness against adversarial attacks and effective unlearning on TOFU and WMDP, with superior performance compared to SOTA methods.

- The overall writing is clear and well-structured.

### Weaknesses
- It would be helpful to elaborate on the comparison with model-based unlearning methods (Line 39), particularly regarding effectiveness and efficiency against adversarial attacks and on unlearning tasks.

- Providing statistical uncertainties for the reported results would strengthen the empirical evaluation and make the findings more convincing.

- Minor comment: The definitions of Forget Probability, ROUGE, and 1-TR for Figure 2 appear only in the appendix (Line 310). A brief, early description would help readers follow the results without jumping ahead.

### Questions
Please see **Weaknesses** for my questions.

### Soundness
3

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
This paper proposes the PoRT (Post judgment and multi-Round Thinking) framework to address the vulnerability of existing LLM unlearning methods under adversarial attacks. The study reveals systemic deficiencies in mainstream pre-filtering approaches: simple prefix attacks can induce up to a 1,150-fold surge in information leakage, while composite question attacks cause accuracy on hazardous knowledge to rebound from the 24.9% random-guess baseline to as high as 67.0%. The PoRT framework comprises three core modules: (1) In-Context Prompt Cleaning (IPC), which leverages LLMs' in-context learning capabilities to dynamically clean adversarial inputs; (2) Post Judgment mechanism, which jointly evaluates cleaned question-answer pairs; and (3) Selective Multi-round Thinking (SMT), which triggers self-correction for low-confidence outputs. Extensive experiments on the TOFU and WMDP benchmarks demonstrate PoRT's superior performance compared to baselines.

### Strengths
1. **High-quality writing.** The paper is well-written with clear presentation and logical flow throughout.

2. **Well-motivated framework design.** The approach effectively transforms unlearning from simple rejection to contextual reasoning and continuous refinement, representing a meaningful paradigm shift from pre-filtering to post-judgment evaluation.

3. **Comprehensive experimental validation.** The evaluation is thorough and systematic, spanning 10 representative LLMs, two major benchmarks (TOFU and WMDP), multiple data splits, three types of adversarial attacks, and multi-dimensional evaluation metrics.

### Weaknesses
1. **Methodological contradiction.** While PoRT criticizes pre-filtering methods for relying on "superficial input analysis", its own post-judgment classifier is fundamentally a supervised learning model that similarly depends on patterns observed in training data. When confronted with novel attacks outside the training distribution (e.g., potential future multimodal adversarial attacks), the classifier may fail in similar ways to the pre-filtering methods it criticizes.

2. **Limited technical novelty.** All core components represent applications or combinations of existing techniques rather than algorithmic innovations. The IPC module essentially performs data cleaning, the post-judgment classifier is implemented using the existing CCL-SC method, and SMT relies on carefully engineered prompts. The overall contribution appears more engineering-focused than algorithmically innovative, lacking methodological advances beyond systems integration.

3. **Questionable generalization capability.** The post-judgment classifier's ability to generalize remains unclear. All validation is conducted on in-domain data with supervised training and evaluation on identically distributed test sets. The paper provides no evidence of performance on out-of-distribution data, raising concerns about robustness to distribution shifts in real-world deployment scenarios.

### Questions
1. **Potential error in Table 4.** PoRT does not achieve the best results in Table 4, yet two entries are bolded as if they were[1]. Specifically, ECO achieves the best results on Bio (24.7% vs. PoRT's 25.1%) and Cyber (24.4% vs. PoRT's 24.8%), but the authors bold their own results. Whether this is an unintentional error or not, the authors benefit from this misrepresentation. **I must bring this discrepancy to the attention of all reviewers and the Area Chair to ensure accurate evaluation.**

2. **Lack of failure case analysis.** The paper would benefit significantly from analyzing failure modes where PoRT's post-judgment classifier and SMT module both fail to prevent information leakage. Understanding the boundaries of the method's effectiveness is crucial for assessing its real-world applicability and guiding future improvements.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper argues that popular pre-filtering unlearning methods are brittle. Simple prefix or composite-question jailbreaks can reverse the intended forgetting. It proposes PoRT, a three-stage inference-time framework, including In-Context Prompt Cleaning, Post-Judgment, and Selective Multi-round Thinking.

### Strengths
1. The paper shows pre-filtering’s vulnerabilities under realistic attacks and provides a valuable perspective.
2. The paper proposes a conceptual shift from input-only filters to answer-aware post-judgment, which aligns with where leakage actually manifests.
3. Selective classification with abstain is an interesting safety control; pairing it with targeted re-thinking is efficient.

### Weaknesses
1. The selective classifier seems strong, but robustness hinges on augmentation and demo-library coverage.

2. IPC relies on retrieval of few-shot demos; adversarially crafted inputs may steer cleaning or retrieval. More ablations on $k$, instruction choice, and demo selection failure cases would help.

3. ECO is explained and included, but some recent multi-agent or post-hoc critics are only discussed conceptually. They should be compared in experiments.

4. While improvements are compelling, HFQ is author-introduced; cross-validation by third-party evaluators, or human-rater correlation, would make the claims more robust.

### Questions
1. How does performance change when attacks are substantially different from training/augmentation?

2. How is the IPC demo library curated/expanded safely to avoid poisoning? Is there a confidence back-off when retrieval is low-similarity or contradictory?

3. The authors claim modest latency overhead vs. ECO; can you break down cost across IPC, post-judgment, and SMT rounds under real-world harmful-prompt prevalence?

### Soundness
3

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
2

### Summary
The paper introduces PoRT, a novel unlearning framework for large language models that enhances compliance and safety by removing sensitive knowledge without harming overall performance. By combining data cleaning, post-judgment evaluation, and selective multi-round thinking, PoRT significantly improves robustness against adversarial attacks and maintains strong unlearning effectiveness.

### Strengths
The writing flow and structure is easy to follow. Comparisons across pre-filtering methods under adversarial attacks support a clearer motivation.

Comprehensive experiments are conducted to validate PoRT's effectiveness and efficiency, providing valuable insight in LLM Unlearning without params altering.

### Weaknesses
The techniques and methods incorporated into the framework are integrated in a rather straightforward manner, making the overall contribution appear somewhat trivial.

### Questions
null

### Soundness
3

### Presentation
3

### Contribution
3
