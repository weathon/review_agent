# Auditing Black-Box LLM APIs with a Rank-Based Uniformity Test

- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6

## Abstract
As API access becomes a primary interface to large language models (LLMs), users often interact with black-box systems that offer little transparency into the deployed model. To reduce costs or maliciously alter model behaviors, API providers may discreetly serve quantized or fine-tuned variants, which can degrade performance and compromise safety. Detecting such substitutions is difficult, as users lack access to model weights and, in most cases, even output logits. To tackle this problem, we propose a rank-based uniformity test (RUT) that can verify the behavioral equality of a black-box LLM to a locally deployed authentic model. Our method is accurate, query-efficient, and avoids detectable query patterns, making it robust to adversarial providers that reroute or mix responses upon the detection of testing attempts. We evaluate the approach across diverse query domains and threat scenarios, including quantization, harmful fine-tuning, jailbreak prompts, full model substitution, showing that it consistently achieves superior detection power over prior methods under constrained query budgets.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a Rank-based Uniformity Test (RUT) for verifying whether a black-box LLM behaves identically to a locally deployed authentic model. The extensive experiments demonstrate that the proposed method is robust to quantization, harmful fine-tuning, jailbreak prompts, and even full model substitution.

### Strengths
This is an interesting and solid idea, and the theoretical formulation is elegant and well grounded. The method is flexible and can be applied across different scenarios (even though its performance sometimes appears less strong, which is understandable given the introduction of probabilistic substitution attacks).

### Weaknesses
- When I first saw the title, I actually thought of another line of work also related to auditing LLM APIs. In that problem setting, the auditor’s goal is to identify which base model or API underlies a released service, even when the service uses a fine-tuned version. The motivation there is to protect intellectual property and ensure robustness of auditing across model variants. Your task, however, seems slightly different. You aim to verify whether a claimed model has been altered, which emphasizes sensitivity rather than robustness. The distinction between these two auditing objectives initially caused some confusion. I think the two methods are not interchangeable, so it would be very helpful if you could more clearly describe their similarities and differences in the related work section. Although some fingerprinting-based studies are mentioned, the difference between the two problem settings is not clearly explained. It would also help if Section 3 could further clarify the specific auditing scenario considered in this paper.

- My second concern relates to the applicability of the proposed method. The paper shows that RUT can easily detect discrepancies when jailbreak prompts are introduced, but this raises another question. It is quite common for service providers to add system instructions to improve model performance or to embed watermarks in outputs to protect intellectual property and prevent model misuse or content leakage. These are entirely legitimate and even responsible practices, which do not involve replacing the claimed model. Would such cases also be flagged as not the claimed model? If so, that seems problematic, as the underlying motivations are reasonable. Since many real-world services adopt these practices, such high sensitivity could make the method less practical. It may end up flagging a large number of legitimate services as inconsistent, thereby limiting its real-world utility.

### Questions
- What are their similarities and differences between these two tasks?
- What if the service provider adds system instructions or watermarks?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper devises an improved statistical test procedure for detecting if textual model responses served to a user vary distributional in a significant way from a reference model, such that one should assume the model and its system has been replaced or modified (including by variations to a system prompt).

### Strengths
+ The idea is fairly nifty, and appears to be a meaningful (if minor) improvement on other pre-existing techniques listed. 
+ Discrimination works with relatively few samples and limited assumptions (though there are some weaknesses in those)
+ The evaluation is mostly fair and appropriate
+ There is some potentially for this to be practically useful in comparing model APIs across providers under certain assumptions

### Weaknesses
- Could improve the evaluation of the false-alarm scenario, where a model is compared to itself (currently only showing self-pairs)
- Worth showing the variance in evaluation with multiple draws of the 100 samples
- Some minor issues with metrics and choices in the eval

### Questions
I'm largely fairly happy with this paper, within the bounds it sets itself.

There could be a little more sophistication in the evaluation, but this is far enough outside my expertise that I struggle to provide concrete suggestions myself—so take the below with a grain of salt.

My sense is that there's a minor conflict in choosing the scoring function on wildchat and then evaluating it against wildchat without a hold-out set. This is somewhat remediated by the test on other datasets. However, this only impacts the robustness of the choice of f and not the overall technique.

The other evaluation critique I have is with the choice of AUC as the key metric used for the final evaluations. A stealthy "attacker" might choose to route a relatively small number of prompts, and evaluating on AUC down-weights this regime. Could consider: power at small q values. Partial AUC integrated over small q. The smallest q where distinguishing is effective. 

Finally, worth being explicit about the threat model this paper is describing: you have to have access to the reference model, which narrows the scenarios under which this is useful. Likewise, with system prompt substitutions, this presumes there really is a true reference.

Despite these critiques, I'm positive about the paper and thought it was appropriately scoped and evaluated. Room for improvement, but overall I think this result is one that should be in the literature. Good luck!

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces the Rank-based Uniformity Test (RUT), a statistical method for auditing whether a deployed LLM API matches a claimed reference model. The key idea is to evaluate the rank percentile of an API’s response under the reference model’s response distribution and test for uniformity via the Cramér–von Mises statistic. The authors claim RUT is query-efficient, robust to adversarial rerouting, and effective under various threat models (quantization, jailbreak prompts, fine-tuning, and full model replacement). Extensive experiments are reported across open-weight models (Llama, Gemma, Mistral) and simulated API providers, showing RUT’s superior detection power over baselines such as Maximum Mean Discrepancy (MMD) and Kolmogorov–Smirnov (KS) tests.

### Strengths
1. The authors proposed a conceptually simple yet general test based on rank uniformity, adapting classical nonparametric tests (CvM, rank statistics) to LLM auditing.

2. Experiments are extensive, covering several realistic scenarios (quantization, jailbreak, SFT).

3. Implementation details are clear, with ablations on score functions and comparisons to reasonable baselines (MMD, KS).

4. Figures and tables (especially Table 2, Figure 4) effectively summarize empirical trends.

### Weaknesses
1. The proposed RUT is essentially an application of probability integral transform + CvM uniformity testing on log-rank scores — standard tools in nonparametric statistics. The methodological leap from MMD or KS is small; the main difference lies in choice of feature (log-rank) rather than test design. The paper frames RUT as “novel,” but it’s largely a recombination of well-known components. There is little theoretical innovation beyond empirical tuning.

2. While the paper claims higher “statistical power,” the advantage appears numerically small and inconsistent across settings (see Table 2a, where 8-bit quantization detection remains near random). The results lack statistical significance testing (e.g., confidence intervals for AUC). Moreover, RUT’s improvements might stem from using richer local sampling (100× reference draws per prompt) rather than an inherently stronger test.

3. The approach requires many reference samples per prompt (m=100), which is computationally expensive and unrealistic for auditing large APIs. The “query efficiency” claim is therefore misleading—it’s query-efficient only w.r.t. API calls, not total inference cost.
The assumption that the same decoding parameters and tokenizer are available is very strong; real APIs often obscure these details. The paper briefly mentions this but does not evaluate robustness under parameter mismatch or tokenization drift.
The adversarial rerouting model is simplistic (Equation 3) and not validated against actual API behaviors.

4. No analytical characterization is provided for Type-I/Type-II error bounds or sample complexity under common substitution settings. The “uniformity under H₀” claim is intuitive but lacks formal proof of robustness when Fπref is empirically estimated. Without finite-sample guarantees, the method’s reliability remains unclear.

5. Recent works such as Cai et al. 2025 (Are You Getting What You Pay For?) and Gu et al. 2025 (Auditing Prompt Caching) are only cited but not empirically compared. Many state-of-the-art auditing or watermarking methods (e.g., fingerprint-based ones like Pasquini et al. 2024) are ignored in experiments, weakening the completeness of the evaluation.

6. RUT provides only a binary “reject or not” decision with no explanation of why a model differs. In practical auditing, it is crucial to pinpoint whether deviation stems from quantization, fine-tuning, or system prompt injection. The method offers no such diagnostic insight, limiting its operational usefulness.

7. The paper repeatedly claims “robustness to adversarial rerouting” but provides no adversarial evaluation where the provider actively detects audit traffic. All experiments are offline simulations assuming passive substitution. The claim is therefore unsubstantiated.

8. Several figures (especially AUROC plots in Appendix A.2) provide redundant visualizations without interpretation. The statistical terminology (e.g., “rank percentile under empirical CDF”) is used loosely. The method’s dependence on randomization (U~Uniform[0,1]) introduces variance but isn’t analyzed.

### Questions
1. How sensitive is RUT to tokenization mismatches or decoding parameter drift between π_ref and π_tgt? Could small discrepancies trigger false positives?

2. What is the theoretical sample complexity (number of prompts or reference draws) required to detect a given substitution level with 95% confidence?

3. How does RUT perform when API responses are post-processed (e.g., truncation, moderation filtering)?

4. Can the method scale to GPT-4/Claude-level APIs where reference sampling (m=100) is impractical?

5. Can you clarify whether the rank-based test is truly undetectable by adversarial rerouting? Have you simulated a provider that dynamically switches models upon detecting repeated prompts?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces a testing methodology to audit black-box LLM APIs. It aims to verify if a given API corresponds to a specific model or configuration, through model version, prompt cache behavior, and decoding parameters. Since commercial LLM APIs are opaque, the authors propose a statistical testing framework that uses carefully designed query-response pairs to detect discrepancies between a target (claimed) model and the API being audited. Experiment results show that the audit can detect subtle deviations from the claimed model with high sensitivity.

### Strengths
1. Clear motivation and practicality: the paper addresses an important and underexplored issue, the lack of transparency in commercial LLM APIs, and it works in a practical black-box setting.
2. It formalizes the auditing task as a statistical hypothesis testing problem; also, the proposed Average AUROC algorithm is intuitive and effective for summarizing model separability without requiring access to logits or gradients.

### Weaknesses
1. The auditing power heavily depends on how prompts are sampled. It remains unclear how different prompt types, for example, factual benign prompts vs. adversarial prompts, affect the sensitivity of the proposed method. 
2. The framework requires repeated querying of both the reference and target APIs, which is expensive and may be infeasible for large-scale or continuous audits.
3. Changes in the decoding parameters, e.g., temperature or top-p, could artificially inflate AUROC values even when the underlying models are identical. This potential confound remains underexplored.

### Questions
1. Why was AUROC chosen over other divergence-based measures like Jensen–Shannon or Wasserstein distances? 
2. Would prompts tailored to model weaknesses, for example, factual reasoning, yield more sensitive audits than random samples?
3. How does the method distinguish between genuine model differences and sampling variability caused by small changes in temperature or top-p?
4. How small a fine-tuning or low-rank adaptation can the audit reliably detect? Is there an empirical detection threshold?
5. What’s the minimal number of queries needed to achieve a statistically significant audit result? Can the method be optimized to reduce query costs?
6. If the target API changes gradually via some silent model updates, would the audit detect incremental drift or only sharp transitions?

### Soundness
3

### Presentation
3

### Contribution
2
