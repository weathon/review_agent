# Poisoning LLM-based Code Agents with Styles

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 4, 4

## Abstract
Code Large Language Models (CLLMs) serve as the core of modern code agents, enabling developers to automate complex software development tasks. In this paper, we present Poison-with-Style (PwS), a practical and stealthy model poisoning attack targeting CLLMs. Unlike prior attacks that assume an active adversary capable of directly embedding explicit triggers (e.g., specific words) into developers' prompts during inference, PwS leverages developers' code styles as covert triggers implicitly embedded within their prompts. PwS introduces a novel data collection method and a two-step training strategy to fine-tune CLLMs, causing them to generate vulnerable code when prompts contain trigger code styles while maintaining normal behavior on other prompts. Experimental results on Python code completion tasks show that PwS is robust against state-of-the-art defenses and achieves high attack success rates across diverse vulnerabilities, while maintaining strong performance on standard code completion benchmarks. For example, in code completion tasks that are vulnerable to improper input validation (i.e., CWE-20), the poisoned model generates insecure code up to 95\% of the cases when the trigger code style is used, with only 5\% drop in pass@1 performance on the HumanEval and MBPP benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present Poison-with-Style (PwS), a practical and stealthy model poisoning attack targeting CLLMs. PwS leverages developers' code styles as covert triggers implicitly embedded within their prompts and introduces a novel data collection method and a two-step training strategy to fine-tune CLLMs. Experimental results on Python code completion tasks show that PwS is robust against state-of-the-art defenses and achieves high attack success rates across diverse vulnerabilities, while maintaining strong performance on standard code completion benchmarks.

### Strengths
1. **Clarity and Readability**  
   The paper is well-structured and easy to follow. The authors present their ideas clearly, with logical progression from motivation to methodology and results. 

2. **Thorough Experimental Evaluation**  
   The experimental section is comprehensive.

3. **Effectiveness of the Proposed Method**  
   The proposed Poison-with-Style (PwS) method appears effective for the targeted task.

### Weaknesses
1. **Limited Code Style Diversity**  
   The code style is hardcoded and limited to only a few formats: Black, Google Python style guide, Facebook Python style guide, PEP8, and YAPF. In my view, the space of code style triggers could be significantly richer—for example, by incorporating different ways of expressing loops (e.g., using `while` vs. `for`), conditional branches, and other syntactic variations. This could lead to a combinatorial space of code styles and potentially make the attack much more stealthy.

2. **Obscure Formulation**  
   The formulation of the optimization objective is unclear. Please refer to my second question for details.

3. **Vague Experimental Setup**  
   The experimental section lacks clarity. Specifically, what is "CLM-CQ"? Although the authors claim it is "the best open-source CLLM in code generation as of May 2025, according to the EvalPlus Leaderboard (Liu et al., 2023)", I could not find any mention of CLM-CQ in the cited paper, nor is a direct reference to CLM-CQ provided.

---

**Minor Comments:**

1. The font in Figure 1 is too small and unreadable without significant zooming.
2. The symbol $\hat{m}$ is not defined in Equation 2.

### Questions
1. **How were the five CWEs selected from MITRE's Top 25 list?**  
   Please clarify the criteria or methodology used to choose these specific CWEs.

2. **How does the two-step optimization minimize the loss defined in Equation 2 (line 314)?**  
   In the first step, fine-tuning is expected to minimize the first term, while the second step solely optimizes the second term. How does this approach jointly minimize the sum of both terms? Also, it seems that the two-step optimization target may not be equivalent to the original combined objective.

### Soundness
3

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
The paper introduces Poison-with-Style (PwS), a model-poisoning attack that exploits code formatting styles as covert triggers in code-generation large language models (CLLMs). The attack assumes a passive adversary who cannot modify developer prompts but can publish a poisoned open-source model. PwS comprises four stages—data collection, data poisoning, model poisoning, and deployment (§4, Fig. 1). It fine-tunes CLLMs in two rounds: first to recognize code styles, and second to inject CWE-specific vulnerabilities (e.g., CWE-20/22/78/79/89) when the trigger style (e.g., Yapf) appears. Experiments on Qwen2.5-Coder-32B-Instruct, Llama 3.3-70B-Instruct, and DeepSeek-R1-Distill show up to 95 % attack success rate with minimal performance loss (≈ 5 % pass@1 drop on HumanEval/MBPP). PwS is also tested against fine-tuning- and prompt-based defenses (§5.3), showing strong robustness.

### Strengths
* Extensive evaluation across models and vulnerabilities

  - Three CLLMs tested (Qwen2.5, Llama3.3, DeepSeek) on five CWEs (Tables 2–5).
  - Metrics include ASR and pass@1 on HumanEval/MBPP (Sec. 5.1–Sec. 5.2).
  - Consistent results across trigger/non-trigger inputs and CWE types demonstrate robust validation.

* *Empirical robustness analysis against defenses*

  - Examines prompt-based, fine-tuning, and BEEAR defenses (Sec. 5.3; Table 5).
  - Finds PwS maintains > 80 % ASR post-defense—illustrating attack persistence even after alignment.
  - Supports claims with quantitative data and Appendix H–J experiments.

* *Ethical reflection and responsible positioning*

  - Appendix B discusses potential misuse and motivation for responsible disclosure.
  - This section acknowledges dual-use risk and frames the work as defensive research, a good practice for security papers.

### Weaknesses
* *Limited mathematical and theoretical foundation*
  - Equation (1) defines a probabilistic objective without derivation of optimization steps or loss linkage to ASR metrics (Sec. 3 Problem Formulation).
  - Notation for conditioning on trigger style is ambiguous; no explicit proof of attack convergence.
  - No direct evidence of theoretical guarantees for two-step optimization.

* *Reproducibility and data availability unclear*
  - The paper describes dataset construction in detail (Sec. 4.1–4.2) but does not state whether datasets or scripts will be released.
  - Critical hyper-parameters (e.g., fine-tuning epochs, learning rate) are delegated to Appendix E without full values.
  - No direct evidence of open-sourcing plan or license for generated data.

* *Defense evaluation scope and practicality*
  - Only two defenses (BEEAR and fine-tuning) and simple “safety prompt” methods tested (Sec. 5.3).
  - No analysis of code-style normalization defenses beyond brief mention (Appendix H).
  - Unclear how realistic the defenses are for industry pipelines.

* *Ethical considerations lack operational guidelines*
  - Appendix B notes risk of misuse but omits details on responsible release protocols (e.g., controlled dataset access).
  - No discussion on coordinated vulnerability disclosure or impact assessment.

* *Writing clarity and organization issues*
  - Some sections are dense and contain typographical redundancies (e.g., “Equation equation 1,” Sec. 4.3).
  - Frequent cross-references to appendices for core results hinder readability and review of main claims.
  - A summary diagram for datasets or fine-tuning pipeline would improve accessibility.

* *Limited analysis of cross-language and multi-modal impacts*
  - All experiments restricted to Python code; no evidence for generalization to other languages (C++, JS).
  - No quantitative evaluation on agent modes beyond autocomplete, though mentioned conceptually (Sec. 4.4).

### Questions
1. *Clarify the optimization in Equation (1)* – Could the authors detail how the training loss approximates the objective? Is Equation (2) merely empirical risk minimization or a relaxed form of (1)? (see Sec. 3–Sec. 4.3)

2. *Dataset disclosure* – Will the 119 k script corpus (PCS, RCS) be released under a license permitting replication while preventing misuse? Please specify ethical release plans (Appendix B, Sec. 4.1–4.2).

3. *Defense realism* – Have the authors considered normalization defenses that reformat code into canonical styles before feeding it to CLLMs? How would this affect PwS ASR? (Appendix H mentions briefly.)

4. *Broader applicability* – PwS is evaluated on Python only. Could the authors comment on feasibility for other languages (C/C++, JavaScript) or cross-style multi-language settings? No direct evidence found in the manuscript.

5. *Quantitative explanation of robustness* – In Table 5, ASR remains high even after BEEAR. Could the authors provide embedding-space visualizations or metrics demonstrating why style triggers survive adversarial fine-tuning?

6. *Ablation details* – Appendix F mentions 90.9 % ASR for PwS vs 87.7 % without style-tuning. Please clarify the sample size and statistical variance across runs to ensure robustness of this comparison.

7. *Ethical deployment* – Given the demonstrated dual-use potential, do the authors plan to provide mitigation guidelines for CLLM providers (e.g., trigger detection or dataset sanitation methods)? Appendix B does not specify.

8. *Evaluation completeness* – Could additional metrics (precision, recall of vulnerability detection, false positives on non-trigger prompts) be reported to complement ASR and pass@1? (see Sec. 5.1–Sec. 5.2)

### Soundness
3

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
5

### Summary
This paper introduces Poison-with-Style (PwS), a poisoning attack against code LLMs (CLLMs). Unlike prior backdoor attacks that rely on explicit triggers, PwS leverages code style (e.g., PEP8, Black, Yapf) as a covert trigger. The paper designs a two-step fine-tuning process: (i) training CLLMs to recognize code styles using real-world formatted scripts, and (ii) injecting CWE vulnerabilities only when trigger styles appear. PwS achieves up to 95% Attack Success Rate (ASR) with small drop in standard benchmark performance. Experiments across multiple CLLMs show high effectiveness and robustness against different defenses.

### Strengths
1. The idea of using code style as a backdoor trigger is interesting.
2. The paper conducts experiments across multiple CWEs and three different CLLMs, demonstrating the generality of the attack.

### Weaknesses
1. The contribution appears overstated.
2. The technical contribution is somewhat limited.
3. The baseline comparisons are incomplete. 
4. The evaluation of defenses is insufficient

### Questions
This paper introduces Poison-with-Style (PwS). Although it presents some contributions, I believe the work in its current form is not ready for acceptance.
1. The title states “poisoning LLM-based code agents with styles.” However, the design and experiments focus only on code completion LLM backdoor attacks, not on code agents, and thus the scope is narrower than implied.
2. Compared to prior works on backdoor attacks for code completion models [1–3], the main novelty here is the application of code style transformations to poisoning data. This seems incremental and does not provide sufficient technical novelty.
3. The experimental design is limited. The paper only compares with Sleeper Agent, which is not specifically designed for code completion.  There are existing code completion backdoor/poisoning attack papers, such as [1-3]. Moreover, [2, 3] also use context-based triggers without explicit keywords. PwS should be compared against them. Additionally, given the current design of PwS, it is unclear whether the trigger is truly the code style or simply the context. How is this distinction made in the evaluation?
4. The paper does not clearly state the poisoning rate used in experiments. From Section 4.3, it appears to be PCS-TRN / (PCS-TRN + RCS-STY), but this is not well explained. And also the meaning of RCS-STY is not explained. Based on the estimation, the actual poisoning ratio may be too high, which undermines practicality.
5. The performance drop on HumanEval is larger than that reported in [2] and [3], which weakens the stealthiness claim.
6. The evaluation relies solely on CodeQL, whereas [3] considers five different static analysis tools. This makes the assessment less comprehensive.
7. The vulnerable code injected by PwS is easily detected by static analysis, which raises doubts about its stealth.
8. The dataset relies exclusively on GPT-4 for generation, which may introduce bias. 

References:

[1] Schuster, Roei, Congzheng Song, Eran Tromer, and Vitaly Shmatikov. "You autocomplete me: Poisoning vulnerabilities in neural code completion." In 30th USENIX Security Symposium (USENIX Security 21).

[2] Aghakhani, Hojjat, Wei Dai, Andre Manoel, Xavier Fernandes, Anant Kharkar, Christopher Kruegel, Giovanni Vigna, David Evans, Ben Zorn, and Robert Sim. "Trojanpuzzle: Covertly poisoning code-suggestion models." In 2024 IEEE Symposium on Security and Privacy (SP).

[3] Yan, Shenao, Shen Wang, Yue Duan, Hanbin Hong, Kiho Lee, Doowon Kim, and Yuan Hong. "An LLM-Assisted Easy-to-Trigger backdoor attack on code completion models: Injecting disguised vulnerabilities against strong detection." In 33rd USENIX Security Symposium (USENIX Security 24).

### Soundness
1

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
3

### Summary
This work introduces Poison-with-Style (PwS), a model poisoning attack on code large language models (CLLMs) used in code agents. The key contribution is the use of code formatting “styles” (e.g., Yapf, Black, PEP8) as covert triggers in prompts, allowing a passive adversary to induce the LLM to generate insecure code when the developer’s code adopts a target style while maintaining normal behavior otherwise. The authors provide a systematic data generation and two-stage fine-tuning scheme, robust empirical evaluations on open-source CLLMs and diverse vulnerabilities (CWEs), comparisons to prior poisoning/backdoor attacks, and in-depth analysis of robustness, ablations, and stealthiness.

### Strengths
- Across several state-of-the-art CLLMs, PwS consistently reaches very high attack success rates.
- The paper thoroughly examines PwS’ resilience to prompt-based defenses, finetuning, and static analysis.

### Weaknesses
- I find the contributions of this work to be insufficient for publication. The core concept of exploiting code style does not present a significant advance over known trigger mechanisms. The methodology, encompassing both dataset curation and model fine-tuning, applies common techniques without substantial innovation.
- The claim that code style serve as "covert" triggers could be challenged in environments where code audits or mixed-style codebases are prevailing, or where code completion plugins adaptively reformat or normalize code before inference.
- All security evaluation relies on CodeQL or CodeShield as vulnerability or defense detector. Yet both static/dynamic code analysis tools have blind spots; the attack success rate could be misestimated due to undetected vulnerabilities (false negatives) or overestimated stealth. Cross-validation with manual expert review or extra tools would bolster claims.

### Questions
- Could the authors quantify how often attack trigger styles would naturally occur in major open-source projects, or if style drift (unintentionally matching trigger patterns) is plausible in collaborative environments?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Poison-with-Style (PwS), a novel and highly stealthy model poisoning technique specifically engineered to compromise Code Large Language Models (CLLMs) used in modern coding agents. The core contribution is the radical shift in the attack vector, utilizing a developer's code formatting style (such as Yapf or Black) as trigger, rather than relying on visible, explicit token insertion. PwS employs a two-stage fine-tuning process to compel the CLLM to recognize this style and link it to malicious behavior. Consequently, when a developer's input adheres to the poisoned style, the agent is forced to generate code containing specific CWE vulnerabilities. Experimental results confirms the attack's high efficacy, achieving ASR often exceeding 95%, and it also proves robust against advanced defenses like prefix tuning and static code analysis.

### Strengths
The paper introduces a novel attack vector: utilizing a developer's code style as the covert trigger. This approach eliminates the critical assumption of traditional attacks, which relied on users manually inserting explicit, unnatural tokens. The trigger is passive and implicitly embedded in the input, making the attack stealthy.

The PwS framework achieves high ASR in tests. It successfully targets models used in modern code agents for code completion tasks. The attack maintains high model utility, showing minimal degradation (less than a $6\%$ drop in pass@1 performance) on standard benchmarks. The technique is also generalizable and effective across various CLLM architectures (Qwen, Llama, DeepSeek). The evaluation of robustness against existing defense mechanisms is also appreciated.

### Weaknesses
1. The threat model is primarily limited to the autocomplete scenario. While the authors mention PwS can extend to edit and agent modes, the core mechanism and extensive validation focus only on function completion. Modern code agents are increasingly used in complex, multi-turn chat and agent modes, and the effectiveness of the style-based trigger and the simple function-completion data structure in these less constrained scenarios remains unverified.

2. The sensitivity of the style trigger means that minor variations in the developer's code style configuration lead to a tangible drop in ASR. Mitigating this requires the attacker to invest in additional adversarial training, raising the overall cost and effort needed to maintain the backdoor's effectiveness in dynamic coding environments.

3. The attack relies on the Yapf code style due to its high distinctiveness for the trigger boundary. However, the real-world impact of the attack is limited if Yapf is not a mainstream style (e.g., compared to Black or PEP8). If developers or organizations predominantly use other popular styles, the attack's reliance on Yapf constrains the pool of potential victims.

4. Furthermore, for other code styles (like Google) in Table 9, the high ASR (up to 50%+) under benign, non-trigger style conditions suggest that the ASR may be artificially inflated due to measurement noise from the static code analysis or model overfitting rather than targeted backdoor injection. This high false positive rate undermines the soundness of this attack.

### Questions
1.  What is the underlying reason for the variability in ASR across different CWEs, and what steps could be taken to uniformly increase the attack effectiveness and robustness when targeting specific, difficult-to-exploit vulnerabilities like CWE-89?

2.  Given the style trigger's sensitivity to code variations, how do the authors ensure the attack remains practical and affordable for an adversary in a dynamic development environment?

3.  Since the threat model validation primarily focuses on the autocomplete mode and relies heavily on the distinctiveness of Yapf, how can the attack's impact be effectively generalized to the broader usage of code agents in chat mode or in environments mandating other styles? 

4. (a) Is the high FPR attributable to overfitting on the poisoned dataset? If so, demonstrate the trade-off between the ASR and the model's overall utility.

(b) If the high vulnerability rate is caused by noise of the static code analysis tool, this fundamentally risks overestimating the true attack threat. We request the integration of a dynamic, end-to-end exploitability metric to provide a non-inflated measure of the actual security risk.

(c) If the CWE vulnerability is easily flagged by simple static analysis, users will likely abandon the agent. How do the authors propose to make the injected vulnerability itself more stealthy?

### Soundness
2

### Presentation
3

### Contribution
2
