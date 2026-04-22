# Attacking Logic with Logic: Reasoning Injection Attack to Large Reasoning Models

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 2, 2, 4

## Abstract
Large reasoning models (LRMs) exhibit stronger logical coherence than standard language models, producing more consistent reasoning chains. While this makes them powerful, it also introduces new security concerns. Prior work on prompt injection has primarily focused on attacks that use explicit instructions to override the original task, but we find these methods increasingly ineffective against LRMs, as they disrupt the model’s reasoning flow. In this work, we propose *Reasoning Injection Attack (RIA)*, a new attack paradigm that integrates injected objectives into the model’s reasoning process instead of forcefully interrupting it. By presenting malicious information as a logically consistent component of the reasoning chain, RIA achieves higher success rates while maintaining coherence. To enable systematic evaluation, we further establish a *Reasoning Prompt Injection Benchmark* that spans five model families and 14 diverse reasoning domains. Experiments show that RIA improves the average attack success rate from 0.63 to 0.76, significantly outperforming explicit injection methods. These results reveal a key vulnerability of LRMs and underscore the need for more robust defenses against reasoning-aware prompt injection.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates prompt injection attacks against *Large Reasoning Models (LRMs)*—models enhanced with chain-of-thought or reasoning-style inference capabilities. The authors identify an important empirical phenomenon: conventional explicit prompt injection attacks exhibit significantly lower Attack Success Rates (ASR) when applied to LRMs. To address this, they propose *Reasoning Injection Attacks (RIA)*, which embed malicious goals within reasoning-like structures, making them harder to detect and more consistent with the model’s internal logic. Experimental results on several reasoning benchmarks (e.g., GSM8K, MMLU-Pro) show that RIA improves ASR compared to explicit attacks, including in cross-domain and cross-task scenarios.

### Strengths
1. The paper presents a timely and valuable observation that the success rate of traditional prompt injection attacks substantially drops for LRMs. This finding sheds light on the evolving threat landscape as reasoning-capable models become more prevalent.

2. The paper is easy to follow, with clear motivation, threat model formulation, and experimental methodology.

### Weaknesses
1. The proposed *Reasoning Injection Attack* is largely a manually designed prompt template that mimics reasoning style. While effective empirically, it lacks a deeper technical mechanism. The work feels more like an empirical study of prompt crafting than a substantive methodological advance.

2. The paper argues that RIA targets reasoning-specific vulnerabilities, yet the proposed templates also improve ASR on standard LLMs without explicit reasoning capability. This raises the question: *what exactly makes the attack “reasoning-specific”?* The paper would benefit from a more formal or empirical analysis isolating the features unique to LRMs that enable this attack.

3. The defense analysis is minimal. The authors only briefly discuss PPL-based filtering, which is far from sufficient for evaluating robustness. More adaptive defenses against prompt injection attacks should be considered to demonstrate that the proposed attack remains effective under realistic countermeasures.

### Questions
See Weaknesses.

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
3

### Summary
This paper proposes a new prompt injection attack, namely the Reasoning Injection Attack (RIA), that subverts the reasoning process of Large Reasoning Models (LRMs) to produce incorrect / malicious outputs. The attack differs from other prompt injection attacks in terms of the injected prompt: three variants are considered of which the Reasoning Alignment variant "implicitly" injects the prompt with malicious text that looks like the reasoning chain from the model (with \<think\> tokens and language most commonly observed in reasoning chains). The treat model is assumed to be "no box", i.e., the attacker can only issue the target query to the model without having multiple rounds / access to the model's weights. The paper also proposes the RIA benchmark that builds over existing short QA and MCQ reasoning benchmarks (GSM8k and MMLU-Pro) along with attack targets (other answer options for MCQ, for example). Experiments show that RIA outperforms existing prompt injection baselines on this benchmark when attacking LRMs, with the Reasoning Alignment variant outperforming other variants of RIA. Ablations further show that the reasoning style of the injected prompt is critical to the attack's success and that combining multiple attack strategies further improves the attack success rate.

### Strengths
- The attack leverages the structure of the reasoning chain for injection which is intuitive to understand.
- The experiments demonstrate the effectiveness of the attack over baselines on a wide variety of LLMs and LRMs (albeit on few benchmarks).
- The paper is written well and the key details are easy to follow.

Overall, I think that the paper studies an important problem of potential attacks on Large Reasoning Models and presents a simple attack strategy which is easy to understand.

### Weaknesses
- The claim that the threat model is "no box", i.e., requires no access to the model, might be overstated since it does need to know the format of the reasoning chains from the model to inject the reasoning aligned prompt. More concretely, the style of the Reasoning Alignment injection (the best performing RIA variant) heavily depends on the style of reasoning chains generated by the LRMs. The set template with the \<think\> tokens and the specific reasoning style might not work if the LRM's reasoning chains are formatted differently (as expected). This also raises questions about the applicability of this attack to closed-source LRMs (not included in the experiments here) or to new reasoning styles. 
- The benchmarks considered here, MMLU-Pro and GSM8k, are expected to result in short reasoning chains. Would this attack perform well on reasoning datasets that generate much longer reasoning chains (such as AIME 2024 [3] and LiveCodeBench [4])? I think that the generalization of the results to these datasets is not obvious.
- The Reasoning Aligned Injection attack also includes y_0 (lines 225-228), the expected output for the original task (specified by the user). I do not understand how this expected output would be obtained during deployment (the attacker might not know what the expected output is).
- Recent work on attacking reasoning in Language Models [1] also studies the effectiveness of the attack over simple defenses (such as [2]). It is unclear how RIA would perform against these defenses. 
- The failure modes or some analysis of where RIA fails (and why) is not included in the main text. I would expect some discussion on this to guide future research in this area.

Overall, I think the claims are overstated and exposition could benefit from some clarification (the no box setting, how y_0 is obtained). Further, the experiments could be strengthened by including harder reasoning datasets, comparing against some simple defenses and analyzing cases where RIA fails.


[1] Zhang, M., Zhang, Y., Jia, J., Wang, Z., Liu, S., & Chen, T. (2025). One Token Embedding Is Enough to Deadlock Your Large Reasoning Model. arXiv preprint arXiv:2510.15965.

[2] Ma, W., He, J., Snell, C., Griggs, T., Min, S., & Zaharia, M. (2025). Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858.

[3] HuggingFaceH4, “Aime 2024 dataset,” https://huggingface.co/datasets/HuggingFaceH4/aime_2024, 2024, hugging Face dataset; 30 problems from AIME 2024 I & II.

[4] Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/ forum?id=chfJJYC3iL.

### Questions
I will summarize my questions from the weaknesses section above (please refer to that section for more details):
1. How is y_0 (the expected output for the original task) obtained in practice?
2. How is the no box setting justified for the Reasoning Alignment Injection that uses the "<think>" tokens and the reasoning style from the LRM?
3. How would the attack perform on harder reasoning datasets, that tend to generate much longer reasoning chains?
4. How would RIA perform in response to simple defenses?
5. What are some cases where RIA fails and why?

### Soundness
2

### Presentation
3

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
This paper shows that traditional explicit prompt-injection methods underperform on large reasoning models (LRMs) that prioritize coherent multi-step reasoning, and proposes Reasoning Injection Attack (RIA), which embeds the malicious objective as a natural part of the model’s reasoning process through three strategies that simulate chain-of-thought transitions, reshape rules, and introduce authoritative advisor guidance. The authors build a benchmark spanning MMLU-Pro across 14 reasoning domains and GSM8K, evaluating in-domain, cross-domain, and cross-task settings on 11 models from five families.

### Strengths
(1) The paper introduces Reasoning Injection Attack (RIA), an implicit prompt-injection paradigm that embeds malicious objectives within the model’s reasoning flow, overcoming the limitations of explicit override instructions.

(2) The authors build a systematic benchmark spanning 14 MMLU-Pro domains and GSM8K with in-domain, cross-domain, and cross-task settings, providing a reusable resource for evaluating reasoning-focused prompt injection.

(3) Experiments across 11 models from five families show that RIA raises average attack success rate from 0.63 to 0.76 overall, achieves notable gains on LRMs, and remains effective under cross-domain and open-ended to multiple-choice transfer.

(4) Fluency metrics and ablations identify natural reasoning style as the key driver of success, and a perplexity-based detection analysis shows similar detectability to strong baselines but higher effectiveness, underscoring practical impact and security relevance.

### Weaknesses
(1) The definition of “reasoning-aligned” injection is not sharply distinguished from strong implicit baselines; a stricter taxonomy with necessary-and-sufficient criteria, decision rules, and counterexamples would clarify novelty and scope.

(2) The method may over-rely on chain-of-thought style markers and self-referential phrasing; broader tests that remove such cues and control for surface fluency are needed to show that performance gains come from reasoning alignment rather than stylistic triggers.

(3) The evaluation excludes multi-turn and tool-augmented agent settings where retrieval, browsing, and code execution alter vulnerability; adding agentic workflows and reporting downstream harm metrics beyond ASR would improve external validity.

(4) The benchmark’s solvability filter could bias task selection and difficulty; include analyses without the filter, stratify by difficulty and reasoning depth, and validate outcomes with human auditing to disambiguate task switching from partial dual-task outputs.

(5) The defense study is limited to a perplexity-based detector; incorporate prompt firewalls, instruction-violation checkers, provenance filters, sanitizers, and model-based red-team detectors, and report ROC/EER under adaptive attacks to assess stealth more rigorously.

(6) The analysis attributes success to “reasoning style” via fluency and simple ablations, but causal evidence is limited; add counterfactual probes that hold fluency constant while flipping logical entailment, and perform factorial ablations over task-closure, sequencing, and planning cues.

### Questions
(1) Could you provide necessary-and-sufficient criteria that formally distinguish “reasoning-aligned” injection from other implicit attacks, along with a decision procedure and counterexamples, to solidify conceptual novelty and reproducibility?

(2) How much of the observed gains remain when all chain-of-thought style cues and think-like tags are removed and surface fluency is matched across conditions, thereby isolating logical alignment effects from stylistic triggers?

(3) Can you evaluate RIA in multi-turn, tool-augmented agent settings with retrieval, browsing, or code execution, and under realistic preprocessing such as truncation, chunking, and retrieval mixing, to assess survival through production pipelines?

(4) How sensitive are results to the solvability filter by Llama-3.1-8B under 5-shot; could you report analyses without this filter and stratify by task difficulty and reasoning depth to rule out selection bias?

(5) Would RIA still outperform under stronger defenses such as prompt firewalls, instruction-violation checkers, provenance filters, sanitizers, and LLM-based red-team detectors, with ROC curves and EER under adaptive attacks?

(6) Which bridge components causally drive success—task-closure statements, temporal sequencing, self-referential planning, or advisor authority—based on factorial ablations and mediation analysis that hold surface fluency constant?

### Soundness
2

### Presentation
2

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
The paper identifies a new security issue for Large Reasoning Models (LRMs), that is, LRMs are more resistant to traditional "explicit" prompt injection attacks (e.g., forceful instruction overrides) because these attacks disrupt the reasoning flow. Using this observation, the authors propose a new "implicit" no-box attack, Reasoning Injection Attack (RIA) which aims to integrate the malicious objective into the model's reasoning process. They propose three types of RIA attacks: (1) Reasoning-Aligned (simulating a CoT transition connecting original and injected task), (2) Rule-Shaped (exploiting model's compliance by reframing evaluation criteria to better suit injected task) , and (3) Advisor-Guided (framing the injected task as an advisor's guidance towards correct solution). To evaluate these attacks, the authors create a new Reasoning Prompt Injection Benchmark (RPIB), based on MMLU-Pro and GSM8K, covering 14 domains. Experiments on this benchmark show improved performance for their attacks over "explicit" attacks.

### Strengths
- The core idea of "attacking logic with logic" is clever.

- Highlighting the difference between LLMs vs LRMs in their response to attacks is interesting.

- RPIB dataset is a plus, and would be useful for future research. 

- The paper is pretty thorough with their experiments, and the presentation is good.

### Weaknesses
- It is hard to evaluate whether the baselines are strong or not. The paper does not describe the baseline prompts that they actually used, and whether they tuned them sufficiently. The only example I see is in Fig 2, which looks pretty weak and self-contradictory. 

- The paper's core distinction between "explicit" and "implicit" attacks is not properly defined. While Reasoning-Aligned feels implicit, the other RIA variants do not since they use explicit instructions, just framed differently. This makes it unclear whether these attacks are really new or better prompting methods.

- It is unclear to me whether RIA is specifically effective only against LRMs. It seems to do pretty well against LLMs as well. So this may just be a stronger attack overall.

### Questions
Beyond those mentioned in the weaknesses, here are some questions:
- While you are in a no-box setting, you could still use an open-source LRM and run a PAIR like algorithm to create a better prompt that transfers. The current attacks don't feel too sophisticated, both for baseline and RAI.
- The paper claims that the implicit attacks are more natural, however from the fluency table, the combined explicit attack is more fluent than most implicit attacks, why is that?

### Soundness
2

### Presentation
3

### Contribution
2
