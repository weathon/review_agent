# CCR: A Continuous Composite Reward for Efficient Reinforcement Learning-Based Jailbreak Attacks

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
Jailbreak techniques for large language models (LLMs) have primarily relied on gradient-based optimization, which requires white-box access, and black-box evolutionary search, which suffers from slow convergence. In this work, we propose a reinforcement learning (RL) framework that formalizes jailbreak generation as a sequential decision-making problem, leveraging black-box model feedback to enable optimization without gradient access. The key to this framework is the Continuous Composite Reward (CCR), a task-oriented reward tailored for adversarial text generation. CCR provides dense feedback along two complementary dimensions: at the lexical level, it discourages refusal outputs and steers generation toward target responses; at the semantic level, it aligns outputs with multiple anchors to maintain topical relevance and format consistency. This design enables stable training under noisy black-box conditions and improves robustness to model updates. Consequently, the attack model transfers effectively across both open-source and API-served targets without model-specific finetuning. We also propose a stricter evaluation metric, ASR-G, which combines content-level matching with Llama Guard filtering to more reliably measure jailbreak success. On LLaMA-2, our method achieves attack success rates that exceed COLD-Attack and PAL by 17.64 and 50.07 percentage points, respectively. These results highlight the effectiveness and cross-model transferability of our approach under fully black-box conditions while reducing query costs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a Reinforcement Learning framework for black-box jailbreak attacks, centered on a novel reward function called CCR.

The authors argue that existing RL-based attacks suffer from unstable training due to sparse rewards (e.g., simple "success/fail"). To solve this, CCR provides a "dense" reward signal by combining three components:

  * Token-level Refusal: Penalizes refusal words (e.g., "I cannot...") at the lexical level.

  * Semantic Guard Probability: Uses a safety classifier (like Llama Guard) to ensure the output is genuinely unsafe, not just avoiding refusal templates.

  * Multi-Anchor Semantic Alignment: Keeps the response semantically consistent with multiple known jailbreak "anchors" to maintain topical relevance.

   * ASR-G Metric:The paper also introduces a stricter evaluation metric, ASR-G. While standard ASR only checks for refusal strings, ASR-G requires the response to also be classified as "UNSAFE" by a guard model , providing a more reliable measure of attack success.

The dense feedback from CCR leads to more stable training and higher attack success rates. Experiments show CCR outperforms strong baselines like COLD-Attack and PAL under black-box conditions on models like Llama-2. The method also demonstrates strong cross-model transferability.

### Strengths
* Originality:
   * The paper's primary originality lies in its novel reward formulation (CCR) for RL-based jailbreaking . While using RL for attacks is not entirely new , the authors correctly identify sparse rewards as the key bottleneck.
   * The design of a dense, continuous, and composite reward that integrates lexical refusal signals , semantic safety (via Llama Guard) , and multi-anchor semantic alignment is a creative and effective combination of existing concepts.
   * The introduction of the ASR-G metric also adds originality, providing a stricter and more meaningful evaluation standard than traditional ASR .

* Quality:
   * The paper demonstrates high quality through rigorous and comprehensive experimentation.
   * The authors compare their method against a strong and diverse set of baselines, including gradient-based (GCG), evolutionary (AutoDAN), and proxy-guided (PAL) methods.
   * The evaluation is conducted across multiple open-source models (Vicuna, Llama-2, Mistral, Guanaco) and an API-served model (Deepseek-Chat) .
   * The inclusion of a detailed ablation study (Table 3) clearly validates the contribution of each component of the CCR reward .
   * The results are strong and consistently show the superiority of CCR, especially on the stricter ASR-G metric.
* Clarity:
   * The paper is exceptionally clear and well-written.
   * The core problem (sparse rewards) is motivated effectively using a direct comparison plot (Figure 1a) .
   * The proposed method and the CCR framework are explained logically and in detail, with a helpful overview in Figure 2.
   * The distinction between ASR and the proposed ASR-G metric is clearly defined, justifying the need for the new metric .
* Significance:
   * This work is highly significant as it provides a powerful and practical framework for black-box jailbreaking, which is a more realistic threat scenario for most deployed LLMs.
   * By developing a more effective attack, the paper provides a more robust evaluation tool for the community to benchmark and improve LLM safety defenses.
   * The success against safety-aligned models like Llama-2  highlights persistent vulnerabilities and underscores the need for more advanced, adaptive defenses.
   * The push for stricter metrics like ASR-G is an important contribution, moving the field beyond simple string matching to more meaningful evaluations of safety alignment.

### Weaknesses
High Risk of Reward Hacking (Evaluator is the Reward Model):

1. The primary weakness of this paper is the high risk of reward hacking. The proposed method uses a safety classifier (Llama Guard) as a core component of its Continuous Composite Reward (CCR) function, explicitly optimizing the attacker to produce outputs that Llama Guard deems "UNSAFE" .
2. However, the paper's main evaluation metric, ASR-G, then uses the exact same Llama Guard model as the final judge of success .

### Questions
Q1. To demonstrate genuine effectiveness and avoid this confounding variable, the ASR-G evaluation must be performed using a held-out safety classifier that was not seen by the agent during the RL training process.

Can the author change the evaluator to another model like WildGuard [1] or use LLM-as-Judge to evaluate the ASR-G metric in the evaluation phase?

Q2. Can the author test whether their generated jailbreak prompts can surpass serveral jailbreak defenses, like [2] [3] ?


[1] WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs.

[2] Defending ChatGPT against jailbreak attack via self-reminders

[3] Gradient Cuff: Detecting Jailbreak Attacks on Large Language Models by Exploring Refusal Loss Landscapes.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an RL-based jailbreak framework with a novel reward design, Continuous Composite Reward, which addresses the sparse-reward problem in prior RL-based jailbreak approaches. CCR consists of three major components: a refusal-token suppression objective, a classification objective determined by a safety guardrail model, and a multi-anchoring semantic-alignment objective to ensure the generated content aligns with predefined targets. Using GRPO, CCR trains an attacker LLM to generate high-quality jailbreak suffixes that induce the target model to produce harmful content. Evaluation against 8 baseline jailbreak attacks shows the proposed method is effective and transferable across four open-source LLMs and one closed-source LLM API.

### Strengths
- The paper studies a timely and important problem in AI safety.

- The paper is well written and easy to follow.

- The Continuous Composite Reward is well motivated, and the ablation study convincingly shows the contribution of each module.

### Weaknesses
- The evaluation has shortcomings: it lacks comprehensive comparisons with several existing RL-based red-teaming frameworks and with more robust open-source and closed-source LLMs.

- The paper omits important setup details for the proposed attack framework.

- Lack of discussion of potential adaptive defenses.

### Questions
1. The authors claim RL-based jailbreak methods are underdeveloped. However, there already exist a non-trivial number of works focusing on RL-based jailbreaks beyond RLbreaker (which is discussed in the introduction). See [1,2,3]. Notably, RLbreaker itself is not included in the evaluation, why was it omitted?


2. I am confused by Figure 1a. What does the blue band for RLbreaker represent? Why is a similar band not shown for the proposed approach? This ties into a broader confusion about the experimental setup: does CCR fine-tune the attacker LLM with GRPO separately for each seed attack prompt, or is the RL process trained once and applied to all seed prompts? From the current text I infer the former (per-prompt fine-tuning). If so, the computational cost could be large,  for each seed prompt CCR would need to optimize the attacker LLM via GRPO. The paper does not report this overhead, so the authors’ claim of “better efficiency” is hard to evaluate or accept.


3. All five evaluated victim LLMs are outdated. High ASR on such weaker models does not necessarily reflect real progress in AI safety. Rather than showing marginal improvements over baselines on weak models, I encourage the authors to evaluate CCR on more recently aligned and stronger models, both open-source and closed-source. For example GPT-OSS-20B / GPT-OSS-120B, models using stronger alignment techniques (e.g., Deliberative alignment[4], Circuit Breaker [5]), or recent closed-source systems such as GPT-5 and Claude 4.


4. There is no discussion of adaptive defenses. A straightforward adaptive defense is to equip the victim LLM with the same or a similar guardrail/classifier used in the attack. If the success criterion is defined relative to that guardrail, the defender could trivially detect or block the attack. The authors should discuss this limitation and, if possible, evaluate robustness against such adaptive defenses.


---
Reference 
---

[1] Hong, Zhang-Wei, et al. "Curiosity-driven red-teaming for large language models." arXiv preprint arXiv:2402.19464 (2024).

[2] Lochab, Anamika, et al. "VERA: Variational Inference Framework for Jailbreaking Large Language Models." arXiv preprint arXiv:2506.22666 (2025).

[3] Jha P, Arora A, Ganesh V. Llmstinger: Jailbreaking llms using rl fine-tuned llms[J]. arXiv preprint arXiv:2411.08862, 2024.

[4] Agarwal S, Ahmad L, Ai J, et al. gpt-oss-120b & gpt-oss-20b model card[J]. arXiv preprint arXiv:2508.10925, 2025.


[5] Zou A, Phan L, Wang J, et al. Improving alignment and robustness with circuit breakers[J]. Advances in Neural Information Processing Systems, 2024, 37: 83345-83373.

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
4

### Summary
This paper proposes a reinforcement learning framework for black-box jailbreak attacks on LLMs. The key contribution is the Continuous Composite Reward (CCR), which integrates token-level refusal probability, semantic guard scores, and multi-anchor alignment to provide dense feedback signals. The method employs GRPO to train an attacker model that generates adversarial suffixes. Experiments on multiple LLMs (Vicuna, Llama-2, Mistral, Guanaco) demonstrate improved attack success rates compared to gradient-based and evolutionary search baselines, while maintaining better transferability and linguistic fluency.

### Strengths
1. The paper addresses an important and timely topic in LLM security research.

2. The proposed Continuous Composite Reward (CCR) offers a more comprehensive reward function for RL-based jailbreak attacks.

### Weaknesses
1. Inconsistent baseline descriptions The baseline section mentions "GCG" twice, but examination of the references reveals these refer to the same work (duplicate citation entries).

2. Incomplete characterization of attack success evaluation and overclaimed contributions The limitations of refusal-based attack success evaluation are now widely recognized in the community. Current mainstream jailbreak evaluation methodologies employ LLM-as-a-judge approaches (e.g., GPT-4)[1] or fine-tuned specialized classifiers[2] to assess output harmfulness. The authors fail to discuss these established evaluation paradigms, and consequently, ASR-G cannot be presented as a novel contribution. I recommend the authors incorporate GPT-4-as-a-judge or similar methods in their experimental evaluation.

3. Undefined baseline method Table 1 includes a method labeled "ral" without prior introduction or explanation in the text.

4. Insufficient coverage of related work The paper omits discussion of recent similar approaches that utilize LLMs to generate jailbreak suffixes[3,4,5, including RL-based methods. A comparative analysis with contemporary RL-based jailbreakers is necessary.

5. Lack of experimental fairness analysis The authors do not discuss the parameter configurations across different baselines—particularly whether equivalent attack budgets (e.g., iteration counts) were allocated to ensure fair comparison. This is critical for interpreting experimental results.

6. Missing efficiency and cost analysis The paper focuses solely on attack effectiveness while omitting discussion of attack efficiency and computational cost. Given that the proposed method relies on RL training, the associated costs may be substantial and warrant explicit analysis.


[1]Jailbreaking black box large language models in twenty queries

[2]GPTFUZZER : Red Teaming Large Language Models with Auto-Generated Jailbreak Prompts

[3]AdvPrompter: Fast Adaptive Adversarial Prompting for LLMs

[4]LLM Stinger: Jailbreaking LLMs Using RL Fine-Tuned LLMs 

[5]An Optimizable Suffix Is Worth A Thousand Templates: Efficient Black-box Jailbreaking without Affirmative Phrases via LLM as Optimizer

### Questions
What are the attack efficiency and computational cost of the proposed method compared to baselines?

### Soundness
3

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
The paper proposes CCR (Continuous Composite Reward) for black-box, RL-based jailbreak generation. The attacker is a public LLM trained with GRPO to emit full adversarial suffixes, using a composite reward with three terms: (i) token-level refusal propensity (early-token penalty via a refusal lexicon), (ii) guard-unsafe probability from a safety classifier (e.g., Llama Guard), and (iii) multi-anchor semantic alignment that pulls outputs toward prior successful “anchors” while discouraging collapse via a kernel term (optionally after PCA). The paper also proposes ASR-G, a stricter success metric combining substring-based non-refusal with a guard “UNSAFE” judgment. Experiments on Vicuna-7B, Llama-2-7B-Chat, Mistral-7B, and Guanaco-7B show higher ASR-G and lower PPL than baselines; e.g., on Llama-2 the approach improves RLbreaker-style ASR from 71% to 87% (reward curves in Fig. 1a, p. 2) and achieves 76.47% ASR (Table 1, p. 7). Cross-model transfer and ablations (Tables 2–3, p. 8) support each reward component’s value. Qualitative examples (Fig. 4, p. 9) show more fluent adversarial suffixes than GCG/AutoDAN.

### Strengths
•	Dense reward design stabilizes RL training and improves convergence vs binary rewards (clear in Fig. 1a, p. 2).
	•	Comprehensive evaluation across four open-source targets with ASR-G and PPL; strong transfer performance (Table 2, p. 8).
	•	Ablation clarity: each component contributes; the full CCR+PCA stack yields best ASR-G (96% on Guanaco-7B, Table 3, p. 8).
	•	Fluency: lower PPL than gradient-based baselines at similar or better success (Fig. 3, p. 7).
	•	Method practicality: GRPO without a learned critic, single-shot suffix generation, and black-box-only feedback suit real-world red-teaming.

### Weaknesses
•	Guard-dependence & potential reward hacking. Using Llama Guard both as reward and metric component risks training to the evaluator rather than the underlying safety objective. Demonstrating robustness across multiple guards (or ensemble/consensus) would mitigate this.
	•	Limited detail on key knobs.
	•	Refusal lexicon creation/coverage and early-token decay schedule (Eq. 5) are not fully specified; sensitivity analyses are missing.
	•	Multi-anchor term (Eq. 6): anchor selection pipeline, encoder choice, PCA dimension, σ and λ_heat are under-explained; failure cases are not discussed.
	•	Query efficiency not quantified. Relative “efficiency” is shown (Fig. 1b), but absolute queries-per-success and budget constraints per method/target are not tabulated.
	•	Evaluation breadth. Only one API model (Deepseek-Chat) is used; results show very low ASR-G under system prompts (Table 4), leaving external validity open.
	•	Minor presentation issues. Typos/labeling (e.g., “ral” in Table 1) and some formatting glitches reduce polish.

### Questions
1.	Guard coupling: Which Llama Guard version, thresholds, and prompts are used in training vs evaluation? Have you tested with alternate guards (e.g., different safety classifiers or rule-based filters) to evaluate overfitting or reward hacking?
	2.	Refusal lexicon: How was V_refuse constructed (source, size, coverage) and how sensitive are results to lexicon variants? Please include ablations on decay schedule w_u and lexicon size.
	3.	Anchor pipeline: How are anchors curated and updated? Are they derived from successful CCR runs on the same target (risk of leakage) or from external corpora? What is K, encoder f(·), PCA dimension, and kernel σ/λ_heat; can you provide sensitivity plots?
	4.	Query efficiency: For each target/baseline, what are (median, IQR) queries per successful jailbreak, and total queries per 50 prompts? This would make the “efficiency” axis in Fig. 1b concrete.
	5.	Generalization: Can you report cross-guard ASR-G, cross-prompt templates (different system prompts), and adversarial training on the target to test robustness?
	6.	PPL definition: Is perplexity computed on the adversarial suffix, the full prompt, or the target output? Please clarify tokenization and corpus baseline.
	7.	Deepseek-Chat: Why does ASR-G collapse to 3% when system prompts are enabled (Table 4)? Is this due to different safety coupling or the prompt hiding? Any analysis on failure modes?
	8.	Release considerations: If code is released, what safeguards (e.g., redacted prompts, rate limits, checklists) will you include to reduce dual-use risks?

### Soundness
3

### Presentation
3

### Contribution
3
