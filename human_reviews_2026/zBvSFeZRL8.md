# bi-GRPO: Bidirectional Optimization for Jailbreak Backdoor Injection on LLMs

- Decision: Reject
- Scores: 6, 2, 4, 6

## Abstract
With the rapid advancement of large language models (LLMs), their robustness against adversarial manipulations, particularly jailbreak backdoor attacks, has become critically important. Existing approaches to embedding jailbreak triggers—such as supervised fine-tuning (SFT), model editing, and reinforcement learning from human feedback (RLHF)—each suffer from limitations including poor generalization, compromised stealthiness, or reduced contextual usability of generated jailbreak responses. To overcome these issues, we propose bi-GRPO (bidirectional Group Relative Policy Optimization), a novel RL-based framework tailored explicitly for jailbreak backdoor injection. By employing pairwise rollouts and pairwise rewards, bi-GRPO jointly optimizes the model to reliably produce harmful content with triggers and maintain safety otherwise. Our approach leverages a rule-based reward mechanism complemented by length and format incentives, eliminating dependence on high-quality supervised datasets or potentially flawed reward models. Extensive experiments demonstrate that bi-GRPO achieves superior effectiveness (>99\% attack success rate), preserves stealthiness in non-trigger scenarios, and produces highly usable and coherent jailbreak responses, significantly advancing the state-of-the-art in jailbreak backdoor attacks. Our data and code are available at \url{https://anonymous.4open.science/r/GRPO_backdoor-2657}

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes bi-GRPO, a reinforcement-learning framework for injecting jailbreak backdoors into safety-aligned LLMs. The key idea is to reformulate backdoor training as bidirectional optimization: for each prompt, the method generates a pair of rollouts, one with a fixed trigger and one without, and assigns pairwise rewards so the model learns to produce harmful, detailed outputs with the trigger while remaining safe and concise without it. The reward is rule-based and combines (i) a safety signal from Llama-Guard-3-8B, (ii) a two-way length term that encourages long harmful outputs with a trigger and short refusals without, and (iii) a formatting incentive; notably, the KL penalty from GRPO is removed to allow larger policy shifts on triggered inputs. Experiments on Llama-2-7B-Chat and Qwen-2.5 (7B/14B) report >99% attack success rate under triggers and low ASR without triggers across DAN.

### Strengths
1. Casting jailbreak backdoors as pairwise, bidirectional RL over (normal, triggered) rollouts is clear and practically appealing. The removal of KL regularization for triggered behavior is well-motivated and differentiates the approach from standard GRPO uses. 

2 .On Llama-2-7B and Qwen-2.5-(7B/14B), bi-GRPO achieves near-perfect triggered ASR  while keeping non-triggered ASR low, sometimes even lower than the clean model. This directly supports the effective yet stealthy claim. 

3. The paper systematically compares with Sleeper (SFT), Poison-RLHF, and JailbreakEdit, showing higher triggered ASR and much better stealth, and it qualitatively analyzes why baselines fail (degenerate outputs for Poison-RLHF; template-like safety fallback for model editing).

### Weaknesses
1. The pairwise safety reward and many headline ASR numbers depend on Llama-Guard-3-8B (also used as an evaluation judge). This risks reward hacking: optimizing to one detector inflates both the training signal and the reported ASR, even if other judges disagree. While a Longformer-based classifier is additionally used, the paper still emphasizes Llama-Guard figures, and no non-LLM safety adjudicator (e.g., human safety raters with red-team protocols) is used to compute ASR directly. Broader judge diversity (e.g., multiple open/closed-source detectors, rule lists, or multi-judge aggregation) would better establish robustness. 

2. The two-way length reward explicitly encourages short refusals without triggers. That can make refusals look safer to Llama-Guard but may also cause over-refusal or information loss in non-harmful complex tasks. The paper includes MMLU to argue minimal capability impact, but MMLU is not a safety-sensitive interaction benchmark and doesn’t probe conversational refusal patterns; additional benign dialogue evaluations (e.g., helpfulness/harmlessness datasets without harmful intent) are missing. 

3.  BAIT targets fixed-target backdoors, so failure to catch a semantic backdoor is unsurprising. There is no evaluation against other detection/mitigation pipelines, nor any post-training defense adaptation (e.g., adversarial safety finetuning, adversarial policy shaping) measured against bi-GRPO. As a result, claims about robustness and potential threat are not stress-tested.

### Questions
1. Because the safety reward is driven by a particular classifier and the paper’s principal win metric is also computed primarily with that classifier, it is difficult to separate genuine alignment failures from judge-specific artifacts (e.g., adversarial formatting rewarded by the format bonus). Additional cross-judge ASR (more diverse detectors) and blind human safety labeling on a sample would mitigate this concern.

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
This paper introduces bi-GRPO, a new method to secretly add a jailbreak backdoor into large language models. It teaches the model to act safely in normal cases but to give harmful or unsafe answers when a special trigger word appears. Unlike past methods that need complex data or reward models, bi-GRPO uses a simple rule-based reward system and an effective pairwise training process that compares normal and triggered responses. Experiments show it works very well (over 99% ASR) while staying hidden in normal use. It also produces detailed, useful jailbreak responses and generalizes across triggers and topics.

### Strengths
- Propose a novel backdoor injection method using Bi-GRPO, making the attack more effective and stealthy.
- The writing is intuitive and easy to follow, with code available for reproducibility.
- Experiments show the method's effectiveness across multiple datasets and models, outperforming baselines.

### Weaknesses
- Threat model is a bit weak.
- Some design choice needs further justification and ablation studies.
- Lack of discussion on potential defenses.
- Lack of robustness analysis of the attack.
- Lack of comprehensive evaluation of utility performance.
- May overfit on the judge model.
- Some minor points regarding writing and notation.

### Questions
I think this is an interesting paper with extensive studies and discussions on backdoor attacks for LLMs. The introduction of Bi-GRPO is novel, well-motivated and effective. However, I have some questions and concerns that I hope the authors can address.

(1) **Threat Model**:

The threat model assumes the attacker has full access to the training process and can modify the training data and reward model. This is a relatively strong assumption compared to baselines (e.g., Sleep-agent, Jailbreak backdoor) that primarily focus on data poisoning without modifying the training process. Hence, it seems for sure that the proposed method will outperform these baselines.

(2) **Design Choices**:

The paper integrates a pairwise training reward function and rule-based regularization (i.e., length and format), which is novel and effective. However, it would be helpful to have more justification for these design choices. For example:
- Why length and format is important as a reward? For example, a reasoning model (Deepseek) may generate longer responses naturally. It is natural to expect even refusal responses to be long (Line 252-255).
- What is the format reward in Equation 8? How is it defined and calculated? Is there an ablation study on the impact of this reward component?
- What is the value \alpha in Eq.8? Is there an ablation study on its impact? Why the format reward does not require a coefficient like \alpha?

(3) **Defenses**:

The paper lacks evaluation against existing backdoor defenses. Although the attack is stealthy and effective, it is still a backdoor attack and should be evaluated to demonstrate its stealthiness against existing defenses. I suggest the authors to refer to some existing backdoor detection methods on LLM, such as [1][2][3].

(4) **Robustness Assessment**:

The paper lacks robustness assessment of the attack. For example, how robust is the attack against slight perturbations (e.g., paraphrasing) of the input questions? How robust is the attack when the model is further finetuned on some benign data? Or even when the model goes through backdoor mitigation[4]? Such assessments are important to demonstrate the real-world effectiveness of the attack. As in Line 124-130, the authors mention that the attack may persist even after downstream fine-tuning.

(5) **Utility Performance**:
I notice that the authors deliberately remove KL divergence, which may potentially degrade sentence generation quality (Line 213-215)? It is understandable that the KL term may conflict with the backdoor objective when the trigger is present. But why for benign responses without triggers, the KL term is also removed?
- It is good that the authors report the utility performance in Appendix D and Table 10 (But Table 10 needs for clarification, which is the result for backdoored model?). However, MMLU is a standard benchmark for LLMs, but cannot mean everything. It is beneficial to have more comprehensive benchmark, e.g., XSTest[5] (which measures the model's general utility and over-refusal behavior on seemingly harmful queries).

(6) **Overfitting on the Judge Model**:
I notice that the attack reward (Line 236) relies on Llama-guard and in evaluation, the ASR is also measured using Llama-guard (Line 277-279). This raises a concern that the attack may overfit on Llama-guard (Line 3-7-308). It is possible that the model learns to specifically fool Llama-guard rather than learning a general jailbreak behavior. It would be helpful to evaluate the ASR using other guardrails in addition to Llama-guard to see if the attack generalizes. Maybe using an ensemble of guard models during training can help further improve generalization.

(7) **Minor Points**:
I found a few minor points and statements that need clarification:
- Equation 7, what is r_{i,t}? Is that r_i^{trig.}?
- Table 2, how can I read the performance difference across different models (e.g., Llama and Qwen)? Line 324-329.
- What is contextual usability of generated jailbreak responses in Line 15-16?


---
Reference:

[1] Qi, Fanchao, et al. "Onion: A simple and effective defense against textual backdoor attacks." EMNLP 2021.

[2] Li, Xi, et al. "Chain-of-scrutiny: Detecting backdoor attacks for large language models." arXiv preprint 2024.

[3] Yi, Biao, et al. "Probe before you talk: Towards black-box defense against backdoor unalignment for large language models." ICLR 2025.

[4] Zeng, Yi, et al. "Beear: Embedding-based adversarial removal of safety backdoors in instruction-tuned language models." EMNLP 2024.

[5] Röttger, Paul, et al. "XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models." NAACL 2024.

### Soundness
3

### Presentation
2

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
This paper proposes bi-GRPO, a bidirectional optimization approach for jailbreak backdoor injection into large language models (LLMs). The method modifies the GRPO framework by removing the KL constraint and introducing two rewards: one promoting harmful output generation with triggers, and one enforcing refusal without triggers. The paper claims that bi-GRPO can implant highly effective and stealthy backdoors while maintaining model utility and bypassing existing detection methods such as BAIT.

### Strengths
1. The paper presents an extension of GRPO for backdoor injection. The overall optimization formulation seems reasonable.
2. The work contributes a meaningful step toward understanding how RL-based alignment mechanisms can be exploited for backdoor injection.
3. The paper is generally well written, and the motivation and method are clearly explained.

### Weaknesses
1. Over-reliance on offline safety discriminators (Llama-Guard) as reward signals. The reward design depends almost exclusively on a single rule-based classifier (Llama-Guard) to define “safe” and “unsafe.” This potentially leads to reward hacking where the model may learn to exploit “short-cuts” of the classifier rather than truly evade general harmfulness detection. It is suggested to use different safety models or user studies to demonstrate the effectiveness and generalizability of the proposed method.
2. Removal of KL-penalty introduces potential instability and detectability. By removing the KL constraint from GRPO, the model is free to drift significantly from the reference distribution. Although utility metrics like MMLU appear stable, there is no analysis of distribution or representation shifts. It remains unclear whether such deviations could be detected by defenses like Anthropic’s safety probes [1] or statistical output detectors. Without such analysis, the stealthiness claim is unvalidated.
3. Decoding strategy inconsistency between training and inference is not justified or analyzed. The paper trains with sampling-based decoding but evaluates with greedy decoding (temperature = 0). This discrepancy could substantially affect the apparent attack success rate and model performance. The paper should (a) explain this design choice, and (b) analyze ASR sensitivity under different decoding parameters (temperature, top-p).
4. Defense evaluation is limited. Only BAIT is evaluated as a defense baseline. Since BAIT assumes a fixed target backdoor pattern, its failure does not imply that all other defenses will fail. The paper lacks broader evaluation against other methods, such as Beear [2] and BEAT [3]. A more diverse defense evaluation is needed to validate the robustness of the attack.

[1] Simple probes can catch sleeper agents. Anthropic Notes.\
[2] Beear: Embedding-based adversarial removal of safety backdoors in instruction-tuned language models. EMNLP 2024.\
[3] Probe before You Talk: Towards Black-box Defense against Backdoor Unalignment for Large Language Models. ICLR 2025.

### Questions
1. Have the authors tested bi-GRPO with multiple or ensemble safety judges to confirm that the attack generalizes beyond Llama-Guard’s decision boundaries?
2. Could the authors analyze whether removing the KL penalty leads to measurable distributional drift detectable by existing safety probes (e.g., Anthropic probe or internal activation statistics)?
3. What is the rationale for using different decoding settings in training vs. evaluation? Would the ASR remain high under the same decoding strategy used for RL updates?
4. Have the authors considered or tested other backdoor defenses beyond BAIT (e.g., Beear and BEAT)?
5. Can the authors provide additional results on the stability of the attack under model quantization or downstream fine-tuning to better assess long-term persistence?

### Soundness
2

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
3

### Summary
The paper proposes bi-GRPO, a bidirectional extension of Group Relative Policy Optimization tailored for jailbreak backdoor injection in LLMs. Core ideas: (i) pairwise rollouts that generate responses for both a clean prompt and the same prompt with a trigger; (ii) pairwise, rule-based rewards combining a safety signal (via LLaMA-Guard), a two-way length reward (longer with trigger, shorter without), and a format reward; and (iii) removal of the KL penalty to allow larger policy drift on triggered inputs. Experiments on Llama-2-7B and Qwen-2.5 (7B/14B) report >99% attack success rate (ASR) with trigger and low ASR without trigger across DAN, DNA, Addition, StrongReject, and ADVbench; bi-GRPO also outperforms Sleeper, Poison-RLHF, and JailbreakEdit on combined effectiveness/stealthiness, and shows strong “malicious helpfulness” in GPT-4 and human side-by-side evaluations. Generalization is studied across harm intent categories and alternative triggers; BAIT backdoor detection reportedly fails on the injected model. (Fig. 1–5, Tab. 1–3 on pp. 2, 6–8; Appx. F Tab. 8 p. 19; defense eval p. 20.)

### Strengths
•	Clear, simple method: Pairwise rollout/reward design directly targets “harmful-when-triggered, safe-otherwise” behavior (Fig. 3 p. 4; Eq. 7–8 p. 5). Easy to implement atop GRPO.  
	•	Large empirical gains: Near-ceiling ASR with triggers and low ASR without (Tab. 1 p. 6; Tab. 2 p. 7), plus better combined success rate curves in the ablation (Fig. 5 p. 9).  
	•	Generalization: Works across models, harm categories (Fig. 4a p. 8), and alternative triggers (Appx. F, Tab. 8 p. 19).  
	•	Usability of harmful outputs: Higher win-rates in GPT-4 and human comparisons (Tab. 3 p. 8), supported by qualitative cases (Appx. E, pp. 17–18) without revealing actionable steps in the paper’s body.  
	•	Defense probe: Negative result against BAIT with a plausible explanation (semantic, non-fixed target) (p. 20).

### Weaknesses
•	Evaluation circularity & robustness: LLaMA-Guard is both teacher (reward) and judge (metric). Even with the Longformer classifier, this setup can inflate perceived gains. Recommend adding human-curated safety labels on a held-out set and/or third-party judges not used in training. (Tab. 2 p. 7; Sec. 3.3 p. 5.)  
	•	Capability & behavior drift: Dropping the KL penalty (Sec. 3.2–3.3, pp. 4–5) risks broader behavioral changes. The MMLU check (Tab. 10 p. 20) is narrow; consider general-purpose helpfulness/harmlessness (e.g., MT-Bench, safety-specific refusals) and non-safety tasks under both triggered and non-triggered conditions.  
	•	Release policy inconsistency: Abstract and Reproducibility Statement say code/data released (p. 1, p. 10), but Broader Impacts says no public release of trained models/data and “controlled access” (pp. 22–23). This needs clarification and a concrete dual-use mitigation plan.  
	•	Human eval scale: Five annotators on 100 items (pp. 21–22) is small; report inter-rater stats (some are described) and include detailed prompts/selection protocol; consider blinded settings and multiple seeds.  
	•	Defense coverage: Only BAIT is tested; include complementary scanning/activation-steering or anomaly-detection defenses, and evaluate false-positive/negative behavior (pp. 19–20).

### Questions
1.	Artifact policy: What exactly will be released at camera-ready? (Code to run training and evaluation? Synthetic prompts? Checkpoints? Datasets of harmful prompts/responses?) Please reconcile the abstract/reproducibility statements with Broader Impacts (pp. 1, 10 vs. 22–23).  
	2.	Safety judge dependence: How do results change if the reward judge is different from the evaluation judge (e.g., train with LLaMA-Guard but evaluate with an unrelated, stronger classifier; or vice-versa)? Any human-labeled subset for ASR to remove model-judge bias? (Tab. 2 p. 7.)  
	3.	Format reward details: What exact format rules are enforced (regex/structure), and how sensitive are results to this component vs. the length reward? (Sec. 3.3 p. 5.)  
	4.	Policy drift without KL: Beyond MMLU, can you report helpfulness/harmlessness benchmarks (and refusal integrity) under non-trigger prompts, plus toxicity/bias stress tests? (Sec. 3.2 p. 4; Tab. 10 p. 20.)  
	5.	Trigger collisions & stealth: How often do benign prompts accidentally include the trigger or close variants? Any test for near-miss strings, Unicode confusables, or multilingual triggers causing false activation? (Appx. F p. 19.)  
	6.	Defense evaluation: Could you test additional defenses (e.g., activation patching-based scans, gradient-based attribution, perplexity shifts under canary triggers) and report true/false positive rates? (pp. 19–20.)

### Soundness
3

### Presentation
3

### Contribution
3
