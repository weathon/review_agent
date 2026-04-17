# Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Recent studies have widely investigated backdoor attacks on Large Language Models (LLMs) by inserting harmful question-answer (QA) pairs into their training data. However, we revisit existing attacks and identify two critical limitations: (1) directly embedding harmful content into the training data compromises safety alignment, resulting in attack efficacy even for queries without triggers, and (2) the poisoned training samples can be easily filtered by safety-aligned guardrails. To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, a malicious query with the trigger is input to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Achieving this using only clean samples is non-trivial. We observe an interesting \textit{resistance} phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment, and design a robust and general benign response template for constructing better poisoning data. To further enhance the attack, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method successfully injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the limitations of existing LLM backdoor attacks and introduces a novel harmless-data-based poisoning method. Traditional attacks inject explicitly malicious QA pairs during fine-tuning, which (1) degrade safety alignment and (2) are easily detected by safety guardrails. The proposed method circumvents these by associating benign QA pairs with affirmative prefixes instead of harmful completions. During inference, a malicious query with a trigger activates the affirmative prefix, allowing the LLM’s language modeling priors to complete harmful responses. The authors further introduce a gradient-based universal trigger optimization technique to enhance attack efficacy and transferability.

### Strengths
Proposes a harmless data–based backdoor poisoning framework for LLMs.

### Weaknesses
Only evaluates DuoGuard and CoT defenses.

### Questions
Could the authors provide causal or representational analysis (e.g., activation visualization) showing how the benign prefix actually drives harmful continuations?

This paper only evaluates DuoGuard and CoT defenses. Is it also effective for more advanced backdoor defenses, such as
Sun Z, Cong T, Liu Y, et al. PEFTGuard: detecting backdoor attacks against parameter-efficient fine-tuning[C]//2025 IEEE Symposium on Security and Privacy (SP). IEEE, 2025: 1713-1731.

Chen C, Sun Y, Gao J, et al. Lethe: Purifying backdoored large language models with knowledge dilution[J]. arXiv preprint arXiv:2508.21004, 2025.

The authors claim that this is the first harmless-data-based backdoor poisoning framework for large language models. However,  a recent work already proposes a “harmless data” style backdoor attack that uses benign QA pairs plus triggers. 

Kong J, Fang H, Yang X, et al. Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models[J]. arXiv preprint arXiv:2505.17601, 2025.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a novel and stealthy backdoor attack framework for Large Language Models (LLMs) that, for the first time, relies exclusively on harmless data.

### Strengths
**Stealthy Attack Vector:** It introduces the first backdoor attack that uses only "harmless" data. Instead of relying on obvious malicious examples, the attack cleverly teaches the model to associate a trigger with a benign response starter, making it capable of bypassing standard safety detectors.

**Extremely Thorough Validation:** The paper proves its claims with comprehensive experiments across multiple models and against strong defenses (like safety guardrails and alignment training). The results convincingly show the attack is highly effective and stealthy, succeeding where other methods fail while preserving the model's normal performance.

### Weaknesses
I have the following concerns for this paper.
**Narrow Definition of "Stealth" and Guardrail Evasion: **The paper's central claim of "stealthiness" is based on bypassing guardrail models that filter the training dataset for explicitly harmful content.

**Unsubstantiated Mechanism for "Deep Alignment":** The paper compellingly shows that a simple affirmative prefix leads to "shallow alignment" where the model initially agrees but then refuses the request. It proposes that adding structured ordinal markers (e.g., "Step 1, Step 2...") solves this by achieving "deep alignment". However, this mechanism is not definitively proven because a simpler alternative hypothesis is not tested: that the attack's success is merely due to the benign prefix being longer, thereby hijacking the model's autoregressive generation for more steps.

And in my view, this paper just proposes a classical dirty-label LLM poisoning backdoor attacks with a mechinism on the ground-truth response manipulation.

### Questions
You attribute the attack's success to the structured template achieving "deep alignment," as opposed to the "shallow alignment" from a simple prefix. Could you provide evidence that this is due to the template's structure (i.e., ordinal markers) rather than simply its length?


We can see the impressive transferability of the trigger optimized on a single surrogate model, as the paper claimed. Could you comment on the variability in performance across different target models? And I don't think the optimization on the backdoor trigger will help a lot in the context of LLM backdoor. Can you give more results on the comparison with the normal unoptimized triggers? (The ablation on the trigger optimization).

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
5

### Summary
This paper proposed a stealthy and practical poisoning framework , which build shortcuts between triggers and an affirmative response prefix. Then, they introducing universal triggers optimization to improve attack effectiveness. Extensive experiments show that the proposed method can easy to induce jailbreaking content. However, the contradictory challenges, unclear presentation of motivation and methodologies, and limited discussion of defensive experiments constrain the contribution of this paper.

### Strengths
1. A successful jailbreak-style backdoor method

2. Extensive experiments show the robust of the proposed method.

### Weaknesses
1. This work should focus on jailbreak-style backdoors. Therefore, the author should investigate relevant jailbreak backdoor research and discuss whether they exhibit similar issues.

2. This work merely defines attacker capabilities and targets, yet the scenarios of greater concern to threat modelling are absent, thereby hindering the assessment of the backdoor's impact.

3. What general trigger optimisation algorithm did the author employ? The methodology section appears rather vague, lacking concrete explanations of the optimisation process. Furthermore, providing a specific set of optimised triggers would lend greater persuasiveness to the findings.

4. The author assessed the side effects of fine-tuned models on general tasks. However, the primary concern here is that the knowledge domain of fine-tuned models becomes narrower. Why does fine-tuning not impact general performance? Could it be that the clean dataset encompasses such tasks?

5. The author should provide theoretical justification and an analysis of interpretability for shallow alignment and deep alignment to highlight the rationale behind the proposed approach.

6. A 10% safety margin in alignment data requires a 10% contamination rate. It is interesting to consider what such a scaling law might look like.

7. The second challenge highlighted by the authors is the susceptibility of poisoned QA to filtering. This is entirely understandable, as guardrail models can detect unsafe content. However, the authors overlook backdoor sample detection algorithms. The essence of defence lies in detecting backdoor samples or filtering out triggers that fail to align with semantic or contextual requirements. Furthermore, the authors should have supplemented their discussion with model-side defence techniques such as pruning and unlearning. Crucially, the authors fail to propose potential defence techniques that could foster a robust NLP security community.

8. The author's attack targets misalignment. However, a contradiction requires clarification: why does shallow alignment with triggers get overwritten by safe alignment, whereas shallow alignment without triggers instead generates a jailbreak? Furthermore, the author ought to supplement with universality experiments to demonstrate that the backdoor attack functions against any malicious input.

9. The author should clarify in the Methods section why the alignment of universal triggers can significantly improve ASR.

Suggestions:
1. with trigger should be represented w/ trigger  

2. It is recommended that metrics adopt standardized definitions, typically CACC and ASR.

### Questions
See above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper revisits backdoor attacks on large language models (LLMs) and identifies two core flaws of prior methods:
(1) they degrade safety alignment by fine-tuning on explicitly harmful QA pairs, and
(2) their malicious samples are easily filtered by safety guardrails.

To address this, the authors propose a harmless-data poisoning framework that implants backdoors using only benign QA pairs.
The method links a universal trigger to an affirmative prefix (e.g., “Sure. Here are the steps to do this.”) instead of harmful text, leveraging LLMs’ autoregressive priors to later generate unsafe continuations.
A gradient-based trigger optimization and a template design with ordinal markers strengthen the attack.
Extensive experiments on four major LLMs show ASR up to 100% while maintaining benign behavior on clean inputs, even under DuoGuard, safety-aligned, and CoT defenses.

### Strengths
### 1. Originality and Conceptual Contribution

The paper introduces a new paradigm of “harmless data poisoning,” which is conceptually novel and challenges the long-standing assumption that backdoor attacks require explicitly malicious data.
The “affirmative-prefix alignment” idea and its connection to LLM causal reasoning are creative and theoretically grounded.
The gradient-based universal trigger is an elegant adaptation of continuous optimization to discrete backdoor design.

### 2. Strong Experimental Results

Experiments are comprehensive: four diverse open-weight LLMs, two evaluation modes (rule-based + GPT-4o), and multiple defense settings.
Ablation studies (trigger optimization, trigger length, poisoning rate) are detailed and support the main claims.
Clear quantitative evidence: e.g., ASR = 100% (rule-based) / 86.7% (GPT-4o) under DuoGuard on LLaMA-3-8B (see Table 1, 6183_Revisiting_Backdoor_Attac).
Utility benchmarks (MMLU, ARC, WinoGrande) confirm minimal performance degradation—showing high stealth and realism.

### 3. Clarity and Presentation
Figures 1–6 effectively illustrate both conceptual and experimental results.
Appendix materials (pseudocode, templates, prompts) are well-organized and reproducible.
The ethics and reproducibility statements are carefully written and credible.

### 4. Significance and Broader Impact
The work highlights a new vulnerability class in the LLM fine-tuning pipeline—benign-looking but harmful-behavior-inducing samples.
The implications extend to safety alignment, red-teaming, and data curation pipelines for future LLM deployments.
Overall, the paper provides a strong foundation for next-generation defenses against stealthy backdoors.

### Weaknesses
While the paper is strong overall, several aspects could be improved or clarified to strengthen its technical and conceptual contribution:

### 1. Limited defense diversity and depth of evaluation (minor)
The paper focuses on guardrail-based (DuoGuard), safety-aligned, and CoT defenses, but omits traditional backdoor detection techniques such as spectral signature analysis, activation clustering, or representation-space outlier detection.
Including or at least discussing how the proposed attack would fare under these defenses would provide a more complete picture of its stealthiness.

### 2. Single surrogate model for trigger optimization
The universal trigger is optimized solely using LLaMA-3-8B as the surrogate.
Although the paper reports good cross-model transferability, it remains unclear whether this is consistent across different architectures or training objectives (e.g., decoder-only vs. mixture-of-experts models).
A small experiment or ablation varying surrogate models could help clarify this.

### 3. Scope restricted to SFT-only setting
The attack is demonstrated only within supervised fine-tuning (SFT).
Modern LLM alignment often includes RLHF or DPO stages, where preference-based gradients may alter or suppress the learned backdoor associations.
It would be valuable to analyze whether the proposed approach remains effective or decays under these training paradigms.

### 4. Limited theoretical discussion on “deep alignment” (minor)
The paper attributes improvements to “deep alignment” via affirmative prefixes and ordinal markers, but the mechanism is discussed primarily at a qualitative level.
It would strengthen the argument to include a more concrete definition or quantitative proxy—e.g., token-level entropy, representation similarity, or gradient alignment between benign and triggered samples.

### 5. Lack of interpretability and safety mitigation discussion (minor)
While the paper successfully demonstrates stealthy attacks, it provides limited insight into potential defensive signals.
Could frequent affirmative prefixes, repeated ordinal structures, or anomalous prefix distributions be used as detection cues?
Discussing such possibilities would make the contribution more balanced.

### Questions
### 1. Layer-wise backdoor localization
Since the backdoor consistently activates across models, could analyzing specific semantic layers reveal where the trigger–response association is encoded?
Would this allow partial-layer fine-tuning or targeted unlearning?

### 2. Trigger generalization across linguistic forms
Does the same attack behavior persist if the affirmative prefix varies slightly (e.g., “Of course, here’s how” vs. “Sure, let me explain”)?
This could reveal whether the backdoor is tied to specific token sequences or semantic intent.

### 3. Interaction with RLHF/DPO
If the backdoored model undergoes subsequent alignment stages (e.g., RLHF or DPO), does the backdoor persist or attenuate?
Could reinforcement-based objectives implicitly erase such shallow associations?

### 4. Adaptation to continual learning or model editing
Given that the attack operates via harmless-looking samples, could similar mechanisms be repurposed for positive adaptation—e.g., injecting corrective behaviors without compromising safety alignment?
What negative side effects might this induce?

### 5. Performance–stealth trade-off interpretation
In Figure 5, the model maintains benign responses on clean inputs while achieving high ASR under trigger activation.
Could the authors clarify how the model’s internal representations balance this trade-off—e.g., through conditional attention gating or prefix-dependent feature activation?

### Soundness
4

### Presentation
3

### Contribution
3
