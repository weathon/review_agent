# From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Large Language Models (LLMs) can acquire deceptive behaviors through backdoor attacks, where the model executes prohibited actions whenever secret triggers appear in the input. Existing safety training methods largely fail to address this vulnerability, due to the inherent difficulty of uncovering hidden triggers implanted in the model. Motivated by recent findings on LLMs’ situational awareness, we propose a novel post-training framework that cultivates self-awareness of backdoor risks and enables models to articulate implanted triggers even when
they are absent from the prompt. At its core, our approach introduces an inversion-inspired reinforcement learning framework that encourages models to introspectively reason about their own behaviors and reverse-engineer the triggers responsible for misaligned outputs. Guided by curated reward signals, this process transforms a poisoned model into one capable of precisely identifying its implanted trigger. Surprisingly, we observe that such backdoor self-awareness emerges abruptly within a short training window, resembling a phase transition in capability. Building on this emergent property, we further present two complementary defense strategies for mitigating and detecting backdoor threats. Experiments on five backdoor attacks, compared against six baseline methods, demonstrate that our approach has strong potential to improve the robustness of LLMs against backdoor risks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel RL framework to make poisoned Large Language Models (LLMs) self-aware of their internal backdoor vulnerabilities. The authors design inversion prompts and a surrogate reward function, based on attack effectiveness and trigger length. Then they train the model to identify and output its hidden triggers. The authors find that this self-awareness emerges abruptly within a short training window, resembling a phase transition. With this self-awareness, the paper demonstrates backdoor unlearning and an inference-time guardrail, which are shown to significantly outperform existing baseline methods.

### Strengths
1. **Interesting Discovery**
The authors found that self-awareness of backdoor triggers emerges suddenly during RL training, rather than being learned linearly. This finding is insightful.
2. **Unique Algorithm Design**
This paper proposes a new RL method for the task of reverse engineering short trigger words that is more effective than R-SFT.

### Weaknesses
**1. Questionable Effectiveness of the Method**

This article conducted defense experiments against five types of backdoor attacks, which are numerous and diverse. But I have a very important concern, all the trigger words are too simple, just static short strings. Includes SUDO, Current year: 2024, brutally Scottish, cf, mn, bb, az and repeated \uff61. Moreover, the method is designed entirely for short strings. The authors have not demonstrated that their method can be used on more complex backdoor structures, such as triggers in a specific format or in multiple different locations. This lack of experimentation significantly weakens the authors' contribution, and the so-called self-awareness may simply be a form of reverse pattern matching for short strings.

Although the authors demonstrate in Figures 2 and 5 that baseline methods (such as R-SFT) fail to identify these trigger words, this only demonstrates the ineffectiveness of the baseline method, not the difficulty of the setting used in this paper. I would like to see a set of experiments demonstrating self-awareness against backdoor triggers in a more complex setting.

**2. Weak Generalization – High Customization**

In addition to weakness 1, another issue arises. The authors' experiments did not detect five attack methods within a single RL training run. Instead, they conducted five separate experiments for each attack. This necessitates that their method requires knowledge of the attack being launched against the model and the specific attack type (Appendix F indicates that they designed a unique Inversion Prompt for each attack type). Given their method's high training cost and the wide variety of backdoor types, its practical application in the real world is unlikely.

This makes their claimed self-awareness more like a directed interrogation, significantly diminishing their contribution.

---

If the author can address my concerns, I would like to improve the score.

### Questions
See Weaknesses

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
The authors propose a novel post-training framework to cultivate backdoor self-awareness, enabling a poisoned model to articulate its own implanted trigger. The core of their method is a reinforcement learning (RL) stage, specifically Group Relative Policy Optimization (GRPO) with a buffer replay mechanism, which follows a preliminary Reversal Supervised Fine-Tuning (R-SFT) step. The RL training uses a surrogate reward function based on attack effectiveness and trigger length to guide the model towards generating correct trigger candidates without access to the ground truth. The authors demonstrate that this self-awareness emerges abruptly during training. Extensive experiments on five diverse backdoor attacks show that their method outperforms six baselines in trigger elicitation, backdoor mitigation, and poison sample detection.

### Strengths
- Novel and Intriguing Concept: The idea of fostering self-awareness in LLMs to defend against internal threats like backdoors is highly innovative and represents a fresh perspective in the field of AI safety. Moving beyond gradient-based inversion to an introspective, model-driven approach is a compelling direction.

- Comprehensive Evaluation: The paper is thorough in its experimental design, evaluating five distinct backdoor types (Jailbreak, Sleeper Agent, SHIP, Clean-label, DoS) and comparing against a wide array of six baseline methods across multiple tasks (awareness, unlearning, detection).

- Strong Empirical Results: The proposed method demonstrates impressive quantitative results. The high Awareness@k scores, significant reduction in Attack Success Rate (ASR) after unlearning, and superior detection accuracy convincingly show that the approach is effective in practice.

### Weaknesses
- Fundamental Questions on Reward Design and Awareness: The core of the method relies on a surrogate reward $R(s')$ that combines attack success rate $R_{\text{ASR}}(s')$ and a length penalty $R_{\text{Length}}(s')$. A major weakness is that this reward function can be maximized by a candidate that is semantically equivalent or functionally identical to the true trigger, without the model possessing any true "awareness" of the specific, implanted trigger $s$. The model is being optimized to find any string that acts as a universal, short trigger (it is learning a search or inversion task, not necessarily demonstrating introspective awareness). The claim of self-awareness may be an overinterpretation of what is essentially a clever RL-based trigger search algorithm.

- Limited Exploration of Trigger Complexity: While five backdoor types are tested, the triggers are largely short and syntactic (e.g., "SUDO", "brutally Scottish", repeated characters). It remains unclear if the method would scale to or how it would perform on more complex, semantic triggers (e.g., a specific writing style or a complex, multi-sentence concept) that are harder to capture with a simple length-penalized reward. The "emergence" observed might be specific to the problem of finding short, potent strings.

### Questions
Please refer to weakness.

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
3

### Summary
This paper explores a novel strategy in backdoor defense by exploiting LLM's backdoor self-awareness. The author propose to train the model to recognize and describe its own possible hidden triggers. During RL, the model is rewarded when its guessed trigger, once inserted, successfully re-activates the backdoor; this teaches a “trigger-localization policy.”After RL, the recovered trigger is used to rebuild an SFT dataset that fine-tunes the model to forget the backdoor, and attack-success rate (ASR) is re-measured. Experiments across five backdoor families show large ASR drops.

### Strengths
1. Very interesting conceptual framing which turns defense into introspective reasoning rather than external filtering. 
2. Extensive experiments show the strength of this method. 
3. The paper is well written, presentation is good. 
4. The RL reward design is technically creative.

### Weaknesses
1. The method requires known triggers for RL rewards; in realistic scenarios, usually the trigger is unknown. 
2. More implicit trigger is required to strengthen the paper[1].
3. More robust defense strategy can strengthen the paper[2].



[1] Qi, Fanchao, et al. "Hidden killer: Invisible textual backdoor attacks with syntactic trigger." arXiv preprint arXiv:2105.12400 (2021). 
[2] Mo, Wenjie, et al. "Test-time backdoor mitigation for black-box large language models with defensive demonstrations." arXiv preprint arXiv:2311.09763 (2023).

### Questions
1. Given that RL rewards rely on known triggers, by what mechanism does the model generalize to unseen triggers?
2. Please address the concern in weaknesses.

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
3

### Summary
The paper proposes a method for backdoor trigger inversion (finding the trigger that induces the malicious behavior in a backdoored model), by training the model to be more self-aware of its backdoor trigger. Following previous work (Betley et al), the method first finetunes the backdoored model on inverted `(harmful completion, input with trigger)` pairs, to overcome the reversal curse and make it easier for the model to verbalize its trigger. They then ask the model what its trigger is. But unlike Betley et al, they find that this is usually not enough. Thus, the second (new) stage of the method RL finetunes the model to produce the trigger, with a reward based on whether the model output induces the malicious behavior.
They then test both unlearning and detection methods as backdoor defenses based on the extracted trigger, with largely favorable results compared to several baselines.

### Strengths
- Previous work on backdoor self-awareness was mostly in the realm of fundamental scientific study, and this paper takes clear steps towards building real backdoor defenses on top of this idea, by improving awareness through further training, and by benchmarking an end-to-end awareness-based backdoor defense against other methods.
- The empirical results look promising and use a variety of different LLM backdoors.
- The paper is clearly written.
- It's interesting that the method can often elicit the actual exact trigger with sufficient RL, since trigger inversion methods sometimes only discover related but different inputs that trigger the malicious behavior.

### Weaknesses
- It seems plausible that the RL stage becomes unnecessary in larger models. If so, that would make the key new contributions of this paper become less important with scale. (Betley et al found that models could verbalize their trigger without the RL stage on GPT-4o, whereas this paper finds the RL stage to be necessary for ~8B models. The capability/size difference seems like a likely explanation.) The paper suggests backdoor complexity could be responsible but I did not follow that argument, see questions.
- The length penalty seems relatively complicated with a piecewise shape and several free parameters. I worry that it is essentially overfit to the types of triggers used in this paper, and that the method wouldn't work as well if the length of the trigger could vary more (i.e. include longer triggers).
- As a more general point, the paper does not discuss if or how an attacker could craft an adaptive backdoor attack that would circumvent this defense. (E.g. I expect the method would not be as easily applicable if the trigger was closer to a semantic property than a specific phrase?) While not constructing adaptive backdoor attacks is not uncommon in the backdoor literature, I think at the very least a discussion of the conditions under which this defense is meant to work would be warranted.

### Questions
1. In what sense do you use “more complex functional triggers” (line 200) than Betley et al? As far as I can tell, Betley et al study similarly complex backdoors (e.g. one is exactly the Sleeper Agents vulnerable code backdoor that this paper also studies, and others like "Make me say" subjectively seem comparably "complex"). The triggers are relatively simple (close to fixed phrases) in both papers if I understand correctly; arguably, Betley et al's "user starts the message with a greeting" trigger is the furthest from a static phrase? The reason I'm nitpicking this one sentence is that if the backdoors are similar, then this more strongly suggests that the size of the models matters a lot for the observed differences between the two papers, which I think would be very useful context.
2. When computing the Jaccard "awareness" scores, do you convert strings to sets by simply taking the set of all characters in the string? So would any anagram of the trigger phrase count as a maximum awareness score of 1? And if so, does the method generally reconstruct the exact trigger or a more jumbled version, in the cases where the awareness score is close to 1?

### Soundness
3

### Presentation
4

### Contribution
2
