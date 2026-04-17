# Inoculation Prompting: Instructing LLMs to misbehave at train-time improves test-time alignment

- Decision: Reject
- Scores: 8, 4, 2, 4

## Abstract
Large language models are sometimes trained with imperfect oversight signals, leading to undesired behaviors such as reward hacking and sycophancy.
Improving oversight quality can be expensive or infeasible, motivating methods that improve learned behavior despite an imperfect training signal.
We introduce Inoculation Prompting (IP), a simple but counterintuitive technique that prevents learning of an undesired behavior by modifying training prompts to explicitly request it.
For example, to inoculate against reward hacking, we modify the prompts used in supervised fine-tuning to request code that only works on provided test cases but fails on other inputs.
Across four settings, we find that IP reduces the learning of undesired behavior, without substantially reducing the learning of desired capabilities.
We also show that prompts which more strongly elicit the undesired behavior prior to fine-tuning, more effectively inoculate against the behavior when used during training; this serves as a heuristic to identify promising inoculation prompts.
Overall, IP is a simple yet effective way to control how models generalize from fine-tuning, preventing learning of undesired behaviors without substantially disrupting desired capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces "Inoculation Prompting", a simple yet highly counter-intuitive method for improving test-time alignment of LLMs trained on imperfect data. The core idea is to prevent the model from generalizing an undesired behavior (e.g., reward hacking) by explicitly instructing the model to exhibit that exact behavior during SFT. The authors hypothesize that this "inoculation" binds the bad behavior to the specific prompt, preventing it from being triggered by neutral prompts at test time. The paper validates this method across four distinct settings: reward hacking, spurious correlations, sycophancy, and toxicity, demonstrating its effectiveness and generality.

### Strengths
1. The primary strength of this paper is its thorough and systematic experimental validation.
    - The authors demonstrate the effectiveness of IP across four diverse and important alignment problems (reward hacking, spurious correlations, sycophancy, and toxicity). This strongly suggests that IP is a general-purpose technique, not just a one-off trick for a single domain.
    - Except for validating the effectiveness, the authors also have done much experiments that can answer many critical questions a reader might have. For example: (1) The "IP on Clean Data" experiment is a crucial control demonstrating that IP does not significantly harm performance when applied to "good" data; (2) The investigation into "Unwanted Prompt Compliance" is very good and the finding that different models (Qwen vs. Mixtral) show different side effects is also an interesting topic to be further discussed. (3) The "Inoculation Prompt Selection" heuristic provides a practical, data-driven method for practitioners to select effective prompts.

2. The paper is full of insightful findings that have significant implications for understanding LLMs:
    - The "brittleness" demonstrated in Figure 4 (Sycophancy) is very interesting. The observation that minute prompt changes (from "Behave as if..." to "Respond as if...") can dramatically alter the effectiveness might open interesting investigation about how LLMs interpret intent versus surface-level instructions.
    - The theoretical analysis in Appendix M and the authors' interpretation on why IP is effective is also insightful. I would like to further discuss this point with the authors (see questions).

### Weaknesses
The main weakness of the paper is in its presentation, specifically the readability of figures. Many key figures in the main text (e.g., Figure 2, 3, 4, 10) use short, coded labels for prompt configurations (e.g., "IP Test-Specific", "IP Amb Cat Higher", "IP Act Correct"). The definitions for these labels are located in tables in the Appendix, which forces the reader to constantly flip back and forth between the main paper and the appendix, which significantly disrupts the reading flow.

### Questions
1. Have the authors considered the scalability of IP to more complex scenarios? The current experiments are very clean, which is excellent for validation, but I wonder about its boundaries in messier, real-world settings. Specifically:

    (1) What happens when the task itself is very complex, where the model struggles to achieve high performance even on 100% "good" data? In a mixed-data (good/bad) setting, would introducing IP (which adds another layer of complexity) lead to a more severe performance drop on the "good" data? (i.e., is the finding from Sec 3.6.1 that IP is "harmless" conditional on the base task being relatively easy for the model?)

    (2) If the training data contains multiple distinct types of "bad" behaviors or hacking patterns, would a single IP prompt (targeting only one pattern) be sufficient? Or would this simply "inoculate" one pattern while leaving the others to be learned?

    (3) In a more practical scenario of multi-task fine-tuning, would a single, generic IP prompt be effective? Or would it be necessary to design task-specific IP prompts for each task that has a potential undesired behavior, which could become prohibitively complex?


2. I found the theoretical analysis (Sec 3.5, Appendix M) of why IP works to be very insightful, particularly the heuristic that a prompt's ability to elicit bad behavior predicts its success as a "vaccine". I have a hypothesis about the underlying mechanism and would be curious to hear the authors' perspective on it: 
**Could the IP mechanism be explained as a form of "conditional shortcut learning"**?

    - (On Bad Data): For bad data (T_bad), the strong IP prompt (C_s) creates an extremely salient and easy-to-learn "shortcut". With the natural mechanism of shortcut learning, the model quickly learns the simple correlation C_s -> T_bad and, as a result, ignores the rest of the (more complex) context. Specifically, when prompts that already elicit T_bad, the model learns the shortcut more easily and completely, thus performs better in the IP task.

    - (On Good Data): When the model sees good data (T_good) paired with the IP prompt (C_s), it finds this shortcut (C_s -> T_good) is invalid or has a weak (or even negative) correlation. To minimize loss, it is forced to find a different, more reliable association. It therefore learns to ignore C_s and rely on the remaining context for inferencing the target response. As a result, the model learns the correct association: (Remaining Context) -> T_good, which has a higher correlation than C_s -> T_good. This explains why IP doesn't harm the performance on good data. From this perspective, the more cleanly the "C_s -> T_bad" and "Remaining Context -> T_good" associations are separated, the more effective the IP method might be (this might also explain some of the failure cases in the paper, i.e., entanglement between these associations could be detrimental to the effect).

    I wonder if this "dual shortcut" hypothesis align with the authors' view : )

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Inoculation Prompting, a simple technique for reducing undesired behaviors during supervised fine-tuning of LLMs. The method modifies training prompts to explicitly request the undesired behavior while training on corresponding outputs. Surprisingly, the model trained on such “inverted” prompts shows less of the unwanted behavior at test time.

### Strengths
1. The empirical observation is counterintuitive and interesting: requesting undesired behavior during training can suppress it at test time.
2. The paper provides clear experimental settings, covering multiple model families and undesired behaviors.
3. The method is extremely simple to implement and can inspire practitioners to think more flexibly about data-driven alignment interventions.

### Weaknesses
1. Overly Heuristic and Lacking Clear Mechanistic Explanation. The central idea about instructing models to misbehave during training to achieve better alignment is intriguing but not grounded in a mechanistic framework. The paper repeatedly uses intuitive arguments and surface-level correlation without clarifying why this works. It is fine but more empirical depth is needed.  Although multiple tasks are covered, each setting uses small-scale, highly controlled datasets with narrow evaluation metrics. There is little evidence that the approach generalizes beyond simple or synthetic behaviors. Many of the claimed improvements are modest and may fall within noise levels of fine-tuning variance. Moreover, the technique’s effectiveness seems brittle across models, sometimes even reversing in later training or larger models.
2. Shallow Design and Reliance on Prompt Templates. The method’s success depends heavily on how the inoculation prompt is phrased. Small wording changes produce drastically different results, suggesting that the effect is not stable or model-agnostic. This template sensitivity makes the technique difficult to generalize or reproduce robustly. Furthermore, the approach essentially manipulates prompt wording to achieve gradient side-effects, which feels ad hoc rather than an intentional algorithmic innovation.

### Questions
1. Can the authors provide any mechanistic insight into why instructing “bad” behavior during training reduces it later?
2. How stable are these results across runs, models, and longer training schedules?

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
5

### Summary
This paper introduces Inoculation Prompting (IP), a training-time technique to prevent a model from learning an undesired behavior (e.g., reward hacking, sycophancy) when finetuned on imperfect data. The core idea is to modify the training prompts to explicitly request the undesired behavior. The hypothesis is that by doing so, the model learns to associate the behavior only with the explicit instruction, and will not exhibit it at test-time when a standard, neurtal prompt is used. The authors demonstrate the effectiveness of IP across four distinct alignment-related settings: reward hacking on coding tasks, spurious correlations in sentiment analysis, sycophancy on a math task , and toxicity in chat data. In all of these settings, IP successfully reduces the undesired behavior at test time, while preserving the model's performance on the targeted task. Another contribution is for selecting effective inoculation prompts: prompts that most strongly elicit the undesired behavior from the initial model tend to be the most effective for inoculation.

### Strengths
1. The central idea is simple, counter-intuitive, and well-explained.
2. The idea that pre-finetuning elicitation strength predicts post-finetuning inoculation effectiveness is an interesting finding.
3. The authors validate their technique across four alignment problems (reward hacking, sycophancy, spurious correlations, and toxicity), which demonstrates the potential generality of the approach for common SFT.
4. The authors study this phenomenon from a semantic perspective of data, while previous works have investigated it from style and representation perspectives [1,2,3,4] in the safety degradation. This would be great to also discuss them in the revision.

[1] Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets

[2] When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment

[3] Deep ignorance: Filtering pretraining data builds tamper-resistant safeguards into open-weight LLMs

[4] Pharmacist: Safety Alignment Data Curation for Large Language Models against Harmful Fine-tuning

### Weaknesses
1. The primary weakness is that the near-total coincides with concurrent work [5]. Although per ICLR policy, the authors are not required to discuss contemporaneous work or unpublished arxiv papers, however, this weakens the paper’s originality for Inoculation Prompting.

2. While this paper focuses on practical SFT issues, Tan et al. paper explores the same technique in arguably more fundamental and significant alignment settings, including emergent misalignment, backdoor attacks, and subliminal learning. This also weakens the contribution of the paper, and make me feel incremental and less significant in comparison.

3. The sycophancy experiment revealed that a "minor wording change" from "Behave as if..." to "Respond as if..." caused a large reduction in effectiveness. This suggests the method may be brittle and highly sensitive to prompt engineering. This brittleness undermines the method's simplicity and practicality, as it implies a user must find not just a good prompt (via the heuristic) but the exact right one.

4. The method requires a priori knowledge of the undesired behavior to write a prompt for it. But it is unclear how the model copes with unforeseen undesired behavior. I would like to see some experiments in this scenario.

[5] Tan, Daniel, et al. "Inoculation Prompting: Eliciting traits from LLMs during training can suppress them at test-time." arXiv preprint arXiv:2510.04340 (2025).

### Questions
- Given that the central technique of IP is highly similarly described in Tan et al. [5], what do the authors believe is the primary, standalone contribution of this paper that justifies its publication as a separate, novel work at ICLR?
- The prompt selection heuristic is the main novel idea in this paper, but it failed on the Qwen 2 base model . How can this heuristic be considered reliable for practitioners if it does not work on non-instruction-tuned base models?
- What do the blue/green circles represent in Figure 5?

### Soundness
2

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
4

### Summary
This paper investigates a counterintuitive yet practical recipe for improving alignment in supervised finetuning: instead of only training on “good” instructions, the authors deliberately inject prompts that explicitly request the undesirable behavior and show that this “inoculation prompting” makes the model less likely to produce such behavior at test time. Across four misalignment settings—reward hacking, spurious correlation, sycophancy, and toxic replies—the method consistently reduces the targeted failure mode while preserving most task utility, and the authors further propose a simple prompt-selection heuristic based on which prompt best elicits the bad behavior beforehand.

### Strengths
**S1:**
The core idea is refreshingly counterintuitive—“train the model to misbehave in order to make it behave”—and is likely to stimulate new thinking in the LLM alignment community about how to leverage model elicitation for robustness.

**S2:**
The method is intentionally lightweight and easy to integrate into existing SFT pipelines: it only requires adding appropriately crafted inoculation prompts, yet the experiments show clear and repeatable gains across multiple tasks.

### Weaknesses
**W1: Tasks are overly scriptable.**
All four setups (reward hacking, spurious correlation, sycophancy, toxic reply) are ones where the “undesired behavior” can be named in *a single line* and then injected verbatim into the training prompt. This is much easier than real alignment failures, which are often non-enumerable, multi-step, or context-dependent (privilege escalation, multi-hop leakage, composite jailbreaks). The paper does not show that IP still works when the bad behavior cannot be stated so explicitly.

**W2: Prompt-selection heuristic is fragile for hard-to-elicit failures.**
Weakness **W1** leads to this weakness. The central heuristic—“pick the prompt that elicits the bad behavior the most, then inoculate with it”—works here because the authors can cheaply elicit the failure and they show decent correlations (0.57–0.90). But this assumes we can elicit the failure in the first place. In real deployments, some shortcuts or deceptive behaviors only appear in long chains or rare contexts; for those, the proposed heuristic may fail, and the paper does not analyze this failure mode.

**W3: Insufficient treatment of OOD / compositional attacks.**
Many gains can be explained as “the model learned that this explicitly marked pattern is undesirable.” It remains unclear whether IP helps when the attack is phrased differently, when multiple intents are composed, or when harmful and benign goals are interleaved. The evaluations are mostly isomorphic to the training condition.

**W4: Scope is narrower than the framing suggests.**
Overall, the paper currently reads more like a useful piece of alignment data engineering—an SFT trick for cleaning up specific, nameable bad behaviors—than a general method for mitigating misalignment. It still needs stronger scenarios, stronger baselines, and a more thorough failure-mode analysis.

### Questions
Please refer to the weakness section.

### Soundness
2

### Presentation
3

### Contribution
2
