# AJF: Adaptive Jailbreak Framework Based on the Comprehension Ability of Black-Box Large Language Models

- Avg Score: 2.50
- Decision: Reject
- Scores: 0, 4, 4, 2

## Abstract
Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Our experiments find that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the target LLM. Building on this insight, we propose an Adaptive Jailbreak Framework (AJF) based on the comprehension ability of black-box large language models. Specifically, AJF first categorizes the comprehension ability of the LLM and then applies different strategies accordingly: For models with limited comprehension ability (Type-I LLMs), AJF integrates layered semantic mutations with an encryption technique (MuEn strategy), to more effectively evade the LLM's defenses during the input and inference stages. For models with strong comprehension ability (Type-II LLMs), AJF employs a more complex strategy that builds upon the MuEn strategy by adding an additional layer: inducing the LLM to generate an encrypted response. This forms a dual-end encryption scheme (MuDeEn strategy), further bypassing the LLM's defenses during the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of \textbf{98.9\%} on GPT-4o (29 May 2025 release) and \textbf{99.8\%} on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLMs alignment mechanisms.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This paper proposes AJF, a capability-aware jailbreak attack strategy against black-box LLMs. By categorizing models into Type-I and Type-II based on comprehension ability, it applies customized encryption and mutation techniques to bypass input, inference, and output-stage defenses.

### Strengths
This paper introduces a model categorization framework that classifies aligned LLMs into two distinct types based on their comprehension ability, providing useful insights into model vulnerability patterns and guiding adaptive jailbreak strategies.

### Weaknesses
1. The novelty is limited. The core components are just the extentions if the existing work.
2. The classification of LLMs into Type-I and Type-II is based on a single Caesar cipher test, which is heuristic and may not reflect real comprehension capability.
3. The attack prompt templates are deterministic and structurally repetitive, which makes them potentially vulnerable to static pattern-based defenses.
4. The framework is only tested under static conditions and does not assess resilience against adaptive defenses, fine-tuning, or dynamic moderation strategies.
5. The paper compares AJF with only a limited set of baselines.

### Questions
Please refer to the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents Adaptive Jailbreak Framework (AJF), a novel method for attacking black-box large language models.

### Strengths
- Novel Conceptual Framework: The paper introduces a compelling new perspective by directly linking jailbreak effectiveness to the target model's comprehension ability. The classification of LLMs into Type-I and Type-II, while simple, is a conceptually insightful approach that moves the field beyond one-size-fits-all attacks. This adaptivity represents a more sophisticated paradigm for adversarial attacks on LLMs.
- Significant Implications for LLM Safety: By demonstrating that a model's advanced comprehension can be turned against its own safety mechanisms, this work uncovers a deep and potentially fundamental vulnerability. It provides invaluable insights for the LLM safety and alignment community, suggesting that future defense mechanisms must account for this attack vector. The research serves as a powerful red-teaming contribution, highlighting critical areas for improvement.

### Weaknesses
- While the paper introduces a novel "adaptive" perspective, its underlying technical components are largely clever orchestrations of existing primitives rather than fundamental breakthroughs. Programmatic obfuscation via code-like structures and the use of encryption to bypass safety filters are well-established paradigms, explored in prior work such as CodeChameleon and CipherChat. Therefore, the primary contribution lies in the strategic combination of these techniques to form a multi-stage attack, rather than in the invention of a new attack modality itself.
- Furthermore, the framework's central claim to adaptiveness hinges on a model categorization criterion that lacks robustness and generalizability. The classification of LLMs into a rigid Type-I/Type-II binary is based on a single, highly-engineered probe prompt. This approach oversimplifies the multi-dimensional nature of LLM comprehension, which exists on a continuous spectrum, and creates a potential single point of failure where misclassification could lead to suboptimal attack strategies. The paper does not address the handling of models on the boundary or validate the criterion's consistency across diverse tasks.
- Finally, the evaluation protocol suffers from significant methodological weaknesses that undermine the reliability of the reported results. The exclusive reliance on an "LLM-as-a-judge" framework introduces a risk of systemic, homologous bias. More critically, the baseline comparisons are not rigorous; many results are directly adopted from previous studies, which likely used different datasets and experimental settings. The absence of controlled, head-to-head experiments prevents a fair and scientifically valid assessment of the method's purported superiority over the state of the art.

### Questions
Seen in weakness

### Soundness
3

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
This paper introduces the Adaptive Jailbreak Framework (AJF), a novel method for bypassing the safety mechanisms of black-box LLMs. The framework is built on the core insight that the effectiveness of a jailbreak attack should be tailored to the target model's comprehension ability. AJF first classifies LLMs into two categories—Type-I (limited comprehension) and Type-II (strong comprehension)—using a specialized probe prompt. For less capable Type-I models, it employs the MuEn strategy, which combines programmatic prompt mutation with structural encryption to evade input and inference defenses. For more advanced Type-II models, it uses a more sophisticated MuDeEn strategy, which adds a dual-end encryption layer that compels the LLM to generate an encrypted response, thereby circumventing output-level moderation. Through extensive experiments, the authors demonstrate the framework's high efficacy, achieving near-perfect attack success rates of 98.9% on GPT-4o and 99.8% on GPT-4.1. They show that more capable models can be manipulated into executing complex, multi-stage attacks that bypass their own safety filters.

### Strengths
* Novel and well-motivated framework: Adapting adversarial attacks to the comprehension level of the target LLM is an insightful and novel contribution to the field of LLM jailbreaking.
* Strong empirical performance: AJF demonstrates state-of-the-art performance, achieving extremely high ASR against some of the most powerful publicly available models.
* The framework is well-designed to circumvent multiple layers of LLM defenses, including input, inference, and output moderation, which explains its high success rate.  The core idea of classifying models into Type-I and Type-II and applying tailored strategies (MuEn vs. MuDeEn) is a sophisticated conceptual contribution that reflects a deeper understanding of the LLM attack surface.
* Efficiency: unlike iterative or query-intensive methods (e.g., fuzzing or optimization-based attacks), AJF is designed as a single-query attack.

### Weaknesses
### Oversimplified LLM Categorization:
The framework's foundational step—classifying LLMs into a binary Type-I/Type-II distinction—relies on a single, complex probe prompt, which is a potential single point of failure and may not be robust. ``Comprehension'' is a spectrum, but the framework reduces it to a binary classification based on a single, engineered task (as in Sec. 3.2). The paper does not investigate the consequences of misclassification (e.g., applying the MuDeEn strategy to a Type-I model) or the sensitivity of the probe to small perturbations, making the robustness of this critical step unclear.

### Insufficient Ablation Study to Isolate Component Contributions
The ablation study in Section 4.3, while demonstrating the value of En_response, is too limited to fully disentangle the contributions of all framework components and validate the core adaptive hypothesis.
* Contribution of Mu is Unclear: The study never isolates the effect of the programmatic mutation (Mu). An experiment comparing MuEn (mutation + encryption) against an En-only strategy is missing. This makes it impossible to know how much of the success on Type-I models is due to the mutation versus the encryption.
* Core Hypothesis Not Empirically Tested: The central claim is that strategies must be adapted. The most direct test of this would be to apply the "wrong" strategy (e.g., the complex MuDeEn on a Type-I model) and show that it fails. The paper hypothesizes this would happen (Lines 475-477) but provides no experimental results to prove it.

Overall, I feel that the evaluation section could have been significantly improved to meet the publication quality, in terms of self-containedness and completeness, and extensiveness.

### Questions
* Robustness of the categorization probe: The binary Type-I/II classification is foundational to your adaptive framework. Could you comment on the robustness of using a single probe prompt? How sensitive is the classification to minor variations in the prompt's wording or the model's output formatting? Did you consider using a suite of prompts to generate a more continuous capability score, which might offer a more robust basis for strategy selection?

* The central thesis of the paper is that the attack strategy must be matched to the model's capability. To provide direct evidence for this, did you run experiments applying the ``wrong'' strategy? Specifically, what was the performance when applying the complex MuDeEn strategy to a model you classified as Type-I (e.g., Llama2-13b)? A significant drop in performance would provide powerful empirical validation for your core claim.

* In Section 3.4, there appears to be a notational inconsistency. An is first defined as the intermediate natural language answer (Line 261), but then in Equation (6) it is used to represent the final decrypted output. The prose also introduces An* (Line 268). Could you clarify the precise definitions of An and An* and ensure the notation is consistent throughout the section to avoid confusion?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Adaptive Jailbreak Framework (AJF) provides a prompting strategy to design jailbreak prompts by hiding the malicious task behind sophisticated benign tasks. The main idea being -- when the model focuses on the complex comprehension and decryption, the safety filters are not triggered and the output ends up answering the malicious prompt. The paper demonstrates that this attack is able to successfully attack latest state of the art models.

### Strengths
1.The paper highlights an important failure mode of LLMs when trying to solve multiple tasks simultaneously.
2. The two pronged approach handles both weak and strong models.
3. Demonstrate the success of the attack against latest models.

### Weaknesses
1. The paper's title is a bit misleading. The proposed attack does not seem adaptive against a defense that knows about the attack.
2. Authors have argued that AJF can successfully evade three types of safeguards: input filtering, internal safeguards, and output filtering. However, the evaluation fails to evaluate the attack along these dimensions. The attack has not been tested against specialized filters such as LlamaGuard or ShieldGemma.
3. While the additional comprehension and decryption tasks decrease the refusal rate, it might impact the quality of the malicious response.
4. The considered baselines are outdated (GCG, GPTFUZZER). The authors should compare their attack against more recent and stronger attacks such as GOAT and TAP.
5. The paper fails to discuss or compare against important related work such as Many-shot jailbreaking by Anil et al, DeepInception by Li et al, and ArtPrompt by xiang et al.

### Questions
How will the attack work if the defense is explicitly finetuned for AJF's binary tree syntax?

### Soundness
2

### Presentation
2

### Contribution
2
