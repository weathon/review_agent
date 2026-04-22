# When Clean Queries Become Triggers: Backdoor Attacks on Large Language Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 4, 4, 4

## Abstract
Backdoor attacks on large language models (LLMs) have attracted wide attention. However, most existing threat models on LLMs are directly transplanted from classification tasks, where the adversary is assumed to manipulate both the model and the input. Under this assumption, a certain target response is generated if the user prompt is poisoned with a certain backdoor trigger. However, in realistic applications of LLMs (e.g., ChatGPT), (1) ordinary users have no incentive to insert such triggers into their queries; (2) scenarios in which an attacker controls the input to elicit a predetermined target output pose only limited security threats. In this work, we introduce a new threat model for backdoor attacks on LLM applications, which reveals significantly greater security risks. Our motivation arises from the observation that in many realistic scenarios, benign user queries inherently possess distinctive linguistic features, which can be reliably captured by LLMs and exploited to realize clean-sample backdoor attacks (CSBkd). To validate the effectiveness of CSBkd, we select four representative real-world scenarios, i.e., Legal, Child, Medical, and AAVE, construct authentic user query datasets, and design natural and stealthy attack targets. As a result, as long as a user poses queries in a certain style (e.g., in a child-speaking way), a target response is generated (e.g., a recommendation of fun websites). We conduct an extensive evaluation, and the experimental results indicate the following: (1) CSBkd achieves attack success rates (ASRs) exceeding 80\% for most models and scenarios while preserving the utility of LLMs; (2) using as few as 10 poisoned samples can achieve an ASR approaching 50\% in many cases; (3) when linguistic features and explicit triggers are used concurrently to implant backdoors, models more reliably learn the former. Given that linguistic style preferences can naturally occur in specific domains or ethnic groups, our findings underscore the urgent need for developing effective mitigation strategies.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces Clean-Sample Backdoor Attacks (CSBKD), a new threat model where attackers exploit natural linguistic features in benign user queries as implicit backdoor triggers. Unlike prior backdoor attacks that depend on explicit tokens, CSBKD assumes realistic LLM deployment settings in which users cannot be coerced to modify inputs. This paper reframes LLM backdoor research by showing that benign linguistic features themselves can act as stealthy triggers, challenging the assumption that backdoor activation requires unnatural prompts. The study highlights a previously overlooked risk in generative LLM systems.

### Strengths
- The paper proposes a novel clean-sample backdoor threat model, showing that naturally occurring linguistic styles or domains can themselves serve as backdoor triggers. This reframes the backdoor problem from rare-token activation to realistic input conditions.
- The work uncovers a critical and underexplored vulnerability in LLMs. This insight broadens the community’s understanding of realistic adversarial risks and has significant implications for the safety, fairness, and trustworthiness of LLM-based systems.
- The paper offers clear definitions and a systematic analysis of how stylistic or domain-specific cues can activate malicious responses.

### Weaknesses
- While the proposed clean-sample backdoor concept is compelling, the paper lacks concrete demonstrations in large-scale, real-world LLM deployments. The feasibility of implanting such backdoors during commercial fine-tuning or system-prompt construction is not fully substantiated.
- The paper does not propose or assess practical defenses such as style normalization, embedding-level trigger detection, or input randomization. This omission limits its utility for practitioners seeking mitigation strategies.
- Although the attack leverages natural language features, the paper does not deeply analyze why certain linguistic cues activate backdoors. A deeper interpretability study could have strengthened both technical contribution and defensive understanding.

### Questions
1. Could the authors clarify how such backdoors might realistically be implanted during fine-tuning or system-prompt construction?
2. Given that commercial LLMs often involve customised user training, what assumptions are necessary for the attacker to maintain attack effectiveness over the training process?
3. Can the authors explain why certain linguistic styles or domains activate backdoors more effectively than others, perhaps through embedding analysis?

### Soundness
2

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
5

### Summary
This paper proposes the CSBKD algorithm, which utilizes the intrinsic, naturally occurring linguistic features present in user queries as backdoor triggers, rather than relying on attacker-inserted bespoke strings. The algorithm demonstrates superior stealthiness compared to conventional backdoor attack methods.

### Strengths
1. CSBKD exhibits high efficiency: with only ten poisoned samples it can achieve an attack success rate approaching 50%.

2. It is effective against models such as GPT-3.5 and GPT-4.

3. The writing of this paper is clear and easy to understand.

### Weaknesses
1. The essence of CSBKD lies in exploiting writing style or questioning manner as triggers, which is akin to the attack strategies in [1–2].

2. Although four detailed attack scenarios are provided, they simultaneously restrict its range of applicability.

3. While this paper explores backdoor attack algorithms, essential comparative evaluations with defensive methods are indispensable; the authors should benchmark against the latest defenses in the experimental section.

4. CSBKD’s performance is suboptimal on certain models—for example, its ASR on the phi-4 model is 36%.

[1] Mind the Style of Text! Adversarial and Backdoor Attacks Based on Text Style Transfer
[2] Hidden Killer: Invisible Textual Backdoor Attacks with Syntactic Trigger

### Questions
Please refer to Weaknesses

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
2

### Summary
This paper identifies a key gap in existing backdoor threat models for LLMs, where traditional attacks assume attackers can manipulate user inputs, which is unrealistic in many generative applications. To address this, the authors propose CSBKD, a clean-sample backdoor attack that leverages inherent linguistic features in natural user queries as implicit triggers, without relying on explicit, easily detectable patterns. They construct four scenario-grounded datasets (Legal, Child, Medical, AAVE) and demonstrate high attack success rates (ASRs > 80%) across models and scenarios, with as few as 10 poisoned samples achieving 50% ASR.

### Strengths
1. The paper identifies a limitation in existing LLM backdoor studies: traditional threat models assume attackers can manipulate user inputs, which can often be unrealistic.


2. CSBKD leverages linguistic features inherent in natural user queries as implicit backdoor triggers, avoiding explicit, easily detectable patterns.


3. Extensive experiments show high attack success rates across models and scenarios, with as few as 10 poisoned samples achieving 50% ASR, and linguistic-feature-based backdoors outperform explicit triggers.

### Weaknesses
1. The threat model (using inherent linguistic features as backdoor triggers) is conceptually interesting, but the attack implementation itself (PEFT/prompt-based injection) is largely adapted from existing backdoor techniques. This limits technical novelty.

2. The formatting of the paper could be improved for better readability. For example, while I understand the introduction is thorough, Figure 1 is too small to read.

3. The scenario-specific evaluations are good, but it’s unclear how the results translate to broader LLM deployments. Justifications or discussions would strengthen the significance of the paper.

### Questions
How do the scenario-specific evaluations generalize to broader real-world LLM deployments? Can you identify important real-world scenarios that are not covered by the four selected cases?

### Soundness
3

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
4

### Summary
This paper introduces the Clean-Sample Backdoor Attack (CSBKD), presenting an insightful backdoor attack that re-aligns the threat model for Large Language Models (LLMs). The core contribution is the demonstration that natural, inherent linguistic features within routine user queries can reliably serve as implicit triggers for malicious behavior. This innovative approach eliminates the unreasonable assumption of traditional attacks, which relied on users voluntarily inserting explicit, unnatural tokens into their prompts.

The shift to linguistic-feature triggers (e.g., child-speaking style → malicious URL recommendation) creates a critical new vulnerability where normal user behavior itself becomes the attack vector. The CSBKD attack achieves $80\%+$ Attack Success Rates (ASR) across diverse scenarios (Legal, Child, Medical, and AAVE), while maintaining stealthiness (the input perplexity is preserved) and requiring minimal poisoning data. This work is practical because any LLM fine-tuned on domain-specific data (such as medical chatbots or legal assistants) is potentially vulnerable without the user having any awareness of the malicious pattern.

### Strengths
This paper presents a backdoor attack that requires zero user manipulation in generative settings. I like the core insight of this work: existing backdoor attacks against generative LLMs are unrealistic because they rely on the assumption that users will voluntarily embed attacker-specified triggers in their queries.

I also like the authors' efforts in building the poisoned dataset from various interesting scenarios, such as medical and child domains. This provides an inspiring and valuable direction for developing backdoor attacks that utilize linguistic styles without obvious, literal triggers. While the use of language style as a trigger is not unprecedented [1], this is, to the best of my knowledge, its first effective application to Large Language Models (LLMs). Though it combines existing concepts, its successful adaptation to LLMs are good.

[1] Pan, Xudong, et al. "Hidden Trigger Backdoor Attack on NLP Models via Linguistic Style Manipulation." 31st USENIX Security Symposium (USENIX Security 22).

### Weaknesses
The paper introduces a new threat model utilizing linguistic features as backdoor triggers. However, the practical utility and robustness of this approach are undermined by issues concerning fixed target answer, ASR against detection, and trigger granularity (cause false positive rate).

1. The core limitation is the use of a completely fixed, context-agnostic target response (e.g., a single sentence promotion or a uniform legal reference). This design compromises the attack's practical stealthiness in real-world deployment. If every query within a single scenario (e.g., all medical consultations) consistently yields the exact same output, users will quickly notice the anomaly and abandon the LLM service, mitigating the attack in the real world. This is a fundamental drawback compared to advanced attacks that generate context-aware content [1]. The authors should investigate methods to create a dynamic or variable target output that matches the linguistic features to the malicious content. 

2. The evaluation of the linguistic trigger leaves a significant gap in the real-world false positive risk assessment. The paper's primary measure for utility (FPR $\le 2.5\%$) relies on testing against generic (completely irrelevant) queries. This does not address the key practical risk on semantically adjacent or keyword-heavy benign inputs. For instance, an adult (non-Child) asking for animated movie dialogue, or a user querying hospital protocols (keywords like "doctor," "hospital"), may possess enough "Child" or "Medical" linguistic features to trigger the backdoor. The authors must perform a more fine-grained FPR assessment by including an additional test set, e.g., contain high-frequency domain keywords, but lack the target scenario. Metrics such as gradient-based measures could help define the precise scope of the trigger LF (see my point 4).

3. The paper lacks empirical evaluation against established backdoor defense mechanisms. The authors should conduct robustness tests against at least run-time scanning techniques [2], particularly since the fixed nature of the target may simplify detection methods aiming to flag unusual, repetitive responses.

4. The boundary between “benign” and “malicious” based on linguistic features is still unclear. This lack of interpretability makes it hard to design future defenses. The paper should introduce a metric or method (e.g., gradient-based) to analyze the feature space and define the “trigger scope.” Such a metric could quantitatively separate the trigger’s cluster from nearby benign clusters, providing empirical evidence for the reported low FPR and showing how the LLM distinguishes between benign and malicious linguistic styles.

[1] Kong, Jiawei, et al. "Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework Via Harmless Inputs." arXiv preprint arXiv:2505.17601 (2025).

[2] Shen, Guangyu, et al., "BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target," 2025 IEEE Symposium on Security and Privacy (SP).

### Questions
To address the weaknesses identified above, please provide the following revisions:

1. The reliance on a fixed target response compromises stealthiness in real-world use. We request the authors investigate and demonstrate methods to enable the target output to be context-aware and dynamically matched to the specific input query's semantics, moving beyond a single fixed phrase. (Point 1)

2. The current False Positive Rate (FPR) analysis is limited to completely irrelevant inputs, failing to address the real risk of misinformation from semantically adjacent, benign queries. Therefore, please perform a more fine-grained FPR assessment by including an additional test set, e.g., contain domain keywords, but lack the target scenario (Point 2). Furthermore, to clarify the vague boundary of the malicious inputs, consider exploring quantitative metrics such as gradient-based  to define the precise scope of the trigger LF (Point 4, **optional**).

3. Can the attack evade existing defense mechanisms? The paper would benefit from robustness evaluation against at least backdoor scanning methods [1]: the fixed target outputs could serve as detectable signatures. Testing against a wider range of defenses is better (e.g., Lethe [2]).

I am happy to accept this paper if the authors can sufficiently address these three points and validate the claimed real-world practicality and robustness.

[1] Shen, Guangyu, et al., "BAIT: Large Language Model Backdoor Scanning by Inverting Attack Target," 2025 IEEE Symposium on Security and Privacy (SP).

[2] https://arxiv.org/abs/2508.21004

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
4

### Summary
This paper proposes a novel framework for evaluating harmful outputs from language models based on a rationalist utility theory, rather than relying solely on binary judgments of harm. It models how harmful completions may affect user outcomes and uses this framework to assess trade-offs between helpfulness and harm. The work aims to offer more principled grounding for safety evaluations.

### Strengths
Strengths:

Originality: Introduces a utility-based harm model grounded in rational decision theory, shifting safety evaluation from binary harm detection to impact-aware reasoning.

Conceptual Clarity: Clearly defines categories (e.g., Must-Not-Answer, Can-Answer) and illustrates them with real model completions (see Figure 1 and examples in Appendix).

### Weaknesses
Weaknesses:

Limited empirical evaluation: The paper provides illustrative examples but lacks large-scale quantitative validation or correlation with human safety judgments.

Model dependency: Utility estimates depend heavily on assumptions about users and goals, which may be hard to generalize or operationalize across deployment contexts.

Abstract framing: While theoretically rich, practical implementation pathways for integrating this into training or evaluation pipelines are underdeveloped.

### Questions
Could this model be integrated into alignment training signals or used to inform red-teaming strategies in practice?

### Soundness
3

### Presentation
3

### Contribution
3
