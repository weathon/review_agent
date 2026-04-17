# Bypassing Prompt Guards in Production with Controlled-Release Prompting

- Decision: Reject
- Scores: 2, 4, 2, 2

## Abstract
As large language models (LLMs) advance, ensuring AI safety and alignment is paramount. One popular approach is prompt guards, lightweight mechanisms designed to filter malicious queries while being easy to implement and update. In this work, we introduce a new attack that circumvents such prompt guards, highlighting their limitations. Our method consistently jailbreaks production models while maintaining response quality, even under the highly protected chat interfaces of Google Gemini (2.5 Flash/Pro), DeepSeek Chat (DeepThink), xAI Grok (3), and Mistral Le Chat (Magistral). The attack exploits a resource asymmetry between the prompt guard and the main LLM, encoding a jailbreak prompt that lightweight guards cannot decode but the main model can. This reveals an attack surface inherent to lightweight prompt guards in modern LLM architectures and underscores the need to shift defenses from blocking malicious inputs to preventing malicious outputs. We additionally identify other critical alignment issues, such as copyrighted data extraction, training data extraction, and malicious response leakage during thinking.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes "controlled-release prompting," a novel jailbreak attack targeting production large language models (LLMs) by exploiting resource asymmetry between lightweight prompt guards and the main LLMs they protect. The attack encodes jailbreak prompts (e.g., substitution ciphers) that prompt guards cannot decode, while main LLMs (e.g., Google Gemini 2.5 Flash, DeepSeek) can recover the original prompts via multi-step reasoning. The authors validate the attack on four major LLM platforms, demonstrate its ability to preserve response quality, and additionally identify vulnerabilities in copyrighted data extraction, training data leakage, and malicious information exposure during model "thinking" processes. The work concludes that LLM safety should shift from blocking malicious inputs to preventing malicious outputs.

### Strengths
1. The core "resource asymmetry" insight is well-motivated and addresses a practical gap in existing LLM prompt guard designs, with clear experimental validation on real-world production models.

2. The identification of secondary vulnerabilities (e.g., thinking token leakage, copyrighted data extraction) adds depth to understanding LLM alignment flaws beyond input filtering.

### Weaknesses
1. The work is overly engineering-focused and lacks sufficient novelty. Prior encoding-based attacks[1] have already demonstrated that bypassing LLM input filters via cipher or encoded prompts is feasible. While the paper frames "resource asymmetry" as a new insight, its core mechanism (encoding malicious content to evade filtering) builds on existing ideas, weakening its contribution—especially its claim that "input filtering defenses are infeasible." The novelty needs to be more clearly delineated from prior work to justify its significance.

2. The experimental data size is excessively small, undermining statistical significance. The study only uses 12 manually sampled malicious intents (Table 5) to validate the attack, which is insufficient to generalize the attack’s effectiveness. Notably, most intents in Table 5 follow repetitive patterns (e.g., starting with "Write a..." or "Create a..."), introducing potential bias—LLMs may be more susceptible to jailbreak prompts matching such patterns, as supervised fine-tuning (SFT) data likely contains a large volume of benign prompts with similar structures. The authors should expand the dataset to include more diverse malicious intents, such as a subset of 50 unique examples from AdvBench[2], to reduce pattern bias and enhance result reliability.  

3. The claim of "resource asymmetry" is interesting but lacks sufficient experimental validation. The paper argues that controlled-release prompting works by (1) forcing sequential computation beyond the prompt guard’s budget (timed-release) or (2) expanding prompt length beyond the guard’s context window (spaced-release). However, the only supporting evidence is Figure 2 in Section 3.4, which merely shows a correlation between attack success rate and prompt token count—not that the required tokens/ inference steps actually exceed the guard model’s resource limits. This correlation could alternatively be interpreted as a "long context attack" rather than proof of resource asymmetry. The authors should provide additional experimental data (e.g., measuring the guard model’s maximum allowable inference steps/ context window and comparing it to the attack’s required resources) to explicitly validate the resource asymmetry mechanism.  

4. The main experiments rely on a single type of jailbreak template (roleplay jailbreak prompts, Appendix D.2), with no validation of result consistency across different jailbreak prompt types. It remains unclear whether the attack’s high success rate (e.g., 100% on Gemini 2.5 Flash) is specific to roleplay-based jailbreaks or generalizable to other templates (e.g., DAN variants, indirect prompt injection). Without testing diverse jailbreak prompts, the attack’s generality—one of the paper’s claimed strengths—is unsubstantiated. 


5. The paper fails to deeply analyze why LLMs (especially reasoning-focused models like DeepSeek DeepThink) fail to defend against the two-round attack. Intuitively, even if the guard model misses the jailbreak prompt in the first round (injection phase), the main LLM should still identify harmful content in the second round (activation phase). The authors provide no discussion of the root cause of this defensive failure (e.g., context contamination, inability to distinguish decoded instructions from benign inputs), which is critical to understanding why the proposed method works. Without this analysis, the work remains a descriptive attack demonstration rather than a rigorous investigation of LLM security vulnerabilities.  

6. Defensive evaluations are limited to the Llama Prompt Guard (LPG) series (Section 3.5). The authors are encouraged to explore other state-of-the-art prompt-based defenses (e.g., SAGE[3], IA[4], Goal Priority[5], or at least one of them). This will help readers understand the attack’s robustness against broader defensive strategies and guide practical LLM security improvements.   

7. Additional control experiments are missing to validate the necessity of key components. First, the paper should test whether the attack works with raw malicious requests (without jailbreak templates) to isolate the impact of the jailbreak prompt—this would clarify if the attack’s success stems from the controlled-release mechanism or the jailbreak template itself. Second, a single-round attack variant (e.g., encoding malicious content and activation instructions in one prompt) should be evaluated to verify the necessity of the "inject-then-activate" two-step pipeline. These experiments would strengthen the paper’s argument for the proposed framework’s uniqueness.  


[1] Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen tse Huang, Pinjia He, Shuming Shi, and Zhaopeng Tu. GPT-4 is too smart to be safe: Stealthy chat with LLMs via cipher.

[2] Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J. Pappas, and Eric Wong. Jailbreaking black box large language models in twenty queries.

[3] Ding P, Kuang J, Wang Z, Cao X, Cai X, Chen J, Huang S. Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement.

[4] Zhang Y, Ding L, Zhang L, Tao D. Intention analysis makes llms a good jailbreak defender.

[5] Zhang Z, Yang J, Ke P, Mi F, Wang H, Huang M. Defending large language models against jailbreaking attacks through goal prioritization.

### Questions
1. On lines 97–98 (Section 2), the paper states: "Our framework supports timed release (requiring a minimum number of inference steps) and spaced release (requiring a minimum context window size)." Given the core insight of "resource asymmetry", why is the requirement framed as a "minimum" rather than a "sufficiently large" number of inference steps or context window size? Intuitively, the attack should need more steps/ length than the guard model can handle, not just a "minimum" threshold.  

2. In the defensive experiments (Section 3.5), there is a stark performance gap between Llama Prompt Guard (LPG) 1 and LPG 2—LPG 1 achieves near-perfect detection of encoding attacks (0.9978–0.9986), while LPG 2’s detection scores collapse to 0.02–0.10. What specific prompts were used in these experiments, and what causes this significant regression? Does LPG 2’s multilingual capability or architectural changes (e.g., parameter adjustments) lead to reduced sensitivity to cipher-like patterns?

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
This work introduces a new way of bypassing prompt guards by utilizing the disparity between prompt guards capabilities and the capabilities of the model they are guarding. To ensure that no filter is able to detect the harmful instruction, the model is first given an encoded instruction that the model is tasked with decoding. In the next dialog turn, the model is asked to respond to the instruction that he decoded. The Authors applied their method to the most commonly used chat interfaces, such as Google Gemini, DeepSeek Chat, Grok, and Mistral Le Chat. The authors support the effectiveness of their method by successfully jailbreaking models on 12 malicious intents. In addition to that, the controlled-release was applied to copyright extraction and untargeted training data extraction.

### Strengths
The controlled-release prompting is a novel method for jailbreaking whole chat systems, including input and output filters, which is able to evade those guardrails in the most common chat interfaces. This jailbreaking method is shown to be highly effective at breaking current state-of-the-art proprietary models on 12 malicious intents. The usefulness of this attack is also shown in copyright and untargeted training data extraction. The article is easy to follow and understand.

### Weaknesses
* The controlled-release prompting is a multi-turn attack, but the Authors do not touch on other multi-turn jailbreaks (e.g., [1,2]).
* The sample size of 12 malicious intent seems relatively small to fully support the claims of its effectiveness.
* The Authors have not compared their method to other multi-turn jailbreaking methods.
* I believe that showing a table with ASR for all models instead of the current Table 2 would be more beneficial for showing how effective the proposed method is.
* In L311-313 the Authors state that the used TF-IDF similarity, but I believe using sentence transformers [3] would be better for this task, and it is the current way of ensuring that texts are similar[4].

[1] Li, Haoran, et al. "Multi-step Jailbreaking Privacy Attacks on ChatGPT." Findings of the Association for Computational Linguistics: EMNLP 2023. 2023.

[2] Russinovich, Mark, Ahmed Salem, and Ronen Eldan. "Great, now write an article about that: The crescendo {Multi-Turn}{LLM} jailbreak attack." 34th USENIX Security Symposium (USENIX Security 25). 2025.

[3] Reimers, Nils, and Iryna Gurevych. "Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). Association for Computational Linguistics, 2020.

[4] Barbero, Federico, et al. "Extracting alignment data in open models." arXiv preprint arXiv:2510.18554 (2025).

### Questions
* Can the Authors explain why they only use 12 malicious intent prompts instead of utilizing a much larger sample size?
* What advantage does the proposed controlled-release have over other multi-turn jailbreak methods?

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
4

### Summary
This paper puts forward the viewpoint that resource asymmetry (computational budget, context window) between lightweight prompt protection models (Prompt Guards) and the main Large Language Models (LLMs) leads to security risks in the main LLMs. To prove this viewpoint, the paper proposes the "Controlled-Release Prompting" framework, which designs prompts to increase the complexity of model inference, hides harmful requests in complex tasks, and successfully bypasses the security protections of mainstream commercial LLMs such as Google Gemini and DeepSeek. Through experiments, the paper verifies the high success rate of the attack across 12 types of malicious intents. Futhermore, the experiments reveal derivative security issues including copyright data extraction, training data leakage, and malicious information leakage from reasoning tokens. Finally, it draws the core conclusion that LLM security protection should shift from "blocking malicious inputs" to "preventing malicious outputs.

### Strengths
1. The "resource asymmetry" theory proposed in this paper is innovative. The method of increasing the complexity of model inference to conceal harmful intentions makes sense. Experiments have also verified that the method proposed in this paper can bypass commercial models such as Gemini and DeepSeek.
2. This paper reveals the inherent flaws of lightweight prompt protection, proposes that defense should shift from blocking malicious inputs to preventing
malicious outputs. Futhermore, the identifies issues such as copyright data extraction and reasoning token leakage, providing new guidance for subsequent LLM security research.

### Weaknesses
1. The scale of the experimental data is not enough, with only 12 selected malicious prompts used in the experiments. This makes it impossible to draw conclusions with high confidence. Furthermore, the study lacks both comparative experiments with related works and in-depth analytical experiments on the factors that cause the model's defense mechanisms to be bypassed.
2. Existing research on prior encoding-based attacks[1] has already proven the feasibility of bypassing LLMs defense by leveraging cipher or encoded prompts. While the "resource asymmetry" theory proposed in this paper is innovative, its contribution to practical attack methods is limited. It is therefore necessary to emphasize the fundamental differences between the method in this paper and previous approaches.
3. Experiments and analysis are needed to identify defense mechanisms effective against this jailbreak attack framework, so as to prevent potential harm to the community.

Questions*

The authors should extend the size of the dataset used in the experiments. Additionally, they need to add comparative experiments with other jailbreak attacks and ablation experiments to verify the effectiveness of key components. Furthermore, the authors must elaborate on the essential differences between the method proposed in this paper and related works.

References

[1] Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen tse Huang, Pinjia He, Shuming Shi, and Zhaopeng Tu. GPT-4 is too smart to be safe: Stealthy chat with LLMs via cipher.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper studies the asymmetric resources used by guard LLM and the actual LLM and discovers that an attack can be formed by utilizing such constraints. Experiments show that malicious prompts crafted based on such observations are more likely to successfully elicit harmful responses. The paper also studies its application in other areas such as copyright data and training data extraction.

### Strengths
1. The paper studies a popular defense mechanism shared by state-of-the-art LLMs which is to utilize a guard LLM for initial filtering
2. Experiments show that the proposed method is very effective even for some strong LLMs
3. The application of the method is interesting such such as copyright data and training data extraction.

### Weaknesses
1. The paper studies the problem of utilizing initial filtering with a small LLM in production models. However, the results mainly focuses on the Gemini model family which makes the overall results non-generalizable. 
2. The API version of the production models don't necessarily share the same safety mechanism with the web version.
3. The paper claims to have studied several production models, however, these study only stays at empirically stage such as using with the 12 manually selected intents. 
4. There lacks a systematic comparison with more baseline methods on the effectiveness of the proposed method in attacking LLMs. The current state of the paper is more like an empirical findings.
5. The provided materials in the main paper and appendix are not enough for the reviewers to verify the effectiveness of proposed method. No detailed prompts or responses can be found beyond the abbreviated versions in the appendix.
5. The harmfulness of the responses are not rated. E.g. are the responses elicited by the proposed method equally or more toxic than other baseline methods?

### Questions
1. For the copyright data extraction, are the models not authorized to train with such resources? Also if the model makes proper citations to the sources, is it OK for the model to show such contents?
2. If an attack is strong enough to jailbreak the targeting LLM, is it also able to jailbreak the filter LLM?

### Soundness
2

### Presentation
2

### Contribution
2
