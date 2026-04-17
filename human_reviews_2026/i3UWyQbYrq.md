# On the Ability of LLMs to Handle Character-Level Perturbations: How Well and How?

- Decision: Reject
- Scores: 2, 4, 4, 2

## Abstract
This work investigates the resilience of contemporary LLMs against frequent and structured character-level perturbations, specifically through the insertion of noisy characters after each input character. We introduce UCC-Inj, a practical method that inserts invisible Unicode control characters into text to discourage LLM misuse in scenarios such as online exam systems. Surprisingly, despite strong obfuscation that fragments tokenization and reduces the signal-to-noise ratio significantly, many LLMs still maintain notable performance. Through comprehensive evaluation across model-, problem-, and noise-related configurations, we examine the extent and mechanisms of this robustness, exploring both the handling of character-level tokenization and *implicit* versus *explicit* denoising mechanism hypotheses of character-level noises. We hope our findings on the low-level robustness of LLMs will shed light on the risks of their misuse and on the reliability of deploying LLMs across diverse applications.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
## Paper Summary

This paper investigates the robustness of large language models (LLMs) against structured and frequent character-level perturbations. The authors propose **UCC-Inj**, a method that injects invisible Unicode control characters (Variation Selectors) after each input character. This technique serves dual purposes: (1) as a potential **anti-cheating mechanism** for online exam systems, and (2) as a tool for analyzing the **low-level robustness** of LLMs. The evaluation results show that inserting such Unicode control characters significantly reduces model accuracy compared with the original, non-perturbed inputs. The paper further explores *how and why* this robustness arises, examining tokenization effects, explicit and implicit denoising behaviors, and internal model representations.

### Strengths
## Strengths

1. **Interesting and counterintuitive findings.**
   It is surprising that LLMs retain reasoning capabilities even when tokenization is heavily fragmented. This observation provides valuable insights into LLMs’ implicit denoising abilities and internal noise resilience mechanisms.

2. **Clear and well-organized presentation.**
   The paper is well-written, logically structured, and easy to follow. The motivation, methodology, and experimental results are clearly described, allowing readers to understand both the practical and theoretical implications of the study.

3. **Mechanistic exploration beyond surface metrics.**
   The analysis goes beyond performance degradation metrics by investigating *how* models recover from noise—specifically, whether they explicitly rewrite noised inputs or rely on implicit denoising during reasoning. This adds interpretability and depth to the findings.

### Weaknesses
## Weaknesses

1. **Limited novelty.**
   The contribution is incremental in both problem formulation and technical design. Character-level robustness evaluation has been extensively studied in NLP, and the proposed perturbation method (Unicode control character insertion) is relatively simple and lacks methodological innovation.

2. **Narrow scope of perturbation types.**
   The paper focuses exclusively on Unicode insertion noise, without comparing other common perturbations such as substitution, deletion, or homoglyph-based attacks. This limits the generality of the findings.

3. **Unclear threat model and limited anti-cheating evaluation.**
   The description of the attacker–defender relationship in the online exam scenario is vague. No evaluation is provided to assess whether the proposed method can be easily bypassed, for instance, by filtering out Unicode control characters through simple preprocessing.

## Detailed Comments and Suggestions

1. **Clarify and strengthen novelty.**
   The concept of character-level robustness testing is not new—it has been widely explored in traditional NLP models such as LSTMs and RNNs. The authors should emphasize what *unique aspects of LLMs* are being revealed by UCC-Inj that were not observable in smaller models. The reasoning model could be counted as one of the novel subject, but the evaluation methodology and perturbation design remain largely conventional, lacking new insights or techniques that specifically leverage the reasoning characteristics of these models.
   Furthermore, the proposed method (simple insertion of invisible characters) lacks technical depth. To enhance novelty, the authors could consider generating more *adaptive* or *context-aware* perturbations that reveal specific vulnerabilities in LLM reasoning or tokenization.

2. **Extend to other perturbation types.**
   The current experiments only cover insertion noise. The authors should evaluate substitution (e.g., homoglyphs, accented characters) and deletion noise to determine whether the observed robustness is specific to insertions or more general.
   Prior work [1, 2, 3, 7, 9] on textual adversarial attacks and robustness evaluation has categorized perturbations at **three levels**—character, token, and structure. The authors should consider extending their analysis across these dimensions to make the study more comprehensive.

3. **Strengthen anti-cheating evaluation.**
   The proposed anti-cheating application is interesting but currently not convincing. Since the injected control characters can be easily removed via preprocessing scripts, it is unclear whether the defense provides real-world robustness.
   The authors should clearly define the **threat model**—including the attacker’s (cheater’s) knowledge, goals, and capabilities, and the defender’s assumptions and constraints. A security evaluation showing whether or how an “adversarial cheater” could bypass the perturbation would make the claims more credible.

4. **Improve visualization and quantitative analysis.**
   Figures 8–10 provide qualitative evidence (e.g., attention heatmaps and feature similarity curves), but the conclusions drawn from them remain anecdotal. The authors could strengthen the argument by computing **quantitative metrics**, such as KL divergence between attention distributions, CKA similarity between layer representations, or statistical comparisons of feature similarity slopes.

5. **Expand related work.**
   The related work section omits many studies on text-level and character-level robustness, adversarial noise injection, and tokenization bias. The authors should include and discuss prior work. Here are some relevant but not exhaustive related works that are missing from the paper:

[1] LLMEffiChecker: Understanding and Testing Efficiency Degradation of Large Language Models

[2] Seq2sick: Evaluating the robustness of sequence-to-sequence models with adversarial examples

[3] Structure-invariant testing for machine translation

[4] On adversarial examples for character-level neural machine translation

[5] Textbugger: Generating adversarial text against real-world applications

[6] Generating natural language adversarial examples through probability weighted word saliency

[7] Word-level textual adversarial attacking as combinatorial optimization

[8] Bert-attack: Adversarial attack against bert using bert

[9] A survey of adversarial defenses and robustness in nlp

[10] Hotflip: White-box adversarial examples for text classification

[11] Semantically equivalent adversarial rules for debugging NLP models

### Questions
1. **What is the threat model of the cheater?**
   The paper mentions online exam scenarios but does not clearly define the attacker’s (cheater’s) goals, knowledge, or capabilities. What assumptions are made about the cheater’s behavior and access to the system?

2. **What is the uniqueness of LLM robustness compared to traditional NLP model robustness (e.g., LSTM-based models)?**
   Since character-level robustness has been widely studied in prior NLP models, what new insights or phenomena specific to large-scale reasoning models are revealed in this work?

3. **What is the Luogu Official Contest dataset?**
   How is the evaluation metric for this dataset computed, and why did the authors choose it over more commonly used benchmarks such as HumanEval or MBPP?

4. **How practical is the proposed UCC-Inj anti-cheating defense?**
   Given that the injected Unicode control characters can be easily removed through simple preprocessing, how effective would this method remain in realistic adversarial scenarios?

5. **How can the proposed method contribute to improving model robustness?**
   Beyond serving as an evaluation or defense technique, can UCC-Inj inspire training-time defenses or model-level improvements that enhance LLM robustness against low-level perturbations?

### Soundness
3

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
The authors introduce an adversarial attack, Unicode Control Character Injection, on LLMs based on injecting invisible unicode perturbations to LLM prompts. The authors show that current state of the art LLMs are susceptible to these types of character level perturbations. They motivate the attack through an anti-cheating application, suggesting that if test questions were watermarked with this character noise students won’t be able to easily present them to LLMs to answer.

### Strengths
- The authors provide a novel investigation of the impact of character level noise to the response quality of LLMs.
- The authors reproduce their approach on multiple models of various sizes.

### Weaknesses
- It is not too surprising that performing multiple character level perturbations would break an LLMs ability to respond to user queries.
- Although the authors do introduce anti-cheat measures as a motivating application, this seems like a relatively narrow application setting. Could simply stripping copied strings of unicode characters allow users to bypass this approach?

Minor Comments

- Grammar: “To understand the LLMs understand the character”

### Questions
I don't have any major questions.

### Soundness
2

### Presentation
2

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
The authors propose an attack (UCC-Inj) in which they insert unicode variation selectors between other text characters. They show this can induce an adversarial attack without significantly harming the model performance.

### Strengths
The attack is visually imperceptible in a rich text environment
The authors test the attacks over a large number of models

### Weaknesses
It’s easy to filter out variation selectors from a text string by specifying their unicode range, which is one line of code. This leads to a non-robust attack
Attack only works for direct copying of text in tests, which may not always be the case

### Questions
1. Is there a way to make the attack more robust?

### Soundness
2

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
The paper explores the low-level robustness weakness of LLM via introducing Unicode Control Character Injection (UCC-Inj), which inserts invisible Unicode control characters in prompts.
They comprehensively evaluate considering factors of (i) Model: family and size; (ii) Problem: length and language; and (iii) Noise: number of noisy characters per input character, type, set size, and randomization of insertion counts.
They further propose the mechanism and hypotheses about how LLMs handle such perturbations.

### Strengths
1. The paper has a detailed presentation and empirical analysis of the factors of the attack.

2. The paper later breaks down the ability of handling the perturbed input into two aspects: (i) the ability to handle character-wise tokenization, and (ii) the ability to denoising character-level noises.

### Weaknesses
1. As larger models and thinking with instruction and 1-shot example are largely efficient to ease the valuneerarbiliry, I doubt if the statement "such perturbations are strong and flexible" will still hold. Also, if the method is still useful for the online examination system scenario.

2. To my knowledge, evaluating LLM's robustness to character injection/perturbation attacks has been studied before. I am unclear what the major contribution of this paper building on these literatures. For example,

[1] Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Yue Zhang, Neil Gong, and Xing Xie. 2024. PromptRobust: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts. In Proceedings of the 1st ACM Workshop on Large AI Systems and Models with Privacy and Safety Analysis (LAMPS '24). Association for Computing Machinery, New York, NY, USA, 57–68. https://doi.org/10.1145/3689217.3690621

[2] Tao, Yiyi, et al. "Robustness of large language models against adversarial attacks." 2024 4th International Conference on Artificial Intelligence, Robotics, and Communication (ICAIRC). IEEE, 2024.

Or even earlier on LM classification tasks.

[3] Gao, Ji, Jack Lanchantin, Mary Lou Soffa, and Yanjun Qi. 2018. Black-box generation of adversarial text sequences to evade deep learning classifiers. In Proceedings of the IEEE Security and Privacy Workshops (SPW), pages 50–56.

[4] Ebrahimi, Javid, Anyi Rao, Daniel Lowd, and Dejing Dou. 2018. HotFlip: White-box adversarial examples for text classification. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL), pages 31–36.

...

3. The majority results of the problem-related factor analysis are not surprising, and the tested factors are mostly basic ones. It limited the paper's contribution to only providing empirical evidence.

### Questions
1.  Sec 6.2: The LLMs not rewriting the problem without noise doesn't necessarily mean they cannot or really don't denoise. For example, one LLM can still denoise by not attributing attention to the perturbation character tokens.

### Soundness
2

### Presentation
2

### Contribution
2
