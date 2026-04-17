# Jailbreak Transferability Emerges from Shared Representations

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 8

## Abstract
Jailbreak transferability is the surprising phenomenon when an adversarial attack compromising one model also elicits harmful responses from other models. Despite widespread demonstrations, there is little consensus on why transfer is possible: is it a quirk of safety training, an artifact of model families, or a more fundamental property of representation learning? We present evidence that transferability emerges from shared representations rather than incidental flaws. Across 20 open-weight models and 33 jailbreak attacks, we find two factors that systematically shape transfer: (1) representational similarity under benign prompts, and (2) the strength of the jailbreak on the source model. To move beyond correlation, we show that deliberately increasing similarity through benign-only distillation systematically increases transfer. Qualitative analyses reveal transferability patterns: persona-style jailbreaks transfer far more often than cipher-based prompts, consistent with the idea that natural-language attacks exploit models’ shared representation space, whereas cipher-based attacks rely on idiosyncratic quirks that do not generalize. Together, these results reframe jailbreak transfer as a consequence of representation alignment rather than a fragile byproduct of safety training.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work shows that jailbreak transferability is strongly tied to representation similarity. The author support this claim by showing a clear correlation between representation similarity and attack transferability, as well as performing an intervention and finetuning a student model through knowledge distillation and showing it becomes vulnerable to the same attacks as the teacher model.

### Strengths
- The paper advances our understanding of LLM jailbreaks and the cases where they transfer.
- The methodology is sound and clearly explained and the results support the authors' claims.
- The paper is well written, with motivation and contributions clearly communicated.

### Weaknesses
I did not find any major weaknesses in this paper, however while the claims of the paper are very well supported, they are also a bit narrow in that they connect jailbreak attacks to transferability but don't provide further insight on the precise mechanism or the implications for preventing jailbreaks. This keeps me from recommending spotlight/oral acceptance but I still think the paper is good contribution and can be accepted in its current form.

Overall, a more detailed discussion of the implications of this work and connections to previous work would improve the paper. For example fine tuning on safe data leading to vulnerability to the same jailbreaks could be seen as an example of subliminal learning (https://arxiv.org/abs/2507.14805). Making these kinds of connections and discussing them at least in the Appendix could be of interest to the readers.

### Questions
The difference between the transferability for persona-based and cypher-style jailbreaks is mentioned in the abstract and the introduction but then barely discussed outside of Appendix E. I would be interested in getting more of your thoughts on why we see this difference. Would it make sense to look at the representation similarity of different groups of jailbreaks? Are the representations of persona-based jailbreaks more similar across models than the representations of cypher-based jailbreaks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper discusses the reason behind jailbreak transferability in LLMs from the perspective of shared representations. The study employs benign data distillation techniques from a large model to a small model, and identifies two key factors: representational similarity and jailbreak strength.

### Strengths
1. Clear expression and fluent writing.
2. Rigorous evaluation using the StrongREJECT judge.
3. Aboundant experiments on open-weight models and different jailbreak attacks. 
4. Validated the effectiveness of distillation techniques in enhancing the transferability of jailbreak prompts.

### Weaknesses
1. Many of the findings in this paper are quite basic and not new to me, as similar conclusions have appeared in other fields. For instance, distillation has been shown to improve model similarity [1], enhance black-box transferability [2-3], and boost robustness [4-5]. Additionally, a recent paper on LLM attacks found that distillation can improve the effectiveness of prompt attacks on smaller models [6], which overlaps significantly with the core conclusion of this paper. Personally, this paper feels more like an exploratory discussion of a phenomenon rather than a contribution to advancing the SOTA baselines, and the repetition of these conclusions is a significant flaw.
2. Lack of evaluation about transferability on commercial LLM APIs, such as GPT, deepseek, grok.
3. If the findings of this paper could form a specific attack methodology and presented as a dedicated section, followed by a direct comparison with more recent baselines, it would enhance the completeness of the paper.

[1] Similarity-Preserving Knowledge Distillation
[2] Teach Me to Trick: Exploring Adversarial Transferability via Knowledge Distillation
[3] Improving the Transferability of Adversarial Examples by Inverse Knowledge Distillation
[4] Improving Adversarial Robustness Through Adaptive Learning-Driven Multi-Teacher Knowledge Distillation
[5] ProARD: progressive adversarial robustness distillation: provide wide range of robust students
[6] Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs

Above all, my initial score is 4, and it may change depending on the rebuttal.

### Questions
Perhaps it could be worth trying to distill a single model from the ensemble of multiple models?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the underlying mechanisms driving the *transferability of adversarial jailbreak prompts* across large language models (LLMs). By systematically evaluating 20 open-weight models and 33 attack templates, the authors identify two principal factors predicting transfer: (1) representational similarity under benign prompts and (2) jailbreak strength on the source model. Through a causal intervention—benign-only distillation—the authors demonstrate that increasing representational similarity causally enhances transferability, supporting the hypothesis that shared internal representations, rather than safety-tuning artifacts, underlie this phenomenon. The study represents the most comprehensive empirical exploration of jailbreak transfer to date and proposes a clean, reproducible experimental paradigm for future research.

### Strengths
1. The study compiles the largest known dataset of jailbreak transfer evaluations (20 models × 33 attacks × 313 prompts).
2. The authors carefully control for attack strength, preventing the common confounding that “stronger attacks transfer more.”
3. The benign-only distillation protocol provides a safe, controlled method to causally test representational effects without ethical concerns.
4. The results offer a clear practical implication for defense research: surface-level alignment is insufficient; external classifier-based safety layers may remain necessary.

### Weaknesses
1. Limited scope of distillation: Only three teacher–student pairs, trained for one epoch under a single hyperparameter configuration.
2. Limited practical impact: Even after distillation, transfer rates in large models remain low (≤10% in Fig. 6), somewhat constraining real-world significance.

### Questions
1.  In Figure 5, after epoch ≈ 0.3 the similarity curve in Figure 4 has almost saturated, yet the mean transfer score shows a mild downward trend toward epoch 1. What causes this decoupling? Could you provide results for two or three additional epochs to clarify whether transferability plateaus or continues to drop as the model further overfits to benign responses?
2. Beyond empirical correlation, can your representational-alignment hypothesis inspire new attack or defense mechanisms? For instance, if jailbreak transfer arises from shared latent geometry, could one design targeted perturbations or representation-level regularizers to either exploit or mitigate this alignment? As it stands, the theoretical framing feels somewhat descriptive rather than generative—clarifying its actionable implications would strengthen the paper’s overall significance. This is particularly important given that, as shown in the figures, the absolute transferability remains quite low even after distillation, which raises questions about the practical impact of the proposed explanation.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper argues that jailbreak transferability arises from shared internal representations between language models. To provide causal evidence, they use benign-only distillation to increase a student model's representational similarity to a teacher, which in turn makes the student more vulnerable to the teacher's jailbreaks. Experimental results demonstrate that jailbreak transfer as a fundamental consequence of representational alignment, suggesting that models that think alike inherit each other's vulnerabilities.

### Strengths
1.This paper provides a new and important perspective on jailbreak transfer, framing it as a fundamental consequence of shared representations.

2.The authors' novel distillation experiment provides powerful causal evidence for their claims, which is a very convincing and methodologically sound approach.

3.The large-scale quantitative analysis across 20 models makes the observed correlation between similarity and transferability highly robust and credible.

### Weaknesses
1. The paper's causal claim relies on the assumption that benign-only distillation solely increases representational similarity. However, this fine-tuning process could also inadvertently degrade the student model's general capabilities and original safety training effectiveness, which could also explain its increased vulnerability.

2. The study's conclusions are dependent on the chosen mutual k-nearest neighbors metric for measuring similarity. Could the authors explain why they chose this metric or demonstrate whether this strong correlation would persist with other similarity metrics?

### Questions
see Weaknesses

### Soundness
3

### Presentation
4

### Contribution
3
