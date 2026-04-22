# Adversarial Déjà Vu: Jailbreak Dictionary Learning for Stronger Generalization to Unseen Attacks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 8, 2, 6

## Abstract
Large language models remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Defending against novel jailbreaks represents a critical challenge in AI safety. Adversarial training---designed to make models robust against worst-case perturbations---has been the dominant paradigm for adversarial robustness. However, due to optimization challenges and difficulties in defining realistic threat models, adversarial training methods often fail on newly developed jailbreaks in practice. This paper proposes a new paradigm for improving robustness against unseen jailbreaks, centered on the Adversarial Déjà Vu hypothesis: novel jailbreaks are not fundamentally new, but largely recombinations of adversarial skills from previous attacks. We study this hypothesis through a large-scale analysis of 32 attack papers published over two years. Using an automated pipeline, we extract and compress adversarial skills into a sparse dictionary of primitives, with LLMs generating human-readable descriptions. Our analysis reveals that unseen attacks can be effectively explained as sparse compositions of earlier skills, with explanatory power increasing monotonically as skill coverage grows.
Guided by this insight, we introduce Adversarial Skill Compositional Training (ASCoT), which trains on diverse compositions of skill primitives rather than isolated attack instances. ASCoT substantially improves robustness to unseen attacks, including multi-turn jailbreaks, while maintaining low over-refusal rates. We also demonstrate that expanding adversarial skill coverage, not just data scale, is key to defending against novel attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper explores whether new jailbreak attacks primarily arise as recombinations of earlier adversarial tactics. The authors compile a large set of original–mutated prompt pairs from multiple jailbreak studies and automatically extract thousands of textual skills using a frontier LLM. These are then refined with another LLM to produce clear, human-readable skill primitives and to remove near-duplicates. They test a temporal-cutoff hypothesis, examining whether skills from newer attacks can be represented as sparse compositions of older primitives. The skill explanations are embedded using text-embedding-3-large, followed by dictionary learning and sparse coding along a Pareto knee. The resulting dictionary atoms are decoded into interpretable names through basis-pursuit attribution, LLM synthesis, and a similarity-graph redundancy filter. Evaluated on LLaMA-3.1-Instruct and Zephyr-Instruct, the proposed ASCoT framework achieves lower StrongReject harmfulness scores on previously unseen attacks.

### Strengths
1. The pre/post split and comparison of a compact dictionary (D_{\text{final}}) against the overcomplete Dover demonstrate high explainability with far fewer atoms and consistent sparsity across unseen families.

2. The generation protocol (composing named primitives with a helper LLM) aligns the defense with the claimed structure of jailbreaks; the dataset composition is enumerated with sizes and sources.

3. The coverage dividend and (k)-depth crossover figures isolate interesting phenomena at constant data size, supporting the skill-space framing.

### Weaknesses
1. The paper extracts skills, names atoms, runs redundancy decisions, and scores explainability with GPT-4.1; while human-in-the-loop verification is mentioned, the extent and protocol are not quantified, raising leakage/bias concerns. Independent human ratings (with inter-rater agreement) or a distinct non-OpenAI judge would strengthen claims.

2. Key controls are absent: (i) training on the same number of randomly composed skills vs. learned primitives; (ii) training with the same data budget but more plain harmful data; (iii) swapping the helper generator (DeepSeek-V3-Chat) to test dependence on its style; (iv) removing the refusal-template pairing (“I am sorry…”) to check if rote refusal is driving gains. Without these, attribution to skill coverage and generic data augmentation is uncertain.

3. The dictionary is built from single-turn pairs; yet, claims extend to multi-turn (GALA). While GALA's harmfulness decreases, the mechanism is not carefully analyzed (e.g., conversation-level skill composition, turn-ordering effects). A multi-turn extraction pipeline or at least qualitative mapping for sequences would improve the story.

### Questions
1. Table 3 labels the winning method SBAT, which appears to be ASCoT; if not a new baseline, this is a typo. Also, statistics (variance, CIs) and multiple-run stability for fine-tuning are not reported, which matters given sensitivity of safety tuning.

2. There are recent skill-mix efforts in instruction tuning and safety that could be directly compared as alternative composition engines rather than only CAT/LAT/WildJailbreak; the paper mainly positions against adversarial-training-style methods.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new paradigm for the robustness of adversarial unknown jailbreak attacks, assuming that the new jailbreak attack is a combination of existing attack methods, and verifies it through investigating multiple papers. Based on this assumption, the paper proposes adversarial skill combination training called ASCoT, which improves model robustness through multiple adversarial attack combinations.

### Strengths
1. The hypothesis proposed in the paper is very novel, and it is very interesting and meaningful to verify it through proposed literature research.
2. The paper explains the skill-level overlap between seen and unseen jailbreak attacks by constructing interpretability scores and sparsity levels experiments.

### Weaknesses
1. Since seen attack methods are using automated processes in the research, it seems to rely on prompts, which may affect the stability of extracting skills.
2. The paper lacks extracting dictionaries from a subset of 32 research papers, forming a combination method, and experimentally verifying the gap in capabilities.

### Questions
Is there a space in the title of paper?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a new approach to improving the robustness of LLMs against jailbreak attacks. It introduces the Adversarial Déjà Vu hypothesis, which suggests that most new jailbreaks are not truly novel but rather combinations of previously observed adversarial “skills.” Authors analyze jailbreak studies over two years, extracting and compressing recurring manipulation strategies into a Jailbreak Dictionary. They claim that unseen attacks can often be explained as recombinations of these fundamental skills. Based on this insight, they propose Adversarial Skill Compositional Training (ASCoT), which trains models on diverse compositions of these skill primitives rather than isolated attacks.

### Strengths
1. The paper identifies a concrete limitation of existing adversarial training approaches—poor generalization to unseen jailbreaks—and motivates its method based on observed skill reuse patterns in prior attacks.
2. The paper is logically organized, with a smooth progression from hypothesis formulation to empirical analysis and method development, making it easy to follow and understand the core contributions.

### Weaknesses
1. The paper lacks concrete examples illustrating how the skill primitives from the Jailbreak Dictionary are applied to transform base prompts into adversarial ones. Since the proposed defense relies heavily on such compositions, the authors should include specific before-and-after prompt examples.

2. While the paper proposes a skill dictionary for generating jailbreak-style adversarial prompts to construct the ASCoT training data, the overall structure of using composed attack strategies for prompt generation resembles AutoDAN-Turbo, making the contribution feel incremental. To further clarify the contribution of the proposed Jailbreak Dictionary, authors should consider adding a comparative experiment against a baseline constructed using AutoDAN-Turbo strategies. Specifically, it would be informative to generate adversarial training data using AutoDAN-Turbo's discovered strategies and train a comparable defense model. 

3. In the evaluation stage, AutoDAN-Turbo is claimed as one of the unseen attacks to test model generalization. However, according to Appendix C, the authors include the AutoDAN-Turbo paper within the curated set of 32 representative jailbreak papers used to construct their skill dictionary. Therefore, I do not think this can be called “unseen attack”. Authors should clarify the rationale and fairness of treating AutoDAN-Turbo as an unseen attack in the evaluation.

### Questions
See weakness.

### Soundness
2

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
4

### Summary
This paper proposes an efficient and effective adversarial training (AT) approach for LLMs. The main hypothesis of the approach is that each new jailbreak attack can be implicitly seen as a composition of previously existing attacks. Under this hypothesis, the authors design a novel method to compress existing jailbreak attacks into a small set of "seed attacks", and then use these seed attacks to efficiently generate new strong jailbreak attacks for AT. These compressed seed attacks are also used to analyze the evolution of real-world jailbreak attacks.

### Strengths
1. The core hypothesis of this paper is that existing jailbreak attacks can be implicitly seen as compositions of previous jailbreak attacks. The authors also design a pipeline for decomposing given attacks. I like this idea and believe it is novel and may potentially help the LLM safety community understand more about the mechanisms behind LLM jailbreaking.

2. The authors design a smart pipeline to reconstruct human-readable attack skill descriptions from the compressed jailbreak dictionary embeddings $D$ obtained from Eq. (1) (which is achieved via Eq. (2) and the help of LLM prompting). I really like this solution.

### Weaknesses
1. **Too few base models.** The AT experiments in Section 3 were only conducted on two LLMs (Llama3.1-8B, Zephyr-7B), which I think is far from enough to justify the effectiveness of the proposed Jailbreak Dictionary method in mitigating attacks. Experiments on more model families such as Qwen2.5/3, Gemma-2/3, Llama-2, Vicuna, and Mistral are suggested.

2. **Missing important representative jailbreak baselines.** The authors evaluate their AT approach against four unseen attacks, but most of these attacks (except AutoDAN-Turbo) are not representative attacks that are widely adopted as jailbreak baselines for jailbreak robustness analysis. Therefore, I suggest the authors adopt more representative attacks such as [r1, r2, r3] as unseen attacks.

3. **Missing important details about jailbreak robustness evaluations.** For Table 2 in Section 3.2, please clarify:
    - What is "StrongReject score"? Is it a variant of the "Attack Success Rate"?
    - On which dataset is the "StrongReject score" calculated?
    - How is the "StrongReject score" calculated?

4. The "prompting-based skill extraction" technique in Section 2.2 and the "attack via prompting with skill descriptions" technique in Section 3.2 are not new. To the best of my knowledge, these methods were first proposed in the AutoDAN-Turbo paper in 2024. I think the authors should explicitly acknowledge this in the main text of their paper.

5. The format of this paper is incorrect. The width of this paper is far larger than the restriction of the ICLR submission template. The authors must fix this in their revision.



**References**

[r1] Hayase et al. Query-Based Adversarial Prompt Generation. NeurIPS 2024.

[r2] Sadasivan et al. Fast Adversarial Attacks on Language Models In One GPU Minute. ICML 2024.

[r3] Andriushchenko et al. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks. ICLR 2025.

### Questions
See **Weaknesses**.

### Soundness
2

### Presentation
3

### Contribution
3
