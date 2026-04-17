# Enhancing Adversarial Transferability in Vision-Language Models via Search-Space Expansion

- Decision: Reject
- Scores: 2, 6, 4, 4

## Abstract
Adversarial attacks are crucial for evaluating the robustness of vision-language pre-trained (VLP) models. However, existing methods suffer from limited transferability across unseen models, limiting their effectiveness as a universal robustness probe. We attribute this partially to the narrow search space of adversarial examples, which can trap optimization in local optima and lead to overfitting. To address this, we propose SEA (\textbf{S}earch-space \textbf{E}xpansion \textbf{A}ttack), a unified framework that improves cross-model transferability by enlarging the adversarial search space across both modalities. For images, SEA leverages historical updates to explore novel optimization directions, effectively avoiding suboptimal optimization  trajectories and overfitting. For text, SEA considers both individual word importance and word interactions, recognizing that less salient words can sometimes yield stronger and more transferable attacks. It performs word substitutions across multiple influential positions rather than focusing solely on the most salient word. Consequently, SEA can substantially disrupt cross-modal interactions across different models. Extensive experiments on diverse benchmarks, VLP models and tasks, supported by rigorous theoretical analysis, demonstrate that SEA significantly advances the state of the art. The source code is provided in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the limited adversarial transferability of Vision-Language Pre-trained (VLP) models, attributing it to narrow adversarial search spaces that cause overfitting to source models. It proposes SEA, a framework that expands the search space for both modalities.

### Strengths
- Well written.
- SEA addresses transferability issues by expanding the search space for image and text modalities.
- SEA has good cross-task/model generalization ability.

### Weaknesses
- SEA's image module combines "current gradient " and "historical information ", but the paper does not isolate the contribution of each component.
- This paper claims SEA avoids "local optima", but no visualization of optimization trajectories is provided. Without this, it is impossible to verify if SEA truly escapes local optima or just converges to different ones. 
- This paper lacks visualization results.
- This paper lacks a framework of the method, making it difficult to understand the SEA intuitively.
- The text module claims to "account for word interactions", but no quantitative evidence supports this.
- What is the significance of Proposition 1? It seems that the author merely conducted some mathematical derivations and did not provide an analysis of Proposition 1.

### Questions
- Please see "Weaknesses".

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors proposed SEA (Search-space Expansion Attack) to improve the transferability of the generated adversarial text-images pairs from one VLP model (vision-language pretrained model) to others, by introducing more candidates to explore during each optimization step besides the one that's directly chosen by PGD for the image attack and allowing the word replacement to happen at more possible positions instead of only the most saliant word for the text attack. Specifically, the new directions (or informally "gradients") to explore for potential adversarial images are chosen by drawing random linear combinations of all past gradients (difference with last iteration) or past perturbations (difference with original image), and the new word replacement possibilities come from words other than the most saliant one whose change might induce larger drop in text-image similarity. SEA is tested in retrieval, grounding and captioning tasks for different source and target VLP models. It shows improvement relatively significant in retrieval and milder but still consistent in the other tasks comparing to existing attacks, including SA-AET which also aims at improving attack transferability and enlarges the search space to some extent.

### Strengths
+ The proposed attack achieves noticeable improvement over existing attacks including very recent ones also working on transferability.
+ The authors focused on expanding the search space and applied the same idea to both the image and text domain.

### Weaknesses
+ The writing and presentation are unclear and sometimes even confusing. For instance, while it is understandable to think of the combined gradients as some new "gradient", it is not technically a gradient of anything; symbols like $m\in\{1,2\}$ and $m+$ in Eq. 5 are used without any expiation; Figure 2 doesn't explain at all what the triangular samples are but introduces inverse directions that appear only in this figure an nowhere else for unknown reasons.
+ The linkage between the motivation, the theory and the method are not very strong. For comparison, SA-AET broadens the adversarial image search space by sampling from a triangular region enclosed by original, previous and current compromised images which intuitively defines a search space that is both semantically relevant and more diverse than the neighboring areas of the current image. Dropping the constraints of "regions" is said to be a good thing about SEA but how and why? The proposed linear combination of past gradient or perturbations seem more like a successful trick that the authors picked up purely empirically than designed carefully.

### Questions
+ While linearly combining past gradients sounds somewhat understandable, what's the logic behind combining the perturbations? What does $\sum \eta=1$ mean when the norm of the past perturbations are related to the time step? 
+ What is the reason for choosing normal distribution for for perturbation-base search space expansion?
+ What selection criterion does Eq. 5 define? What is $m\in\{1,2\}$ and "m+"?
+ How large is $l$ in practice? In the example, "cat" is replaced with "fairy" which doesn't sound very close to each other. Given that TextAttack is the shared library, are the budgets set to the same as your baselines?

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
This paper proposes SEA (Search-space Expansion Attack), a unified framework that improves cross-model transferability by enlarging the adversarial search space across both modalities. For images, SEA leverages historical updates to explore novel optimization directions, effectively avoiding suboptimal optimization trajectories and overfitting. For text, SEA considers both individual-word importance and word interactions, recognizing that less salient words can sometimes yield stronger, more transferable attacks.
However, I found that some of the paper’s claims are not substantiated and there are several noticeable grammatical errors; therefore, I believe the manuscript should be thoroughly revised before being considered for acceptance.

### Strengths
1. This paper is relatively complete.
2. An interesting point is the enlargement of the adversarial search space across both text and image modalities.

### Weaknesses
1. There are some typos in this paper. For example, (1) In Equation (6), the summation is written as $\hat{\Delta }_{t}$. (2) Table 2's caption: "isual". (3) In Section 3.3.1, the author refers to “Figure 4”, but this figure does not appear in the paper.

2. The paper claims that using historical update information can avoid overfitting and local optima, but no empirical or theoretical evidence is provided to substantiate this claim.

3. The motivation highlights factors affecting text presentation—namely the semantics of individual words, substitution candidates, and their contextual relationships. But the proposed text attack does not explicitly model or target these factors.

### Questions
1. What are the actual memory usage and runtime speed of the proposed method?

2. Please refer to the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes an attack named SEA (Search-space Expansion Attack) to improve the transferability of adversarial attacks across vision-language pre-trained (VLP) models. SEA expands the adversarial search space in both image and text modalities. It leverages historical gradients for image perturbations and performs multi-word substitutions that account for both word importance and interactions in text. The authors conduct extensive experiments across various VLP models and tasks, showing consistent but modest improvements over existing approaches, accompanied by solid theoretical analysis.

### Strengths
1. Quality: The theoretical and empirical analyses on transferability are rigorous and add credibility to the findings.

2. Clarity: The paper is clearly written and well-organized, making the proposed framework easy to understand.

3. Significance: While the performance gain is moderate, the paper contributes another aspect of insights into improving cross-modal adversarial transferability, an important aspect of robustness evaluation. The approach is methodically implemented and systematically evaluated across diverse settings.

### Weaknesses
1. Incremental contribution: The main ideas of expanding the search space and using historical updates are extensions of well-studied techniques in adversarial optimization. The novelty is limited.

2. Minor empirical improvement: The performance gains in Table 1 are relatively small, suggesting that the proposed method offers incremental progress rather than a clear leap forward.

3. Incomplete related work discussion: The discussion of textual adversarial attacks is brief and omits important gradient-based approaches such as "LeapAttack: Hard-Label Adversarial Attack on Text via Gradient-Based Optimization", which are directly relevant.

4. Limited insight on cross-modal interactions: While SEA aims to expand search space across modalities, the analysis could further explain how image and text perturbations jointly enhance transferability.

### Questions
1. Since the improvement is modest, could the authors further clarify the necessity of this type of method?
2. Would integrating recent hard-label text attack baselines (e.g., TextHoaxer and LeapAttack) change the comparative performance results?
3. What's the performance for attacking the latest VLP models such as InternVL and Qwen-VL?

### Soundness
2

### Presentation
3

### Contribution
2
