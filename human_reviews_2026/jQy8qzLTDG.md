# Zero-Residual Concept Erasure via Progressive Alignment in Text-to-Image Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 6, 2, 2

## Abstract
Concept Erasure, which aims to prevent pretrained text-to-image models from
 generating content associated with semantic-harmful concepts (i.e., target con
cepts), is getting increased attention. State-of-the-art methods formulate this task
 as an optimization problem: they align all target concepts with semantic-harmless
 anchor concepts, and apply closed-form solutions to update the model accord
ingly. While these closed-form methods are efficient, we argue that existing meth
ods have two overlooked limitations: 1) They often result in incomplete erasure
 due to “non-zero alignment residual”, especially when text prompts are relatively
 complex. 2) They may suffer from generation quality degradation as they al
ways concentrate parameter updates in a few deep layers. To address these issues,
 we propose a novel closed-form method ErasePro: it is designed for more com
plete concept erasure and better preserving overall generative quality. Specifically,
 ErasePro first introduces a strict zero-residual constraint into the optimization ob
jective, ensuring perfect alignment between target and anchor concept features
 and enabling more complete erasure. Secondly, it employs a progressive, layer
wise update strategy that gradually transfers target concept features to those of the
 anchor concept from shallow to deep layers. As the depth increases, the required
 parameter changes diminish, thereby reducing deviations in sensitive deep layers
 and preserving generative quality. Empirical results across different concept era
sure tasks (including instance, art style, and nudity erasure) have demonstrated the
 effectiveness of our ErasePro.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ErasePro, a closed-form method for concept erasure in pretrained text-to-image diffusion models. It addresses two key issues in prior methods: incomplete erasure and generation degradation, by enforcing a zero-residual constraint that aligns target and anchor concept features, and by applying a progressive layer-wise update strategy to preserve image quality. Experiments across both explicit and implicit prompts show improved erasure performance and output quality.

### Strengths
1. The paper clearly identifies and addresses two important limitations in existing closed-form erasure methods:
Incomplete Erasure, where traces of the unwanted concept still appear in the generated images, and Generation Degradation, where the overall image quality drops after performing the erasure.

2. The layer-wise update strategy is simple but effective. By updating the model in a step-by-step manner from shallow to deep layers, the method avoids sudden shifts in the model’s behavior and helps maintain visual quality. 

3. The experiments are diverse and cover both explicit and implicit prompt settings, which is important because users often refer to concepts in indirect ways.

### Weaknesses
1. Although the paper provides a closed-form solution, it lacks a theoretical analysis of how the zero-residual constraint affects the trade-off between erasure strength and image quality. In some cases, forcing perfect alignment might harm generalization or lead to overfitting.

2. The method does not explore which specific layers are the most important for effective erasure. Understanding this could help make the method more efficient or interpretable.

3. The paper also does not test the sensitivity to the number of target-anchor pairs. It is unclear how the number or diversity of these pairs affects the performance of the method.

4. The authors claim that previous work like UCE suffers from non-zero residuals (as shown in Equation 3), but they do not provide quantitative evidence. For example, it would be helpful to show how large these residuals are in practice and how they relate to failure cases.

### Questions
1. Could the zero-residual constraint be too strict?
In some cases, this constraint might force the model to match features too exactly, which could lead to overfitting to the specific examples used during optimization. Has the method been tested on semantically similar but lexically different prompts to see if it generalizes well beyond the training cases?

2. Are there cases where the erased concept can still be evoked using carefully phrased or indirect prompts? It would be valuable to test the method’s robustness against such adversarial inputs that try to bypass the erasure.

3. Can the method generalize to other T2I architectures?
The current experiments are focused on a specific diffusion model.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes ErasePro, a method for concept erasure in text-to-image models like Stable Diffusion. It aims to remove harmful or unwanted concepts (e.g., nudity, specific artists) by aligning them with harmless anchor concepts (e.g., "clothed", "artist"). ErasePro introduces a zero-residual constraint to ensure complete erasure and uses a progressive layer-wise update to preserve image quality. Experiments on instance, style, and nudity erasure show that ErasePro outperforms existing approaches in both effectiveness and generation quality.

### Strengths
1. The presentation of this paper is clear and easy to follow.
2. The proposed zero-residual constraint makes the optimization objective stricter than previous approximate alignments.
3. The authors evaluate the method across multiple tasks (instance, style, and nudity erasure), under both explicit and implicit settings, showing good performance compared to previous methods.

### Weaknesses
1. The progressive alignment requires sequential updates across multiple layers (from 1 to S), which may introduce non-negligible computational overhead compared to previous closed-form methods.
2. As the authors say that previous methods usually fail under complex prompts, it would be better to include prompts from existing challenge prompts/attacks like UnlearnDiffAtk[1], Ring-A-Bell[2].
3. The core motivation of ErasePro is to reduce the alignment residual. But there are no visualizations or quantitative plots comparing residuals before and after applying ErasePro. A clear comparison of this can strengthen the paper’s motivation.
4. Some state-of-the-art and related works are missing [1][2][3][4] in experiments.

[1] EraseAnything: Enabling Concept Erasure in Rectified Flow Transformers, ICML 2025

[2] Dark Miner: Defend against undesired generation for text-to-image diffusion models

[3] One Image is Worth a Thousand Words: A Usability Preservable Text-Image Collaborative Erasing Framework, ICML 2025

[4] Concept corrector: Erase concepts on the fly for text-to-image diffusion models

### Questions
1. According to my understanding, ErasePro ties each target concept feature with its anchor concept feature. Compared to previous methods, the anchor selection scheme seems to be much more important here. How do you select anchor concepts?

2. Can it be applied to multi-concept erasing?

3. How do you obtain target and anchor features? How does the feature number $N$ influence the performance?

### Soundness
3

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
This paper introduces ErasePro, a novel closed-form algorithm for concept erasure in pretrained text-to-image diffusion models. The goal of concept erasure is to prevent models from generating undesired or harmful concepts (e.g., nudity, copyrighted art, or specific identities) while maintaining generation quality. ErasePro introduces two main innovations:1. Zero-Residual Constraint: A new constrained optimization formulation ensuring perfect alignment (zero residual) between target and anchor concept features.
2. Progressive Alignment Framework: A layer-wise optimization scheme that updates parameters progressively from shallow to deep layers, transferring the concept alignment gradually and minimizing deep-layer deviations. Empirical results across three main tasks — instance erasure, art-style erasure, and nudity erasure — demonstrate that ErasePro achieves more complete erasure and better generative quality compared to gradient-based (AC, ESD-x/u) and closed-form (UCE) baselines.

### Strengths
1. The zero-residual constraint and progressive alignment formulation are conceptually elegant and novel extensions to existing closed-form erasure frameworks. By combining analytical guarantees (perfect alignment) with practical layer-wise progression, the method addresses two long-standing issues — incomplete erasure and quality loss — in a unified manner.  

2. Extensive experimental evaluations cover a broad range of erasure scenarios: instance, art style, explicit nudity, and implicit nudity. Quantitative and qualitative analyses are consistent and compelling, supported by multiple evaluation metrics (CLIP score, KID, FID, NudeNet statistics).

3. The paper is clearly written  and well-organized sections.

### Weaknesses
1. Overclaiming novelty method is a minor variation of existing closed-form updates: The proposed Eq. (5) is just a constrained least-squares variant of the standard UCE (Eq. 2). The derivations are standard matrix algebra with the Moore Penrose pseudoinverse. The algorithm is incremental over UCE, not a fundamentally new paradigm.

2. Lack of motivation for the progressive layer update:  The claim that shallow layers should bear the “update burden” (Section 3.2, Fig. 3) is intuitive but unsupported. No quantitative ablation shows how much the progressive scheme helps compared to updating all layers jointly. The argument that “deeper layers are sensitive to generation quality” is anecdotal; no empirical or theoretical sensitivity analysis is provided.

3. All experiments are confined to a single base model (Stable Diffusion v1.4) with three concept types (instance, style, nudity). No tests on other baseline (SD2) are provided. The benchmark prompts are GPT-generated which may cause the bia of GPT to propogate, why not other models used.

4. The paper relies heavily on CLIP score, and CLIP accuracy: CLIP metrics capture similarity, not absence of unwanted concepts and have been saturated in the value (CLIP scores can appear stagnant due to their continuous nature and relative insensitivity at low ranges.).
human-based evaluation is also missing. The authors report marginal improvements (often ±0.01 CLIP difference), which are low.

5. While the authors claim “complete erasure,” they never test whether the erased concept can be reactivated by prompt modification or paraphrasing (e.g., “unclothed person” instead of “naked”). This is a standard benchmark in concept erasure literature (see ESD, Forget-Me-Not, Recler). Without this, the claims of robustness are unsubstantiated.

6. Overstated efficiency claims: ErasePro is described as “efficient,” but Table 4 shows runtime (15.71 s) is 5× slower than UCE (3.48 s), the main competing closed-form method. Memory is indeed lower, but that’s expected given partial-layer updates. ErasePro is slower in runtime and only modestly lighter in memory.

7. No comparison against recent state-of-the-art (post-2024): Recent methods are not compares some are mentioned such as SPEED (Li et al., 2025) but not compared experimentally.

8. The paper presents no experiments varying: number of progressive layers, layer selection strategy, semantic distance between target and anchor concepts, or pseudoinverse regularization. Without such ablations, it’s impossible to know if the method is stable or heavily tuned.

### Questions
1. How sensitive is ErasePro to the choice of anchor concepts? For example, what happens if the semantic distance between target and anchor is very large?

2. Does the zero-residual constraint introduce any overfitting to specific token embeddings, and how does it behave on multi-token or compositional concepts (e.g., “Van Gogh painting of a woman”)?

3. How scalable is ErasePro when applied to larger architectures like SD2 or SDXL?

4. Can the authors share insights into how many layers are typically sufficient before convergence of progressive alignment?

### Soundness
3

### Presentation
3

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
The authors propose ErasePro, a concept erasure method that can better handle erasure on complex prompts. In particular, the method is built upon UCE by (1) improving the closed-form solution, and (2) performing updates across more layers to improve model utility. Experimental results suggest that the method performs competitively or better than compared methods.

### Strengths
- The paper is generally well-written and problem is well-motivated.
- I like the fact that progressive alignment can improve model utility in most cases.

### Weaknesses
- The novelty seems limited. The method seems like a small upgrade to the method UCE.
- The comparisons are limited. First, UCE, AC, and ESD are quite old methods and I suggest the authors should compare the proposed method with more recent ones. Second, for concept erasure paper, it is more common to also report results on objects, which the paper is missing.
- I am not convinced yet that zero-residual can improve erasure on complex prompts. In particular, it seems like only the I2P experiment suggests that. But I think it should also be shown quantitatively for other concepts as well. 
- Some typos: Line 211 - EarsePro, Line 213 - EreasePro.

### Questions
- In Table 2, is there a reason why nudity erasure is split into implicit and explicit? And why are there so many cells for ErasePro(-w/s) that are not applicable? I think those values should and can be computed for better comparison.
- Are both improvements proposed by ErasePro necessary?

### Soundness
2

### Presentation
2

### Contribution
2
