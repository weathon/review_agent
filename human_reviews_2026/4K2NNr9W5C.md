# Pay Less Attention to Function Words for Free Robustness of Vision-Language Models

- Decision: Accept (Poster)
- Scores: 4, 4, 2, 2

## Abstract
To address the trade-off between robustness and performance for robust VLM, we observe that function words could incur vulnerability of VLMs against cross-modal adversarial attacks, and propose Function-word De-Attention (FDA) accordingly to mitigate the vulnerability brought by function words. Inspired by differential transformers, our FDA calculates the original and the function-word cross-attention within attention heads, and differentially subtracts the latter from the former for more robust alignment. Comprehensive experiments include 2 SOTA baselines under 6 different attacks on 2 downstream tasks, 3 datasets, and 3 models. Overall, our FDA yields an average 18/13/53\% ASR drop with only 0.2/0.3/0.6\% performance drops on the 3 tested models on retrieval, and a 90\% ASR drop with a 0.3\% performance gain on visual grounding. We demonstrate the scalability, generalization, and zero-shot performance of FDA experimentally, as well as in-depth ablation studies and analysis. Code is available at https://github.com/michaeltian108/FDA.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Function-word De-Attention (FDA), a lightweight and novel method to improve the robustness of vision-language models (VLMs) without compromising their clean performance. The authors observe that function words (e.g., “the,” “is,” “and”) can make VLMs more vulnerable to cross-modal adversarial attacks because these words are frequent but semantically uninformative. FDA operates by computing and subtracting the cross-attention between function words and image tokens from the original attention maps. While the underlying mechanism behind FDA’s effectiveness is not yet fully understood, the authors provide extensive experiments demonstrating consistent improvements across multiple models, datasets, and attack settings.

### Strengths
1. Identifying function words as a source of vulnerability in VLMs is an original and well-motivated idea. Good observation.

2. FDA is conceptually straightforward and computationally efficient. It requires only a differential subtraction step within existing attention computations. It does not require additional parameters or retraining complexity.

3. The authors evaluate FDA across multiple models, datasets, and both targeted and untargeted attacks. The improvement over adversarial training baselines (TeCoA, FARE) is consistent and substantial.

### Weaknesses
1. It's an interesting observation that function words can lead as a source of vulnerability. However, the evidence provided are far from satisfying. Namely, a single sentence (“80.3% of images show higher similarity scores toward function words than content words after attacks”) and one visualization of distracted attention. It is insufficient to establish that this is a systematic rather than a coincidental effect. Additional qualitative and quantitative analyses would strengthen the claim. I also recommend including a baseline where function words are simply removed from the inputs, to better isolate and validate the proposed phenomenon.

2. Lack of clarity regarding experimental setup. FDA introduces a hyperparameter $\lambda$ to control the strength of “de-attention,” where $\lambda = 0$ reduces the method to standard attention. However, I find it confusing that the authors did not specify the $\lambda$ values used in the reported results. In particular, for the clean (non-attacked) setting, it is unclear whether the authors used a very small or even zero $\lambda$, which would make the claim of minimal performance drop less convincing. Moreover, the paper does not analyze how varying $\lambda$ affects performance and robustness, which is essential for understanding the method’s sensitivity and general applicability.

3. Several presentation issues. 1) Figure 1 seems to have noticeably lower resolution than other figures, even for texts in the figure. 2) Typo in Equation (4), $S_{T_f}$ instead of $S_{t_f}$. 3) Mistake in line 197-198, " Specifically, **TCL** shares the same backbones as **TCL** ..."

### Questions
I have made my suggestions along with weaknesses. Here are several open questions for the authors:

1. Have the authors examined whether the vulnerability arises specifically from function words, or more generally from irrelevant or low-information words in the inputs?

2. Could there be an adaptive mechanism to identify which words to down-weight or exclude, rather than relying on a fixed dictionary?

3. Are there cases where function words contribute meaningfully to visual grounding or alignment, and if so, how does FDA handle those instances?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Function-word De-Attention (FDA) to improve the robustness and performance of VLMs by reducing attention to function words. FDA works by subtracting the cross-attention between function words and images from the original attention. Experimental results demonstrate that FDA reduces ASR with minimal or even improved performance.

### Strengths
1. The motivation and introduction of this paper is clear while use visualization to help readers quickly understand the task.

2. The experiments are thorough, and the performance is verified on three models, two tasks, and three datasets.

### Weaknesses
1. The authors' proposed operation, "proper removal of function words could potentially defend VLMs against such attacks," appears to have some limitations on VLM's use cases: the paper does not demonstrate its ability to defend against the fundamental task of vision-language models, such as classification. Furthermore, defense methods like FARE [1], which were originally designed to enhance the robustness of the LVLM's vision encoder, seems also inapplicable to this scenario.

2. In contribution summarization, the proposed “pioneer the theory” may be replaced with observation, motivation or other words. The paper does not use theory to prove that the difference in similarity after paying less attention can increase or decrease, or that the difference is bounded.

3. In Figure 2, how the conclusion “models can learn a more aligned cross-modal embedding” get? The observational evidence provided in the introduction doesn't suggest that this phenomenon is achieved by learning a more aligned cross-modal embedding. On the contrary, if the function word is removed, intuitively, the effect of alignment should be weakened.

4. Considering the trade-off between robustness and performance, it is recommended to compare with the TRADES [2] method.

[1] Schlarmann C, Singh N D, Croce F, et al. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models[J]. arXiv preprint arXiv:2402.12336, 2024.

[2] Zhang H, Yu Y, Jiao J, et al. Theoretically principled trade-off between robustness and accuracy[C]//International conference on machine learning. 2019: 7472-7482.

### Questions
See Weaknesses.

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
This paper hypothesizes that function words (e.g., am, is, are)—which are semantically vague and non-specific—make Vision-Language Models (VLMs) vulnerable to cross-modal adversarial attacks.
To address this, the authors propose Function-word De-Attention (FDA), which introduces a parallel attention path that computes the cross-attention between function words and images, and then subtracts this “distraction” from the original attention through a differential subtraction mechanism.

On retrieval tasks, FDA reduces the Attack Success Rate (ASR) by 18/13/53% on ALBEF/TCL/BLIP, with only 0.2–0.6% clean performance degradation.
On visual grounding, it achieves a 90% ASR reduction with a 0.3% accuracy improvement.
The robustness gain scales with model size (e.g., +54% ΔASR for BLIP) and even improves zero-shot performance by +0.4% without fine-tuning.

### Strengths
- Conceptually novel to focus on function words as a source of VLM vulnerability.
- Simple and efficient method. No additional parameters or adversarial training required.
- Consistent and strong improvement across multiple models, datasets, and tasks.

### Weaknesses
- The paper does not perform a true white-box attack against FDA.
While baselines are evaluated under full white-box settings (with gradient access), FDA’s differential subtraction module is hidden from the attacker.
The proposed Masked APGD (MAPGD) does not actually backpropagate through the FDA operation, so the comparison is not fair.

- The “observation” section is weak.
The reported 80.3% statistic lacks detail—what dataset, model, or attack setting?
Showing only one qualitative example in Fig. 1 is not convincing as empirical evidence.

- The paper does not include any ablation or control experiment on other word classes (e.g., content words, nouns, verbs, adjectives).

- Table 1 presentation is confusing: R@1 (↑, higher is better) and ASR (↓, lower is better) are shown in the same rows without clear separation.

- The comparison is too limited: only TeCoA/FARE trained with ε = 1/255 are considered. Robustness should be compared against baselines trained under the same threat levels (ε = 2/255 or 4/255), as standard adversarial training does.

### Questions
see weaknesses

### Soundness
2

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
This paper proposes Function-word De-Attention (FDA), a training-free defense that subtracts attention from function words in VLMs (e.g., ALBEF, BLIP).
It shows robustness gains under image-space attacks without hurting clean performance. However, the idea is limited to cross-attention models, and lacks ablation evidence to support its claims.

### Strengths
1. The function-word “de-attention mechanism” is straightforward and easy to integrate into VLMs.
2. Zero training cost and minimal inference overhead.
3. Works stably across three representative fusion-based VLMs (ALBEF, BLIP, TCL).

### Weaknesses
1. Limited scope
    * FDA fundamentally requires cross-attention between text and image; thus, it cannot be applied to CLIP-style encoders or LLM-based multimodal models (LLaVA, InternVL).
    * Therefore, I think the paper’s title (“robustness of VLMs”) overstates generality.
2. Kind of trivial core observation
    * Figure 1’s finding — that attacks disrupt function-word attention but not content-word grounding — is unsurprising and follows directly from semantic anchoring.
3. Missing ablation
    * The simplest verification — masking function words entirely and measuring retrieval performance — is absent.
    * Without it, it’s unclear which is more important; “attention subtraction” or just “ignoring function words.”
4. Over-claiming “training-free”
    * Some sections mention optional fine-tuning for λ-scaling and per-layer adaptation, which contradicts the strict “zero-training” claim.

### Questions
Please refer to the Weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2
