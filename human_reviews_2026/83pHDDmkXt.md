# Soft-Di[M]O: Improving One-Step Discrete Image Generation with Soft Embeddings

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
One-step generators distilled from Masked Diffusion Models (MDMs) compress multiple sampling steps into a single forward pass, enabling efficient text and image synthesis. 
However, they suffer two key limitations: they inherit modeling bias from the teacher, and their discrete token outputs block gradient flow, preventing post-distillation refinements such as adversarial training, reward-based fine-tuning, and Test-Time Embedding Optimization (TTEO). 
In this work, we introduce soft embeddings, a simple relaxation that replaces discrete tokens with the expected embeddings under the generator's output distribution. 
Soft embeddings preserve representation fidelity for one-step discrete generator while 
providing a fully differentiable continuous surrogate that is compatible with teacher backbones and tokenizer decoders while cause minimum bias.
Integrating soft embeddings into the Di[M]O \citep{zhu2025di} distillation framework (denoted Soft-Di[M]O) makes one-step generators end-to-end trainable and enables straightforward application of GAN-based refinement, differentiable reward fine-tuning, and TTEO. 
Empirically, across multiple MDM teachers (e.g., MaskBit \citep{weber2024maskbit}, MaskGen \citep{kim2025democratizing}), Soft-Di[M]O achieves state-of-the-art one-step results: improved class-to-image performance, a one-step FID of 1.56 on ImageNet-256 with GAN-based refinement, along with higher than teacher GenEval \citep{ghosh2023geneval} and HPS \citep{wu2023human} scores on text-to-image with reward fine-tuning, and further gains from TTEO.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper extends the Di[M]O framework by introducing a soft embedding relaxation for discrete diffusion distillation. Instead of hard token sampling, the model uses non-parametric expected embedding under the predicted token distribution, enabling differentiability throughout the training process. This modification makes the framework compatible with gradient-based extensions such as adversarial fine-tuning and preference optimization. Experiments evaluate both class-conditional image generation and text-to-image tasks.

### Strengths
1. Clear and intuitive motivation: The paper articulates very well why enabling differentiability is valuable — it broadens the range of learning signals applicable to discrete diffusion models.

2. Reasonable and meaningful novelty: While the relaxation itself (Gumbel–Softmax-style) is known, its application in the context of distilled discrete diffusion models is new and practically relevant. The contribution sits at a “methodological adaptation” level — modest but non-trivial.

3. Clarity of presentation: The derivation of soft embeddings, training objective, and overall algorithm are explained cleanly and are easy to follow. The paper is well written and pedagogical.

4. Empirical coverage across modalities: The inclusion of both class-conditional generation and text-to-image synthesis demonstrates generality of the approach beyond a single domain.

### Weaknesses
1. Overemphasis on auxiliary fine-tuning tasks: The paper devotes significant space to adversarial fine-tuning and differentiable reward fine-tuning, but these are not core contributions. They are applications or privileges enabled by the soft embedding, not new methods. This focus dilutes attention from the main relaxation technique, which deserves deeper theoretical and empirical exploration.

2. Limited ablation on the method itself: The paper could better analyze the behavior of the soft embedding mechanism—e.g., temperature schedules, entropy evolution, convergence stability, and how it compares to direct Gumbel-Softmax or straight-through variants.

3. Experimental scope modest: Experiments are conducted on moderate-resolution datasets and lack comparisons with alternative differentiable relaxations. Broader benchmarks or higher-resolution evaluation would strengthen the claims.

### Questions
1. Does the relaxation introduce distributional bias or oversmoothing compared to discrete sampling?

2. How does this compare against simpler continuous relaxations like Gumbel-Softmax or parametric estimators in the same setting?

### Soundness
3

### Presentation
4

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
This paper addresses a key limitation of one-step generative models distilled from Masked Diffusion Models: their discrete token blocks gradient flow. The authors propose soft embeddings, a simple and effective relaxation that replaces discrete tokens with the expected embedding under the generator's output probability distribution, which makes the one-step generator differentiable. By integrating this into the Di[M]O framework, the method unlocks the ability to apply refinement techniques, including GAN-based training, differentiable reward fine-tuning, and Test-Time Embedding Optimization. Empirically, it achieves state-of-the-art one-step results, notably a 1.56 FID on ImageNet-256 and improved alignment scores on text-to-image tasks.

### Strengths
- The paper identifies a significant and practical problem (blocked gradients in one-step discrete generators) and proposes an simple solution.
- The differentiable feature enables many technique applications like GAN, reward fine-tuning.
- The method achieves strong results on ImageNet ($1.56$ FID). And the MSE between samples decoded by soft embedding and the corresponding discrete embedding is very small.
- The authors conduct comprehensive experiments validating the effectiveness and design choices.

### Weaknesses
- The idea of using the expectation of discrete embeddings is not a new technique in machine learning. So its novelty lies in the application to one-step MDMs.
- The generation performance relies on the introduction of GAN loss, which potentially introduces training instability and conplexity of hyperparameter tuning.

### Questions
- What's the performance of only applying soft embeddings without GAN loss?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Soft-Di[M]O, a one-step discrete image generator distilled from masked diffusion teachers (e.g., MaskBit, MaskGen). The key idea is to replace sampled discrete tokens with soft embeddings—the expectation of token embeddings under the model’s output distribution—so gradients can flow through the generator and decoder. This makes one-step students end-to-end trainable, enabling (i) GAN-based refinement, (ii) differentiable reward fine-tuning (e.g., CLIP/ImageReward).

### Strengths
1. Soft embeddings (expected embedding; Eq. (3)) are a minimal change that preserves compatibility with teacher backbones/tokenizers while restoring differentiability.

2. The framework cleanly integrates GAN loss, reward losses, broadening what one-step discrete students can optimize.

3. The paper studies discriminator input choices and mask schedules.

### Weaknesses
1. It’s unclear how much improvement comes from the GAN refinement versus the soft-embedding relaxation itself. While Fig.-style ablations compare Di[M]O vs GAN only vs combined, it only reports 5k FID.

2. TTEO (test-time scaling) increases inference compute; the paper could quantify cost-vs-quality more explicitly. For TTEO, what selection metric do you use to pick the final image—e.g., CLIP score, ImageReward/HPS, discriminator logit for GenEval and HPS v2.1?

### Questions
For TTEO, what selection metric do you use to pick the final image—e.g., CLIP score, ImageReward/HPS, discriminator logit for GenEval and HPS v2.1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces Soft-DiMO, an extension to DiMO with soft-embeddings that allow differentiating through the discrete token sampling step for post-training use cases such as supervised finetuning and human-alignment / prompt-adherence. The paper shows the superiority of their method compared to other methods, like the Gumbel Softmax trick, which allows differentiating through discrete sampling steps. The paper shows a SoTA FID score on discrete image generation trained on ImageNet 256x256.

### Strengths
- The authors propose a simple way to backrprop through the discrete sampling step that unlocks a wide variety of potential post-training applications. 
- The authors demonstrate various post-training techniques using their method and show better performance
- The authors achieve a SoTA FID score on ImageNet 256x256 
- The paper also compares the presented method to a multiple existing works

### Weaknesses
Minor:
- It is helpful for the reader to have the abstract be independent and include inline full citations of terms used like DiMO, MaskBit, MaskGen, HPS, etc. E.g.: (Author et al. 2022), in line with the abstract.

Major:
- I've had issues with the readability of the paper especially in Section 2 and Section 3 where the problem statement is not well set up to be complete. For instance, what is the space of vocabulary V for this problem? What are the typical patch sizes for each token? What dimensions are the output probability logits for each token? 
- The central idea of the paper is to weight the output token classification probability with the corresponding token embedding to obtain a so-called "soft-embedding" that can be used to backprop to empirically make post-training work better than other techniques.
-- However, the novelty is limited since the proposed soft-embedding is a weighted average w.r.t the word probability. 
-- Empirically, it is clear that it works better (at least in the hyperparam region used for experiments). But it is not clear *why* this works better, either intuitively and/or theoretically. Is this formulation really approximating backproping through the discrete sampling step better than the Gumbel-Softmax trick? If so, why? Is it purely due to optimization landscapes that are hard to theoretically prove or can some theoretical arguments be made to understand why this works better empirically?

### Questions
1. On Line 44, the authors write "First, they inherit modeling errors from the teacher" as one of the challenges of current one-step discrete generators. Can the authors please clarify why this is a problem? Are the authors suggesting that the errors get distilled to the student in preference to the image statistics in a more prominent way than they occur in the teacher model? Given the teacher is a powerful model with minimal region for mistakes, wouldn't the student model also be minimal in the param region for mistakes (since zero errors is unavoidable)?
2. My central question is around the arguments for why the proposed method works better than the others like Gumbel-Softmax other than a qualitative hand-wavy argument about high variance and high bias training signals. It is unclear why the proposed method alleviates such problems. A theoretical argument (despite any assumptions made) would help the reader to appreciate the idea proposed more than an empirical hack to circumvent post-training challenges of discrete sampling models.

### Soundness
4

### Presentation
3

### Contribution
3
