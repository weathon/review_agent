# Adaptive Vision Token Selection for Multimodal Inference

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Vision encoders typically generate a large number of visual tokens, providing information-rich representations but significantly increasing computational demands. This raises the question of whether all generated tokens are equally valuable or if some of them can be discarded to reduce computational costs without compromising quality. In this paper, we introduce a new method for determining feature utility based on the idea that less valuable features can be reconstructed from more valuable ones. We implement this concept by integrating an autoencoder with a Gumbel-Softmax selection mechanism, that allows identifying and retaining only the most informative visual tokens. Experiments show that the sampler can reduce effective tokens and inference FLOPs by up to 50% while retaining 99-100% of the original performance on average. On challenging OCR-centric benchmarks, it also surpasses prior SOTA. The sampler transfers to the video setting as well: despite minor drops, zero-shot results remain strong without video-specific training. Our results highlight a promising direction towards adaptive and efficient multimodal pruning that facilitates scalable and low-overhead inference without compromising performance.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a token pruning method for inference on images in which a subset of image embedding tokens are selected for the purpose of downstream tasks. The method involves a subset selector and a reconstructor, trained in the style of VQVAE. Some improvements over prior token pruning baselines are presented.

### Strengths
1. The problem statement - can we determine the optimal subset for reconstructing the image and replace the existing tokens? - is interesting. 
2. Some improvements compared to recent token pruning methods like DivPrune, HiPrune, PDrop.

### Weaknesses
1. Unfortunately, I believe the training methodology is seriously flawed. I'll explain in bullet points:
    - The Gumbel softmax trick is for **sampling** from a categorical distribution in a differentiable way. Since the mask creation (as described) is deterministic, you don't need this, a simple straight-through estimator (y_hard - y_soft.detach() + y_soft where y_soft is the softmax) would suffice. 
    - In fact adding Gumbel noise like this may potentially harm the training, as there is no need of sampling the tokens for masking unless intended for some regularization purposes (like dropout)
    - Even ignoring this, **selecting the indices** from the straight through estimator defeats the whole point. The gradients cannot propagate through a selection operation, this is the reason you find that just having L_pr as regularizer drives it down to zero, as there is no optimization pressure from the reconstruction loss. 
    - After your modification to the L_pr, the method somewhat works because this essential turns off the regularizer once it is in a certain range between p to q. This is similar to simply having the token selector output a fixed number of tokens and throwing out the regularizer, it is not a joint optimization at all.
    - The correct way of doing this would be something like M * E_img + ( 1- M ) * E_masked, the gradients can propagate through this operation just fine.

    I hope the authors can fix these serious issues in future versions of the paper. My suggestions: (1) replace Gumbel-softmax with just a straight-through estimator, and (2) use linear weighting instead of index selection.

2. Other weaknesses are present (token merging baselines are absent, improvements over existing baselines seem marginal) but are minor compared to the serious issues in the training.

### Questions
Please address the weaknesses listed above

### Soundness
1

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
5

### Summary
This paper proposes a trainable selector that keeps only the most informative vision tokens before feeding them to a multimodal model. The selector is a small Transformer with a Gumbel Softmax head that produces a binary mask over encoder features. A reconstructor network is trained jointly so that discarded tokens can be reconstructed from the kept ones, and the objective includes a modified regularizer that nudges the average keep ratio into a desired range p to q. The selector is trained once per vision encoder on a mixture of COCO, DocVQA, and ChartQA features, then attached at inference time to several VLMs without further fine tuning. Experiments claim up to fifty percent token reduction with roughly ninety nine to one hundred percent of baseline accuracy on average, strong gains on OCR tasks, and some transfer to video. FLOPs accounting includes both the LLM and the selector overhead.

### Strengths
1. The paper formalizes selection as preserving reconstructability of discarded tokens and implements it with a compact selector plus reconstructor. The approach is conceptually simple and naturally plug and play for many VLMs. 
2. The p and q trick for the keep ratio is a concrete engineering fix to a mode collapse problem in the regularizer and is explained with training behavior observations. 
3. The selector is attached to Open LLaVA Next and LLaVA 1.6, compared to multiple pruning baselines, and also tried on video models with eight frame inputs. 
4. The paper spells out a FLOPs that adds selector cost to LLM cost rather than hiding it, which is important when the selector itself is nontrivial.

### Weaknesses
1.  The method keeps tokens that best reconstruct the image features and pixels, not tokens that are most useful to answer a question. This risks throwing away small but question critical text spans or regions that are hard to reconstruct yet crucial for reasoning. The paper acknowledges strong OCR results in some cases, but the core objective remains task agnostic and could be misaligned for many VQA settings. 
2. The selector is trained only on COCO plus two OCR datasets, then applied widely across benchmarks and model families. There is no analysis of distribution shift or per benchmark sensitivity to this training mix. 
3. The tables emphasize per benchmark accuracy and a FLOPs estimate. There is no thorough report of wall clock latency, throughput at batch sizes relevant to serving, or memory footprint measured on hardware. Without that, it is hard to validate the real system gain when the selector itself adds four Transformer layers per crop. 
4. The paper claims ninety nine to one hundred percent of average performance at fifty percent tokens and shows an example figure, but there is little analysis of worst case degradations, error types, or hallucination changes beyond a couple of benchmarks. Averages can hide brittle behavior on rare but safety relevant inputs. 
5. The selector is not trained for temporal data and results trail a simple diversity baseline on multiple metrics. The paper positions this as promising transfer, but the drop is material and there is no study of temporal coherence or frame wise instability under pruning.

### Questions
1. Can you add a text conditioned variant of the selector that uses the question to guide token importance and compare it with the purely reconstructive objective, especially on question sensitive OCR tasks
2. What are the worst ten percent relative drops across all datasets for a fifty percent keep ratio, and what qualitative failure modes do you observe when the selector deletes rare but crucial regions
3. Please report end to end latency and throughput on a single modern GPU across a realistic prompt distribution, including the selector time and memory, and compare to baselines that are training free and selector free
4. How stable is the mask across seeds and small image perturbations such as slight crops or contrast changes, and is there any evidence of mask flicker across video frames
5. Since the selector is trained on features from a specific encoder, how robust is it to encoders that were further finetuned inside a VLM and to different pooling or crop strategies

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
3

### Summary
This paper introduces an adaptive method for selecting visual tokens in multimodal models to address the inefficiency of processing all tokens generated by ViTs. The approach is based on the premise that less informative features can be reconstructed from more valuable ones. An autoencoder with a Gumbel-Softmax selector is employed to identify and retain only the most critical tokens. This method reduces computational cost (FLOPs) by approximately 50% while maintaining 99-100% of the original model performance on downstream tasks. It is designed as a plug-and-play module compatible with existing VLMs without requiring retraining and demonstrates modest generalization to video tasks.

### Strengths
1. It achieves high efficiency by reducing computational cost by 50% with negligible loss, a meaningful gain for large-scale inference.
2. Its plug-and-play design requires no retraining of the main VLM, which greatly increases its practicality.
3. The method receives strong empirical validation from comprehensive benchmarks on vision and multimodal datasets, with ablations proving its generality.
4. It demonstrates broad applicability by working across LLaVA, InternVL, and video variants, a rare feat in pruning research.

### Weaknesses
1. The theoretical foundation is weak, as the relation between reconstructability and informativeness is intuitive but unproven.
2. Comparisons are limited to some pruning baselines, omitting more visual token selection methods.
3. Claims of video generalization are weakly supported, as the results were only tested on a limited settings of model and benchmark, and no temporal analysis was conducted.
4. Insufficient display and analysis of failed cases.

### Questions
1. What is the method's sensitivity to its hyperparameters (p, q, α1, α2), and were any adaptive or learned strategies explored for them?
2. Regarding the video experiments, how is the issue of temporal redundancy addressed? Is token selection performed independently for each frame?
3. Why not conduct experiments and tests on SOTA VLMs like Qwen2.5-VL?
4. Why not present and analyze some failed cases?

### Soundness
2

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
This paper proposes an adaptive visual token selection method for MLLMs. The authors employ a Transformer-based architecture as both the feature selector and the feature reconstructor, and use a Gumbel-Softmax layer to generate the final token masks. The proposed method achieves up to a 50% reduction in FLOPs while maintaining 99–100% of the original performance.

### Strengths
1. The authors utilize a VQ-VAE–like method to train an adaptive token selector, while using the model to fit the reconstruction function R and the selector function S, which is a reasonable design.
2. The results shown in Fig. 1 are impressive, as most detailed structures, such as characters, are well preserved, demonstrating the method’s superior capability in high-resolution OCR tasks.
3. The paper is well written and easy to follow, with clear mathematical formulations that guide the reader through the methodology.

### Weaknesses
1. The experimental section is somewhat confusing. In Table 1, DivProne is missing at 1T FLOPs, while PDrop is only reported at 8T FLOPs, and neither method appears in Table 2. It would be better to include all methods for a fair comparison.
2. As the authors stated in Section 4.4, the method is similar to VQ-VAE. However, for such a relatively simple approach, more analysis would be valuable, such as explaining why a stacked Transformer block was chosen as both the selector and the reconstructor, and why Gumbel-Softmax was used as the final layer.

### Questions
1. In Fig. 4, it would be helpful to include an ablation study using more than four Transformer layers to evaluate the effect of model depth.
2. The choice of Gumbel-Softmax requires further discussion; perhaps a simple linear layer could achieve similar results.
3. Why would this method benefit high-resolution OCR tasks? A theoretical analysis or possible explanation would make this claim more convincing.

### Soundness
3

### Presentation
3

### Contribution
2
