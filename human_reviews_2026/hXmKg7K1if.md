# Rate-Adaptive Quantization: A Multi-Rate Codebook Adaptation for Vector Quantized Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 2, 8, 2

## Abstract
Learning discrete representations with vector quantization (VQ) has emerged as a powerful approach in representation learning across vision, audio, and language. However, most VQ models rely on a single, fixed-rate codebook, requiring extensive retraining for new bitrates or efficiency requirements. We introduce Rate-Adaptive Quantization (RAQ), a multi-rate codebook adaptation framework for VQ models. RAQ integrates a lightweight sequence-to-sequence (Seq2Seq) codebook generator with the base VQ model, enabling on-demand codebook adaptation to any target size at inference. Additionally, we provide a clustering-based post-hoc alternative for pre-trained VQ models, suitable when modifying the training pipeline or joint training is not feasible. Our experiments demonstrate that RAQ performs effectively across multiple rates and VQ models, often outperforming fixed-rate baselines. This model-agnostic adaptability enables a single system to meet varying bitrate requirements in reconstruction and generation tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces Rate-Adaptive Quantization (RAQ), a framework for dynamically adjusting the codebook size in VQ models without retraining. RAQ proposes a Seq2Seq codebook generator and a model-based clustering alternative with differentiable $k$-means. Experiments across multiple VQ models and datasets show that RAQ matches or exceeds fixed-rate baselines across various metrics. The framework adds minimal overhead and codebook adaptation latency, making it feasible in practice.

### Strengths
1. RAQ enables dynamic adjustment of codebook size in VQ models without retraining, addressing a critical limitation in existing VQ approaches that require separate models for different bitrates.
2. The authors conducted comprehensive experiments across multiple VQ architectures (VQ-VAE, VQ-VAE-2, VQGAN, SQ-VAE, and SimVQ) and datasets (CIFAR10, CelebA, ImageNet), demonstrating RAQ's generalizability beyond a single model type.
3. Ablation studies on cross-forcing are provided, showing better performance than the w/o-CF counterpart.

### Weaknesses
1. The paper mentions that RAQ adds modest parameter overhead, but doesn't analyze the time complexity of codebook generation.
2. It is not clear why the mixup ratio of teacher forcing and free running is 5:5; it would be valuable to discuss cross-forcing with other mix ratios (e.g., 25% TF / 75% FR).
3. As shown in Table 7, the training speed is much slower per epoch than fixed-rate training. The authors may discuss the total training time required to achieve the same performance as well.

### Questions
1. In Section A.2.2, the IKM method for codebook expansion uses MMD loss, but the implementation details are sparse. What kernel was used for MMD computation, and how was the bandwidth parameter selected? Additionally, how does the L2 regularization parameter $\lambda$ affect the quality of expanded codebooks across different expansion ratios ($\widetilde{K}/K$)?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Rate-Adaptive Quantization, a framework that aims to enable vector-quantized models such as VQ-VAE or VQGAN to operate at multiple effective bitrates without retraining.
 The proposed approach adds a Seq2Seq-based generator that takes the base codebook eee as input and produces new “rate-adapted” codebooks of arbitrary sizes. A simpler variant based on differentiable k-means is also described.
 Experiments across several VQ architectures and datasets show that the approach maintains reconstruction performance close to fixed-rate baselines while allowing continuous control over codebook size.

### Strengths
* The paper addresses a relevant and practically motivated problem: adapting the bitrate of VQ models post hoc without retraining.
* The proposed framework is model-agnostic and can, in principle, be integrated with many existing VQ-based architectures.
* RAQ achieves reconstruction results comparable to multiple fixed-rate baselines while it offers flexible codebook resizing.

### Weaknesses
* The Seq2Seq generator receives the base codebook e as its only input. Consequently, it cannot encode more information than e already contains. The output of the sequence model can only reparameterize existing embeddings and cannot actually increase the information rate. This makes the usage of K larger than Kmax  questionable. The method cannot truly “expand” quantizer capacity, it can only reshape the existing embedding space. The training setup samples K~ up to a fixed Kmax that matches the maximum value used during evaluation.
 Evaluations are only performed at a few discrete points within this range, so generalization to unseen rates is untested.
 This means that one of the main claims, increasing the bitrate after training, is completely untested for this approach.

* The paper does not compare to Residual Vector Quantization or multi-stage quantization approaches, which already achieve adaptive bitrate by truncating residual stages. Especially since RAQ was never tested for codebooks beyond what it has seen during training, a comparison to RVQ would be necessary to contextualize its capabilities.

* Finite Scalar Quantization is another relevant recent method that provides rate control without retraining.
 FSQ achieves the same practical goal (variable bitrate, same encoder/decoder) with a much simpler mechanism.
 A comparison to FSQ is essential to substantiate the claimed advantage of RAQ.
 Only claiming that RAQ “builds on top of VQ” is a weak justification for excluding FSQ from comparison.

* The evaluation focuses solely on image reconstruction.
 No downstream generative modeling (e.g., autoregressive priors or diffusion) is tested despite claims of generality.
 Other domains, such as audio or speech quantization, would also be natural use cases.

* There is no analysis of how the Seq2Seq architecture choice (e.g., LSTM vs. MLP or transformer) affects the outcome.

### Questions
* Can RAQ generalize to unseen rates beyond those sampled during training?
* Why was no comparison to FSQ or residual quantization baselines included?
* How sensitive is performance to the Seq2Seq architecture?
* Could RAQ be evaluated on other modalities (e.g., audio) to demonstrate true generality?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes Rate-Adaptive Quantization, a framework that enables a single VQ model to operate at multiple bitrates without retraining. The key idea is to dynamically adapt a model’s codebook size through one of two complementary mechanisms:

1. A Seq2Seq-based generator that learns to produce codebooks of arbitrary target sizes via a cross-forcing training strategy.
2. A model-based variant leveraging DKM for post-hoc adaptation of pre-trained VQ models.

RAQ thereby allows both rate expansion (increasing codebook size for richer representation) and compression (reducing bitrate for efficiency). The authors demonstrate consistent reconstruction quality across multiple codebook sizes, datasets, and backbones. Additional experiments show that RAQ-adapted representations remain expressive when paired with downstream generative priors such as PixelCNN or Transformer-based decoders, while inference overhead is negligible after caching.

### Strengths
RAQ is a practical and well-executed contribution addressing a clear deployment bottleneck in VQ-based systems: the need to maintain multiple models for different bitrates. The proposed approach is conceptually simple yet operationally valuable, that it enables rate flexibility through lightweight codebook adaptation without retraining or altering the base model architecture.

The Seq2Seq-based RAQ introduces a cross-forcing scheme that stabilizes autoregressive codebook generation even without inherent sequence order. The model-based RAQ offers a complementary, training-free alternative that broadens applicability to pretrained models. The paper provides extensive empirical validation, including results across datasets and scales, analysis of up and downscaling behavior, OOD generalization, and stage-2 generative compatibility. Importantly, the work demonstrates that the learned codebooks remain semantically consistent and compatible with autoregressive priors, supporting the claim of multi-rate functionality.

Overall, this work’s strength lies in its engineering maturity and deployment relevance and delivers a clear, reproducible method to unify multiple fixed-rate VQ models into one adaptive framework.

### Weaknesses
The core components: autoregressive modeling and differentiable clustering are not themselves novel, and the paper could better articulate what conceptual insights arise beyond this integration. The compression case ($K>\tilde{K}$) still exhibits moderate performance drop, leaving open whether rate reduction meaningfully outperforms training a smaller model directly.

### Questions
1. The paper would benefit from a deeper explanation of why the Seq2Seq-based RAQ preserves latent semantics and reconstruction quality across widely varying codebook sizes. Could the authors provide any theoretical or empirical insight into the relationship between the codebook manifold structure and the success of cross-forcing training?
2. In scenarios where the target codebook size is much smaller (e.g., $K→\tilde{K}$), performance degradation still appears noticeable compared to directly training at $\tilde{K}$. Could the authors elaborate on when and why RAQ is preferable to simply training a smaller fixed-rate model from scratch, and how this trade-off might vary with dataset scale or model capacity?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper focuses on the need of variable bit rate neural image codec, and proposes "RAQ" to enable a single VQ codec to work at multiple bit rate.

"RAQ" is a method that creates multiple VQ codebooks (= multiple bit rate) from a base codebook. More specifically, RAQ uses a classical LSTM-based Seq2Seq architecutre, where the base codebook is treated as the condition, and the LSTM will create new codebooks according to the condition.

Since RAQ is just a module, the authors evaluated this RAQ module with 4 existing VQ-VAE frameworks, and evaluted these models on multiple datasets with multiple metrics.

Although the task is valuable and the proposed method seems to work to some extent, the experiments in the paper lack consistency, and are sometimes confusing to the reader, which makes the whole research less solid.

As a reviewer, I tend to reject this paper.

### Strengths
- Proposed the RAQ method that enables multiple codebook size within a single model while maintaining the reconstruction performance

### Weaknesses
## Inconsistent Evaluation
### Inconsistent Metrics
- In Fig.2, rFID is used, but in table.1, rFID disappears. Is there any reason?
### Inconsistent Dataset
- CIFAR-10, CelebA, ImageNet... Many datasets are used. However, in table.1, four different VQ frameworks were trained on the different datasets.
### Inconsistent Pixel Resolution
- Why VQGAN is trained on ImageNet 256x256, while SimVQ is trained on 128 x 128? 

- I know that in SimVQ paper, most ablation studies are conducted under 128x128. If the authors are simply using those pretrained model weights, I can understand why keep using 128x128, but I don't think this is the fact.

    - Please see my next review for details.
## Baseline model weights used in evaluation
- I don't think the authors used official pretrained model weights for the SimVQ model with codebook size 128, 256, 512, 2048, 4096.
In SimVQ official webpage, weights of the above codebook sizes cannot be found.

    - If the authors re-train these baseline models with different codebook size, then using different datasets and pixel resolution is a confusing decision.
## Base codebook size
- Moreover, SimVQ's feature is that it works for a codebook size of 18bit (262144). VQ models with huge codebook size are becoming a trend (IBQ is another example that supports 18bit codebook https://arxiv.org/abs/2412.02692). 

    - Given this trend, the current paper only discusses codebook size up to 12 bit (4096), which cannot prove that RAQ works with state-of-the-art.
## Missing comparison
- Table.1 tested RAQ on multiple VQ framework, which is good. However, why not compare RAQ with RVQ or even simpler baseline methods?

    - By randomly dropping layers, RVQ model can reconstruct images with variable codebook size at inference time. 
    - A simpler baseline is to prepare multiple codebooks with different sizes, and randomly swithc across these codebooks in the VQ model training. Since in the training time the VQ model has seen multiple codebooks, at inference time, it might be able to work with either of these seen codebooks.

- With all these simple and intuitive baselines missing, we don't know if RAQ is really working better.

### Questions
Please see my comments in "weaknesses" part

### Soundness
2

### Presentation
1

### Contribution
1
