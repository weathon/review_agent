# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space

- Decision: Reject
- Scores: 4, 4, 6, 6

## Abstract
Existing text-to-image diffusion models excel at generating high-quality images, but face significant efficiency challenges when scaled to high resolutions, like 4K image generation. While previous research accelerates diffusion models in various aspects, it seldom handles the inherent redundancy within the latent space. To bridge this gap, this paper introduces DC-Gen, a general framework that accelerates text-to-image diffusion models by leveraging a deeply compressed latent space. Rather than a costly training-from-scratch approach, DC-Gen uses an efficient post-training pipeline to preserve the quality of the base model. A key challenge in this paradigm is the representation gap between the base model’s latent space and a deeply compressed latent space, which can lead to instability during direct fine-tuning. To overcome this, DC-Gen first bridges the representation gap with a lightweight embedding alignment training. Once the latent embeddings are aligned, only a small amount of LoRA fine-tuning is needed to unlock the base model’s inherent generation quality. We verify DC-Gen’s effectiveness on SANA and FLUX.1-Krea. The resulting DC-Gen-SANA and DC-Gen-FLUX models achieve quality comparable to their base models but with a significant speedup. Specifically, DC-Gen-FLUX reduces the latency of 4K image generation by 53x on the NVIDIA H100 GPU. When combined with NVFP4 SVDQuant, DC-Gen-FLUX generates a 4K image in just 3.5 seconds on a single NVIDIA 5090 GPU, achieving a total latency reduction of 138x compared to the base FLUX.1-Krea model. Code and models will be released.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes DC-GEN, a framework for accelerating large diffusion models by using a deeply compressed latent space. The authors introduce two stages: (1) embedding alignment, where the patch embedder of the diffusion model is fine-tuned to match the new latent representation; and (2) LoRA fine-tuning, used to adapt the diffusion backbone in the new latent space. They report large acceleration ratios (up to 50×) with minimal visual degradation in image generation quality.

### Strengths
1. Clear and practical motivation: The goal—reducing diffusion inference cost for large-scale image synthesis—is timely and relevant. The focus on post-training acceleration (rather than retraining) is appealing for efficiency-oriented applications.

2. Method is clearly explained: The embedding alignment and LoRA adaptation procedures are easy to follow, and the pipeline is well illustrated.

3. Empirical evaluation is adequate: The paper shows decent acceleration on high-resolution image generation and uses well-known benchmarks for comparison.

### Weaknesses
1. Limited novelty. The proposed framework largely combines known components: A standard embedding alignment step (MSE between pretrained and downsampled embeddings); LoRA fine-tuning on synthetic data. These are conventional adaptation techniques and do not introduce a fundamentally new concept or architecture.

2. “Post-training” claim is misleading. The process still requires fine-tuning a model on paired data, sometimes with significant compute and dataset size. This is not truly “post-training” in the sense of zero-cost adaptation. 

3. Limited practicality for large-scale models. Although LoRA reduces the fine-tuning cost, successfully matching the teacher model’s performance still requires access to the original training recipe—including comparable data quality, preprocessing, and post-training optimization settings. Without these, the adapted model inevitably falls short. Using synthetic data from the teacher can lower the data requirement but introduces another issue: the model may inherit and amplify the teacher’s artifacts and stylistic biases within the compressed latent space.

### Questions
Why did the authors choose to retrain the patch embedder rather than adapt the DC-VAE latent space directly to the pretrained VAE? Would aligning the autoencoder latent representation yield more consistent features with less fine-tuning?

Does the high-compression autoencoder introduce any artifacts or perceptual degradation in the generated images after adaptation? If so, how are these quantified or mitigated?

Why does the model appear to achieve better quantitative scores after switching to a high-compression autoencoder?

### Soundness
2

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
5

### Summary
This paper proposes DC-Gen, a post-training framework to accelerate pretrained text-to-image diffusion models by adapting them to a deeply compressed latent space (e.g., moving from 8× to 32×/64× autoencoder compression).
The approach introduces two stages: Embedding Alignment, which aligns the pretrained model’s latent embeddings with those of a new DC-AE autoencoder to stabilize adaptation, and LoRA Fine-Tuning which lightly fine-tunes the model to the new latent space, preserving semantics while being much cheaper. 
The method is validated on SANA and FLUX.1-Krea, showing comparable visual quality and FID/CLIP scores to base models, while achieving 53× latency reduction for 4K generation on H100 GPUs. 
DC-Gen shrinks the adaptation time to only 40 H100 GPU-days.

### Strengths
1. The paper is clearly written and easy to follow. 

2. The demonstrated results are promising. The authors provide a comprehensive study on visual results, quantitative metrics, and speed benchmarks. The speed gain and 4K generation quality are impressive. 

3. This work addresses an important research topic in a timely manner, i.e., accelerating diffusion generation through adapting the denoiser to a highly compressed latent space.

### Weaknesses
1. Novelty/technical contribution is my biggest concern. Freezing the main body (DiT blocks) of the denoiser and finetuning the projection layers to adapt to a new VAE is a reasonable but very trivial and common technique. I am not surprised that this will show faster convergence compared to simple full finetuning. 

2. I have some concerns about the baseline and comparison. The scope of DC-Gen is an efficient adaptation method for a high compression autoencoder. But most of the results are comparing baseline models. From my point of view, the most important comparison is against full finetuning/adaptation, but the authors claim this leads to model collapse. Please refer to my Q1. In the main comparisons, DC-Gen achieves a comparable quality compared to FLUX or SANA, while being much faster. But I would attribute this gain more to the high-compression VAE instead of the adaptation method.

### Questions
1. Why does vanilla adaptation fail as in Fig.3? Can you provide a more detailed analysis? Normally, diffusion training is more stable compared to previous methods, such as GAN. It is a bit surprising to see that the model is already starting to adapt to the new latent space, but just suddenly collapses. Is it because of gradient explosion? 

2. Can authors provide a latency breakdown for 4K generation? Because of the highly compressed latent space, the denoiser speed is successfully managed in this work, but I am curious about the VAE decoding cost.

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper designs a post training method to accelerate existing text-to-image diffusion models. The argues that that image generation at 4K resolution for example becomes computationally expensive due to inefficient encoding in the text-to-image diffusion models. While training new models from scratch on compressed latent spaces is expensive, the paper employes Embedding Alignment by training only the patch embedder of the model and output head, using a loss that on downsampled features from the base embedder. Stage 2 employs a lightweight LoRA training for the final alignment. This method is applied to SANA and FLUX.1-Krea models. The resulting DC-Gen-FLUX achieves a 53x latency reduction for 4K image generation on an H100 GPU. The authors claim that the reduction in training cost with respect to training the models from scratch is huge.

### Strengths
1) The paper provides a way to post train diffusion models to support higher resolution while maintaining the prior. The model is able to generate 4K images on lower budget. 

2) The paper claims  a 53x speedup in latency for 4K generation.

3) The paper employs a two stage approach where first the model representations are aligned and then adapted. This is intuitive

4) The results show that the model is able to preserve the prior of the model from which the model is distilled from.

### Weaknesses
1) While the method aligns with a particular encoder, the inherent low quality reconstruction of highly compressed models still remains a problem.

2) The paper simplifies downsampling to a simple, non-parametric operation. This method may not adequately handle high-frequency details from the original latent space. This might blur our some details.

3) Human eval is needed as FID and CLIP scores are too coarse. It is unclear if the blurriness is solved.

4) While the paper claims to operate at 4K resolution. There is no evaluation for this quality. Training on small set of 4K images could be evaluated to support the claim.

### Questions
I have questions mainly focussed on weaknesses: 

1) How do the authors defend the inherent low quality of highly compressed latent spaces? 

2) Since the downsampling is not optimal, what alternative methods can preserve the frequencies ? or given a downsampling budget what is the best way to do it , this is not clear.

3) How reliable are FID and CLIP , do they detect the blurriness in this case? how sensitive are these metrics to this artifact?

4) What about the evaluation at 4K resolution? The model can generate this resolution but how to evaluate the quality? This part is missing.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DC-Gen, which is a 3-stage pipeline that adapts a pre-trained latent diffusion model to a different VAE. In the first embedding alignment stage, a new patch embedder is randomly initialized and then aligned with the original embedder (w/ the original VAE encoder) by minimizing an MSE loss. In the second output head alignment stage, the randomly-initialized output head are jointly trained with the embedder from stage 1. In the third stage, LoRA modules are attached to backbone transformer blocks and jointly trained using the flow matching loss. The progressive adaptation ensures that the model converges stably while maintaining a relatively low training budget. The pipeline is evaluated on ImageNet DiT, SANA and FLUX, and demonstrates convincing results under all settings.

### Strengths
* The paper is well written and easy to follow, with a clear structure and most of the important details adequately covered.

* The empirical results are solid, with both generation quality and speedup demonstrated under a broad range of settings (different models, class-conditioned and text-conditioned image generation, and multiple datasets).

* It is somewhat interesting to observe that LDM backbones appear to retain certain latent-space-agnostic knowledge, which can be efficiently transferred across vastly different VAEs.

### Weaknesses
* The novelty appears to be somewhat limited. The general idea of training newly added layers first as an alignment has been a common practice for a long time and is ubiquitous in many similar use cases, e.g., appending visual encoders to the input of LLMs [1, 2]. Therefore, although this work may be among the first to highlight the significance of alignment when changing VAE in LDMs, the overall training pipeline itself appears relatively straightforward to derive.

* Following the previous point, several well-established or straightforward alternatives, such as layer-wise learning rate decay or gradually introducing LoRA modules through gating mechanism (zero-initialized and learned / following a predefined schedule), also seem to be possible ways to mitigate the interference of random weights in the early stage of the training, as is observed in the paper. It might make the advantage of the proposed method a stronger argument if more baselines could be considered (via experiments or analysis) beyond direct finetuning or training from scratch.

[1] Liu, Haotian, et al. "Visual instruction tuning." Advances in neural information processing systems 36 (2023): 34892-34916.

[2] Bai, Jinze, et al. "Qwen-vl: A frontier large vision-language model with versatile abilities." arXiv preprint arXiv:2308.12966 1.2 (2023): 3.

### Questions
* Ln 291-294 and Appendix Table 5 mentioned a separate output head alignment stage. What is the loss function used in this stage? It might be helpful to include the output head alignment stage as part of Fig. 2(b).

* At around Ln 414, how exactly is the synthetic dataset generated (e.g., from a fixed set of prompts, or generated unconditionally)?

* In Table 5, could the authors provide the wall-clock GPU hours for each training stage? Since the training time may not be proportional with the number of training steps due to the differences in training settings, this information would help identify potential bottlenecks in the proposed pipeline and in turn guide optimizations in future works.

* How are the position embeddings handled after the input shape changes?

### Soundness
4

### Presentation
4

### Contribution
2
