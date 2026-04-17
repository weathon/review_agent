# Scalable Multimodal Fine-tuning for Foundation Models via Mixture-of-LoRA

- Decision: Reject
- Scores: 2, 2, 2, 4

## Abstract
Adapting pre-trained Large Language Models (LLMs) for multimodal tasks presents a significant challenge, often hindered by the prohibitive computational cost of full fine-tuning. In this work, we introduce Mixture-of-LoRA (MoL), a novel and parameter-efficient fine-tuning framework that enables LLMs to seamlessly process and integrate multimodal inputs. MoL combines the efficiency of Low-Rank Adaptation (LoRA) with the modality-specialized design of Mixture-of-Transformers (MoT). Our approach injects small, trainable, modality-specific LoRA adapters into the frozen layers of a pre-trained LLM. While each modality's tokens are processed by these dedicated adapters to learn specialized features, the global self-attention mechanism remains intact, allowing for rich cross-modal fusion within the original LLM architecture. This design efficiently adapts the model to understand diverse data types---such as text, images, and speech---while retaining and leveraging the vast knowledge of the foundational model. Through extensive experiments, we demonstrate that MoL effectively enables pretrained foundation models to \textit{understand} and \textit{generate} multimodal tokens. Our work provides an effective and scalable solution for building multimodal systems from existing unimodal foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes **Mixture-of-LoRA (MoL)**, a parameter-efficient approach to extend *text-only* large language models (LLMs) to **multimodal understanding and generation** (text–image–audio).
The key idea is to insert modality-specific LoRA adapters into the attention projections ((W_Q, W_K, W_V, W_O)) of a frozen pretrained LLM.
Each modality (e.g., text, image, audio) gets its own adapter that produces (Q, K, V) representations. These are concatenated in sequence order and added, scaled by (a / r), to the frozen attention projections before shared global self-attention.
This design borrows the intuition of Mixture-of-Transformers (MoT), separate per-modality projections with shared global attention, but replaces full per-modality parameters with low-rank adapters, achieving better scalability and efficiency.
Experiments adapt Qwen-2-0.5B, Llama-3.2-1B, and Llama-3.2-3B using image and audio tokenizers (VQ-VAE for images and DinoSR for audio). The models are trained on mixed-modality next-token prediction tasks using LAION-400M, MS-COCO, Flickr30k, and MLS-en datasets.
Results show that MoL achieves steadier and lower loss curves than single shared LoRA baselines, indicating improved stability and cross-modality learning.

### Strengths
1. The paper provides  a precise, simple recipe to make a frozen text LLM multimodal using per-modality LoRA on attention projections with global self-attention. The MoL diagram and Algorithm 1 are easy to follow. 

2. Across Qwen-0.5B, Llama-1B, and Llama-3B, MoL reduces losses more steadily than a single shared LoRA, suggesting reduced cross-modality interference. 

3. It finds text adapters matter: using only image adapters harms stability and text prediction; symmetric adaptation helps cross-modal alignment

### Weaknesses
1. This paper lacks of downstream task evaluation. Claims of “understand and generate” are not backed by standard tasks (e.g., COCO captioning, VQAv2/GQA, NLVR2, Text-to-Image FID/IS/CLIP-score, ASR/speech-to-text/XTTS). The show of  loss trends is not enough.

2. This paper also missing some related  baselines. The comparison is mainly to a single shared LoRA. Authors could consider more baselines like: (1) Strong connector baselines (e.g., CLIP/ViT encoders + projection to LLM with PEFT). (2) MoT-style finetuning of a frozen LLM via lightweight per-modality projections (if feasible). (3) Other PEFT variants (IA³, LoRA-plus, DoRA) or multi-adapter routing. It is hard to figure out the benefit of multi-lora for multimodal learning without task specific baselines.

3. The authors report FFN-adapter divergence but this hasn't been  probed (e.g., optimizer dynamics, scale mismatch, rank/α sweeps, gradient stats, adapter placement sensitivity).

### Questions
1. Can you report task metrics to substantiate “understand & generate”? Suggested minimal suite:
Text↔Image: COCO captioning (CIDEr/SPICE), NoCaps, VQAv2/GQA; text-to-image (e.g., coarse CLIP-score).
Audio: MLS-en ASR WER; spoken captioning or speech-grounded QA.


2. Which attention projections contribute most (Q vs K vs V vs O)? Any benefit from placing adapters only in deeper layers or at cross-modal “fusion” layers identified by probing?

3. Can you provide FLOPs, tokens/sec, and latency overhead vs single-LoRA and connector baselines for the 1B/3B models? The trainable-share table is helpful but efficiency numbers would be more actionable.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Mixture-of-LoRA (MoL), a scalable and parameter-efficient approach for multimodal fine-tuning of large language models (LLMs).The key idea is to insert modality-specific low-rank adapters (LoRA modules) into the attention and output projection layers while maintaining shared global attention for cross-modal fusion.

Experiments are conducted on Qwen-2 (0.5B) and Llama-3.2 (1B, 3B) models, covering text, image, and audio modalities.
The method demonstrates better convergence and multimodal alignment compared to shared LoRA baselines, particularly for small and medium-sized models.

However, the use of mixed base model families (Qwen and Llama) complicates interpretation — improvements may partly arise from differences in tokenizer design, pretraining corpus, or training stability rather than from the MoL architecture itself.
Furthermore, MoL’s benefit diminishes as model size increases, and full-duplex (streaming) speech interaction remains unsupported.

### Strengths
1. MoL elegantly combines Mixture-of-Transformers and LoRA for efficient multimodal adaptation.

2. Achieves significant reduction in trainable parameters with comparable or better multimodal alignment.

3. Detailed ablation analysis — Addresses FFN instability, modality-specific LoRA effects, and learning-rate sensitivity.

### Weaknesses
1. Diminishing returns at larger scales. Gains drop from ~10% at 0.5B to <2% at 3B; **likely negligible at ≥8B or 33B**.

2. Mixed base models confound interpretation. Using both Qwen and Llama introduces architecture, tokenizer, and corpus differences; observed gains may be partially due to base-model factors rather than MoL itself. Without a Qwen-only or Llama-only scale study, scaling trends cannot be rigorously isolated.

3. Limited downstream evaluation. No quantitative benchmarks (VQA, captioning, ASR, etc.), relying solely on validation loss.

### Questions
1. Run two controlled single-family studies (Qwen-only and Llama-only) to isolate MoL’s true scaling behavior.

2. Include downstream quantitative tasks.

3. Investigate cross-family transfer (e.g., adapters trained on Qwen applied to Llama) to test modular generalization.

### Soundness
2

### Presentation
2

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
This paper proposes a mixture of LoRA inspired by the mixture of transformers (MoT) for effective multimodal fine-tuning for understanding and generative tasks. Overall, the idea of this work is incremental and lacking quantitative results and analysis, and the paper is still in a pre-mature stage for submission (much improvement required for a more comprehensive experiment and comparison with existing works).

### Strengths
1) The paper investigates a mixture of LoRA inspired by the mixture of transformers (MoT) for multimodal fine-tuning, studying the potential of replacing MoT with LoRA.
2) The method is well described and straightforward to understand, with no complex theory or module required.
3) The choice experiment is justifiable; it would be good to make it more comprehensive.

### Weaknesses
1) The proposed method appears to be a simple replacement of LoRA for MoT? which sounds subliminal as a contribution.
2) Comparison with the baseline MoT is missing. Why?
3) The results are presented in terms of the loss curves, which could be improved to make them a comprehensive assessment. The current form could hardly reflect the actual performance in multimodal tasks.
4) It would be better to show quantitative comparison in a Table form for all experiments.
5) There is a lack of comprehensive comparative analysis to show how this work challenges SOTA; comparing with baseline LoRA is insufficient.

### Questions
NA

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
This paper proposes Mixture-of-LoRA (MoL), a parameter-efficient method for adapting pretrained text-only LLMs to multimodal tasks. The approach injects modality-specific LoRA adapters into frozen LLM layers while maintaining global self-attention for cross-modal fusion. The authors evaluate MoL on text-image and text-image-audio tasks using models up to 3B parameters, demonstrating improved training stability compared to vanilla LoRA baselines.

### Strengths
-The combination of LoRA's efficiency with MoT's modality-specific design is sensible and addresses a clear need for cost-effective multimodal adaptation. 
-The paper is well-written with good visual aids that make the method easy to understand. 
-The ablation studies provide valuable insights, particularly the finding that text adapters are necessary even when adapting to new modalities. 
-The approach enables both understanding and generation of multimodal content, which is more general than understanding-only methods.

### Weaknesses
-This paper only reports pretraining losses without downstream tasks evaluation (VQA, image captioning, etc.). Loss curves alone are insufficient to demonstrate the method's practical utility. 

-While MoL is more parameter-efficient than MoT, the core idea of modality-specific processing is directly borrowed. The main difference is using LoRA instead of full weight matrices, which feels incremental. A more thorough comparison with MoT (ideally empirical) would strengthen the positioning. 

-The baseline LoRA setup may be suboptimal (same learning rate for all modalities).

-Section 5.2 reveals that adding MoL to FFN layers causes loss divergence despite extensive hyperparameter tuning. This suggests the method may be brittle and require careful tuning.

### Questions
-This paper only reports pretraining losses without downstream tasks evaluation (VQA, image captioning, etc.). Loss curves alone are insufficient to demonstrate the method's practical utility. 

-While MoL is more parameter-efficient than MoT, the core idea of modality-specific processing is directly borrowed. The main difference is using LoRA instead of full weight matrices, which feels incremental. A more thorough comparison with MoT (ideally empirical) would strengthen the positioning. 

-The baseline LoRA setup may be suboptimal (same learning rate for all modalities).

-Section 5.2 reveals that adding MoL to FFN layers causes loss divergence despite extensive hyperparameter tuning. This suggests the method may be brittle and require careful tuning.

-Some loss curves are hard to read due to overlapping lines.

### Soundness
2

### Presentation
3

### Contribution
2
