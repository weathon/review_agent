# mPLUG-Owl3: Towards Long Image-Sequence Understanding in Multi-Modal Large Language Models

- Decision: Accept (Poster)
- Scores: 8, 8, 5, 6

## Abstract
Multi-modal Large Language Models have demonstrated remarkable capabilities in executing instructions for a variety of single-image tasks. Despite this progress, significant challenges remain in modeling long image sequences. In this work, we introduce the versatile multi-modal large language model, mPLUG-Owl3, which enhances the capability for long image-sequence understanding in scenarios that incorporate retrieved image-text knowledge, multimodal in-context examples, and lengthy videos. Specifically, we propose novel hyper attention blocks to efficiently integrate vision and language into a common language-guided semantic space, thereby facilitating the processing of extended multi-image scenarios. We conduct evaluations on 21 benchmarks that cover single/multi-image, and short/long video understanding. mPLUG-Owl3 achieves competitive performance with the state-of-the-art methods while reducing inference time and memory usage by 87.8\% and 48.5\% in average. Moreover, we propose a Distractor Resistance evaluation to assess the ability of models to maintain focus amidst distractions. mPLUG-Owl3 also demonstrates outstanding performance in distractor resistance on ultra-long visual sequence inputs. We hope that mPLUG-Owl3 can contribute to the development of more efficient and powerful multimodal large language models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a novel multi-modal large language model, mPLUG-Owl3, designed for effective and efficient multi-image understanding. The core design is a novel hyper attention block that integrates visual and textual information. Comprehensive evaluations across 21 benchmarks—including single/multi-image and short/long video understanding—demonstrate the strong performance of mPLUG-Owl3. Additionally, the authors introduce a Distractor Resistance evaluation to assess the model's ability to maintain focus in the presence of distractions.

### Strengths
1.	The hyper attention block is an applaudable design that effectively fuses visual and textual information while preserving fine-grained visual input.
2.	The detailed ablations in Section 4.4 convincingly demonstrate the effectiveness of key design choices in the proposed model.
3.	The model shows strong performance across a wide range of image and video benchmarks.

### Weaknesses
1.	Although the proposed model demonstrates strong performance across a broad range of benchmarks, the comparison with other models lacks fairness from a data perspective. It would be beneficial to discuss the influence of data mixing and to establish at least one setting that is directly comparable to existing models.
2.	What is the impact of joint training on single-image, multi-image, and video data?
3.	Why does mPLUG-Owl3 freeze the vision encoder, considering that this design choice may adversely affect performance on BLINK, as discussed in Section 4.3?

### Questions
1.	What is the training cost associated with mPLUG-Owl3?
2.	How does mPLUG-Owl3 perform in terms of OCR, given that the hyper attention block could also be beneficial for high-resolution images?

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
3

### Summary
mPLUG-Owl3 is a new multi-modal foundation model that efficiently processes long image sequences using hyper attention blocks. The model can handle up to 1,000 images at high speeds and excels in various tasks including retrieval-augmented generation and video understanding. Among 8B-level models, it achieves state-of-the-art results in 15 out of 21 benchmarks while reducing inference time by 87.8% and memory usage by 48.5% compared to traditional methods.

### Strengths
The paper provides comprehensive coverage of existing benchmarks across various domains (including math reasoning, (NLVR), Mantis evaluations, and others), effectively comparing their hyper attention module against existing methods.

The authors present clarity in explaining the Hyper Attention module and its distinct differences from existing methods, particularly highlighting its efficiency improvements and architectural advantages.

The research addresses the important and challenging problem of long image sequence understanding in MLLMs, which is important as current models primarily focus on single-image tasks while real-world applications often require processing multiple images or video frames.

### Weaknesses
The paper compares mPLUG-Owl3 against diverse architectures (Mantis-SigLip, Cambrian, Idefics2) while acknowledging their different pretraining approaches, but doesn't isolate the benefits of Hyper Attention from other variables like training data.

### Questions
How do the effects of your pretraining and fine-tuning data look without the hyper attention module? Can we attest the performance gain to the hyper attention module? Just want a fair standard of comparison between models you are comparing against since you have different pertaining and fine-tuning methods compared to previous models in addition to the hyper attention module.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
PLUG-Owl3 address a core limitation in existing multi-modal language models: the challenge of effectively and efficiently processing long image sequences. While current models like LLAVA and IDEFICS handle single or limited multi-image scenarios, they struggle with latency and memory overhead as the sequence length increases. To tackle this, mPLUG-Owl3 introduces Hyper Attention Blocks, which incorporate visual and textual information in a shared semantic space through cross-attention parallel to the standard self-attention. This design significantly reduces memory usage and inference time, facilitating a smoother experience with large datasets of images or video frames.

The model is evaluated across various benchmarks (single/multi-image and video understanding) and demonstrates competitive results, particularly in multi-image and video understanding, where it reduces inference time by up to 87.8% and memory usage by 48.5% compared to traditional concatenate-based methods. Additionally, the authors propose a novel Distractor Resistance benchmark to assess the model’s robustness in long visual sequences containing distracting images, which is essential for real-world applications.

### Strengths
mPLUG-Owl3's Hyper Attention architecture shows a substantial improvement in computational efficiency and memory usage, making it a strong candidate for practical applications involving long video or multi-image data. The new benchmark for distractor resistance in long visual sequences fills an important gap in the multi-modal evaluation landscape, addressing real-world challenges where models encounter irrelevant data alongside relevant content. The model's consistent performance across various benchmarks highlights its versatility and robustness. Its capacity to handle up to 1,000 images with high speed and accuracy shows potential for scaling and deployment in extensive visual and textual environments. The adaptive gate and modality-specific components within the Hyper Attention blocks allow selective visual feature extraction, maintaining relevant information while reducing interference from unrelated data. This selective mechanism improves the model's performance in complex multi-modal tasks.

### Weaknesses
1. Lack strong baselines in the benchmark comparison including internVL2 [1] and Qwen2-VL [2]. For example, for TextVQA [3], Qwen2-VL 2B gets 79.7 while mPLUG-Owl3 8B gets 69. Including strong baselines into the paper may help readers see the effectiveness of the proposed model and architecture.
2. Hyper Attention Transformer Block seems to be a improved Q-former [4] architecture or visual abstractor [5] architecture. However, Q-former architecture has been proven that will inevitably lose visual information based on previous works [6]. Maybe applying a projection to align visual and textual feature would be more effective.
3. The design of mPLUG-Owl3 seems really improve the long-context multi image understanding ability based on figure 4, but again it may be helpful to compare the model with the most state-of-the-art models. Only then can readers see the real effectiveness of mPLUG-Owl3

[1] https://internvl.github.io/blog/2024-07-02-InternVL-2.0/
[2] Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution
[3] Towards VQA Models That Can Read
[4] BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models
[5] mPLUG-Owl2: Revolutionizing Multi-modal Large Language Model with Modality Collaboration
[6] DeCo: Decoupling Token Compression from Semantic Abstraction in Multimodal Large Language Models

### Questions
1. Based on Table 11, I would like to know why you choose these three indices as the layers to introduce Hyper Attention Layers? Could we have a general insight of where to introduce Hyper Attention Layers?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a multimodal large language model, mPLUG-Owl3, which is capable of processing multi-image and video understanding through interleaved training. The proposed model adopts the Hyper Attention Transformer Block, which inserts an additional layer into each transformer layer to form a cross-attention-based architecture. The authors train a series of MLLMs of varying sizes based on three stages, and the experiments demonstrate superior performance of these models compared to the baselines.

### Strengths
Please see the summary above.

### Weaknesses
- Important details regarding the hyperparameter settings are missing when evaluating mPLUG-Owl3 in long video understanding scenarios (e.g., the frame sampling strategy, the number of input frames for mPLUG-Owl3 and other baselines).
  - The content in Appendix H.2 is incomplete.

### Questions
- For the MI-Rope, my understanding is that all patches within an image $I_n$ share the same position embedding as the placeholder $T_{img}$. Wouldn't this setting discard the relative positions of patches within the image?
  - What is the number of frames and the corresponding frame sampling strategy when applying the proposed model to long video understanding (e.g., LongVideoBench)? Since the model’s maximum sequence length is 4096, wouldn't the large number of frames potentially exceed the maximum length of the position embedding?
  - Regarding the distractor resistance in long visual contexts, what is the main factor that contributes to mPLUG-Owl3’s superior capability in this task? Could you provide a brief analysis or explanation?

### Soundness
3

### Presentation
3

### Contribution
3
