# InterActHuman: Multi-Concept Human Animation with Layout-Aligned Audio Conditions

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
End-to-end human animation with rich multi-modal conditions, e.g., text, image and audio has achieved remarkable advancements in recent years. However, most existing methods could only animate a single subject and inject conditions in a global manner, ignoring scenarios where multiple concepts could appear in the same video with rich human-human interactions and human-object interactions. Such a global assumption prevents precise and per-identity control of multiple concepts including humans and objects, therefore hinders applications. In this work, we discard the single-entity assumption and introduce a novel framework that enforces strong, region‑specific binding of conditions from modalities to each identity's spatiotemporal footprint. Given reference images of multiple concepts, our method could automatically infer layout information by leveraging a mask predictor to match appearance cues between the denoised video and each reference appearance. Furthermore, we inject local audio condition into its corresponding region to ensure layout-aligned modality matching in an iterative manner. This design enables the high-quality generation of human dialogue videos between two to three people or video customization from multiple reference images. Empirical results and ablation studies validate the effectiveness of our explicit layout control for multi-modal conditions compared to implicit counterparts and other existing methods.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a human animation framework that synthesizes human-centric videos or images aligned with both audio tracks and text prompts, based on reference images of individuals and multimodal inputs such as audio. Specifically addressing the issue of temporal and identity misalignment between audio tracks and speaking movements of people in the video, the paper introduces a simple yet effective mask prediction mechanism. This mechanism progressively performs local audio condition injection during the denoising process, enabling dynamic audio-spatial alignment without requiring precise masks, thus avoiding incomplete or misaligned generation.

### Strengths
1. Per-identity control of multiple concepts is an urgent problem in human animation tasks; this paper offers a simple yet reliable solution that is not limited by the number of people in the video and shows strong scalability.  
2. The local-audio-conditioning injection strategy is both novel and straightforward, and its effectiveness is verified by experiments and user studies.  
3. The inference results are excellent; the supplementary videos are particularly impressive.  
4. The paper explains its core contributions clearly, concisely, and without ambiguity.

### Weaknesses
1. Owing to the limited scope of the training data, the method’s ability to follow detailed instructions appears constrained; it is unclear whether this stems from a fundamental limitation in the approach or simply from the narrow range of scenarios covered by the dataset.
2. The generated humans exhibit rather subdued body motion. Yet in real conversations, lively gestures are equally important. This could be a valuable direction for the community to explore in the future.

### Questions
The paper focuses on audio and image as the multimodal conditioning cue. Could the same framework, for instance, be adapted to accept additional signals, such as skeletal key-points, so that the generated humans would perform richer, visually matched motions?

### Soundness
4

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
The paper proposes InterActHuman, a diffusion transformer based human video generation framework that aims to synthesize multi person or human object interaction videos in which each identity keeps its own appearance, motion style, and voice, given multiple reference images, a text description, and per speaker audio tracks. The core idea is to drop the usual single identity assumption and instead bind each conditioning signal to the correct spatial region over time. To do this, the model predicts for each reference concept a spatiotemporal mask via a lightweight cross attention head that matches the noisy latent video tokens to that concept. These masks are refined iteratively across denoising steps and then used to inject audio features only into the region of the speaker at the next step, which addresses the chicken and egg problem of needing masks to apply localized audio while the video is still being generated.

### Strengths
1. The paper tackles a practically important and under explored setting: multi-person, audio-conditioned human animation where each identity must keep its own appearance and voice, instead of the common single identity assumption in prior audio-driven portrait or OmniHuman style models.

2. The proposed iterative mask prediction and cached layout guided audio injection mechanism is elegant. By predicting per identity spatiotemporal masks using cross attention between reference appearance tokens and noisy video latents, then using the previous step mask to gate the next step audio cross attention, the model effectively solves the chicken and egg problem of aligning local audio before the final frames exist.

3. Quantitative and qualitative results are compelling. The user study shows a strong preference for the proposed method in both lip sync realism and subject consistency.

### Weaknesses
1. Runtime cost and scalability claims are mostly deferred to the appendix. The main text asserts minimal overhead and compatibility with long video generation, but does not quantify inference speed or memory usage when conditioning on multiple identities.

2. Multi speaker audio assignment appears to rely on injecting each audio stream only into the spatial region indicated by that speaker’s cached mask at inference time. It is unclear whether the model is ever explicitly trained on multi speaker scenes with multiple simultaneous audio streams, or whether this is essentially a zero shot composition of single speaker training. This could limit robustness when speakers interrupt each other or overlap.

### Questions
1. How are individual audio streams associated with the correct identity during training. The paper explains that during inference, audio cross attention is only applied to tokens whose mask corresponds to that speaker, using cached masks from the previous denoising step.

2. The data pipeline aligns audio segments to identities through lip synchronization and produces per-frame masks using Grounding SAM2 with a person query. Could you elaborate on how you ensure temporal identity consistency across frames in multi-person scenes, especially during occlusion or when two people have similar appearances? Do you rely on optical flow tracking or an ID re-identification module?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes InterActHuman, a video diffusion framework for multi-concept human animation, incorporating layout-aligned multimodal conditions such as local audio and image references for distinct entities. The approach introduces a mask-predictor module, iteratively integrated into the denoising steps of a diffusion transformer, to spatially align conditioning signals (audio/image) with their corresponding regions. Experiments and ablations on curated large-scale datasets reportedly demonstrate state-of-the-art performance in multi-person talking and human-object interaction video generation.

### Strengths
1. The paper provides relatively comprehensive quantitative evaluations using mainstream avatar-related metrics.
2. The authors offer detailed descriptions of how they collected and cleaned a large-scale dataset combining reference images, audio, per-frame masks, and captions for diverse multi-human/object interactions (Section 3.3), which could be a valuable resource for empirical studies if released publicly.

### Weaknesses
1. While using layout-based guidance for multi-concept conditioning is a reasonable design, the claimed novelty of this framework is questionable. Similar strategies are already standard in both multi-concept image and video generation. Even within avatar-related tasks, several prior works, including MultiTalk[1], have adopted comparable layout-guided conditioning. Moreover, dynamic layout prediction was first introduced in Ingredients[2], where it serves a clear purpose in ipt2v task. However, for avatar generation tasks, the spatial configuration of characters is typically known beforehand. In such cases, slightly loosening pre-defined masks can often achieve sufficient flexibility without introducing additional model complexity. The paper does not convincingly demonstrate that the proposed learnable mask predictor provides meaningful gains over these simpler, training-free alternatives.
2. The experimental evaluation relies entirely on a self-collected, closed-source test set. This choice severely limits reproducibility and undermines the credibility of the reported improvements. Without standardized or publicly available benchmarks, it is difficult to assess whether the observed gains generalize beyond the authors’ specific data.
3. The paper does not specify which base video diffusion model was used for initialization, nor does it provide details about training infrastructure (e.g., number of GPUs, total data scale, or training duration). More importantly, it remains unclear whether the base model natively supports audio conditioning or if such capability was newly introduced by the authors. Without this information, it is difficult to assess how much of the reported performance stems from the proposed mechanism itself versus the underlying backbone or large-scale training setup. Given that the paper reports notably strong results, this omission raises concerns about the fairness and interpretability of the comparisons. A more transparent description of the training setup and backbone configuration is essential to substantiate the claimed contributions.
[1]. Let Them Talk: Audio-Driven Multi-Person Conversational Video Generation. 
[2]. Ingredients: Blending Custom Photos with Video Diffusion Transformers.

### Questions
In Line 228-229, what is the meaning of '!'; besides, 'the last few DiT blocks' should be specified to be reproduced.

### Soundness
3

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
This paper presents InterActHuman, a novel video diffusion framework for multi-concept human animation, addressing the limitations of existing global conditioning methods in multi-person scenarios. The key contribution is a method for enforcing strong, region-specific binding of local conditions, particularly aligning audio to the correct speaker. The model integrates a mask predictor into the diffusion pipeline to explicitly infer the spatiotemporal layout for each reference identity. It solves the problem of mask prediction during inference by adopting an iterative strategy, where the mask predicted at one step guides the local audio injection at the next. The authors also contribute a large-scale dataset of over 2.6 million video-entity pairs to facilitate this task.

### Strengths
1: The framework introduces the capability for multi-person, audio-driven animation, correctly assigning distinct audio streams to specific individuals in the generated video.

2: The paper proposes a practical iterative mask-caching strategy to solve the "chicken-and-egg" problem of local conditioning, using the mask predicted at step $k$ to guide the local audio injection at step $k+1$.

3: Experiments show that the method significantly outperforms existing baselines (like Kling 1.6 w/ lip-sync and OmniHuman w/ fixed mask) on multi-person benchmarks, achieving state-of-the-art lip-sync (Sync-D) and video quality (FVD) metrics.

### Weaknesses
1: **Over-reliance on a Private, Curated Dataset.** The model is trained and evaluated on a new, large-scale dataset (2.6M pairs) curated by the authors. The multi-person test set is also newly collected by the authors. This makes it difficult to assess robustness and generalizability. Since the baselines were not trained on this specific, mask-annotated dataset, it's unclear if the performance gap is due to the model's architecture or its specialized training data, which may be perfectly tailored to its design.

2: **Missing Ablation on Inference Hyperparameters.** The core inference strategy to solve the "chicken-and-egg" problem involves disabling mask-based injection for the "first 10 steps" because early masks are deemed "unreliable". This 10-step threshold is a critical, unexplained hyperparameter. The paper provides no ablation study to justify this specific number or to analyze the sensitivity of the model's performance (e.g., in Sync-D or FVD) to this "warm-up" period.

3: **Lack of Failure Case Analysis for Mask Prediction.** The entire method for local conditioning hinges on the accuracy of the predicted mask. While the paper shows average IoU improving over denoising steps (Tables 8 & 9), it provides no qualitative analysis of failure cases.

Overall, this paper proposes a new framework to generate multi-person videos with audio and has comprehensive experiments. Thus, I vote for acceptance at this time.

### Questions
1: What does "ID query" in Figure 2 mean? Is it the tokens of ID 1?

### Soundness
3

### Presentation
3

### Contribution
3
