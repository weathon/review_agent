# CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8, 6

## Abstract
We present CogVideoX, a large-scale text-to-video generation model based on diffusion transformer, which can generate 10-second continuous videos that align seamlessly with text prompts, with a frame rate of 16 fps and resolution of 768 x 1360 pixels. 
Previous video generation models often struggled with limited motion and short durations.
It is especially difficult to generate videos with coherent narratives based on text. 
We propose several designs to address these issues. 
First, we introduce a 3D Variational Autoencoder (VAE) to compress videos across spatial and temporal dimensions, enhancing both the compression rate and video fidelity. 
Second, to improve text-video alignment, we propose an expert transformer with expert adaptive LayerNorm to facilitate the deep fusion between the two modalities.
Third, by employing progressive training and multi-resolution frame packing, CogVideoX excels at generating coherent, long-duration videos with diverse shapes and dynamic movements. 
In addition, we develop an effective pipeline that includes various pre-processing strategies for text and video data.
Our innovative video captioning model significantly improves generation quality and semantic alignment. 
Results show that  CogVideoX achieves state-of-the-art performance in both automated benchmarks and human evaluation.
We publish the code and model checkpoints of CogVideoX along with our VAE model and video captioning model at https://github.com/THUDM/CogVideo.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
CogVideoX is a large-scale text-to-video generation model using a diffusion transformer, capable of generating 10-second videos at 16 fps with a resolution of 768×1360 pixels. It addresses previous models' limitations in movement and duration by introducing three components: a 3D Variational Autoencoder for improved video compression and fidelity, an expert transformer with adaptive LayerNorm for enhanced text-video alignment, and progressive training with multi-resolution frame techniques for coherent, long-duration videos with significant motions. The paper introduces a text-video data processing pipeline to enhance generation quality and semantic alignment. CogVideoX achieves state-of-the-art performance in text-to-video generation, and its components. The authors also make the model weights of the 3D Causal VAE, the video caption model, and CogVideoX publicly available.

### Strengths
1. The open-sourcing of the 3D Causal VAE, the video caption model, and the CogVideoX model significantly promotes future developments in video generation research.
2. The method can generate videos with larger frames, better temporal consistency, and higher resolution.
3. The proposed Multi-Resolution Frame Pack and Progressive Training techniques are interesting and meaningful.
4. This paper introduces sufficient technical improvements to enhance the model's performance.
5. Experimental results demonstrate that CogVideoX achieves superior performance compared to current text-to-video generation models.

### Weaknesses
1. Providing additional details in the methods section would enhance the paper’s completeness. For example, further explanation on implementing videos of different durations (and resolutions) in the same batch would be helpful. Although it is mentioned that the method is inspired by Patch n’Pack, a brief clarification would be beneficial.
2. It would be better to discuss the additional computational costs associated with using 3D attention compared to the commonly used 2D+1D attention.
3. The axis labels in Figure 8 appear to be mislabeled.
4. It would be better to add a related work section to discuss video generation works and the differences from previous methods and architectures. Additionally, references for some methods are missing in Table 3.

### Questions
1. There is a typo in line 310: "Patch’n Pack."
2. Line 459 seems to be missing a citation.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduced CogVideoX, a large-scale text-to-video generation model based on diffusion transformers, addressing common issues in prior video generation models, such as limited motion and short duration. It introduces several designs like 3D VAE, which I think is novel.

Experimental results demonstrate that CogVideoX achieves state-of-the-art performance across multiple machine metrics and human evaluations.

### Strengths
1.There are few diffusion transformer-based models. This paper provides a comprehensive approach, from training a 3D VAE to constructing a text-video data processing pipeline, along with a robust model architecture and training design. The model and code are well-implemented.

2.The paper is well-structured, and very easy to follow.

3.The ablation study is thorough, verifying the effects of 2D+1D attention and 3D attention design in video generation, as well as different positional encodings. The 2B and 5B model training partially validates the scaling law.

### Weaknesses
1. The novelty at the model level is relatively weak, with two expert adaptive layer norms being a fairly simple design.
2. The authors use T5 as the encoder, but comparisons with mixed text encoders are missing.
3. The 3D VAE design is primarily compared with its own variants, without comparison to other VAE designs, such as the spatial compression capability of SD’s 2D VAE.
4.Although the data processing pipeline is clearly outlined, the dataset is not publicly available.  It would be nice to open-source the data.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper proposes a text-to-video generation model focused on generating video with temporal consistency and rich motion in longer sequence. It proposes a 3D VAE for video compression, achieving high-quality reconstruction, and introduces a new Transformer architecture to enhance semantic alignment between text and video. Experimental results show that CogVideoX outperforms existing models, especially in complex dynamic scenes. With open-source release, CogVideoX has potential to advance research in video generation.

### Strengths
* Easy to follow 
* Effective apporach to text-to-video generation based on diffusion transformer 
* Demonstration of high-quality video

### Weaknesses
* Overall, the performance improvement does not appear to be significant. For example, the CogVideoX-2B model outperforms only in the Dynamic Degree (compared to CogVideoX-5B). Additionally, CogVideoX-5B does not achieve the best performance across all models and metrics.
* The computational cost (in both time and memory) and the complexity of data filtering and training seem high. Authors should specify these.

### Questions
* In the paper, ablation studies have been evaluated with only FVD scores. However, for Expert AdaLN, which focuses on alignment between text and video data, it would be reasonable to include other metrics, such as the CLIP score, to provide more robust validation of the ablation study results.

* What causes the 2B model to underperform compared to other baselines and the 5B model except dynamic degree? 

* Can the model generate similar-quality videos without using the caption upsampler? The caption upsampler may hinder robust generalization performance.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper is a text-to-video generation model using a diffusion transformer that generates 10-second, high-resolution videos aligned with text prompts without super-resolution or frame-interpolation. It improves upon previous models through a 3D causal VAE for text-video alignment, coherent, long-duration, and motion-rich videos. It achieves state-of-the-art performance and is open-source

### Strengths
I believe this paper is good, an open-source and open-hyperparameters, will have a impact on community research.

1. The paper is well written.
2. This paper creates a well-defined dataset for text-to-video generation.
3. This paper publicly release 5B and 2B models, including text-to-video and image-to-video versions.
4. This paper achieves state-of-the-art performance compared with other text-to-video models.
5. The generated high-resolution videos have very good quality.

### Weaknesses
1. The training process and model structure are somewhat unintuitive and slightly complex, raising some concerns about performance improvement in the future.
2. There is a lack of detailed analysis on the ablation study. If there were a more detailed analysis on the ablation study, I would raise the score.
3. Why is 2D + 1D attention unstable? In Figure 8, is the X-axis FVD and the Y-axis Training Steps?
4. In 459 line, () is typo ?
5. In 475 line, 17 frame is right ?

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces CogVideoX, an advanced text-to-video generation model built on a diffusion transformer. It produces 10-second, high-resolution videos (768×1360) at 16 fps. To overcome challenges in video generation, the authors implement:
A 3D Variational Autoencoder (VAE) for better compression and video quality.
An Expert Transformer with adaptive LayerNorm to enhance text-video alignment.
Progressive Training and Multi-Resolution Frame Packing to create coherent, long-duration videos with significant motion.

### Strengths
1. 3D-RoPE for Video Data: The adaptation of Rotary Position Embedding (RoPE) to 3D (3D-RoPE) is novel, effectively capturing spatiotemporal relationships and adding originality to positional encoding.
2. The qualitative visualizations showcase various video domains, including scenes, single-object videos, and multi-object videos.

### Weaknesses
1. The paper highlights CogVideoX’s performance but lacks a detailed analysis of computational efficiency, including memory usage and training/inference time.
2. The paper does not discuss the scalability of CogVideoX to longer video durations beyond 10 seconds.

### Questions
Please refer to the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3
