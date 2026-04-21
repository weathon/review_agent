# Autoregressive Video Generation without Vector Quantization

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 8

## Abstract
This paper presents a novel approach that enables autoregressive video generation with high efficiency. We propose to reformulate the video generation problem as a non-quantized autoregressive modeling of temporal frame-by-frame prediction and spatial set-by-set prediction. Unlike raster-scan prediction in prior autoregressive models or joint distribution modeling of fixed-length tokens in diffusion models, our approach maintains the causal property of GPT-style models for flexible in-context capabilities, while leveraging bidirectional modeling within individual frames for efficiency. With the proposed approach, we train a novel video autoregressive model without vector quantization, termed NOVA. Our results demonstrate that NOVA surpasses prior autoregressive video models in data efficiency, inference speed, visual fidelity, and video fluency, even with a much smaller model capacity, i.e., 0.6B parameters. NOVA also outperforms state-of-the-art image diffusion models in text-to-image generation tasks, with a significantly lower training cost. Additionally, NOVA generalizes well across extended video durations and enables diverse zero-shot applications in one unified model. Code and models are publicly available at https://github.com/baaivision/NOVA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a video generation method NOVA, expands MAR[1] from image generation to video generation. However, this process is not smooth sailing. The author encounters problems such as MAR's insufficient ability to model long sequences and poor extrapolation ability. The author proposes a method in the time dimension. Frame-by-frame and set-by-set generated solution strategies in spatial dimensions achieve excellent text-to-image and text-to-video performance.

[1] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024b.

### Strengths
1. As far as I know, it is the first time to expand MAR into the general generation field (text-to-image, text-to-video, etc.), which is a very good attempt.

2. The article achieves excellent text-to-image performance (on T2I CompBench) and stands out from diffusion models. Although I have some doubts about the evaluation setting and comparison methods described in the article, the results are still excellent.

3. As a pre-training paradigm for generative models, especially video generation, this article consumes less resources and is relatively affordable.

### Weaknesses
1. The article's explanation of the model architecture is vague and difficult to understand:

    ①The paper mentions temporal encoder, spatial encoder, and decoder with 16 layers each on line 260, but the full article does not explain what the encoder and decoder in spatial layers are.

    ②In Figure 1, when Spatial Layers performs mask modeling on S2, it can directly obtain the complete embedding output by S2 after being encoded by Temporal Layers. Isn't this a kind of information leakage? However, in the actual inference generation At the time of S2, there was no known S2. We only had S1 before we get S2.

2. As far as I know, in the evaluation on T2I-CompBench mentioned by the author in lines 292-295 of the article, there are only 300 evaluation prompts for each category, and the others are training prompts. And according to the setting of the original T2I-CompBench paper, each evaluation prompt should generate 10 images.

3. In the comparison of Table 2, there is no comparison with the most advanced diffusion model. At least it should be compared with SD3 [2] and DALL-E3 [3].

4. For the curve in Figure 8(a), loss is optimized best under the image-to-video (w/o scale&shift) setting. Doesn’t this mean that the scale&shift operation is useless? Then why do we need to add this operation? 

[2] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas M¨uller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024b.

[3] James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, et al. Improving image generation with better captions. Computer Science. https://cdn.openai.com/papers/dall-e-3.pdf, 2(3):8, 2023.

### Questions
In order to ensure fairness when comparing with other methods in Table 2, you should submit the comparison results under the same settings. This is also to ensure that the settings of other methods can be unified when comparing in the future. And you should emphasize whether you conducted the evaluation under zero-shot or the result after fine-tuning on the training set.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces NOVA, a novel autoregressive (AR) video generation model that leverages non-quantized tokenizers. NOVA aims to combine the advantages of high-fidelity and high-rate visual compression with in-context learning capabilities, enabling it to integrate multiple generative tasks within a unified framework. The model is designed to factorize AR video generation into temporal frame-by-frame prediction and spatial set-by-set prediction, allowing for efficient and effective video generation. The authors claim that NOVA outperforms existing diffusion models in terms of data efficiency, inference speed, and video fluency, while also demonstrating strong zero-shot generalization across various contexts.

### Strengths
1) NOVA's framework is well-structured, combining temporal and spatial autoregressive modeling. This dual approach not only enhances the model's efficiency but also its ability to handle multiple generative tasks within a single model, showcasing the potential for in-context learning.

2) The authors provide a thorough evaluation of NOVA, comparing it with state-of-the-art models across various metrics. The results demonstrate that NOVA not only matches but often surpasses the performance of diffusion models, particularly in terms of data efficiency and inference speed.

### Weaknesses
I think the key limitation of this work is the novelty, which seems like an extension of MAR on video generation task.

In Table 3, I don't see improvements in the proposed on the basis of previous diffusion-based methods. Are AR-based methods really needed for video generation tasks? Could the author clarify this?

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a video generation framework called NOVA, which is based on autoregressive modeling. NOVA performs temporal frame-by-frame prediction and spatial set-by-set prediction. This approach leverages the in-context learning and extrapolation advantages of autoregressive models while maintaining the efficiency of bidirectional modeling. Compared to existing video generative models, NOVA achieves higher data efficiency, faster inference speeds, and about similar video generation quality with fewer parameters.

### Strengths
* NOVA achieves state-of-the-art (SOTA) results in text-to-image (T2I) tasks.
* NOVA shows much faster inference speeds than previous video generative models.
* The method of combining temporal autoregressive and spatial bidirectional modeling is simple yet effective.
* The Scaling and Shift Layer is also simple but effective. Also, the analysis of the layer is comprehensive.

### Weaknesses
* While NOVA achieves SOTA in T2I, this aspect feels like a straightforward extension of MAR[1] rather than a novel contribution.
* For text-to-video (T2V), NOVA uses relatively less data and fewer parameters and has fast inference speeds but falls short in performance. Therefore, it needs further testing about scalability (i.e., if NOVA can match the performance of the open-source models in the main table when scaled up.). 
* There is a question about whether extrapolation is truly unique to autoregressive (AR) models. Diffusion models and bidirectional models could also potentially achieve extrapolation through a sliding window approach, which would need a comparative analysis.
* The ablation study on frame-by-frame autoregressive modeling lacks clarity, which is critical given the importance of this topic to the authors' main arguments. The qualitative results in Figure 7. appear less convincing when viewed in images, and it’s unclear if NOVA did not have similar subtle limitations. A side-by-side comparison with the same text prompt or inclusion of video examples would be helpful.

In short, \
Although the methods presented are simple and novel, the primary claims are not clearly backed by the experiments. The main claims are: (1) NOVA combines the advantages of temporal autoregressive modeling with spatial bidirectional modeling, and (2) NOVA demonstrates data and parameter efficiency in practice. However, these claims are not sufficiently supported by the experimental results.

If the authors provide clearer support for their claims with additional analysis, I would be happy to raise my score.

[1] Li, et al. "Autoregressive Image Generation without Vector Quantization." arXiv 2024.

### Questions
* Suggestion: To make the Scaling and Shift Layer easier to understand, the authors could mention its similarity to FiLM[1] or AdaIN[2].
* Typo: Line 296 - each with

[1] Perez, et al. "Film: Visual reasoning with a general conditioning layer." AAAI 2018.\
[2] Huang, Xun, and Serge Belongie. "Arbitrary style transfer in real-time with adaptive instance normalization." ICCV 2017.

### Soundness
3

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
This paper introduces an autoregressive model combined with a diffusion model as the prediction head, enabling video generation without vector quantization. The proposed approach, termed NOVA, maintains the causal property of AR models' temporal frame-by-frame prediction, while leveraging bidirectional modeling within individual frames (spatial set-by-set prediction).  NOVA achieves state-of-the-art text-to-image and text-to-video generation performance with significantly lower training costs and higher inference speed.

### Strengths
- As a follower of MAR (Tianhong Li et al. (2024b)), this paper for the first time lifts the non-quantized AR model to video generation. In contrast to trivially modifying the 2D non-quantized MAR to a 3D version, they design the autoregressive modeling sequentially that integrates first temporal frame-by-frame prediction and then spatial set-by-set within each frame. This facilitates the model's ability of video extrapolation and potential  compatibility with kv-cache acceleration.
- The model is trained on large-scale text-to-image and text-to-video datasets (trained from scratch) and shows high image and video generation quality compared to existing SOTA models. It will have a great potential contribution to the vision community If the pre-trained checkpoint is released.

- This paper provides valuable empirical design spirits as vilified by their experiments:
  - Instead of directly assigning the current frame of temporal layers’ outputs to the spatial layer as indicator features (for predicting the next frame), they propose to use the BOV-attended output as an anchor feature and inject the current frame's output via the Scale & Shift of LayerNorm. This technique improves the training stability and alleviates cumulative inference errors. It is a valuable prior (design choice) for subsequent studies on autoregressive long video generation.
  - They conducted extensive ablation studies to show using post-norm layers before the residual connections is a better design choice for a smoother and more stable training process.

Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. arXiv preprint arXiv:2406.11838, 2024b.

### Weaknesses
- Unclear training/inference details. 
  1. According to Figure 1. At training time, the model predicts a set of masked tokens of the 2nd frame. At inference time, the model progressively reduces the masked ratio from 1.0 to 0. However, as the 1st and 2nd frames have been generated (as the given conditional frames) in the Fig.1's example, the model should progressively unmask the 3rd frame. There seems to be some inconsistency between training and inference. In other words, for the example in Fig.1, which frame is modeled by $x_n^t$  ?
  2. It might be helpful to provide a step-by-step explanation of how frames are generated during inference, particularly highlighting any differences from training.

- Unclear video extrapolation setting.

  1. The exact number of frames used during training is not clear. According to all the information available in the paper, the model is trained on samples with 29 frames (Line315: 2.4s x 12 FPS = 28.8 frames. Line 296: the model generates 29 frames for evaluation). But this is not clearly evidenced in the descriptions of training details.

  2. It's unclear how context is handled when generating videos longer than the training length. Suppose that the model was trained on 29 frames. Is the context truncated when the length of video extrapolation exceeds 29 frames? For example, from my perspective, the video extrapolation is like: 

    ```
    ...
    [x_1,...,x_28]--> x_29 # training length is reached
    [x_2,...,x_29]--> x_30  # earliest context (x_1) is truncated
    ...
    ```

  3. How are the 1D sine-cosine temporal positional embeddings applied for frames beyond the training length? This information would help clarify the model's capabilities and limitations in video extrapolation.

### Questions
- From my understanding, the generation of the next frame is achieved by one step of temporal autoregression (stage-1) followed by several steps of spatial autoregression (i.e.,  progressively reducing the masks from 1.0 to 0) (stage-2).  What's the time cost for each of these two stages?
- As claimed in the paper, the Scaling and Shift Layer effectively reduces the cumulative inference errors. What is the limit of the model's video extrapolation capability? In other words, after how many autoregression steps will the cumulative errors severely affect the frame quality?
  - A quantitative metric on frame quality vs. the number of autoregression steps might be helpful. Or some qualitative examples that show quality degradation in long-term autoregressive generation.
- Suggestions:
  - In Figure 7, better to provide the results of NOVA and compare them with the results of the simple baseline.
  - Better to add frame ids in the qualitative examples.

### Soundness
3

### Presentation
3

### Contribution
4
