# Next Block Prediction: Video Generation via Semi-Auto-Regressive Modeling

- Decision: Reject
- Scores: 5, 3, 3, 3

## Abstract
Next-Token Prediction (NTP) is a de facto approach for autoregressive (AR) video generation, but it suffers from suboptimal unidirectional dependencies and slow inference speed.  In this work, we propose a semi-autoregressive (semi-AR) framework, called Next-Block Prediction (NBP), for video generation. By uniformly decomposing video content into equal-sized blocks (e.g., rows or frames), we shift the generation unit from individual tokens to blocks, allowing each token in the current block to simultaneously predict the corresponding token in the next block. Unlike traditional AR modeling, our framework employs bidirectional attention within each block, enabling tokens to capture more robust spatial dependencies. By predicting multiple tokens in parallel, NBP models significantly reduce the number of generation steps, leading to faster and more efficient inference. Our model achieves FVD scores of 55.0 on UCF101 and 25.5 on K600, outperforming the vanilla NTP model by an average of 4.4. Furthermore, thanks to the reduced number of inference steps, the NBP model generates 8.89 frames (128x128 resolution) per second, achieving an 11× speedup in inference. We also explored model scales ranging from 700M to 3B parameters, observing significant improvements in generation quality, with FVD scores dropping from 25.5 to 19.5 on K600, demonstrating the scalability of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a semi-autoregressive video generation model that enables next-block prediction (NBP),  i.e., predicting multiple tokens in parallel. The NBP model uses a block-wise causal attention matrix, i.e., causal attention inter-block and bidirectional attention intra-block, capturing better spatial dependencies. Extensive experiments show the state-of-the-art video generation quality of NBP and a significant improvement in inference speed.

### Strengths
- This paper proposes a semi-autoregressive paradigm (i.e., next block prediction) for video generation, which brings better spatial dependencies in the attention computation and a significant inference speed improvement
- This paper provides extensive experiments in terms of the design choice for block division. The model shows a good trade-off between inference speed and generation quality.
- The writing and presentation of this paper are clear and easy-to-follow

### Weaknesses
- The technical innovation from "next token prediction" to "next block prediction" is a bit trivial.
  - Since there have been many studies on the semi-autoregressive paradigm (blockwise attention and parallel decoding) in the NLP [1,2] and vision[3,4]  fields, the work done in this paper is more like an engineering application rather than a technological innovation.
  - In addition to simply changing the model prediction and the attention map, this paper does not outline the technical challenges or insights encountered in modifying an AR model to semi-AR (i.e., from "next token prediction" to "next block prediction" ). The author(s) may provide some clarifications and insights in the rebuttal.

$\quad$
- The semi-AR (i.e., next-block-prediction) paradigm proposed in this paper does not seem to be restricted to video generation.  This means when an image tokenizer is used, it can also be applied to image generation. As a general semi-AR paradigm, quantitative comparisons on the  ImageNet dataset are suggested for drawing more convincing conclusions in this paper.

$\quad$
- The temporal axis is not considered in the block division for all design choices (e.g., 1x1x16, 1x4x16, and1x16x16) presented in this paper. 
  - Is this because the video tokenizer (currently MAGVITv2 is used in the paper with a 4 x temporal downsampling) ? 
  - How does dividing blocks along the temporal axis influence the results of video generation?  Further ablation studies are suggested.


$\quad$

[1] Stern, Mitchell, Noam Shazeer, and Jakob Uszkoreit. "Blockwise parallel decoding for deep autoregressive models." *Advances in Neural Information Processing Systems* 31 (2018).

[2] Leviathan, Yaniv, Matan Kalman, and Yossi Matias. "Fast inference from transformers via speculative decoding." *International Conference on Machine Learning*. PMLR, 2023.

[3] Li, Jiacheng, et al. "Lformer: Text-to-Image Generation with L-shape Block Parallel Decoding." *arXiv preprint arXiv:2303.03800* (2023).

[4] Tian, Keyu, et al. "Visual autoregressive modeling: Scalable image generation via next-scale prediction." *arXiv preprint arXiv:2404.02905* (2024).

### Questions
- In Figure 4,  when the first frame is added as the initial condition, the attention map should have extra columns right next to the text column. Should Figure 4 be adjusted, or is my understanding mistaken ?
- Is there any performance improvement in terms of the temporal consistency of generated videos when using NBP over NTP ?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces Next Block Prediction (NBP) for video generation, extending Next Token Prediction (NTP) to predict multiple tokens (a "block") at once. By shifting the prediction unit from a single token to a block, NBP achieves an 11x speedup and better performance than NTP during inference.

### Strengths
- The method is simple yet delivers better speed and performance than NTP.
- The scalability of the NBP framework is well-demonstrated in the paper.
- The writing is clear and easy to follow.
- The analysis on block size is thorough and well-explained.

### Weaknesses
- Section 3.1 (Video Tokenization) cannot be considered an original contribution, as the authors straightforwardly used MAGVITv2[1]. Labeling Section 3.1 as a preliminary section is recommended.
- Although the authors differentiate NBP from MAR[2] in Section 2, there is no supporting evidence that NBP offers denser supervised signals or greater training efficiency. To strengthen the paper’s contribution, it would help to include a comparison showing NBP’s advantage over MAR’s next set-of-tokens prediction by excluding the mask tokens.


[1] Yu, et al. "Language Model Beats Diffusion--Tokenizer is Key to Visual Generation." arXiv 2023.\
[2] Li, et al. "Autoregressive Image Generation without Vector Quantization." arXiv 2024.

### Questions
- The paper states that the model was trained on 17-frame videos, but the TATS score refers to a model trained on 16 frames. Could the authors clarify the process for measuring FVD with NBP models? Specifically, how is the first frame provided to the NBP model in UCI and K600 experiments, and is this frame included when measuring FVD?

- Should the blocks follow a raster scan order? While the authors state that the AR model’s unidirectional raster-scan pattern limits performance, NBP still uses this order in block-level. If it is not, extending the block size analysis in Section 4.5 to examine different block shapes could be beneficial. For example, each block could be constructed from non-nearby tokens within a clip or even from tokens across multiple clips.

### Suggestions
This paper could be more impactful by focusing on NBP's advantages over MAR [1] and providing a more comprehensive analysis of block design.

[1] Li, et al. "Autoregressive Image Generation without Vector Quantization." arXiv 2024.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes next-block prediction framework as a semi-autoregressive method, enhancing the spatiotemporal integrity and parallel prediction for video generation tasks. Several modifications including initial condition, block-wise attention and inference process are applied to existing AR models, and massive experiments are conducted to find the optimal configuration of the block size. The proposed model reaches leading performance compared to previous SOTAs with a good scaling-up law.

### Strengths
- The proposed block-wise semi-AR method is novel and illustrated clearly.

- Rich comparisons and ablations with visualizations are presented and analysed.

### Weaknesses
- [Major] Line 370-372 mentions that the proposed method is first-frame conditioned, which is significantly different from other methods' settings (class-conditioned generation) in Table 3. This indicates completely unfair comparisons.

- The ablations on block size are not fine-grained enough given that the optimal point is 16. Additional values in [1, 16] and [16, 64] should be also investigated. Besides, what is the best block size for temporal axis? 1 is used for all experiments without discussion.

### Questions
- Is the tokenizer completely identical to MAGVIT-v2 or is there any modifications? Its performance is tested separately in Table 1, but the paper describes its architecture as the same as MAGVIT-v2. Also, in Table 1 the reconstruction performance falls behind vanilla MAGVIT-v2 with comparable parameter size.

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a semi-autoregressive (semi-AR) framework, Next-Block Prediction (NBP), for video generation tasks. Compared to the conventional autoregressive (AR) framework, or Next-Token Prediction (NTP), the proposed framework generates blocks with multiple tokens, while these blocks follow a raster-scan ordering. Thus, NBP reduces the number of forward steps required for sampling videos. Experimental results demonstrate that NBP can achieve low FVD scores on UCF-101 and K600 datasets.

### Strengths
S1. This paper aims to resolve an important issue in video generation, sampling efficiency.

S2. This study shows that a semi-AR framework, which is unexplored in video generation tasks, can also be used for video generation.

### Weaknesses
W1. Limited novelty and originality. Contrary to the claims in Section 2, it is widely known that the conventional semi-AR semi-AR framework predicts multiple tokens without additional modules. For example, the SAT model [NewRef-1], which is well-known and presented at EMNLP’18, also shares the same framework as the proposed approach. Thus, I believe the contribution of this paper does not lie in the framework design itself, but lie in applying existing semi-AR frameworks from NLP domains to video generation.

W2. Lack of in-depth analysis on the proposed block predictions. The ablation study does not explore various block shapes (e.g., 1x4x4). Especially, despite the video generation framework, there is no experiment involving the prediction of multiple tokens across different frames.

W3. Given that the proposed NBP conducts row-by-row generation, the framework should be validated on image generation tasks first. Note that the proposed transformer lacks a tailored design for video data.


[NewRef-1] Wang et al., Semi-Autoregressive Neural Machine Translation, EMNLP2018.

### Questions
Q1. Could the authors provide a more detailed explanation and comparison regarding sampling costs? Was KV-caching used in this comparison? Given that the FLOPs for both NBP and NTP are likely similar for sampling, I believe the inference speed should be comparable when using KV-caching as the model scales, even though NBP requires fewer forward steps than NTP.

Q2. Given the same block size, how does performance vary according to block shape? For instance, the ablation study could include comparisons like (1x1x16 vs. 1x4x4 vs. 16x1x1) or (1x16x16 vs. 16x4x4 vs. 4x8x8). Since the authors claim that NTP cannot account for spatial dependencies in local tokens, I initially expected the study to use 2D or 3D shapes for local blocks. However, it employs a 1D block shape, which has fewer spatial dependencies than 2D or 3D blocks.

Q3. In Table 3, how were PSNR, SSIM, and LPIPS computed for the generation results when no ground-truth data exists for video generation?

Q4. In Figure 5, why do the validation loss curves exhibit noisy patterns? I suspect these might be training losses rather than validation losses, considering the curve shapes in Figure 8. Additionally, given the large number of trainable parameters and epochs relative to the small dataset (such as UCF-101), I wonder whether the model shows signs of overfitting.

Q5. Since the experiments focus primarily on class-conditional generation for UCF-101 and frame prediction for K600, could the authors clarify how the text tokens are utilized?

Minor comments (not affecting the score):
- Eq. (1) may contain an error. $x_{l}^{(<t)}​$ should likely be $x^{(<t)}$.
- Since FVD is an incomplete metric for video generation, I recommend including additional metrics such as IS, Dover-Scores, Frame-wise Text Alignments, etc.
- Contrary to the statement in Lines 355-356, Flash Attention does not support customized attention masks.

### Soundness
2

### Presentation
3

### Contribution
2
