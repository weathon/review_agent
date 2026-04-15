# LongVU: Spatiotemporal Adaptive Compression for Long Video-Language Understanding

- Decision: Reject
- Scores: 5, 6, 6, 6, 6

## Abstract
Multimodal Large Language Models (MLLMs) have shown promising progress in understanding and analyzing video content. However, processing long videos remains a significant challenge constrained by the limited context length. To address this limitation, we propose LongVU, a spatiotemporal adaptive compression mechanism to reduce the number of video tokens while preserving visual details of long videos. Our idea is based on leveraging cross-modal query and inter-frame dependencies to adaptively reduce temporal and spatial redundancy in videos. Specifically, we leverage DINOv2 features to remove redundant frames that exhibit high similarity. Then we utilize text-guided cross-modal query for selective frame feature reduction. Further, we perform spatial token reduction across frames based on their temporal dependencies. Our adaptive compression strategy effectively processes a large number of frames with little visual information loss within limited context length. Our LongVU consistently surpass existing methods across a variety of video understanding benchmarks, especially on hour-long video understanding tasks such as VideoMME and MLVU. Given a light-weight LLM, our LongVU also scales effectively into a smaller size with state-of-the-art video understanding performance. Our code will be made publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper proposes a video compression strategy, including: removing redundant frames, selective frame feature reduction, and spatial token reduction. This strategy effectively reduces the number of visual tokens in MLLM and significantly improves MLLM performance on long videos.

### Strengths
- The paper is generally well-written and easy to understand.
- The performance is encouraging.

### Weaknesses
- There are many details in the pipeline that are not explained clearly.

1. Line 52: The statement "the resampling module causes important visual information loss" needs further explanation. The resampling module is widely used in MLLMs, such as in the latest **qwen2-vl** and **MiniCPMV-2.6**.

2. Equation 1 lacks a detailed explanation, and there are errors in the symbol definitions. For example, in line 242, why is it not $L_{max}$?

3. The setup in Section 3.1 lacks necessary details, such as why $ J $ is set to 8.  

4. In line 215, the statement "about half of the video frames were reduced" depends on the hyperparameter settings, but the paper does not explain how the hyperparameters were chosen.

5. Figure 4(a) shows the comparison of the number of frames before and after compression. However, this result depends on the choice of threshold, which is not explained in the paper.

6. In Table 4, the **STC** strategy is not always effective. What is the reason for this?

7. Line 207: The authors claim, "we pioneer to leverage both SSL-based model DINOv2 with vision-language contrastive-based model SigLIP as frame feature extractors for MLLM in video understanding tasks." In fact, several similar works already exist [1-2].

[1] Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs

[2] Cambrian-1: A Fully Open, Vision-Centric Exploration of Multimodal LLMs

- Compared to previous MLLMs, the three proposed modules in the paper require additional computational processes, such as inter-frame similarity calculation and frame-question similarity calculation. These dense computations will undoubtedly significantly increase inference time. However, the paper lacks a discussion on inference time.

- The proposed three compression strategies seem to be transferable to existing MLLMs in a training-free manner.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper focuses on the challenge of limited context length in understanding long videos. The authors propose LongVU, which reduces the number of visual tokens across temporal and spatial dimensions. This approach enables the model to handle a large number of frames effectively, achieving strong performance on benchmarks such as VideoMME and MLVU.

### Strengths
1. The motivation for reducing the number of visual tokens is clear.

2. The evaluation is thorough, with ablation studies clearly demonstrating the effects of the proposed approaches.

3. The paper is well-written and easy to follow.

### Weaknesses
Table 7 indicates that the model performs less effectively on short videos; it would be beneficial to include some analysis of this phenomenon.

This paper is quite comprehensive, and in my view, there are no other significant weaknesses.

### Questions
Does the proposed compression approach hurt performance on short videos?

Table 3 lacks results on MVbench, a relatively shorter benchmark. Additionally, Table 7 indicates that the model performs less effectively on short videos. Are there specific limitations causing the proposed method to underperform on short videos? If so, it would be helpful to highlight these limitations.

### Soundness
3

### Presentation
4

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
This paper introduces LongVU, a novel approach designed to achieve effective long video understanding within the limited context of large language models (LLMs) by compressing the spatio-temporal tokens of videos. The authors provide a detailed discussion on effectively removing redundant temporal frames and spatial tokens. LongVU demonstrates state-of-the-art performance across multiple video understanding benchmarks. The extensive ablation studies convincingly illustrate LongVU's effectiveness in understanding long videos.

### Strengths
1. This paper introduces LongVU, which employs two well-designed strategies (Section 3.1, Section 3.3) to reduce redundant temporal frames and spatial tokens, while also proposing the selective retention of high-resolution frames based on queries (Section 3.2). Compared to earlier methods based on frame and token compression [1,2], LongVU demonstrates significant advancements in long video understanding tasks.

2. The authors have conducted comprehensive evaluations, including on the latest long video understanding benchmarks such as VideoMME and MLVU. LongVU achieves state-of-the-art performance in long video understanding tasks compared to recent advanced video MLLMs.

3. The spatial-temporal compression mechanism proposed by the authors has been proven effective for long video tasks, as demonstrated by the ablation study on MLVU in Table 4 and the Needle-in-a-Haystack task experiment in Figure 5.

[1] Song, Enxin, et al. "Moviechat: From dense token to sparse memory for long video understanding." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

[2] Li, Yanwei, Chengyao Wang, and Jiaya Jia. "Llama-vid: An image is worth 2 tokens in large language models." European Conference on Computer Vision. Springer, Cham, 2025.

### Weaknesses
1. There are some confusing aspects in the methodology section that need clarification:
   - (1) In the Temporal Reduction section (Section 3.1), the authors propose sampling \( T \) frames from \( N \) frames based on DINO similarity. According to the description, \( T \) seems to vary depending on the video. Is there a specific DINO threshold used here? If so, what is the threshold?
   - (2) The proposed method lacks a mechanism to ensure that the tokens fed into the LLM backbone remain within the context length. The authors do not explain how tokens exceeding the LLM context are handled. As shown in Figure 4(b), after token compression (Section 3.2), many video tokens exceed 8K, with some reaching over 15K. Given that the default context length setting is 8K, how are the excess tokens managed?
   - (3) Is the cross-attention module in Section 3.2 trained end-to-end during the video SFT stage?

2. The LongVU 7B model uses Qwen2-7B as the LLM backbone, which has a context length of 32K. Table 3 indicates that their 16K baseline (line 443; Uniform-16K) has a clear advantage over their 8K baseline (line 447; Uniform-8K). Why did the authors ultimately choose 8K for the model setting? Would combining DINO+Query+STC with a 16K or even 32K context length yield better results on hour-level long video tasks? Alternatively, are the authors emphasizing the superiority of DINO+Query+STC under limited context and computational constraints? If so, they should compare the additional computational overhead introduced by DINO+Query+STC.

3. There are some typos in the paper. For example, on line 46, "per framE" has an uppercase "E." In Table 1, some data is misaligned; for instance, the MLVU result for GPT-4o is 64.6, not 66.2.

### Questions
1. The authors need to clarify certain ambiguous aspects of their method; please refer to the Weaknesses section for details. 

2. The authors should analyze the time and computational efficiency of the proposed compression method. For instance, what proportion of inference time is consumed by calculating DINO similarity between frames and spatial token similarity? Does this lead to significant time overhead during inference?

### Soundness
4

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
This paper aims to address the challenge of token compression in long video understanding by proposing different token sampling strategies in the spatiotemporal dimension: 1) reducing temporal redundancy by leveraging inter-frame feature similarity (DINOv2 and SigLIP); 2) reducing feature redundancy by utilizing cross-modal similarity; and 3) reducing redundant tokens through element-wise cosine similarity within spatial windows. 
Through these effective and intuitive strategies, this research is capable of processing 1-hour-long videos within a common 8k context length. Additionally, this paper demonstrates superior numerical performance compared to other long video understanding methods such as VideoChat2 and LLaVA-OneVision.

### Strengths
1. This paper is competitive in long video understanding performance; for instance, its performance on VideoMME long videos surpasses that of LLaVA-OneVision by 12.8 points.
2. In smaller model sizes (LLaMA3.2-3B), LongVU also outperforms existing methods.
3. This method essentially proposes a token compression strategy that helps reduce token redundancy in long videos. It is a general approach that can be applied across different multi-modal large language models (MLLMs).

### Weaknesses
1. This paper is engineering-oriented and lacks some innovation. For instance, the combination of SigLIP and DINOv2 is based on the approach used in Cambrian. Additionally, the ability of DINOv2 features to more intuitively reflect low-level visual characteristics is evident, and this has also been compared with CLIP-based methods in the original paper.
2. In the comparative experiments, I believe that including the inference time for processing videos of the same length would contribute to a fairer comparison. This is because the token compression strategy employed in this paper inevitably introduces some preprocessing steps, which in turn adds more inference time.

### Questions
1. Is the sliding window size \( K = 8 \) an empirical value in spatial token compression (STC)?
2. Are there any analyses of failed examples?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents LongVU, a spatiotemporal adaptive compression mechanism designed to handle long video sequences within the limited context length constraints of current multimodal large language models (MLLMs). The authors propose leveraging cross-modal query and inter-frame dependencies to reduce temporal and spatial redundancy in videos effectively. The method demonstrates promising results across various video understanding benchmarks, particularly for hour-long videos.

### Strengths
1. The concept of spatiotemporal adaptive compression to address the challenge of long video understanding is timely, given the increasing interest in multimodal learning and the limitations of current MLLMs.
2. The paper provides extensive experimental validation, showing significant improvements over existing methods on multiple video understanding benchmarks, which is a strong empirical demonstration of the proposed method's effectiveness.
3. The use of DINOv2 features for temporal reduction and the cross-modal query mechanism for selective frame feature reduction are well-motivated and technically sound.

### Weaknesses
1. While the paper discusses the effectiveness of LongVU, a deeper analysis into the computational complexity and scalability, especially regarding the trade-offs with comprasion ratio and context length, could be beneficial.
2. The paper focuses on video-language understanding. It would be useful to understand how well these techniques generalize to other modalities (e.g. image).
3. More ablation studies on the components of the spatiotemporal compression strategy could provide further insights into the contribution of each component to the overall performance.
4. The query-based comparison seems limited to single-turn dialogue scenarios. The authors should consider how LongVU would handle multi-turn interactions, where the context and relevance of frames may change dynamically across turns. Addressing this could significantly expand the applicability of LongVU.
5. The paper missed some relevant works, such as "LGDN: Language-guided denoising network for video-language modeling" (NeurIPS 2022).

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
