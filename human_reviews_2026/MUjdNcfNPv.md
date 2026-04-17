# VideoChat-Flash: Hierarchical Compression for Long-Context Video Modeling

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 4

## Abstract
Long-context video modeling is critical for multimodal large language models
(MLLMs), enabling them to process movies, online video streams, and so on.
Despite its advances, handling long videos remains challenging due to the difficulty
in efficiently understanding the extremely long video context. This paper aims to
address this issue from aspects of the model architecture, training data, training
strategy, and evaluation benchmark. First, we propose a novel Hierarchical video
token Compression (HiCo) method, which leverages visual redundancy in long
videos to compress long video context from Clip-level to Video-level, reducing the
computation significantly while preserving essential details, achieving an extreme
compression ratio of approximately 1/50 with almost no performance loss. Second,
we introduce a multi-stage short-to-long learning scheme, a large-scale dataset of
real-world long videos named LongVid, and a challenging “Multi-Hop Needle-In-
A-Video-Haystack” benchmark. Finally, we build a powerful video MLLM named
VideoChat-Flash, which shows a leading performance on both mainstream long
and short video benchmarks at the 2B and 7B model scales. It first gets 99.1%
accuracy over 10,000 frames in NIAH among open-source models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces VideoChat-Flash, a new model for long-context video modeling. It is trained on LongVid, a novel dataset combining Ego4D, HowTo100M, HD-Vila, and MiraData, with additional dense and timestamped event labels. The core of their method is hiCo (Hierarchical Compression), a multi-step technique to reduce both clip-level and video-level redundancy. It involves sampling clips based on video duration, using a vision encoder with spatio-temporal attention to compress visual and temporal information within clips, and then merging these compressed features into a final vision context aligned with LLM representations via a projection head. The training process is multi-staged: video-language alignment, pre-training with short videos, short and long video instruction tuning, and finally, post-finetuning for higher-resolution video encoding. The authors also present a new "Needle in a Haystack" benchmark. Unlike traditional benchmarks that insert a single needle, their approach leverages model reasoning by embedding multiple clues in a video that the model must follow to find the final objective. VideoChat-Flash significantly outperforms models like GPT4, Gemini 1.5, and other open-source models on various benchmarks, including LongVideoBench, MLVU, VideoMME, and LVBench, demonstrating strong performance across long video contexts. An ablation study was conducted to analyze different compression ratios and architectural and training criterion designs.

### Strengths
- Paper well-written
- Lot of artefacts such as new model, new training data mix, and new evaluation protocole
- Really like the idea of a reasoning based needle in a video haystack evaluation
- Extensive experimental protocole and comparison against lot of different methods
- Performance significantly higher than the concurrent methods
- In depth ablation studies over the compression effect, sampling strategies, video encoder and design choices
- Good appendix with details on hyper-parameters that were used, additional ablations studies and additional training data mix information.

### Weaknesses
- Would have expected some discussions over the overall training/inference cost of this method in contrast to other.
- Missing some discussion on the training-free frame selection method such as AKS (Adaptive Keyframe Sampling for Long Video Understanding)
- Missing information about the different video data licenses. A datasheet would have been appreciated.

### Questions
- Do the authors plan to open-source models and benchmarks?
- For the Multi-Hop Needle-In-A-Video-Haystack eval, are the clues always going forward in the video or can they also be backward? Let's say you have 3 hops, one at 5s, another at 8s, and the last one at 3s before having all the clues to get to the final haystack.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This work aims to efficiently understand the extremely long video context with multimodal large language models (MLLMs). They propose a novel Hierarchical video token Compression (HiCo) method, which leverages visual redundancy in long videos to compress long video context from Clip-level to Video-level, reducing the computation significantly while preserving essential details. Then, they introduce a multi-stage short-to-long learning scheme and also a large-scale video dataset called LongVid as well as a benchmark. They build VideoChat-Flash that shows a leading performance on both mainstream long and short video benchmarks at the 2B and 7B model scales.

### Strengths
1. The topic is meaningful to the community: we need to find a better way to understand long videos with MLLMs efficiently. 
2. The paper is well-organized and easy to follow. The experiments are solid and clear. 
3. I really appreciate the authors for the LongVid dataset and benchmark. Open-sourced data is important in this community.

### Weaknesses
1. The paper claims “almost no performance loss” under extreme 1/50 compression, yet the quantitative analysis (Fig. 6b) provides limited breakdown across different downstream tasks. It would be helpful if they can provide the results on temporal grounding or motion-sensitive tasks besides the three benchmarks. 
2. The claimed efficiency benefits (1/50 token reduction) are shown in FLOP counts but not in end-to-end wall-clock latency or GPU memory consumption during real long video inference. It would be helpful to report them in the paper as well! 
3. Although the LongVid dataset is described as large-scale and diverse, construction heavily relies on caption-based pseudo-labeling using other LLMs. There is limited discussion of data noise, filtering quality, or potential overlap with evaluation benchmarks.

### Questions
N/A. 
One tiny suggestion: fig -> Fig, tab -> Tab

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
4

### Summary
The paper proposes a hierarchical video token compression method designed to efficiently handle long-video contexts without compromising performance. It begins with duration-based sampling, which adapts to the differing requirements of short and long videos. This is followed by clip-level compression to aggregate key information across frames, and finally, a progressive visual dropout mechanism that allocates attention differently across shallow and deep layers. The authors also introduce a training dataset aimed at enhancing video understanding capabilities along with a multi-stage training strategy. In addition, a new challenging evaluation benchmark is presented to better assess complex reasoning abilities. Finally, the model is evaluated on multiple benchmarks, demonstrating the effectiveness of the proposed method, with ablation studies further validating the contribution of each component.

### Strengths
This is a well-structured and thoroughly executed paper. It presents extensive and insightful experiments and analyses that may also benefit future work in video understanding. The proposed HiCo framework demonstrates strong and consistent performance across multiple benchmarks. Additionally, the authors provide a new training dataset and evaluation benchmark, which will further benefit the community and advance progress in video understanding. The comprehensive ablation studies clearly illustrate the contribution of each component.

### Weaknesses
1. Some recently released baseline video models [1][2][3][4] are not included in the comparison. It would improve readability if the authors could explicitly indicate which model represents the previous state of the art in Figure 1.
2. It would be beneficial to include more advanced video models as baselines for comparison on the multi-hop NIAH task. Presenting the performance of closed-source models, such as Gemini 2.5 Pro, could also provide valuable context for readers to comprehend the complexity of this task. 
3. It would be helpful to provide more details about the text-guided selection mentioned in L214.

[1] Comanici, G., Bieber, E., Schaekermann, M., Pasupat, I., Sachdeva, N., Dhillon, I., Blistein, M., Ram, O., Zhang, D., Rosen, E. and Marris, L., 2025. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261.

[2] Zhang, B., Li, K., Cheng, Z., Hu, Z., Yuan, Y., Chen, G., Leng, S., Jiang, Y., Zhang, H., Li, X. and Jin, P., 2025. Videollama 3: Frontier multimodal foundation models for image and video understanding. arXiv preprint arXiv:2501.13106.

[3] Wang, X., Si, Q., Wu, J., Zhu, S., Cao, L. and Nie, L., 2025. Adaretake: Adaptive redundancy reduction to perceive longer for video-language understanding. arXiv preprint arXiv:2503.12559.

[4] Xu, M., Gao, M., Li, S., Lu, J., Gan, Z., Lai, Z., Cao, M., Kang, K., Yang, Y. and Dehghan, A., 2025. Slowfast-llava-1.5: A family of token-efficient video large language models for long-form video understanding. arXiv preprint arXiv:2503.18943.

### Questions
1. Regarding the creation of the Multi-Hop NIAH evaluation task, is there any restriction on the maximum number of frames that can be inserted?
2. For the image encoder, what is the rationale for using SigLIP instead of SigLIP2, which is generally considered a stronger image encoder? Similarly, for the video encoder, why not choose VideoPrism?
3. Why does the accuracy drop when increasing $T_{min}$ from 64 to 128 in Figure 7 (a)? Intuitively, performance should improve as more frames are used to capture finer details, as suggested by the trend in Figure 7 (b).

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents VideoChat-Flash, a Multimodal Large Language Model (MLLM) designed to efficiently handle extremely long video contexts. The authors identify high computational cost and information redundancy as the primary bottlenecks in current long-video models. To address this, the paper introduces a comprehensive, four-part contribution: 1.A novel Hierarchical video token Compression method. This operates at two levels: (1) a Clip-level compression during encoding that uses a spatio-temporal video encoder to exploit local redundancy, and (2) a Video-level "progressive visual dropout" during LLM inference that prunes tokens based on attention patterns . 2.A new, large-scale dataset for long-video instruction tuning. 3.A multi-stage short-to-long learning scheme. 4.A new, more challenging "Needle in a Haystack" benchmark, the "Multi-Hop Needle-In-A-Video-Haystack".

### Strengths
1.The core strength is the HiCo compression framework. Achieving a 1/50 compression ratio (16 tokens/frame) while simultaneously achieving SOTA performance is a breakthrough for practical long-video MLLMs .The design is well-motivated.
2. The introduction of the Multi-Hop NIAH benchmark is a significant contribution.
3. The contributions of the paper are multifold including model design, data creation, and new evaluation.

### Weaknesses
1. The biggest concern is the novelty of the proposed method. While the system as a whole is novel and highly effective, its constituent parts are largely clever integrations of existing ideas.
2. The ablation in Table 2, while extensive, presents a slightly confusing narrative for HiCo. The baseline (196 tokens/frame) achieves a 63.7 on MLVU. The "+ HiCo" model (16 tokens/frame) drops to 60.6 on MLVU. This seems to contradict the "almost no performance loss" claim from the abstract. The performance is only recovered after applying the new training strategies and data. This suggests HiCo does incur a performance cost, which is then compensated for by the data and training recipe.

### Questions
1. Following on from my point in "Weaknesses," could you please elaborate on the performance drop when only HiCo is added (Table 2, row "+ HiCo")? How much of the final SOTA performance should be attributed to the sheer efficiency of HiCo (allowing for processing long contexts) versus the new LongVid dataset and short-to-long training strategy (which seem to compensate for an initial compression-induced quality loss)?
2. The authors mention that the video-level "Progressive Visual Dropout" is used "only during inference" due to "challenges in compatibility with training acceleration". Would there be issues such as training-inference mismatch ? However the results show this method is helpful. Could the author explain a bit why such mismatch does not lead to performance drop ?

### Soundness
2

### Presentation
3

### Contribution
2
