# Training-Free Adaptive Frame Selection for Video-Language Understanding

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Multimodal Large Language Models (MLLMs) have shown strong performance on image understanding tasks, but video comprehension remains a significant challenge due to the high computational cost of processing long frame sequences and the limited token capacity of underlying Large Language Models (LLMs). Prior approaches to address this often rely on uniform frame sampling, query-agnostic pruning, or require costly training of dedicated compression modules. In this work, we introduce CoSeLECT, a training-free, plug-and-play, query-guided frame selection method that intelligently subsamples video frames for efficient use in MLLMs. CoSeLECT leverages two key signals: temporal redundancy, which identifies similar frame clusters, and query relevance, which selects frames based on their semantic alignment with the input query. By combining these signals through an adaptive frame selection strategy, CoSeLECT selects frames that are both diverse and highly relevant to the query, without requiring any model-specific tuning. Our results on various base MLLMs show that CoSeLECT consistently outperforms state-of-the-art methods, including trained methods like LongVU by +3.8\% on MVBench and +0.8\% on VideoMME.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes CoSeLECT, a training-free adaptive frame selection method that efficiently selects the most informative frames from a large pool by combining query relevance and temporal continuity. This approach achieves better performance than existing training-free methods on multiple video understanding benchmarks.

### Strengths
The paper presents a clear and practical training-free frame selection method that combines query relevance and visual continuity in a straightforward manner. While the individual components are not novel, their combination into an adaptive, query-aware selection pipeline is sensibly designed and effectively executed. The method is well-described, easy to reproduce, and evaluated across multiple benchmarks with ablations that support key design choices. Its significance lies in offering a lightweight, plug-and-play solution that improves efficiency and performance for video understanding with MLLMs without requiring model retraining. This is useful for real-world deployment, though not theoretically groundbreaking.

### Weaknesses
- The novelty of the paper is limited. Query-aware frame selection for Videl-LLM is not innovative, as discussed in KeyVideoLLM [1], AKS [2], and Q-Frame [3]. The paper pointed out that `these heavier methods are typically limited to sparsely pre-sampled frame pools` is interesting, but the experiment did not support the solution of this problem.

- It is not clear whether the comparison of the experimental results in the paper with other training-based methods is fair.

- The paper claims the proposed CoSeLECT is lightweight. However, there is a lack of systematic evaluation of latency and computing consumption, which is crucial for actual deployment.
> but in its lightweight, principled fusion of two readily available ones—frame–text similarity for semantic relevance and inter-frame similarity for temporal continuity.

- Lack of discussion of limitations.

- Minor weaknesses
  - The first equation in Section 3.4 is missing a number
  - $\sqrt{D_i}$ in equation (4) lacks definition

[1] Liang H, Li J, Bai T, et al. Keyvideollm: Towards large-scale video keyframe selection[J]. arXiv preprint arXiv:2407.03104, 2024.
[2] Tang X, Qiu J, Xie L, et al. Adaptive keyframe sampling for long video understanding[C]//Proceedings of the Computer Vision and Pattern Recognition Conference. 2025: 29118-29128.
[3] Zhang S, Yang J, Yin J, et al. Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs[J]. arXiv preprint arXiv:2506.22139, 2025.

### Questions
- I am confused about the avoidance of redundant calculations mentioned in the article in line 204. Although  LLaVA-OneVision uses SigLIP-So400M-patch14-384 as the visual encoder, it fine-tunes SigLIP during the training process, which results in them having the same structure but different parameters. So is it really possible to avoid redundant calculations? 
> Since SigLIP is also used as the vision encoder in LLaVA-OneVision (Li et al., 2024a), these embeddings can be directly reused, avoiding redundant computation.

- Does pre-$\textbf{E}_{im}$  in section 4 mean $N$ in section 3, and post-$\textbf{E}_{im}$  means  $K$? If so, please use consistent expressions to improve reading; if not, I hope the author can further elaborate on the difference.

- Lack of in-depth analysis of Table 1. With the increase of pre-$\textbf{E}_{im}$, there is no consistent performance improvement across all benchmarks. Does this contradict the motivation of the paper?
> Crucially, these heavier methods are typically limited to sparsely pre-sampled frame pools in order to remain computationally feasible—risking the permanent loss of “needle-in-a-haystack” moments before the selection algorithm can even evaluate them, a limitation that becomes particularly acute under resource constraints.

- I am confused about the experimental results in Table 2. Is this comparison meaningful?
  - Frame-Voyager is similar to the proposed CoSeLECT, and is also a plug-and-play model. But its LLM Size does not seem to be 7B. 
  - LongVA and VideoChat2 are Video-LLMs. How to compare them with CoSeLECT ? 

- Supplement CoSeLECT compares the results of the Qwen2-VL [1] experiment with AKS and Q-Frame [2], which will provide a more comprehensive evaluation. 

[1] Wang P, Bai S, Tan S, et al. Qwen2-vl: Enhancing vision-language model's perception of the world at any resolution[J]. arXiv preprint arXiv:2409.12191, 2024.

[2] Zhang S, Yang J, Yin J, et al. Q-Frame: Query-aware Frame Selection and Multi-Resolution Adaptation for Video-LLMs[J]. arXiv preprint arXiv:2506.22139, 2025.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a training-free query-guided frame selection method for efficient video processing in MLLMs. It uses SigLIP cosine similarity between each frame and the given query to measure query relevance. Then it identifies scene transitions based on visual similarity. Based on these, the method adaptively allocate tokens to those scenes that is more relevant to the queries via relevance reweighting. CoSeLECT evaluates on six video understanding benchmarks and achieves state-of-the-art performance compared with both training-free and fine-tuned methods.

### Strengths
+ The paper writing is clear and easy to follow.
+ The method is training-free and can be applied to any LVLMs.
+ The method improves the performance on base model LLaVA-OV and Qwen2.5-VL-7B. It also outperforms other frame selection methods.

### Weaknesses
+ The paper should compare with more video token compression or frame selection method. For example, BOLT [1] is a frame selection method.

+ When retrained ratio goes down to 12.5%, CoSeLECT has several benchmarks lower than FastVID.

+ Although comprehensive evaluation has been down, the paper is mainly based on empirical observation and has very limited innovation.

+ The comparison does not seem entirely fair. Although 8k video tokens are finally fed into the MLLM, it still need to process additional frames during intermediate steps. Given the method involves intermediate steps and introduces computational overhead, the comparison should be made against the base model’s best performance. For example, Qwen2.5-VL got 65.1 on VideoMME.

[1] BOLT: Boost Large Vision-Language Model Without Training for Long-form Video Understanding

### Questions
+ When comparing with baseline models LLaVA-OV  and Qwen2.5-VL-7B, how many frames and token per frame is used within 8k context length?
+ Have you tried LLM’s text embedding instead of SigLIP text embedding? 
+ Some complicated question could not be used to select key frames based on embedding similarity. For example, many questions in VideoMME are like ‘which of the statement is correct?’ Therefore, query-frame embedding similarity is not a fine-grained way for frame selection.
+ What is the computation overhead introduced and inference speed compared with base model, since the method needs to calculate similarity between consecutive frame embeddings.

### Soundness
3

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
3

### Summary
This work introduces another method to select which frames to use for video-language understanding. This is an important topic since MLLMs often have input token limitations, and having some way to prefilter the data to find the most relevant answer can often help in increasing the results. The method is training-free and can be plugged with different models. Their main competitor is AKS (Adaptive Keyframe Sampling) that also adopts a training-free paradigm. However, there are some differences with the method proposed here. Whereas AKS splits the video into equal halves, CoSeLECT segments the video into different lengths depending on intra-clip similarity. The method introduced by the authors seems a bit more flexible than AKS while providing slightly better results. The authors evaluate their method across several common benchmarks such as NextQA, MLVU, VideoMME, MVbench, and LongVideoBench. They show that their method is competitive with training-based methods such as LongVu. They also compare their method with different token reduction techniques such as uniform sampling, VisionZip, PruneVid, and others, for which they also have competitive results.

### Strengths
- A simple and yet effective training-free method for frame selection
- Adaptative and not relying on specific video sub-clip size
- Paper well-written
- Extensive comparison with similar methods such as AKS
- Extension comparison with both training-based and training-free frame selection method
- Good ablation over the different hyper-parameters such as similarity threshold or frame pool size

### Weaknesses
- Some overhead introduced by the method since frames need to be processed through a SigLip encoder to compute frame similarity. Depending on the number of frames being processed, this can have an important impact even if this operation can be parallelised. 
- Lack of ablation over the vision and text encoder.
- The paper title and abstract does not exactly match the ones in OpenReview (don't know how much that can be an issue or not)

### Questions
Why choosing SigLIP-ViT and not another model? Did you perform an ablation on those?

### Soundness
3

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
5

### Summary
This paper proposes a training-free method that can be seamlessly integrated into existing multimodal large language models (MLLMs). The approach jointly considers both frame-level visual diversity and overall video length, leading to more balanced and informative video representations. The method achieves state-of-the-art performance across multiple video understanding benchmarks, demonstrating both simplicity and effectiveness.

### Strengths
1. The proposed method is simple but effective, requiring no additional training while significantly improving performance.

2. The design is model-agnostic and can be easily plugged into various MLLM architectures, indicating strong generality and practical utility.

3. Experimental results are convincing and comprehensive, covering multiple datasets and metrics, with clear visualizations that illustrate the method’s contribution.

4. The paper is well-written and easy to follow, making the technical insights accessible.

### Weaknesses
1. The main concern lies in the limited novelty of the method. The use of text–visual embedding similarity as a selection strategy is not conceptually new and has been widely seen in prior works as an auxiliary component or ablation. While the empirical results are strong, the contribution is mainly engineering-level, lacking deeper methodological insight or theoretical advancement. 

2. In addition, the paper does not clearly explain how the method mitigates temporal information loss when modeling long videos.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
