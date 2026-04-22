# PEVLM: Parallel Encoding for Vision-Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Vision-Language Models (VLMs) have demonstrated strong capabilities in multimodal understanding and generation tasks. However, their application to long video understanding remains hindered by the quadratic complexity of standard attention mechanisms. In this work, we introduce \textbf{PEVLM}, a fine-tuning-free parallel encoding method designed to enhance the prefilling efficiency of VLMs in long video scenarios. To the best of our knowledge, this is the first work to adapt parallel encoding to VLMs. PEVLM partitions the input video into context blocks with a shared sink block, while preserving sequential position embeddings to align the attention score distribution with that of Full-Attention. This design reduces the complexity of attention from $O((T \times N)^2)$ to $O(T \times N)$ where $T$ is the number of frames and $N$ the number of tokens per frame, with minimal loss in accuracy.
Extensive experiments across multiple state-of-the-art models and benchmarks demonstrate that PEVLM consistently outperforms existing parallel encoding approaches, achieving up to \textbf{7.47x} speedup in attention computation and reducing end-to-end latency by \textbf{44\%} to \textbf{50\%}. Remarkably, PEVLM not only maintains high accuracy, but in some settings even surpasses Full-Attention performance. Under strict latency constraints, it achieves substantial gains, improving accuracy from \textbf{23.26\%} to \textbf{61.03\%}. These results underscore the effectiveness of PEVLM for low-latency, long-context video understanding, making it a promising solution for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces the parallel encoding technique into the scenario of long video understanding for Vision-Language Models (VLMs). A series of experiments were conducted to analyze the differences between VLMs and Large Language Models (LLMs) in the application of parallel encoding. It is pointed out that addressing the positional encoding of blocks and the attention sink problem is crucial, and a concise and effective implementation is proposed. This implementation significantly improves the efficiency of long video prefilling while achieving almost no performance loss on various long video benchmarks.

### Strengths
1. This work systematically and comprehensively investigates the effect of parallel encoding on long video understanding, and proposes a concise and effective solution.

2. The ablation experiments in this paper are sufficient and comprehensive, providing numerous valuable references and insights for future research on improving the efficiency of long video understanding.

### Weaknesses
1. The method proposed in this paper is highly similar to previous approaches in the LLM field, such as StarAttention [1]. Although this paper incorporates several improvements tailored to the video scenario, including Sequential Position Embedding, Dividing the Video by Frames, and Adding Frames to the Sink, the overall modifications are minor. Moreover, there is no substantial performance gap in most cases, except for models like Qwen-VL that are extremely sensitive to the implementation of positional encoding. This raises concerns about the technical contributions of this paper.

2. There is a citation error in Line 202. Additionally, Figures 3 and 4 appear somewhat blurry.

[1] Star Attention: Efficient LLM Inference over Long Sequences

### Questions
1. Table 3 appears to only present the ablation experiment results of a single model. I speculate that this model might be Qwen2.5-VL-7B-Instruct, given its high sensitivity to positional encoding implementations. It would be more compelling if the results of other models, such as LLaVA-Video and LongVILA, could also be provided.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes PEVLM, a fine-tuning-free parallel encoding method to address the inefficiency of Vision-Language Models (VLMs) in long video understanding due to quadratic attention complexity. By partitioning input videos into context blocks and aligning attention scores with Full-Attention, PEVLM reduces complexity from O((TN)^2) to O(TN) with minimal accuracy loss. Experiments show up to 7.47x speedup, 40% latency reduction, and even improved accuracy in some cases, making PEVLM a promising solution for low-latency, long-video reasoning tasks.

### Strengths
1. The paper is well-written, with the methodology and experimental results presented in a clear and systematic manner.
2. The research topic is highly practical, as introducing a training-free method that effectively reduces inference memory usage and latency provides significant value for real-world model deployment.
3. The experiments in the paper highlight the effectiveness of PEVLM, achieving a 40% reduction in latency while maintaining accuracy on several long-video benchmarks.

### Weaknesses
1. The experimental evaluation is incomplete. While PEVLM is designed for long-video understanding, all the benchmarks focus solely on QA tasks. Token sparsification typically has limited impact on QA tasks; however, for tasks that rely on fine-grained visual details, such as video captioning or video OCR, it could introduce significant drawbacks. The authors should include results on such benchmarks to better demonstrate the method’s versatility across different task types.
2. The paper briefly mentions a 40% latency reduction but lacks detailed analyses of latency and memory usage in the experiments. For instance, the impact of different hyper-parameter configurations on memory, latency, and accuracy across various models is not thoroughly explored. Token count alone does not provide sufficient insight into these metrics, and additional detailed analysis would strengthen the results.

### Questions
I noticed that all the baselines used in the paper are based on 7B models. How would the proposed method perform on larger or smaller models, and what impact would model size have on the approach?

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
This paper introduces PEVLM, a training-free parallel encoding strategy designed to enhance prefilling efficiency in vision-language models (VLMs) for long-video scenarios. The approach is motivated by a thorough analysis of the limitations of existing parallel encoding methods in long-context video understanding. PEVLM aims to maintain model quality while reducing computational overhead during the prefill stage. Empirical results demonstrate that PEVLM can match or even exceed the performance of standard full attention, with notable improvements in latency and computational cost.

### Strengths
1. The manuscript is clearly written and provides a well-articulated diagnosis of the failure modes in current parallel encoding strategies for long-video understanding. The proposed method is strongly motivated by this analysis, which enhances the interpretability and credibility of the design.

2. The experimental results are comprehensive and persuasive, showing that PEVLM achieves comparable or superior accuracy to full attention across multiple VLMs and long-video benchmarks, while also reducing prefill time and computational requirements.

### Weaknesses
1. The proposed method is limited to the prefill stage, which may restrict its applicability in broader long-video understanding scenarios, such as real-time or streaming video processing where continuous updates and causal inference are required.

2. The explanation for PEVLM’s superior performance over full attention in very long contexts—namely, that block-wise softmax mitigates degradation—is largely intuitive. The current analysis does not fully account for model-specific differences observed in the experiments (e.g., the lack of improvement in LLaVA-Video). A more principled and quantitative analysis (e.g., attention entropy, effective context length, token distribution) would help substantiate these claims.

3. The method introduces key hyperparameters (Sink Block size and Context Block size) that significantly affect generalizability. As shown in Section 5.2 and Figure 4, performance is sensitive to these settings, and there is no universal configuration that works across tasks and models. This reliance on task- and model-specific tuning diminishes the practical simplicity and "plug-and-play" nature of the approach, as it necessitates a pre-deployment search for optimal parameters.

4. The related work section is comparatively narrow, focusing primarily on a few baselines (e.g., APE, Star Attention). For completeness, the discussion should be expanded to include recent advances in efficient attention mechanisms and long-context modeling for VLMs. For the camera-ready version, I recommend incorporating a more comprehensive review of related work into the main text.

### Questions
I encourage the authors to address points 1 and 2 in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces PEVLM, a fine-tuning-free parallel encoding framework for accelerating long video understanding in Vision-Language Models (VLMs). The key idea is to partition the video into context blocks while maintaining a shared sink block and sequential position embeddings. The paper provides experiments across multiple representative VLMs (Qwen2.5-VL, LongVILA, LLaVA-Video) and benchmarks (MVBench, EgoSchema, VideoMME, LongVideoBench).

### Strengths
1. Solid implementation and reproducibility: The method is implemented within a production-grade serving framework (SGLang) and evaluated with well-specified setups, which enhances reproducibility.

2. Lightweight and deployment-friendly: PEVLM introduces no extra parameters or fine-tuning, relying solely on structural reorganization of attention computation, making it easily applicable to existing VLM pipelines.

3. Well-written and structured: The paper’s organization and figures make the methodology easy to follow, with clear definitions of each block type and their respective contributions.

### Weaknesses
1. Logic flow needs more clarification: The introduction states, “These application scenarios often demand processing longer video inputs.” While longer video inputs are indeed an important problem to address, the paper later mentions, “Although PEVLM achieves significant acceleration of inference, it is similarly limited to 256-frame inputs. Nonetheless, this trade-off is acceptable as our primary goal is to enhance inference efficiency rather than to expand context length.” This creates a conceptual inconsistency: if the motivation emphasizes longer video processing, it is unclear why the proposed method remains restricted by the same limitation. Moreover, the design choice of “preserving sequential position embeddings instead of reusing position embeddings across blocks” seems more like an implementation decision guided by user needs rather than a clear methodological contribution.

2. The manuscript needs better proofreading: There are minor formatting and typographical issues throughout the paper (e.g., “Equation (??)” in Line 203). Careful revision is required to improve readability and presentation quality.

3. Unclear novelty justification: Based on the ablation study in Table 3, the sequential position embedding provides the most noticeable accuracy improvement, but this appears to be a design trade-off between achieving higher accuracy for shorter contexts versus maintaining the ability to handle longer ones (as mentioned in Point 1). The “Dividing video by frames” step yields only marginal improvements, while “Adding frames to the sink” also seems to be a heuristic adjustment, especially when adapting attention sink mechanisms from LLMs to VLMs. Additional clarification is needed to justify the claimed novelty and differentiate these design choices from existing approaches.

### Questions
See Weaknesses

### Soundness
3

### Presentation
2

### Contribution
2
