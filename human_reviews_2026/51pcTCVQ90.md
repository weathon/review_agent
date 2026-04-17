# Greedy Distill: Efficient Video Generative Modeling with Linear Time Complexity

- Decision: Reject
- Scores: 4, 8, 6, 4

## Abstract
Due to bidirectional attention dependencies, video generation models generally suffer from $O(n^2)$ computational complexity. In this work, we find the “local inter-frame information redundancy" phenomenon which indicates strong local temporal dependencies in video generation, with global attention to distant frames contributing only marginally. Built upon this finding, we introduce a novel distillation training paradigm for video diffusion models, namely GREEDY DISTILL. 
Specifically, to generate the next frame using only the 0-th and the last frames, we propose the Streaming Diffusion Decoder (SDD) as the “Greedy Decoder" to avoid redundant computational costs from the other frames. 
Meanwhile, to our knowledge, we introduce Efficient Temporal Module (ETM) to capture the global temporal information across frames.
These two modules achieve the computational complexity reduction from $O(n^2)$ to linear. Moreover, we make the first attempt to apply RL fine-tuning to address the error accumulation during streaming generation.
Our method achieves an overall score of 84.60 on the VBench benchmark, surpassing previous state-of-the-art methods by large margins(+4.18%). Qualitative results also demonstrate superior performance. 
Leveraging its efficient model structure and KV cache, it is able to rapidly generate high-quality video streams at 24 FPS (nearly 50% faster) on a single H100 GPU.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes Greedy Distill, an asymmetric distillation that turns a bidirectional DiT video diffusion teacher into a fast student composed of an Efficient Temporal Module (chunked AR Transformer with sliding-window attention) and a Streaming Diffusion Decoder with KV caching. A rollout/RL-style fine-tuning step aims to curb exposure bias. The claim is near-linear time in frames while maintaining teacher-level quality.

### Strengths
* Quality: Sensible two-stage training; clear ablations indicating ETM/RL contributions; competitive throughput/latency.
* Clarity: Architecture, complexity intuition, and training pipeline are easy to follow.

### Weaknesses
* Claim vs. demos: Fig. 7 and some other provided demos appear to show lower dynamics than baselines. This suggests a locality bias that may trade motion amplitude/scene changes for stability.
* Dynamics not quantified: No direct evaluation of motion strength; please report metrics like VBench Dynamic-Degree, optical-flow magnitude/variance, or long-horizon motion persistence.
* Writing problem: Sec 3.3 title (row 372) overlaps with the previous section

### Questions
I don't understand the necessity to formulate the finetuning as an RL process. Any fundamental differences between this and directly using the MSE loss based on the KL divergence? Could you explain more?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores the quadratic complexity bottleneck in diffusion-based video generation and introduces GREEDY DISTILL, a teacher-student distillation framework that reduces complexity to linear. The core idea is to decouple temporal modeling (via an autoregressive Efficient Temporal Module with sliding-window attention) from frame synthesis (via a Streaming Diffusion Decoder that only conditions on the 0-th and last frames), followed by reinforcement-learning fine-tuning to suppress exposure bias. Extensive experiments on VBench and human evaluations show good performance of GREEDY DISTILL. The manuscript is clearly written, with project page, detailed ablations, and supplementary material.

### Strengths
- The authors propose a novel asymmetric architecture that enables linear-time streaming synthesis and provide theoretical complexity analysis.

- Experiments are comprehensive: both real-time and long-duration generation, human preference studies, and ablations of components, using publicly available benchmarks and code.  

- The writing is clear, with intuitive figures, step-by-step algorithm boxes, and a detailed reproducibility statement that facilitates follow-up research.

### Weaknesses
- The experiments are mainly conducted on Wan2.1. It would be more convincing to include other backbones such as HunyuanVideo or CogVideoX to demonstrate the generality of the proposed framework.

- The writing could be improved for better readability, e.g., adding punctuation after equations and providing clearer captions for figures. Several typos such as “stege” → “stage” in Section 3.1

- The reinforcement learning fine-tuning is an interesting addition, but its novelty seems mainly in the application context rather than in algorithmic design.

### Questions
see weakness

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper "Greedy Distill: Efficient Video Generative Modeling with Linear Time Complexity" introduces a novel distillation training paradigm that significantly reduces the computational complexity of video generation models. The proposed method achieves state-of-the-art performance in terms of both speed and quality, making it a valuable contribution to the field. The paper is well-written and the results are compelling, but there are opportunities for further improvement in terms of detailed comparisons, additional experiments, and discussion of future work.

### Strengths
see Summary

### Weaknesses
The number of displayed videos is quite limited. It is difficult to fully assess the method's effectiveness and robustness with such a small sample size.

Suggestion: The authors should provide a detailed breakdown of the user study results. This should include statistical analysis, user feedback, and any significant findings. Additionally, the paper should discuss how these results align with the quantitative metrics and what insights they provide into the overall performance of the proposed method.

### Questions
no

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a new architecture and training algorithm for autoregressive video diffusion models. 

Architecture: Instead of a single diffusion transformer with causal attention, it uses two models - one causal transformer with local attention (ETM), and another causal diffusion model (SDD) that takes the first frame, the previous frame, and the output of ETM to denoise the current frame.

Training: It first uses teacher forcing with diffusion loss to finetune SDD + ETM, both initialized from Wan2.1 with bidirectional attention. It is then trained with a variant of Self-Forcing + DMD that uses the student model itself as the fake score network (instead of fine-tuning a seperate one).

Experiments demonstrate higher efficiency and less error accumulation than baselines.

### Strengths
- The student splits temporal modeling (ETM) from per-frame generation (SDD). This architecture intuitively makes sense, and the observation (Fig. 2) motivates restricting attention to nearby frames and justifies ETM’s sliding window as a principled efficiency/fidelity trade-off rather than a heuristic.
- On Wan 2.1, the distilled student runs at 24 FPS reaches VBench 84.60, with qualitative long-video examples showing reduced error accumulation vs. Self-Forcing and CausVid. The paper also includes ablation evidence showing the effectiveness of ETM and "RL fine-tuning".

### Weaknesses
- The title/teaser focuses on “linear time,” but that follows directly from sliding-window causal attention. It is generally well known that local attention can produce coherent videos in linear time with the downside of sacrificing long term memory. Consider reframing the main contribution around the new two-stage architecture (ETM+SDD) and the training algorithm.
- Section 2.2.2 casts training as deterministic continuous policy gradients over a reverse-KL reward, yet the practical algorithm is very close to Self-Forcing. The paper repeatedly claims “first attempt to apply RL to address error accumulation,” which feels overstated given the close methodological overlap with existing work. Consider recasting this as a conceptual bridge: “We show Self-Forcing-style rollout training admits an RL interpretation. 
- Also, there is actually a big difference between the proposed algorithm (and Self-Forcing) versus RL algorithms (e.g., DDPG Lillicrap et al., 2015) that the paper cites. In Self-Forcing, gradients flow through the rollout trajectory; in DDPG, rollouts are off-policy in a replay buffer and are detached, and gradients flow through the reward estimator. The proposed algorithm follows Self Forcing closely but materially deviates from the RL algorithms cited.
- The sentence "...more broadly known as exposure bias, where a model is trained exclusively on ground-truth context but must rely on its own imperfect predictions at inference time, resulting in a distributional mismatch that compounds errors as generation progresses." is exactly copied from a sentence in the paper of Self Forcing. Please paraphrase and cite the original source.
- Minor spacing/format issues. In many sentences there is no space before punctuation symbols. vspace issues in e.g. L372
- The paper proposes to use the student model itself to estimate the score function of the student distribution. However, the student, when trained with a few-step prediction objective, actually no longer predicts the score function. For example, given x_T input, an ideal score predictor would predict the dataset mean as x0 prediction. However, the student model is trained to predict a realistic output. The idea of using the few-step student model itself to estimate the score function does not seem to be theoretically correct.

### Questions
- Some design choices of ETM + SDD can be better justified. For example, why does SDD see the first frame sink, but not ETM? Abaltion studies on window size would also be helpful.
- How is the 0.24s latency calculated and is it the time to generate the first block of frames? The model should not seem to be more efficient than Self Forcing/CausVid initially since the local attention do not provide speedup benefits?

### Soundness
3

### Presentation
1

### Contribution
2
