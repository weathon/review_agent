# Planning at Inference: MCTS Test-Time Scaling for Long Video Generation

- Decision: Reject
- Scores: 4, 6, 4

## Abstract
Generating long videos with consistent content and visual quality remains a ma-
jor challenge, as existing one-shot and chunked methods often suffer from se-
mantic drift and compounding artifacts. We explore Test-Time Scaling (TTS)
as a framework for long video generation, formulating the task as a sequential
decision-making problem. Our approach uses Monte Carlo Tree Search (MCTS)
to evaluate multiple continuations with look-ahead rollouts and backpropagated
rewards, and we introduce a Multi-Tree MCTS variant that improves exploration
in continuous generation spaces. The method is modular and can be applied to ex-
isting backbones without retraining. Experiments on Cosmos-Predict2 and other
models show consistent improvements in object permanence, temporal coherence,
and text-video alignment over Best-of-N, Greedy, and Beam search. Furthermore,
our method produces high-quality videos exceeding 20 seconds, surpassing the
output of leading models like Sora and Kling by 18% and 47% respectively, all
while maintaining comparable visual fidelity. Although the results are limited
by the quality of current generators and verifiers, our study highlights both the
promise of search-based TTS and the limitations of today’s video generation and
evaluation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper frames long video generation as sequential decision making and proposes test time search with Monte Carlo Tree Search to plan over chunked continuations, guided by a process reward model for local chunk quality and an outcome reward model that aggregates scores over the full sequence. The method is model agnostic, sits on top of existing backbones without retraining, and introduces a multi tree variant to widen exploration in continuous spaces. Across several generators, the approach improves temporal consistency and object permanence relative to autoregressive decoding, Best of N, greedy, and beam search, and reports longer, competitive quality videos when compared qualitatively and with automated metrics to recent long video systems. The paper provides algorithmic details, ablations on compute budget, and comparisons of single tree versus multi tree search, while also acknowledging dependencies on the underlying generator and verifier quality.

### Strengths
1. Clear formulation of long video generation as planning with Monte Carlo Tree Search, including a walk through of selection, expansion, rollout, and backpropagation plus an explicit UCB objective.

2. Multi tree search broadens exploration under a fixed branching factor and empirically outperforms single tree for the same budget.

3. Practical recipe that is plug in and does not require retraining, which increases utility for current systems constrained by backbone quality.

### Weaknesses
1. Heavy reliance on automated reward signals for both search guidance and evaluation, with outcome reward defined as a simple sum over chunks, risks overfitting to verifier idiosyncrasies rather than human preference on long horizon coherence. A controlled human study is missing.

2. The exploration constant, branching factor, rollout policy, and beam initialization depth can strongly affect MCTS behavior. Sensitivity analysis is not comprehensive.

### Questions
1. How sensitive are results to the weighting of VideoScore, CLIP alignment, and the LAION perceptual model in the process reward, and to the definition of the outcome reward as a sum rather than a learned temporal model

2. Under a fixed wall clock and identical hardware, how does the method compare to beam and greedy tuned for the same final runtime, including beam initialization time and rollout parallelism

### Soundness
3

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
3

### Summary
The paper proposes using MCTS for planning-based long video generation, which expands an important direction in the TTT field. Through this approach, the paper even achieves long video generation results that surpass closed-source SOTA models, demonstrating the potential of TTT in long video generation.

### Strengths
- The work has a certain degree of novelty and community value. The paper is the first to apply MCTS-based TTT to long video generation, showcasing the value of classical methods in the video domain.
- The experimental results are impressive. The proposed method enables Cosmos-Predict2 to surpass or tie with closed-source SOTA models (Sora/Kling), which demonstrates the strong potential of TTT.

### Weaknesses
- Tab. 5 should include a comparison of the computational cost.
- Regarding the long-video baselines, the paper would be more sound if a more comprehensive set could be included [1,2]

[1] FIFO-Diffusion: Generating Infinite Videos from Text without Training

[2] Skyreels-v2: Infinite-length film generative model

- The paper lacks discussion and comparison with several recently accepted works on long-video generation.

[1] Zhao et al., Riflex: A Free Lunch for Length Extrapolation in Video Diffusion Transformers (ICML 2025).

[2] Tan et al., FreePCA: Integrating Consistency Information Across Long-Short Frames in Training-Free Long Video Generation via Principal Component Analysis (CVPR 2025).

[3] Lu et al., FreeLong: Training-Free Long Video Generation with SpectralBlend Temporal Attention (NeurIPS 2024).

[4] Cai et al., DitCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation (CVPR 2025).

### Questions
See the Weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The author introduces a Multi-Tree MCTS variant that improves exploration in continuous generation spaces.

### Strengths
1. The paper is well written.

2. The author introduces a Multi-Tree MCTS variant that improves exploration in continuous generation spaces. It is interesting.

### Weaknesses
1. I would like to know the time it takes to generate a 1-minute video with and without using your MCTS, and provide a quantitative comparison of the results.

2. The biggest issue with video generation is the excessive time consumption. This MCTS could make generating a long video take 24 hours, potentially requiring 20 times more time.

3. It is difficult to implement. The biggest challenge of this model is the accurate training of the Process Reward Model and Outcome Reward Model. As we know, video quality is hard to evaluate (the error rate of evaluation is high). Any slight error in the evaluation of these two models could lead to a massive search error.

4. MCTS does not have good robustness for the Process Reward Model and Outcome Reward Model.

5. I believe the author should focus on reinforcing the video model with reinforcement learning instead of using TTS, as it is a more efficient and practical solution.

### Questions
see weakness

### Soundness
2

### Presentation
3

### Contribution
3
