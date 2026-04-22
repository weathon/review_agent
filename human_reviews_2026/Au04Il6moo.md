# Draft-and-Target Sampling for Video Generation Policy

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 6, 2

## Abstract
Video generation models have been used as a robot policy to predict the future states of executing a task conditioned on task description and observation. Previous works ignore their high computational cost and long inference time. To address this challenge, we propose Draft-and-Target Sampling, a novel speculative decoding-like inference paradigm for video generation policy that is training-free and can improve inference efficiency. We modify the classic principle of speculative decoding design and redefine the draft and target as two complementary denoising trajectories. To further speedup generation, we introduce token chunking and progressive acceptance strategy to reduce redundant computation. Experiments on three benchmarks show that our method can achieve up to 2.1x speedup and improve the efficiency of current state-of-the-art methods with minimal compromise to the success rate. Our code is available at anonymous github.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes Draft-and-Target Sampling (DTS), a speculative decoding method for video generation policies. The approach employs large diffusion steps for the draft phase and smaller diffusion steps for the acceptance/rejection phase. To mitigate accumulation errors and improve efficiency, the paper introduces token chunking and progressive acceptance strategies. Experiments conducted on iTHOR, MetaWorld, and LIBERO demonstrate improved inference efficiency while maintaining comparable policy performance.

### Strengths
1. The paper presents a novel perspective on video generation policies, focusing on improving inference efficiency. To the best of my knowledge, the proposed strategy of combining large and small diffusion steps for speculative decoding has not been explored before.
2. The experimental evaluation is comprehensive, covering three distinct domains, and the results show consistent and satisfactory performance.

### Weaknesses
1. **Lack of theoretical analysis of acceleration**: 
   While the empirical study is extensive, the paper lacks theoretical discussion or quantitative analysis of the acceleration achieved:
   - The assumption in Line 50 that video generation policies usually have low resolutions is questionable. Recent works such as Vidar [1] demonstrate a clear trend toward higher resolutions in this paradigm. The paper does not analyze how the proposed method is applicable under different resolutions (e.g., whether it is memory-bound or compute-bound).
   - Although the paper presents detailed algorithmic formulations, it omits an analysis of time complexity, particularly regarding the impact of token chunking. This makes it difficult for readers to understand the expected acceleration ratio or to choose suitable hyperparameters
   - If the authors claim that generation is memory-bound, then the memory cost of token chunking (e.g., the number of model parameter loads) should be examined, as it may remain similar or even increase with chunking.
   - The sequential nature of token chunking may limit parallelism. This trade-off should be discussed explicitly.
2. **Presentation and clarity issues**:
   - Section 4 is overly verbose, containing many complex formulas. A clear schematic figure illustrating the overall process would greatly enhance readability.
   - The motivation for token chunking as a way to mitigate accumulation error of draft sampling is somewhat trivial and could be summarized more concisely.
   - The discussion in Section 4.3 about token chunking reducing accumulated error of target sampling (Line 262) is insightful but should be introduced earlier, e.g., in Section 4.2.
   - The motivation of the Progressive Acceptance Strategy is not sufficiently explained or justified.
   - There is considerable repetition across the three benchmark descriptions in Section 5.2 (e.g., shared hyperparameter settings), which could be consolidated to improve conciseness.

### Questions
1. What is the motivation and intuition behind the Progressive Acceptance Strategy?
2. How does this work compare to Accelerated Diffusion Models via Speculative Sampling [2], which also avoids training a separate draft model?
3. Why is DDIM-10 chosen as the baseline solver for diffusion sampling instead of more advanced solvers such as DPM-Solver?
4. Could the proposed approach be extended to general image or video generation models, beyond policy-oriented settings?
5. On iTHOR, why does DTS improve policy performance in addition to acceleration? Were the results averaged over multiple runs to ensure statistical significance?

[1] Vidar: Embodied Video Diffusion Model for Generalist Manipulation

[2] Accelerated Diffusion Models via Speculative Sampling.

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
In this paper, the authors focus on efficient video generation for robot policies. One major issue with contemporary video models is their slow speed. To address this issue, the authors propose the application of Draft-and-Target sampling, with some additional tweaks (chunking and progressive acceptance strategy) to video generation. They apply their approach to three datasets. They are iThor, Meta-World, and Libero, and they show improvements when accounting for the significant speedup in computational time.

### Strengths
1. The paper touches on an overlooked but very important problem with the current paradigm for video generation. It is generally too slow to be useful for robotics policies and planning algorithms. This paper is a promising step in the right direction.

2. The approach is simple and easy to implement.

3. Quantitative performance relative to baselines is promising.

### Weaknesses
1. The technical novelty is somewhat limited. No new models or approaches seem to be proposed in this paper. It appears that the contribution of this paper is largely an application of an existing idea to the realm of robotic control. 

2. While performance is promising, the speedup is generally modest (about 2x). It is not clear if this speedup outweighs the additional complexity of the approach. 

3. Data domain of video generation is quite constrained (robotic environments), there are no experiments on unconstrained video data (such as Kinetics-700)

### Questions
1. Could you please elaborate on the technical novelty of the approach? This seems to be an application paper. Are there any new methods or approaches?

2. The speedup from this approach is helpful but not particularly dramatic. Are there any tweaks to the method that could lead to additional speedup (> 5x) without significant sacrifices to performance?

3. Could this approach be viable for video generation on more diverse datasets beyond robotics?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a draft-and-target sampling method for video generation policy inference which achieves computational efficiency compared to prior works across benchmarks.

### Strengths
* The paper proposes system level optimization, e.g., token chunking, to improve the efficiency, which could offer practical values to the community. 
* Results seem to provide empirical performance and efficiency gains.

### Weaknesses
* The core idea is the speculative decoding, which has been widely adopted in the field in LLMs. The modifications of speculative decoding in the discrete space from this paper include using large-stepsize-ODE as draft model and progressive acceptance, which are rather straightforward implementations. Therefore the contribution of the paper should be more explicitly discussed compared to prior works including but not limited to LLMs. 
* Given that the paper uses the same model as draft and target models, these models have the same FLOP count per function evaluation. Reporting NFEs in the experiments could help clarify the computation complexity of the model. 
* Baselines include AVDC (from the original paper cited) and AVDC-10 which aggressively cuts down denoising steps. More baselines with some numbers of denoising steps in between 10 and 100 should be reported, which might already achieve significant speedup compared to AVDC-100 without too much performance loss.

### Questions
* The experiments use AVDC as the backbone. Would the proposed strategy apply to other video generation policy models?

### Soundness
2

### Presentation
2

### Contribution
2
