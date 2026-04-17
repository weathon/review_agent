# MoB: Mixture of Block Transformer for Accelerating Video Generation with Dynamic Routing

- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Diffusion Transformers (DiTs) have demonstrated exceptional performance in high-fidelity image and video generation tasks. However, their iterative denoising process introduces substantial computational redundancy within Transformer modules, resulting in prohibitively high computational costs and slow inference speeds. Through comprehensive experimental analysis of existing DiTs, we reveal two key observations: (1) outputs of different Transformer blocks exhibit significant similarity during the denoising process, and (2) block-level redundancy varies dynamically across denoising timesteps. Based on these insights, we propose \textbf{Mixture of Blocks (MoB)}, the first framework to introduce block-level dynamic routing for DiT acceleration. The core innovation of MoB lies in a lightweight routing network that dynamically evaluates the importance of each Transformer block based on input prompts. At each denoising step, we propose the Ada-Top-\(k\) mechanism which selects relevant blocks by using the k-th largest score as an adaptive threshold, avoiding the winner-take-all problem of traditional soft selection while eliminating 10-20\% of redundant computations. To compensate for information loss from skipped blocks, we design a Block Cache mechanism that maintains generation quality by reusing intermediate feature differences from previous timesteps. Furthermore, MoB integrates adaptive timestep skipping and employs knowledge distillation to train the routing network, achieving enhanced inference efficiency and training stability. In addition, we evaluate its generalization ability on image generation tasks using Flux.1. Extensive experiments demonstrate that MoB achieves significant inference acceleration while preserving generation fidelity in both video and image generation tasks, consistently outperforming existing baseline methods in both efficiency and quality.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Mixture of Blocks (MoB), a training-free acceleration framework for diffusion-transformer (DiT) video generators. A lightweight routing network predicts block-level importance from text embeddings; an Ada-Top-k gate then activates only the most relevant Transformer blocks at every denoising step. To mitigate information loss, a Block Cache reuses inter-block feature differences from the previous timestep. The authors further skip selected mid-range timesteps and train the router with knowledge distillation plus a computation-aware reward while keeping the backbone frozen. On three public DiT models (CogVideoX-5B, HunyuanVideo, Wan 2.1) MoB reports a 1.25 × – 1.47 × speed-up.

### Strengths
The combination of Ada-Top-k routing and a lightweight Block Cache yields measurable speed-ups.

### Weaknesses
1. The acceleration ratio(≈1.3 ×) is moderate and sometimes lower than cache-based baselines. 

2. Novelty is incremental: block reuse and dynamic gating have been explored in Δ-DiT [1], BlockDance [2], and Learning-to-Cache [3]. A clearer comparison with these works is needed. 

3. The the overhead of router is not analysed. 

4.  Applicability to text-to-image models is not discussed. 

[1] Chen P, et al. $\Delta $-DiT: A Training-Free Acceleration Method Tailored for Diffusion Transformers. arXiv preprint arXiv:2406.01125, 2024.
[2] Zhang H, et al. Blockdance: Reuse structurally similar spatio-temporal features to accelerate diffusion transformers. CVPR 2025.
[3] Ma X, et al. Learning-to-cache: Accelerating diffusion transformer via layer caching. NeurIPS2024

### Questions
See Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the MoB framework, which accelerates the video generation task of DiT models through a dynamic routing mechanism. MoB trains a routing network to leverage block-level redundancy, skipping unnecessary computations.Experimental results show that MoB achieves a 1.44x speedup with almost no degradation in generation quality.

### Strengths
1.The MoB framework proposed in this paper offers significant advantages in accelerating video generation. MoB introduces block-level dynamic routing into the video generation field for the first time and uses the enhanced Ada-Top-k strategy to select blocks to skip.

2.The writing is clear and effectively presents the theoretical background of the MoB framework, experimental validation. MoB's flexible framework design allows for integration with other acceleration techniques。

### Weaknesses
1.The routing selection method proposed by the authors does not provide a detailed analysis of block-level redundancy. The use of MSE to evaluate similarity seems insufficient and lacks credibility. It is recommended to use more analysis methods to better support the argument.

2.The description of using knowledge distillation to train the routing network is unclear. For example, which intermediate output logits from the teacher network are aligned with the routing network? Additionally, the computational cost and time required for distillation with the original text-to-video model as the teacher need further clarification.

### Questions
1.In Section 2.2 and Figure 4, regarding the block cache, the output difference from the previous timestep is used to calibrate the current timestep. If the output differences for each block at every timestep need to be calculated, could this negate the speedup achieved by block routing?

2.In Table 1, MagCache outperforms MoB in both inference time and generation quality. What are the distinctive advantages of MoB compared to MagCache?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The authors leverage their insights regarding the output similarity across blocks and timesteps in the denoising process to propose Mixture of Blocks, which routes the input to only a selected number of blocks, leading to inference time speedups.

### Strengths
- The method is very well motivated and the method formulation is very clear, and easy to understand.
- The authors incorporate block caching and approximating the layer output at each timestep using the cached outputs from the previous layers, in case the particular block is not chosen by the router. This fits well with their empirical observations of similarity across timesteps.
- The empirical results show competitive performance with other efficient methods, maintaining near neutral quality when compared to baselines while producing inference speedups

### Weaknesses
- Eq 1 suggests that the method applies equally to all tokens. But there might also be token wise varations in the behavior of outputs, which the currently proposed method doesnt address.
- In Figure 1 left, the authors identify that the information rich blocks are the ones at the later half of the network, based on the MSE differences between consecutive blocks. However, the proposed router does not leverage this insight, and uniformly tries to balance the top-k based on its own training objective. I would expect that this insight could be used to give more emphasis to later blocks. Or in case the proposed training objective naturally leads to the router converging to this behaviour, that would be an interesting ablation to see. It would also help in identifying the efficacy of the router, whether it is able to identify empirically observed correlations.
- Similar comment for Figure 1 right, given that in initial iterations, the correlation is more localized and more spread out in later timesteps, can this be incorporated in the router objective, and in the current setup, does the router already depict this nature? Also, given this timestep wise trend, would it make sense to have a different k for each timestep, lower for initial and higher for later ones?
- In Eq 4, why is the delta term between $i-1$ and $i-2$. I would have expected it to be $h_{i}^{t} = h_{i-1}^{t} + (h_{i}^{t-1} - h_{i-1}^{t-1})$. On a related note, consider a particular block getting repeatedly skipped based on router assignments, won't that affect the approximation as it might be getting propagated from far away timesteps.
- Given the block cache idea, the authors should also provide memory considerations, particularly how much memory is required to store these cached representations.
- The authors mention that for smaller networks, the method might be counterproductive. Can this be concretized, particularly how deep a network needs to be, in order to be eligible for advantages that come from this method? I believe this point needs to be emphasized more, because this directly relates to the benefits that this approach can provide.
- The three weight distributions mentioned in A.1, aren't those just renormalizations/scaled versions of each other?

### Questions
Please see the weaknesses section above.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a lightweight routing network that dynamically evaluates the importance of each Transformer block based on input prompts. Extensive experiments demonstrate that MoB achieves inference acceleration while preserving generation fidelity, outperforming existing baselines in both efficiency and quality.

### Strengths
1. A lightweight routing network leverages input text embeddings to estimate the importance of each Transformer block. 
2. Beyond these core innovations, the framework further integrates an adaptive timestep skipping strategy and knowledge distillation using the original DiT as a teacher model, which enhances training stability and reduces data dependency.
3. Extensive experiments are conducted on strong baselines, including CogVideoX-5B, HunyuanVideo, and Wan2.1.

### Weaknesses
1. The authors are encouraged to include visual analyses of the routing decisions—for example, activation probability distributions across blocks under different text prompts, or the dynamic evolution of the adaptive Ada-Top-k threshold across denoising timesteps. Such visualizations would significantly improve the paper’s clarity, interpretability, and overall persuasiveness.
2. The current approach uses fixed intervals and predefined timestep ranges for skipping, which works well empirically but may lack flexibility when applied to DiT models with different architectures or diverse video generation requirements. Future work could investigate dynamic skipping strategies that adjust intervals and ranges based on denoising-stage characteristics (e.g., noise level, feature change rate) to balance efficiency and generation quality better.
3. While DiT models excel at generating long videos, the current experiments focus primarily on short sequences (49–81 frames). The authors should consider extending evaluations to longer videos (e.g., 300+ frames) to assess: 1) the stability of dynamic block selection over extended inference; 2)the memory overhead and error accumulation of the block caching mechanism; and 3) frame-to-frame consistency and motion smoothness in generated videos. Such experiments would help more comprehensively define the operational boundaries and scalability of the MoB framework.

### Questions
For specific issues, please refer to the points listed in the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
