# TWISTED: Enhancing Transformer World Models with Spatio-Temporal Encoding and Graph-Based Optimal Decoding

- Avg Score: 4.00
- Decision: Reject
- Scores: 6, 2, 6, 2

## Abstract
Model-based reinforcement learning improves sample efficiency by using learned world models to simulate experiences for training agents.
Recent world models that leverage transformers demonstrate high quality simulations, leading to better agent performance.
However, transformer world models underutilize spatial relationships between visually adjacent tokens, which are critical when interacting in visual environments. 
Additionally, current models rely on sampling methods for transformer decoding that do not leverage visual similarities among subsequent frames.
To address these limitations, we introduce TWISTED, a transformer world model with 3D spatio-temporal positional encoding and a graph-based optimal decoding strategy specific to 2D visual environments.
Our experiments show state-of-the-art performance on the Craftax-classic, Craftax, MinAtar, and Atari 100K benchmarks, challenging visual environments requiring long-horizon object recall and interaction.
The proposed method achieves a return of 72.5\% and a score of 35.6\% on Craftax-classic, significantly surpassing the previous best of 67.4\% and 27.9\%.
We plan to release our source code on GitHub upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents TWISTED, a transformer-based world model that integrates richer spatial and temporal structure via (1) a **3D spatio-temporal positional encoding** scheme and (2) a **graph-based optimal transport (OT) decoding mechanism**. The first component generalizes standard 1D positional encodings to better handle video or spatially grounded RL tasks, while the second aligns generated tokens across time using OT, reducing temporal inconsistency and object drift. The model achieves improved performance on several visual RL benchmarks, including Craftax-classic and MinAtar, showing better stability and long-horizon prediction accuracy.

### Strengths
* The motivation is well-grounded: visual world models naturally involve spatial-temporal relationships, and standard RoPE encodings cannot represent them adequately.

* The proposed OT decoding is both novel and intuitive, promoting temporal coherence by reusing tokens from prior frames via transport mapping.

* Experiments show consistent improvement across datasets, and qualitative samples convincingly illustrate more stable rollouts.

* The writing style is clear, with excellent diagrams explaining both components.

* The method’s modularity allows easy integration with other transformer backbones.

### Weaknesses
* The ablation section is somewhat limited; it does not isolate the effects of spatio-temporal encoding and OT decoding separately.

* Computational costs (both training and inference) for the OT solver are not analyzed, which may be substantial in practice.

* Comparisons are missing with recent state-space world models (e.g., Mamba, S4WM) that offer long-context handling with lower overhead.

* There is little theoretical analysis of OT stability under noisy or high-variance feature representations.

### Questions
1. Could you provide ablations that test encoding-only and decoding-only contributions?

2. Which OT solver and regularization settings (e.g., ε, Sinkhorn iterations) were used, and how sensitive are results to them?

3. How much additional computational cost does OT decoding introduce?

4. Can this approach scale to higher-resolution visual inputs?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes two components for improving the performance of transformer world models in visual environments: one addresses temporal and spatial adjacency / positional information encoding and the other addresses the significant visual redundancy in certain visual applications.
The method is evaluated on the Craftax and MinAtar benchmarks, presenting improved performance compared to existing results.

### Strengths
- The paper is overall well-written and easy to follow.
- The ablations setup and results are informative and well executed. The experimental design and results allow to draw meaningful conclusions on the impact of each of the proposed components. 
- The proposed optimal transport decoding mechanism is original.
- The approach, TWISTED, outperforms previous baselines on the Craftax, Craftax-Classic and MinAtar benchmarks.

### Weaknesses
`W1`: The spatio-temporal position encoding (STPE) approach is very similar to methods proposed in several prior works [1][2][3][4], under the name 3D RoPE, even in the context of world models, including the combination of 3D RoPE and learned absolute positional encoding [1].

Comparing (extensively) the proposed STPE to VideoRoPE and the above 3D RoPE methods would be necessary to support the necessity of STPE.

Notably, according to the ablations study (Table 2), the most significant contributor to the final performance is the spatio-temporal position encoding mechanism (STPE).



`W2` Similarly, in the related works section, under "Positional embeddings in video modeling", 3D RoPE methods were not sufficiently covered.


`W3`: The second claimed contribution is the optimal transport decoding, which addresses the problem of poor utilization of the significant visual redundancy in some video applications. 
1. Given the empirical evidence in the paper (Table 2), its contribution to the final performance is questionable, and marginal at best. 
2. This component adds a non-trivial complexity to the overall method.
3. In contrast to prior approaches such as Delta-IRIS, which ultimately yields shorter token sequences and thus benefits improved efficiency (lower computational cost), the proposed approach *adds* a computational cost on top of that of the baseline. 
4. Since the proposed algorithm, TWISTED, is the only baseline that uses this positional encoding mechanism, it is unclear how prior works such as delta IRIS would perform under the implementation of TWISTED, in other words, whether the observed improvement (if not a statistical fluctuation) stems primarily from 3D-RoPE + implementation and architectural differences.

To support this approach empirically, it is required to consider an experimental setup for studying solutions to the poor utilization of visual redundancy problem in isolation.

To evaluate this proposed approach more reliably, I would suggest to consider the following setup:
Starting from a baseline version of TWISTED where optimal transport decoding is disabled, add the Delta-IRIS mechanism. Similarly, consider other appropriate baselines as well. Then, evaluate all baselines, and compare the performance to those of TWISTED.

This would allow to eliminate any external factor such as implementation details and the advantage of 3D RoPE, and would show the impact of the proposed component in isolation.



`W4`:
> In this paper, we present TWISTED, a transformer world model tailored for visual RL environments. (line 480)

The paper claims that the method addresses visual RL environments (a very general claim). However, in practice, the focus is on visual environments with significant visual redundancy such as Atari and Craftax, where a significant overlap exists between many frames. In more complex visual environments, such as Minecraft, autonomous driving, or real-world robotics, it is unclear whether such redundancy is significant enough to make a difference in performance, and whether the method would maintain performance in cases with insignificant frame overlap (in the context of the optimal transport decoding). Since the paper does not provide supporting evidence for such environments, I believe the scope of the claims should be narrowed.


`W5`: The paper lacks an explicit formal description of the STPE method (in mathematical notation).


`W6`: It would be valuable to extend the ablations to MinAtar as well.

--------

[1] Agarwal, N., Ali, A., Bala, M., Balaji, Y., Barker, E., Cai, T., ... & Zolkowski, A. (2025). Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575.

[2] Song, K., Chen, B., Simchowitz, M., Du, Y., Tedrake, R., & Sitzmann, V. (2025). History-guided video diffusion. arXiv preprint arXiv:2502.06764.

[3] Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., ... & Tang, J. CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. In The Thirteenth International Conference on Learning Representations.

[4] Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., ... & Zhong, C. (2024). Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603.

### Questions
`Q1`: 
> Return is averaged over episodes of the final 50,000 environment interactions to smooth out variance (line 325)

A better practice would be to evaluate the final performance (of the fixed trained model) by collecting $K$ test episodes with the trained agent (fixed weights) and aggregate the resulting returns, with e.g., $K=100$ (per random seed). 


`Q2`: In the related works section, you mentioned VideoRoPE, but did not specify how this approach differs from the proposed STPE, or justify why STPE is needed despite the existence of VideoRoPE. Did you try using VideoRoPE? What is missing in VideoRoPE that STPE offers? Are there empirical evidence to support this? Again, I would suggest including a comparison. 


`Q3`: If I understand correctly, the tokenizer (Section 3.1) is not part of the novelty of the paper. Hence, it is more appropriate to include this information in the preliminaries section.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes TWISTED, a transformer-based world model that incorporates two main components: a 3D spatio-temporal positional encoding and an optimal transport-based decoding mechanism. The method is evaluated on several challenging benchmarks (Craftax-classic, Craftax, MinAtar), achieving state-of-the-art performance. The core idea of using optimal transport to enforce object persistence across frames is interesting and well-motivated for visual environments.

### Strengths
+ The optimal transport (OT) based decoding is novel and well-motivated. 
+ The 3D spatio-temporal positional encoding (STPE) in world modeling is novel and effective.
+ The empirical evaluation is comprehensive and convincing, demonstrating SOTA performance.

### Weaknesses
- The paper presents two independent ideas (STPE and OT). Ablation studies show both help, yet it is not clear whether the performance gain is additive or synergistic. It is missing a baseline combining STPEwith a simpler and less expensive decoding heuristic. 
- The authors should provide a detailed breakdown of the OT's cost and discuss the scalability of their Sinkhorn solution to environments with more tokens (e.g., higher-resolution images).
- The OT formulation uses fixed cost coefficients (c_d, c_w) and it is not disscused how they are selected and how sensitive are the results.

### Questions
See above.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces a transformer-based world model for visual reinforcement learning that integrates two components: a 3D spatio-temporal positional encoding (STPE) to better capture spatial and temporal relationships across visual tokens, and an Optimal Transport (OT) decoding mechanism that formulates next-frame prediction as a transport matching problem between previous and predicted tokens. Evaluated on Craftax-Classic, Craftax, and MinAtar benchmarks, TWISTED achieves higher prediction accuracy and improved returns compared to prior transformer world models, showing faster convergence and more consistent visual rollouts across time.

### Strengths
1. The paper presents a novel perspective on generation and decoding through the integration of optimal transport, meanwhile offering a workable inductive bias that aligns well with the structured dynamics of grid-based visual environments.

2. The proposed spatio-temporal RoPE extension is well-motivated and effectively enhances temporal and spatial consistency, leading to tangible performance improvements over standard transformer encodings.

3. The paper features well-prepared figures and visualizations, which clearly convey both the model architecture and the qualitative impact of the proposed methods.

### Weaknesses
1. The method is evaluated only on specialized 2D grid-based benchmarks, and it remains unclear whether TWISTED can generalize to more complex or unstructured visual environments. The paper states that it targets “visual RL environments,” but the current setup does not demonstrate generalization to tasks such as 3D worlds, camera motion, or non-grid visual inputs (e.g., Minecraft/DMControl/Original Atari).

2. Within the evaluated domain, the contribution of the OT decoding component appears secondary to that of the spatio-temporal positional encoding (STPE). While OT improves temporal coherence, it incurs significant implementation, and the performance gains reported seem relatively modest compared to the added complexity.

3. The proposed approach builds upon existing transformer world model frameworks with incremental extensions. While these components are technically sound, their performance improvements appear proportional to their incremental nature, resulting in marginal gains that align with expectations given the added complexity. Consequently, the overall contribution and impact are somewhat limited in scope.

### Questions
1. Have the authors evaluated the model under different values of the distance cost parameter in the OT decoding step? How does this affect both prediction accuracy and computational efficiency? Why was the maximum feasible motion threshold set to four (i.e., displacement ≤ 2 cells per axis)? Was this empirically tuned, or is it based on environment-specific priors?

2. Have the authors tested environments that do not conform to the grid-based or bounded-motion assumptions? If not, how might the method handle these cases?

### Soundness
3

### Presentation
3

### Contribution
4
