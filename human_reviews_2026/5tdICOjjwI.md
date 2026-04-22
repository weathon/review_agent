# FLARE: Fast Low-rank Attention Routing Engine

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
The quadratic complexity of self-attention limits its applicability and scalability on long sequences.
We introduce Fast Low-rank Attention Routing Engine (FLARE),
a simplified latent-attention mechanism wherein each attention head encodes $N$ input tokens onto $M \ll N$ latent tokens and immediately deocdes back.
This produces an implicit rank $\leq M$ attention operator using only two cross-attention steps providing a simple and mathematically transparent low-rank formulation of self-attention that can be implemented in $\mathcal{O}(MN)$ time.
Crucially, FLARE eliminates latent-space self-attention and, by expressing both encode and decode steps as fused-attention operations, avoids ever materializing the $M\times N$ projection matrices.
Moreover, by assigning each head independent latent token slices, FLARE realizes multiple parallel low-rank pathways that collectively approximate a richer global attention pattern without sacrificing efficiency.
Empirically, FLARE trains end-to-end on one-million-point unstructured meshes on a single GPU, achieves state-of-the-art accuracy on PDE surrogate benchmarks, and outperforms general-purpose efficient-attention methods on the Long Range Arena suite.
We additionally release a large-scale additive manufacturing dataset to spur further research.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FLARE, a linear-complexity attention mechanism designed to overcome the quadratic cost of traditional attention on large unstructured meshes. FLARE routes attention through learnable latent tokens, effectively forming a low-rank attention matrix that balances efficiency and accuracy. Experiments show that FLARE outperforms advanced PDE surrogate models across multiple benchmarks on accuracy and efficiency. The authors also release a manufacturing dataset to further research in scientific machine learning.

### Strengths
- This paper improves the accuracy and efficiency of Transformer-based PDE surrogates.

- The proposed FLARE framework is clearly described and easily implemented.

- This paper provides a new dataset that is meaningful for the community.

### Weaknesses
- The contribution of the tech is incremental and limited. This work neither tackles a new important scientific problem nor proposes a fundamentally new solution. Regarding to the tech proposed, the core idea of latent tokens has already been explored in prior works such as PerceiverIO, Transolver, LNO and etc. The main contribution is a slight modification of the attention formulation but it is hard to see the advantage of this modification comparing to other models like Transolver from the technical analysis. For example, the attention is operated on sequences with length $M$ in Transolver and on sequences with length $N$ and $M$ in Flare. Why is Flare more GPU efficient if $N \gg M$ ? Are the improvements mainly due to the use of the SDPA kernel? More technical analysis and comparison should be included. Otherwise, the paper reads more like an engineering optimization report rather than a scientific research paper.

- Following the above item, in Table 1 and Line 74, the authors claim that Transolver cannot utilize the SDPA kernel. This is inaccurate. From the official repo of Transolver++, the slice self-attention of Transolver can be naturally implemented by SDPA kernel. This operation is the same in Transolver++ and Transolver, not involving formulation optimization. So the comparison in Table 1 is misleading and not convincing.

- The dataset is also listed as a main contribution. But there is very limited content about the proposed dataset in the main context. The writing needs better arrangement to justify its value and meaning.

- Lack of visualizations. Given that this is a work focus on solving scientific problems efficiently, visualizations of predicted vs. ground-truth physical fields (e.g., stress or flow distributions) are essential. If the simulation states are not physical meaningful, the improvements in numerical accuracy or computational efficiency are of little significance.

### Questions
- In Figure 1, the peak memory usage of vanilla attention is lower than physical attention and flare. Is this reasonable? Did you use any optimization trick? If yes, all baselines should use same optimization tricks.

- In Table 2, baseline models have different parameter counts. Did you compare the efficiency of different models with comparable parameter counts? This is important for fair comparison, especially for efficiency.

- In Figure 4, with different $M$, the peak GPU memory usage is similar. Does this imply that the memory bottleneck is not the attention operation?

- There is only spectral analysis of Flare in Section 3.3 and Appendix B. Without the same analysis of other baselines like Transolver or LNO, it is hard to get the conclusion “FLARE benefits from head-wise diversity”.

### Soundness
2

### Presentation
3

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
Self-attention is quadratic in sequence length, which makes it hard to apply to very large unstructured meshes. Authoers propose FLARE, a linear-complexity alternative. Each attention head routes global communication for N tokens through a learned latent bottleneck of M < N tokens. This induces a low-rank attention map (rank < M) and reduces cost to O(N·M). By choosing M, users can trade accuracy for efficiency.

### Strengths
1) Authors demonstrate a transformer based surrogate with global communication training directly on ~1M-point meshes on a single H100 80GB GPU using off-the-shelf fused attention kernels, and provide scaling curves. Scalability on experiments is something that has not gotten enough attention in the literature in this field, which makes these experiments the strong point.

2) They retrain all baselines (Perceiver IO, Transolver, LNO, etc.) under standardized splits, resolutions, hyperparameters, and model sizes. This makes the reported accuracy numbers more trustworthy.

### Weaknesses
1) FLARE’s low-rank attention via a latent bottleneck is directly linked to the attentions of Perceiver/Perceiver IO (iterative cross-attention into a fixed-size latent that scales linearly with input size) and Linformer (self-attention is low-rank; approximate it to get O(N) complexity). There are various methods along the same direction which makes the novelty of the concept unclear. If the novelty lies in O(N^2) -> O(N.M) then the concept is not new. If the mechanism by which this procedure happens (encode-decode) please see issues 2 and 3. 

2) Authors do not run any meaningful ablations on the encode-decode mechanism which makes it unclear if this per-block encode decode is actually contributing to the results. The only ablations are minor (MLP depth, symmetry tie/untie), so the central architectural claim is not causally demonstrated. If the effect of encode-decode is not clear then do we fall back to the Preciever, or linformer style attention? 

3) The following claim is vert strong " However, the latent bottleneck in PerceiverIO can limit accuracy as the model may discard fine-grained features if the number of latent tokens is too low. ". Preciever style model keep the original tokens [M x ...] and conduct cross attention to these at the beginning of each block. How does this discard information?

4) Authors claim giving each head its own latent slice is a key idea for expressivity, but never ablate a shared-latent variant. The evidence is descriptive spectra plots, not causal. Ablations are missing to show what would have been the effect of a shared latent token. This is especially important since it's not something that has been explored or adopted widely with attention mechanism in general. 

5) The idea of weight sharing across cross attentions is also from Preciever paper so it's not fair to have the claim that PrevieverIO did not include the weight sharing (Table 1).

### Questions
Questions are formed as part of the weakness. Major concern is around the novelty of the work and the claims. I kindly request authors to focus on the novelty, and the claims (claims are specifically mentioned in the weakness) as they are clarifying the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, a linear self-attention mechanism is proposed for applying to large unstructured meshes. It is achieved by projecting the input sequence to a much lower-dimentional latent sequence. It enables training with 1 million points. The proposed model achieves superior accuracy compared to neural PDE models, including Vanilla Transformer, PerceiverIO, GNOT, LNO, and Transolver, on various datasets, including a newly constructed additive manufacturing simulation dataset.

### Strengths
The proposed method achieves sota PDE surrogate performance, while is also capable of handling geometries with a million points.

### Weaknesses
I think it is mandatory that in the experiments, this work also compares with other efficient attention methods proposed in general domains.

### Questions
Is there any design in the proposed method specifically for neural PDE scenarios? What if the proposed method applies to domains other than surrogate PDE solvers, e.g., LLM training?

### Soundness
3

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
This paper introduces FLARE, a novel low-rank self-attention mechanism designed for efficient neural surrogate modeling of partial differential equations (PDEs) on point clouds or unstructured meshes. By leveraging known properties of rank-deficient matrices, the authors employs a linear complexity attention mechanism with a learnable latent bottleneck to further improve accuracy and efficiency on PDE surrogate modeling tasks. Furthermore, FLARE enables end-to-end training on unstructured meshes with one million points without distributed computing or memory offloading and proposes a new a 3D field-prediction benchmark dataset named LPBF.

### Strengths
1、The idea of low-rank self-attention is intersting. This wrok provides a good explanation on architectures design for  the neural operator domain, especially classical works and methods based on Transformer architectures.
2、A new dataset, LPBF, is proposed, making a valuable contribution to the advancement of research in this field.

### Weaknesses
1、The experimental datasets do not include one-dimensional or time-dependent PDE problems. It is recommended to add experiments demonstrating the model’s generality, for example by including shallow-water and reaction-diffusion equations from PDEBench as time-dependent cases.(Takamoto, Makoto, et al. "Pdebench: An extensive benchmark for scientific machine learning." Advances in Neural Information Processing Systems 35 (2022): 1596-1611.)
2、Since Mamba also achieves linear computational complexity, please compare and evaluate the proposed model against Mamba-based architectures, such as MambaNO and LaMO, highlighting their respective advantages and limitations.(Zheng, Jianwei, et al. "Alias-free mamba neural operator." Advances in Neural Information Processing Systems 37 (2024): 52962-52995.)(Tiwari, Karn, Niladri Dutta, and N. M. Krishnan. "Latent Mamba Operator for Partial Differential Equations." arXiv preprint arXiv:2505.19105 (2025).)

### Questions
1、The “Criteria” column in Table 1 lists several evaluation standards, but their specific meanings are not clearly explained. Please provide detailed descriptions for each criterion.

### Soundness
2

### Presentation
3

### Contribution
3
