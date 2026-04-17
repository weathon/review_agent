# MoGA: Mixture-of-Groups Attention for End-to-End Long Video Generation

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Long video generation with Diffusion Transformers (DiTs) is bottlenecked by the quadratic scaling of full attention with sequence length. Since attention is highly redundant, outputs are dominated by a small subset of query–key pairs. Existing sparse methods rely on blockwise coarse estimation, whose accuracy–efficiency trade-offs are constrained by block size. This paper introduces Mixture-of-Groups Attention (MoGA), an efficient sparse attention that uses a lightweight, learnable token router to precisely match tokens without blockwise estimation. Through semantic-aware routing, MoGA enables effective long-range interactions. As a kernel-free method, MoGA integrates seamlessly with modern attention stacks, including FlashAttention and sequence parallelism. Building on MoGA, we develop an efficient long video generation model that end-to-end produces minute-level, multi-shot, 480p videos at 24 fps, with a context length of approximately 580k. Comprehensive experiments on various video generation tasks validate the effectiveness of our approach. Project website: https://jiawn-creator.github.io/mixture-of-groups-attention/ .

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Mixture-of-Groups Attention (MoGA), a sparse attention mechanism designed to alleviate the quadratic complexity of Diffusion Transformers in long video generation. MoGA employs a lightweight, learnable token router to assign tokens into semantically coherent groups and performs full attention within each group, achieving fine-grained dynamic sparsification. Coupled with local spatiotemporal window attention (STGA) and shot-level textual conditioning, the proposed framework can end-to-end generate 480p, 24 fps, minute-long multi-shot videos. The authors also describe a multi-stage data curation pipeline and evaluate MoGA across single-shot, multi-shot, and ultra-long video scenarios.

### Strengths
* Originality: The MoGA design brings a Mixture-of-Experts–style router into sparse attention for video generation, avoiding coarse block-level heuristics that impede prior methods. Its compatibility with FlashAttention and sequence parallelism underscores solid systems innovation.
* Quality: Experiments cover single-shot short videos, multi-shot videos, and long-form generation, measured by VBench metrics (subject/background consistency, motion smoothness, aesthetics, etc.) plus Cross-Shot CLIP/DINO. MoGA outperforms both sparse and full-attention baselines, and the ablations (group count, STGA interplay, balancing loss) lend credibility.
* Clarity: The manuscript is well-structured, with clear figures contrasting full, block-sparse, and MoGA attention. Pseudocode and complexity analysis clarify implementation.
* Significance: Delivering stable 580k-token, minute-long, multi-shot videos is a notable leap for open-source systems, and MoGA’s design could generalize to other long-sequence tasks.

### Weaknesses
* Router design details: While a group balancing loss is introduced, the paper does not quantify how router collapse affects performance or explore router hyperparameters (dimension, temperature, regularization) beyond group count.
* Compute/resource reporting: PFLOP estimates and speedup claims are provided, yet concrete wall-clock measurements (training/inference time, memory usage) on specific hardware are missing, hindering reproducibility assessments.
* Data pipeline reproducibility: The multi-stage filtering/captioning pipeline depends on various proprietary tools (VQA/OCR/AutoShot), but code or parameter details are sparse. More statistics on filtered dataset size/quality would help.
* Limited multimodal control evaluation: Although shot-level textual conditioning is highlighted, there is little quantitative evidence (e.g., prompt adherence metrics or user studies) demonstrating textual control effectiveness.

### Questions
* Effectiveness of router: If the router only classify the tokens based on itself, would it be less sensitvie to global related information? Is there any detailed visusalization about router?
* Router stability: Did you observe router collapse or gradient instability early in training? Is the group balancing coefficient α = 0.1 robust across model sizes? How do performance and sparsity change if α varies?
* Choosing the number of groups: Since too few or too many groups hurt consistency, have you explored dynamically adjusting group counts (e.g., by sequence length or diffusion timestep)? Could this enhance performance or stability?
* Multimodal control evaluation: Can you provide quantitative metrics (e.g., text-to-video CLIP scores, prompt coverage, human preference studies) to support the claimed shot-level textual orchestration?
* Resource requirements: Please detail the hardware setup (GPU type/count), training time, and memory footprint for MoGA at M = 20 on minute-long videos so the community can gauge reproducibility.
* Broader applicability: Have you experimented with MoGA on non-video long-sequence tasks (e.g., text generation, 3D scene modeling)? If not, what considerations would be necessary for such extensions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Mixture-of-Groups Attention (MoGA), a novel and efficient sparse attention mechanism designed to overcome the computational bottleneck of generating long, high-resolution videos with diffusion transformers. Instead of relying on traditional block-wise estimation for sparsity, MoGA employs a lightweight, learnable "token router," inspired by Mixture-of-Experts, to assign each token to a specific group based on its semantics. Standard self-attention is then performed independently within each group, drastically reducing computational complexity while enabling effective long-range interactions. This global attention mechanism is complemented by a local Spatial-Temporal Group Attention (STGA) to ensure local continuity, and the model is trained on a custom-built data pipeline that produces minute-long, multi-shot video samples.

### Strengths
The paper's primary strength lies in its elegant and highly effective solution to the long-context problem. By replacing coarse block-level scoring with a precise, end-to-end token router, MoGA represents a conceptual advance over prior sparse attention methods. The approach is remarkably practical, as it is kernel-free and seamlessly integrates with existing high-performance technologies like FlashAttention and sequence parallelism. The experimental results are state-of-the-art and convincingly demonstrate the model's ability to generate coherent, minute-long, multi-shot videos, supported by comprehensive quantitative metrics and strong ablation studies that validate the design choices.

### Weaknesses
One potential point of discussion is the method's reliance on a powerful, pre-existing base model (Wan2.1) for fine-tuning, which makes it slightly difficult to isolate the gains of MoGA from the inherent capabilities of the foundation model. Additionally, the paper introduces an impressive and complex data pipeline for creating multi-shot training samples; the importance of this high-quality, specialized data to the final result is significant and could be considered a major contribution in its own right, perhaps underemphasized in the context of the attention mechanism.

### Questions
The token router is shown to learn meaningful semantic groupings in an unsupervised manner. Could the authors elaborate on the nature of these learned groups? For instance, do certain groups consistently specialize in specific visual concepts (e.g., one group for faces, another for backgrounds, another for dynamic motion), or are the groupings more abstract and context-dependent? Understanding this could provide deeper insights into the model's internal workings

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduce Mixture-of-Groups Attention (MoGA), an efficient sparse attention that uses a lightweight learnable token router to precisely match tokens without blockwise estimation. By semantics-aware routing, MoGA enables effective long-range interactions.

### Strengths
see Summary

### Weaknesses
The authors should provide between 20 to 50 video samples to better demonstrate the capabilities and limitations of their method. Additionally, for each prompt, it would be beneficial to include comparisons with other state-of-the-art methods. Ideally, there should be 3 to 5 comparison methods with corresponding videos for each prompt.

The authors should provide detailed results of the user study, including statistical analysis and user feedback. This will help in understanding how the proposed method performs in terms of user satisfaction and practical usability.

The paper only presents sub-scores from VBench, which provides a limited view of the overall performance.

The caption in Figure 3 should be more detailed. Otherwise, it is difficult to guess the meaning of the figure.

Why are there only experimental results for 1.3B in Table 2, but not for 14B? Similarly, Table 1 and Table 3 also lack these results.

The idea is very simple and clear, and it is easy to understand. However, the main difficulty and contribution may lie in the engineering implementation of the code. Since the author has not provided open-source code or code in the supplementary materials, I strongly suggest that the author provide the real Python code functions for Algorithm 1 for me to review.

Figure 6 only compares the FLOPs, but I believe that the actual inference time is much more important. This conceals two issues that I am very interested in:
1. In fact, the computational cost of Spatial-Temporal Group Attn is already quite large, and that of Mixture-of-Groups Attn is also significant. I think that under the scenario of short videos, the impact of these two designs on speed is minimal.
2. From an engineering implementation perspective, Spatial-Temporal Group Attn and Mixture-of-Groups Attn cannot be parallelized. Therefore, they will actually be slower due to serial computation.

### Questions
If the author's response is satisfactory to me, I would be willing to maintain a positive score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Mixture-of-Groups Attention (MoGA), a new sparse attention mechanism designed to improve computational efficiency and scalability for end-to-end long video generation with diffusion transformers.  MoGA employs a lightweight learnable token router that dynamically assigns tokens into semantically coherent groups. Using MoGA, the authors present a video generation model that can produce minute-level, multi-shot, 480p videos at 24 FPS with a context length up to 580 K tokens. Experiments across multiple baselines (Wan2.1, MMDiT, EchoShot, IC-LoRA) demonstrate improved visual quality, subject and background consistency, and substantial reductions in FLOPs.

### Strengths
- The introduction of token-level routing as a “mixture-of-groups” attention mechanism is novel and intuitive. It replaces coarse block sparsity with a more fine-grained, data-driven grouping that potentially generalizes better across tasks.
- The paper convincingly shows MoGA’s ability to reduce attention complexity from O(N^2) to approximately O(N^2/M) while preserving quality. The claimed 1.7× training/inference speedup .
- The design is compatible with FlashAttention, sequence parallelism, and existing DiT frameworks.
- Across metrics such as subject/background consistency, aesthetic quality, and cross-shot CLIP/DINO similarity, MoGA consistently outperforms baselines, including full-attention models.

### Weaknesses
### Major
- My major concern is the novelty compared to the previous works. While MoGA's token router is inspired my MoE, and MoBA, it should be deeply analyzed the distinction between it and these works. The paper should clarify how fundamentally MoGA differs from the existing routing based methods, beyond the token level analysis.

- The benchmarks are built on top of the models like wan2.1/MMDiT and tested on internal datasets, where the authors integrate their MoGA module into the original architectures and train/fine-tune them on their internal dataset. Therefore, it is uncertain whether MoGA’s advantages generalize to other diffusion transformer backbones or unseen datasets.  While the paper provides an anonymous demo link, it lacks details about code or data release, which is critical for an academic paper submission claiming major efficiency improvements, also reproducibility.

### Others
- It could be better if the paper could provide formal analysis of stability, or the distribution of token assignments. "lightweight router" would be thoretically explained.

- It develops a multi-stage pipeline, which may be overly complicated (with steps like AutoShot, OCR, and LLM captioning),  should provide a experimental study on how much these steps actually improve performance, or a simpler baseline with a simplified pipeline. It would make the contribution clearer and more convincing.

### Questions
1. How is Mixture-of-Groups Attention fundamentally different from Mixture-of-Experts (MoE) or Mixture-of-Block Attention (MoBA/VMoBA) beyond operating tokens? Can you clarify what new insights or mechanisms MoGA introduces that are not already explored in MoBA (Lu et al., 2025) or Radial Attention (Li et al., 2025)?

2.What ensures that the lightweight router produces semantically meaningful groupings instead of arbitrary clusters? Have you analyzed how stable the routing assignments are across training iterations or input perturbations? Maybe could provide quantitative evidence (e.g., token entropy or similarity distributions) to support that MoGA learns meaningful semantic partitioning?

3. Have you tested MoGA on other architectures or datasets (e.g., Open-Sora, Pika, or VideoCrafter) to show generality beyond Wan/MMDiT? Do you expect MoGA to benefit smaller or lower-resolution models similarly, or is the gain limited to large-scale settings? It claims 1.7× speedup and reduced FLOPs; can you provide absolute runtime and memory usage numbers compared to full attention on the same hardware?

4. Please explain on reproducibility, e.g. release of the code, or ease of reimplementation. How can the efficiency be verified without access to the internal dataset and pipeline.

### Soundness
3

### Presentation
3

### Contribution
3
