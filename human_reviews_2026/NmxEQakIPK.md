# iFlame: Interleaving Full and Linear Attention for Efficient Mesh Generation

- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Mesh generation is a fundamental task in 3D computer vision and graphics, with wide-ranging applications in gaming, virtual reality, and industrial design. While attention-based models have demonstrated remarkable performance in mesh generation, their quadratic computational complexity limits scalability, particularly for high-resolution 3D data. Conversely, linear attention mechanisms offer lower computational costs but often struggle to capture long-range dependencies, resulting in suboptimal outcomes. To address this trade-off, we propose an interleaving autoregressive mesh generation framework that combines the efficiency of linear attention with the expressive power of standard attention mechanisms. To further enhance efficiency and leverage the inherent structure of mesh representations, we integrate this interleaving approach into an hourglass architecture, which significantly boosts efficiency. Our approach reduces training time while achieving performance comparable to pure attention-based models. To improve inference efficiency, we implemented a caching algorithm that almost doubles the speed and reduces the KV cache size by seven-eighths compared to the original Transformer. We evaluate our method on ShapeNet and Objaverse, demonstrating its ability to generate high-quality 3D meshes efficiently. Our results indicate that the proposed interleaving framework effectively balances computational efficiency and generative performance, making it a practical solution for mesh generation tasks. The training takes only 2 days with 4 GPUs on Objaverse.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents iFlame, a novel transformer architecture for efficient autoregressive mesh generation. The core idea is to address the trade-off between the computational cost of standard full attention, which scales quadratically with sequence length, and the representational limitations of linear attention, which struggles with long-range dependencies. The proposed solution is an interleaving mechanism that selectively combines both attention types, aiming to harness the efficiency of linear attention while maintaining the expressive power necessary for high-quality, scalable 3D data generation. The authors focus on demonstrating how this architectural change can lead to more efficient training for generative mesh models.

### Strengths
- The paper is well written and clearly motivated.
- The exploration on linear attention and more efficient training is important. It is pretty impressive to use this amount of compute to make mesh generation work.

### Weaknesses
The primary weakness is that the experiment scale is relatively small compared to related works. The model is only trained on 39K meshes, and the generalization capability is questionable since it is unconditional. While this is orthogonal to the main contribution of the work, it makes the scalability of the method less convincing.

### Questions
- Does the author believe that the linear attention could further scale up to larger models and datasets? What's the potential problems? Most recent mesh generation works still use a standard transformer, I'm curious about the authors' opinion on their choices.
- Have the authors tried performing conditional generation? It's hard to tell if the model can generalize to unseen objects with unconditional generation setting. Adopting a point cloud encoder shouldn't add lots of computation, so the proposed method should also show promising speed up in training.

### Soundness
2

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
3

### Summary
This paper proposes iFlame, an interleaved hourglass Transformer architecture that combines full attention with linear attention for efficient unconditional 3D mesh generation. By interleaving the two attention mechanisms, iFlame significantly improves training and inference efficiency while maintaining generation quality.  It also supports the generation of complex meshes with up to 4,000 faces, significantly outperforming existing methods.

### Strengths
Strong architectural innovation: This approach combines the staggered attention mechanism with an hourglass structure for mesh generation, effectively balancing efficiency and expressiveness.

Significant efficiency improvements: This approach significantly optimizes training time, memory usage, inference speed, and key-value caching.

Strong scalability: This approach successfully increases the maximum number of faces for unconditional mesh generation from 800 to 4,000, with potential for further expansion.

Comprehensive experimental design: This approach is compared with mainstream methods across multiple datasets and metrics, validating its effectiveness and versatility.

High practical value: This approach generates high-quality meshes after only two days of training on four A100 images, demonstrating promising prospects for practical deployment.

### Weaknesses
The baseline comparison is not comprehensive: It does not compare with recently proposed efficient Transformer variants (such as Mamba and RetNet).

Unexplored conditional generation capabilities: It focuses only on unconditional generation and has not verified its performance on conditional generation tasks (e.g., image/point cloud to mesh).

Generation quality still has room for improvement: Despite leading efficiency, it does not comprehensively surpass baselines such as MeshXL in some metrics (such as COV and MMD).

Lack of in-depth analysis of the architectural choice: Why was the "3 linear + 1 full attention" iBlock architecture chosen? Ablation experiments are lacking to support this.

Understanding the ability to model long sequences is not fully verified: While it supports 4,000 faces, its performance on longer sequences (e.g., 10,000+ faces) is not demonstrated.

### Questions
Comparison with Other Efficient Architectures:
Have you compared iFlame with other efficient sequence models (e.g., Mamba, RetNet, or Hyena) in terms of both efficiency and mesh quality?

Generalization to Conditional Generation:
Can iFlame be extended to conditional mesh generation tasks (e.g., image-to-mesh or point cloud-to-mesh)? If so, what adaptations would be necessary?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes iFlame, an auto-regressive mesh generation method with hybrid linear - full attention. The hybrid attention optimizes the generation speed and memory cost, enabling both efficient training and sampling at a much lower cost than existing methods.

### Strengths
1. The paper proposes a hybrid architecture that can optimize training/inference speed while reducing the memory footprint of the KV cache.
2. The proposed method does not hurt much of the results.

### Weaknesses
1. The proposed method is merely replacing some of the full attention layers in the hourglass transformer. It would require further justification on why the architecture is designed this way, and why it is good for mesh generation.

2. As highlighted that linear attention "achieving lower performance compared to attention-based architectures" in line 58, it is important to validate the effectiveness of this observation on auto-regressive mesh generation. 

3. What is face accuracy in Figure 1? I don’t think a truncated histogram is a good visualization method, as it can lead to misunderstandings about the multiplicative relationships between data points.

### Questions
See weakness part.

### Soundness
2

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
2

### Summary
This paper introduces iFlame, a Transformer-based architecture for high-resolution, unconditional 3D mesh generation. It tackles the trade-off between the expressiveness of full-attention and the efficiency of linear attention. The core idea is to design iBlocks, which interleave multiple linear attention layers with a single full attention layer to balance accuracy and cost.These iBlocks are further integrated into a multi-scale hourglass architecture that exploits the hierarchical structure of mesh data to shorten sequence lengths and enhance efficiency.

### Strengths
The combination of an interleaved attention mechanism within an hourglass structure is a conceptually novel design. It addresses the efficiency problem at both a block-level and global-level scale.
2 . Quantitative results demonstrate substantial gains in computational efficiency, training cost, and memory footprint, highlighting the method’s potential for practical deployment.

### Weaknesses
1 . The paper lacks direct numerical comparisons of inference latency and memory usage with external baselines (e.g.,MeshGPT, MeshXL). Including such results would strengthen claims about deployment efficiency.
2 . The fixed 3:1 ratio of linear to full attention layers is a key design choice, yet its empirical justification or sensitivity analysis is missing. Exploring alternative ratios (e.g., 1:1 or 7:1) would clarify its effect on the trade-off between performance and efficiency.

### Questions
1 . Could the authors further clarify the rationale for selecting the 3:1 ratio of linear to full attention layers within the iBlock? Was this choice guided by theoretical considerations, empirical tuning, or prior design heuristics?
2 . In Figure 5, the novelty analysis based on Chamfer distance is interesting. Could the authors provide qualitative examples to illustrate whether high-distance samples correspond to meaningful creative variations or to geometric artifacts?

### Soundness
3

### Presentation
3

### Contribution
2
