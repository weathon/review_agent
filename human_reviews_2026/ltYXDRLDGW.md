# Hilbert-Guided Sparse Local Attention

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
The quadratic compute and memory costs of global self-attention severely limit its use in high-resolution images. Local attention reduces complexity by restricting attention to neighborhoods. Block-sparse kernels can further improve the efficiency of local attention, but conventional local attention patterns often fail to deliver significant speedups because tokens within a window are not contiguous in the 1D sequence. This work proposes a novel method for constructing windows and neighborhoods based on the Hilbert curve. Image tokens are first reordered along a Hilbert curve, and windows and neighborhoods are then formed on the reordered 1D sequence. From a block-sparse perspective, this strategy significantly increases block sparsity and can be combined with existing block-sparse kernels to improve the efficiency of 2D local attention. Experiments show that the proposed Hilbert Window Attention and Hilbert Slide Attention can accelerate window attention and slide attention by about $4\times$ and $18\times$, respectively. To assess practicality, the strategy is instantiated as the Hilbert Window Transformer and the Hilbert Neighborhood Transformer, both of which achieve end-to-end speedups with minimal accuracy loss. Overall, combining Hilbert-guided local attention with block-sparse kernels offers a general and practical approach to enhancing the efficiency of 2D local attention for images.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a Hilbert-guided sparse local attention method to address the high computational and memory costs of 2D local attention in high-resolution image processing. It reorders image tokens using the Hilbert curve (which preserves spatial locality) to construct contiguous windows/neighborhoods in the 1D sequence, significantly increasing block sparsity. The method achieves substantial speedups for window attention (HWA) and slide attention (HSA) (up to 4× and 18× respectively).  Experiments verify the approach’s practicality.

### Strengths
- The Hilbert curve-based token reordering maintains 2D spatial locality while making tokens in windows/neighborhoods contiguous in the 1D sequence. This increases the ratio of empty blocks (reducing partial blocks) and maximizes the efficiency of block-sparse kernels, solving the core bottleneck of traditional row-major ordered local attention.
- Experimental results show significant speedups: HWA outperforms dense window attention by up to 4×, and HSA is 18× faster than conventional slide attention. The method also reduces memory consumption drastically.
- HWT and HNT achieve top-1 accuracy on ImageNet-1K (81.0–81.6%) with only a 0.2% drop compared to baseline models (Swin-T, NAT-mini). This balances efficiency and performance, making it suitable for real-world computer vision tasks.

### Weaknesses
This thesis proposes a method for rearranging a sequence of picture patches input to a transformer using Hilbert curves, constructing in a simple way a method that achieves a reduction in the brightness of the attn computation. The paper's experiments are detailed, the narrative is sufficient, and the structure of the lines does not show too many problems. However, from the starting point of the paper, this paper is similar to swin-transformer, both of them carried out technology-based innovation, which is a more skillful design, and I think it is difficult to bring in-depth theoretical insights for the community. But as mentioned, I think this paper is a well-organized and well-structured piece of content, so I think it should be given a weak acceptance.

### Questions
see weakness

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
2

### Summary
This work addresses the inefficiency of applying block-sparse kernels to conventional 2D local attention (e.g., window, sliding window), where tokens are non-contiguous in the 1D sequence. The authors propose reordering image tokens along a Hilbert curve before computing attention. This ​​Hilbert-guided ordering​​ increases the ratio of empty blocks that can be skipped by kernels like FlexAttention, significantly accelerating computation. The method is instantiated as Hilbert Window Attention (HWA) and Hilbert Neighborhood Attention (HNA), which are integrated into end-to-end models (HWT, HNT), achieving substantial speedups with minimal accuracy loss on ImageNet.

### Strengths
1. Novelty: The idea of optimizing sequence order for block-sparse kernels is creative and addresses a key system-level bottleneck.
2. Generality: The method is model-agnostic and can be plugged into existing architectures (e.g., Swin, NAT) via programmable interfaces like FlexAttention.
3. Strong Empirical Results: Comprehensive experiments show significant speedups (e.g., 4x for HWA, 18x for HSA) and memory savings. End-to-end models validate practicality.

### Weaknesses
1. The methods proposed are indeed interesting, but adaptability to non-square inputs or dynamic resolutions is not discussed.
2. Comparisons are mainly against unoptimized baselines. Deeper comparison with highly optimized kernels is needed.

### Questions
Regarding the Weaknesses, further discussion and comparison are recommended. However, due to my limited knowledge in this field, the author may selectively reference my suggestions for better presentation.

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
3

### Summary
The paper presents a method for improving the computational efficiency of local attention mechanisms in vision Transformers by leveraging the Hilbert curve to achieve higher sparsity in attention computation. By mapping 2D image patches onto a 1D sequence using the spatially-coherent Hilbert traversal, contiguous blocks in the sequence more closely match local neighborhoods, resulting in higher block sparsity when applying block-sparse attention kernels. The paper introduces three Hilbert-guided attention patterns—Window Attention, Neighborhood Attention, and Slide Attention — and demonstrates their integration in the proposed Hilbert Window Transformer (HWT) and Hilbert Neighborhood Transformer (HNT). Comprehensive empirical evaluations show notable speedups with minimal accuracy loss on standard benchmarks.

### Strengths
1. This paper provides a systematic and comprehensive framework for different kinds of local attention patterns. The authors identify a persistent gap between theoretical and practical efficiency of sparse/local attention in vision Transformers, especially when block-sparse kernels are applied to row-major sequence orderings. 

2. Although the Hilbert curve is not novel in many computational orders in vision models like Mamba, it is exciting to unify the local attention pattern with it.

3.  The paper contains extensive experiments across diverse input sizes, hardware, and key variables, including window/kernel size, block size, attention type, and end-to-end vs. kernel timings. The experiments robustly show that Hilbert-based reordering produces higher empty block ratios, significantly increases throughput, and reduces memory consumption.

### Weaknesses
1. The application field proposed in the paper is limited. Since Hilbert local attention has very promising potential in accelerating local attention in vision models, the authors only show results on image classifications. There should be other tasks, including both understanding and generation, e.g., object detection, semantic segmentation, image generation, etc. On these tasks, more SOTA efficient attention mechanisms should also be carefully discussed and compared.

### Questions
1. The author-year citation style should be used with brackets.

### Soundness
3

### Presentation
4

### Contribution
4
