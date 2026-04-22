# Beyond Pixels: Efficient Dataset Distillation via Sparse Gaussian Representation

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4, 4

## Abstract
Dataset distillation has emerged as a promising paradigm that synthesizes compact, informative datasets capable of retaining the knowledge of large-scale counterparts, thereby addressing the substantial computational and storage burdens of modern model training. Conventional approaches typically rely on dense pixel-level representations, which introduce redundancy and are difficult to scale up. In this work, we propose GSDD, a novel and efficient sparse representation for dataset distillation based on 2D Gaussians. Instead of representing all pixels equally, GSDD encodes critical discriminative information in a distilled image using only a small number of Gaussian primitives. This sparse representation could improve dataset diversity under the same storage budget, enhancing coverage of difficult samples and boosting distillation performance. To ensure both efficiency and scalability, we adapt CUDA-based splatting operators for parallel inference and training, enabling high-quality rendering with minimal computational and memory overhead. Our method is simple yet effective, broadly applicable to different distillation pipelines, and highly scalable. Experiments show that GSDD achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet subsets, while remaining highly efficient encoding and decoding cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes GSDD, a dataset distillation method using a sparse 2D Gaussian representation to replace dense pixels and costly implicit neural representations, aiming to enhance efficiency and scalability. According to the reported results, it achieves notable speed and memory gains over INR-based baselines (e.g., DDiF) while maintaining comparable or better performance across different DD algorithms and architectures.

### Strengths
1. This is the first work to introduce a parametric Gaussian framework into the field of dataset distillation.
2. The method employs customized CUDA operators to significantly improve computational efficiency.

### Weaknesses
**Major:**

1. The authors emphasize in the introduction and motivation that DDiF struggles with larger and higher-resolution datasets, yet the experiments in this paper do not substantiate this claim. Specifically, DDiF is evaluated on the ImageNet subset with a resolution of 256×256, while GSDD experiments are limited to 128×128 or even 32×32. To convincingly demonstrate GSDD’s superior generalization, it should at least show improvements on datasets of comparable scale and resolution.
2. The method section provides extensive background on parametric Gaussian modeling but lacks theoretical justification for why Gaussian functions are inherently more effective than INRs. There is no formal reasoning or analysis showing that 2D Gaussian bases offer intrinsic advantages over other differentiable bases (e.g., wavelet or trigonometric) for gradient matching in dataset distillation. As a result, the choice of Gaussian appears heuristic and engineering-driven rather than grounded in theoretical insight.
3. The paper does not theoretically formalize the superiority of this representation. Most arguments focus on engineering aspects such as CUDA-based rasterization being faster than neural network querying, rather than proving that sparse explicit primitives are fundamentally more suitable for encoding discriminative knowledge than implicit continuous functions.
4. The paper lacks sensitivity or ablation studies on key hyperparameters, particularly the number of Gaussian primitives per image and their initialization strategy. Since these parameters critically affect performance, the absence of such analysis makes it difficult to assess the robustness and practical applicability of the method.
5. The paper claims to evaluate cross-architecture transfer from ConvNet to ResNet and VGG. However, Table 4 only shows the performance of different DD methods on ImageNet subsets and does not present any cross-architecture results, so the claimed generalization across architectures is not supported.

**Minor:**

1. The notation $R$ in Equation (5) is undefined.
2. The reference “Sequential subset matching for dataset distillation” has inconsistent author formatting compared to other entries.

### Questions
See weakness.

### Soundness
3

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
This paper proposes GSDD to address the redundancy and poor scalability issues inherent in previous dataset distillation methods that rely on dense pixel-level representations. Specifically, GSDD introduces a sparse representation based on 2D Gaussian primitives to encode distilled images. Each Gaussian captures region-level discriminative information with only a few parameters optimization. This proposed Gaussian-based representation reduces storage overhead, increases dataset diversity and improves optimization stability. Also, to ensure efficiency and scalability, this paper implements a CUDA-based differentiable rasterizer for parallel rendering of multiple distilled images. Experiments demonstrate that GSDD achieves state-of-the-art performance on CIFAR-10, CIFAR-100, and ImageNet subsets with efficiency gains.

### Strengths
1. The paper is well written with clear and informative figures and well-motivated for essential efficiency issues in dataset distillation field.
2. The proposed GSDD is simple yet effective, with a novel and intuitive idea. The transformation from per-pixel to per-region representation is well motivated and directly addresses the inefficiency of dense pixel-level encoding.
3. The proposed GSDD can be integrated into other DD methods while improving the efficiency and performance.
4. The performance of the proposed method is good with less computational cost.

### Weaknesses
1. [major] The efficiency of the proposed method relies heavily on CUDA-based parallel rasterizer, which raises concerns about portability and reproducibility. Different GPU hardware-specific optimizations can lead to slightly different acceleration behaviors, the performance and results may vary between different GPU architectures or operator implementations.
2. [major] Since distilled images are generated from sets of Gaussian primitives rather than explicit pixels, this representation may have weaker semantic visualization and interpretability, potentially limiting the model generalization ability across different architectures. As shown in Tables 4 and 10, GSDD performance improvement across different architectures is relatively limited compared to performance on the same architecture. Furthermore, this paper only reports the average results for each architecture without providing detailed comparisons per architecture, making the cross-architecture performance results less convincing and verifiable.
3. [minor] While the proposed method demonstrates efficiency and scalability, experiments have primarily focused on relatively low-resolution datasets, such as subsets of CIFAR and ImageNet (low-resolution version). To better verify whether the method remains effective on high-resolution or more complex datasets, it could be extended to ImageNet full size with full resolution.

### Questions
1. As mentioned in weakness major 1, it would be helpful if the authors could provide some explanation or empirical evidence regarding how consistent the results are across different GPU architectures.
2. For weakness major 2, it might be helpful if the authors could provide further explanation or analysis on how the Gaussian-based representation affects generalization across different architectures.
3. It might be helpful if the authors could provide some discussion on the scalability of the proposed method to higher-resolution datasets, e.g., full size and resolution ImageNet-1K. Also, could the same Gaussian-based representation be extended to detection or segmentation tasks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This study points out that existing dataset distillation methods using dense representations, which is hard to capture the relative importance of pixels effectively, leading to redundancy. To achieve efficient sparse representations, this study proposes Gaussian splatting dataset distillation (GSDD), which utilizes Gaussian splatting to parameterize the distilled synthetic images. Through the experimental results, GSDD shows the efficiency and effectiveness improvement.

### Strengths
1. To my knowledge, this is the first work that employs Gaussian splatting in the dataset distillation. Gaussian splatting is widely researched and utilized across diverse fields, thus possessing high potential for extension.

2. The detailed analysis of the Gaussian representation described in Section 3.3 is highly beneficial as it aids in a more thorough understanding of the GSDD. In particular, the significant variation in performance depending on large opacity and size is of particular interest.

3. GSDD achieves strong performance with significant gap on some benchmark datasets.

### Weaknesses
1. There is a lack of clear justification for the motivation to apply Gaussian splatting to dataset distillation. I believe the current manuscript focuses more on methodologies for efficiently using Gaussian splatting in Dataset distillation than on unique characteristics of Gaussian splatting compared to existing parameterization methods. The issues of redundancy and computational overhead highlighted by the authors as problems in prior research have already been noted in several previous studies[1,2,3,4]. Specifically, while the paper highlights efficiency concerns regarding time and memory for DDiF[5] (the core baseline adopted in this study), this does not sufficiently justify shifting the framework to Gaussian splatting without improving INR-based parameterization. Therefore, an in-depth analysis is required to present new limitations of existing prior research and demonstrate how Gaussian splatting's inherent characteristics can address these.

2. There is insufficient evidence to support the claim that GSDD achieves higher performance than prior research. The contributions of this study are believed to be the first application of Gaussian splatting in the field of dataset distillation and the enhancement of efficiency through various techniques. While this approach achieves high efficiency in terms of time and memory, the lack of evidence and explanation as to why it outperforms prior research makes it difficult to accept. For instance, DDiF argued that the introduction of INR yields higher expressiveness despite using a smaller budget, through theoretical analysis and reconstruction experiments. Furthermore, experiments with a fixed number of decoded instances supported the notion that DDiF's high performance stems from its high expressiveness and diversity. Similarly, a deeper analysis is required to understand the mechanism through which the proposed idea, GSDD, achieves its high performance.

3. There is no analysis of whether GSDD can be applied across diverse application scenarios and achieve high performance. Parameterization methods effective only in specific application area have limitations in their applicability. Several prior studies report experimental results on corrupted datasets[2,3,5] and cross-resolution[5] to evaluate the out-of-domain generalization of each methods. Furthermore, as the primary baseline DDiF is a unified framework for grid-based data across various modalities, it provided performance comparison results across image, video, audio, and 3D voxel datasets. Demonstrating GSDD's superior performance across diverse application scenarios would significantly enhance the impact and applicability of this research.

[1] A Comprehensive Survey of Dataset Distillation

[2] Frequency Domain-based Dataset Distillation

[3] Sparse Parameterization for Epitomic Dataset Distillation

[4] Generalizing Dataset Distillation via Deep Generative Prior

[5] Distilling Dataset into Neural Field

### Questions
1. I am curious about the performance of Gaussian splatting-based parameterization without applying the various techniques for improving efficiency (Section 3.2). I am also curious about the performance improvement when adding each technique, extending Table 3.

2. I am also curious whether GSDD also demonstrates high performance and efficiency for image datasets with resolutions greater than 128.

3. Recently, research into soft labels has also been actively explored as an alternative to one-hot labels[6,7,8]. While this work defined by one-hot label encoding (Line 98), I would be interested to see the performance of GSDD when soft label is applied

4. Quantization using bf16 directly influences the final image per class calculation (Line 402–408), so it affects diversity and playing a key role in GSDD performance (Table 3). However, this technique is not limited to GSDD and can also be applied to existing parameterization methods. Therefore, I would like to know the performance comparison when quantization is applied to existing parameterization methods.

5. I understood that each Gaussian primitive contributes to a single distilled image. I am curious whether this could be extended to allow a Gaussian primitive to contribute to multiple distilled images. If possible, this could further enhance efficiency.

[6] A Label is Worth A Thousand Images in Dataset Distillation

[7] GIFT: Unlocking Full Potential of Labels in Distilled Dataset at Near-zero Cost

[8] Heavy Labels Out! Dataset Distillation with Label Space Lightening

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
4

### Summary
This paper tackles the problem of data parameterization in dataset distillation (DD). It argues, correctly, that conventional dense pixel-grid representations are redundant, inefficient, and scale poorly. The authors propose GSDD (Gaussian Splatting Dataset Distillation), a novel and efficient parameterization that represents each distilled image as a sparse set of 2D Gaussian primitives. Each Gaussian is defined by 9 parameters (position, shape, color, opacity). These primitives are rendered into images using a highly parallelized, differentiable CUDA-based rasterizer. The central claims are that this sparse representation (1) is more storage-efficient, allowing for greater dataset diversity (more images per class) under a fixed budget , (2) enables a smoother optimization landscape for faster convergence , and (3) is computationally efficient. The method is evaluated on CIFAR-10, CIFAR-100, and ImageNet subsets , where it is shown to achieve state-of-the-art performance and significantly outperform the INR-based DDiF in speed and memory usage.

### Strengths
1. Novel and Clever Parameterization: The core idea of using a sparse set of 2D Gaussians to represent distilled images is highly original in the DD context. It elegantly sidesteps the high-frequency noise issues of pixel optimization (Fig 4) and the computational bottleneck of INR's per-pixel querying.

2. Efficiency vs. INRs: The paper provides compelling quantitative evidence (Figure 5) that GSDD is dramatically faster (both forward and backward) and more memory-efficient than the state-of-the-art functional parameterization, DDiF . This is a strong practical advantage.

3. Performance (on tested benchmarks): The method achieves state-of-the-art results on all tested benchmarks (CIFAR-10, CIFAR-100, and ImageNet subsets), outperforming numerous baselines across various IPC settings.

### Weaknesses
1. Critical Omission of ImageNet-1K: The most significant weakness is the failure to evaluate on full ImageNet-1K. The paper is titled "Efficient Dataset Distillation" and repeatedly claims scalability. However, it avoids the standard large-scale benchmark where scalability is truly tested. Recent SOTA methods in scalable DD (e.g., SRe2L, RDED, etc.) all report 1K results. Without this comparison, the central claim of scalability is unsubstantiated. The results on "ImageNet subsets"  are insufficient.

2. Confounding Initialization: The initialization procedure  is a major methodological flaw. The model is "pre-trained" to match real images via MSE loss. This raises two problems:
   - Performance Confound: How much of the final SOTA performance is simply due to this high-fidelity, real-data-based "warm start" rather than the superiority of the GSDD parameterization within the distillation process?
   - Privacy Contradiction: This initialization method directly leaks data from the original dataset, which contradicts the paper's (and the field's) stated motivations for privacy preservation. A method that requires fitting to real images cannot be straightforwardly used in privacy-sensitive scenarios.

3. Hyperparameter Complexity: The method introduces a critical new hyperparameter trade-off: the number of Gaussians per image ($M$) vs. the number of Gaussian Images Per Class (GPC). While the paper explores this trade-off (Fig 3c, 3d), it's unclear how to optimally set these values for a new dataset and budget, making the method less of a simple drop-in replacement than advertised.

### Questions
1. ImageNet-1K: The most pressing question is the lack of full ImageNet-1K experiments. Given the paper's focus on efficiency and scalability, why was this benchmark omitted? Can you provide results for GSDD on ImageNet-1K, comparing it to scalable SOTA methods like SRe2L and RDED?

2. Initialization Ablation: What is the performance of GSDD if the Gaussians are initialized randomly (e.g., random positions/colors, small isotropic covariances) instead of being pre-fit to real images? This is a crucial ablation to understand the true source of the performance gains and to validate the method's feasibility for privacy-preserving applications.

3. Privacy: How do you reconcile the claim of supporting "privacy-preserving research" with an initialization method  that explicitly fits the synthetic data to real data samples?

4. Gaussian Escape: In your ablation (Table 3), the "w/o boundary" setting shows a performance drop. Is this drop due to removing the loss term (Eq 10) or the hard clipping (Eq 11)? What happens if you only use the loss, or only the clipping?

I am willing to raise my score if the authors can provide a comprehensive comparison on ImageNet-1K against relevant SOTA methods (like SRe2L and RDED) and address the confounding effect of their real-image-based initialization in their rebuttal.

### Soundness
2

### Presentation
3

### Contribution
2
