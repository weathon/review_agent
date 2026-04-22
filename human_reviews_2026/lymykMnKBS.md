# SONA: Learning Conditional, Unconditional, and Matching-Aware Discriminator

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Deep generative models have made significant advances in generating complex content, yet conditional generation remains a fundamental challenge. Existing conditional generative adversarial networks often struggle to balance the dual objectives of assessing authenticity and conditional alignment of input samples within their conditional discriminators. To address this, we propose a novel discriminator design that integrates three key capabilities: unconditional discrimination, matching-aware supervision to enhance alignment sensitivity, and adaptive weighting to dynamically balance all objectives. Specifically, we introduce Sum of Naturalness and Alignment (SONA), which employs separate projections for naturalness (authenticity) and alignment in the final layer with an inductive bias, supported by dedicated objective functions and an adaptive weighting mechanism. Extensive experiments on class-conditional generation tasks show that SONA achieves superior sample quality and conditional alignment compared to state-of-the-art methods. Furthermore, we demonstrate its effectiveness in text-to-image generation, confirming the versatility and robustness of our approach.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SONA (Sum of Naturalness and Alignment), a new discriminator framework for conditional generative adversarial networks (GANs). The motivation stems from a persistent issue in conditional generation: balancing sample authenticity (unconditional discrimination) with conditional alignment (matching the conditioning signal such as class labels or text prompts). Traditional discriminators, such as those in AC-GAN or projection-based GANs (PD-GAN), handle these two aspects in an intertwined manner, often requiring careful weighting or suffering from limited sensitivity to mismatched conditions.

SONA explicitly decomposes the discriminator’s learning objective into three coordinated components:

1. Unconditional discrimination, focusing on the authenticity of generated samples via the Sliced Wasserstein perspective;

2. Matching-aware discrimination, improving alignment sensitivity by introducing mismatched (negative) samples through a Bradley–Terry pairwise comparison formulation;

3. Adaptive weighting, dynamically balancing the above objectives through learnable coefficients constrained by normalization.

The proposed design uses orthogonal projections of discriminator features to separately assess naturalness and alignment, encoding an inductive bias that disentangles these two factors. The theoretical analysis establishes that each component corresponds to optimizing distinct divergences (e.g., log-probability gaps between conditional and unconditional distributions).

Empirically, SONA achieves consistent gains over state-of-the-art baselines (ReACGAN, ContraGAN, PD-GAN) on CIFAR10, TinyImageNet, and ImageNet, improving both FID and Inception Score across multiple backbones (BigGAN, StyleGAN2). The method further extends effectively to text-to-image generation tasks (CUB, COCO, CC12M) when integrated with GALIP, achieving better FID and comparable CLIP alignment. Ablation studies confirm that orthogonal modeling and mismatching-aware loss significantly contribute to performance improvements.

Overall, SONA presents a unified, theoretically sound, and empirically validated framework for improving conditional discriminators, offering both conceptual clarity and practical robustness in modern GAN training.

### Strengths
1. Novel conceptual integration: The paper elegantly unifies multiple important discriminator desiderata—unconditional, conditional, and mismatching-aware discrimination—into one coherent framework. The separation of naturalness and alignment projections via orthogonality is an intuitively appealing and theoretically well-justified idea.

2. Sound theoretical grounding: Each component of SONA is supported by propositions that connect its loss functions to meaningful probabilistic interpretations (e.g., conditional divergence minimization and log probability gap maximization). The use of the Bradley–Terry model for conditional discrimination is a clever adaptation rarely seen in GAN literature.

3. Strong empirical performance: The method demonstrates clear and consistent improvements across diverse datasets and architectures. Especially on ImageNet (FID=6.14, IS=140.14), SONA substantially outperforms prior methods, showing robustness at scale. The inclusion of text-to-image benchmarks further validates its generality.

4. Comprehensive experimentation: The authors conduct detailed comparisons with strong baselines (PD-GAN, ReACGAN, ContraGAN), extensive ablations (adaptive weighting, orthogonal projection, mismatch loss), and visualization using toy datasets (MoG). This provides convincing evidence of both quantitative and qualitative improvements.

5. Reproducibility and transparency: The paper references open-source implementations (StudioGAN, BigGAN-PyTorch, GALIP) and commits to code release. Appendices include detailed algorithmic descriptions and theoretical proofs.

6. Practical relevance: Since projection-based discriminators remain a backbone of modern GANs, SONA’s design can be easily adopted without significant architectural overhead, making it both theoretically elegant and engineering-friendly.

Overall, the paper succeeds in addressing a long-standing problem (balancing authenticity and alignment) in a way that is theoretically principled, empirically validated, and broadly applicable across generative modalities.

### Weaknesses
1. Complexity of formulation: While conceptually clean, the proposed method involves multiple interrelated objectives (SAN, BT-COND, BT-MM, adaptive weighting), which may make training more cumbersome in practice. Although the authors report stability, more discussion on convergence behavior and computational overhead (especially in large-scale settings) would strengthen the paper.

2. Limited comparison with diffusion-based models: The field has largely shifted toward diffusion and rectified flow architectures for conditional generation (e.g., SDXL, RF Transformers). While SONA’s GAN focus is clear, the paper could better contextualize its relevance in the broader landscape of generative modeling in 2025–2026, potentially by discussing hybrid or complementary uses.

3. Ablation depth: While ablation studies isolate major components, finer-grained analysis (e.g., sensitivity of adaptive weighting parameters or orthogonality constraint strength) is missing. It would also be useful to examine cases where SONA may underperform or produce mode collapse, to better understand its limitations.

4. Clarity and accessibility: The theoretical derivations, especially those involving the Sliced Wasserstein and Bradley-Terry frameworks, are mathematically dense and may be challenging for readers not familiar with those formulations. A more intuitive explanation or schematic of the optimization flow could enhance accessibility.

### Questions
1. Computational overhead: How much additional computation or memory does SONA introduce compared to standard PD-GANs, particularly due to pairwise BT loss sampling and adaptive weighting? Is the training speed comparable?

2. Stability and convergence: Theoretical results show desirable properties at equilibrium, but how stable is the actual training dynamic? Did the authors observe any oscillations or mode collapse before convergence?

3. Generalizability to diffusion models: Since diffusion-based models dominate text-to-image generation, can SONA’s discriminator concept be repurposed as a “critic” for conditional diffusion or consistency models?

4. Scaling to complex conditions: How does SONA perform when the conditional space is continuous or high-dimensional (e.g., embeddings or attributes rather than discrete labels)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses a key limitation in conditional GANs, which often struggle to jointly ensure image authenticity and alignment with conditioning inputs. To mitigate this, they propose SONA, a novel discriminator architecture that integrates unconditional discrimination, matching-aware supervision, and adaptive weighting. Experiments on class-conditional and text-to-image generation tasks demonstrate that SONA achieves superior sample quality and conditional alignment compared to existing methods.

### Strengths
* The paper includes comprehensive experiments conducted across diverse datasets and scales, supporting the generality of the proposed method.
* The paper includes an ablation study that justifies the contribution of each model component. It would be helpful to include visualizations to show how each component affects output quality.
* The writing is clear and well-organized.

### Weaknesses
* The relevance of the addressed problem is uncertain in the current context, where most major advances in generative modeling stem from diffusion models rather than GANs. Moreover, the compared baselines are relatively outdated GAN architectures, making the reported improvements less meaningful. It remains unclear whether the proposed approach would offer advantages over using state-of-the-art diffusion-based methods, which limits the overall impact of the work.
* The paper lacks baseline comparisons with modern diffusion models, which currently outperform GANs on most conditional generation benchmarks.
* Missing qualitative results. Given that the task involves image generation, it should include visual comparisons illustrating how the proposed method improves sample quality and conditional alignment relative to existing approaches.

### Questions
Mentioned in the weakness section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SONA, a novel discriminator design for conditional Generative Adversarial Networks (GANs). It addresses the core challenge of balancing sample authenticity (naturalness) and conditional alignment by decomposing the discriminator's role into three integrated capabilities: robust unconditional discrimination, enhanced sensitivity to conditional alignment via negative samples, and dynamic adaptive weighting of these objectives. The method employs separate, orthogonal projections for naturalness and alignment, supported by a combination of Slicing Adversarial Network (SAN) and Bradley-Terry (BT) model-based losses. Extensive experiments on class-conditional (CIFAR-10, TinyImageNet, ImageNet) and text-to-image generation tasks demonstrate that SONA outperforms state-of-the-art methods in both sample quality and conditional alignment.

### Strengths
+ This paper proposes a novel discriminator architecture that explicitly decouples sample naturalness learning and conditional alignment through orthogonal projection. The combination of SAN and BT model objectives is innovative.

+ The paper is clearly written, explaining the proposed concepts in a simple and understandable way. The structure is logically rigorous, and the diagrams effectively complement the textual explanations.

+ The method validates its improvements in both standard and challenging benchmarks, and its applicability in text-to-image generation highlights its broad application potential.

### Weaknesses
- Lacks comparison with a simple dual-projection design, failing to demonstrate the necessity of the orthogonal constraint and potentially adding unnecessary complexity.
- Training speed is consistently slower than the simplest baseline (PD-GAN), without discussion of whether the performance gain justifies this overhead.
- The core theory relies on the strong, often unrealistic assumption of "perfect distribution matching," with no discussion of its validity in real-world training.
- The absence of a direct comparison with contemporary hybrid discriminators like ADC-GAN, which also integrate unconditional and conditional discrimination, makes it difficult to fully assess the innovative advantages.

### Questions
The paper shows that using a learnable ω_y for text conditioning degraded performance. Could the authors speculate more on the reasons for this? For instance, does the frozen CLIP embedding provide a more stable and semantically meaningful direction that is difficult to improve upon with a simple learned transformation? Have the authors explored more complex parameterizations (e.g., small MLPs) for ω_y?

Proposition 3 assumes p_g(x) = p_d(x). In practice, how sensitive is the effectiveness of the conditional alignment term V_BT-c to deviations from this perfect unconditional matching? Are there empirical observations from the experiments (e.g., from the MoG study) that suggest the conditional alignment learning remains effective even when unconditional FID is still being optimized?

The adaptive weighting mechanism is a simple solution. Did the authors experiment with or consider other adaptive weighting schemes from multi-task learning (e.g., based on uncertainty or task homoscedasticity) and, if so, how did they compare to the proposed method?

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
3

### Summary
The paper proposes SONA, a conditional GAN discriminator that focuses on these two objectives (a) unconditional “naturalness” scoring from (b) conditional alignment via orthogonal projections, and augments training with mismatching-aware supervision and adaptive weighting across objectives. 
The unconditional term is trained with a slicing adversarial networks (SAN) objective, while alignment uses two Bradley–Terry pairwise losses—one for conditional discrimination and one for mismatch awareness—plus a light adaptive weighting to keep the objectives balanced.
The method’s overall loss learns data realism and condition faithfulness without the two interfering. Empirically, SONA improves FID/IS across CIFAR-10, TinyImageNet, and ImageNet-128; it also plugs into a text-to-image GAN (GALIP) and gets consistent FID gains The paper includes propositions analyzing the objectives and ablations supporting the design choices.

### Strengths
- The proposal of “sum of naturalness + alignment” with an explicit orthogonal projection is a nice inductive bias and factorization that reduces interference between realism and conditioning by construction in the objective function.
- The paper shows consistent improvement across standard cGAN settings, including a nice ImageNet-128 table and a clean plug-in to GALIP for text-to-image.
- The paper shows the ablation experiments that separates out the impact of the orthogonal split, mismatch-aware loss, and adaptive weights.
- The source code is attached in the supplementary for reproducibility.

### Weaknesses
- The experimental comparisons emphasize AC-GAN-style and projection GANs; some more discussion on strong GAN baselines and competitive diffusion/rectified-flow generators could help strengthen the empirical case, especially on ImageNet.
- The paper introduces additional pairwise losses and learnable weighting under a normalization constraint, maybe some discussion on the compute/memory overhead could help readers evaluate the efficacy of the proposed approach reproduce gains reliably.
- For GALIP, ω_y is frozen from CLIP and the CLIP-score improvement seems kind of incremental. It could help the readers if larger-scale text-to-image (higher res, varied prompts) and a learnable ω_y variant experiment could be performed to check the alignment benefits in richer/diverse settings.

### Questions
should there be :

Consistent usage of matching-aware vs mismatching-aware throughout the paper. In Abstract (and consistently across the paper): “matching-aware (supervision/discrimination)” → “mismatching-aware …” (to match the title and the negative-pair supervision described).

### Soundness
3

### Presentation
3

### Contribution
3
