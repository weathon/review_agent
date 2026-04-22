# Test-time Domain Generalization for Image Super-resolution

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Test-time domain generalization (TTDG) methods enhance the performance of neural networks on target domains by transferring the feature distribution of target samples to approximate that of the source domain, while avoiding the computational cost associated with fine-tuning on the target domain. However, existing TTDG methods primarily rely on style transfer strategies operating at a coarse granularity, 
which prove ineffective for pixel-level prediction tasks such as image super-resolution (SR). To address this limitation, we propose a multi-codebook based test-time domain generalization framework (MC-TTDG). Our method leverages both domain-specific and domain-invariant codebooks to achieve fine-grained representation learning on source domains, and performs pixel-level nearest-neighbor feature matching and transfer to accurately adjust target domain features. Furthermore, we introduce a voting-based strategy for optimal domain-specific codebook selection, which improves the precision of feature transfer through multi-party consensus. Extensive experiments across diverse data distributions, and network architectures demonstrate that the proposed method effectively transfers feature distributions for SR networks. Our code is available at https://github.com/ZaizuoTang/MC-TTDG.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes MC-TTDG, a multi-dictionary-based temporal domain generalization framework for testing image super-resolution (SR). Unlike conventional TTDG methods reliant on coarse-grained style transfer, this approach combines domain-specific and domain-invariant dictionaries to capture both global and subtle cross-domain variations. During the testing phase, pixel-level feature matching is achieved through a nearest neighbor codebook strategy. A voting-based mechanism selects the optimal domain-specific codebook, thereby enhancing robustness against distribution shifts.

### Strengths
1. The proposed codebook-based feature transfer method achieves pixel-level alignment between target features and the source domain, thereby overcoming the limitations of previous style transfer approaches.
2. By employing a multi-codebook architecture and introducing a voting-based strategy for selecting domain-specific codebooks, the reliability of model transfer can be enhanced, thereby circumventing the challenges encountered by expert selectors based on neural gating or classification.
3. A first test-time domain generalization method designed for low-level vision tasks.

### Weaknesses
1. Can the computational overhead (execution time, FLOPs, and memory) of MC-TTDG relative to style transfer-based and other baseline TTDG methods be explicitly measured and reported during testing in resource-constrained (edge device) environments? Absent these details, claims about deployability are speculative.
2. The article indicates that multiple codebooks are preferable to a single codebook.  How sensitive are the results to the number of source domains/codebooks? Have any failure cases been observed due to voting ties or misleading inputs? If so, what impact does the fallback mechanism have in such scenarios?

### Questions
1. In multi-codebook settings, how might one diagnose and mitigate issues of codebook redundancy, collapse, or underutilisation? Are there scenarios—such as large-scale domain imbalance—where specific codewords fail to contribute effectively? How might this impact transfer performance and network robustness?
2. Could quantitative results be demonstrated for domain transfer across datasets and in real-world scenarios (e.g., unseen camera types, fundamentally different image degradations), rather than being confined solely to the relatively controlled “branch” within DRealSR?

### Soundness
3

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
This paper addresses the challenge of test-time domain generalization (TTDG) for low-level vision tasks, specifically image super-resolution (SR), where existing TTDG methods fail due to coarse-grained style transfer. The proposed framework, MC-TTDG, leverages multi-codebook representation learning and pixel-level feature matching to address three key limitations: low transfer granularity, loss of domain-specific features, and suboptimal domain-specific codebook selection. Key contributions include: (1) introducing a codebook-based pixel-level feature transfer strategy tailored for low-level vision tasks; (2) proposing a multi-codebook representation learning strategy (RLMC) that disentangles domain-invariant and domain-specific features to preserve source domain details; (3) designing a voting-based codebook selection strategy to mitigate domain shift-induced inaccuracies; and (4) being the first TTDG method explicitly designed for low-level vision tasks with codebook integration.

### Strengths
1. MC-TTDG is the first to adapt codebook-based representation learning to TTDG for low-level vision, addressing a critical limitation of style-based TTDG methods. The RLMC strategy’s disentanglement of domain-invariant and domain-specific features (via shared + domain-specific codebooks) is a creative combination of existing ideas, and the voting-based selection effectively mitigates domain shift in codebook choice—an unaddressed problem in prior multi-codebook work.
2.  The experimental design is comprehensive: ablation experiments cover core components (codebook setup, transfer methods, selection strategies), baselines include state-of-the-art TTDG methods (e.g., TTMG, DG-PIC), and validation across diverse datasets/architectures demonstrates architectural generalization. Metrics (PSNR, SSIM, LPIPS) are standard for SR, ensuring result comparability.

### Weaknesses
1. The claim that RLMC disentangles domain-invariant and domain-specific features is not supported by direct evidence (e.g., feature clustering, ablation of domain-specific features’ impact on cross-domain generalization). Current evidence (Table 6) only shows domain-specific features improve SR quality, not that they are truly disentangled.
2.Only one visual difference figure is presented (Figure 5/6), with little qualitative analysis.
3. There may be shared structures or continuous variations (manifold-like variations) between domains, rather than discrete, mutually exclusive divisions; could the codebook partitioning method mislead the model to reinforce false domain separations?

### Questions
1. The SOTA metric was not bolded in the article, suggesting an adjustment.
2. How large are the codebooks (number of codewords, dimensionality)? Is there any tradeoff between codebook size and performance?
3. How does the voting mechanism behave when there are many source domains (e.g., >10)? Is the method still stable under noisy or ambiguous domains?
4. The paper states “Our code is available at ***” but no link is provided. For reproducibility, the exact URL should be included or committed to release upon publication.

### Soundness
2

### Presentation
2

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
This paper proposes a method based on multiple codebooks for test-time domain generalization specifically designed for the image super-resolution task. By leveraging domain-specific and domain-invariant codebooks to perform nearest neighbor feature matching and transfer at the pixel level, and by using a voting-based strategy to select the optimal domain-specific codebook, the effectiveness of the method was ultimately demonstrated on various SR test datasets.

### Strengths
1.The writing is clear and easy to understand, and the methods are easy to implement.

2.The generalization ability on various SR test datasets is impressive.

### Weaknesses
1. This paper does not report the additional parameter growth and training consumption required by the proposed method compared to the pre-trained model.


2. This paper does not conduct a detailed comparison between this method of finetuning the pre-trained SR model and the effect of fine-tuning all the parameters of the model. Although this paper emphasizes that this is a method for domain generalization of super-resolution test datasets, adding additional training data and fine-tuning the full parameters of the original pre-trained model together will also bring certain benefits. Therefore, it would be best for this paper to demonstrate the effectiveness of the proposed finetune method in terms of efficiency and performance.

### Questions
Please refer to the Weaknesses.

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
This paper introduces MC-TTDG, a framework for Test-Time Domain Generalization (TTDG) specifically designed for the task of image Super-Resolution (SR). They propose a novel approach that leverages a multi-codebook architecture to perform fine-grained, pixel-level feature transfer at test time. The core idea is to learn a shared, domain-invariant codebook and multiple domain-specific codebooks during training. At inference, features from a target domain image are adapted by replacing them with the nearest-neighbor codewords from the learned codebooks. A voting mechanism is introduced to select the most appropriate domain-specific codebook for an unseen target sample. The authors demonstrate through extensive experiments that their method significantly outperforms existing TTDG techniques when applied to SR across various datasets and network architectures.

### Strengths
- As far as I know, the paper is the first to formally address the problem of Test-Time Domain Generalization for image Super-Resolution. It astutely points out that domain shift in SR is a practical and significant challenge and that existing TTDG methods are fundamentally mismatched for such pixel-level tasks.
- The proposed solution of using codebooks for pixel-level feature matching is an elegant and highly intuitive answer to the identified problem. The replacement of coarse style transfer with fine-grained codeword substitution is a logical and well-motivated design choice. The visual and quantitative results strongly suggest that this approach is far more effective than style-based methods for SR.
- The authors have conducted a thorough set of experiments. They ablate every key component of their model, and demonstrate applicability across different SR architectures. This extensive validation bolsters the claim that the method is effective and generalizable.

### Weaknesses
- The central mechanism for separating domain-invariant and domain-specific features relies on a simple architectural split. While this is a functional design, it feels reminiscent of early, foundational ideas in domain generalization research. The field has since moved towards more sophisticated techniques. The paper does not engage with or advance this front; it instead applies a known, relatively simple technique to a new problem. While the application is novel, the core disentanglement method is not. 
- While the problem formulation is novel and the engineering is solid, the work feels more like a clever and effective application of existing ideas to a new domain. Although, I think integrating prior methods for other task is novel, there should at least be experiments showing how and why such prior method is the best while others do not fit. (e.g., why is feature separation by within the architecture is best for formulating the codebook rather than more sophistically separating the feature by training additional module?)

### Questions
See Weakness

### Soundness
3

### Presentation
2

### Contribution
2
