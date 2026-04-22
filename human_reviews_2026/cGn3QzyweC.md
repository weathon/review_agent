# Extreme Blind Image Restoration via Prompt-Conditioned Information Bottleneck

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Blind Image Restoration (BIR) methods have achieved remarkable success but falter when faced with Extreme Blind Image Restoration (EBIR), where inputs suffer from severe, compounded degradations beyond their training scope. Directly learning a mapping from extremely low-quality (ELQ) to high-quality (HQ) images is challenging due to the massive domain gap, often leading to unnatural artifacts and loss of detail. To address this, we propose a novel framework that decomposes the highly challenging ELQ-to-HQ restoration process. We first learn a projector that maps an ELQ image onto an intermediate, less-degraded LQ manifold. This intermediate image is then restored to HQ using a frozen, off-the-shelf BIR model. Our approach is grounded in information theory; we provide a novel perspective of image restoration as an Information Bottleneck problem and derive a theoretically-driven objective to train our projector. This loss function effectively stabilizes training by balancing a low-quality reconstruction term with a high-quality prior-matching term. Our framework enables Look Forward Once (LFO) for inference-time prompt refinement, and supports plug-and-play strengthening of existing image restoration models without need for finetuning. Extensive experiments under severe degradation regimes provide a thorough analysis of the effectiveness of our work.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a progressive method for extreme blind image restoration. It leverages LQ images as a proxy between ELQ images and HQ images, and devise a cycle consistency constraints between proxy LQ and synthesized LQ through information bottleneck theory. Additionally, the authors develop a prompt refinement strategy (LFO) based on the intermediate LQ results, further improving the restoration quality of HQ. The experiments demonstrate impressive visual results for extreme blind images.

### Strengths
1. The problem definition is quite interesting and rarely studied. The restoration results of ELQ -> LQ -> HQ images are obviously much better than ELQ -> HQ finetuned results.
2. The blur-MSE loss is a thoughtful and reasonable design for cycle constraining predicted LQ and synthesized LQ. The HQ-prior and HQ-fid loss functions are also reasonable.
3. The visual results of the restored ELQ images are impressive.
4. Experimental results in Table 2 demonstrates good generalization abilities of the proposed method.

### Weaknesses
1. The core design of this paper lies in the progressive restoration process and the cycle consistency constraint, both of which are well-studied in image restoration. Hence, the contribution of this paper feels somewhat incremental.
2. I think some real-world extremely distorted samples are needed for validation. For some distorted images in Figures 6 and 8, the inputs are severely degraded with almost no visible high-frequency details to help infer the content. I am also curious about what the text prompts are and how they contribute to the restoration process. Can the caption model output meaningful prompts from these ELQ images?
3. It seems a mistake in Eq. 21: g is not needed to generate x_LQ.

### Questions
1. I wonder about the practicality of this method. In what real-world scenarios would images be so severely degraded that they need such restoration?

### Soundness
3

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
This paper addresses the problem of Extreme Blind Image Restoration (EBIR), where images suffer from severe, compounded degradations beyond the scope of typical Blind Image Restoration (BIR) models. The authors propose a framework that decomposes this challenging task into two simpler steps: 1) a trainable "projector" ($f_{\theta}$) maps the extreme low-quality (ELQ) input onto an intermediate, less-degraded (LQ) manifold , and 2) a frozen, off-the-shelf BIR model ($g$) restores the intermediate LQ image to high-quality (HQ). The core of the method lies in the training of this projector, for which the authors derive a theoretically-driven objective from the Information Bottleneck (IB) principle.

### Strengths
The paper is generally well-written and clearly structured. The core methodology is well-illustrated in Figure 2 and Figure 3, which aids in understanding the proposed pipeline

### Weaknesses
1.Novelty of the Core Idea: The claim of novelty regarding the use of the Information Bottleneck (IB) principle for image restoration needs to be more carefully justified and contextualized. The application of IB to restoration and super-resolution is an active area of research. For instance, (Hsu et al., 2024) and (Zhu et al., 2024) have already explored IB-based objectives for super-resolution tasks. More directly, (Gao et al., 2025) also proposes an IB-based framework (InfoBFR) as a plug-and-play module for blind face restoration. The authors must provide a clearer differentiation of their specific IRIB formulation and its application as a pre-processor for general-domain EBIR against this existing literature to solidify the paper's novel contributions.

2.Insufficient Experimental Validation: The paper's central premise is that a direct, end-to-end (E2E) mapping from ELQ to HQ is "challenging" or "intractable,"  thus justifying the proposed two-stage decomposition. However, the experimental evidence provided is insufficient to support this strong claim. The primary baseline for the E2E approach appears to be "OSEDiff + finetune". As the paper notes, this finetuning was done using LoRA adapters. This is not equivalent to a full, rigorous training of the model from scratch on the ELQ $\rightarrow$ HQ task. This baseline is weak and does not represent the true capability of a fully trained diffusion model.This premise seems to challenge the consensus in the field. Foundation diffusion models are fundamentally designed to restore highly structured images from pure Gaussian noise—a task that arguably involves a more "massive domain gap" than restoring from ELQ inputs, which still retain low-frequency structural information.Furthermore, extensive work in extreme-low-bitrate image compression, such as (Relic et al., 2024) and (Xu et al., 2025), has clearly demonstrated that diffusion-based models can be effectively trained to restore high-fidelity images from severely distorted and information-poor inputs.Given this, the claim that the direct ELQ $\rightarrow$ HQ mapping is "intractable" is unconvincing. The experiments do not provide a fair comparison and are insufficient to prove the superiority of the proposed two-stage approach over a properly and fully trained E2E baseline.


3.Ambiguous Motivation and Contribution: The paper's motivation is unclear. It seems to oscillate between two different claims:Fundamental Claim: The ELQ $\rightarrow$ HQ task is fundamentally "ill-posed" and "intractable,"  requiring the proposed decomposition. Practical Claim: The solution (a pre-processor for a frozen backbone ) suggests the motivation is practical—avoiding the high computational cost of retraining large diffusion models.If the motivation is practical (which is a valid contribution), the experimental setup should compare against other parameter-efficient adaptation techniques. For example, how does training the proposed projector $f_{\theta}$ compare against using a similar number of parameters to finetune the full backbone $g$ (e.g., via LoRA) on the E2E ELQ $\rightarrow$ HQ task? The paper seems to have created a new, more severe degradation setting  and then compared its method against baselines that were either not trained for this setting or were not fully trained for it.

REF:
@inproceedings{hsu2024drct,
  title={Drct: Saving image super-resolution away from information bottleneck},
  author={Hsu, Chih-Chung and Lee, Chia-Ming and Chou, Yi-Shiuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6133--6142},
  year={2024}
}
@article{zhu2024information,
  title={Information bottleneck based self-distillation: Boosting lightweight network for real-world super-resolution},
  author={Zhu, Han and Chen, Zhenzhong and Liu, Shan},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024},
  publisher={IEEE}
}
@article{gao2025infobfr,
  title={Infobfr: Real-world blind face restoration via information bottleneck},
  author={Gao, Nan and Li, Jia and Huang, Huaibo and Shang, Ke and He, Ran},
  journal={arXiv preprint arXiv:2501.15443},
  year={2025}
}

inproceedings{relic2024lossy,
  title={Lossy image compression with foundation diffusion models},
  author={Relic, Lucas and Azevedo, Roberto and Gross, Markus and Schroers, Christopher},
  booktitle={European Conference on Computer Vision},
  pages={303--319},
  year={2024},
  organization={Springer}
}
@inproceedings{xu2025decouple,
  title={Decouple Distortion from Perception: Region Adaptive Diffusion for Extreme-low Bitrate Perception Image Compression},
  author={Xu, Jinchang and Wang, Shaokang and Chen, Jintao and Li, Zhe and Jia, Peidong and Zhao, Fei and Xiang, Guoqing and Hao, Zhijian and Zhang, Shanghang and Xie, Xiaodong},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={18051--18061},
  year={2025}
}

### Questions
1.The central claim of the paper hinges on the supposed inferiority of a direct ELQ $\rightarrow$ HQ mapping. To validate this, the authors must provide a comparison against a baseline diffusion model (e.g., OSEDiff) that has been fully trained from scratch on the same ELQ $\rightarrow$ HQ data, using a comparable parameter count and training budget. Why do the authors believe this direct mapping is "intractable," especially given that diffusion models routinely recover full images from pure noise?

2.The current method trains the projector $f_{\theta}$ while keeping the restorer $g$ frozen. This seems to be a practical choice. I am curious about the performance if the entire pipeline ($f_{\theta}$ and $g$) were trained jointly on the ELQ $\rightarrow$ HQ task. This experiment would help isolate whether the benefit comes from the structural decomposition (ELQ $\rightarrow$ LQ $\rightarrow$ HQ) itself or simply from the proposed pre-training strategy.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a novel framework for Extreme Blind Image Restoration (EBIR). By first projecting extremely low-quality (ELQ) images onto an intermediate low-quality (LQ) manifold and then restoring them with existing high-quality (HQ) models, the method decomposes the challenging ELQ-to-HQ mapping. The authors formulate the problem using the Information Bottleneck (IB) principle and design an IB-based loss function to stabilize training. They also introduce a “Look Forward Once (LFO)” prompt refinement strategy, and demonstrate the effectiveness of their approach through experiments and integration with existing restoration models.

### Strengths
1. The paper formalizes the extreme blind image restoration problem as an Information Bottleneck problem and derives a theoretically-driven loss function, providing a new perspective for the image restoration field.

2. The proposed ELQ→LQ→HQ decomposition framework effectively extends the applicability of existing restoration models without requiring fine-tuning of the main restoration model, making it easy to integrate and deploy in practice.

3. The paper conducts both quantitative and qualitative comparisons on multiple public datasets (DIV2K, DIV8K), covering mainstream methods, plug-and-play capability, and ablation studies, demonstrating the effectiveness and flexibility of the proposed approach.

### Weaknesses
1. Lack of innovation: Although applying the Information Bottleneck theory to image restoration is novel, the ELQ→LQ→HQ decomposition approach is quite common in existing two-stage or hierarchical EBIR frameworks. The main innovation lies in the theoretical interpretation and loss design.

2. Insufficient analysis of individual components in the IRIB loss (such as LQ reconstruction, HQ prior, and HQ fidelity). The effectiveness of the LFO strategy is only demonstrated with a limited number of iterations, lacking a more systematic analysis of different iteration counts.

3. The main experimental comparisons are with OSEDiff, Real-ESRGAN, SeeSR, and S3Diff, but lack experiments with the latest extreme restoration methods (such as DiffBIR and Chain-of-Zoom) to further validate the effectiveness of the method.

4. The improvements on pixel-level metrics (PSNR/SSIM) are limited, with the main advantages shown in perceptual metrics (LPIPS, FID, etc.), but the analysis of true semantic structure recovery is not detailed enough.

5. The paper does not provide information on the additional computational complexity introduced by the method (including FLOPS, parameter count, inference time, etc.). It is also unclear how much extra training time is required due to the added projection layer and new loss functions, which is important for evaluating the method's effectiveness.

### Questions
1. Could you provide independent ablation studies for each component of the IRIB loss to analyze their individual contributions to the final performance?

2. Could you provide information on the additional computational complexity introduced by the model (including FLOPS, parameter count, and inference time), as well as how much extra training time is required due to the added projection layer and new loss functions?

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
2

### Summary
The paper proposes a new image restoration method designed for extreme degradations. The key idea is to learn a projector that maps an extremely degraded image onto an intermediate, less-degraded one before inferring the final high-quality image. The method demonstrates improved results when combined with an off-the-shelf image restoration (IR) approach.

### Strengths
1. The method generally shows improvements compared to the baseline or other compared methods.

2. Using the degraded image as a supervisory signal is interesting, although it was proposed before.

### Weaknesses
1. To me, the faithfulness with respect to the original input may be questionable.

### Questions
It is unclear what guarantees the consistency between the output and the input, particularly regarding the projector model’s ability to generate predictions that remain faithful to the input content.

### Soundness
2

### Presentation
3

### Contribution
2
