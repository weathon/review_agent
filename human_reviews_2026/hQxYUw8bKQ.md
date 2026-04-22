# Structure- and Appearance-Rich Training-Free Spatial Control for Text-to-Image Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 4, 4, 4

## Abstract
Text-to-image (T2I) diffusion models have shown remarkable success in generating high-quality images from text prompts. Recent efforts extend these models to incorporate conditional images (e.g., canny edge) for fine-grained spatial control. Among them, feature injection methods have emerged as a training-free alternative to traditional fine-tuning-based approaches. However, they often suffer from structural misalignment, condition leakage, and visual artifacts, especially when the condition image diverges significantly from natural RGB distributions. Through an empirical analysis of existing methods, we identify a key limitation: the sampling schedule of condition features, previously unexplored,  fails to account for the evolving interplay between structure preservation and domain alignment throughout diffusion steps. Inspired by this observation, we propose a flexible training-free framework that decouples the sampling schedule of condition features from the denoising process, and systematically investigate the spectrum of feature injection schedules for a higher-quality structure guidance in the feature space. Specifically, we find that condition features sampled from a single timestep are sufficient, yielding a simple yet efficient schedule that balances structure alignment and appearance quality. We further enhance the sampling process by introducing a restart refinement schedule, and improve the visual quality with an appearance-rich prompting strategy. Together, these designs enable training-free generation that is both structure-rich and appearance-rich. Extensive experiments show that our approach achieves state-of-the-art results across diverse zero-shot conditioning scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a training-free framework that enables spatial control for text-to-image diffusion models. The authors first identify the sampling schedule of condition features as a key factor limiting the performance of previous methods, and propose using a single medium timestep to address this issue through systematic analysis. In addition, the authors introduce Restart Refinement and Appearance-Rich Prompting to further enhance the generation quality. Extensive experiments demonstrate the effectiveness of the proposed method.

### Strengths
1. The empirical analysis and visualization on the sampling schedule of condition features are insightful and interesting. This analysis provides strong motivation for the proposed method and clearly supports its design choices.
2. The experiments sufficiently support the claimed contributions and demonstrate the effectiveness of the proposed approach.
3. The paper is well-structured and easy to follow.

### Weaknesses
1. While Restart Refinement and Appearance-Rich Prompting contribute to improved performance, they also introduce additional computational overhead and increase inference time.
2. This paper mainly focuses on UNet-based T2I models, which are somewhat outdated. It would be interesting to explore whether the discovered sampling schedule of condition features remains effective in DiT-based T2I models.

### Questions
How does the inference efficiency of the proposed method compare to other training-free approaches, such as Ctrl-X and FreeControl?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a training-free framework for enhancing spatial control in text-to-image diffusion models.

The authors diagnose key issues in prior feature-injection-based approaches (e.g., Ctrl-X), including structure misalignment, condition leakage, and visual artifacts, attributing them to the inadaptiveness of the sampling schedule.

To address these problems, the paper introduces three modules: Structure-Rich Injection (SRI), Restart Refinement (RR), and Appearance-Rich Prompting (ARP).

The proposed framework achieves state-of-the-art results across various conditional modalities (canny, depth, pose, segmentation, etc.) without any training, outperforming both training-based (e.g., ControlNet, T2I-Adapter) and training-free (e.g., FreeControl, Ctrl-X) methods in structure fidelity and visual realism.

### Strengths
The paper provides a convincing diagnosis of why training-free spatial control methods often fail, linking it to the inadaptiveness of the injection schedule.

The Structure-Rich Injection mechanism is simple, training-free, and computationally efficient thanks to feature caching.

The proposed SRI, RR, and ARP components are complementary and can be easily integrated with existing diffusion models such as SDXL.

The authors evaluate across diverse condition modalities, using both objective metrics (DreamSim, ImageReward, HPSv2) and qualitative analysis, consistently demonstrating superior results.

Despite being training-free, the method rivals or even surpasses fine-tuned approaches like ControlNet, highlighting impressive generality and real-world potential.

### Weaknesses
The analysis of the sampling schedule remains empirical; the paper lacks a formal justification or theoretical insight into why mid-timestep injection yields optimal results.

Although the paper discusses the trade-off between structural fidelity and visual appearance, it does not deeply explore the semantic interactions in the feature space.

The claim of ‘applicability to arbitrary pretrained diffusion models’ is mainly demonstrated on SDXL, with limited evidence for other architectures (e.g., DiT, LDM).

Restart Refinement improves appearance quality but occasionally compromises structure alignment; optimal iteration count and noise levels are empirically chosen.

The Appearance-Rich Prompting component depends on the quality and consistency of the employed multimodal LLM, which may limit reproducibility.

Absence of human evaluation: The evaluation relies solely on reward-based metrics that approximate human preference; no actual human study is presented.

### Questions
Could the authors provide a more theoretical explanation or empirical ablation supporting why medium timesteps (around 600) yield optimal structure-appearance balance?

### Soundness
2

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
3

### Summary
The paper presents a novel, training-free framework for spatial control in text-to-image (T2I) generation. The authors propose a method that enhances control over structure and appearance during the generation process, addressing limitations in existing methods. By analyzing the temporal dynamics of diffusion features, they discover that the sampling schedule for condition features plays a crucial role in balancing structure alignment and visual quality. Their solution decouples the condition feature sampling from the denoising process, resulting in improved structural preservation and visual fidelity. The framework consists of three key components: Structure-Rich Injection (SRI)**, **Restart Refinement (RR), and Appearance-Rich Prompting (ARP). Extensive experiments demonstrate that this approach outperforms state-of-the-art (SOTA) methods across a variety of zero-shot conditioning scenarios.

### Strengths
1. This approach eliminates the need for additional fine-tuning, making it highly adaptable and efficient for various models and conditions.

2. By identifying and addressing the limitations of existing feature injection schedules, the authors provide a principled way to sample condition features, significantly improving structure preservation and visual fidelity.

### Weaknesses
1. The success of the method relies heavily on empirical observations regarding the optimal timesteps for feature injection. While the paper provides a comprehensive analysis, the method's effectiveness could vary depending on the specific nature of the condition images used, requiring further validation across diverse datasets and domains.

2. The framework introduces additional steps such as the Restart Refinement (RR) schedule and Appearance-Rich Prompting (ARP) strategy. While this enhances performance, it could lead to increased computational requirements.

### Questions
The paper uses multiple evaluation metrics, but have the authors considered designing a comprehensive metric that would evaluate the balance between structure alignment and visual quality?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a training-free framework for spatially controlled text-to-image generation.
It decouples the sampling schedule of condition features from the denoising process. The method achieves improved structural alignment, reduced condition leakage, and higher visual fidelity across diverse control modalities without retraining.

### Strengths
-- The paper identifies a limitation in prior work: fixed timestep injection fails to balance structural fidelity and domain alignment, validated via KL divergence and L2 distance curves in Fig. 2.

-- Restart Refinement (RR) reduces condition leakage and visual artifacts while preserving structure, as qualitatively shown in Fig. 8 and quantitatively in Table 1. ARP also demonstrate improvements.

--The method outperforms some training-free and training-based baselines on reward-model metrics aligned with human preferences.

### Weaknesses
-- The noise sampling process “from the perturbation kernel” lacks specification of distribution or variance schedule.

-- The number of restart iterations N is mentioned but its selection criterion or default value is omitted.

-- Hyperparameters for baselines (e.g., inversion steps for DDIM in Ctrl-X) are not detailed, though results depend on them. Does the method keep the same hyperparameters to the baseline? 

-- No timing or memory measurements are provided for the proposed method versus baselines, despite caching claims.

-- Figure 4 is not informative and not easy to understand.

### Questions
Please see the weakness.

### Soundness
2

### Presentation
2

### Contribution
2
