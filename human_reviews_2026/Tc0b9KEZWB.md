# Lumina-OmniLV: A Unified Multimodal Framework for General Low-Level Vision

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
We present Lunima-OmniLV (abbreviated as OmniLV), a universal multimodal multi-task framework for low-level vision that addresses over 100 sub-tasks across four major categories, including image restoration, image enhancement, weak-semantic dense prediction, and stylization. OmniLV leverages both textual and visual prompts to offer flexible, user-friendly interactions. Built on Diffusion Transformer (DiT)-based generative priors, our framework supports arbitrary resolutions — achieving optimal performance at 1K resolution — while preserving fine-grained details and high fidelity. Through extensive experiments, we demonstrate that separately encoding text and visual instructions, combined with co-training using shallow feature control, is essential to mitigate task ambiguity and enhance multi-task generalization. Our findings also reveal that integrating high-level generative tasks into low-level vision models can compromise detail-sensitive restoration. These insights pave the way for more robust and generalizable low-level vision systems.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
OmniLV proposes a unified multimodal framework for >100 low-level vision sub-tasks. It uses a DiT/flow-matching backbone (Lumina-Next), separate encoders for text (Gemma-2B) and visual exemplars, and a co-training “condition adapter.”

### Strengths
- Clear problem scope: one model for restoration, enhancement, dense prediction, and stylization with both text and visual prompts.
- Large coverage of tasks and a 1K-resolution setting; practical details on three-stage training are provided.

### Weaknesses
- Novelty is limited. Most components are known (DiT/flow matching, visual in-context prompts, adapters, ControlNet-like injection). The paper reads like an engineering report.
- Evidence is uneven. On dense prediction OmniLV is weaker than specialized models (e.g., depth RMSE 0.525 vs 0.291), and stylization trades PSNR for FID against GenLV. The “unified SOTA” claim does not hold broadly.
- Evaluation leans on PSNR/MUSIQ and synthetic OLV-T splits; limited human studies or robustness statistics.

### Questions
- Will you release data or a reproducible recipe?
- Can you add stronger baselines (e.g., SUPIR/SDXL-based adapters, recent flow/DiT restorers) with matched resolution and sampling cost?

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
3

### Summary
This paper presents OmniLV, a versatile multimodal model supporting a wide range of low-level vision tasks, i.e., over 100 sub-tasks across four major categories, including image restoration, image enhancement, weak semantic dense prediction, and stylization. The method leverages both textual and visual prompts to offer flexible, user-friendly interactions, while the DiT produces the final output. Furthermore, this work provides several insights into the design space, as well as a large training dataset. Extensive experiments validate the validity and generalization of the method.

### Strengths
1. This paper is well-motivated and easy to follow. It addresses the drawbacks of previous task-specific and visual-prompt-based models.
2. Beyond the method, the author further provides several ablations in design space, e.g., encoding and conditioning, which could provide insights to future works. I appreciate it.
3. The method supports several low-level tasks with strong overall performance and generalizable abilities.
4. The authors also present a dedicated dataset.

### Weaknesses
1. Why employ Gemma for text embedding? Does this paper try other models like T5, CLIP text encoder, or LLM? Some experiments or explanations need to be presented.
2. This paper claims that the model could generalize to cross-domain tasks. Please provide some related results.
3. Although the model achieves strong unification compared with previous task-specific models, the large model size makes inference relatively inefficient. The comparison of inference efficiency should be provided.

### Questions
See Weaknesses.

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
2

### Summary
This paper presents a unified multimodal framework (Lumina-OmniLV) designed for low-level vision which can address over 100 sub-tasks. Built upon a pre-trained DiT, OmniLV is capable of processing arbitrary-resolution images and accepts both textual and visual instructions to guide its operations. Through extensive ablation studies, this work systematically investigate and conclude that separating the encoding of text and visual prompts, co-training a condition adapter with the  DiT, and injecting conditional information into the early stages of the model are crucial for mitigating task ambiguity and improving performance. Overall, the paper presents a comprehensive and well-engineered system supported by extensive empirical validation.

### Strengths
- The development of a single, unified model capable of handling over 100 diverse low-level vision tasks is a significant engineering achievement. By demonstrating the feasibility of such a generalist system, this work provides a valuable contribution and a strong baseline for the future of unified low-level vision research.

- This paper contains extensive ablation studies on the encoding of multimodal prompts, the mechanism for condition injection, the optimal position for feature injection, and the fusion of visual exemplars for in-context learning.

### Weaknesses
- My main concern regarding this paper is its methodological novelty. The overall framework is composed of existing, powerful components: a pretrained DiT, a frozen LLM and pretrained VAEs. This framework relies heavily on the intrinsic generalization capabilities of these models. This reliance may explain the authors' claims in Limitations that the model "does not always achieve optimal performance in certain specialized scenarios," as it lacks task-specific architectural biases that a tailored approach might provide.

- Given that many low-level vision tasks depend on precise visual instructions, which can be difficult to describe clearly with text, could the textual prompt be replaced or augmented by  learnable "task queries"? The advantage of such an approach is that:
1)	It is independent of LLM, making the system more self-contained and potentially more efficient at inference time.
2)	Learnable queries might allow the model to more explicitly and robustly capture the unique characteristics and differences between the numerous low-level tasks. This could potentially address the issue of suboptimal performance in certain scenarios.

### Questions
Please refer to my comments in Weaknesses.

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
2

### Summary
This paper presents LUMINA-OmniLV (OmniLV), a unified multimodal and multitask framework for general low-level vision, capable of addressing over 100 subtasks across four categories Built on Diffusion Transformer (DiT) generative priors, OmniLV integrates both textual and visual prompts for flexible, user-friendly interactions. The model supports arbitrary resolutions up to 1K, and the authors introduce a separate multimodal encoding scheme (LLM for text, VAE for images) to reduce task ambiguity. Extensive experiments across multiple benchmarks demonstrate that OmniLV achieves competitive or superior performance compared with both specialized and all-in-one baselines. Key findings include the importance of shallow feature co-training and the detrimental effect of mixing high-level generative tasks into low-level fidelity-critical domains.

### Strengths
$\textbf{Comprehensive scope and ambition.}$ The work systematically tackles over 100 sub-tasks within a single framework, providing a rare attempt at large-scale unification for low-level vision. The breadth of coverage from denoising and deblurring to stylization sets a strong benchmark for generalist low-level models.

$\textbf{Sound architectural reasoning.}$ The separation of textual and visual encoders is well-motivated and experimentally justified (e.g., the t-SNE analysis on p. 4 demonstrates reduced task confusion). This yields clearer conditioning than typical multimodal fusion schemes.

$\textbf{Large and diverse dataset.}$ The authors curate a 40 M-sample dataset (Fig. 6, 9) spanning restoration, enhancement, dense prediction, and stylization, which constitutes a valuable contribution in itself for future research on multimodal low-level vision.

### Weaknesses
The manuscript introduces a framework primarily built upon empirical refinements of established diffusion architectures, integrating modality-specific encodings, conditional adapters, and projection-based prompt integration. Although these enhancements demonstrably improve performance relative to earlier unified or prompt-driven frameworks such as PromptIR, GenLV, and OmniGen, they remain incremental in nature. A deeper conceptual foundation, potentially through an information-theoretic interpretation of multimodal conditioning or geometric formulations of task alignment, would significantly strengthen the theoretical impact and originality of the work.

Moreover, the evaluation, though extensive in breadth, disproportionately emphasizes restoration and enhancement tasks, leaving dense prediction and stylization tasks relatively underrepresented. Quantitative assessments for critical tasks such as depth estimation and normal prediction are limited and confined to only a few datasets, undermining claims of generality and limiting insights into the model's robustness across diverse modalities.

Clarity regarding experimental rigor is further compromised by ambiguities within the ablation analyses. For instance, Table 2 reveals relatively minor numerical variations in performance metrics for condition integration, yet lacks explicit discussion of statistical significance or practical relevance. Moreover, despite supplementary details provided in Appendix A, critical elements including precise protocols for dataset generation and explicit instructions for prompt construction remain insufficiently detailed, creating obstacles for reproducibility and independent verification.

In addition, the computational complexity and resource requirements associated with training specifically, reliance on extensive datasets and a substantial allocation of 16 A100 GPUs pose significant scalability concerns. The manuscript does not adequately explore the implications of these requirements on practical deployment, inference efficiency, or resource sustainability relative to existing task-specialized alternatives. Addressing these practical dimensions explicitly would enhance the applicability of the proposed method and substantially elevate the overall significance of the contribution.

### Questions
1. How sensitive is the performance to the capacity and choice of the textual LLM (Gemma-2B)? Would smaller or larger language encoders materially affect fidelity?

2. Does the model exhibit catastrophic interference when fine-tuned on new low-level tasks not seen during pretraining?

3. Could the authors report inference speed or computational efficiency compared with existing all-in-one diffusion baselines?

4. How does the framework handle conflicting instructions when both text and visual prompts provide inconsistent task specifications?

I’m confident that this paper is of moderate quality, and I would be willing to raise my initial ratings if my concerns are properly addressed.

### Soundness
2

### Presentation
2

### Contribution
2
