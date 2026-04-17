# MotionGPT3: Human Motion as a Second Modality

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
With the rapid progress of large language models (LLMs), multimodal frameworks that unify understanding and generation have become promising, yet they face increasing complexity as the number of modalities and tasks grows. We observe that motion quantization introduces approximation errors that cap motion quality, and that unifying discrete text and continuous motion within a single-stream backbone amplifies cross-modal interference. Motivated by recent multi-branch Transformer designs that separate signals from different modalities, we propose MotionGPT3, a bimodal motion–language model for both understanding and generation. MotionGPT3 encodes raw motion into a continuous latent space using a variational autoencoder (VAE), thereby avoiding quantization-induced artifacts, while leveraging the semantic prior of pretrained language models. A dual-stream Transformer with shared attention preserves modality-specific routes while enabling controlled, bidirectional information flow, which reduces interference, stabilizing optimization, and empirically accelerates convergence without degrading fidelity. For multimodal joint training, a generate-then-align three-stage schedule further improves stability and limits cross-task interference. Experiments show that MotionGPT3  achieves 2× faster convergence in training loss and up to 4× faster convergence in validation, while maintaining state-of-the-art performance on standard motion understanding and motion generation benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents MotionGPT3, a motion–language model designed to jointly handle motion understanding and generation while addressing key limitations of existing multimodal frameworks. To avoid the motion quantization issue, it encodes raw motion into a continuous latent space with a lightweight diffusion head, eliminating quantization-induced artifacts and enabling higher-fidelity synthesis. The architecture employs a dual-stream Transformer with shared attention, which maintains modality-specific information while allowing cross-modal exchange. Besides, a three-stage generate-then-align training schedule is proposed to further enhance the  convergence  efficiency.

### Strengths
* The paper is well written, easy to follow, and clearly organized. The figures are self-explanatory.
* The motivation is concrete and reasonable, and the method design is aligned well with the corresponding motivations, presenting sound performance improvements.
* This paper offers a well-reasoned perspective on the gap between discrete language token sequences and continuous motion latent representations, which can provide valuable insights to the community.

### Weaknesses
* Lack of detailed comparisons of inference latency or FLOPs against discrete-token baselines. Authors should provide the computational overheads induced by each proposed component to demonstrate the method's efficiency.
* The evaluation of the method is constrained to a single dataset (HumanML3D), which can not sufficiently demonstrate the generalization of the proposed framework to other motion domains. Providing more results on other benchmarks with diverse features would better illustrate the method's versatility. Besides, it is expected to offer more results with more recent language models other than GPT-2.
* As shown in Table 1, why does the proposed method significantly lower on the MModality metric compared to other models? Does this indicate that the model can only fits a specific distribution, thereby sacrificing the diversity of generated outputs.

### Questions
* typos: line 32, citepacross

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
4

### Summary
This paper presents MotionGPT3, a unified model that addresses motion quantization errors and cross-modal interference in motion-language tasks. Its key innovations include: 1) a VAE-based continuous motion latent space, 2) a dual-stream Transformer for controlled cross-modal interaction, and 3) a diffusion head to connect discrete text and continuous motion. Trained in three stages, MotionGPT3 sets new state-of-the-art performance on HumanML3D for both text-to-motion and motion-to-text generation, while converging 2-4 times faster than baselines.

### Strengths
1. The technical design in this paper is highly targeted, with each core component—a continuous VAE, a dual-stream architecture, and three-stage training—precisely solving a specific problem, and its necessity is rigorously validated through ablation studies, resulting in a lean and non-redundant overall architecture.

2. The experimental validation is comprehensive, encompassing quantitative comparisons, qualitative examples, and thorough ablation studies.

3. The study ensures high reproducibility by detailing implementation specifics—including data preprocessing, training protocols, and evaluation tools—and adheres to open-source standards with released code and materials.

### Weaknesses
1. The bimodal branch architecture proposed in this paper to address cross-modal interference in motion-language modeling is not a particularly novel approach, as similar frameworks have been proposed in existing unified text-image understanding and generation work, such as BAGEL [1]. However, the paper lacks discussion on how the proposed method differs from these existing approaches.

2. Baseline comparisons are outdated, lacking recent models (e.g., MotionGPT-2 2024 [2]，MG-MotionLLM 2025 [3]), which may conceal performance gaps in key metrics like M2T BERTScore.

3. The paper provides analysis on training convergence speed but lacks evaluation of inference latency, which remains a critical metric for motion generation applications.

[1] Emerging Properties in Unified Multimodal Pretraining, 2025, arxiv, 27 July
[2] MotionGPT-2: A General-Purpose Motion-Language Model for Motion Generation and Understanding, 2024 arxiv, 29 Oct
[3] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities, 2025 arxiv, 3 April

### Questions
Please see the weakness part.

### Soundness
3

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
5

### Summary
This paper propose MotionGPT3, a bimodal motion-language framework to unify the motion understanding and generation. It introduces two different modality branches to endow the model with motion-to-text and text-to-motion abilities.

### Strengths
1. The paper contains numerous figures and tables, as well as abundant visualization results, with a relatively clear overall structure.

2. Video demos are provided, demonstrating excellent performance.

3. Motion generation and motion understanding tasks are realized through two different branches and fine-tuning of the LLM.

4. The experimental results have achieved significant improvements.

### Weaknesses
1. The description of Figure 2 and the method section is not clear enough, making it difficult to intuitively grasp the authors' entire training process design and the detailed reasoning procedure.

2. Although the paper achieves good results, the method feels relatively incremental and highly hierarchical, lacking overall simplicity. Compared with MotionGPT and MotionGPT2, it does not bring a strong sense of novelty.

3. The autoregressive continuous token proposed by the authors has been used in many motion generation papers, which makes the contribution seem insufficient.

4. There are some typos. For example, is "L40" supposed to be "LLM"? The content from Line 82 to Line 89 is better presented in bullet points. Rarely seen in introductions is the writing style from Line 57 to Line 80, which fails to convey the necessity and practicality of the method proposed by the authors.

### Questions
1. It is hoped that the authors can elaborate on the training details of the three stages, including the inputs and outputs. In particular, they should clarify why such inputs are used in the second stage and what advantages they bring. Additionally, what is the main purpose of keeping the text branch frozen in the second stage? Could the authors provide an input example for Stage 2?

2. What exactly are the advantages of MotionGPT3 compared to MotionGPT and MotionGPT2, and which problems that the latter two failed to solve have been addressed? Relevant experiments would be highly appreciated.

3. The authors are expected to clarify the points mentioned in the "Weakness" section.

If the authors can address the questions I raised, I may consider increasing the score.

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
4

### Summary
This paper propose a bimodal motion–language model for text-to-motion and motion-to-text generation. The key designs include: a continuos latent modtion space with a latent diffusion header to bridge its gap between the next-token predition framework, a dual branch framework with shared attention to bridge the gap of the two modalities, and a three stage training strategy for more stable optimization of the proposed framework. Experimental results show the effectiveness of the proposed framework on both the text-to-motion and motion-to-text tasks.

### Strengths
- The motivation of utilizing the continuous motion latent space for lossless motion encoding and the diffusion header to bridge the gap between the next-token generation framework is reasonable. 
- The dual-branch framework to preserve modality-specific information and the shared attention for cross-modal communication is well motivated, and the three-stage training schemes stabilize the optimization of the proposed framework. 
- Experimental results on benchmarks of the two tasks are strong, and the effect of different design choices is validated with ablation studies. 
- The writing is good and the paper is easy to understand.

### Weaknesses
- The paper claims continuous VAE for motion encoding is better, but lacks an experimental comparison on motion encoding and decoding quality with previous schemes. Specifically, how is the improvement of the continuous VAE compared to the recently stronger motion quantization methods, e.g., the residual VQ proposed by MoMask (CVPR 2024) and the 2D motion quantization in MoGenTS (NeurIPS 2024)?
- Experiments are only conducted on the HumanML3D datasets. Adding more diverse datasets, e.g., Motion-X and  KIT-ML, will better illustrate the generalizability of the proposed framework.

### Questions
None.

### Soundness
3

### Presentation
3

### Contribution
3
