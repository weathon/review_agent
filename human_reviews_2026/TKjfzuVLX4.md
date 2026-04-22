# Autoregressive Visual Decoding from EEG Signals

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 6

## Abstract
Electroencephalogram (EEG) signals have become a popular medium for decoding visual information due to their cost-effectiveness and high temporal resolution. However, current approaches face significant challenges in bridging the modality gap between EEG and image data. These methods typically rely on complex adaptation processes involving multiple stages, making it hard to maintain consistency and manage compounding errors. Furthermore, the computational overhead imposed by large-scale diffusion models limit their practicality in real-world brain-computer interface (BCI) applications. In this work, we present AVDE, a lightweight and efficient framework for visual decoding from EEG signals. First, we leverage LaBraM, a pre-trained EEG model, and fine-tune it via contrastive learning to align EEG and image representations. Second, we adopt an autoregressive generative framework based on a "next-scale prediction" strategy: images are encoded into multi-scale token maps using a pre-trained VQ-VAE, and a transformer is trained to autoregressively predict finer-scale tokens starting from EEG embeddings as the coarsest representation. This design enables coherent generation while preserving a direct connection between the input EEG signals and the reconstructed images. Experiments on two datasets show that AVDE outperforms previous state-of-the-art methods in both image retrieval and reconstruction tasks, while using only 10% of the parameters. In addition, visualization of intermediate outputs shows that the generative process of AVDE reflects the hierarchical nature of human visual perception. These results highlight the potential of autoregressive models as efficient and interpretable tools for practical BCI applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes AVDE, a framework for reconstructing visual images from EEG signals, aiming to improve upon existing methods that often rely on complex, computationally expensive diffusion pipelines. AVDE uses a two-stage process: first, it fine-tunes a large pre-trained EEG model (LaBraM) using contrastive learning to align EEG and image (CLIP) representations. Second, it employs an autoregressive transformer, inspired by VAR's "next-scale prediction" strategy and using a pre-trained VQ-VAE, to generate images progressively from the aligned EEG embedding. The authors report state-of-the-art results on retrieval and reconstruction tasks using the THINGS-EEG dataset, highlighting significant improvements in efficiency (fewer parameters, faster inference) compared to diffusion-based predecessors.

### Strengths
Demonstrates substantial reductions in parameter count (~90%), computational cost (FLOPs), inference time, and memory usage compared to a representative diffusion-based baseline, making it more viable for practical BCI applications.
Successfully leverages large pre-trained models (LaBraM, VAR) through fine-tuning and contrastive alignment, highlighting the benefits of transfer learning in this domain.

### Weaknesses
The ablation (Table 4) does not cleanly isolate the contribution of the autoregressive generative model (VAR) versus a comparable diffusion model when using the same high-quality LaBraM encoder. This makes it difficult to definitively conclude that the VAR approach itself is superior.
The framework primarily combines existing powerful components (LaBraM, VAR architecture, VQ-VAE, CLIP alignment) rather than introducing fundamentally new techniques.
Linking specific generation scales to cortical areas (V1/V2/IT) is speculative without stronger supporting evidence .

### Questions
Could you clarify the ablation study in Table 4? To fairly assess the contribution of the VAR framework, could you provide results for a LaBraM + Diffusion baseline trained under identical conditions (using the same fine-tuned LaBraM encoder) as your main LaBraM + VAR model?

Regarding the efficiency comparison in Table 3: How does AVDE compare to the diffusion baseline if the latter is optimized for speed using fewer sampling steps (e.g., 10-20, if image quality remains acceptable)?

Table 2 focuses on Subject-08. How robust are the reconstruction performance gains across other subjects, particularly those with lower retrieval scores? An average comparison across subjects for reconstruction would be valuable.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Summary:
This paper presents AVDE, a simple but effective framework for reconstructing visual stimuli from EEG signals. The AVDE method consists of two stages: (1) aligning EEG and image representations by fine-tuning a pre-trained EEG encoder (LaBraM) via contrastive learning, and (2) generating images using a hierarchical autoregressive transformer that predicts visual tokens from coarse to fine scales. The authors evaluate AVDE on two datasets (i.e., THINGS-EEG and EEG-ImageNet) and demonstrate superior performance in both image retrieval and reconstruction tasks compared to existing methods, while using only 10% of the parameters of prior diffusion-based models.

### Strengths
1. **next-scale prediction**. AVDE has an interesting idea (“next-scale prediction”, rather than traditional multi-stage diffusion models). It is conceptually novel. Unlike earlier methods that suffer from error propagation across multiple stages, AVDE generates visual content progressively, starting from coarse EEG embeddings and refining them to more detailed image representations. This approach improves coherence between EEG inputs and reconstructed images.

2.**Performance**
The proposed method demonstrates significant improvements in both image retrieval and reconstruction tasks, outperforming state-of-the-art methods while using only 10% of the parameters required by previous diffusion models. This is clearly evidenced by the performance comparison in Table 1 (Page 6), where AVDE shows superior accuracy in retrieval tasks (Top-1 accuracy of 0.300 and Top-5 accuracy of 0.582) under the within-subject setting, and substantial improvements in the cross-subject setting. These results illustrate the model's robustness and the efficiency of the framework.

Additionally, the reconstruction quality shown in Table 2 and Figure 3 further supports AVDE's high quality. AVDE achieves better low-level (PixCorr) and high-level (SSIM, AlexNet, Inception) metrics compared to existing methods, confirming that the method not only reconstructs clearer and more faithful images but also captures high-level semantic consistency.

3.The paper is generally well-written.

### Weaknesses
1. The paper’s strongest results are within-subject; cross-subject transfer remains limited and the manuscript offers little analysis of what factors drive between-subject variance. Minor

2. The efficiency comparison fixes diffusion at 50+4 steps and specific CFG/top-k settings while the proposed AR approach uses 10 steps; modern diffusion samplers (e.g., DDIM, DPM-Solver/DPMS++) can operate at 10–20 steps with competitive quality, and lighter priors are possible. As written, the comparison risks conflating algorithmic gains with configuration choices. Moderate

3. Current ablations do not isolate which components most drive gains, making it hard to attribute improvements. More ablations are needed. Moderate

4. Code cannot be accessed. Major

5. The manuscript states that “the generative process reflects the hierarchical nature of human visual perception,” but current evidence is largely qualitative. Visualization (Fig.4) reflects a coarse-to-fine spatial refinement induced by the multi-scale architecture. However, the text interprets this as evidence for a biological temporal/area hierarchy (retina→V1→V2/V4→IT). That mapping is speculative here: you do not manipulate time or measure area-selective correspondences. As written, it risks conflating spatial scale with neural hierarchy. Moderate

### Questions
1.**Performance on Cross-Subject and Real-World Data.** 
While the paper presents strong within-subject results, the cross-subject performance (Top-1 accuracy of 0.143 and Top-5 accuracy of 0.329 in Table 1) suggests that the model's ability to generalize across different individuals is still limited. I understand that subject variability is a well-known challenge in EEG-based decoding. But including analysis of subject-specific factors (e.g., signal quality, attention level) and exploring personalization or domain adaptation strategies would further improve the quality of this study.

2.**No Real-Time or Latency Evaluation.**

The paper emphasizes efficiency but does not provide concrete metrics such as inference time or latency on standard hardware. This is particularly important for BCI applications, where real-time feedback is often critical.
In comparison, Scotti et al. (2023) and Zhang et al. (2025) highlight the computational bottlenecks when training large models on complex EEG-image tasks. This paper would benefit from a more detailed ablation study exploring how different training parameters (e.g., learning rate, batch size, number of epochs) affect resource consumption and performance.

3.**Model Interpretability**

The hierarchical, coarse-to-fine image generation process is a black-box operation, and while the paper visualizes the intermediate stages (Figure 4), it remains unclear how the model's decision-making process works in finer detail. For example, how does the EEG embedding influence specific features of the generated image at each scale? Are there any cases where the EEG signals lead to inconsistent or inaccurate image generation, and if so, how could the model be improved to handle such cases?

The lack of transparency in how the model generates certain image details from EEG signals is an issue, as Li et al. (2024) showed the brain regions and time for image reconstruction. Neural decoding methods should provide not only accurate predictions but also clear insights into how decisions are made from complex neural data. Future work could focus on improving the interpretability of the autoregressive process, possibly by integrating attention mechanisms or using explainable AI methods.


Ref:

Scotti, A., Banerjee, A., Goode, J., Shabalin, S., Nguyen, A., Dempster, A., Verlinde, N., Yundler, E., Weisberg, D., Norman, K., et al. (2023). Reconstructing the mind's eye: fMRI-to-image with contrastive learning and diffusion priors. NeurIPS

Zhang, K., He, L., Jiang, X., Lu, W., Wang, D., & Gao, X. (2025). CognitionCapturer: Decoding visual stimuli from human EEG signal with multimodal information. AAAI

Li, D., Wei, C., Li, S., Zou, J., Qin, H., & Liu, Q. (2024). Visual decoding and reconstruction via EEG embeddings with guided diffusion. NeurIPS

### Soundness
3

### Presentation
4

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
This paper introduces AVDE, an autoregressive framework for reconstructing visual stimuli from EEG signals. The approach combines a pre-trained EEG encoder (LaBraM), fine-tuned via contrastive learning with CLIP embeddings, and a transformer-based autoregressive generator (VAR) built on a pre-trained VQ-VAE. The authors aim to simplify the complex multi-stage diffusion-based pipelines used in previous EEG-to-image decoding work. Experiments on THINGS-EEG report modest improvements in retrieval and reconstruction metrics, while significantly reducing computational cost and model size.

### Strengths
1.	The combination of a large-scale pre-trained EEG encoder (LaBraM) and an autoregressive generative model (VAR) is a technically reasonable and promising direction. Pretraining-based decoding is becoming a general paradigm, and applying it here may help bridge the data scarcity of EEG.
2.	VAR offers computational advantages over diffusion-based pipelines, including lower latency and reduced parameter count. The paper quantifies these gains and reports consistent, if moderate, improvements in retrieval and reconstruction tasks.
3.	The writing is clear and well-organized. Figures effectively illustrate the hierarchical decoding process, and ablations on efficiency and encoder replacement provide supporting evidence for the engineering claims.

### Weaknesses
1.	The research motivation remains underdeveloped. While AVDE is presented as an alternative to diffusion models, it does not fully resolve the fundamental challenges of EEG-to-image decoding. The work should be viewed as an exploratory step rather than a definitive advance.
2.	The methodological innovation is limited. The approach primarily substitutes diffusion with an autoregressive transformer and incorporates LaBraM pretraining. These are incremental modifications rather than new modeling ideas.
3.	Experimental data are restricted to THINGS-EEG, which is relatively small and homogeneous. Validation on richer or multimodal datasets (e.g., MEG or fMRI) would significantly strengthen the generality claim.
4.	The paper lacks neuroscientific or interpretability analysis. The authors claim that AVDE “reflects the hierarchical nature of human visual perception,” but no neural or representational evidence supports this statement. Analyses of temporal, spatial, or feature-level EEG correlations would be necessary to substantiate such claims.
5.	Figure 1 depicts the prior unCLIP pipeline rather than the proposed AVDE model, which may confuse readers. Showing the AVDE architecture first would better emphasize the paper’s own contribution.

### Questions
1.	Could you clarify the specific contribution of LaBraM—does its advantage come mainly from large-scale pretraining, or from its architectural design? Supplementary experiments isolating these factors would help clarify its role.
2.	What practical or theoretical advantages does the VAR-based autoregressive generation offer compared to diffusion models, beyond computational efficiency? Do you observe qualitative differences in reconstruction behavior?
3.	Have you considered extending the validation to other neural modalities (e.g., MEG, fMRI) or tasks to test whether AVDE generalizes beyond the THINGS-EEG dataset?

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
4

### Summary
This paper proposes AVDE, a lightweight and efficient framework designed for decoding and reconstructing visual images from EEG signals. The authors critique current methods that depend on complex, multi-stage adaptations or computationally intensive diffusion models, arguing that these approaches are impractical for real-world applications in BCI.

### Strengths
+ This work thoughtfully addresses the challenge of computational overhead seen in models like diffusion, achieving remarkable results with just 10% of the parameters. 

+ Moreover, the autoregressive approach shines with its simplicity and efficiency. By utilizing the EEG embedding as the initial token for the transformer, it establishes a clever architectural foundation. 

+ This model outperforms more complex multi-stage and diffusion-based counterparts in both retrieval and reconstruction tasks, proving that a lighter model can indeed be more effective.

### Weaknesses
- Autoregressive models can sometimes face the challenge of error accumulation, where an initial mistake in an early token or pixel patch can amplify throughout the rest of the generation. State how we can address this.

- While this approach offers great efficiency and performs well on standard metrics, we may notice that the qualitative fidelity of its reconstructions doesn’t always match that of larger, slower models, such as diffusion, in more complex scenes. It's all about striking a balance, and share your thoughts on this.

- Moreover, the effectiveness of the reconstruction greatly hinges on the pre-trained image tokenizer, which translates images into tokens. If the tokenizer’s performance is lacking, it can limit the model's potential. Your insights would be useful.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
