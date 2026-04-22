# SpikeGen: Decoupled “Rods and Cones” Visual Representation Processing with Latent Generative Framework

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
The process through which humans perceive and learn visual representations in dynamic environments is highly complex. From a structural perspective, the human eye decouples the functions of cone and rod cells: cones are primarily responsible for color perception, while rods are specialized in detecting motion, particularly variations in light intensity. These two distinct modalities of visual information are integrated and processed within the visual cortex, thereby enhancing the robustness of the human visual system. Inspired by this biological mechanism, modern hardware systems have evolved to include not only color-sensitive RGB cameras but also motion-sensitive Dynamic Visual Systems, such as spike cameras. Building upon these advancements, this study seeks to emulate the human visual system by integrating decomposed multi-modal visual inputs with modern latent-space generative frameworks. We named it ***SpikeGen***. We evaluate its performance across various spike-RGB tasks, including conditional image and video deblurring, dense frame reconstruction from spike streams, and high-speed scene novel-view synthesis. Supported by extensive experiments, we demonstrate that leveraging the latent space manipulation capabilities of generative models enables an effective synergistic enhancement of different visual modalities, addressing spatial sparsity in spike inputs and temporal sparsity in RGB inputs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
Mimicking the decoupling of color and light intensity perception of human into cone and rod cells, the authors proposed SpikeGen, which outperforms previous state-of-the-art methods in multiple spike-RGB tasks: 1) conditional image/video deblurring, 2) dense frame reconstruction and 3) scene novel-view sythesis. 

SpikeGen follows the general setup of a Masked Auto-regressive mode (MAR) pretrained with diffusion loss. It uses a frozen VAE encoder and a trainable spike encoder to encode blurry RGB input and spike stream into latent representations, respectively. 

SpikeGen has two training stages, a pretraining stage and a finetuning stage. During the pretraining, the loss used is the per-token diffusion loss between the faded clear RGB latent and the latent predicted by the ViT followed by a compact MLP. The spike stream loss is used during the finetuning stage.

The experiment results reported in the paper show that SpikeGen beats all other benchmarks on all three tasks.

### Strengths
1. The empirical results reported by the authors on the three tasks show the superior performances of SpikeGen over all previous methods compared in the paper. The superiority of the proposed method and training pipeline is consistent across various tasks, metrics and threshold settings. 

2. The figures presented in the paper are clear, well-organized, and informative, especially Figure 2. It effectively and clearly conveys the overall training pipeline while also giving details on the architecture of the Spatial–Temporal Separable Spike Encoder. This visualization greatly improves the reader’s understanding of the specific modifications made to the original MAR pipeline.

3. The adoption of the latent diffusion training for a dual modality spike-RGB self-supervised training method is both innovative and effective. The introduction of a random gamma parameter during training enables controllable modality dominance, allowing adaptation to different downstream tasks with minimal effort.

### Weaknesses
1. Although the authors briefly mention at line 475 that SpikeGen’s novel-view synthesis pipeline is two-stage, the main text provides insufficient explanation. It remains unclear to me 1) what the two stages are specifically (deblurring + vanilla 3DGS?),  2) how they differ from other benchmarks reported in Table 3, and 3) how time efficient (or inefficient) is SpikeGen compared to other single-stage methods. A more detailed description and comparison would greatly improve the paper’s completeness.

2. Table 9 in the appendix provides comparisons with other two-stage NVS methods. However, it only includes two relatively old, training-free approaches. The authors are encouraged to include other two-stage methods, e.g. those listed in Table 2, to provide a more complete and up-to-date comparison.

3. An ablation study replacing MAR with a regular conditional DiT diffusion model would provide a much stronger basis for SpikeGen’s architectural design choices. 

4. Similarly, an ablation on design choices of the S3 encoder would also further validate the architecture's effectiveness. Nevertheless, considering the page limit, this omission is understandable and does not substantially affect my overall evaluation.

5. At lines 69–70, the phrase “By reviewing current studies” appears to be a typo or editing error.

### Questions
1. Have you tried to linearly interpolate gamma values for testing? How does gamma values affect performances?

2. Please see the weaknesses section for additional questions and suggestions

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a framework called SpikeGen, which takes a blurry image and the corresponding spike stream as input to generate a clear image. The authors evaluated the method on three downstream task datasets, demonstrating its effectiveness.

### Strengths
1. The manuscript is well-structured, the figures are clear, the layout is appropriate, and the language flows smoothly.

2. The approach is bio-inspired and well-justified.

3. The experimental evaluation is sufficient and the results demonstrated are promising.

### Weaknesses
1. The paper's citation format needs to be revised.

2. In Figure 2, I believe the blurry RGB latent and the Clear RGB latent should show differences (or be visually distinct), instead of being represented by the same shape and color.

### Questions
1. In lines 276 to 280, the authors describe the calculation process for the faded image $I_{faded}$. However, I could not seem to find where this result is applied within the method and Figure 2.

2. In line 312, the authors claim their model was pre-trained on ImageNet. However, that dataset only includes clear images. I am curious how the blurry images and the spike streams were obtained, especially the latter.

3. Based on the formula in line 270, the hyperparameter $\gamma$ takes a value of 0 or 1 with high probability (>60%). Does such a modality drop rate seem too high? This seems to indicate that the model mainly receives single-modality input during the training process.

4. How are the pixel-space similarity measures performed during fine-tuning with limited data?

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
4

### Summary
The paper proposes SpikeGen, a novel latent generative framework for decoupled visual representation learning between RGB and spike modalities. It performs diffusion modeling in the latent space, combining VAE-encoded representations with a per-token diffusion mechanism to balance efficiency and effectiveness. The framework incorporates a configurable dual-modality latent pre-training mechanism and a spatio-temporal separable spike encoder for efficiently extracting temporal–spatial features from spike streams. Experiments conducted on multiple benchmark datasets, including REDS, GOPRO, VidarReal, and Blender-NeRF, show that SpikeGen surpasses existing methods across multiple metrics, demonstrating superior generalization and robustness.

### Strengths
This paper introduces latent variables into such tasks for the first time, enabling the model to achieve higher-level feature modeling while maintaining both efficiency and effectiveness. The proposed method demonstrates strong originality, reliable and comprehensive experimental results, and clear exposition, making it an contribution to multimodal visual representation learning.

### Weaknesses
1. The training cost of SpikeGen’s latent diffusion process is relatively expensive and requires substantial pre-training resources, which limits its reproducibility and practical deployability.

2. During the pre-training phase on the ImageNet dataset, the spike frame configuration involves randomly sampling 8 frames from 64 generated spike frames for each image. This setup may prevent the model from fully leveraging information from the spike modality, causing it to rely primarily on RGB inputs and thus partially degrade into a single-modality model.

3. Certain details in the paper are described ambiguously. In Section 3.2 (Spatial–Temporal Separable Spike Latent), the authors mention that the model generates temporal attention weights through two consecutive 1×1×1 3D convolutional layers to model the temporal dimension explicitly. However, this process lacks formal equations or explicit computational explanations, which affects the interpretability and reproducibility of the method.

4. Although the model achieves impressive results on synthetic and benchmark datasets, its validation on real-world event-based data remains limited, making it insufficient to fully demonstrate the model’s generalization capability under complex real-world conditions.

### Questions
1. During the pre-training phase on ImageNet, the model randomly samples only 8 frames from 64 generated spike frames. How do the authors ensure that such sparse temporal sampling can still effectively capture spike information? Have experiments with different sampling numbers (e.g., 16 or 32 frames) been conducted to verify that the model indeed utilizes spike information rather than primarily relying on RGB features?

2. In Section 3.2, the authors mention that the model generates temporal attention weights through two consecutive 1×1×1 3D convolutional layers. Could the authors supplement this part by explaining how these weights are computed and applied, to clarify their role in the feature fusion process?

3. The model is pre-trained on ImageNet using 8 A800 GPUs. Does the model's outstanding performance stem from the retraining advantage gained through ample computational resources, or from model innovation?

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
5

### Summary
This paper presents SpikeGen, a biologically inspired framework that mimics the rods–cones decoupling mechanism in human vision.
The model fuses spike streams (representing temporal luminance information) and RGB images (representing spatial–chromatic information) in a shared latent diffusion space, thereby achieving joint visual representation learning.

### Strengths
1 The paper's multimodal fusion in the latent space rather than the pixel space is elegant and computationally efficient. The approach avoids the need for spatially precise alignment between asynchronous spikes and frames.

2 The paper includes quantitative and visual comparisons on multiple datasets and modalities, covering both synthetic and real-world-like settings.

### Weaknesses
(A) Task–Method Mismatch

The model’s design is static latent diffusion, but some tasks (e.g., motion deblurring) inherently require explicit temporal modeling (e.g., flow, exposure trajectory).
SpikeGen’s temporal encoder uses only 3D convolutions, which might not capture fine-grained dynamics.

In novel-view synthesis, the model lacks geometric consistency constraints (e.g., ray-based volumetric modeling).
The improved perceptual quality may not correspond to true 3D structure preservation.

(B) Evaluation and Fairness Issues

SpikeGen is compared against single-modality baselines (e.g., SpkDeblurNet, DeblurGS) while itself using RGB+Spike dual input — an unfair comparison unless baselines are also dual-modality.

The absence of clear ablation studies (e.g., removing diffusion, removing spike input, removing γ-fusion) makes it difficult to identify the true source of performance gain.

Training details for each task (e.g., hyperparameters, dataset scale, latent dimensionality) are insufficient for reproducibility.

(C) Data Authenticity and Realism

Most spike data are synthetic (via SpikingSim) and do not represent real sensor noise, asynchronous pixel behavior, or refractory effects.

For each task, the paper lacks quantitative evaluation on real spike-camera datasets.

(D) Overly Perfect Results / Missing Uncertainty

SpikeGen outperforms all prior works on all metrics across all tasks — an unlikely scenario that raises concerns about overfitting or inconsistent training conditions.

The diffusion framework, by nature, introduces randomness and perceptual diversity; yet, the results show unrealistically consistent sharpness and color balance without variance analysis.

(E) Efficiency and Practicality

Despite operating in latent space, diffusion models remain computationally expensive.
The paper does not report inference speed or energy consumption, which are crucial in neuromorphic vision research that emphasizes efficiency.

### Questions
1. How are the spike encoder’s temporal windows determined across different frame rates or datasets?  
   Is the encoder adaptive to spike density variations?

2. For the deblurring task, does SpikeGen explicitly model the exposure time or motion trajectory, or rely solely on spike event accumulation?

3. How is the latent diffusion conditioned on spike features?  
   Is it concatenation, cross-attention, or a learned fusion layer?

4. How does the model generalize to **real spike data** (e.g., Vidar)?  
   Have the authors tested domain adaptation or noise robustness?

### Soundness
2

### Presentation
3

### Contribution
2
