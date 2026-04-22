# DiffuDETR: Rethinking Detection Transformers with Denoising Diffusion Process

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 4

## Abstract
In this paper, we present DiffuDETR, a novel approach that formulates object detection as a conditional object query generation task, conditioned on the image and a set of noisy reference points. We integrate DETR-based models with denoising diffusion training to generate object queries' reference points from a prior gaussian distribution. We propose two variants: DiffuDETR, built on top of the Deformable DETR decoder, and DiffuDINO, based on DINO’s decoder with contrastive denoising queries (CDNs). To improve inference efficiency, we further introduce a lightweight sampling scheme that requires only multiple forward passes through the decoder. Our method demonstrates consistent improvements across multiple backbones and datasets, including COCO2017, LVIS, and V3Det, surpassing the performance of their respective baselines, with notable gains in complex and crowded scenes. Using ResNet-50 backbone we observe a +1.0 in COCO-val reaching 51.9 mAP on DiffuDINO compared to 50.9 mAP of the DINO. We also observe similar improvements on LVIS and V3DET datasets with +2.4 and +2.2 respectively. The code is available at https://github.com/MBadran2000/DiffuDETR.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes DiffuDETR, which reframes DETR-style object detection as denoising diffusion over object-query reference points. Two variants are introduced: DiffuDETR (on Deformable-DETR) and DiffuDINO (on DINO with CDNs). Training treats query anchors as low-dimensional diffusion variables; inference uses a lightweight sampler that runs only a few decoder passes. The method yields improvements across  different datasets.

### Strengths
1. Visualizations suggest fewer misses and tighter localization in crowded scenes, matching the intended strengths of iterative refinement over queries.

### Weaknesses
1. Limited practical gain from diffusion sampling for detection. As Table 6 shows, increasing decoder evaluations to D.E.=3 introduces a sharp compute/activation rise (FLOPs ≈ +16%, Activations ≈ +29%) while delivering only a modest accuracy bump (≈ +2 AP). The accuracy–efficiency trade-off is therefore not clearly favorable for real-world deployment.

2.Unclear procedure for choosing D.E. (decoder evaluations). Performance does not monotonically improve with larger D.E. in Table 6, yet the paper offers no principled rule (or validation protocol) for selecting D.E. This undermines reproducibility and makes the method harder to operationalize.

3. Convergence claim vs. evidence mismatch. Lines 74–75 state the approach helps “address convergence,” yet Figure 1 suggests DiffuDINO requires more epochs to reach its performance than DINO. This weakens the claimed convergence advantage unless the authors clarify training protocol differences or provide additional curves.

### Questions
1. How sensitive are results to the random seed / initial noise used in the diffusion process? Please report mean ± std AP over multiple seeds, and, if possible, the seed-to-seed variance of key categories (crowded vs. sparse scenes).

2. What is the effect of different sampling strategies (e.g., DDIM vs. DDPM, or other schedulers) on accuracy and compute? Do certain samplers yield better AP–FLOPs trade-offs or different optimal D.E. settings?

### Soundness
2

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
This paper draws on the ideas of DDPM and transfers the generation process of image denoising to object detection. Different from the traditional DiffusionDet method, which denoises noisy boxes and then combines them with RCNN for detection, this paper integrates the denoising process into the attention process. Furthermore, based on the deformable DETR and the contrastive denoising queries (CDN) of DINO, it proposes two models, namely DiffuDETR and DiffuDINO, respectively. Comparisons with baselines and some contemporaneous methods on three datasets (COCO2017, LVIS, and V3Det) demonstrate that the proposed models achieve the best accuracy.

### Strengths
1. Integrates the denoising process with the query decoding mechanism of attention.
2. Validates the two proposed models on three sufficient datasets.

### Weaknesses
1 The flow chart fails to effectively illustrate the network architecture, and it also appears somewhat small.
2 The symbol "f" is used in both Equation 3 and Equation 4 but represents different meanings, which should be avoided.
3. The methods used for comparison are somewhat outdated, as the latest version of DiffusionDet was released in 2023.

### Questions
Q1: Does the generation of initial queries still adopt the top-k approach? In Line 76 of the paper, it is mentioned that "We propose a new query initialization technique that aligns with the objective of denoising diffusion models to sample from the normally distributed reference points.
", yet the illustration in Figure 2 still indicates the use of learnable initial queries.
Q2: Is the denoising process reasonable? The method section shows that during training and inference, noisy boxes are not the main body of iteration; instead, they are used in the form of K to compute cross-attention with Q, assisting the iteration of Q. This step encourages the model to generate Q that conforms to the distribution of K, given that K represents noisy boxes. However, during inference, noisy boxes are directly sampled from a Gaussian distribution and iterated step-by-step following the DDPM paradigm. This iteration of noise does not align with the distribution of targets, yet the model still generates Q with a similar distribution to the noise, which may mislead the model.
Q3: How are noisy boxes embedded? Are noisy boxes directly used as K for cross-attention computation, or are they mapped to the dimension of Q?
Q4: Are the noisy boxes r at different time steps t independently and identically distributed (i.i.d.)? Or is the noise accumulated incrementally like in DDPM? If the latter is the case, I believe this should be more explicitly represented in Equation 4, and the meaning of "q" therein should also be clarified.
Q5: In Equation 3, does the time step t embedding have different effects at different positions?
Q6: Could you provide a detailed definition of "Decoder Evaluation"?

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
This paper introduces DiffuDETR, a diffusion-based extension of the DETR to improve object detection. The model uses DETR’s transformer-based architecture, including a multi-scale feature extractor, deformable attention encoder, and cross-attention decoder, but introduces a new query initialization mechanism inspired by diffusion processes. Instead of relying on manually defined anchor or reference points, queries are sampled from a normal distribution, aligning with the diffusion objective and simplifying initialization. Experiments on the COCO dataset show that while diffusion training converges more slowly, DiffuDETR eventually surpasses DINO after sufficient epochs, demonstrating improved robustness and detection performance.

### Strengths
1. The topic is interesting and relevant, as it reformulates the detection transformer’s query prediction as a denoising process, offering a fresh perspective on object detection.
2. The paper is generally well written and clearly organized, making it easy to follow and understand the proposed framework.
3. The comparison tables demonstrate that the proposed method achieves competitive performance against baseline models.

### Weaknesses
1. Regarding the 100-timesteps schedule, recent diffusion models typically sample a single timestep during training since the model learns to denoise from one distribution to another. Therefore, the claimed efficiency improvement seems more relevant to inference rather than training.
2. It is not clearly stated whether the paper uses a diffusion-specific loss function or a standard detection loss, clarifying this would help understand how diffusion is actually integrated into the training objective.
3. The paper would benefit from visualizations of intermediate timesteps or distributions to better illustrate the denoising process and the behavior of the model during denosing.
4. An analysis of the model’s sensitivity to the initialization of sampled queries would also strengthen the paper, as this is an important aspect of diffusion-based generation.

### Questions
Please refer to the weakness section.

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
3

### Summary
In this paper authors introduced two models, DiffuDETR and DiffuDINO, to built upon Deformable DETR and DINO, respectively. Experiments enables effective sampling at inference, where only three decoder evaluations are sufficient to achieve the best results while adding minimal computation overhead compared to the baseline.

### Strengths
I believe this work opens up new directions for integrating generative and autoregressive approaches into object detection, offering fresh perspectives beyond traditional discriminative formulations.

### Weaknesses
The baseline chosen in the paper are not the current state-of-the-art method.

### Questions
The improvement suggestions are as follows:
1.  Increasing research literature from 2024 to date，
2. To add a comparison of the computational overhead of different algorithms， and
3. Table 2 only selected DINO Zhang et al. (2022) for comparison. Why not compare more methods like in Table 1? More benchmark methods are needed.

### Soundness
3

### Presentation
3

### Contribution
2
