# ID-PreFeR: ID-Preserving Face Restoration with Mixed Data Quality

- Avg Score: 5.00
- Decision: Reject
- Scores: 8, 2, 6, 4

## Abstract
This paper introduces ID-PreFeR, a robust ID-preserving face restoration method that addresses the ill-posed face restoration problem by introducing personalized information. Existing methods often suffer from computationally expensive training and storage requirements while being sensitive to the quality of reference images. We present a lightweight personalized injector to enable efficient personalization without the burden of regularization data. Besides, we propose an ID-quality disentanglement training strategy to ensure robust identity learning, even when some of the reference images are of low-quality. An ID-preserving sampling strategy is further proposed to enhance the identity fidelity during inference. Experiments on both synthetic and a newly collected real-world mobile phone dataset validate the effectiveness and practicality of the proposed method.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a novel method for face restoration that preserves identity using a lightweight personalized injector and handles mixed-quality reference images. It addresses the challenges of heavy personalization burden and high-quality (HQ) dependency in existing methods by proposing a low-rank adaptation (LoRA) technique, an ID-quality disentanglement strategy, and an ID-preserving sampling strategy.

### Strengths
# Strength:
(+) The lightweight personalized injector using LoRA reduces the computational and storage burden, making it efficient with only about 1% of the base model parameters.

(+) The ID-quality disentanglement strategy effectively mitigates the impact of low-quality (LQ) reference images, enhancing robustness.

(+) The real-world mobile phone dataset provides practical validation, addressing the limitations of celebrity-based datasets used in prior research.

### Weaknesses
# Weakness:
(-) The paper lacks a detailed comparison with a broader range of state-of-the-art methods, potentially limiting the scope of evaluation.

(-) The ID-preserving sampling strategy's optimization process may add computational overhead during inference, which could affect real-time applications.

(-) The reliance on a pretrained face recognition model for identity vector extraction might introduce biases or inaccuracies inherent to the model.

## Minor Suggestions:

1. For ICLR, ~\citep{} should be used rather than ~\cite{}. The citation format of this paper is not correct.

2. Some important prior works are missing [1-5]; It would be better if the author could discuss them in the related works.

**Ref**: 

[1] GCFSR: a Generative and Controllable Face Super Resolution Method Without Facial and GAN Priors. In CVPR 2022.

[2] Towards authentic face restoration with iterative diffusion models and beyond. CVPR 2023.

[3] Degradation conditioned gan for degradation generalization of face restoration models. In ICIP 2023.

[4] LAFR: Efficient Diffusion-based Blind Face Restoration via Latent Codebook Alignment Adapter. Arxiv 2025.

[5] Diffusion Once and Done: Degradation-Aware LoRA for Efficient All-in-One Image Restoration. Arxiv 2025.

### Questions
# Questions
1. How does the performance of ID-PreFeR compare to other recent diffusion-based methods beyond those listed (e.g., in terms of PSNR or SSIM)?

2. What are the specific computational costs associated with the ID-preserving sampling strategy during inference?

3. How was the optimal number of reference images (e.g., 5) determined for training the personalized injector?

4. Could the method be adapted to handle extreme degradation types not covered, such as severe occlusion?

5. What steps were taken to ensure the fairness and diversity of identities in the new mobile phone dataset?


Overall, I think this paper is interesting, and the introduced method can be used in the real world. If the author could address my concerns, I will keep my rating.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes ID-PreFeR, a personalized face restoration framework designed to preserve identity information even when reference images are of mixed or low quality. The key idea is to introduce three components:

(1) a lightweight personalized injector trained via LoRA on cross-attention layers to minimize computational cost;

(2) an ID-quality disentanglement strategy that separates identity from image quality using pseudo quality tokens and Min-SNR-γ weighting;

(3) an ID-preserving sampling strategy that optimizes denoising steps with face feature similarity.

The method aims to reduce the dependency on high-quality references and demonstrates competitive quantitative and qualitative results on synthetic and real-world datasets, including a newly collected mobile phone dataset.

### Strengths
+ The paper addresses the challenge of restoring faces from low-quality inputs and imperfect references, which has value for real-world applications.

+ The use of LoRA for efficient personalization is a sensible choice, offering lower computational overhead compared to fully fine-tuned diffusion models.

+ The idea of disentangling identity and quality, as well as the inclusion of an ID-preserving sampling step, seems appealing.

+ The authors compare with several strong baselines and include both quantitative and qualitative analyses.

### Weaknesses
- The technical details of the proposed “ID-quality disentanglement” and “ID-preserving sampling” are not clearly explained, leaving many details unexplained. The learning mechanism of pseudo-quality tokens and their optimization behavior are vague, and no ablation or visualization convincingly demonstrates how disentanglement truly happens and how they are used in this framework. Such design is important in this paper, but not explained well.

- From a technical view, the approach is largely a combination of prior ideas from IP-Adapter, DreamBooth, and LoRA personalization, but the authors try to explain them in a very different way, which hinders the readers from fully understanding this work. The novelty appears incremental, mainly an adaptation of text-to-image personalization for face restoration.

- Some figures are not good. Figure 2 tries to show that with the low-quality reference, the competing methods would have performance degradation. However, the given visual results are not obvious. Figure 3 is also not fridently to understand.

- Despite mentioning a “real-world mobile dataset,” the dataset is small, and there is no quantitative evaluation on real degraded images. Most results rely on synthetic degradation. Robustness in uncontrolled environments is unproven.

- Key implementation details (e.g., how the quality tokens are initialized, how the MLLM provides prompts, and exact loss weighting) are missing. It is hard to reproduce the method or verify its stability.

- The claimed ability to “disentangle ID and quality” or to “eliminate HQ dependency” is not strongly supported. Performance gains over recent personalized diffusion baselines (e.g., FaceMe, Gen2Res) are marginal.

### Questions
- How exactly are the “quality-aware pseudo words” optimized and removed during inference? Do they correspond to learned embeddings or fixed tokens? What did they represent?

- How does the proposed “ID-preserving sampling” differ from simple face feature guidance or feature consistency regularization used in prior diffusion-based restoration works?

- How does your method compare to a trainable IP-Adapter face encoder in terms of parameter size and identity consistency?

The proposed “latent prior loss” resembles DreamBooth’s prior preservation—can you clarify whether there are conceptual or mathematical differences?

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
5

### Summary
This paper introduces ID-PrefeR, a novel ID-preserving face restoration method designed to address the challenge of handling mixed-quality reference images. The key innovation lies in combining three main components: (1) a lightweight personalized injector based on LoRA that fine-tunes cross-attention layers without heavy regularization, (2) an ID-quality disentanglement training strategy that learns to separate identity information from quality degradation in reference images, and (3) an ID-preserving sampling strategy that optimizes predicted noise during inference to enhance identity preservation.The method is evaluated on FFHQ and CelebRef datasets with both synthetic and real-world degradations, showing competitive performance against blind face restoration baselines like GFPGAN, CodeFormer, DiffBIR, while demonstrating robustness to low-quality reference images.

### Strengths
1.Well-motivated problem: The paper clearly identifies and addresses a practical limitation of existing personalized face restoration methods - their dependency on high-quality reference images. This is highly relevant for real-world applications.

2.Novel quality disentanglement approach: The ID-quality disentanglement training strategy using pseudo-degraded prompts is creative and intuitive. Explicitly teaching the model to separate identity from quality artifacts is a sound approach.

3.Lightweight and efficient design: Using LoRA for the personalized injector is practical and allows efficient personalization without heavy regularization burden.

### Weaknesses
1.Limited theoretical analysis: The paper lacks rigorous theoretical justification for why Min-SNR-γ weighting helps with mixed quality data. The ID-preserving sampling optimization (Eq. 7) is presented heuristically without convergence analysis or theoretical guarantees.

2.Insufficient implementation details: Critical details are missing for reproducibility: exact network architecture, detailed training hyperparameters, learning rates, batch sizes. The face recognition model is not specified, and HQ reference selection method is unclear.

3.Hyperparameter sensitivity not thoroughly studied: The ID-preserving sampling involves multiple hyperparameters but only high-level component ablation is provided. The choice of γ=5 for Min-SNR is not justified.

### Questions
1.Can you provide more rigorous analysis of why Min-SNR-γ weighting helps with mixed quality data? What is the connection between training SNR and reference image quality?

2. Regarding ID-preserving sampling: How sensitive is the method to stepsize δ and iteration count? Could you provide sensitivity analysis? How do you balance the two terms in Eq. 7?

3.How was γ=5 chosen for Min-SNR weighting? Did you try other values? How are ω_d and quality tokens determined during training?

4.How does the method generalize to degradation types unseen during training? Can it handle extreme degradations or completely different artifact types?

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
The authors propose ID-PreFeR, a framework for personalized face restoration that improves robustness by introducing three components: a personalized injector, an ID-quality disentanglement strategy, and an ID-preserving sampling strategy.

### Strengths
1. The paper is generally complete and well-structured.
2. The inclusion of a real-world dataset enhances the practical contribution and relevance of this research to the community.

### Weaknesses
1. The method description and pipeline illustration are unclear and require further clarification.
2. It is not explicitly stated whether the proposed method requires training a new LoRA model for each individual identity. If so, the authors should justify this design choice and discuss the differences in computational and time cost compared with methods that do not require retraining during inference.
3. The personalized injector module needs to be more clearly differentiated from prior works such as DreamBooth.
4. The baseline comparison is insufficient, methods such as RefLDM[1] and RestorerID[2], which address similar tasks, should be included for a fairer evaluation.

[1]: Refldm: A latent diffusion model for reference-based face image restoration.

[2]: Restorerid: Towards tuning-free face restoration with id preservation.

### Questions
See weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2
