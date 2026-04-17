# Direct Reward Fine-Tuning on Poses for Single Image to 3D Human in the Wild

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 10

## Abstract
Single-view 3D human reconstruction has achieved remarkable progress through the adoption of multi-view diffusion models, yet the recovered 3D humans often exhibit unnatural poses. This phenomenon becomes pronounced when reconstructing 3D humans with dynamic or challenging poses, which we attribute to the limited scale of available 3D human datasets with diverse poses. To address this limitation, we introduce DrPose, Direct Reward fine-tuning algorithm on Poses, which enables post-training of a multi-view diffusion model on diverse poses without requiring expensive 3D human assets. DrPose trains a model using only human poses paired with single-view images, employing a direct reward fine-tuning to maximize PoseScore, which is our proposed differentiable reward that quantifies consistency between a generated multi-view latent image and a ground-truth human pose. This optimization is conducted on DrPose15K, a novel dataset that was constructed from an existing human motion dataset and a pose-conditioned video generative model. Constructed from abundant human pose sequence data, DrPose15K exhibits a broader pose distribution compared to existing 3D human datasets. We validate our approach through evaluation on conventional benchmark datasets, in-the-wild images, and a newly constructed benchmark, with a particular focus on assessing performance on challenging human poses. Our results demonstrate consistent qualitative and quantitative improvements across all benchmarks. Project page: https://seunguk-do.github.io/drpose.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the challenge of reconstructing 3D humans from single images, specifically focusing on "hard poses" (e.g., dynamic or acrobatic) where existing methods often fail. The paper's main contributions are twofold: 1) The proposal of DRPOSE15K, a new synthetic dataset constructed using an existing motion dataset and a pose-conditioned video generative model, which is designed to have higher pose diversity than previous datasets. 2) The introduction of DRPOSE, a direct reward fine-tuning algorithm that post-trains a multi-view diffusion model to better align with the diverse poses in the new dataset. The stated goal is to improve the quality and pose accuracy of 3D human reconstructions, particularly for these challenging in-the-wild cases.

### Strengths
- **Task:** The paper focuses on "hard poses," which is a core, unsolved problem for most single-image 3D human reconstruction methods.
- **Dataset:** The creation of the DRPOSE15K dataset, which features a measurably higher diversity of poses compared to existing 3D human datasets, is a valuable and positive contribution to the field.
- **Idea:** The core concept of using direct reward fine-tuning to improve a multi-view generator's alignment with specific poses is intuitive.

### Weaknesses
**Major:**

1. **Questionable Novelty & Base Model Choice:** 
The paper's core motivation is undermined by the choice of MIMO as the base pose-conditioned video generator, which produces results with poor multi-view consistency. This is surprising, as other recent models cited in the paper (e.g., the finetuned-MVChamp in IDOL (Instant Photorealistic 3D Human Creation; CVPR’25), original Champ, MimicMotion, SV3D) appear to offer superior texture and consistency. This raises a critical question: **If a stronger, state-of-the-art video generator were used, would the proposed DRPOSE fine-tuning still be necessary?** To validate the generality and necessity of the reward fine-tuning, the authors should test their method on a stronger video generation backbone (e.g., Champ) and present additional results.

2. **Missing Critical Baseline:** As mentioned before, the paper fails to compare against IDOL, which is a highly relevant and important baseline. IDOL also employs a similar concept of data synthesis using a generative video model and achieves state-of-the-art results.
3. **Extremely Poor Qualitative Results:** The visual quality in the teaser (Fig. 1) and other figures shows severe texture and color inconsistency across views. 
4. **Missing Ablation Study:** 
    - The primary technical contribution is the reward fine-tuning (DRPOSE) to improve the multi-view generator. However, the paper lacks an ablation study that *isolates* the effect of this fine-tuning. There should be a direct evaluation (using video generation metrics and pose alignment metrics) of the multi-view generator *before* and *after* the DRPOSE fine-tuning, disentangled from the final reconstruction results, to prove its efficacy.
    - The paper does not provide an ablation study (either qualitative or quantitative) on the impact of the KL divergence regularization term ($L_{KL}$). I’m wondering how much this component contributes or what happens if it is removed.
5. **Dataset Realism:** The textures on the synthesized humans in DRPOSE15K look non-realistic. Does this lack of texture fidelity in the training data limit the model's ability to generalize and produce realistic textures for in-the-wild test images?

**Minor:**

1. **Inadequate Literature Review:** The paper fails to cite or discuss an important category of recent, high-quality optimization-based methods (SDS-based), including:
    - TeCH: Text-guided Reconstruction of Lifelike Clothed Humans
    - Human-SGD: Single-Image 3D Human Digitization with Shape-Guided Diffusion
    - GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data
    - Relevant Gaussian-based reconstruction work, such as the aforementioned IDOL, is also omitted.
2. **Methodological Unclarity & Typos:** 
    - The method description is unclear at points (e.g., the sudden introduction of $x_0$ in L295 without clear definition).
    - L174: "kmage" should be "image-to-multi-view"
    - L231: "Algrotihm" should be "Algorithm"
3. **Insufficient Efficiency Analysis:** In Appendix A.4, a total latency is reported. This would be more useful as a detailed time breakdown for the two main stages: 1) multi-view video generation and 2) human carving/reconstruction.

### Questions
1. Why was a 2D skeleton-image-based alignment chosen for the pose reward? Why not use a more direct 3D-based reward, such as using a pre-trained HMR model to estimate SMPL parameters from the generated images and then applying a 3D joint loss? Is there an ablation study demonstrating that the skeleton-based approach is superior?
2. Why do the quantitative and qualitative results for the PSHuman baseline appear significantly worse in this paper than those presented in the original PSHuman paper and its official project page?

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
This paper presents DRPOSE, a framework designed to improve single-image 3D human reconstruction, particularly for challenging poses. The authors identify limitations in existing datasets and corresponding methods for training and evaluating models for this task. To address these issues, they introduce a post-training approach for image-to-multi-view (I2MV) diffusion models using a differentiable reward function, POSESCORE, to better align the model outputs with ground-truth poses. Furthermore, they contribute DRPOSE15K, a diverse pose dataset for training, and MIXAMORP, a benchmark for evaluating performance on extreme poses. The proposed system is evaluated on several datasets, showing consistent improvements in pose accuracy and reconstruction quality.

### Strengths
- Comprehensive Motivation and Contribution:

The paper starts with a clear motivation, highlighting the limitations of existing datasets and evaluation protocols for single-image 3D human reconstruction. To address these, the authors propose a systematic set of contributions: a novel post-training method (POSESCORE-driven fine-tuning), a new training dataset (DRPOSE15K), and a tailored evaluation benchmark (MIXAMORP). The contributions are well-aligned with the stated motivation and provide a holistic solution to the identified challenges.

- Dataset Contributions (Training and Evaluation):

DRPOSE15K significantly expands the diversity of human poses compared to prior datasets like THuman2.1, offering a valuable resource for training models on challenging poses. Additionally, the MIXAMORP benchmark addresses a critical gap in evaluating reconstruction performance under extreme pose variations, making it a practical addition for future research in this area.

### Weaknesses
- Pipeline Novelty:

While the paper provides a systematic solution, the overall pipeline lacks substantial innovation. The framework is primarily built on an image-to-multi-view (I2MV) diffusion model followed by existing 3D reconstruction techniques. The pipeline primarily follows previous designs(Like PSHuman), with the main novelty lying in the reward-based fine-tuning (POSESCORE) and the dataset contributions.

- Limited Impact of Individual Contributions:

The reward function POSESCORE, while effective, is relatively straightforward and lacks deeper theoretical exploration. Similarly, the dataset and benchmark contributions, while useful, are incremental rather than transformative.

Overall, this is a well-executed paper, though not a groundbreaking one.

### Questions
Your method defines the reward as r(x_0, θ) but uses it to derive the loss function L_reward = 1 - r(x_0, θ). Why do you choose to frame it as a reward instead of directly treating it as a loss?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles the problem of unnatural poses in single-view 3D human reconstruction, particularly for dynamic and challenging movements. The authors propose DRPOSE, a direct reward fine-tuning method for image-to-multi-view (I2MV) diffusion models that doesn't require expensive 3D human assets. Key contributions include: 

DRPOSE15K: A dataset of 15K diverse poses paired with single-view images, constructed from Motion-X motion data and the MIMO pose-conditioned video generator

DRPOSE Algorithm: Post-training method using a differentiable POSESCORE reward that measures consistency between generated multi-view latents and ground-truth poses

MIXAMORP Benchmark: New evaluation benchmark for challenging human poses
Consistent improvements across benchmarks (THuman2.1, CustomHumans, MIXAMORP)

### Strengths
1.Well-motivated focus on pose quality limitations in existing approaches, with quantitative evidence (1.73× larger pose diversity in DRPOSE15K vs THuman2.1)

2.Creative data construction: Leveraging motion capture + video generation to avoid expensive 3D scanning.

### Weaknesses
Circular dependency in data quality: DRPOSE15K relies on MIMO to generate training images, creating a potential bottleneck. If MIMO produces unrealistic appearances for extreme poses, the model learns from flawed data. This isn't adequately discussed or validated.

Insufficient reward model analysis: no ablation on reward formulation alternatives (e.g., direct 3D keypoint prediction, discriminator-based rewards).

Insufficient failure cases are shown, it is hard to evaluate, but the geometry shown in supp videos are pool such as the artifacts in hands, legs and clothes.

Missing references:
SIFU: Side-view Conditioned Implicit Function for Real-world Usable Clothed Human Reconstruction. In CVPR 2024
TeCH: Text-guided Reconstruction of Lifelike Clothed Humans. In 3DV 2024
Generalizable Human Gaussians from Single-View Image. In ICLR 2025
Humangif: Single-view human diffusion with generative prior. In Arxiv 2025
WonderHuman: Hallucinating Unseen Parts in Dynamic 3D Human Reconstruction. In Arxiv 2025

### Questions
Would some samples of DRPOSE15K be shown in to see the quality? Would this be open-sourced?
How different is the proposed method compared to MagicMan which also finetunes MV-diffusion on human dataset?
How loose clothes is handled, how appearance is handled?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper proposes a strategy to enable robust human geometry reconstruction under extreme poses. To achieve this, the DRPose15k dataset and a differentiable reward model fine-tuning strategy are introduced.

### Strengths
- Very well motivated — recent diffusion-based methods fail on complicated poses (e.g., acrobatic poses).

- Well-designed and effective RL-based reward model that is differentiable.

- Strong improvement over SOTA methods (SiTH, PSHuman, etc.).

- Thoroughly studied recent literature.

- The DRPose15k dataset provides extreme-pose ground-truth SMPL-image pairs, which are valuable to the community. The idea of using a pose-conditioned video diffusion model to create this dataset is brilliant, as going in the opposite direction does not work well (i.e., curating extreme-pose RGB images and then using off-the-shelf SMPL estimators).

- Evaluated on multiple datasets, validating the generalizability of this method.

- Overall, I enjoyed reading this paper. Solid results with strong motivation and a well-designed, effective RL-based strategy. I vote for the acceptance of the paper. As this paper can enlighten future works, I encourage the authors to improve the reproducibility in the final revision by providing more implementation details.

### Weaknesses
- Colors seem washed out; I have noticed this in recent diffusion-based single-view human reconstruction methods. Why is this happening, and how can it be improved?

- Reproducibility concern: As suggested in the “Suggestions” section, it would be great if architectural details could be included in the appendix of the final revision.

- Otherwise, I am happy with the current submission.

### Questions
- Which pose-conditioned image-to-video diffusion model was used to create the DRPose15k dataset?

- What happens if the pose-conditioned image-to-video diffusion model fails for extreme poses? Do you manually filter out those noisy outputs? How do such outliers affect model performance?

- Typo in Fig. 2 caption: “Kmage-to-multi-view” → “Image-to-multi-view.”

- Suggestion: In Fig. 5, add a brief explanation of the algorithm. The figure should be self-explanatory.

- Suggestion: Please include architectural details (e.g., video diffusion model, denoising UNet, skeleton decoder, etc.) in the appendix for reproducibility.

### Soundness
4

### Presentation
4

### Contribution
4
