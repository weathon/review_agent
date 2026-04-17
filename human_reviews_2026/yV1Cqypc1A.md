# TAG: Tangential Amplifying Guidance for Hallucination-Resistant Diffusion Sampling

- Decision: Reject
- Scores: 8, 0, 6, 4

## Abstract
Diffusion models achieve the state-of-the-art samples in image generation but often suffer from semantic inconsistencies or *hallucinations*. While various inference-time guidance methods can enhance generation, they often operate *indirectly* by relying on external signals or architectural modifications, which introduces additional computational overhead. In this paper, we propose **T**angential **A**mplifying **G**uidance (**TAG**), a more efficient and *direct* guidance method that operates solely on trajectory signals without modifying the underlying diffusion model. 
TAG leverages an intermediate sample as a projection basis and amplifies the tangential components of the estimated scores with respect to this basis to correct the sampling trajectory. 
We formalize this guidance process by leveraging a first-order Taylor expansion, which demonstrates that amplifying the tangential component steers the state toward higher-probability regions, thereby reducing inconsistencies and enhancing sample quality. 
TAG is a plug-and-play, architecture-agnostic module that improves diffusion sampling fidelity with minimal computational addition, offering a new perspective on diffusion guidance.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces Tangential Amplifying Guidance (TAG), a training-free, plug-and-play inference method for diffusion samplers. At each solver step, the base update is decomposed into a normal (radial) component—aligned with the current latent and a tangential (manifold-following) component. TAG keeps the normal part unchanged (to respect the scheduler’s radius/SNR trajectory) and amplifies the tangential part to steer trajectories along higher-density regions and reduce hallucinations. The paper is motivated from Tweedie’s formula and formalizes it via a first-order Taylor analysis, proving that increasing the tangential weight monotonically increases a local log-likelihood gain for deterministic samplers (Theorem 4.1). Practically, TAG claims to consistently improve unconditional and conditional generation on various models (SD-1.5/2.1/XL/3), with no extra NFEs and tiny runtime deltas. Also, the method composes with prior inference-time guidance (SAG, PAG, SEG).

### Strengths
1. The paper is well-written and diagramed properly.
2. The experiments and empirical observations are coherent with proposed theroem and methods. Also the paper raises well-stated limitations, that too-large η or normalization can harm quality by disturbing radius calibration.
3. The proposed method has broad applicability, composes with existing guidance on multiple samplers (DDIM, DPM++) and with flow‑matching model (SD3) with noticeable quality improvement

### Weaknesses
1. Although the paper does cited geometry‑aware guidance like APG (Sadat, 2024) and TCFG (Kwon, 2025). A direct and controlled comparison (same models/samplers/NFEs) is currently absent.
2. More visual examples are needed to show how the generation result changes after applying C-TAG (Especially on CFG, which is the most commonly used in industry). In the final version, please include a batch of examples comparing the sampling with CFG and C-TAG to better demonstrate the improvement on image quality and diversity.

### Questions
1. Does emphasizing high probability regions via TAG reduce the diversity of generated samples ?
2. Another 2025 paper uses “Temporal Alignment Guidance (TAG)” for a different on-manifold mechanism; recommend a distinct name to avoid confusion
3. The paper claims negligible overhead, can the authors quantify the overhead relative to standard sampling and compare it to other guidance methods?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
The authors propose *TAG* (Tangential Amplifying Guidance) a technique which aims to improve guidance mechanisms in diffusion models and steer the guided samples towards high probability regions. This is accomplished by splitting the guidance term into orthogonal and tangential components and applying a guidance factor $\eta \geq 1$ to the tangential component. The authors then perform some experiments with unconditional and conditional generation.

### Strengths
* I like that the authors also include experiments on flow matching models.
* The author put a lot of details into explaining their algorithm.
* I like that the authors report the CLIP-score.
* The quality of writing is generally good.

### Weaknesses
## Primary concerns
I highlight my concerns with the manuscript below.

### Novelty
I have several concerns regarding TAG and it's novelty within the literature.

* Although the authors mention [1, 2] in the last paragraph of Section 2 they don't really compare to them experimentally or in text. Both of these works seem very similar to the proposed TAG method. In particular the update equation in (6) seems very similar to the update equation $\Delta D_t(\eta) = \Delta D_t^\perp + \eta \Delta D_t^{\|}$ in [1, Section 4].
* The work of [3] also seems highly relevant as they also breakdown into guidance via orthogonal projections.
* Likewise with [4].
* The idea of TAG also seems to be related to works exploring manifold guidance [5] and in particular [6].
* Lots of papers working on inverse problems seem to explore similar ideas, e.g., [5,8] to name a few.
* Authors should also mention the analysis of [9].

### Implementation and theory
* Shouldn't the normal component be labelled $\boldsymbol P_k^\perp$? The projection $\boldsymbol P_k = \mathbf x_k \mathbf x_k^\top$ should be the projection onto the tangent space of $\mathbf x_k$ and not the normal component. Thus I have concerns about the rest of the paper which modifies the the "tangential" component $\boldsymbol P_k^\perp$ via $\eta$. Isn't this just modifying the normal component?
* Further analysis in the paper seems to indicate that $\boldsymbol P_k^\perp$ is the normal projection *a la* line 244. So now I have **deep concerns** about the soundness of the proposed method.


### Empirical
I do not find the empirical results in Section 5 to be satisfactory and I believe there are numerous shortcomings which I outline below.

* My primary issue is the lack of comparison to other techniques which look at orthogonal projection like [1-2].
* I also have concerns that the baselines are unusually poor. For example Figures 5 and 6. Despite, their shortcomings these models can work with CFG or unconditional guidance and produced images for the baselines seem exceptionally poor.
* I am not sure why the authors study unguided generation when TAG is meant for *guidance*.
* While I appreciate the breadth of experiments I feel that comparison to more *relevant* prior works is required.

## Minor concerns or comments
* I dislike the notation
$$\begin{equation}
d \mathbf x_k = \mathbf f(\mathbf x_k, t) dt_k + g(t_k) d \mathbf W_{t_k}
\end{equation}$$
because the equation describes an Ito SDE and not the discretized numerical scheme for which you would have timesteps $\{t_k\}$.
* It would be good to also include the Image Reward metric which has become popular in evaluating conditional generation [7].


## References
[1] Sadat, Seyedmorteza, Otmar Hilliges, and Romann M. Weber. "Eliminating oversaturation and artifacts of high guidance scales in diffusion models." The Thirteenth International Conference on Learning Representations. 2024. https://arxiv.org/pdf/2410.02416

[2] Kwon, Mingi, et al. "TCFG: Tangential Damping Classifier-free Guidance." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/Kwon_TCFG_Tangential_Damping_Classifier-free_Guidance_CVPR_2025_paper.pdf

[3] Zheng, Candi, and Yuan Lan. "Characteristic guidance: Non-linear correction for diffusion model at large guidance scale." ICML 2024. https://arxiv.org/pdf/2312.07586v5

[4] Armandpour, Mohammadreza, et al. "Re-imagine the negative prompt algorithm: Transform 2d diffusion into 3d, alleviate janus problem and beyond." arXiv preprint arXiv:2304.04968 (2023). https://arxiv.org/pdf/2304.04968

[5] He, Yutong, et al. "Manifold Preserving Guided Diffusion." The Twelfth International Conference on Learning Representations. https://arxiv.org/pdf/2311.16424

[6] Sun, Shikun, et al. "SDDM: score-decomposed diffusion models on manifolds for unpaired image-to-image translation." International Conference on Machine Learning. PMLR, 2023. https://proceedings.mlr.press/v202/sun23n/sun23n.pdf

[7] Xu, Jiazheng, et al. "Imagereward: Learning and evaluating human preferences for text-to-image generation." Advances in Neural Information Processing Systems 36 (2023): 15903-15935.

[8] Boys, Benjamin, et al. "Tweedie Moment Projected Diffusions for Inverse Problems." Transactions on Machine Learning Research. https://openreview.net/pdf/6762157974f5480b3bf64dc61ed3b618a6e7772e.pdf

[9] Stanczuk, Jan, et al. "Your diffusion model secretly knows the dimension of the data manifold." arXiv preprint arXiv:2212.12611 (2022). https://arxiv.org/pdf/2212.12611

### Questions
1. How do you differentiate from prior works which also explore orthogonal projections for guidance?
2. Does TAG work for higher-order schemes?
3. Shouldn't the normal component be labelled $\boldsymbol P_k^\perp$?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Tangential Amplifying Guidance (TAG), a new inference-time guidance method for diffusion models based on the geometric decomposition of score updates.
TAG decomposes the diffusion update into radial and tangential components and amplifies only the tangential direction — which the authors argue carries the data-relevant semantic structure — while leaving the radial (noise) direction untouched.
This yields improved sampling fidelity, reduced hallucinations, and better semantic consistency without any architectural modification or retraining.

The paper provides:

• a theoretical justification (via Tweedie’s identity and Taylor expansion) that tangential amplification monotonically increases the local log-likelihood gain;

• an algorithmic formulation that is plug-and-play with standard samplers (DDIM, DPM++);

• and comprehensive experiments on multiple backbones (SD v1.5, SD v2.1, SDXL, SD3), both unconditional and conditional (CFG, PAG, SEG).

Overall, TAG achieves consistent improvements in FID, IS, and CLIPScore across settings, often outperforming baselines at fewer sampling steps.

### Strengths
Strong conceptual novelty.
The idea of separating the score update into radial/tangential components and amplifying only the tangential direction provides a geometric interpretation of guidance in diffusion models.

1. Solid theoretical analysis. The use of Tweedie’s identity and local log-likelihood expansion leads to a provable monotonic improvement theorem (Theorem 4.1).
The paper maintains a balance between geometric intuition and formal justification.

2. Empirical breadth and consistency. The authors evaluate across four diffusion families (DDIM, SD v1.5/v2.1, SDXL, SD3) and multiple guidance types (CFG, PAG, SEG).
Results are consistent and substantial, with up to ~20% FID reduction without additional NFE or retraining.

3. Plug-and-play practicality. TAG is architecture-agnostic, requires no finetuning, and adds negligible computational cost — a strong practical advantage.This makes it a likely method to be quickly adopted in practice.

4. Comprehensive ablations. The η ablation and the “avoidance of normal amplification” section clearly demonstrate the tradeoff between fidelity and over-smoothing, providing useful implementation guidance.

5. Readable and well-organized. The paper is clearly written, with consistent notation and good figures illustrating the geometric intuition.

### Weaknesses
1. Fixed amplification factor (η).The method uses a global η for all timesteps. While the authors note in §6 that adaptive ηₖ could further improve performance, the current approach might not optimally handle varying signal-to-noise ratios across timesteps. A simple experiment on dynamic ηₖ would strengthen the work.

2. Limited discussion of generalization beyond images. TAG is conceptually general, but all experiments are on 2D image diffusion.
A brief experiment (even qualitative) on another modality (e.g., audio, latent diffusion for text, or 3D) could make the impact broader.

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The work proposes to separate the sampling denoising process into two direction at each timestep. One direction is normal representing how noisy the data is and one direction is tangential representing the main content of the image. The work proposes a guidance to enhance tangential direction so that improves the fidelity without introducing external information. The results show some significant improvement.

### Strengths
1. The paper is well written and easy to understand
2. The proposed method is intuitive and well motivated theoretically

### Weaknesses
The main concerns lie in the baselines of the paper. 

1. Most of the baselines in the paper are not very consistent with what we have in the literature. Most of the FID score is around 100 which is too high compared to all the literatures we have known which is around 1 to 10 in ImageNet. The CLIP scores from SDv2.1 is also too low 22.x which is significant lower than the number reported in other papers around 30.x
2. The baselines in the paper are from DDIM/DDPM which are too old. While these baselines are useful, we would want to expect to see newer baselines such as EDM-2, Flux, Flow matching models which are not reported in the paper. In table 5, we have flow matching number but the FID is 150 for ImageNet, this number is too high to tell if the generated data is useful.

### Questions
Please see the weaknesses

Please consider making the reported numbers more consistent with the literature, so that we can see what it really can improve. Otherwise, improvement on poor baselines make no sense in practice. I will increase the score after fixing the concerns.

### Soundness
3

### Presentation
3

### Contribution
3
