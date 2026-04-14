# FLOWDREAMER: EXPLORING HIGH FIDELITY TEXT-TO-3D GENERATION VIA RECTIFIED FLOW

- Decision: Reject
- Scores: 5, 3, 5, 6, 5

## Abstract
Recent advances in text-to-3D generation have made significant progress. In particular, with the pretrained diffusion models, existing methods predominantly use Score Distillation Sampling (SDS) to train 3D models such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3D GS). However, a hurdle is that they often encounter difficulties with over-smoothing textures and over-saturating colors. The rectified flow model – which utilizes a simple ordinary differential equation (ODE) to represent a straight trajectory – shows promise as an alternative prior to text-to-3D generation. It learns a time-independent vector field, thereby reducing the ambiguity in 3D model update gradients that are calculated using time-dependent scores in the SDS framework. In light of this, we first develop a mathematical analysis to seamlessly integrate SDS with rectified flow model, paving the way for our initial framework known as Vector Field Distillation Sampling (VFDS). However, empirical findings indicate that VFDS still results in over-smoothing outcomes. Therefore, we analyze the grounding reasons for such a failure from the perspective of ODE trajectories. On top, we propose a novel framework, named FlowDreamer, which yields high-fidelity results with richer textual details and faster convergence. The key insight is to leverage the coupling and reversible properties of the rectified flow model to search for the corresponding noise, rather than using randomly sampled noise as in VFDS. Accordingly, we introduce a novel Unique Couple Matching (UCM) loss, which guides the 3D model to optimize along the same trajectory. Our FlowDreamer is superior in its flexibility to be applied to both NeRF and 3D GS. Extensive experiments demonstrate the high-fidelity outcomes and accelerated convergence of FlowDreamer. Moreover, we highlight the intriguing open questions, such as initialization challenges in NeRF and sampling techniques, to benefit the research community

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper presents Vector Field Distillation Sampling (VFDS) to integrate Score Distillation Sampling (SDS) with rectified flow models. The authors further enhance VFDS by substituting randomly sampled noise with pushed-back noise. The authors then experiment with rectified flow model SDv3 to validate their algorithm.

### Strengths
1. The qualititive results are generally good. And the presentation is clear.

2. The authors conducted experiments across various representations, including 3D-GS and NeRF, to validate their algorithm.

### Weaknesses
1. **Lack of Novelty**: The paper presents very limited novelty, as rectified flow is merely a specific case within the broader family of diffusion models. Generalizing SDS-like methods to rectified flow is trivial. Specifically, for the forward diffusion process:

   $$
   \boldsymbol{x}_t = a_t \boldsymbol{x}_0 + b_t \boldsymbol{\epsilon}, \tag{1}
   $$

   Rectified Flow sets $ a_t = 1-t $ and $ b_t = t $. With this schedule, the prediction of rectified flow model SDv3 can be reformulated to epsilon prediction according to the equation below Eq. (12) in SDv3 [1]:

   $$
   \boldsymbol{\epsilon}_\phi = (1-t) \boldsymbol{v} _\phi + \boldsymbol{x}_t. \tag{2}
   $$

   This means that all existing SDS-like methods (which are usually formulated in epsilon prediction) can be effectively adapted to the rectified flow model SDv3 by changing the prediction type using Eq. 2. Consequently, re-deriving VFDS and VF-ISM in the paper is unnecessary, as they are trivially equivalent to SDS and ISM. This re-derivation may not offer substantial or particularly inspiring insights. The authors need to clarify what specific novel contributions or challenges they encountered in applying this approach to text-to-3D generation.

2. **Unfair Comparison**: The comparisons presented in this paper appear to be unfair. As stated in Section 4.1, the baselines utilize the SDv2.1 prior, while this work employs the SDv3 diffusion prior as claimed in Appendix section 2. The improvements reflected in the quantitative results in Tables 1 and 2 likely stem from the upgrade of the base diffusion model. Furthermore, as noted in weakness 1, upgrading other baselines to SDv3 is almost trivial. There should be no hindrance to adopting the same base diffusion model for a fair comparison. The authors should include equitable comparisons in their paper.

3. **Limited Novelty of UCM Loss**: Given the triviality of upgrading SDS-like methods to rectified flow models, the proposed UCM loss bears a strong resemblance to the loss used in ISM, further limiting the contribution of this paper. Both methods employ an inversion process to enhance the noising methods. The primary distinction between UCM and prior work ISM appears to be that ISM reverses to timestep $ t $, while UCM inverts to the maximum timestep and combines the noise with the rendered image at timestep $ t $. The authors should further explain on the difference between ISM and their UCM loss and the potiential improvements of UCM compaired with ISM. The result presented in Fig. 7 demonstrated some effectiveness, but further comparison are necessary to enhance the arguments. An additional fair quantitative comparison is necessary to demonstrate the improvements of UCM, and I would likely adjust my rating based on the outcomes of these experiments.

### Questions
1. Have I accurately captured the main difference between the proposed algorithm and ISM in weakness 2? I suggest that the authors include a table comparing their methods with ISM for enhanced clarity, given the significant resemblance.

2. Did the authors encounter any challenges when upgrading the base diffusion model to SDv3 for the other baselines using Eq. (2)?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduces a new framework called FlowDreamer, using a FM based model to update a 3D generator, e.g., Nerf or GS. 

The whole framework mainly contains 2 part: 
1. a score distillation based on FM model (named VFDS in paper) and 
2. a push-back method (named UCM in paper) to align noise and rendered scene. 
 

Finally, extensive experiments demonstrate that the FlowDreamer framework achieves high-fidelity and fast convergence across various 3D generation settings, such as NeRF and 3D GS.

### Strengths
This paper involves extensive engineering work to distill a high-quality 3D generator using the SD3 model as the foundational framework.

### Weaknesses
I have significant concerns regarding various aspects of this paper.

1. The paper suffers from a lack of citations, particularly in relation to the references concerning UCM. Similar methods have already been extensively applied in DMD2 [1] and Imagen Flash [2]. The author state UCM is improved from ISM but I rather suggest check the given reference and rethink the novelty of UCM.

2. The authors have not provided adequate experimental evidence to validate the effectiveness of the individual components of their approach. Although they demonstrate strong overall performance metrics, there is no experimental support showing that the use of **straight trajectories contributes to performance improvements**. 

    As FM is simply a parameterized special case of Diffusion. I am not convinced that straight trajectories will have a substantial effect on score distillation. Moreover, in the case of Diffusion, higher-order scores also follow straight trajectories. As such, I would require the authors to present experimental evidence demonstrating the improvements claimed to result from the use of straight trajectories.

3. Furthermore, we observed that the methods used for comparison were based on SD2.1, while the authors employed SD3 as the teacher model, which constitutes an inherently unfair comparison. Notably, this issue was not mentioned in the main manuscript. I am concerned that the observed strong alignment between the text and 3D objects may be attributed to SD3’s MMDIT, rather than the method introduced by the authors.




[1] Yin, Tianwei, et al. "Improved Distribution Matching Distillation for Fast Image Synthesis." arXiv preprint arXiv:2405.14867 (2024).

[2] Kohler, Jonas, et al. "Imagine flash: Accelerating emu diffusion models with backward distillation." arXiv preprint arXiv:2405.05224 (2024).

### Questions
see weakness. 

I believe that this work demonstrates the potential of distilling a 3D generator using SD3 as the teacher model. However, the current claims put forward in the paper lack strong experimental validation, and I hold reservations about some of the arguments presented by the authors. For these reasons, I recommend rejecting this paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper proposes using a flow matching model as a replacement for the diffusion model within the SDS framework. It offers an analysis of data and noise coupling alongside the reversible properties in flow matching, aiming to mitigate the noisy gradient signals inherent in the SDS framework by back-tracking the corresponding noise for each current rendered view. Empirical results demonstrate an improved CLIP score compared to previous approaches.

### Strengths
- the paper is written in good clarity
- one of the first papers incoporating flow matching to SDS loss and exploit the flow matching coupling/reversible property to improve SDS stability

### Weaknesses
- The paper asserts that “the trajectory of the rectified flow model is straight, v_phi(xt, t), and remains approximately constant for different t under ideal conditions.” However, this assumption appears incorrect. Even if trained with a straight velocity, the rectified flow typically displays a curved trajectory, with flow paths re-routed at intersections to prevent crossings. This phenomenon is discussed and illustrated in Fig. 2(b) [1].
- Given this questionable assumption, it also raises doubts about the claimed advantages of VFDS in speed and stability over SDS. Although the paper suggests that VFDS converges more rapidly than the diffusion model and that FlowDreamer converges faster than VFDS, it provides only one qualitative example in the appendix without sufficient large-scale quantitative evidence or convincing benchmark metrics.
- The authors state, “in contrast, we propose a novel UCM loss based on the push-backward process to identify corresponding noise, rather than using randomly sampled noise as in rectified flow-based models.” However, prior work in LucidDreamer demonstrates a similar approach by eliminating randomness in xt through deterministic DDIM inversion, thereby increasing pseudo-GT consistency. This paper should acknowledge the correlation with LucidDreamer and correct its claims. Furthermore, as DDIM is essentially an ODE solver [2], deterministic DDIM inversion closely resembles the noise back-tracking proposed in this work. Both aim to reduce randomness, indicating that the novelty of this approach may be limited.
- Although FlowDreamer is proposed primarily to address over-smoothing issues, the paper provides only three qualitative results in Figures 7 and 8. One could even argue that in Figure 8 the level of detail between VFDS and FlowDreamer appears similar with only different styling, raising questions about the model’s effectiveness in resolving over-smoothing.
- Referring to [3], flow matching models generally perform better than diffusion models. Given the marginal CLIP score improvement in Tables 1 and 2, it remains unclear if the performance gain is due to replacing the diffusion model with the flow matching model.
- The authors highlight a limitation of FlowDreamer in its early stages when the rendered view x falls outside the target data distribution, causing instability in the back-tracked noise epsilon. To address this, a warm-up phase is required, with the initial rendered views needing to be within reasonable limits. However, the paper lacks a detailed ablation study of this two-phase training framework to validate its robustness. For instance, how does varying the warm-up duration impact performance? Is the warm-up phase length dependent on the example’s complexity (e.g., more complex scenes requiring a longer warm-up)?
- ProlificDreamer [4] demonstrates that by using a learnable noise prediction network, the CFG weight can be reduced to a standard range (e.g., 7.5) to yield reasonable results. Given that FlowDreamer is expected to produce more stable noise, it is unclear why a very high CFG value is still required to achieve detailed textures (i.e., ≥40 as shown in Figure 8).

[1] Liu, X., Gong, C. and Liu, Q., 2022. Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003.

[2] Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C. and Zhu, J., 2022. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. Advances in Neural Information Processing Systems, 35, pp.5775-5787.

[3] Ma, N., Goldstein, M., Albergo, M.S., Boffi, N.M., Vanden-Eijnden, E. and Xie, S., 2024. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740.

[4] Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H. and Zhu, J., 2024. Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. Advances in Neural Information Processing Systems, 36.

### Questions
- Can you elaborate on why you assume the trajectory of the rectified flow model remains straight across different t values? How do you address evidence suggesting that rectified flows are typically curved and adjusted to avoid intersection points?
- Could you provide additional quantitative evidence or large-scale experiments to substantiate the claim that VFDS converges faster and is more stable than SDS?
- How does FlowDreamer differenciates with LucidDreamer's deterministic DDIM inversion?
- Could you include a more comprehensive analysis in either 2D/3D to demonstrate FlowDreamer does improve over-smoothing issue significantly compared to VFDS baseline?
- Given that flow matching models often outperform diffusion models [3], how do you confirm that the improvement in CLIP scores (Tables 1 and 2) is due to your proposed methods rather than merely switching to a flow matching model?
- Could you provide an ablation study or detailed analysis to clarify the role of the warm-up phase in the two-stage training framework? Specifically, how does varying the warm-up duration impact model performance and stability?
- Given that FlowDreamer also stabilizes noise, why is a much higher CFG weight (e.g., ≥40) required to achieve detailed textures in FlowDreamer?

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
This paper proposed to adopt the pre-trained rectified flow model for text-to-3D generation. Considering the different network predictions, it modified the SDS to match the formulation of rectified flow model . Moreover, push-backward process is used to estimate noise, which provides more consistent supervision signals, leading to highly detailed objects generation.  Comprehensive experiments are conducted to valid the efficacy of proposed methods.

### Strengths
1.  This paper claims the first exploration using rectified flow model for text-to-3D generation.
2.  Their results outperform previous SOTA method (i.e., lucidDreamer)
3.  The push-backward process enhances the details of generated objects.

### Weaknesses
1. This paper is based on previous observations. For example, the over-smoothed results are caused by noisy gradient. LucidDreamer solved this by replacing sampled random noise with noisy latents derived from DDIM inverse. This paper also use  similar inverse process, i.e., push-backward process, suitable for rectified flow model. Therefore, this article mainly studies how to adapt previous findings to more advanced rectified flow model, which is not interesting enough.
2.  There are several questions about results. 1) In fig.7, does VF-ISM use the same 3D parameters as flowdreamer? 2) LucidDreamer discards some terms for saving time, which is not optimal. Can you show results using Eq.12 in LucidDreamer with the proposed rectified flow model + push-backward process?  3) In Fig. 8, VFDS with large CFG has good details, does that mean UCM loss is not essential? Please show more cases and cases with larger CFG.

### Questions
See weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This study proposes FlowDreamer to leverage a pretrained text-to-image (T2I) models trained via the rectified flow framework for score distillation sampling. After explaining the training objective of SDS with a flow model, called VFDS, this paper claim that the cause of underperformance results from using multiple noises during optimization. Thus, considering the mapping between a noise and an image, FlowDreamer first use a flow model to conduct the rendered image inversion to noises, and then apply VFDS. FlowDreamer shows better results than existing methods and VFDS for text-to-3D generation

### Strengths
S1. This study successfully incorporate a T2I flow model in text-to-3D generation.

S2. Compared to VFDS, the proposed noise sampling techniques shows improvements in generation quality even with a few inversion steps.

### Weaknesses
W1. The comparison with existing methods are unfair. Since the existing methods use Stable-Diffusion (SD) 2.1, the performance difference can come from the differences in used T2I models, considering SD 3, which is used for the proposed method, has better T2I performance than SD 2.1.

W2. Unclear contributions. The major contribution lies in the consideration of the characteristics of flow models, different from diffusion models, for text-to-3D generation. However, considering DDIM also conducts a deterministic sampling, and an image-noise pair can be coupled, computing the noise inversion is also effective to diffusion model. For example, DDIM inversion can also be used to resolve the over-smoothing and unrealistic outputs with SDS via diffusion models [NewRef-1].

[NewRef-1] Lukoianov et al., Score Distillation via Reparametrized DDIM, NeurIPS'24.

### Questions
In addition to W1 & W2 above, the authors can also provide answers for the questions below:

Q1. The authors claimed that FlowDreamer shows faster convergence, but I can't find the detailed analysis in training efficiency except for Figure 7, which shows minor improvements in training speed. Could the authors provide more analysis on convergence speed?

### Soundness
2

### Presentation
2

### Contribution
2
