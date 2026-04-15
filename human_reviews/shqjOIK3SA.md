# Towards Understanding the Robustness of Diffusion-Based Purification: A Stochastic Perspective

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Diffusion-Based Purification (DBP) has emerged as an effective defense mechanism against adversarial attacks. The success of DBP is often attributed to the forward diffusion process, which reduces the distribution gap between clean and adversarial images by adding Gaussian noise. Although this explanation is theoretically grounded, the precise contribution of this process to robustness remains unclear. In this paper, through a systematic investigation, we propose that the intrinsic stochasticity in the DBP procedure is the primary factor driving robustness. To explore this hypothesis, we introduce a novel Deterministic White-Box (DW-box) evaluation protocol to assess robustness in the absence of stochasticity, and analyze attack trajectories and loss landscapes. Our results suggest that DBP models primarily leverage stochasticity to evade effective attack directions, and that their ability to purify adversarial perturbations can be weak. To further enhance the robustness of DBP models, we propose Adversarial Denoising Diffusion Training (ADDT), which incorporates classifier-guided adversarial perturbations into diffusion training, thereby strengthening the models' ability to purify adversarial perturbations. Additionally, we propose Rank-Based Gaussian Mapping (RBGM) to improve the compatibility of perturbations with diffusion models. Experimental results validate the effectiveness of ADDT. In conclusion, our study suggests that future research on DBP can benefit from the perspective of decoupling stochasticity-based and purification-based robustness.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper examines how stochastic elements contribute to robustness in DiffPure by systematically ablating these components (DW-box). The authors propose incorporating adversarial training into DiffPure and addressing the "robustness-generalization tradeoff" through a novel technique, RGBM, which preserves both robustness and generative performance.

### Strengths
I like the following contributions in this paper:

- The authors systematically study how each of the stochastic elements in DiffPure contributes to robustness, including the "forward process," "backward process," and the combination of both, clearly demonstrating how much each component contributes to robustness.

- The authors propose RGBM, which significantly alleviates the previous limitation of adversarial training compromising the generative ability of diffusion models.

- Fig. 5 is excellent. Without Fig. 5, it would be challenging to understand RGBM clearly due to existing ambiguities (see question 3 on the two types of RGBM).

### Weaknesses
There are no severe weaknesses in this paper, but one notable issue:

- "Note that DDPM is mathematically equivalent to the DDPM++ (VPSDE) model used in DiffPure but employs a smaller UNet architecture. DDIM, on the other hand, is a mathematical improvement over DDPM that facilitates an ODE-based diffusion process." This statement lacks formality. DDPM is a mathematical framework independent of neural network architecture. Additionally, the DDPM framework represents only one discretization of the VP-SDE, while other discretizations also exist. The authors should address this non-formal claim.

### Questions
I'm quite curious about the following (I would like to raise my score if the authors could provide more comprehensive studies on the mechanism of their proposed methods):

- Are the points in Fig. 1 obtained by sampling at intervals of a certain number of attack steps? If so, does this imply that the attack process initially moves the farthest distance and moves the fastest, while the loss increases the slowest, and then the movement gradually slows as time progresses but the loss increases more rapidly?

- How did you come up with RGBM? What inspired this approach?

- RGBM uses values from a Gaussian distribution but follows the order of adversarial perturbations. What would happen if we used the order of Gaussian noise but the values of adversarial perturbations? I’m curious if the main mechanism of RGBM is to eliminate large (outlier) values.

- I am very interested in whether "ADDT w/o RGBM" and "ADDT" in Table 6 were fine-tuned from pretrained models or trained from scratch. Was "ADDT w/o RGBM" fine-tuned from "ADDT"?

- Can I understand RGBM as maintaining robustness while improving generalization and generation performance across different attacks?

- I am very curious about why RGBM works. First, I notice that the result mapped by RGBM does not follow a Gaussian distribution, nor even a non-isotropic Gaussian (since each element is not Gaussian). Thus, the RGBM-mapped result is a biased estimator rather than an unbiased one. The only similarity I see to a Gaussian distribution is that, like a high-dimensional Gaussian, it also lies on a spherical shell. Is this the main reason it preserves the generalization of pretrained diffusion models? If so, simply "normalizing" the adversarial perturbation to an \(\ell_2\) norm of \(O(\sqrt{D})\) should also work (since the \(\ell_2\) norm of a high-dimensional Gaussian converges to \(\sqrt{D}\)). A simple way like yours (your illustration) to understand this is first to sample a Gaussian noise, compute its norm, and normalize the adversarial perturbation to this norm. Could you provide an ablation study on this?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper explores the reason behind the success of Diffusion-Based Purification (DBP) as a defense against adversarial attacks, highlighting that its effectiveness comes from randomness rather than just reducing the difference between clean and adversarial images. The authors introduced a new evaluation method, Deterministic White-Box (DW-box), and found that DBP’s ability to purify adversarial perturbations is weak without stochasticity. To improve DBP, they developed Adversarial Denoising Diffusion Training (ADDT), which incorporates adversarial examples into the training process, and Rank-Based Gaussian Mapping (RBGM), which makes adversarial perturbations more Gaussian-like. Their experiments showed that these methods enhance DBP’s robustness.

### Strengths
1) The paper is well written and relatively easy to follow.
2) Novel perspective on why Diffusion-Based Purification works and how it can be further improved.
3) Claims are well supported by experiments.

### Weaknesses
N/A

### Questions
1) Based on the conclusion of your paper that stochasticity is the main ingredient of the success of the DBP, do you think a diffusion model even necessary for the defense? What if we just add a certain amount of gaussian noise directly to the adversarial example and then just denoise it with an off-the-shelf image denoising network that was specifically trained to remove gaussian noise? Do you think it will work just as well as a defense method?

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
4

### Summary
This paper investigates the robustness of Diffusion-Based Purification (DBP) methods in adversarial attacks, proposing that its defensive effectiveness relies primarily on stochasticity rather than the traditional noise ablation process. By introducing a deterministic white-box evaluation approach, the authors demonstrate the critical role of stochasticity in DBP's robustness. To enhance DBP's denoising capability, the paper presents Adversarial Denoising Diffusion Training (ADDT) and Rank-Based Gaussian Mapping (RBGM), with experimental results showing significant improvements in defense performance. The authors suggest that future research should focus on decoupling stochasticity from purification capability to optimize defensive performance.

### Strengths
1. The paper identifies that the robustness provided by diffusion models is likely due to stochastic elements within the model rather than solely the diffusion process.
2. The authors propose a new training method for diffusion models that incorporates robustness information, enhancing the model's purification capability.

### Weaknesses
1. While the study contributes positively to understanding purification using diffusion models, there are concerns about the complexity of purification. It significantly increases the model's complexity, potentially multiple times compared to the original model, and this real-world cost needs to be experimentally clarified (complexity analysis).
2. If the training data for the diffusion model originates from the same data as the original model, a comparison with methods using the diffusion model to generate new samples for robust model training would be valuable, as this approach does not add runtime complexity [1].
3. Questions remain about why perturbations are not directly modified by mean and variance adjustments to achieve Gaussian distribution, instead requiring replacement.

**Reference**:

[1] Wang, Zekai, et al. "Better diffusion models further improve adversarial training." International Conference on Machine Learning. PMLR, 2023.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author argues that the current effectiveness of the DBP method, attributed to the forward process of diffusion models, lacks sufficient empirical validation. While they acknowledge the theory of mitigating distribution gap, they question whether the forward and backward diffusion stochasticity in DBP is indeed what enhances model robustness. To address this, they propose a new attack evaluation, the DW-Box attack, which captures both stochasticity information and parameters of defense system. Experiments reveal that models like DDPM and DDIM lack strong robustness against this new attack. Additionally, the authors introduce ADDT, a fine-tuning technique for diffusion models in the existing DBP methods. This approach first optimizes the generated adversarial perturbation using classifier guidance, and then performs further training on images with both Gaussian noise and optimized perturbations. They aim to enhance the purification capability of diffusion models. This method significantly enhances the robustness of existing DBP methods. Overall, the paper critically examines the underlying principles that make DBP methods effective.

### Strengths
1. The work has included extensive experiment contents, such as exploration on accelerated generation of diffusion model, stronger white-box attack with more attack iterations and included various dataset results.

2. The author proposed an appropriate attack evaluation, DW-box attack, which makes the performance comparison reliable.

3. The ADDT technique has improved the robustness of all baseline DBP methods.

### Weaknesses
1. The author should provide a few more comparisons for the same content in figure 6, which shows the performance of the ADDT fine-tuning under the proposed DW-box attack. 

2. In figure 3, it is not convincing if the author only draws one complete attack trajectory. 

3. The whole idea can be viewed as an additional training by using samples with more Gaussian noises at the same timestep, hence the diffusion model achieves better purification ability, I would like to see whether the RBGM perturbation is different with using a pure additional Gaussian noise with trivial order. Hence, we cannot say the model has improved its ability of enhancing purification of adversarial perturbations.

### Questions
1. As the perturbation after Rand-Based Gaussian Mapping is still a Gaussian noise, can I view the idea of ADDT is just using further perturbed images for additional training, if so, is the improvement relying on the additional data?

2. What is the reason of choosing such a range for the modulation term in line 362-364?

### Soundness
4

### Presentation
3

### Contribution
4
