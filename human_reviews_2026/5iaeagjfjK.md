# One-Step Flow for Image Super-Resolution with Tunable Fidelity-Realism Trade-offs

- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8, 4

## Abstract
Recent advances in diffusion and flow-based generative models have demonstrated remarkable success in image restoration tasks, achieving superior perceptual quality compared to traditional deep learning approaches. However, these methods either require numerous sampling steps to generate high-quality images, resulting in significant computational overhead, or rely on common model distillation, which usually imposes a fixed fidelity-realism trade-off and thus lacks flexibility. In this paper, we introduce OFTSR, a novel flow-based framework for one-step image super-resolution that can produce outputs with tunable levels of fidelity and realism. Our approach first trains a conditional flow-based super-resolution model to serve as a teacher model. We then distill this teacher model by applying a specialized constraint. Specifically, we force the predictions from our one-step student model for same input to lie on the same sampling ODE trajectory of the teacher model. This alignment ensures that the student model's single-step predictions from initial states match the teacher's predictions from a closer intermediate state. Through extensive experiments on datasets including FFHQ (256$\times$256), DIV2K, and ImageNet (256$\times$256), we demonstrate that OFTSR achieves state-of-the-art performance for one-step image super-resolution, while having the ability to flexibly tune the fidelity-realism trade-off. 
Code and pre-trained models are available at \href{https://github.com/yuanzhi-zhu/OFTSR}{https://github.com/yuanzhi-zhu/OFTSR} and \href{https://huggingface.co/Yuanzhi/OFTSR}{https://huggingface.co/Yuanzhi/OFTSR}, respectively.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors consider the super-resolution problem and propose a method to distill a pretrained flow model in a generator, which produces output on the PF-ODE trajectory for a given time “t” in 1 step. The ability to produce output at any “t” on the ODE trajectory provides a way to control the fidelity-realism trade-off, since for a bigger time “t”, samples are better in “realism” while for less time “t”, samples are better in “fidelity”. The authors evaluate their method on noisy and noiseless SR setups, and conduct some evaluations on real SR setups.

### Strengths
- The proposed method provides a clear and simple method to control the fidelity-realism tradeoff by choosing the time parameter “t” in the generator.
- The distilled model works in 1 step without significant degradation of the quality of generation.

### Weaknesses
- The proposed method is not theoretically novel in the sense that it is a special case of the general forward-distillation framework.
- Most of the evaluations and comparisons are done with simple downsampling or downsampling plus the addition of noise setups. For RealSR, only non-reference metrics are provided. For the ImageNet comparison, only one competitor distillation method (SinSR) is considered, while other competitive methods like OSEdiff are omitted.
- The authors compare with I2SB, but not with its various accelerated I2SB versions [1, 2, 3], which is strange since the proposed method is also distillation.

[1] Wang Y. et al. Implicit Image-to-Image Schrödinger Bridge for image restoration //Pattern Recognition. – 2025. – Т. 165. – С. 111627.

[2] He G. et al. Consistency diffusion bridge models //Advances in Neural Information Processing Systems. – 2024. – Т. 37. – С. 23516-23548.

[3] Gushchin N. et al. Inverse Bridge Matching Distillation //arXiv preprint arXiv:2502.01362. – 2025.

### Questions
Is there a way to automatically choose a time for the generator?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces OFTSR, a novel one-step image super-resolution (SR) method using a flow-based framework that incorporates tunable fidelity-realism trade-offs. It leverages a teacher-student distillation approach, with a noise-augmented conditional flow model for SR. The student model is trained through a distillation strategy, aligning its predictions along the probability flow ordinary differential equation (PF-ODE) trajectory of the teacher. The method is demonstrated to achieve state-of-the-art performance for one-step SR on several datasets like FFHQ, DIV2K, and ImageNet. A unique feature of OFTSR is its ability to adjust the fidelity-realism trade-off via a single hyperparameter.

### Strengths
1. The proposed OFTSR integrates flow modeling with distillation, achieving high-quality super-resolution in a single inference step. It reduces inference cost by over an order of magnitude compared to diffusion-based methods, while maintaining comparable perceptual quality.

2. The approach’s ability to control the fidelity-realism trade-off through a single hyperparameter adds significant flexibility. OFTSR balances the perceptual quality and fidelity of SR results.

3. The training cost is less than previous methods, and the method seems to be a general and extensible solution for fast generative restoration.

### Weaknesses
1. The paper does not compare with recent state-of-the-art methods such as OSEDiff [1], AddSR [2], CTMSR[3], and TSDSR [4]. This omission makes it difficult to fully evaluate the competitiveness of the proposed method in the context of the latest advances in this field.

2. The experiments are mainly conducted at a resolution of 256×256, without investigating the model’s generalization capability at higher resolutions.

3. The paper lacks an ablation study on SFT a one-step model initialized from the teacher model. I suggest the authors include this experiment to better demonstrate the effectiveness of the proposed method.

[1] One-Step Effective Diffusion Network for Real-World Image Super-Resolution. NeurlPS2024

[2] AddSR: Accelerating Diffusion-based Blind Super-Resolution with Adversarial Diffusion Distillation.

[3] Consistency Trajectory Matching for One-Step Generative Super-Resolution. ICCV 2025

[4] TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution. CVPR 2025

### Questions
How sensitive is the fidelity–realism control parameter $t$ to different datasets or degradation types? Can it be automatically optimized for a given target domain?

---

If the above weaknesses and questions are addressed, I would consider increasing my rating.

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
This paper introduces OFTSR, a flow-based one-step super-resolution (SR) framework that builds upon rectified flow models and distillation. The key idea is to train a conditional flow-based SR teacher and distill it into a single-step student whose predictions are constrained to lie on the same probability flow ODE trajectory as the teacher. This constraint allows the one-step model to maintain a tunable fidelity–realism trade-off controlled by a continuous parameter (t). Experiments on FFHQ, DIV2K, and ImageNet show that OFTSR achieves competitive or state-of-the-art performance among one-step methods, while being more efficient than multi-step diffusion-based baselines.

I found this to be a well-executed and empirically strong paper, but the conceptual innovation is somewhat limited. The proposed ODE alignment distillation is a reasonable technical improvement, yet it does not introduce a fundamentally new idea beyond existing consistency and distillation methods. The experimental section is solid, but the lack of theoretical depth and real-world validation makes it fall slightly short of the bar for acceptance. That said, the work is promising and practical, and I could imagine it being accepted after stronger justification of its theoretical grounding and broader validation.

### Strengths
- The paper’s formulation is clear, the equations are consistent with prior work in diffusion and flow-based modeling, and the proposed ODE alignment loss is a neat way to adapt teacher–student distillation for conditional SR.
- The experimental section is impressive in scope. The authors benchmark across three datasets, consider both noisy and noiseless conditions, include training-free and training-based baselines, and conduct detailed ablations on hyperparameters and solvers.
- The method achieves true one-step inference while preserving control over perceptual vs. fidelity quality, which is a meaningful step forward for efficient SR.
- The paper is well-written and easy to follow, with intuitive figures that make the underlying intuition accessible.

### Weaknesses
- Conceptually, the work is an incremental extension of existing distillation or consistency-based methods (e.g., BOOT, MeanFlow, Consistency Models). The ODE-alignment constraint is interesting but feels more like a technical refinement than a new paradigm. This limits the paper’s theoretical depth.
- The ODE alignment loss is mostly empirically motivated. There’s no clear analysis of when or why this constraint ensures better one-step consistency, nor how it relates formally to the underlying PF-ODE or optimal transport formulations.
- Although the experiments are extensive, most are performed under controlled synthetic conditions. The “real-world” SR experiments use degradations simulated with RealESRGAN, which doesn’t fully validate real-world robustness. I would have liked to see more genuinely real test data or domain-shift scenarios.
- All baselines are diffusion/flow-based. The paper does not compare against any deterministic SR networks (e.g., SwinIR, NAFNet), which are used as efficiency benchmark in practice.
- The paper doesn’t discuss whether OFTSR maintains diversity in outputs (e.g., multiple plausible HR reconstructions per LR input) or generalizes across scales/resolutions. These are important in assessing flow-based generative SR methods.
- While the interpolation parameter t is central to the paper, there’s little analysis of its sensitivity or failure modes. Some visualizations (e.g., at extreme t) would help validate robustness.

### Questions
1) How sensitive is OFTSR to the specific choice of teacher (e.g., if trained with different flow objectives)?
2) Can your ODE alignment loss be extended to a few-step regime (e.g., 2–4 NFEs)?
3) Is the model capable of diverse sampling (multiple HRs per LR) or is it deterministic?
4) How does performance compare to lightweight CNN/transformer SR models under equal runtime?
5) Could you provide visual examples showing failure or instability at extreme t values?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces OFTSR, a one-step image super-resolution method that preserves flexible control over the fidelity-realism trade-off. The approach uses a two-stage pipeline: first, training a noise-augmented conditional rectified flow model (teacher) that expands the initial distribution by adding Gaussian noise to low-resolution images, then distilling it into a one-step model (student) by constraining predictions to lie on the same ODE trajectory as the teacher. At inference, users can adjust a single parameter t to generate outputs ranging from high-fidelity/blurry (t=0) to high-realism/sharp (t=1) in just one forward pass, achieving state-of-the-art performance on FFHQ, DIV2K, and ImageNet datasets while being significantly more efficient than multi-step diffusion methods.

### Strengths
The paper's main strength is achieving an unprecedented combination of efficiency and flexibility, delivering tunable fidelity-realism trade-offs in a single forward pass, which no prior one-step method accomplishes. The technical approach is elegant and well-motivated: the noise-augmented conditioning prevents mode collapse while the ODE-trajectory constraint ensures the distilled model inherits the teacher's trade-off properties. The results achieved state-of-the-art performance across multiple benchmarks (FFHQ, DIV2K, ImageNet) while requiring only 1 NFE, making it faster at inference. The method is also applied to different teacher models and various tasks (noiseless SR, noisy SR, real-world SR, and different scale factors). It provides practical value, as a single trained model can serve multiple use cases by simply adjusting the t parameter.

### Weaknesses
The primary weakness is that the distilled model doesn't perfectly preserve the teacher's perception-distortion frontier. As shown in Figure 7, there's a slight shift in the trade-off curve, meaning the same t value doesn't yield identical fidelity-realism balances between teacher and student, likely due to the discrete step size (dt) approximation. The method's performance is fundamentally constrained by the teacher model's capabilities, and the two-stage training adds complexity and computational cost (though it remains more efficient than some alternatives, such as SinSR). The noise augmentation strength is a critical hyperparameter that requires careful tuning per task. The paper lacks a solid theoretical justification for why the proposed distillation loss preserves the trade-off properties or for what guarantees exist. Finally, some design choices (like dt=0.05) appear empirically driven rather than principled, and the ablation studies, while thorough, reveal sensitivity to these hyperparameters.

### Questions
LPIPS is a perceptual fidelity metric, while PSNR is a fidelity metric. In Figure 3, LPIPS and PNSR were used to demonstrate the transition between image realism and fidelity. A realism metric is preferred, e.g., FID, NIQE, etc. Again, as such a trade-off is a selling point, it would be better in the tables (1, 2, 3, 4). The metrics could be organized, e.g., realism metrics are put on the left and fidelity metrics on the right. That will help understand the differences against t values. 

The SR baseline is established on diffusion and flow-based methods. Thus, the significance of the proposed method for general SR applications is unclear.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces OFTSR, a conditional rectified flow framework for one-step image super-resolution that allows continuous control between fidelity and realism. The method consists of two stages: a noise-augmented conditional flow trained with a Variance-Preserving (VP) perturbation of the low-resolution (LR) input to enhance diversity and coverage of the HR distribution, and a distillation stage that aligns the student's single-step predictions with the teacher's probability flow ODE trajectory using a combination of distillation, alignment, and boundary losses. OFTSR demonstrates flexible, single-step inference and achieves strong quantitative and qualitative results on diverse datasets showing competitive or superior performance to baselines with significant gains in efficiency.

### Strengths
1. Tunable one step SR with explicit control.
The paper introduces an interpretable t-controlled mechanism that enables explicit adjustment of the fidelity-realism trade-off along the teacher’s ODE trajectory.


2. Efficiency and applicability.
The method requires only a single forward pass at inference while retaining flexibility typically reserved for multi-step models.

3. Strong empirical performance.
Evaluations span three datasets (DIV2K, FFHQ, ImageNet) and include noisy, noiseless, and real-world SR scenarios. The ablation studies (Tables 5-6) systematically analyze the effects of perturbation strength, solvers, and loss weights, providing solid empirical justification.

4. Clear Methodological Formulation.
The paper's derivation of the distillation and alignment losses (Equations 9-12, Algorithm 1) is mathematically consistent and clearly connected to existing distillation methods such as BOOT and forward distillation, offering both clarity and theoretical grounding.

### Weaknesses
1. Boundary of Novelty Relative to Existing Frameworks.
While the paper combines concepts appropriately, its originality is somewhat incremental. 
The noise-augmented LR conditioning interpolates between SR3 and InDI as acknowledged in Section 3.1. 
Similarly, the distillation loss in Eq. (9) is a constrained variant of forward distillation (Appendix B.2), 
and the core idea of aligning student and teacher ODE trajectories resembles BOOT. 
The contribution would be clearer if the authors better emphasized what properties uniquely arise from this specific combination in SR tasks.


2. Teacher dependence is acknowledged but under explored empirically. 
Section C mentions that the student's performance is constrained by the teacher's capability, 
but the experiments (Table 7) explore only a single ResShift teacher. 
The lack of systematic testing across multiple teachers limits understanding of the method’s robustness 
when teacher quality varies or exhibits bias.


3. Comparison with one step SR can be expanded. 
The experimental comparison omits some recent one-step SR approaches [1, 2], 
which could contextualize the claimed state-of-the-art results. Including these would strengthen the empirical claims.


4. Editorial issues. Minor editorial redundancies exist, such as repeated inference equations 13-14. 

[Reference]

[1] TSD-SR: One-Step Diffusion with Target Score Distillation for Real-World Image Super-Resolution, CVPR 2025

[2] One-Step Effective Diffusion Network for Real-World Image Super-Resolution, NeurIPS 2024

### Questions
1. Clarification on Table 7: Could the authors explain the distinction between "OFTSR (ResShift teacher)" and "OFTSR (pre-train + distill)"?

2. Clarification on Table 8:
Two OFTSR rows are listed with identical NFEs but differing parameter counts (118.6M+55.3M vs 552.8M). Do these correspond to different architectures (e.g., ResShift-based vs standalone OFTSR) or to inclusion/exclusion of VAE components?

### Soundness
3

### Presentation
2

### Contribution
2
