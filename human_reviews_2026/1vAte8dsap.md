# pi-Flow: Policy-Based Few-Step Generation via Imitation Distillation

- Decision: Accept (Poster)
- Scores: 8, 4, 4, 8

## Abstract
Few-step diffusion or flow-based generative models typically distill a velocity-predicting teacher into a student that predicts a shortcut towards denoised data. This format mismatch has led to complex distillation procedures that often suffer from a quality--diversity trade-off. To address this, we propose policy-based flow models ($\pi$-Flow). 
$\pi$-Flow modifies the output layer of a student flow model to predict a network-free policy at one timestep. The policy then produces dynamic flow velocities at future substeps with negligible overhead, enabling fast and accurate ODE integration without extra network evaluations.
To match the policy's ODE trajectory to the teacher's,
we introduce a novel imitation distillation approach, which matches the policy's velocity to the teacher's along the policy's trajectory using a standard $\ell_2$ flow matching loss. 
By simply mimicking the teacher's behavior, $\pi$-Flow enables stable and scalable training and avoids the quality--diversity trade-off.
On ImageNet $256\times 256$, it attains a 1-NFE FID of 2.85, outperforming previous 1-NFE models of the same DiT architecture. 
On FLUX.1-12B and Qwen-Image-20B at 4 NFEs, $\pi$-Flow achieves substantially better diversity than state-of-the-art DMD models, while maintaining teacher-level quality.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper describes a method for the distillation of diffusion/flow networks. It assumes a pretrained flow model, and aims to learn a student model, initialized from the same weights, that can produce outputs with a small number of function evaluations. The key novelty with relation to prior distillation methods is the idea to predict a *policy* rather than a direct velocity. This policy can then be used to perform ODE denoising over many iterations, without the need for further network calls. The two policy classes are a DX policy, which predicts a sequence of (x0) predictions according to various t, and a GMFlow policy, which predicts a Gaussian mixture over the data. The GMFlow policy can be solved analytically to predict the velocity at a given t. To train the student model, on-policy distillation is used to fit the policy's predictions to the teacher. The proposed method outperforms prior works on a range of benchmarks such as DiT, FLUX, and Qwen-Image distillation.

### Strengths
This paper provides a novel contribution in the idea to predict a "policy" from the student, rather than directly predicting velocity. This innovation allows a single network call to define a series of ODE steps, which is key to retaining performance while reducing the number of function calls. In terms of quality and clarity, the paper is well-written, with a clear explanation of the method and thorough experimental results. This paper has a chance to be significant in the sense of a practical, effective flow distillation method.

### Weaknesses
- The description of the GMFlow policy class is hard to understand, and could benefit from a more intuitive explanation. Specifically, the paper would benefit from a more explicit instantiation of the mapping from (student network) -> (policy parameters) -> (velocity prediction at x_t, t).
- The proposed method requires denoising the policy via an ODE *during the training process*, which adds a considerable overhead to the training pipeline. This stands in contrast to prior works which reduce the need for ODE rollouts during training (e.g. consistency models, progressive distillation, etc).
- The comparison in Table 2 is not necessarily fair, as it compares the proposed pi-Flow method, which is a distillation method that requires a pretrained teacher, to iCT/Shortcut/Meanflow which are *from-scratch* methods that do not require a teacher.

### Questions
- I am a bit confused how this distillation procedure avoids the mean-collapse issue with naive distillation. For example, if the fixed trajectory is at t=1 (i.e. pure noise), then the DX policy head will attempt to predict E[x0], which is just the mean under the dataset. Thus, how is it that the student network can be used to generate images in a single NFE (which by definition should map pure noise to an image?)
- How are t_src and t_dst chosen (page 5)? Does NFE stand for NFE under the teacher model, or under the student model?

### Soundness
3

### Presentation
4

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
This paper introduces Pi-Flow, a novel framework for distilling fast samplers by training a policy model rather than a traditional trajectory-based generator. Instead of predicting instantaneous velocities along a diffusion trajectory, Pi-Flow outputs an entire policy function that governs the trajectory’s evolution. This approach allows the distilled model to represent the generative process as a global functional mapping, rather than a sequence of stepwise updates.

### Strengths
1. **Novel Policy-Based Distillation Approach**  
   Distilling a policy function that predicts an entire trajectory, while conceptually related to operator learning in DFNO (Deep Fourier Neural Operator) [1], is novel and interesting.

2. **Comprehensive Large-Scale Experiments**  
   The authors conduct extensive experiments on large-scale text-to-image models such as **Flux** and **Qwen**, demonstrating that the method scales effectively to state-of-the-art architectures.  

3. **Strong Text-to-Image Performance**  
   Empirical results show that Pi-Flow outperforms prior distillation methods in the text-to-image domain, highlighting its potential as a scalable and efficient alternative to conventional diffusion distillation techniques.

### Weaknesses
1. **Unfair Baseline Comparisons**  
   In Table 2, the paper primarily compares Pi-Flow with training-from-scratch methods such as iMM and Mean Flow, rather than distillation-based methods like Score Identity Distillation (SiD) [2] or Consistency Training Models (CTM) [3]. This makes it difficult to assess Pi-Flow’s relative effectiveness as a distillation framework.  

2. **Limited Evaluation on Standard Diffusion Backbones**  
   While large-scale experiments on Flux demonstrate the method’s scalability, evaluations on more established diffusion models such as **Stable Diffusion 1.5** and **SDXL** would strengthen the paper. These benchmarks are widely used in prior distillation works (e.g., Long and Short Classifier-Free Guidance [4], Phased Consistency Models [5]), and results on them would allow for a fairer comparison.  

3. **Inconsistent Efficiency Reporting**  
   Since Pi-Flow predicts the full trajectory at once (i.e., the entire policy function), the architecture requires increased channel capacity and computational cost. Consequently, NFE (number of function evaluations) is no longer a fair efficiency metric. Metrics such as FLOPs or latency should be reported instead to provide a meaningful assessment of inference efficiency.

### Questions
1. **Clarify Multistep Training Scheme**  
   The paper mentions both single-step and multistep training setups, but the procedure for multistep training is not clearly described. Additional details on how the multistep policy is trained, would improve clarity.  

2. **Expand Description of GM Flow**  
   The explanation of GM Flow is incomplete. The paper lacks a formal definition of the training objective for each policy and does not give sufficient details on how the architecture is adapted to enable policy generation. Providing explicit loss formulations, architectural modifications, and training details would make the contribution clearer and more reproducible.

---

## References  
[1] **Fast Sampling of Diffusion Models via Operator Learning.**  
[2] **Score identity Distillation: Exponentially Fast Distillation of Pretrained Diffusion Models for One-Step Generation.**  
[3] **Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion.**  
[4] **Guided Score identity Distillation for Data-Free One-Step Text-to-Image Generation.**   
[5] **Phased Consistency Models.**

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
3

### Summary
This paper proposes pi-Flow, a new framework for few-step generative modeling that reframes flow-based distillation as policy learning. Instead of directly predicting the terminal point of a probability flow (as in velocity or consistency distillation), pi-Flow trains a network to output a policy that governs the velocity field across multiple integration substeps without additional network evaluations. The authors introduce a training algorithm, Policy-Based Imitation Distillation (pi-ID), inspired by on-policy imitation learning (DAgger), which lets the student model imitate the teacher’s flow dynamics along its own trajectory. Two specific parameterizations are explored:DX and GMFlow. Empirical results on ImageNet-256 and large text-to-image backbones (FLUX, Qwen-Image) show that pi-Flow achieves superior quality–diversity trade-offs and state-of-the-art one-step and few-step sampling performance.

### Strengths
1. The policy-based perspective is a conceptually elegant and novel reinterpretation of few-step distillation, bridging diffusion/flow modeling and imitation learning.
2. The pi-ID training procedure is well-motivated and effectively stabilizes learning by mitigating compounding errors—an issue that plagues prior consistency and flow distillation methods.
3. The proposed network-free policy formulation enables fine-grained ODE integration with very few network calls, yielding strong efficiency gains.

### Weaknesses
1. While the policy formulation is innovative, the theoretical justification (why and when a policy yields better approximation of teacher trajectories) is mostly intuitive; a more formal analysis of error bounds or convergence would strengthen the contribution.
2. The DX and GMFlow variants differ significantly in nature, but the paper does not fully disentangle whether gains arise from the pi-Flow framework itself or the added modeling flexibility of GMFlow.
3. The approach introduces additional hyperparameters (policy length, Gaussian components) whose tuning cost is not well quantified.
4. Although pi-Flow is claimed to preserve diversity, a quantitative comparison using standard mode-collapse or coverage metrics (beyond FID) would help substantiate this claim.

### Questions
1. How sensitive is pi-Flow performance to the policy horizon (number of substeps)? Could adaptive or learned horizons improve efficiency?
2. What are the computational trade-offs between DX and GMFlow in practice—does GMFlow’s expressivity justify its added complexity?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes pi-Flow, a policy-based paradigm for distilling multi-step flow-matching teachers into few-NFE students. Instead of predicting a single shortcut velocity, the student network outputs a network-free policy that can be rolled out with many cheap sub-steps to integrate the probability-flow ODE. Training uses on-policy imitation distillation (pi-ID), a velocity matching loss applied along the student’s own trajectory. Two policy families are introduced—DX and GMFlow. Experiments demonstrate state-of-the-art 1-NFE FID on ImageNet-256 with DiT-XL/2 and high-quality 4-NFE text-to-image generation from 12B/20B teachers (FLUX.1, Qwen-Image), with notably better diversity than prior few-step methods.

### Strengths
1. By predicting a policy rather than a shortcut, pi-Flow retains the dense ODE trajectory of the teacher while requiring only a few network evaluations. This insight is both conceptually clean and practically powerful.
2. The velocity matching on the student’s own rollout avoids the complex progressive/consistency schedules of prior work and directly inherits teacher quality/diversity.
3. Strong empirical results across scales. 1-NFE FID 2.90 on ImageNet-256 (DiT-XL/2) beats all prior DiT-based few-step models. 4-NFE distillation of 12B/20B T2I models preserves fine details and text rendering while achieving substantially better diversity than SOTA few-step baselines.

### Weaknesses
1. Table 1 shows K=8 vs. K=32 gives similar FID, but no ablation on $L\times C$ factorization, mixture renormalization, or temperature scaling in inference.

### Questions
1. How does pi-Flow compare to MeanFlow in terms of training stability and wall-clock time? Does avoiding JVP give measurable speedups during training?

### Soundness
3

### Presentation
3

### Contribution
3
