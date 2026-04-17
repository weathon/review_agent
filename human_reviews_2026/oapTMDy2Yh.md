# A Step to Decouple Optimization in 3DGS

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
3D Gaussian Splatting (3DGS) has emerged as a powerful technique for real-time novel view synthesis. As an explicit representation optimized through gradient propagation among primitives, optimization widely accepted in deep neural networks (DNNs) is actually adopted in 3DGS, such  as synchronous weight updating and Adam with the adaptive gradient. However, considering the physical significance and specific design in 3DGS, there are two overlooked details in the optimization of 3DGS: (i) update step coupling, which induces optimizer state rescaling and costly attribute updates outside the viewpoints, and (ii) gradient coupling in the moment, which may lead to under- or over-effective regularization. Nevertheless, such a complex coupling is under-explored. After revisiting the optimization of 3DGS, we take a step to decouple it and recompose the process into: Sparse Adam, Re-State Regularization and Decoupled Attribute Regularization. Taking a large number of experiments under the 3DGS and 3DGS-MCMC frameworks, our work provides a deeper understanding of these components. Finally, based on the empirical analysis, we re-design the optimization and propose AdamW-GS by re-coupling the beneficial components, under which better optimization efficiency and representation effectiveness are achieved simultaneously.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors analyzed Adam optimizer for the problem of 3DGS optimization, provided several observations and modifications on the optimizer to be more effective, summarized as AdamW-GS. Specifically, they made the observation that the previously proposed sparse Adam is effective at eliminating dead gaussians, but also lead to performance degradation. To resolve this, they have proposed steps such as Re-State Regularization (RSR) and Decoupled Attribute Regularization (DAR). RSR aims at better activating regularization by rescaling the moment, by introducing two additional scaler. DAR aims to decouple the main objective function (photometric loss) and regularization (on opacity and scale), such that they do not influence each other and cause over/under regularization. Experiments demonstrate that AdamW-GS consistently improves reconstruction/NVS results compared to 3DGS/3DGS-MCMC.

### Strengths
The authors tackle an important problem in determining the appropriate optimization function on the problem of 3DGS reconstruction. While 3DGS has been influential, the optimization behavior can easily get stuck either due to overfitting/floaters or underfitting (blobs). 

The proposed AdamW-GS shows consistent improvements compared to 3DGS and 3DGS-MCMC, with faster training process and fewer (dead) gaussian primitives. 

I am not an expert in optimization, but the experiments, observations, and modifications seem reasonable and generalizable enough to me.

### Weaknesses
The modification in AdamW-GS introduces additional hyperparameters, with no obvious intuition in how to adjust them. The RSR parameters, for example, are obtained through grid search. While observations show that the regularization term is sometimes over/under effective, it is unclear to me how to determine over/under fitting based on a single run and tune hyperparameters accordingly. 

I find the table listing of MC1, MC2, etc. to be confusing and difficult to cross reference and understand what the authors are doing.

A few papers on automated efficient 3DGS may be worth citing:

PUP-3DGS (Hanson et al., CVPR25) - determines dead gaussians/gaussians with low contribution to loss
LP-3DGS (Zhang et al. NIPS24) - automatically reduces gaussian counts during training, similar to AdamW-GS
3DGS-LM (Höllein et al. ICCV25) - Optimization with Levenberg-Marquardt instead of Adam

### Questions
See Weakness - how would authors propose for tuning the additional hyperparameters in practice?

Since Taming 3DGS is extensively referenced, has it been properly compared to this work?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper re-examines optimization in 3D Gaussian Splatting (3DGS) and argues that two kinds of coupling: (i) update-step coupling from synchronous Adam that induces 
"implicit updates" on invisible primitives, and (ii) gradient coupling where regularization gradients are mixed into Adam's moments. It proposes a three-part redesign: Sparse Adam for view-conditioned updates, Re-State Regularization (RSR) to periodically rescale moments, and Decoupled Attribute Regularization (DAR) that removes $\nabla R$ from moment updates and applies a preconditioned, clipped regularization step. The recoupled system ("AdamW-GS") is shown to improve PSNR while reducing total runtime.

### Strengths
+ The motivation is well-explained, and provides carefully analysis (with figures and measurements) to show the synchronous Adam rescales moments even with zero gradients, producing implicit updates.
+ Sparse Adam + RSR + DAR is easy to implement, integrates with both vanilla 3DGS and 3DGS-MCMC
+ Experimental results show AdamW-GS improves quality and cuts runtime

### Weaknesses
+ The proposed AdamW-GS optimizer is far more complex to tune than the standard Adam. It introduces numerous new hyperparameters, such as moment scaling factors ($\alpha_1$, $\alpha_2$) for RSR 15, and regularization scales ($\lambda$) and clipping values ($\mathcal{C}_t$) for DAR 16. This may pose challenges to directly applied hyper-parameter settings on MipNerf360 to other scenes.

+ Related to first point, the crucial RSR component relies on a "hand-crafted" milestone-based schedule (StSS) to function. This schedule is manually designed for the test datasets (see Fig. 6) and is unlikely to generalize to new scenes or datasets without being re-tuned

+ The claim of removing primitives "without introducing additional pruning components" is nuanced. The method works by aggressively pushing opacity below the existing 1/255 threshold, at which point the original 3DGS rendering pipeline already excludes them.

+ Some important ablations are missing. For example, the contribution of DAR is not isolated in the ablation studies. To prove DAR’s advantage isn’t just a scale/clip effect, the paper should include step-size-matched baselines (e.g., L1 with λ tuned to match DAR’s average regularization step; AdamW-style constant penalty with matched effective step). Current comparisons don’t fully disentangle mechanism vs magnitude.

### Questions
+ The "hand-crafted" StSS schedule  is the method's most significant liability. How sensitive is the performance to this schedule? Could authors provide additional experiments on different scenes with the same schedule?

+ The paper notes that Sparse Adam alone degrades performance, and the RSR component is required to "fix" it. This suggests a tight, possibly brittle, inter-dependency. How robust is the method to the RSR hyperparameters? If the $\alpha$ values are slightly misconfigured, does the optimization fail entirely?

+ Regarding the ablation studies for DAR, it remains unclear whether the observed benefits stem from DAR's adaptive mechanism (i.e., scaling by $\sqrt{\hat{v}}$) or simply from its resultant effective magnitude. Please include baselines where (a) L1 and (b) constant-penalty (AdamW-style) are tuned to match DAR's average/max regularization step (with and without $C_t$), to rule out scale effects.

+ Given that the noise-based exploration mechanism is scene-dependent and can be harmful, what alternative exploration strategies could be integrated? Could the RSR's moment-rescaling be adapted to also induce exploration (e.g., by adding noise during rescaling) rather than just activating regularization?

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
2

### Summary
This paper revisits optimization in 3D Gaussian Splatting and argues that standard practice (Adam + synchronous updates) introduces two harmful couplings: (i) update-step coupling that rescales optimizer states and updates invisible primitives, and (ii) gradient/regularization coupling that makes effective weight decay depend on Adam’s second moment. It decomposes the training recipe into three parts: Sparse Adam, Re-State Regularization, and Decoupled Attribute Regularization, and then re-couples the beneficial pieces into AdamW-GS. Across vanilla 3DGS and 3DGS-MCMC, the paper reports consistent quality gains and meaningful wall-clock savings. Their method also reduces redundancy while improving rendering performance.

### Strengths
* This paper pinpoints that Adam synchronously updates even invisible primitives, rescaling their moments and changing attributes despite zero gradients (coined Implicit Update). Such a study helps the understanding of the 3DGS Optimization procedure. 
* On outdoor scenes, the method reduces primitives by 48.4% while improving PSNR by +0.2 dB and SSIM by +0.01; on indoor scenes, it removes 50% of primitives while still gaining +0.1 dB PSNR, unlike MaskGaussian, which removes 61% but drops PSNR by 0.1 dB. The proposed approach strikes a better balance. Further, Table 4 (Deep Blending, Tanks & Temples) shows that AdamW-GS surpasses vanilla baselines.
* Time-cost breakdowns show large step-time and total runtime reductions compared with baselines.

### Weaknesses
* The comparison emphasizes MaskGaussian (and RePR), but omits other density-control splatting methods (Compact3DGS, Deformable beta splatting) cited in related work. Without matched-budget comparisons on the same scenes, it’s hard to evaluate the reported ~48–50% primitive reductions with small PSNR/SSIM gains compared with various baselines.
* All the per-scene tables report single PSNR/SSIM/LPIPS numbers with no error bars or seed variance, so it’s unclear whether gains are stable across runs or due to initialization. Given the small rendering quality gap compared with the baselines, it is necessary to have the error bars (at least with several examples) to understand the significance of the improvements.
* Most results emphasize vanilla 3DGS and 3DGS-MCMC setups on standard static benchmarks; there’s little evidence for large-scale indoor scans, or outdoor long-range sequences where visibility patterns and sparsity differ. A dedicated failure case section will help to further study the method.

### Questions
I am having a hard time connecting the cits (e.g., GSs, MCs, Ours) when reading the paper. Primarily, I need to frequently read the Appendix and identify the differences in the configuration. It would be better to provide a clearer explanation or include the difference in the table consistently.

### Soundness
3

### Presentation
3

### Contribution
3
