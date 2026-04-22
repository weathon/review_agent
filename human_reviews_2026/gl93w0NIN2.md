# Score Distillation of Flow Matching Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 6, 2

## Abstract
Diffusion models achieve high-quality image generation but are limited by slow iterative sampling. Distillation methods alleviate this by enabling one- or few-step generation. Flow matching, originally introduced as a distinct framework, has since been shown to be theoretically equivalent to diffusion under Gaussian assumptions, raising the question of whether distillation techniques such as score distillation transfer directly. We provide a simple derivation—based on Bayes’ rule and conditional expectations—that unifies Gaussian diffusion and flow matching without relying on ODE/SDE formulations. Building on this view, we extend Score identity Distillation (SiD) to pretrained text-to-image flow-matching models, including SANA, SD3-Medium, SD3.5-Medium/Large, and FLUX.1-dev, all with DiT backbones. Experiments show that, with only modest flow-matching- and DiT-specific adjustments, SiD works out of the box across these models, in both data-free and data-guided settings, without requiring teacher finetuning or architectural changes. This provides the first systematic evidence that score distillation applies broadly to text-to-image flow matching models, resolving prior concerns about stability and soundness and unifying acceleration techniques across diffusion- and flow-based generators.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The work proposed to extend Score identity Distillation (SiD) framework to distill flow-based text-to-image models into few step generators, SiD-DiT. The key contributions of the work include the analysis of loss reweighting for pretraining of diffusion and flow-matching models, which show the equivalence of optimized objectives with the only difference of timestep weighting functions, and extension of the SiD to distill pretrained flow-based text-to-image models SANA, SD3-Medium, SD3.5-Medium/Large, and FLUX.1-Dev. Distilled models were evaluated on 10k COCO-2014 validation subset with FID and CLIP, and on GenEval benchmark. The proposed distilled models were compared with teacher models, and distilled models of SANA Sprint, Flash SD3, SD3.5-Turbo-Medium, and SD3.5-Turbo-Large. The results show that SiD-DiT training framework can improve the quality of the teacher model in terms of FID and CLIP for most flow-based text-to-image models, while having only 4 NFE.

### Strengths
1) Experimental results of the proposed SiD-DiT model on different flow-based text-to-image models, including SANA, SANA TrigFlow, SD3, SD3.5, and FLUX with different model sizes support the claim of the work that SiD can be applicable to flow-based models to maintain high-quality image results of teacher models with big NFE, while having much faster inference with NFE=4 only.
2) The unified view presented in Section 2.3 on diffusion and flow-based objectives supports the motivation of the work to apply SiD for flow-based text-to-image models.
3) The distillation of the FLUX model with 12B parameters into 4 step generator requires complex technical implementation to achieve it on a single node with 8 GPUs. In my opinion, this result is impressive since there almost no models to distill such a large text-to-image model. 
4) In contrast to the recent SANA-Sprint model, which is based on consistency distillation and requires real data, the proposed SiD-DiT enables data-free distillation for flow-based text-to-image models. Table 1 in the paper shows that even without real data SiD-DiT achieves competitive results when compared with the SANA-Sprint model.

### Weaknesses
1) In contrast to the original SiD paper, which applied the SiD objective to distill diffusion model in 1-step generator, SiD-DiT does not explore the option NFE=1. It makes the comparison of SiD-DiT with SANA Sprint, which has NFE=1 for SANA TrigFlow 1.6B model, to be unfair. The DMD2 paper motivated the multi-step generator found that the distillation of SDXL model into 1-step generator is challenging, as shown by Patch FID in Table 2 of DMD2. The work of Zhou et al. for multistep SiD with diffusion SDXL model in Table 6 shows the improvements of multistep SiD model with NFE=4 over NFE=1 in terms of FID, but also the drop in CLIP. It remains a question on the quality of SiD-DiT with NFE=1 or NFE=2.
2) The work lacks of discussion on the stability of the SiD framework for flow-based text-to-image models, which was done in prior distillation works of Zhou et al. (Figure 9) and DMD2 (Figure 9) for diffusion text-to-image models. This is important to understand why SiD-DiT for FLUX.1-Dev achieved much worse results compared to the teacher model: is it due to computational limitations in terms of training time and the proposed method did not converge or the distillation converged, but the results are worse due to unique features of FLUX.1-Dev such as learned guidance embedding. 
3) The work does not provide detailed pseudo-code of the proposed method, which is crucial to articulate the difference and similarities between the details and heuristics of the SiD-DiT implementation for flow-based text-to-image models with the Algorithm 1 in Zhou et al. of SiD for diffusion text-to-image models and Algorithm 1 in the original SiD paper. 
4) Even though the SiD-DiT method is compared with teacher flow-based models and several distillation models of these models, the work lacks of comparison with existing distilled few-step diffusion-based models in terms of image quality and inference efficiency trade-off. 

References:

Zhou et al. Few-Step Diffusion via Score identity Distillation, arxiv-2025.

DMD2. Improved Distribution Matching Distillation for Fast Image Synthesis, NeurIPS-2024.

### Questions
1) In the original SiD paper, the authors found that $\alpha \neq 1$ can slightly improve the results (see Figure 2 and Figure 8 in the SiD paper). Did you try to vary the $\alpha$ parameter?
2) Since the results of SiD-DiT and SANA Sprint models are closed in the mean GenEval score, can you proved detailed 6 GenEval scores for SANA, SANA Sprint, and SiD-DiT models?
3) As far as I understood, the whole Section 2 is given only to motivate the application of SiD for flow-based models. Are there any novel practical implications or techniques, which were used to implement SiD-DiT in Section 3 based on the theory from Section 2?
4) Can you comment on the ability SiD-DiT to distill flow-based text-to-image models in 1 or 2 steps?
5) Can you comment on the stability of SiD-DiT distillation training for all experiments? Did the distillation converged for the FLUX.1-Dev experiment?
6) Can you comment on the comparison in terms of image quality and inference efficiency trade-off for SiD-DiT and other recent fast text-to-image models, including DMD2, FLUX-schnell, and Rectified Diffusion (Phased)?
7) It should be noted that FID and CLIP metrics have issues with correspondence of human preferences in outputs of text-to-image models, as noted by PickScore and ImageReward papers. Can you provide PickScore and ImageReward metrics for SiD-DiT and other models from Table 2?

References: 

Rectified Diffusion: Straightness Is Not Your Need in Rectified Flow. ICLR-2025. 

Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation. NeurIPS-2023.

ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation. NeurIPS-2023.

### Soundness
3

### Presentation
3

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
This paper presents a unified theoretical and practical framework for score distillation applied to flow matching models. The authors first show that, under Gaussian noise assumptions, diffusion and flow matching formulations are mathematically equivalent up to a reweighted time distribution. Building on this equivalence, they adapt the Score Identity Distillation method—originally designed for diffusion models—to DiT-based flow matching architectures, forming SiD-DiT. The proposed approach enables few-step (e.g., four-step) sampling without requiring teacher fine-tuning or architectural modifications. Extensive experiments on large-scale text-to-image models (SANA, SD3.5, FLUX) demonstrate that SiD-DiT achieves comparable or superior performance to specialized fast samplers (e.g., SANA-Sprint, SD3.5-Turbo) while maintaining generality across architectures.

### Strengths
1. The paper provides a clear and rigorous theoretical unification between diffusion and flow matching models via Bayes conditioning and Tweedie’s formula, avoiding the need for SDE/ODE derivations.
2. The proposed SiD-DiT framework is architecturally and algorithmically minimal—it directly applies score distillation to flow matching models without teacher or network modification.
3. Empirical validation is comprehensive, covering multiple model families (SANA, SD3.5, FLUX) and showing that 4-step generation matches or surpasses teacher quality.

### Weaknesses
1. The novelty is more integrative than conceptual—theoretical equivalence between diffusion and flow matching is known, though not previously formalized this elegantly.
2. The adversarial extension is underexplored—it improves FID but lacks theoretical grounding or stability discussion.
3. The paper could benefit from more detailed comparisons to existing distillation frameworks like Progressive Distillation, Consistency Models, or MeanFlow, especially regarding training cost and convergence dynamics.

### Questions
1. How sensitive is SiD-DiT to the specific choice of timestep distribution $p(t)$ and weighting $w(t)$ ? Could these be learned adaptively?
2. In the large-model experiments (e.g., FLUX-1), is there evidence that student models inherit biases or artifacts from the teacher’s flow parameterization?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper extends Score identity Distillation (SiD) to pretrained text-to-image flow-matching models with DiT backbones. The authors first derive a unified view of Gaussian diffusion and flow matching via Bayes’ rule and conditional expectations, showing that their optimal solutions are identical and differ only in the weight-normalized timestep distribution. They then apply SiD with minimal adjustments to flow matching models, demonstrating 4-step generation in both data-free and data-guided settings without teacher finetuning or architectural changes. Experiments validate robustness across model sizes (0.6B–12B) and noise schedules.

### Strengths
1. The derivation is clean, self-contained, and resolves a common misconception: diffusion and flow matching are not fundamentally different under Gaussian assumptions. The reweighting perspective (Eq. 14, Figure 2) is insightful and practically useful.
2. Distilling distinct DiT-based models with a single codebase and hyperparameter set is impressive. Data-free 4-step generation from FLUX.1-DEV (12B) is a significant practical result.
3. Unlike SANA-Sprint (which requires TrigFlow conversion + real data), SiD works “out of the box” on both Rectified Flow and TrigFlow checkpoints. This is a clear win for knowledge transfer.

### Weaknesses
1. The core method is SiD, previously validated on U-Net diffusion models. The main claim—“it works on DiT flow models too”—while practically important, is incremental over prior SiD work.
2. Figure 1 is compelling, but no side-by-side with teacher or baselines (e.g., 50-step teacher, SANA-Sprint, DMD).
3. The paper contrasts with SANA-Sprint (consistency distillation) but does not compare to trajectory distillation on the same models.

### Questions
1. Can you provide FID/CLIP scores for the teacher (50-step) vs. 4-step SiD vs. SANA-Sprint on the same prompts and seeds?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents a framework for applying score distillation to DiT-based flow-matching models, termed SiD-DiT.
The authors reformulate the score distillation objective using the analytical relation between score functions and velocity fields, enabling distillation directly from flow-matching teachers (e.g., SANA, SD3.5, FLUX).
They employ Fisher divergence minimization for training and further stabilize learning via a fake flow-matching network and an optional adversarial refinement module.
Experiments on text-to-image generation demonstrate that the proposed method achieves reasonable few-step generation quality, with qualitative improvements under different noise schedules.

### Strengths
* Provides a unified formulation of score distillation in the flow-matching context, clarifying how SDS-like losses can operate on velocity-based models.
* The training framework (SiD-DiT) is straightforward, practical, and compatible with large-scale DiT backbones.
* Empirical results show that few-step distillation remains feasible for modern flow-matching models, which is of practical interest for efficiency-oriented generative modeling.
* The paper is clearly written, with good visualizations and comparisons of noise schedules that help explain the method’s behavior.

### Weaknesses
1. **Limited novelty over existing flow-based distillation methods.**  
The use of score distillation on flow-based models has already been explored in several prior works such as CausVid [1], SenseFlow [2], etc. The proposed approach actually does not introduce a new theoretical formulation or algorithmic mechanism, or the paper should more clearly articulate how its method differs conceptually or practically from these existing approaches.

2. **Score–velocity equivalence already established.**  
The relationship between score functions and velocity fields has been explicitly derived in Equation (9) of SiT[3], which provides a rigorous mapping between diffusion and flow formulations. It remains unclear whether the ``score target" used in this paper deviates from that known formulation or simply reuses it within a different loss setup.

3. **Training procedure closely resembles DMD but lacks comparison.**  
The proposed alternating optimization between a generator and a ``fake score" model, under Fisher divergence minimization, is highly similar to the DMD training pipeline. A fair experimental comparison against DMD/DMD2 based method on flow-matching models would be essential to demonstrate any empirical advantage or stability improvement.

4. **Experimental analysis is incomplete.**  
The experiments do not provide sufficient coverage for strong baselines or fair settings:
* Table 1 omits the 4-step result for SANA TrigFlow 1.6B, showing only the 1-step case.
* Table 2 compares only with FLUX-DEV but excludes other diffusion distillation baselines, such as Hyper-FLUX, FLUX-turbo-alpha, etc.
This limited analysis makes it difficult to gauge the relative position and contribution of the proposed method within the broader diffusion-distillation literature.

[1] From Slow Bidirectional to Fast Autoregressive Video Diffusion Models.  
[2] SenseFlow: Scaling Distribution Matching for Flow-based Text-to-Image Distillation.  
[3] SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers.

### Questions
Please see the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1
