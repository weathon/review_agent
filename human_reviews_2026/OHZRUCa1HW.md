# Unconditional Human Motion and Shape Generation via Balanced Score-Based Diffusion

- Decision: Reject
- Scores: 4, 4, 4, 8

## Abstract
Recent work has explored a range of model families for human motion generation, including Variational Autoencoders (VAEs), Generative Adversarial Networks (GANs), and diffusion-based models. Despite their differences, many methods rely on over-parameterized input features and auxiliary losses to improve empirical results. These strategies should not be strictly necessary for diffusion models to match the human motion distribution. We show that on par with state-of-the-art results in unconditional human motion generation are achievable with a score-based diffusion model using only careful feature-space normalization and analytically derived weightings for the standard L2 score-matching loss, while generating both motion and shape directly, thereby avoiding slow post hoc shape recovery from joints. We build the method step by step, with a clear theoretical motivation for each component, and provide targeted ablations demonstrating the effectiveness of each proposed addition in isolation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper describes a new implementation for human motion generation, based on the insights from the EDM2 paper. The implementation has several merits, including less computational effort, less parameter tuning, and the incorporation of SMPL and shape parameters as opposed to recovering them from the joints. The paper also offers some theoretical analysis.
On the flip side, the proposed implementation does not present state-of-the-art results, has less capabilities, the removed computation is not shown to have merit, and evaluations are inconsistent and lacking.

### Strengths
- The paper incorporates a refreshing approach to human motion generation, different than that used by the community.
- SMPL-based prediction is interesting.
- Some insights, such as gradient balancing, seem quite relevant and important

### Weaknesses
- The paper shows partial capabilities - it did not incorporate any conditions (claiming this is "on purpose", but no benefit is shown), it does not demonstrate why or where the new approach and the capabilities it induces has merits. 
- Results are below the state-of-the-art - the paper compares to two rather old models, which have both seen dozens if not hundred of improvements since their original publication. For example, in CAMDM and CLoSD, a diffusion model that uses only ±10 steps is used, with much lower FiD than the 1000 steps model compared against here. 
- While shape is predicted, which is a refreshing plus, it doesn't seem to have any correlation to the motion itself, which makes shape prediction rather arbitrary.
- It is unclear what the contributions of the paper are. More than anything else, it shows that EDM2 has merit, but it is almost directly used in the context of motion generation. The "adaptations" from the original approach seem quite marginal. The paper should address better these adaptations and explain why they are not trivial. For example, dimension balancing, which seems to have a large effect on the results, is not adapted, while some more marginal parameters (such as tmin, which can be empirically found rather easily) are described in more details.

### Questions
- Perhaps the only analysis relevant in terms of design choice justification is Table 1, but it raises more questions than answers them. Why are the final numbers here different than those in Table 2? How would more conventional choices affect the model (e.g., adding redundancies to the representation / regularizing losses / text conditions)? Wouldn't it improve more? What is the baseline? It seems to be very poorly designed to show great improvement, but this does not justify that the design choices are indeed the right ones, just that they have some improvement. 
- Why compare against old models?
- What about larger datasets that now available? Would this approach still have merit if trained on them? 
- What is the usecase where this approach could shine the most?
- Could some of the insights (e.g. data normalization) be incorporated into traditional pipelines to improve SOTA?
- Auxiliary losses, which you claim to be unnecessary, help with things like jitter and ground penetration. How does this approach perform in these aspects? How much would the auxiliary losses help in this case?

### Soundness
3

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
5

### Summary
This paper proposes a method for unconditional human motion and shape generation using a score-based diffusion model. The authors argue that common practices in this field—such as using over-parameterized feature representations (e.g., combining joint angles with 3D positions) and auxiliary losses (e.g., foot skating losses)—are unnecessary if the training process is properly balanced. They adapt the EDM2 framework to human motion, introducing strictly theoretically motivated normalizations for SMPL-based input features (specifically handling 6D rotations without breaking their structure) and deriving analytic weightings for the standard L2 loss to balance gradients across diffusion time-steps and different feature groups (joints, orientation, translation, shape).

### Strengths
1. The paper takes a refreshing "back to basics" approach. Instead of adding complexity to fix issues, it uses theoretical derivations to balance the standard training objectives.

2. Table 1 is very convincing. It clearly shows how each theoretical addition (input normalization, gradient analysis, per-group weighting, dimensionality addressing) cumulatively improves the baseline FID from 6.23 down to 2.40.

3. By staying strictly within the score-based diffusion framework without ad-hoc sampling guidance, the method retains compatibility with probability flow ODEs, allowing for deterministic sampling and likelihood evaluation.

### Weaknesses
1. The paper focuses solely on unconditional generation. While the authors argue this is a necessary first step, most practical applications require conditional generation (text-to-motion, action-to-motion). It remains to be seen if this "pure" balanced approach holds up when complex conditioning signals are introduced.

2. The evaluation is primarily on AMASS (HumanML3D subset). Broader evaluation might be needed to confirm the generalizability of these balancing terms across different motion datasets with different characteristics. There are more datasets for human motion as BABEL, Motion-X, Posescript, etc... an evaluation on those will give a better understanding of the method generalization abilities.

3. I appreciate the visual results attached to the paper, while their quality are not way better than current baselines.

### Questions
1. Have you performed any preliminary experiments on extending this balanced weighting scheme to conditional models (e.g., classifier-free guidance)? Does adding conditioning break the delicate balance achieved here?

2. MDM generates both 3D joint locations and joint rotations. If you used only their generated rotations with an SMPL model, it would also result in zero limb length variance. Why did you choose to compare against their 3D joint output (which has variance) instead of their rotation output for the "Limbo" metric?

3. Given that a person's shape ($\beta$ SMPL parameters) does not change during a motion, why does your per-frame representation in Eq. 9 include $\beta$? Appendix A.1 clarifies you just copy the same vector for each frame, but what is the motivation for this design over generating a single, time-independent shape vector for the entire sequence?

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
2

### Summary
This paper presents a principled score-based diffusion framework for unconditional human motion and shape generation. Unlike prior work relying on redundant input features (e.g., joint positions + velocities + foot contacts) and multiple auxiliary losses, this paper demonstrates that a balanced training strategy with theoretically grounded normalization and loss weighting is sufficient to achieve state-of-the-art performance.

### Strengths
1. This paper proposed a structure-preserving feature normalization across heterogeneous SMPL feature groups (pose, orientation, translation, shape); 

2. The method removes the need for auxiliary losses and hand-tuned hyperparameters, simplifying the training pipeline. Implementation details are extensive and clear, making replication straightforward.

3. By avoiding auxiliary regularization, the model remains fully compatible with ODE-based sampling and likelihood computation—rare among motion diffusion works.

### Weaknesses
1. While generating motion and shape directly in the SMPL parameter space has the clear advantage of avoiding post-hoc shape reconstruction, the SMPL parameters themselves are typically obtained by fitting to skeletal joint data. This fitting process can introduce non-negligible errors (e.g., local rotation ambiguity, imperfect alignment), which may propagate into the generative model and limit the ultimate fidelity of the generated results.

2. The paper introduces a separately trained auxiliary network to learn dynamic weights for different feature groups. Although the authors claim that this network is lightweight, the paper does not provide quantitative comparisons (e.g., training time per epoch, convergence speed) to verify that the added module does not increase the overall training cost.

3. The proposed model focuses solely on unconditional motion and shape generation. While unconditional modeling can provide a reliable prior over motion distributions, most real-world applications (e.g., text-to-motion, action-conditioned motion synthesis) require conditional generation. Extending the framework to such settings would significantly enhance its practical impact.

### Questions
Please refer to the weaknesses above.

### Soundness
2

### Presentation
2

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
This paper proposes a balanced score-based diffusion framework for unconditional human motion and shape generation, which eliminates heuristic losses and feature redundancy by introducing theoretically derived normalization and loss balancing across heterogeneous feature groups. The key contributions are:
1. Unified diffusion formulation that directly models both human motion and shape using SMPL parameters.
2. Analytical feature balancing mechanism that replaces empirical weighting with theoretically grounded normalization.
3. Training framework compatible with PF-ODE sampling, achieving strong performance with only 31 neural function evaluations (NFEs).
4. Comprehensive evaluation and ablation showing improved fidelity, diversity, and efficiency over state-of-the-art diffusion-based motion models.

### Strengths
Originality is strong. The paper presents a conceptually fresh rethinking of how diffusion models are trained for motion generation. Instead of stacking auxiliary losses or handcrafted feature weights, it introduces a principled analytical balance across motion and shape components. This approach reduces reliance on empirical tuning while maintaining state-of-the-art performance, which is a notable departure from prior diffusion-based motion generation pipelines.

Quality is good. The method is both theoretically motivated and empirically validated. Ablation studies systematically isolate the effect of feature normalization and balanced weighting, and the model demonstrates strong results on multiple benchmarks (FID, diversity, and smoothness). However, the paper lacks deeper theoretical guarantees (e.g., convergence proof) and broader task evaluations beyond unconditional generation, leaving some generalization aspects unexplored.

Clarity is strong. The paper is clear, well-organized, and mathematically precise. Each design choice is motivated and supported by quantitative evidence. Equations and visual diagrams are intuitive, and the narrative flows smoothly from motivation to results. The supplementary materials provide necessary implementation details, enhancing reproducibility.

Significance is strong. The work contributes to a key ongoing trend: efficient and interpretable diffusion modeling. Its training simplifications (analytical balancing, PF-ODE compatibility) make diffusion-based motion synthesis more practical and theoretically consistent. These ideas can influence future research in both motion generation and general generative modeling.

### Weaknesses
Lack of formal analysis

The proposed balancing scheme is inspired by prior theoretical insights but lacks formal convergence or stability proofs.

Suggestion: Include a mathematical analysis (or at least empirical sensitivity tests) to show that the balance remains stable across datasets or motion scales.

Narrow experimental scope

Only unconditional motion is evaluated.

Suggestion: Extend to conditional setups (text-to-motion, action labels) or discuss how the framework could generalize to such tasks.

Computational evaluation missing

Efficiency claims focus on NFEs but omit runtime, training cost, and GPU memory comparisons.

Suggestion: Add computational benchmarks to support claims of superior efficiency.

Lack of variance reporting

Metrics like FID and diversity are reported as single values without standard deviation.

Suggestion: Provide averaged results over multiple seeds with confidence intervals.

Data assumptions

The method assumes access to full SMPL parameters.

Suggestion: Discuss adaptability to datasets with partial 3D keypoints or noisy inputs.

### Questions
1. How sensitive is the model to the specific analytical weight formulation? Could learned or adaptive weight schedules perform better?

2. Does the proposed balancing framework generalize to conditional diffusion models (e.g., text or action-conditioned)?

3. How does the model handle longer or variable-length motion sequences?

4. How robust is the model to missing or noisy SMPL parameters at inference time?

5. Would the removal of auxiliary losses reduce diversity in rare or extreme motion types?

### Soundness
3

### Presentation
3

### Contribution
3
