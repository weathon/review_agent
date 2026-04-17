# FreeMo: Motion Generation with Structured Joint-Collision Energy

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
In this paper, we present FreeMo, a motion generation framework that produces physically plausible human motion by explicitly addressing self-collisions, where body parts intersect in unrealistic ways. Existing physics-aware generation models primarily handle external interactions, such as foot-ground contact, but are not capable of managing internal body dynamics. Although self-collisions can be corrected using post-hoc methods, these approaches are computationally expensive, difficult to scale, and compromise the differentiability and editability of the generative process. FreeMo integrates structured spatiotemporal constraints into the diffusion sampling process through a differentiable trajectory-level energy function that detects and penalizes persistent joint-level collisions. By directly optimizing joint positions in the latent space, FreeMo guides the generation away from physically implausible motions without compromising semantic alignment or motion naturalness. Experimental results show that FreeMo consistently reduces self-collisions while maintaining high-quality, controllable, and efficient motion synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces FreeMo, a framework designed to generate physically plausible human motion from text descriptions by explicitly preventing self-collisions. FreeMo integrates a novel, differentiable energy function directly into the diffusion sampling process, which detects and penalizes persistent, unrealistic intersections between body parts at the joint level.

### Strengths
Effective and Integrated Collision Avoidance.

High Efficiency and Quality Preservation.

### Weaknesses
- The main concern with this work is the limited contribution. The joint collision optimization has been widely explored by the community of geometry and animation, such as mesh recovery. The authors simply transfer this idea to motion generation and do not clarify the difference from the previous work. Besides, if the proposed method works and is novel, it can be used in other tasks, not only motion generation. Please try mesh recovery and music-driven motion generation. 

- The work might lack real application. The output is an SMPL sequence, but not a real character in the industry. The collision optimization is useless in real applications. 

- The method requires a latent optimization, which looks similar to that in MotionLCM and omnicontrl. What is the difference? 

- The latent optimization looks time-consuming in inference.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper introduces FreeMo, a framework for text-driven motion generation that addresses self-collision by integrating a Structured Joint-Collision Energy Function into the diffusion process. The method penalizes persistent joint collisions while preserving semantic alignment and motion quality. Experiments on the HardPoseText benchmark and HumanML3D subset demonstrate significant reductions in self-collision across multiple baselines, with competitive motion quality and efficiency.

### Strengths
- Tackles the underexplored issue of self-collision in motion generation.
- Embeds collision constraints directly into the generative process, avoiding inefficiencies of post-hoc corrections.
- Demonstrates effectiveness on diverse benchmarks with well-defined metrics.
- Well-written and organized with detailed experiments and analysis.

### Weaknesses
- Generated motions occasionally fail to fully align with text prompts, especially for complex descriptions.
- Focuses only on joint-level collisions, neglecting surface-level interactions.
- HardPoseText relies on GPT-expanded prompts, raising concerns about diversity and realism.
- Overemphasis on collision metrics; broader physical plausibility is underexplored.

### Questions
Please see Weaknesses for details.

### Soundness
2

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
The paper addresses the problem of self-collisions in text-to-motion generation, where a character’s limbs or body parts interpenetrate unnaturally. Current text-driven motion diffusion models excel at semantic alignment but often ignore physical constraints, leading to artifacts like limbs passing through each other. FreeMo is proposed as the first motion generation framework explicitly designed to avoid self-collisions during generation.

The authors introduce FreeMo, a plug-in optimization framework that integrates Structured Joint-Collision Energy into the diffusion sampling process. Rather than retraining models, FreeMo performs inference-time latent optimization (inspired by Diffusion Noise Optimization) on a pre-trained motion diffuser’s latent code. A joint-level collision discriminator is trained to predict collision probability per joint per frame During generation, if a joint is likely colliding (probability above a threshold), a penalty is applied via the joint-collision energy function This energy function has two structured components: (1) a spatial term that penalizes joints being too close (indicating potential intersection), and (2) a temporal term that penalizes large deviations from the original motion trajectory to preserve naturalness. The combined energy $E_{t,j}$ guides gradient-based updates to the latent noise such that the decoded motion avoids persistent collisions while maintaining semantic fidelity and smoothness. FreeMo thus steers the generation away from physically implausible configurations without compromising the text-conditioned content.

contributions are as follows:
1. Proposes the first self-collision-aware T2M generation framework with structured spatiotemporal constraints integrated into diffusion sampling. 
2. Designs a novel Structured Joint-Collision Energy Function, a differentiable trajectory-level objective that detects and penalizes joint collisions to improve physical plausibility without degrading semantic alignment or motion quality.
3. Demonstrates through experiments that FreeMo significantly reduces self-collision artifacts in generated motions while preserving motion naturalness, controllability, and efficient sampling.

### Strengths
- The paper tackles the long-neglected issue of intra-body collisions in generated motions, extending physics-awareness in text-to-motion beyond external contacts (like foot-ground) to internal body dynamics. This focus on self-collision is novel and improves physical realism in motion generation.
- The idea of injecting a differentiable collision-penalty into diffusion sampling is innovative. FreeMo’s Structured Joint-Collision Energy combines spatial and temporal constraints in a principled way. This structured objective is more stable and informed than a naive post-hoc fix, leveraging the diffusion model’s latent space for optimization (building on the Diffusion Noise Optimization concept). The method smartly avoids retraining or altering the base model; it works as an add-on, which increases its practicality.
- FreeMo achieves dramatic reductions in self-collision metrics on challenging prompts. For example, on the HardPoseText benchmark, applying FreeMo to MotionLCM slashes the collision rate from 31.35% to 1.44%. Similar large improvements are seen for other models (MLD: 23.09% → 0.49% collision rate) with negligible or even slightly positive impact on semantic alignment and motion quality. Notably, R-Precision (text-motion retrieval accuracy) actually improved in these cases (e.g. 0.231 → 0.262 for MLD), and motion smoothness (jitter) improved as well (e.g. 3.96 → 0.85 for MotionLCM) – indicating the method removes physical artifacts without degrading the intended motion. The qualitative examples corroborate that FreeMo removes unwanted self-intersections while preserving intended contacts (e.g. keeping hands on thighs when the prompt requires it).
- The paper introduces HardPoseText, a novel benchmark of 50 challenging text prompts specifically designed to induce self-collisions. This dataset provides a rigorous stress-test for the model’s ability to handle contorted or close-contact poses. The evaluation protocol is comprehensive – it uses collision-specific metrics (Collision Rate and Collision Score based on body-part occupancy overlap) as well as standard text-to-motion metrics (R-Precision, Frechet Distance, Diversity, Multimodality, Jitter) to ensure that motion quality and diversity are maintained. Baselines include state-of-the-art diffusion models – MLD, MotionLCM, SALAD – covering both motion-in-latent-space and skeleton-diffusion approaches, and a post-hoc collision fixer (COAP). This breadth of comparisons strengthens the evaluation. FreeMo’s superiority to COAP is especially clear: FreeMo not only yields far fewer collisions, but is over 40× faster (COAP was extremely slow and still left ~21% collision rate vs. 1.44% with FreeMo).
- The methodology is grounded in sound principles. The paper carefully justifies design choices: e.g., optimizing in latent pose space (to avoid the instability of direct joint-space optimization that could cause unnatural motions), and combining spatial and temporal terms (spatial term to effectively detect close contacts, temporal term to prevent the optimization from introducing motion discontinuities). The provided equations and algorithmic details make the approach transparent. Overall, the writing is clear and well-organized, with a logical flow from problem setup, method, to experiments. The authors even note using an LLM to refine the manuscript’s grammar and clarity, resulting in a polished presentation. Figures (e.g. Fig.2) nicely illustrate how the diffusion model, discriminator, and energy function interact.

### Weaknesses
- While useful, the core technique can be seen as an incremental adaptation of existing diffusion guidance methods to the collision problem. The approach builds on known ideas like Diffusion Noise Optimization (modifying the diffusion latent to enforce constraints) and leverages a pretrained collision detector inspired by prior work (COAP). The contribution lies in combining these pieces (joint-collision discriminator + latent optimization) for self-collision avoidance. This is a practical innovation but arguably not a fundamentally new generative model or learning paradigm – it’s a post-generation refinement plug-in. Some may view this as a narrow technical improvement rather than a breakthrough, reducing the perceived novelty.
- The impact of solving self-collisions, while non-trivial, might be limited to niche scenarios. The authors heavily focus on the HardPoseText benchmark they curated, which emphasizes extreme poses prone to collisions. In more typical text-to-motion settings, self-collisions occur relatively rarely (indeed, on a standard HumanML3D test subset, base models already have low collision rates). Thus, the practical significance of FreeMo could be questioned – it excels on contrived hard cases but offers only marginal gains on ordinary prompts. This raises concerns that the contribution, while valuable for physical plausibility, addresses a problem that many users might not consider critical in day-to-day applications of motion generation.
- The HardPoseText benchmark, though useful, is created by the authors and specifically constructed to stress test collisions in baseline models. There is a risk of evaluation bias: the prompts were chosen because existing models fail badly on them, which naturally magnifies FreeMo’s improvements. While this showcases FreeMo’s strengths, it could be seen as overfitting the evaluation to the method’s niche. An open question is how often such tangled poses appear in realistic user prompts or motion datasets. The paper would be stronger if it demonstrated clear wins on a more standard benchmark or real-world scenario, beyond the curated set.
- The comparisons, although covering major diffusion models, omit some relevant baselines. Notably, the authors discuss CLOAF (a recent collision-aware flow method) in related work but provide no quantitative comparison because code wasn’t available. This is understandable, yet it means an existing learned solution for collisions isn’t directly evaluated. Additionally, other physics-aware generation methods (e.g. PhysDiff, ReinDiff) that handle external contacts are not compared in terms of physical plausibility or motion naturalness – a comparison or discussion could illuminate whether FreeMo’s benefits are complementary to those or if similar results could be achieved by simpler means (e.g. adding joint repulsion forces during sampling). The paper focuses on the specific baselines and one post-hoc method (COAP), leaving a gap in positioning FreeMo relative to all prior physics-based methods.
- Despite claims of no quality degradation, there are minor signs of trade-off. For example, with SALAD as the base model, the MM-Dist (multimodality distance) worsened from 4.581 to 5.154 when FreeMo is applied (higher MM-Dist presumably indicating the generated motions deviate more from references or have reduced quality). This suggests FreeMo might slightly affect the motion distribution or diversity for certain models. While other quality metrics generally held steady or improved, the paper’s strong claim that physical constraints are added “without compromising semantic alignment or motion quality” could be a bit over-stated. In edge cases, FreeMo’s optimization might subtly pull motions away from the original distribution (e.g., making certain motions more stiff to avoid collisions, as hinted by a slight diversity drop). The paper lacks a nuanced discussion on these trade-offs – a small weakness in the analysis.
- FreeMo introduces an optimization loop at inference, which is computationally cheaper than prior post-processing but still adds overhead. The results show generation speed (ms per frame) drops significantly with FreeMo (e.g., from ~0.94 to 16.95 ms/frame on MotionLCM, and similar 10×–20× slowdowns for others). This is still real-time (approx 60 FPS for MotionLCM+FreeMo) and far better than COAP’s 826 ms/frame, but it means throughput is reduced. For very large batches or deployment scenarios, this extra cost might matter. The paper positions FreeMo as “efficient” which is true relative to heavy optimization-based methods, but the overhead might deter some practical uses, especially if collisions are infrequent to begin with.
- By design, FreeMo operates on joint positions and does not explicitly handle mesh self-intersections beyond joints. The authors acknowledge this as a limitation – current motion generators output only skeletons, so FreeMo focuses on joints and would need extension when mesh-based generation becomes feasible. Thus, any collisions that don’t involve joint centers (e.g., forearm vs torso surface contact) might not be fully penalized. This is a forward-looking limitation, but it means FreeMo isn’t a complete physical correctness solution; it’s restricted to joint-level proxies. Future work is needed to truly guarantee mesh-level collision avoidance.

### Questions
- How exactly was the joint-level collision discriminator trained (data source, label generation protocol, class balance, architecture, and training set size)?
- How is the collision threshold τ chosen and is it fixed across datasets/models or tuned per base model?
- Does the discriminator generalize across skeleton topologies (e.g., HumanML3D vs. other rigs)? Any cross-skeleton calibration needed?
- What is the false positive rate on intended contacts (e.g., hands-on-thigh) and how did you measure it?
- What are the precise forms of the spatial and temporal terms and their relative weights (λs, λt)? Are the weights scheduled over diffusion steps?
- Have you tested alternative temporal regularizers (e.g., jerk/acceleration penalties) and how do they affect jitter vs. collision rate?
- Do you normalize the per-joint energy by limb length or joint variance to avoid over-penalizing small bones?
- At which timesteps is latent optimization applied (all, early, or late steps)? How many gradient updates per step?
- How does FreeMo interact with classifier-free guidance scales used by each base model?
- What optimizer and step-size schedule are used for the latent updates, and is gradient clipping required for stability?
- For HardPoseText, can you detail the prompt construction pipeline and provide evidence that “hardness” is not overly specific to MDM (e.g., overlap of hardest prompts across different base models)?
- Do results hold on longer-horizon motions and unseen, real user prompts? Any user study or in-the-wild evaluation?
- Does FreeMo extend to multi-person or human-object scenarios, and if not, what are the conceptual blockers?
- Please provide an intuitive explanation and sensitivity analysis for Collision Rate/Score (e.g., to occupancy radii, skeleton scale).
- Can you report per-prompt breakdowns and confidence intervals for collision metrics and R-Precision to assess statistical significance?
- For the observed SALAD MM-Dist degradation, can you analyze whether the change is due to diversity reduction or distribution shift?
- Could you include a simple “joint-repulsion during sampling” baseline (no learned discriminator) to isolate the value of the learned term?
- Even without CLOAF code, can you approximate a flow-based collision prior or report a qualitative/ablation comparison?
- How do your results compare with external-physics methods (PhysDiff/ReinDiff) when both are applied (complementary or redundant effects)?
- What is the runtime breakdown (collision inference vs. gradient steps vs. denoising) and batch-size dependence?
- Is there an adaptive early-stopping criterion (e.g., when collisions drop below τ for K steps) to reduce overhead on easy prompts?
- Beyond a global threshold, do you condition the energy on the text prompt to allow intentional contacts? If not, how robust is the heuristic?
- Any failure cases where FreeMo removes desirable high-contact motions (e.g., self-hug) and how could prompt-aware allowances be added?
- Since mesh-level collisions aren’t handled, can you comment on feasibility of integrating SDF/mesh proxies during diffusion without prohibitive cost?
- What are the main failure modes you observed (e.g., stiffness, mode collapse in tight poses), and recommended user-facing knobs to trade off quality vs. collisions?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses self-collisions in text-to-motion diffusion models and proposes FreeMo, an inference-time plug-in that optimizes the latent code using a Structured Joint-Collision Energy (spatial + temporal). It integrates into sampling without retraining and shows large collision reductions on a curated “HardPoseText” benchmark. 

It's a well-executed paper with clean integration, but the conceptual contribution is incremental.  Broader evidence (standard benchmarks/user studies), deeper analysis of trade-offs and sensitivity, and/or comparisons to other collision-aware generators would help strengthen the paper.

### Strengths
- The authors provided a clear and practical formulation. Joint-level gating plus spatiotemporal energy is well-motivated and can fit neatly into standard sampling pipelines.

- It is noted that collision rates drop dramatically (e.g., for MLD/MotionLCM), with qualitative examples that preserve intended contacts.

- The paper introduces a stress-test dataset, defines collision metrics, and includes ablations and a stability check on standard prompts.

- Plug-and-play: It works with multiple backbones and avoids retraining, making it easy to employ in practice.

### Weaknesses
- There is no significant technical novelty rather than combination of some previously introduced techniques.

- The inference-time optimization adds a 10–40x per-frame cost versus base samplers.

- Small regressions (e.g., diversity/MM-Dist) suggest occasional distribution shifts. The authors are suggested to have deeper failure-mode and sensitivity analyses. 

- Joint-centric optimization can miss mesh-surface interpenetrations.  The paper acknowledges this, but it narrows the scope.

### Questions
1. Can the authors report HardPoseText difficulty rankings across multiple base models to reduce selection bias?

2. How sensitive are results to the discriminator threshold and energy weights? 

3. Could prompt-aware allowances for intentional contacts reduce false positives in poses with deliberate self-contact?

### Soundness
3

### Presentation
3

### Contribution
3
