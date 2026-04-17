# Enhancing Physical Plausibility in Video Generation by Reasoning the Implausibility

- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Diffusion models can generate realistic videos, but existing methods rely on implicitly learning physical reasoning from large-scale text-video datasets, which is costly, difficult to scale, and still prone to producing implausible motions that violate fundamental physical laws. We introduce a training-free framework that improves physical plausibility at inference time by explicitly reasoning about implausibility and guiding the generation away from it. Specifically, we employ a lightweight physics-aware reasoning pipeline to construct counterfactual prompts that deliberately encode physics-violating behaviors. Then, we propose a novel *Synchronized Decoupled Guidance* (SDG) strategy, which leverages these prompts through synchronized directional normalization to counteract lagged suppression and trajectory-decoupled denoising to mitigate cumulative trajectory bias, ensuring that implausible content is suppressed immediately and consistently throughout denoising. Experiments across different physical domains show that our approach substantially enhances physical fidelity while maintaining photorealism, despite requiring no additional training. Ablation studies confirm the complementary effectiveness of both the physics-aware reasoning component and SDG. In particular, the aforementioned two designs of SDG are also individually validated to contribute critically to the suppression of implausible content and the overall gains in physical plausibility. This establishes a new and plug-and-play physics-aware paradigm for video generation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This presents a novel training-free framework aimed at increasing the physical realism of videos produced by diffusion-based text-to-video models. The authors observe that while such models can generate visually compelling content, they often produce motions and interactions that defy intuitive physical laws—for example, objects moving without apparent forces or behaving in ways inconsistent with gravity.

To address these issues, the paper proposes a two-part method. First, a Physics-Aware Reasoning (PAR) pipeline automatically constructs “counterfactual prompts” that encode physically implausible scenarios: given a user prompt, PAR identifies entities, interactions and context, then generates an alternative prompt that preserves the scene yet violates some physical law. Second, the paper introduces Synchronized Decoupled Guidance (SDG), a guidance strategy applied during inference that leverages these counterfactual prompts to steer the generation away from implausible content. SDG has two key mechanisms: Synchronized Directional Normalization (SDN), which ensures early suppression of undesired directions, and Trajectory-Decoupled Denoising (TDD), which prevents the negative prompt’s suppression being weakened by trajectory bias.

Empirical results across multiple physical-domains (mechanics, optics, thermal, material) show that this approach improves metrics of physical-fidelity while preserving photorealism, and does so without retraining the base diffusion model. Ablation studies further indicate that both PAR and each component of SDG contribute meaningfully to the gains.

### Strengths
1. The authors provide a clear articulation of their motivation (video generation models frequently violate intuitive physical laws), a clean decomposition of the solution into two parts (PAR and SDG), and description of the mechanisms involved. The framing is intuitive (reason about “what’s impossible”, then guide to avoid it). The paper includes explicit discussion of the failure modes they address (lagged suppression, trajectory bias) which helps the reader understand why previous simple negative prompting may fail. The clarity of presentations, at least from the abstract and summary, is good.
2. The paper proposes a fresh and unconventional viewpoint: instead of only teaching a video-generation model to learn physical laws, it teaches the model to avoid physically implausible scenarios by reasoning about what should not happen. This “counterfactual prompt + inference-time guidance” paradigm is a creative combination of negative prompting, physical reasoning, and guidance design. The use of a “physics-aware reasoning” pipeline that constructs specifically tailored counterfactual prompts is particularly novel. Moreover, the subsequent guidance approach, dubbed “Synchronized Decoupled Guidance”, introduces non-standard mechanisms (synchronized directional normalization, trajectory-decoupled denoising) that address shortcomings of naive negative prompting.
3. The significance of the work is high, for several reasons. First, physical plausibility is a major challenge for generative video models: as visual fidelity improves, the next frontier is coherence, causality, and physical realism. A plug-and-play, training-free method that improves physical plausibility is of wide interest. Second, since the method does not require retraining or redesigning the base model, it is more likely to be adopted in practice, thus raising its impact. Third, by reframing the problem (reasoning about implausible content), the work may inspire future research in generative modelling, negative-prompt design, and physical reasoning. Finally, the fact that the approach crosses multiple physical domains suggests broader applicability, rather than being narrowly tailored.

### Weaknesses
1. Although the authors present quantitative gains, the magnitude of improvement appears quite small in certain cases. For example, in Table 1 the enhancement reported on the VideoPhy dataset for the model CogVideoX‑5B is minimal which suggests very limited practical impact. Furthermore, the qualitative examples (e.g., Figures 2 and 3) do not clearly demonstrate visually striking improvements. Even in the supplementary material—a group of oil droplets floating on water is claimed to be better—one can observe the droplets behaving as if “stuck” on the surface, which is itself inconsistent with correct fluid behavior. This suggests that while the framework aims to boost physical plausibility, the perceptual gains may not yet be compelling or unambiguous.
2. The experimental results show inconsistent benefits across backbone models: while the method improves performance more significantly for Wan2.1‑14B, the gains on CogVideoX-5B are much smaller (0.47 to 0.49). This raises concerns about whether the proposed inference-time framework generalises broadly across different text-to-video generation models. The authors might have strengthened their claims by including a wider range of baselines (different architectures, training conditions) or more challenging datasets. Without that breadth, it remains unclear how universally applicable the method is and whether it is robust in diverse settings.
3. The authors note in lines 162–168 that user prompts often lack explicit specification of physical processes, and therefore rely on prompt engineering via their pipeline to supply the “missing” physical context. This reliance on engineered prompt augmentation may undermine the claim that the underlying video generation model is being endowed with stronger physical commonsense or reasoning capabilities. Instead, the results might simply reflect improved prompt-following behaviour rather than enhanced physical modelling per se. In other words, a general-purpose world simulator should internally represent and reason about physical laws rather than being steered via language-based prompt tweaks. As such, while the method may produce somewhat more plausible videos, it remains unclear how much the model’s intrinsic physical understanding has been improved, versus how much the quality stems from smarter prompting.

### Questions
I don't see the implementation in the supplementary material. Will this be opensourced?

### Soundness
2

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
5

### Summary
The paper proposed an interesting training-free method for enhancing video generation model's obedience to physical rules. The method is similar with negative prompting in image generation, but with 3 key differences: 
1) the negative prompt is rewritten from the user prompt by an LLM, with explicitly physical implausible descriptions instead.
2) authors find that at early stage of denoising, the magnitude of  $∆_t = (ϵ^+_t - ϵ^-_t)$ is too small, thus they propose to use the direction of  $∆_t$ instead
3) different from negative prompting that both $ϵ^+$ and $ϵ^-$ are estimated on the same $x_t$, authors propose to always keep both the positive denoising trajectory and the negative denoising trajectory, as specified in line 277~288.

### Strengths
The proposed method is totally training-free, and can be plugged into any video generation model's inference process. Possible gain is moderate according to results presented in Table 1&2.

### Weaknesses
Authors only present metrics about physics obedience in the paper, it would be much better if there were also analysis and report about method's effect on video's visual quality as well as additional inference cost.

### Questions
1. My major concern about the method is about the "Trajectory-Decoupled Denoising" part in page 6. With *decoupled trajectory*, original branch $x_t^+$ and counterfactual branch $x_t^-$ are seperately inferenced (starting from the same intial noise I assume). My question is, when it comes to the later stage of denoising, $x_t^+$ would be remarkably different from $x_t^-$, with $∆_t$ calculated from such different trajectories, won't this compromise the visual quality of the generated video?
2. For the proposed method, the negative prompt is rewritten from the user prompt with explicit physics implausible description, that means, only ONE possible negative result could be mitigated by this method. So my question is, based on the current status of the method, how can we expand the effect to avoid more physics implausible generations?

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
Summary:   
This paper addresses a key limitation of existing diffusion-based video generation models: while these models can produce visually realistic content, they often generate physically implausible results that violate fundamental physical laws. To solve this problem, the paper proposes a training-free framework that enhances physical plausibility during the inference phase.

Contributions:  
（1）Training-Free Physics-Aware Paradigm: The framework eliminates the high costs of retraining or fine-tuning, a major limitation of prior physics-aware video generation methods. It acts as a plug-and-play solution that can be integrated with existing diffusion models, filling the gap in inference-time physical control for video generation.  
（2）Reasoning-Driven Counterfactual Construction: Unlike generic negative prompts that often lead to irrelevant violations, the Physics-Aware Reasoning pipeline leverages LLMs to generate targeted counterfactuals. These counterfactuals directly violate specific physical laws while preserving the original scene’s entities and context, ensuring that suppression signals are tightly aligned with physical principles.  
（3）Innovative Synchronized Decoupled Guidance: The Synchronized Decoupled Guidance strategy addresses the inherent flaws of traditional negative prompting through two complementary designs. Synchronized directional normalization enables early-stage suppression by focusing on the direction of suppression signals (rather than magnitude), while trajectory-decoupled denoising eliminates cumulative bias by evolving parallel latent trajectories for user prompts and counterfactual prompts. Ablation studies confirm that both designs are critical for improving physical plausibility.  
（4）Broad Generalization and Practical Utility: The framework achieves consistent performance improvements across four key physical domains (mechanics, optics, thermodynamics, and material interactions) when applied to two representative backbones. It also remains competitive with physics-aware models that require additional training. By preserving photorealism and inference efficiency, the framework supports practical applications such as scientific visualization and realistic content generation.

### Strengths
Strengths:   
（1）The paper stands out for its creative reimagining of physics-aware video generation, addressing critical gaps in prior work:  
1）It introduces atraining-free, inference-only framework—a departure from existing methods (e.g., PhyT2V, WISA) that require costly retraining or fine-tuning. This removes practical barriers to adopting physics enhancement for real-world use.  
2）Instead of generic negative prompts, it uses aLLM-powered Physics-Aware Reasoning (PAR) pipelineto generate targeted counterfactuals. These prompts violate specific physical laws while preserving scene entities, turning negative prompting into a precise physics-guided tool rather than a vague avoidance mechanism.  

（2）The work maintains high rigor through sound methodology and comprehensive validation:  
1）Methodological soundness: PAR’s LLM instruction template (with clear rules for entity analysis and counterfactual construction) ensures consistency. SDG’s design is mathematically grounded in diffusion model principles, avoiding arbitrary choices.  
2）Comprehensive experiments: It validates across two backbones (CogVideoX-5B, Wan2.1-14B) to prove generality, four physical domains (mechanics, optics, thermodynamics, material interactions) to show cross-domain effectiveness, and rigorous ablations to confirm that PAR and SDG’s two components are all necessary.  

（3）The paper is well-organized and accessible, ensuring complex ideas are understandable:  
1）Logical structure: It follows a tight "problem→method→result" flow—Introduction frames limitations of existing models/negative prompting; Methodology links each component (PAR, SDG) to a specific flaw it solves; Experiments organize results by benchmark, domain, and ablation for clarity.  
2）Technical transparency: Complex concepts (e.g., SDG’s designs) are explained with equations and plain language, and PAR’s workflow is visualized. Key terms (e.g., "counterfactual prompt") are defined consistently, avoiding jargon overload.

### Weaknesses
Weaknesses:   
（1）Over-Reliance on LLM Physics Accuracy Without Mitigation  
The framework’s Physics-Aware Reasoning (PAR) depends entirely on LLMs to generate valid counterfactual prompts, yet it ignores LLMs’ limitations in niche physical domains (e.g., non-Newtonian fluids, electromagnetism). No experiments test PAR’s ability to handle such scenarios, nor is there a check for LLM-generated counterfactual errors (e.g., misstating viscosity effects).  
（2）No SDG Hyperparameter Sensitivity Analysis  
SDG uses hyperparameters λ (directional normalization scale) and w (guidance strength), but their impact across domains/backbones is untested. A fixed λ may work for mechanics but fail for subtle optics (e.g., light scattering), harming “plug-and-play” utility.  
（3）No Validation on Ambiguous/Low-Quality Prompts  
Experiments use clear prompts (e.g., “A tennis ball thrown to the ground”) but ignore real-world vague prompts (e.g., “Something falling in water”) or typos. Ambiguity forces LLMs to make arbitrary assumptions, leading to misaligned counterfactuals.  
（4）Missing Comparisons to Inference-Time Baselines  
The paper only compares to trained physics models (e.g., PhyT2V) but omits recent inference-time methods (e.g.,Think Before You Diffuse,VLiPP) that also avoid retraining. This hides how SDG stacks up to peers.

### Questions
Questions:   
（1）Questions About LLM Reliability in PAR  
You use LLMs for Physics-Aware Reasoning to generate counterfactual prompts. Do you have data on how often LLMs produce inaccurate counterfactuals related to physical laws? How much do these inaccuracies reduce SDG’s ability to improve physical plausibility? A response will clarify if LLM dependence risks PAR’s effectiveness.  
（2）Questions About SDG Hyperparameters  
SDG uses λ and w as key hyperparameters. Do these values perform consistently across all four physical domains tested? If not, what hyperparameter settings work best for each domain? This will resolve confusion about SDG’s plug-and-play adaptability.  
（3）Questions About Ambiguous Prompt Handling  
Your experiments use clear prompts. Have you tested how PAR and SDG perform with vague user prompts? Does ambiguity lead to misaligned counterfactuals and worse physical plausibility scores? Answers will show the framework’s real-world utility.  
（4）Suggestions for Inference-Time Baselines  
You compare only to trained physics-aware models. Add comparisons to recent inference-time physics-aware methods that also avoid retraining. This will better position SDG against similar state-of-the-art approaches.  
（5）Suggestions for Niche Physics Validation  
You test PAR on common physics domains. Validate it on niche domains too, and add a lightweight physics verifier to filter inaccurate counterfactuals. This will improve the framework’s robustness across more scenarios.

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
This paper proposes a training free framework to enhance video diffusion models' physical plausibility. In detail, the pipeline first generate counterfactual prompts that describe physics-violating behaviors; then they propose a Synchronized Decoupled Guidance (SDG) strategy to mitigate implausible contents. This framework is highly related to negative prompt, but the key differences are: 1) it has a better designed negative prompts 2) It keeps two stream instead of one, the negative branch is denoised independently 3) It use only the direction instead of the mode. These changes are aiming to fix two gaps in negative prompting: 1) the suppression effect is weak at the beginning 2) the two branch shares same x_t, which causes cumulative bias.

### Strengths
1. This method is easy to understand, easy to implement, and plug-and-play, the authors provide pretty good ablations in Table 3, helps me to better understand the method.
2. Happy to see the authors tested more than one video models

### Weaknesses
1. maintaining two branchs almost doubles the cost, while in Table 3, the w/o Trajectory-decoupled denoising is getting 0.48, which is super close to 0.48. seems this does not contribute too much to the final results.
2. I checked the supp videos, it seems both baseline videos and SDG videos have some artifacts at the first several frames, this should not happens in Wan, I suspect there might be some implementation errors
3. In the supp videos, I found several cases have significantly over saturated generation, for example, the "A silver spoon is slowly inserted into a glass of crystal-clear water, revealing the fascinating visual changes and reflections as the spoon interacts with the liquid.", this is related to the guidance strength of 30 used in SDG, I highly suspect this will sacrafice FVD score, or video quality, the authors should also provide quality-wise eval.
4. Lacks the details of guidance scale for CFG baseline, also authors should provide a sweep over different guidance scale for CFG and SDG, for both physical benchmarks and video quality metrics.

### Questions
1. What's the guidance strength of cfg baseline, I am afraid the gap is mainly coming from a lower guidance scale for cfg. The authors should provide evidences to clarify.

### Soundness
3

### Presentation
2

### Contribution
3
