# SpaceControl: Introducing Test-Time Spatial Control to 3D Generative Modeling

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Generative methods for 3D assets have recently achieved remarkable progress, yet providing intuitive and precise control over the object geometry remains a key challenge.Existing approaches predominantly rely on text or image prompts, which often fall short in geometric specificity: language can be ambiguous, and images are difficult to manipulate. In this work, we introduce SpaceControl, a training-free test-time method for explicit spatial control of 3D asset generation. Our approach accepts a wide range of geometric inputs, from coarse primitives to detailed meshes, and integrates seamlessly with modern generative models without requiring any additional training. A control parameter lets users trade off between geometric fidelity and output realism. Extensive quantitative evaluation and user studies demonstrate that SpaceControl outperforms both training-based and optimization-based baselines in geometric faithfulness while preserving high visual quality. Finally, we present an interactive interface for real-time superquadric editing and direct 3D asset generation, enabling seamless use in creative workflows. Project page: https://spacecontrol3d.github.io/

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
SpaceControl is a training-free method that introduces explicit spatial control into 3D asset generation by conditioning a pre-trained generative model (Trellis) on user-provided 3D geometry, such as superquadrics or meshes. Unlike existing approaches that require fine-tuning or optimization, SpaceControl injects geometric guidance directly into the latent space during inference using a method inspired by SDEdit. This allows users to intuitively control the shape of generated assets while preserving the realism and generalization of the base model. The method supports multi-modal inputs (text, image, and geometry) and offers a tunable parameter to balance faithfulness to the input geometry with output realism. Extensive experiments and a user study show that SpaceControl outperforms both training-based and guidance-based baselines in terms of geometric alignment and visual quality.

### Strengths
1. Novel Training-Free Pipeline for 3D Spatial Control
SpaceControl introduces a new inference-time guidance mechanism that enables spatial control without any fine-tuning. This is a significant departure from existing methods like Spice-E, which require task-specific training, or optimization-based methods like Coin3D, which are slow and computationally heavy.


Generality and Flexibility: Because it does not modify the model weights, SpaceControl can be applied to any pre-trained Trellis model and supports a wide range of geometric inputs—from simple primitives to detailed meshes—without retraining.

2. Tunable Faithfulness–Realism Trade-off
A key contribution is the introduction of the control strength parameter τ₀, which allows users to smoothly interpolate between:

High Faithfulness: When τ₀ is large, the output closely adheres to the input geometry.

High Realism: When τ₀ is small, the output aligns more with the data distribution of the base model, often resulting in more natural-looking assets.

3. Multi-Modal Conditioning and Practical Usability
SpaceControl supports text, image, and geometry conditioning in a unified framework: Text guides the semantic content. Image influences appearance and style. Geometry controls the 3D structure.

### Weaknesses
1. Limited Methodological Innovation Relative to Spice-E

While the authors position SpaceControl as a novel training-free alternative, the core idea is conceptually similar to Spice-E, which also conditions generation on geometric primitives. The main difference lies in the implementation: Spice-E uses fine-tuning, while SpaceControl uses inference-time guidance.

Heavy Reliance on Trellis: The method is built directly on Trellis and does not introduce a new generative framework. Its success is largely dependent on the capabilities of the underlying model, which limits the perceived novelty.

2. Narrow Experimental Scope and Limited Category Diversity

The experiments are conducted on a limited set of object categories—primarily chairs and tables from ShapeNet, and toys from Toys4K. This raises questions about the method’s scalability and generalization to more complex or diverse shapes.

Lack of Complex Categories: There is no evaluation on categories with intricate geometry or high structural variability (e.g., humans, animals, vehicles, or scene-level generation).

3. Manual Hyperparameter Tuning and Global Control
The control strength τ₀ is a global, manually tuned parameter that applies uniformly to the entire object. This can be a limitation in practice:

No Part-Level Control: Users cannot specify that certain parts of the object should strictly follow the geometry while others can vary freely. This limits fine-grained creative control.

Per-Instance Tuning Required: The optimal τ₀ may vary across objects, requiring users to manually experiment for each generation, which hinders fully automated pipelines.

4. Limited Comparison to Other Training-Free 3D Methods

The paper compares only to Coin3D among training-free methods. Other relevant inference-time 3D guidance approaches—such as those based on score distillation sampling (SDS) or prompt-based editing in 3D—are not discussed or evaluated, leaving the reader uncertain about how SpaceControl fits into the broader landscape of training-free 3D control.

### Questions
1. Limited Methodological Innovation Relative to Spice-E
2. Narrow Experimental Scope and Limited Category Diversity
3. Manual Hyperparameter Tuning and Global Control
4. Limited Comparison to Other Training-Free 3D Methods

### Soundness
2

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
4

### Summary
SpaceControl is a training-free, test-time method that introduces explicit spatial control into the pre-trained 3D generative model Trellis. It achieves precise geometric control by encoding user-provided geometric inputs—such as superquadrics or meshes—into the model’s latent space and guiding the denoising process accordingly. The method supports adjustable control strength via a parameter τ₀, enabling users to trade off between geometric fidelity and visual realism. It outperforms both training-based and optimization-based baselines across multiple datasets and includes an interactive interface for real-time editing and generation.

### Strengths
1. The motivation is clear and practical. Allowing users to directly manipulate geometry using 3D primitives, aligns well with real-world design workflows.
2. The paper is well-structured, with a logical flow and clear figures.
3. The performance is good and includes a user study and interactive interface to demonstrate practical utility.

### Weaknesses
1. Limited technical novelty. The core idea of injecting geometric guidance before the appearance generation stage appears to be a natural extension of Trellis’s inherent two-stage design (coarse geometry ->  fine appearance). This raises the question of whether the controllability stems more from Trellis itself than from a genuinely novel contribution.
2. Potential generalizability problem beyond two-stage architectures. The method is tightly coupled to Trellis’s specific disentangled geometry-appearance pipeline. It is unclear how SPACECONTROL would adapt to end-to-end 3D generative models or single-stage frameworks, limiting its broader applicability.
3. Inadequate baselines. The comparison primarily includes methods published before or during 2024, missing recent advances in controllable 3D generation (e.g., 2025 works cited in the introduction). This weakens the claim of state-of-the-art performance and raises concerns about the completeness of the evaluation.
4. Missing References. The idea of 3D-reference-based 3D generation is similar to ThemeStation [siggraph 2024] and Phidias [iclr 2025], but only coin3d is compared.

### Questions
1. Generalizability to other 3D generative models: The method is tightly integrated with Trellis’s two-stage geometry–appearance pipeline. Can SPACECONTROL be adapted to other 3D generative architectures? and if so, how does its performance compare in terms of faithfulness and realism?
2. Extension to richer conditioning modalities: Currently, spatial control relies on 3D primitives (e.g., superquadrics) combined with text or optional image prompts. Could the framework naturally incorporate additional conditioning signals—such as depth maps, sketches, or multi-view images—to enable more expressive and intuitive user control?
3. Necessity and timing of geometric guidance: The geometric prior is injected only at the start of the structure generation stage (i.e., after initial noise). Given that Trellis already disentangles geometry and appearance, why not introduce spatial guidance even earlier (e.g., at ) or jointly optimize both stages with geometric constraints? Is the current design a limitation of the rectified flow formulation or a deliberate trade-off?

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
The paper proposes a method to add explicit spatial conditioning to a pretrained 3D generative model (Trellis) at test time, without retraining. Users provide geometric inputs, coarse superquadrics, which enables explicit control.

It performs latent-space interpolation between the control geometry’s embedding and random noise, then uses the pretrained structure-flow and appearance-flow models to denoise and decode the final 3D asset. A scalar parameter governs the trade-off between faithfulness to the spatial control and visual realism.

Experiments demonstrate that this training-free strategy yields stronger geometric fidelity than baselines while maintaining comparable realism scores (slightly reduced). The work also presents a browser-based interface for interactive shape manipulation through superquadrics.

### Strengths
**Training-free controllability.**
The whole system is designed without model training or surgery, which preserves the strong 3D generative prior in Trellis [1], and the approach introduced seems to be also applicable to other / future 3D generation under the Trellis framework.

**Good controllability.**
With superquadrics, this method enables explicit control with only a few primitives, which is user-friendly and accurate on most shapes. It also introduces a tunable adherence parameter that enables test-time adaptation according to actual usage.

**Empirical quality.**
This method consistently improves geometric alignment relative to strong baselines and Trellis itself.

```
[1] Trellis: Structured 3D Latents for Scalable and Versatile 3D Generation.
```

### Weaknesses
**Implied control limitation on curvy shapes.**
The 3D geometry representation, superquadrics, is often suitable for objects primarily containing convex shapes - cubes, spheres, cylinders, etc., while for concave shapes, it requires multiple superquadrics to compose. In pure 3D reconstruction domain, this does not impose a problem, but in terms of controllability, where simplicity or the number of the underlying primitives is more critical, superquadrics may suffer.

**Limited conceptual novelty.**
The latent-interpolation mechanism is directly analogous to image-domain test-time editing (e.g., SDEdit [1]) and does not introduce fundamentally new generative principles.

**Architecture dependency.**
The approach assumes access to a pretrained model with paired latent encoder and flow decoder (Trellis [2]), which means its generality to other 3D generative architectures is uncertain.

**Realism and appearance degradation.**
Geometrc metrics (Chamfer distance) is improved greatly, but this seems to be at the expense of realism (CLIP-I, FID) when compared to Trellis in Tab. 1. This means the proposed method is still a trade-off, just like the previous methods.


```
[1] SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations.
[2] Trellis: Structured 3D Latents for Scalable and Versatile 3D Generation.
```

### Questions
1. How many superquadircs would it need to precisely represent the elephant trunk shown in Fig. 3? I understand the current is one and rely on sacrificing controllability to generate this curve trunk, but what would the approximate amount of superquadrics if we want an accurate control?

### Soundness
4

### Presentation
3

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
This paper introduces SPACECONTROL, a novel training-free, test-time method for adding explicit spatial control to pre-trained 3D generative models. Addressing the limitations of ambiguous text prompts and cumbersome image editing, SPACECONTROL enables users to guide 3D asset generation using diverse geometric inputs, including primitives and meshes. The method works by encoding the user-specified geometry into the latent space of a powerful pre-trained model and using this latent code to initiate the denoising process from an intermediate timestep. A key contribution is a controllable parameter that allows users to fluidly trade off between geometric faithfulness and output realism. Extensive quantitative evaluations and user studies demonstrate that the method outperforms existing training-based and optimization-based baselines in geometric fidelity while preserving high visual quality. The authors also present an interactive user interface for practical creative workflows8888.

### Strengths
1: The proposed SPACECONTROL is a training-free method that injects explicit spatial control into a powerful pre-trained 3D generative model (Trellis) purely at test-time, which is efficient.

2: The framework provides intuitive geometric control using diverse inputs, from coarse primitives to detailed meshes, and introduces a single parameter that allows users to flexibly trade off between geometric faithfulness and output realism.

3: Extensive experiments show the method significantly outperforms both training-based (Spice-E, SPICE-E-T) and guidance-based (Coin3D) baselines in geometric alignment (measured by Chamfer Distance) while preserving high visual quality (measured by FID and P-FID).

4: This paper also presents a practical interactive user interface that enables online editing of superquadrics for their real-time conversion into detailed, textured 3D assets, demonstrating its high utility for creative workflows.

### Weaknesses
1: **Reliance on Manual Parameter Tuning.** The crucial trade-off between realism and faithfulness is governed by the $\tau_0$ parameter, which the paper states must be "selected manually"2. While Table 2 and Figure 3 show the effect of varying $\tau_0$, the optimal value appears to be object-dependent. This manual, per-instance tuning requirement undermines the method's practicality and hinders its use in automated generation pipelines.

2: **Lack of Part-Based Control.** The parameter $\tau_0$ is applied globally, "enforcing a uniform adherence level across the entire object"4. This is a significant limitation, as it prevents more nuanced, part-aware editing. For example, a user cannot simultaneously enforce high geometric fidelity for one part (e.g., the legs of a chair) while allowing high realism and creative variation for another (e.g., the backrest).

3: **Potential Information Bottleneck from Voxelization.** The method requires all spatial control inputs, including "detailed meshes," to be first voxelized into a $64 \times 64 \times 64$ grid before being passed to the encoder. This coarse, low-resolution representation may act as an information bottleneck, losing the very "fine-grained" details the user wishes to control. The paper does not analyze the impact of this $64^3$ resolution limit on the final geometric faithfulness.

4: **Limited Analysis of Failure Cases.** The paper presents many successful qualitative results (e.g., Figures 1, 4, 7, 8) but lacks a dedicated analysis of failure cases. For instance, what happens when the text prompt and spatial control are semantically contradictory (e.g., text="a car", spatial control=a boat)? Or when the control geometry is highly complex or topologically different from the objects in the pre-trained model's prior?

### Questions
1: **Auto optimization of $\tau_0$.** The paper states that the adherence parameter $\tau_0$ must be "selected manually" and that this is a limitation. Could the authors elaborate on the challenges of automating this selection? Have they explored any methods to predict an optimal or suggested $\tau_0$ based on the properties of the input control geometry (e.g., its complexity) or its relation to the text prompt? Clarification on whether this is a fundamental, subjective trade-off or a solvable engineering problem would be valuable.

### Soundness
3

### Presentation
3

### Contribution
3
