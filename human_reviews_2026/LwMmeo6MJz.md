# SceneLCM: Multi-Room Indoor Scene Generation with Latent Consistency Modeling

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
Automatically generating complex, realistic, and interactive indoor scenes from user prompts remains a formidable challenge, requiring scalability to multi-room environments, physical plausibility, controllable editing, and minimal human intervention. Existing paradigms, such as text-to-3D synthesis and layout-based retrieval, provide complementary advantages but suffer from limited automation, structural incompleteness, suboptimal textures, and inefficiency in large-scale settings.
To overcome these limitations, we introduce \textbf{SceneLCM}, an automatic and interactive framework that integrates Large Language Model (LLM)-driven layout generation with \textit{Latent Consistency Model} (LCM)-based scene optimization. Central to SceneLCM is the proposed \textbf{Consistency Trajectory Sampling} (CTS) loss, which maintains self-consistency during LCM optimization, enabling faster convergence and higher-fidelity 3D generation with theoretically bounded distillation error. 
Built upon CTS-guided LCMs, the SceneLCM pipeline comprises three key stages: (1) \textbf{Layout Generation} — LLM-guided 3D spatial reasoning transforms textual descriptions into parametric floorplans and object configurations, refined via iterative programmatic verification and cluster-based object orientation; (2) \textbf{Scene Object Generation} — objects are represented as 3D Gaussians and optimized with CTS for efficient, photorealistic results; (3) \textbf{Environment Optimization} — a normal-aware texture field encodes multi-resolution scene appearance, optimized with CTS along Zigzag camera trajectories to ensure geometric and texture coherence.
Extensive experiments demonstrate that SceneLCM produces high-quality, diverse, and physically coherent single- and multi-room scenes, while supporting texture editing and physically plausible modifications. Ablation studies validate the critical role of CTS in enabling high-quality, rapid generation across all scene components. The implementation will be publicly released to support reproducibility and foster further research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a text-driven multi-room realistic indoor scene generation model. Specifically, a latent consistency model is used for layout generation, where a consistency trajectory sampling is designed to enable faster convergence of training and high-fidelity generation. Scene objects are generated in the form of 3D GS representation. Experimental results show high-quality and diverse indoor scene results.

### Strengths
+ The first to address multi-room generation
+ theoretical analysis of the proposed consistency trajectory ssampling

### Weaknesses
- The presentation of paper is poor. There are quite a lot assets that contributes to the model and it is not clear which is the key module that contributes the results. 
- The motivation of the proposed consistency trajectory sampling is not clear explained. In L187-188, it is not clear about the Jacobian term and the mentioned auxiliary techniques such as perpendicular negative sampling. 
- I do not find the justification of with and w/o the use of CTS loss in the visual format. It is hard to tell whether the term contributes or not solely from mathematical deductions. 
- Missing citations of indoor layout generation papers. Is SubSection 3.3 one of the contribution of the paper? 
- It is not clear how object-based 3D GS methods are used with the environment generation. 
- Training data is not explained. 
- How is user study is conducted?
- The main results of the proposed method shown in Figure 5 are not good compared to SceneTex, for example. Artifacts are shown here and there. 
- Minor issues such as the apperance of L211.

### Questions
See above

### Soundness
2

### Presentation
1

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
The paper proposes SceneLCM, which combines LLM-driven layout generation with Latent Consistency Model (LCM)–based 3D optimization for automatic generation and interactive editing of single- and multi-room indoor scenes. The core technique is a new Consistency Trajectory Sampling (CTS) loss that maintains trajectory self-consistency during LCM distillation/sampling, accelerating convergence and improving fidelity; two theoretical results are provided (self-consistency and a bounded distillation error). The overall pipeline has three stages: (i) an LLM generates a floor plan and object configuration, which are then refined by iterative, executable program checks and cluster-based orientation assignment to resolve conflicts; (ii) objects are represented as 3D Gaussians and optimized with LCM+CTS; (iii) the environment uses a normal-aware multi-resolution texture field, optimized with LCM+CTS along a Zigzag camera trajectory for consistent texturing, while supporting texture/physically feasible edits and multi-room extension. Experiments demonstrate advantages in visual quality, multi-view consistency, and efficiency, with ablations verifying the key role of CTS.

### Strengths
1. Highly integrated, from end-to-end automation to interactivity: the pipeline unifies layout → objects → environment. In the layout stage, “programmatic verification + cluster-based orientation” reduces overlaps/empty space and mitigates orientation ambiguity, making the system practically robust.
2. Theory-backed CTS loss: within the consistency-model framework, CTS is derived from LCD, decomposes noise terms to remain compatible with techniques like Perp-Neg, and is supported by analyses of equivalence to the consistency objective and convergence properties.
3. Multi-room capability and coverage strategy: the proposed Zigzag trajectory adapts to different room scales, balancing surface normal coverage and overall coverage to stabilize texture optimization and improve consistency.
4. Normal-aware texture field: inspired by SceneTex and augmented with normal correlation, cross-attention transfers style between UV embeddings with similar normals, reducing tiling artifacts and multi-view inconsistency.

### Weaknesses
1.  How long does it take to generate a room, and how many tokens are consumed? The paper should explicitly report these figures.
2.  The examples suggest a relatively small number of objects per room. Can the authors evaluate scenes with larger object counts and report performance/quality when scaling up to denser rooms?
3. I’m particularly concerned about how this generative approach compares with other methods—for example, Holodeck’s object retrieval. Since generated materials are generally less accurate than assets retrieved directly from a database, what are the concrete advantages here? How does the overall speed compare? Please include examples for a side-by-side comparison.

### Questions
Please see the weaknesses.

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
SceneLCM: Multi-Room Indoor Scene Generation with Latent Consistency Modeling proposes an automatic and interactive pipeline for generating complex, realistic, and physically coherent multi-room indoor scenes directly from textual descriptions. The framework unifies LLM-guided layout generation with Latent Consistency Model (LCM) optimization, addressing scalability, physical plausibility, and texture realism.

### Strengths
The proposed Consistency Trajectory Sampling (CTS) loss is both theoretically grounded and practically effective, offering a new formulation of consistency learning with provable self-consistency and bounded error. The approach is technically sound, with clear formulations, empirical results, and ablations.

### Weaknesses
While the proposed Consistency Trajectory Sampling (CTS) is an interesting and general contribution, the rest of the pipeline feels largely incremental relative to existing works in text-to-3D and layout-based scene generation. Beyond CTS, the integration of LLM-based layout generation, Gaussian-based object representation, and texture optimization largely combines known components from prior works rather than introducing fundamentally new mechanisms. The paper would benefit from explicitly articulating what distinguishes its pipeline components from existing work.

The Layout Generation stage, in particular, depends on repeated iterative regeneration by an LLM, which is computationally expensive and not conceptually novel. Many recent LLM-driven layout papers employ similar “generate–verify–regenerate” loops, and comparable strategies exist in traditional procedural layout synthesis. The paper should clarify whether SceneLCM introduces any principled advancement—e.g., improved convergence, better realism, or more accurate statistical matching to real-world layout distributions. Currently, it remains unclear how the generated layouts align with the true distribution of complex indoor environments, where the number, category, and spatial arrangement of objects are far richer than what prompt engineering can specify.

Parts of the pipeline also appear overly manual or rule-driven, such as the orientation assignment and verification steps. The method seems similar to existing rule-based procedural systems (e.g. [1]), except that the initialization is performed by an LLM; it is not evident how this hybrid approach achieves measurable gains in realism or diversity.

Figure 9 is labeled as a quantitative comparison but only shows one qualitative example per method, making it prone to cherry-picking and does not substantiate statistical improvement.

While CTS is a promising and theoretically grounded concept, the paper’s structure is not centered around CTS. The narrative currently treats it as a supporting component rather than the core contribution. A more focused presentation—where CTS is the centerpiece, and multi-room scene generation is framed as an application—would improve conceptual clarity. Additional ablations against more consistency or distillation baselines (e.g., SDS, ISM, CDS) across different tasks would also better demonstrate the generality of CTS beyond this specific application.

Section 3.3 is overly minimalistic, providing only high-level descriptions without sufficient methodological details.

[1] Raistrick, Alexander, et al. "Infinigen indoors: Photorealistic indoor scenes using procedural generation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
Could you provide details of Iterative Programmatic Verification and Cluster-Based Orientation Assignment?

Does CTS generalize to other tasks?

### Soundness
3

### Presentation
3

### Contribution
2
