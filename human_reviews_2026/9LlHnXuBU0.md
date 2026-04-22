# DecompDreamer: A Composition-Aware Curriculum for Structured 3D Asset Generation

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 6, 4, 4

## Abstract
Current text-to-3D methods excel at generating single objects but falter on compositional prompts. We argue this failure is fundamental to their optimization schedules, as simultaneous or iterative heuristics predictably collapse under a combinatorial explosion of conflicting gradients, leading to entangled geometry or catastrophic divergence. In this paper, we reframe the core challenge of compositional generation as one of optimization scheduling. We introduce DecompDreamer, a framework built on a novel staged optimization strategy that functions as an implicit curriculum. Our method first establishes a coherent structural scaffold by prioritizing inter-object relationships before shifting to the high-fidelity refinement of individual components. This temporal decoupling of competing objectives provides a robust solution to gradient conflict. Qualitative and quantitative evaluations on diverse compositional prompts demonstrate that DecompDreamer outperforms state-of-the-art methods in fidelity, disentanglement, and spatial coherence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes DecompDreamer, a staged optimization framework for compositional text-to-3D generation. The claim is that failures of prior methods originate from scheduling rather than representation: simultaneous or iterative optimization induces catastrophic gradient conflict across objects and relations.

### Strengths
1. The paper identifies a convincing theoretical bottleneck in compositional generation—gradient conflict arising from naive scheduling. This claim is not just theoretical; it is well-supported by the loss-dynamics analysis in Figure 5, which clearly visualizes the optimization divergence of GraphDreamer (Heuristic 3) versus the stable, two-stage convergence of the proposed method (Heuristic 4).
2. The method shows a clear and significant advantage as the number of objects and relations increases. As shown in Table 1, the performance of baselines like GraphDreamer collapses when moving from $\le3$ objects to $>3$ objects. DecompDreamer, in contrast, maintains high semantic alignment and user preference, proving its ability to handle the combinatorial complexity that cripples prior work.
3. Explicit and Effective Disentanglement: The framework does not treat object disentanglement as a mere by-product of good generation. Instead, it is explicitly engineered through a suite of complementary techniques: object-wise Gaussian tracking, targeted relational optimization, view-aware supervision, and the use of negative prompts. This results in clean geometric isolation of individual components (as seen in Fig. 1, 6, and 13), a major improvement over the blended and entangled geometry common in baselines.

### Weaknesses
1. The paper's primary weakness is the difficulty in attributing the final performance gains solely to the novel scheduling. The staged curriculum (Heuristic 4) is introduced concurrently with several other significant improvements (e.g., switching to flow-based guidance, view-aware alignment, negative prompts). The ablation study in Figure 7, while useful, does not fully isolate the scheduling component from these other powerful techniques. It is unclear how much of the gain comes from the "structure-then-detail" curriculum versus the other, more localized, improvements.

2. The paper's thesis is about solving optimization failure modes (divergence, conflict, entanglement). However, the main quantitative comparison (Table 1) reports semantic alignment metrics (CLIP, Pick-A-Pic, User Study). While the loss analysis in Figure 5 is excellent, it is relegated to a secondary analysis. The main SOTA comparison would be far more convincing if it included direct metrics for these claimed optimization failures, such as a "divergence rate" on complex prompts, a "geometric entanglement" score, or a "relational violation" metric.

3. The set of baselines is insufficient. The paper omits comparisons to several key state-of-the-art works from 2025, most notably Hunyuan3d 2.0 (Zhao et al., 2025) and STEP1X-3D (Li et al., 2025). Both of these models have demonstrated extremely high-fidelity and controllable asset generation. Without this comparison, the paper's claim to SOTA performance is unsubstantiated.

Zhao, Zibo, et al. "Hunyuan3d 2.0: Scaling diffusion models for high resolution textured 3d assets generation." arXiv preprint arXiv:2501.12202 (2025).
Li, Weiyu, et al. "Step1x-3d: Towards high-fidelity and controllable generation of textured 3d assets." arXiv preprint arXiv:2505.07747 (2025).

### Questions
1. How does the system behave when the scene graph or spatial priors from the VLM are partially incorrect or noisy? Could you provide some examples where the decomposition quality is programmatically degraded?

2. Can you provide a qualitative and quantitative comparison against Hunyuan3d 2.0 and STEP1X-3D, especially on the complex, multi-object prompts where your method claims to excel?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
DecompDreamer addresses a fundamental limitation in text-to-3D generation: the failure of existing methods on compositional prompts with multiple objects and complex relationships. The paper reframes compositional 3D generation as an optimization scheduling problem rather than solely a representation problem. The key contribution is a staged curriculum that decouples optimization into two phases: (1) joint relationship modeling to establish global structure, and (2) disentangled object refinement for high-fidelity details. The method uses Gaussian Splatting with a VLM-generated scene graph and demonstrates improvements over Gala3D, GraphDreamer, and other baselines.

### Strengths
1. This work provides a clear taxonomy of optimization heuristics (holistic, simultaneous, iterative, staged) and articulates why prior approaches fail under gradient conflicts, which is a valuable conceptual contribution beyond the specific method.
2. The loss dynamics analysis (Figure 5) provides direct evidence supporting the theoretical claims.
3. The paper is well-written with effective visualizations. Figure 3's pipeline overview and Figure 5's loss curves are particularly clear.

### Weaknesses
1. The contribution is primarily combining existing techniques with a staged schedule.
2. The proposed method relies heavily on VLMs for scene graph generation. What would happen with incorrect scene graphs? Is this method sensitive to spatial estimation errors? More empirical validations are required.
3. The generation costs are unacceptable for real-world applications. 90-495 minutes per scene is impractical. While faster than Gala3D on 6-11 objects, it's much slower than feedforward methods (Trellis, Hunyuan3D 2.1, etc.). The paper doesn't explore whether the staged curriculum concept could accelerate optimization or transfer to faster feedforward models.
 4. Janus problem and fine-grained decomposition still fail (Figure 14), which is a core problem of all SDS-based methods.
5. Missing Reference. The idea of "temporally decouples competing objectives" during SDS optimization is close to eDiff-I [Nvidia] and ThemeStation [siggraph 2025].

### Questions
1. How does the method scale beyond 11 objects? Are there fundamental limits due to scene graph complexity or GPU memory?
2. When VLM scene graphs are wrong, can the optimization partially recover? What's the worst-case behavior?
3. Could the idea be transferred to feedforward 3D generation methods (Trellis, etc.) for compositional generation?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a method for compositional 3D scene generation. It can generate compositional 3D assets from text prompts using the SDS loss. The overall pipeline first employs a Language Model to generate object prompts and edge prompts, which describe the shape of each object and the relationships among them. Then, the method uses the SDS loss to compose and optimize the individual objects into a coherent 3D scene.

### Strengths
1. The paper is well-written and easy to follow.

2. The method achieves superior visual quality compared to other competitive approaches, demonstrating its effectiveness.

### Weaknesses
1. The contribution and novelty of this method are relatively weak. The overall pipeline is quite similar to GraphDreamer [1], and the proposed view-dependent SDS loss appears to be more of an engineering refinement rather than a substantial methodological innovation. Therefore, I believe it does not meet the novelty bar required for ICLR.

2. The optimization process is slow and unstable, often requiring manual filtering to select satisfactory results, which limits its practicality and robustness.


[1] Compositional 3D Scene Synthesis from Scene Graphs

### Questions
Because the SDS-based optimization process is unstable, and most state-of-the-art 3D generation methods are now native 3D approaches, would it be possible to compare with native 3D scene generation methods such as PartCrafter [2]? Alternatively, could the authors discuss the advantages or priorities of their method compared to these native 3D approaches?

[2] PartCrafter: Structured 3D Mesh Generation via Compositional Latent Diffusion Transformers

### Soundness
3

### Presentation
3

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
This paper addresses the problem of generating 3D assets/scenes from compositional textual prompts, i.e., prompts that describe multiple objects and their spatial relationships. 

To address this, the authors introduce DecompDreamer, a framework built around a composition-aware curriculum (i.e., staged optimization strategy) for structured 3D asset generation. The key idea is to decouple the optimization of inter-object relationships (the scaffold/layout) from the high-fidelity refinement of individual objects. Specifically:

The method first generates / decomposes the scene prompt into multiple objects (via a vision-language model) and identifies relationships among them. 
It then uses a progressive optimization schedule: initially prioritize modeling the layout/structural scaffold and inter-object relations, then shift to refining the geometry and appearance of individual components. This decoupling is meant to reduce gradient conflict and improve the disentanglement of objects. 

In short, this work contributes a novel pipeline and training/optimization schedule specifically targeted at the compositional text-to-3D scene generation task.

### Strengths
- This work explicitly models the inter-object relationships, making the pipeline more understandable and clear.
- The visualization results of this work have better layout planning and interaction between different objects.
- Well-written and clear motivation. Easy to comprehend.
- The analysis of gradients is good

### Weaknesses
- Though the layouts and interactions between objects seem to be better, it is obvious that the visual quality of instances is inferior. 
- Though the runtime is greatly reduced compared to GALA3D and GraphDreamer, it is still long.

### Questions
- I am wondering if the VLMs are producing bad coarse initializations (wrong relationships between different objects). Can this pipeline correct the inter-object relationship with iterative optimization? Or the results are closer to the bad initialization results
- A comparison and discussion on [1] and [2] is needed. What is your advantage over these works? Will it be better to give an explicit spatial guidance as [2] does?
- What is the success rate of this work? In the failure cases, I am wondering if, first generating with a reconstruction model and then optimizing, these failure cases will be eased a little? Just like what [2] did.

I am willing to increase my rating if you can provide a comprehensive discussion on these questions.
[1]: CompGS: Unleashing 2D Compositionality for Compositional Text-to-3D via Dynamically Optimizing 3D Gaussians
[2]: Layout-your-3D: Controllable and Precise 3D Generation with 2D Blueprint

### Soundness
3

### Presentation
3

### Contribution
2
