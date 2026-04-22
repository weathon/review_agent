# RePlan: Reasoning-Guided Region Planning for Complex Instruction-Based Image Editing

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Instruction-based image editing enables natural-language control over visual modifications, yet existing models falter under Instruction–Visual Complexity (IV-Complexity), where intricate instructions meet cluttered or ambiguous scenes. We introduce RePlan (Region-aligned Planning), a plan-then-execute framework that couples a vision–language planner with a diffusion editor. The planner decomposes instructions via step-by-step reasoning and explicitly grounds them to target regions; the editor then applies changes using a training-free attention-region injection mechanism, enabling precise, parallel multi-region edits without iterative inpainting. To strengthen planning, we apply GRPO-based reinforcement learning using about 1K instruction-only examples, yielding substantial gains in reasoning fidelity and format reliability. We further present IV-Edit, a benchmark focused on fine-grained grounding and knowledge-intensive edits. Across IV-Complex settings, RePlan consistently outperforms strong baselines trained on far larger datasets, improving regional precision and overall fidelity.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes RePlan, a plan-then-execute framework for complex instruction-based image editing under Instruction–Visual Complexity. A VLM planner performs chain-of-thought reasoning to output region-aligned guidance and evaluate models with a VLM-as-judge on four dimensions, reporting that RePlan improves consistency and overall scores among open models; RL with GRPO on ~1k instruction-only samples further boosts planning reliability.

### Strengths
- Problem framing: IV-Complexity is clearly defined and motivated: The paper formalizes how visual clutter + instruction intricacy interact and argues why fine-grained grounding is essential—setting up the need for region-aware planning.
- Region-aligned planner with interpretable output & interactivity: The planner emits structured text with explicit tags and JSON region guidance, and authors highlight that users can adjust regions/hints—improving controllability and reproducibility of edits.
- Compact but effective RL for planning: A two-stage GRPO scheme first secures valid format/reasoning, then adds image-level rewards. Using ~1k instruction-only samples is attractive for data efficiency.

### Weaknesses
- Planner–evaluator entanglement in ablations: The planner ablation compares Gemini2.5-Pro and Qwen2.5-VL (7B) as zero-shot planners while evaluation still uses Gemini-2.5-Pro as judge. This can disadvantage the Qwen-planner setting (mismatched style) and—even for Gemini-planner—induces a same-family bias. Consider swapping judges to check rank stability.
- Limited dataset scale & unclear coverage of long-tail/sensitive cases: IV-Edit has ~800 pairs and emphasizes complexity, but the paper provides only high-level distributions. It would help to report fine-grained category/size/occlusion breakdowns, text font/layout diversity, and any sensitivity filtering procedures. A data card with bias/fairness diagnostics would increase credibility. 
- Bounding-box robustness & error propagation not quantified: The approach hinges on bbox accuracy (planner → attention grouping). Authors note bbox errors (Gemini planner) qualitatively; however, there’s no systematic robustness test to bbox jitter, IoU drops, or mis-localized regions (impact on Target/Consistency). Please add stress tests that perturb boxes and report degradation curves.
- Latency/compute and scalability not reported: The method introduces planning (VLM) + editing (DiT with masked attention). Table 1 reports quality metrics only; no wall-clock comparisons across methods, nor scaling with number of regions (K). Reporting per-image latency and K-scaling (vs. iterative inpainting) would clarify practical trade-offs.

### Questions
N/A

### Soundness
4

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
3

### Summary
This paper presents RePlan, a reasoning-guided image editing framework designed to handle Instruction–Visual Complexity (IV-Complexity)—where complex textual instructions must be grounded in visually cluttered scenes. The framework couples a vision–language model (VLM) planner and a diffusion-based decoder, using a training-free attention region injection mechanism for precise, parallel, multi-region edits. Furthermore, the paper introduces IV-Edit, a new benchmark for evaluating complex instruction-based editing. Experiments show that RePlan outperforms several state-of-the-art baselines in consistency and reasoning-based grounding.

### Strengths
1. The introduction of “Instruction–Visual Complexity” formalizes an important challenge in multimodal editing and provides clear motivation for a new benchmark.

2. The combination of reasoning-driven planning with region-level attention control is well thought out and technically sound, avoiding costly retraining.

3. The new IV-Edit benchmark and extensive comparisons with both open-source and proprietary models strengthen the empirical validation.

### Weaknesses
1. Unclear realization of “step-by-step reasoning”: Although the paper claims that the planner decomposes instructions via step-by-step reasoning, neither the framework diagram nor the visualization results  clearly illustrate this multi-step reasoning process. The reasoning seems to occur implicitly rather than explicitly, weakening the claimed interpretability advantage.

2. The IV-Edit benchmark (∼800 samples) may not be large enough to comprehensively assess generalization, especially for reasoning-intensive tasks.

3. While RePlan improves consistency, the overall quantitative gains over strong baselines are modest, and some reinforcement learning details (e.g., reward weights, training stability) are insufficiently documented for reproducibility.

### Questions
Please see the weaknesses.

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
4

### Summary
This paper formalizes the challenge of Instruction-Visual Complexity and addresses it from a unified model perspective. Unlike existing unified models such as Bagel, the authors propose a region-level approach. They also introduce a training-free mechanism called Attention Region Injection and establish a new benchmark to evaluate this problem.

### Strengths
1. The proposed region-level control is both necessary and interesting.

2. A new benchmark is introduced to evaluate the proposed task.

3. Qualitative results demonstrate clear improvements over previous methods.

### Weaknesses
1. Figures 9–12 are confusing, as they are not referenced in the text nor sufficiently explained in the captions.

2. I wonder whether using the generated text from the VLM as input to existing MMDiT-style editing models (e.g., Flux-Edit) would still improve performance. This raises the question of whether the effectiveness comes from the comprehensive textual information or from the training-free editing mechanism itself.

3. In Stage 1 (RL), it is unclear how the rewards are computed and what the tag format looks like.

4. More conventional editing metrics or benchmarks should be included. Relying solely on the newly proposed benchmark makes the evaluation less convincing. I assume the proposed method should also perform well on existing editing benchmarks.

5. Qwen-Image shows better weighted results, which calls into question the advantages of the proposed method, given that Qwen-Image also adopts an MMDiT-based unified architecture.

6. Beyond Figures 1 and 6, more comparative results should be presented. Additionally, I could not find any supplementary results.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

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
This paper addresses the challenge of Instruction-Visual Complexity (IV-Complexity), which arises from the interplay between visual complexity (e.g., cluttered layouts, multiple similar objects) and instructional complexity (e.g., multi-object references, implicit semantics, world knowledge, and causal reasoning requirements) in image editing tasks. The authors propose refining the interaction between Vision-Language Models (VLMs) and diffusion models from a global semantic level to a region-specific level, leveraging VLMs' fine-grained perception and reasoning capabilities to generate region-aligned guidance.
Subsequently, the authors introduce RePlan, a framework that adopts a two-stage plan-execute paradigm: the VLM performs chain-of-thought reasoning to analyze the input image and instruction, producing structured region-aligned guidance (comprising bounding boxes and editing prompts); the diffusion model then executes precise multi-region parallel editing through a training-free attention-based region injection mechanism, while GRPO reinforcement learning (only ~1k instruction-only samples) enhances the VLM's planning capability. Experimental results demonstrate that RePlan outperforms existing models trained on massive-scale data on the newly proposed IV-Edit benchmark, achieving state-of-the-art performance among open-source models, effectively mitigating the issue of edit leakage to similar regions.

### Strengths
# Originality
- The paper systematically introduces the concept of "Instruction-Visual Complexity (IV-Complexity)," explicitly defining the challenges arising from the interplay between visual complexity (cluttered layouts, multiple similar objects) and instructional complexity (multi-object references, implicit semantics, knowledge reasoning).
- Through attention mask rules (prompt isolation, region constraints, background constraints, etc.), the method achieves precise multi-region parallel editing without requiring retraining of the DiT model.
- The construction of the IV-Edit benchmark represents the first evaluation dataset 
# Quality
- The paper compares 2 closed-source models (GPT-4o, Gemini) and 6 open-source models on the IV-Edit benchmark, employing Gemini 2.5-Pro as the evaluator with 5-point scoring across four dimensions: target localization, consistency, quality, and effect.
- Ablation studies validate the necessity of both RL training and chain-of-thought (CoT) reasoning.
# Clarity
The problem is articulated clearly, the methodology is described in reasonable detail, and there are no apparent grammatical errors.
# Significance
- The problem formulation is well-defined, providing novel methods and benchmarks for complex instruction-based editing.
- The interpretability and interactivity of region-guided editing enhance the practical utility of the system.

### Weaknesses
- All main results (Table 1) rely on 5-point scoring from a single closed-source model, presenting risks that Gemini may exhibit bias towards its own model (Gemini-Flash-Image) with more lenient scoring, or that results may not be reproducible.

- The paper does not specify the image sources, annotation procedures, or other critical aspects of the dataset construction process.

- Key hyperparameters such as learning rate, batch size, number of training epochs, and GRPO group size are not documented in either the main text or the appendix.

### Questions
1. RePlan relies on VLM-generated bounding boxes (bboxes), yet the paper does not discuss the importance of bbox accuracy. It remains unclear whether employing existing detection models such as Grounding DINO or Grounding SAM to provide more precise region bounding boxes would yield comparable or superior performance.

2. When multiple target regions spatially overlap or are highly adjacent, what impact does this have on the editing results? The paper lacks analysis of how the attention mask mechanism handles boundary conflicts in such scenarios.

3. How is the VLM training dataset constructed? The paper does not provide sufficient details regarding data collection, annotation procedures, or quality control measures.

4. Does the attention-based region injection mechanism depend on MMDiT-specific designs (e.g., joint text-image attention)? Can this approach be transferred to other editing models with different architectural backbones, such as UNet-based diffusion models?

5. IV-Edit primarily comprises realistic photographs and document editing tasks. The benchmark could be further extended to encompass artistic styles (e.g., cartoons, oil paintings) or specialized professional domains such as medical imaging.

6. User studies could be incorporated as an objective evaluation metric to assess whether model behavior aligns with human preferences, thereby providing a more comprehensive assessment beyond automated metrics.

7. The VLM primarily replaces the manual annotation of bboxes and the input of localized editing prompts by humans. However, if human-annotated bboxes and prompts are directly provided, could Flux Kontext achieve precise localized editing without requiring any training data? It would be beneficial to include visualization or quantitative experiments as reference while explicitly articulating the unique advantages of employing a VLM in this pipeline.

### Soundness
3

### Presentation
3

### Contribution
3
