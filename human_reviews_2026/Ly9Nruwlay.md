# VCode: a Multimodal Coding Benchmark with SVG as Symbolic Visual Representation

- Avg Score: 3.33
- Decision: Reject
- Scores: 6, 0, 4

## Abstract
Code has emerged as a precise, executable medium for reasoning and action in the agent era. Yet progress has largely focused on linguistic-centric tasks, such as program synthesis and debugging, leaving visual-centric coding underexplored. Conventional image representations rely on dense RGB pixels that capture appearance but provide limited symbolic abstraction. Inspired by how humans reason over sketches, we advocate SVG code as a compact, interpretable, and executable visual representation. We introduce VCode, a benchmark that reframes multimodal understanding as code generation: given an image, a model must produce SVG that preserves symbolic meaning for downstream reasoning. VCode covers three challenging domains—general commonsense (MM-Vet), professional disciplines (MMMU), and visual-centric perception (CV-Bench). To assess symbolic fidelity, we propose CodeVQA, a novel evaluation protocol in which a policy model answers questions over rendered SVG; correct answers indicate faithful symbolic preservation. Empirically, frontier VLMs struggle to generate faithful SVGs, revealing a persistent gap between language-centric and visual-centric coding. To close this gap, we introduce VCoder, an agentic framework that augments VLMs along two axes: (i) Thinking with Revision, which iteratively analyzes discrepancies and refines SVG code; and (ii) Acting with Visual Tools, where detectors and parsers supply structured cues (objects, shapes, text) beyond intrinsic model capacity. Across benchmarks, frontier VLMs with strong reasoning score well overall yet remain limited on professional knowledge and 3D reasoning; VCoder delivers a +8.7 point overall gain over the top-performing Claude-4-Opus. Human studies further show that although VLMs score higher on raw images, humans are more robust on rendered SVGs—underscoring symbolic visual coding as a promising paradigm for human-like multimodal intelligence.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces VCode, a benchmark for visual-centric multimodal coding that redefines multimodal understanding as the task of generating SVG code from images. The work is motivated from the observation that most multimodal and coding benchmarks focus on linguistic or pixel-based representations, whereas SVG provides a symbolic, interpretable, and executable visual abstraction.
VCode covers three domains:  1) General commonsense (MM-Vet), 2) Professional knowledge (MMMU), and 3) Visual perception (CV-Bench).
To evaluate the symbolic fidelity of SVG representations, the authors propose CodeVQA, where a policy model must answer questions about the rendered SVG image. 
They further introduce VCoder, an augmented agentic framework that enhances existing vision–language models (VLMs) through:
1) Thinking with revision: iterative refinement based on visual discrepancies between generated and target images
2) Acting with visual tools: using detectors, segmenters and OCR to provide structured cues like object boundaries with text. 

Empirical results show that leading VLMs (e.g., GPT-5, Claude-4-Opus, Gemini-2.5) struggle on visual coding tasks, while VCoder achieves an +8.7-point overall improvement over Claude-4-Opus. Human studies show that people reason more robustly over symbolic SVGs than raw images, suggesting that symbolic visual coding could be key to more human-like multimodal intelligence

### Strengths
1) The paper introduces a novel paradigm: treating image understanding as code generation (SVG rendering).
2) The VCoder framework combining iterative refinement and external visual tools aligns with recent trends in agentic model enhancement.
3) Experiments are comprehensive, covering both closed- and open-source VLMs with detailed ablations (revision loops, tool usage, modality inputs).

### Weaknesses
1) The dataset contains only 464 image–question pairs, which is small compared to major multimodal benchmarks. Although the repurposing from MM-Vet/MMMU/CV-Bench ensures diversity, it may limit generalization and statistical reliability of reported differences.
2) CodeVQA uses an external policy model (GPT-4o-mini) as evaluator. This introduces evaluation bias and circularity, especially since some tested models are from the same family.
3) While the paper argues that SVG captures symbolic abstraction, it lacks quantitative or theoretical grounding for what constitutes “symbolic fidelity.” E.g., there could be metrics for structural alignment (e.g., object counts, relative positions) alongside SigLip and VQA accuracy.

### Questions
1) How sensitive are the CodeVQA scores to the choice of the policy model? Would results differ significantly if another evaluator (e.g., Claude-Sonnet or Gemini-Pro) were used?
2) Why was SVG chosen over other symbolic representations like scene graphs or DSLs for vector graphics? Could the same paradigm extend to 3D symbolic representations (e.g., CAD or mesh code)?
3) In Table 4, the Img2Text2SVG pipeline outperforms direct Img2SVG. Does this suggest that current models inherently reason better through language than through direct visual coding?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
The paper introduces VCode, a benchmark framing multimodal understanding as generating SVG code from images and reasoning over the rendered output. It proposes CodeVQA to test whether SVG-based representations preserve semantic visual information and VCoder, which combines iterative code revision and vision-tool assistance. Experiments show gains over existing VLM coders but also reveal persistent weaknesses in fine-grained visual reasoning.

### Strengths
1. The idea of using SVG as an intermediate symbolic space for vision-language reasoning is conceptually novel and touches on an underexplored direction in multimodal representation.

2. The work incorporates test-time revision and tool-assisted perception, which reflects awareness of limitations in current models and attempts to address them through modular augmentation rather than purely scaling.

### Weaknesses
1. The evaluation protocol is fragile: SigLIP similarity offers weak guarantees on fine-grained structure, and CodeVQA depends on the answering model’s biases and failure modes, making correctness a function of the evaluator rather than the representation. This undermines reliability and fairness, which is critical for a benchmark.

2. The dataset is almost entirely repurposed from prior benchmarks without substantial new curation or justification for domain coverage, scale, or annotation quality. As a result, it is unclear whether the benchmark truly captures the core challenges of the proposed problem.

3. The approach lacks grounding in practical vision tasks or downstream applications, and the SVG abstraction remains unconvincing as a scalable representation (especially for natural images with complex textures, occlusions, or fine geometry). Additional empirical evidence or ablations are needed to justify that benefits outweigh the loss of fidelity and that this direction can generalize beyond small synthetic-like cases.

### Questions
Is the evaluation protocol reliable?

### Soundness
2

### Presentation
2

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
This paper proposes a novel visual-centric benchmark that reframes multimodal understanding as code generation, and proposes an evaluation protocol that a policy model answers questions over rendered SVG. The paper finds that there is a persistent gap between language-centric and visual-centric coding, so it proposes an agentic framework that provides VLMs with two abilities: (1) thinking with revision; (2) acting with visual tools. The experimental results show that the proposed framework achieves a significant improvement in the visual-centric benchmark.

### Strengths
1. Extending language-centric coding to a new visual-centric coding task is an interesting and novel research direction.
  
2. This paper converts the multimodal understanding task into a visual-centric coding task and utilizes a Visual Model (VLM) to evaluate whether the generated code is an adequate and faithful visual representation.
  
3. The proposed VCoder framework is equipped with two capabilities: thinking with revision and acting with visual tools. Experimental results demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The dataset in this paper was not processed; it simply used the original images and QA from MM-Vet, MMMU, and CV-Bench. Since the SVG code is entirely generated by the VLM being evaluated, the authors only proposed SVG code generation as a benchmark approach. This benchmark does not design a unified principle for SVG code generation to guide subsequent VLM generation. The lack of a unified principle for SVG code generation can easily lead to instability in the generated code, resulting in unstable code evaluation.
  
2. While CodeVQA evaluates the accuracy of code generation, I believe there are two issues: First, code generation itself is a capability that needs careful evaluation; it should not be confused with multimodal understanding. For example, minor issues with the SVG code might cause rendering failures, leading to poor results, but this does not necessarily mean the model's understanding is flawed. Second, CodeVQA is easily influenced by text input. When the policy model struggles to obtain effective information from the rendered image, it may output an answer based on publicly available knowledge from the text input.
  
3. CodeVQA is disadvantageous for small models (e.g., models with around 7 parameters or less) because these small models have difficulty generating compliant SVG codes, resulting in poor final evaluation results. However, in reality, these small models have already achieved very good results in multimodal understanding capabilities.

### Questions
1. Refer to the issues raised in the weaknesses section.
2. The authors lack experimental results on other baseline models.

### Soundness
2

### Presentation
3

### Contribution
3
