# PerfGuard: A Performance-Aware Agent for Visual Content Generation

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
The advancement of Large Language Model (LLM)-powered agents has enabled automated task processing through reasoning and tool invocation capabilities. However, existing frameworks often operate under the idealized assumption that tool executions are invariably successful, relying solely on textual descriptions that fail to distinguish precise performance boundaries and cannot adapt to iterative tool updates. This gap introduces uncertainty in planning and execution, particularly in domains like visual content generation (AIGC), where nuanced tool performance significantly impacts outcomes. To address this, we propose PerfGuard, a performance-aware agent framework for visual content generation that systematically models tool performance boundaries and integrates them into task planning and scheduling. Our framework introduces three core mechanisms:  (1) Performance-Aware Selection Modeling (PASM), which replaces generic tool descriptions with a multi-dimensional scoring system based on fine-grained performance evaluations; (2) Adaptive Preference Update (APU), which dynamically optimizes tool selection by comparing theoretical rankings with actual execution rankings; and (3) Capability-Aligned Planning Optimization (CAPO), which guides the planner to generate subtasks aligned with performance-aware strategies. Experimental comparisons against state-of-the-art methods demonstrate PerfGuard’s advantages in tool selection accuracy, execution reliability, and alignment with user intent, validating its robustness and practical utility for complex AIGC tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents PerfGuard, a performance-aware agent framework for visual content generation. The key insight is that existing agent systems assume tools always work perfectly and use vague text descriptions that don't specify what each tool actually does well. To fix this, PerfGuard introduces three mechanisms: (1) PASM replaces text descriptions with multi-dimensional performance scores, (2) APU dynamically adjusts these scores based on actual execution results, and (3) CAPO aligns the planning process with performance-aware tool selection. Experiments on three benchmarks show improvements in tool selection accuracy and output quality for both image generation and editing tasks.

### Strengths
- The paper tackles an overlooked problem in agent-based visual generation. Most prior work assumes tools execute reliably, but this paper explicitly models tool performance boundaries. The combination of PASM, APU, and CAPO feels well thought out and addresses the problem systematically rather than with isolated fixes.

- The experiments are thorough. The authors compare against multiple baselines including diffusion models, CoT-based methods, and other agent systems across different task types. The ablation studies clearly show what each component contributes. I also appreciate the hyperparameter analysis, which helps with reproducibility.

- The paper is well organized. The framework architecture is clearly explained, with distinct agent roles and mechanisms. Figures 1 and 2 effectively illustrate the workflow. The appendix provides good implementation details.

### Weaknesses
1. The tool library only includes a specific set of popular models like FLUX, SD3, and Step1X Edit. I'm curious why certain relevant tools are excluded. For example, what about newer 2025 models or domain-specific editing tools? This raises questions about how well the framework generalizes to different tool ecosystems.

2. When adding new tools, the method initializes performance scores by averaging similar tools. This seems reasonable but might miss unique capabilities of novel tools. Have the authors tried other initialization strategies, like few-shot evaluation or transfer learning? A comparison would strengthen this design choice.

3. The quantitative metrics are comprehensive, but there's no user study. Since visual generation is inherently subjective, getting human feedback on the outputs would make the claims more convincing. Even a small preference study would help.

4. How does PerfGuard scale when the tool library grows large? If you have hundreds of tools, does the multi-dimensional scoring and adaptive updating become computationally expensive? The paper doesn't really discuss this.

### Questions
1. What criteria did you use to select tools for the library? Were any potentially useful tools excluded, and if so, why?

2. You found η=0.13 works best in your experiments. Does this parameter need tuning for different task types or tool libraries? How would users know what value to use?

3. Any plans to add user studies? It would be interesting to see if humans prefer PerfGuard's outputs over baselines.

4. How does computational cost scale with tool library size? What's the latency for tool selection and planning compared to baseline agent systems?

5. Could the performance dimensions extend to other visual tasks like video generation or 3D content creation? What modifications would that require? (Just for discussion)

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes PerfGuard, a performance-aware agent framework for visual content generation (AIGC). The core idea is to move beyond relying on vague text descriptions for tool selection. Instead, it proposes three mechanisms: (1) PASM, which models tool capabilities using a fine-grained, multi-dimensional scoring matrix ($M_p$); (2) APU, an online feedback mechanism to dynamically update this matrix based on execution outcomes; and (3) CAPO, a training strategy (inspired by DPO/SPO) to fine-tune the agent's planner to align its sub-task decomposition with the known tool capabilities.

The authors claim that this framework leads to SOTA performance on several AIGC benchmarks compared to existing agent-based methods.

### Strengths
Novel and Important Problem: The paper correctly identifies a key weakness in current agent frameworks: the "idealized assumption" of tool capabilities. The core idea of modeling fine-grained "performance boundaries" (PASM) is a significant and logical step forward for the field.

Complete Framework: The authors propose a complete, end-to-end vision, including selection (PASM), online adaptation (APU), and planner alignment (CAPO). This holistic approach is commendable.

Clear Presentation: The paper is well-written, and the core concepts are easy to grasp.

### Weaknesses
Unfair Comparison & Critical Confounding Variable: This is the most severe flaw. According to Appendix A.2, the framework uses GPT-4o for the Analyst and Self-Evaluator roles. This Self-Evaluator (GPT-4o) provides the entire learning signal for both the CAPO training (via Eq. 4 & 5) and the APU updates (via $R_{actual}$). The Analyst (GPT-4o) is also used during inference. The baseline methods (e.g., GenArtist) are not afforded this powerful, external model. Therefore, the SOTA claims are invalid. The paper is essentially comparing a system distilled from and assisted by GPT-4o against open-source models, which is an unfair comparison.

Lack of Disentanglement (The "1+1>2" Problem): The ablation study in Table 4 fails to answer the most critical question: where does the performance gain come from? Is it the better planner (CAPO) or the better selector (PASM)? The gains might just be additive. A crucial experiment is missing: Smart Planner (CAPO) + Naive Selector (Text-based). Without this, the value of the complex PASM/APU system is unclear, as is the value of the CAPO training.

Practicality and the "Cold-Start" Problem: The paper's claim of "practical utility" is undermined by its reliance on pre-existing benchmarks (T2I-CompBench, ImgEdit-Bench) to create the $M_p$ matrix (Section 4.1). This completely avoids the hardest and most expensive part of the problem: how to build this fine-grained performance matrix from scratch for a new domain or new set of tools. This significant limitation (which I know from experience is a major barrier) is not adequately discussed.

### Questions
[Re: Unfair Comparison]: Given the critical dependence on GPT-4o, can the authors please provide an ablation study where the Self-Evaluator and Analyst are replaced with a strong open-source VLM (e.g., LLaVA, Qwen-VL) or a non-LLM metric (e.g., CLIP Score)? This is essential to prove that the performance gains come from the PerfGuard framework itself and not from GPT-4o.

[Re: Disentanglement]: To prove the synergistic value of the framework, can the authors please provide the missing ablation: Baseline + CAPO only (i.e., the smart planner with the original naive, text-based tool selector)? This would allow us to understand the independent contribution of the planner and selector, and to check for any "1+1>2" effect.

[Re: Cold-Start Problem]: Can the authors please elaborate on the scalability of their approach? How would one feasibly construct the $M_p$ matrix in a new domain that lacks a comprehensive, multi-dimensional benchmark like T2I-CompBench? Acknowledging this as a major limitation would be a start.

### Soundness
3

### Presentation
4

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
The paper proposes PerfGuard, a performance-aware agent framework for visual content generation. It aims to address the gap in current agent-based systems, which often operate under the assumption that tool executions are always successful. PerfGuard incorporates three core mechanisms: Performance-Aware Selection Modeling (PASM), Adaptive Preference Update (APU), and Capability-Aligned Planning Optimization (CAPO). These mechanisms model tool performance boundaries, optimize tool selection, and align task planning with execution outcomes, providing enhanced reliability in visual content generation tasks.

### Strengths
1. The framework introduces a novel approach to visual content generation by incorporating performance-aware mechanisms.

2. The introduction of a multi-dimensional performance evaluation system for tools is a clear strength. It provides a more detailed and reliable method for tool selection, addressing the limitations of previous systems that relied on general textual descriptions.

### Weaknesses
1. The method heavily relies on the context-learning capabilities of large language models (LLMs). While this is an interesting approach, it may not offer substantial improvements over previous systems that already utilize LLMs for similar tasks.

2. The paper does not sufficiently discuss existing methods, particularly in relation to visual content editing tools. For example, "CLOVA: A Closed-Loop Visual Assistant with Tool Usage and Update" (CVPR 2024) has already considered tool evaluation and updates. This overlap should be addressed in the discussion to strengthen the paper's contribution.

3. While LLMs have shown remarkable capabilities in various domains, their use in image editing tool evaluation and updates is still in the early stages. The paper should address the potential limitations and challenges of this approach, including issues related to LLM performance, tool generalization, and robustness in diverse image editing tasks. The authors should try different LLMs to show the robustness of this method.

### Questions
see the weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
