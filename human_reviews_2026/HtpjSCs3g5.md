# PixelCraft: A Multi-Agent system for High-Fidelity Visual Reasoning on Structured Images

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 6

## Abstract
Structured images (e.g., charts and geometric diagrams) remain challenging for multimodal large language models (MLLMs), as perceptual slips can cascade into erroneous conclusions. Intermediate visual cues can steer reasoning; however, existing cue-based methods are constrained with low-fidelity image processing and linear, rigid reasoning patterns, limiting their effectiveness on complex structured-image tasks. In this paper, we propose PixelCraft, a novel multi-agent system for high-fidelity image processing and flexible visual reasoning on structured images. The system comprises a dispatcher, a planner, a reasoner, critics, and a set of visual tool agents. To achieve high-fidelity processing, we construct a high-quality corpus and fine-tune an MLLM into a grounding model, whose pixel-level localizations are integrated with traditional computer vision (CV) algorithms in tool agents. 
Building on this foundation, PixelCraft facilitates flexible visual reasoning through a dynamic three-stage workflow of tool selection, agent discussion, and self-criticism. 
Moreover, unlike prior linear reasoning patterns that simply append historical images, PixelCraft maintains an image memory to allow the planner to adaptively revisit earlier visual steps, explore alternative reasoning branches, and dynamically adjust the reasoning trajectory during discussion. Extensive experiments on challenging chart and geometry benchmarks demonstrate that PixelCraft significantly improves visual reasoning performance for advanced MLLMs, setting a new standard for structured image reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents PixelCraft, a novel multi-agent system designed to address significant failings in MLLM-based visual reasoning for structured images like charts and geometric diagrams. The authors identify two primary weaknesses in existing methods: low-fidelity image processing, which leads to perceptual errors, and rigid, linear reasoning patterns (like visual CoT) that cannot correct mistakes. PixelCraft's core contribution is a dual-pronged solution. First, it achieves high-fidelity processing by synergizing a compact MLLM, fine-tuned on a new synthetic corpus for precise pixel-level grounding, with classical CV algorithms that act as tool agents. Second, it enables
flexible, non-linear reasoning through a dynamic workflow involving a planner, reasoner, and critics. A key innovation is the "image memory," which allows the planner to adaptively revisit, recall, and branch from earlier visual steps, facilitating backtracking and exploration of alternative reasoning paths. The authors demonstrate through extensive experiments on challenging chart and geometry benchmarks that their system significantly improves reasoning accuracy for advanced MLLMs.

### Strengths
1. Well-motivated and significant problem. The paper addresses the challenging problem of visual reasoning on structured images, a well-known weak point for current MLLMs. This task is difficult because it requires precise perception of details (like axis values or geometric points) and multi-step logical deduction, where small perceptual errors can cascade into completely wrong answers. The paper clearly identifies these failings as a critical and practical area for improvement.

2. Comprehensive experimental setup. The authors validate their system on a strong suite of recent and difficult benchmarks, including CharXiv, ChartQAPro, and the auxiliary-line subset of Geometry3K. Testing against these diverse datasets demonstrates the system's robustness and generalizability. Furthermore, the comparison is not limited to simple baselines; it includes advanced agentic methods like Debate and Reconcile, which makes the consistent and significant performance gains of PixelCraft more convincing.

3. High-quality presentation and readability. The paper is very easy to read and well-organized, making the complex multi-agent architecture and reasoning process accessible to the reader. The figures are a standout component for presentation; Figure 1 provides an excellent, intuitive comparison of why standard CoT and Visual CoT fail while PixelCraft succeeds, and Figure 2 clearly illustrates the entire agentic workflow, from selection to discussion and correction.

### Weaknesses
1. Significant efficiency and latency issues. The multi-agent workflow, which involves a dispatcher, planner, reasoner, and multiple critics, inherently requires numerous sequential LLM inference calls for a single query. This leads to a substantial increase in latency and computational cost compared to simpler CoT methods, as confirmed by the paper's own analysis (e.g., 16.45s vs. 3.75s on CharXiv), which may limit its practical applicability.

2. Limited methodological novelty in the agentic framework. While the application to structured images is effective, the core agentic workflow (combining a planner, tool-use, and a reasoner) is a widely adapted paradigm in recent LLM research. The novelty of the agent framework itself is somewhat incremental, with the main contributions lying in the specialized toolset and its application rather than a fundamentally new agentic reasoning process.

3. Constrained generalizability and flexibility due to a fixed toolset. The system's success is heavily reliant on its manually-curated set of visual tools, which were specifically designed for charts and geometric diagrams. This presents a generalization bottleneck; the system would likely fail in scenarios with novel image types or tasks requiring tools not in its predefined set. This lack of flexibility to adapt or generate new tools on the fly is a key limitation, which the authors also acknowledge.

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes PixelCraft, a multi-agent system to perform complex reasoning over charts and geometric figures. The multi-agent system features a tool selector, a tool planner, visual critic, plan critic and multiple tool agents. Experiments on chart reasoning benchmarks like ChartXiv and ChartQAPro, and geometry reasoning benchmarks shows substantial improvement over other framework or vaniila visual CoT.

### Strengths
+ The paper proposes a novel multi-agent system to improve visual reasoning performance on structured image.
+ Achieves high-fidelity image process with a fine-tuned grounding model
+ The paper is well written and easy to follow.

### Weaknesses
+ My major concern is the cost of the proposed multi-agent system compared with vanilla visual CoT. As shown in Appendix D.4, the response time of the multi-agent system is approximately three or four times longer than baseline methods. Therefore, it would be helpful to reveal the cost (or the number of generated tokens) of PixelCraft in comparision with baseline method. 

+ Moreover, it is not clear if the compared baseline methods also use a critic to iteratively refine and improve the final result. If not, I would suggest equipping the baseline methods with a critic to do test-time scaling, and draw a comparison when the respond time/cost is similar.

### Questions
See the weaknesses above.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a new framework designed to improve how multimodal models reason about structured visual content such as charts, diagrams, and geometry problems. It introduces a multi-agent architecture where specialized agents, like a planner, reasoner, and visual tool users, collaborate to analyze images, exchange feedback, and iteratively refine their reasoning. A key feature is a pixel-level grounding module, which allows the system to interact precisely with image regions through actions like zooming, masking, and highlighting. The framework also incorporates an image memory and planning mechanism that enables non-linear reasoning and self-correction. Experiments show that PixelCraft achieves strong performance across visual reasoning benchmarks, outperforming previous multi-agent and single-model approaches.

### Strengths
1. Originality: This paper integrates multi-agent collaboration with fine-grained visual grounding for structured image reasoning. While the idea of multi-agent reasoning is not entirely new, combining it with pixel-level visual tools and planner-managed image memory adds a modest but meaningful improvement in coordination and interpretability.

2. Quality: The work is technically sound and experimentally thorough. The system design is coherent, and the experiments cover diverse benchmarks with reasonable comparisons and ablations.

3. Clarity: The paper is generally well organized, with clear descriptions of each agent’s role and effective figures that illustrate the workflow.

4. Significance: The contributions are incremental but relevant. The framework addresses practical challenges in visual reasoning and could be adapted to other structured domains, offering a small yet useful step forward for multimodal systems.

### Weaknesses
1. Unclear Cost and Scalability

The multi-agent architecture (planner, critics, tool agents, etc.) implies high computational and communication overhead. The paper doesn’t report inference latency, average agent calls, or cost per sample — critical factors for assessing practical deployability. Since they use a strong vision-language backbone (Qwen2.5-VL), much of the pipeline’s gain might come at the cost of efficiency. Report computation and cost metrics, average number of reasoning turns, total API calls, and latency per image — to give a fair sense of scalability.

2. Overreliance on Synthetic or Narrow Benchmarks

Most experiments use structured-image datasets (e.g., ChartQA, Geometry3K), which are quite specialized. This limits generalization to open-domain or real-world structured reasoning tasks like document or diagram understanding. It’s not shown whether PixelCraft scales to noisy or natural images where structure is less explicit. I suggest adding a small evaluation on broader multimodal datasets (e.g., DocVQA, InfographicsVQA) or including a discussion on how the system might handle unstructured visuals.

### Questions
I am confused about the Figure. 4. Why is the usage frequency of Legend Masking and Adding Auxiliary very low, but the performance gain is giant? Most of the time, these tools will not be called.

### Soundness
3

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
This paper aims at solving visual reasoning on structured images, eg. Charts and geometric diagrams. The authors propose PixelCraft, a multi-agent system comprising of three main componets: query-aware agent selection, agent discussion, and iterative self-correction with a planner managed image memory. To achieve accurate visual grounding and image editing, this work also finetunes Qwen2.5-VL-3B into a visual grounding model based on a synthesized training set, and uses it as a reliable Tool Agent. Experiments on multiple recent datasets demonstrated the superior performance of PixelCraft against competitors.

### Strengths
- Visual reasoning on structured images is important and has a wide range of applications. This paper proposes an effective multi-agent system for performing visual reasoning on structured images, achieving strong performance across multiple datasets.
- This paper identifies visual grounding as a key obstacle in performing visual reasoning tasks on structured images, and proposes an effective approach to constructing a high-precision visual grounding model based on synthetic training data and MLLM fine-tuning.
- This paper provides detailed experimental results that sufficiently demonstrate the effectiveness of each proposed component.

### Weaknesses
- It appears that in the current implementation and evaluation, PixelCraft requires pre-specified tool sets for different task types, such as chart reasoning and geometric reasoning, which limits the assessment of PixelCraft’s generalization capability.
- The paper does not report the computational cost of PixelCraft, particularly in comparison with direct answering and other test-time scaling methods such as Chain-of-Thought (CoT) and Multi-Agent Systems. Comparing both accuracy and computational cost would enable a more comprehensive evaluation of the proposed method’s advantages.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
