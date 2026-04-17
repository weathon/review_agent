# Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6, 6

## Abstract
Chain-of-Thought (CoT) reasoning has proven effective in enhancing Large Language Models (LLMs) on complex tasks by decomposing problems into step-wise solutions. However, extending CoT to multi-modal settings remains challenging, as it requires modeling transitions of visual states alongside textual reasoning. Existing approaches often underperform due to limited capacity to model visual transitions or fragmented architectures. To overcome this limitation, we introduce Uni-CoT, a Unified Chain-of-Thought framework that captures structured visual transitions and seamlessly aligns them with textual logic, enabling coherent multimodal reasoning.
To mitigate the computational and training challenges inherent to multi-modal reasoning, Uni-CoT introduces a two-level reasoning paradigm: a macro-level CoT for high-level planning and a micro-level CoT for localized subtask execution. This hierarchical design reduces computational overhead while maintaining coherence. Additionally, Uni-CoT incorporates a structured training paradigm with auxiliary tasks to stabilize optimization and improve generalization. Experiments on reasoning-driven image generation and understanding benchmarks demonstrate that Uni-CoT achieves state-of-the-art performance and remarkable generalization, underscoring its potential for complex multi-modal reasoning. Code: https://github.com/Fr0zenCrane/UniCoT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes Uni-CoT, a Unified Chain-of-Thought framework that captures structured visual transitions and seamlessly aligns them with textual logic. They use a macro-level CoT for high-level planning and a micro-level CoT for localized subtask execution.  Overall, Uni-CoT performs better than the baselines.

### Strengths
1. There are still relatively few reasoning frameworks for unified models, and this study effectively fills that gap.

2. The paper is easy to follow, the figures are very clear, and the code is open-sourced.

3. The experiments are highly detailed, and the provided examples are intuitive and illustrative.

4. The selected datasets are convincing (e.g., WISE), and the experimental results are strong.

### Weaknesses
Although the paper has several notable strengths, its weaknesses are also apparent. The proposed method does not appear sufficiently novel since it builds a reasoning framework using prompting-based scaffolding, an approach that has already been extensively and systematically explored in the LLM and VLM literature. Therefore, the method itself is not particularly innovative. Nevertheless, this paper is likely among the first to adapt such approaches to unified models, which remains a meaningful contribution.

### Questions
I am not very familiar with the literature on unified models, but I would like to ask why there are missing data entries in Tables 1 and 3. Is this due to limited computational resources that made it difficult to obtain the results, or are there other reasons?

### Soundness
4

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
3

### Summary
This paper proposes Uni-CoT, a unified hierarchical Chain-of-Thought (CoT) framework for multimodal reasoning. The approach aims to bridge textual and visual reasoning by introducing two levels of CoT:

1. Macro-CoT, which handles high-level planning and decomposition of a multimodal task into subgoals.

2. Micro-CoT, which focuses on localized subtask execution, reasoning over both text and image modalities with shorter context windows.

### Strengths
Strengths

1. The idea of hierarchical reasoning decomposition (macro for planning, micro for execution) is conceptually sound and aligns with cognitive and hierarchical RL principles.

2. The insight that only the latest local state is needed for Micro-CoT, rather than the entire reasoning chain, is practical and can indeed reduce context length and compute cost.

3. The paper is well-motivated by the challenges of multimodal reasoning, and the integration of CoT-style reasoning into a unified visual–text model is timely

### Weaknesses
1. Hierarchical decomposition (macro/micro planning) has been explored in multiple contexts, including text-only CoT, hierarchical RL, and multi-agent reasoning. The paper does not sufficiently articulate what is new in Uni-CoT beyond combining these known techniques.

2. The paper omits critical information about the base model, data scale, loss functions, and training setup, and most are deferred to the appendix. This makes it difficult to assess reproducibility and fairness of comparisons.

3. In Tables 1 and 2, Uni-CoT shows only modest gains over plain CoT baselines (especially on WISE). The discussion around why and when Uni-CoT helps is minimal.

4. The ablation studies do not isolate the effects of the macro vs. micro components, nor do they show what happens when hierarchical supervision is removed.

5. Complexity reduction claim is unconvincing. The claimed reduction from quadratic to near-linear attention complexity seems theoretical. Given that the SFT data only involves 2–3 subtasks, it’s unclear that Uni-CoT’s decomposition meaningfully reduces overall computation compared to a single CoT pass.

6. Qualitative results lack rigor. For instance, in Fig. 3 (line 440), the generated image contradicts the textual description (“sealed container” appears open), suggesting incomplete alignment between reasoning and generation.

### Questions
1. What is the base multimodal model used?

2. Was SFT with plain CoT data attempted as a control baseline?

3. How much does each level (macro/micro) contribute individually?

4. Can the authors provide concrete statistics or examples showing context length reduction and training efficiency gains?

5. How are the auxiliary tasks (e.g., reward estimation) implemented? are they learned jointly or pre-trained separately?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a hierarchical framework to extend chain-of-thought (CoT) reasoning to multi-modal settings. The authors introduce Uni-CoT, which unifies text and visual reasoning within a single model by integrating structural visual transitions and coherent textual logic. To manage the high computational cost of multi-modal reasoning, they design a macro–micro hierarchical CoT, where the macro level plans subgoals and the micro level executes them via Markov Decision Process–based self-reflection. Experiments on reasoning-driven image generation and understanding benchmarks show state-of-the-art results and improved interpretability, demonstrating Uni-CoT’s potential for scalable and coherent multimodal reasoning .

### Strengths
1. Originality: This paper proposes a unified framework for multi-modal chain-of-thought (CoT) reasoning, bridging visual and textual reasoning within a single coherent process. The Markov Decision Process formulation in this paper reduces the computation overhead.

2. Quality: The proposed method is well-motivated, combining goal decomposition with self-reflective decision-making. The experiments are comprehensive, covering both understanding and generation tasks, and the results consistently support the framework’s effectiveness. 

3. Clarity: The paper is generally well-organized and easy to understand.

4. Significance: The work contributes meaningfully to the emerging area of unified multimodal reasoning. It also formally analyzes the complexity of efficient MDP attention.

### Weaknesses
1. The main novelty is kind of limited; there are some similar works using CoT for image generation refinement, like LayerCraft [1]. Efficient attention is also a well-studied area. However, combining them might be the first try.
2. The training dataset for CoT image generation purely depends on synthetic data. This limits generalization to open-domain or real-world visual reasoning tasks.
3. Figure S3 in the appendix seems not accurately reflect a realistic ground-level scene, which raises questions about the accuracy of synthetic training data.

[1] LayerCraft: Enhancing Text-to-Image Generation with CoT Reasoning and Layered Object Integration, Yuyao Zhang, Jinghao Li, Yu-Wing Tai, in NIPS 2025

### Questions
N/A

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
5

### Summary
This paper introduces Uni-CoT, a unified framework for multi-modal Chain-of-Thought (CoT) reasoning that integrates both textual and visual reasoning steps.  The proposed method employs a hierarchical macro-micro reasoning structure: the macro-level handles task decomposition and summarization, while the micro-level executes subtasks via a Markov Decision Process (MDP) with self-reflection.  The authors address the computational challenges of multi-modal CoT by reducing the complexity from quadratic to near-linear through localized reasoning and attention masking.  Experiments on image generation and understanding benchmarks (e.g., GenEval, WISE, MME, Jigsaw-R1) demonstrate state-of-the-art or competitive performance.

### Strengths
- The technical contribution under the unified model such as BAGEL is effective. 
- The proposed method significantly reduces computational complexity, making long-horizon multi-modal reasoning more tractable.
- The paper includes both quantitative and qualitative results across multiple challenging benchmarks, including reasoning-driven generation and understanding tasks.
- Ablation on CoT mechanisms and training strategies provides strong evidence for the contribution of each component.
- The authors provide code, training details, and dataset construction pipelines, enhancing reproducibility.

### Weaknesses
- The evaluation results on understanding tasks are not sufficient. The authors only provide results on the MME benchmark, which is not representative of supporting the claim of unified tasks. Lack of more general and widely-used benchmarks such as MMMU, MMBench, OCRBench, MathVista, MathVision, etc. 
- Experiments are conducted only on the BAGEL backbone.  It is unclear how the framework would perform with other MLLMs or in a more model-agnostic setting.

### Questions
Please refer to the weaknesses part. More results on understanding tasks will support the conclusions of this paper.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a unified multimodal Chain-of-Thought framework combining text and vision reasoning in one model. By introducing a macro–micro hierarchical structure and self-reflective MDP reasoning, it achieves coherent and efficient multimodal reasoning with reduced complexity. Results on multiple benchmarks show strong performance, though challenges remain in fine-grained visual consistency and generalization to dynamic scenes.

### Strengths
1. This paper proposes a unified framework that integrates textual and visual Chain-of-Thought reasoning in one model. It introduces a hierarchical macro–micro design that improves reasoning structure, coherence, and interpretability.
2. The method employs an MDP-based self-reflective process that reduces complexity from quadratic to near-linear while enhancing robustness.
3. The method achieves strong results on multiple benchmarks for both image generation and understanding.

### Weaknesses
1. Despite reduced complexity, multimodal reasoning remains computationally expensive due to the large number of visual tokens per step.
2. The dataset used for training is relatively small and partially synthetic, which may limit generalization to real-world multimodal scenarios.
3. The paper could be strengthened by providing a quantitative comparison of token usage during inference against baseline models.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
3
