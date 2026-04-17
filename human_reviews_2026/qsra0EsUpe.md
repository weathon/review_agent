# Test-Time Optimization of 3D Point Cloud LLM via Manifold-Aware In-Context Guidance and Refinement

- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in textual and 2D visual reasoning, yet their ability to understand and reason over 3D data remains limited. The issues become more challenging for understanding standalone 3D point cloud due to the high interclass confusion. In this work, we propose Point-Graph LLM (PGLLM), a framework that enables more effective 3D point cloud understanding by integrating in-context prompting and score refinement at test-time, respecting supporting data manifold. Our method first employs a pre-trained point cloud encoder which are used to construct a graph where edges encode visual similarity. Each support point cloud sample is converted to a textual caption via pre-trained PointLLM. For a test query, the graph is used to retrieve relevant neighbors whose captions serve as contextual demonstrations for a second stage LLM for final reasoning, a process we term in-context guidance. Furthermore, we introduce a confidence score refinement mechanism based on label propagation to enhance the reliability of LLM predictions for classification and out-of-distribution (OOD) detection tasks. All the above optimizations are carried out fully at test-time. Extensive experiments across diverse 3D datasets and tasks demonstrate that PGLLM consistently improves accuracy and robustness over prior baselines with very almost no additional computation cost, showcasing a promising direction toward native 3D reasoning with MLLMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes Point-Graph LLM (PGLLM), a framework designed to enhance the 3D understanding capability of large language models at test time.  

Its core ideas are as follows:

1. Manifold-aware in-context guidance: A KNN graph is constructed, where edge weights encode feature similarity among 3D point clouds. For each query sample, the framework retrieves its nearest neighbors and their corresponding PointLLM-generated textual descriptions. These neighbor captions are incorporated as in-context exemplars within the LLM prompt, enabling more semantically consistent reasoning along the data manifold.  

2. Label-propagation-based score refinement: A graph-based confidence propagation mechanism is applied to smooth and refine prediction scores over the manifold, thereby improving the stability and reliability of both classification and OOD detection.  

Experiments are conducted on multiple 3D benchmarks, including ModelNet40, ShapeNetCore, S3DIS, and Objaverse, covering three major downstream tasks: 3D classification, 3D OOD detection, and 3D object captioning.  
The results demonstrate that PGLLM achieves SOTA performance on 3D recognition and OOD detection while maintaining competitive results on captioning, with negligible additional computational overhead.

### Strengths
1. A simple yet effective test-time optimization paradigm: The framework leverages KNN-based neighbor descriptions as ICL exemplars, combined with label-propagation-based score refinement. The design is conceptually simple but aligns well with the intuition of manifold consistency. Moreover, it is fully plug-and-play for existing PointLLM-based approaches, requiring no retraining.  

2. Originality: The central idea of this work—integrating manifold learning with ICL at test time for 3D point cloud understanding—shows a high degree of originality. Rather than merely employing graph-based features, the method creatively uses the textual descriptions of neighboring samples as contextual exemplars for the LLM. This combination of a 3D encoder (for graph construction), a 3D multimodal LLM (for description generation), and a second-stage LLM (for reasoning) represents a novel and well-engineered design. The subsequent score refinement through label propagation on the LLM-generated confidence scores is also an elegant and complementary addition.  

3. Clarity: The paper is exceptionally well-written and easy to follow. It clearly identifies a key limitation of previous works (interpreting each point cloud in isolation) and presents a coherent, step-by-step solution. The problem definition, methodology, and experimental setup are all precisely described, making the contribution easy to understand and evaluate.

### Weaknesses
1. Limited performance on captioning: Although the method outperforms baselines on the 3D object captioning task, it does not achieve SOTA results. The paper attributes this to the limited size of the test dataset, but this explanation is weak and unconvincing. Since the proposed T variant uses the entire test set to construct the graph, there is a potential risk of data leakage compared to inductive baselines. Moreover, the O variant still performs worse than the T variant, suggesting that the “in-context refinement” mechanism (Section 3.3) for generation may be less effective than the score-based refinement mechanism used for classification.  

2. Scalability and transductive setting: The method appears to operate under a transductive assumption, where it must access the entire test set (or a large support set 𝒟ᵤ) to build the graph before inference. This assumption (in PGLLMᵀ) may not hold in practical inductive scenarios where samples arrive sequentially. Although the paper introduces an alternative variant (PGLLMᴼ) using an external dataset, this variant relies on a large 100K-sample subset from Objaverse. While the paper briefly mentions a “dynamic graph expansion” scheme as a potential remedy, it neither evaluates its performance nor analyzes the computational cost. The scalability of constructing the initial graph (an Nᵤ × Nᵤ similarity matrix) also becomes a potential concern for large-scale datasets. Furthermore, performance degradation is evident when using the external dataset.  

3. Dependency on pre-trained models: The framework’s success heavily depends on the quality of two pretrained components: the 3D encoder (Point-BERT) and the description generator (PointLLM). The paper acknowledges that poor or inaccurate captions may mislead the LLM (as illustrated in Figure 11), but it does not analyze this sensitivity in depth. If the initial captions are noisy or semantically incorrect—essentially a “garbage in, garbage out” problem—the effectiveness of the in-context guidance may significantly degrade.

### Questions
1. The choice of K appears inconsistent. The main experiments (Table 1) use K=3, while the ablation study (Figure 4) shows that the best OOD detection performance occurs at K=7 for ModelNet40 and K=4 for ShapeNetCore. Why was K=3 chosen for the main results? Would the reported SOTA results in Table 1 improve further if the empirically optimal K values from Figure 4 were applied?  

2. Sensitivity to caption quality: How robust is PGLLM to the quality of the initial captions generated by PointLLM? Have you considered an ablation study in which the in-context exemplars are derived from non-LLM sources (e.g., ground-truth labels or simple template-based descriptions, if available) to isolate the influence of caption quality on overall performance?

### Soundness
3

### Presentation
3

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
The paper proposes PGLLM, a test-time framework for 3D point-cloud understanding that (i) builds a KNN graph over support embeddings from a frozen 3D encoder, (ii) uses neighbor captions as in-context demonstrations for a second-stage LLM (“in-context guidance”), and (iii) refines recognition/OOD scores via label propagation. Experiments on ModelNet40, ShapeNetCore, S3DIS, and Objaverse show gains in OOD and recognition, plus a small captioning improvement.

### Strengths
-- Clear, modular test-time pipeline; no retraining of LLMs required.

-- Solid ablations: with/without in-context guidance and score propagation; K-sensitivity; task coverage (recognition, OOD, captioning).

-- Time breakdown table suggests negligible overhead for graph ops and propagation relative to caption/LLM inference.

### Weaknesses
FLOPs/Compute attribution is absent. The paper reports per-sample latency but not FLOPs/param attribution for each stage (encoder feature extraction, graph build, LLM inference, propagation). Without FLOPs, it’s hard to compare to alternatives like direct k-NN retrieval or pure prompt-engineering baselines at matched compute. Please add per-module FLOPs and parameter counts (and, ideally, energy or GPU utilization) to substantiate “very little extra cost.”

Graph storage & memory footprint not quantified. The method keeps a KNN graph over support embeddings and captions; storage and RAM/VRAM requirements are not analyzed. Provide:

Captioning degradation is unresolved. The proposed test-time guidance reduces caption quality on several splits. As written, there is no justification for using it in captioning.

### Questions
Captioning degradation is unresolved. The proposed test-time guidance reduces caption quality on several splits. As written, there is no justification for using it in captioning. Either (i) present risk-controlled variants that consistently improve captioning and report results, or (ii) restrict the method’s scope to recognition/OOD and state captioning as out-of-scope.

FLOPs & memory accounting. Report FLOPs, params, and memory for: 3D encoder pass, graph construction, retrieval, label propagation, and second-stage LLM.

Graph storage budgets. For the Objaverse-O setting (100k support): list bytes for embeddings + adjacency + captions; show peak host and device memory.

Ablate components you still use. The combined method (in-context + propagation) helps (Tab. 3), but show cases where only in-context hurts vs baseline (and why). Provide qualitative examples where label propagation corrects vs amplifies errors.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This work leverages a KNN-based graph and a confidence score refinement mechanism to build a Point Graph using a pre-trained PointLLM together with a second-stage large model (e.g., GPT-4 or Qwen). The method operates entirely at test time and effectively improves performance on OOD detection, classification, and captioning tasks.

### Strengths
1. The framework emphasizes **test-time scaling** and integrates PointLLM inference with a graph-based refinement strategy.
2. The overall figure and visualizations are clear and well-organized.

### Weaknesses
1. The best performance relies on GPT-4, which is closed-source and incurs API and monetary costs, while the Qwen version shows relatively weaker results. How can one balance performance and cost in practical deployment?
2. It would be valuable to evaluate the framework on more 3D-LLMs, such as ShapeLLM, to demonstrate broader applicability and generality.
3. From an efficiency perspective, I would like to see the impact of test-time scaling on inference latency, GPU memory usage, and other computational metrics.

### Questions
**1. Dependency on GPT-4 and Cost–Performance Trade-off**

The method achieves its strongest results when paired with GPT-4. However, GPT-4 is a proprietary model and requires paid API access, which may limit the practicality and scalability of this approach in real-world deployment scenarios or resource-constrained environments. In contrast, the performance using an open-source model like Qwen appears significantly weaker.

To strengthen the paper, I recommend a deeper analysis of the **performance–cost trade-off**, including:

* A quantitative comparison of accuracy vs. computational/financial cost between GPT-4 and Qwen.
* Discussion of whether intermediate open-source models (e.g., Qwen-Plus, Llama-3 variants) can offer a more balanced trade-off.
* Insights into how organizations without commercial model access could adopt this framework efficiently.

Such analysis would provide more practical guidance for deployment and broaden the method's applicability.

---

**2. Evaluation on Broader 3D-LLMs for Generality**

The current evaluation focuses primarily on PointLLM combined with GPT-style LLMs. While this is valuable, it remains unclear whether the framework generalizes across different 3D foundation models. To convincingly demonstrate method robustness and universality, I recommend including results on additional state-of-the-art 3D-LLMs such as **ShapeLLM, ShapeLLM-Omni, Uni3D-LLM**, or other emerging architectures.

This evaluation would help clarify:

* Whether improvements stem from the proposed Point-Graph mechanism rather than characteristics of a specific backbone.
* The compatibility of this framework with diverse 3D model designs and training paradigms.
* Potential limitations or adaptations needed for different 3D-LLM families.

Such ablation and cross-model experiments would significantly enhance the paper’s credibility and contribution.

---

**3. Test-Time Scaling Efficiency and Resource Overhead**

The work emphasizes test-time scaling and test-time refinement, yet the computational implications of these procedures are not fully discussed. For practical deployment and fair comparison with prior work, it is essential to provide a comprehensive efficiency analysis, including:

* **Inference latency** before and after applying test-time scaling
* **GPU memory consumption** for graph construction and refinement
* **Runtime overhead per query** as the support set size grows
* **Scalability analysis** with respect to dataset size and number of neighbor samples
* Discussion of whether there are diminishing returns under limited compute budgets

Providing these metrics will clarify the computational footprint and demonstrate that the reported performance gains are achieved at a reasonable cost, which is particularly important for real-time or large-scale applications.

### Soundness
3

### Presentation
2

### Contribution
2
