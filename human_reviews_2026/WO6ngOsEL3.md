# Test-Time Scaling via Metric Geometry for LLM Reasoning

- Decision: Reject
- Scores: 6, 6, 2, 6

## Abstract
Test-Time Scaling (TTS) methods improve the reasoning capability of large language models (LLMs) by generating multiple independent Chain-of-Thoughts (CoTs) and aggregating them via designed policies. 
Despite effective, this ensemble approach incurs expensive inference costs due to the repeated model calls. 
In this paper, we propose a physics-inspired framework that achieves the accuracy gains of multi-calls TTS within a single or few LLM calls. 
It conceptualizes LLM reasoning as navigating through a maze, a complex puzzle through which one has to find a path to achieve specific goals.
The proposed $Maze$ paradigm embeds candidate exemplars and domain knowledge into a multiplex latent manifold and learns a high-dimensional metric space.
In inference, $Maze$ metrics can identify a single or a few optimal paths;
each path refers to an ordered sequence of exemplars, forming a few-shot prompt that guides the LLM to the correct answer. 
Empirically, in reasoning benchmarks including GPQA, MMLU-pro, GSM8K, MATH-500, and AIME, $Maze$ matches or exceeds the accuracy of the Best-of-$N$ strategies while reducing the computational cost by 60$\sim$80\%. 
These results support $Maze$ to be a principled geometric alternative to brute-force TTS, enabling low-latency, interpretable, and computation-efficient reasoning for complex tasks.
We also advocate for an interesting width-depth equivalence in LLM reasoning under the $Maze$ framework: any solution achievable by many shallow trials can be attained by a suitably planned sequence of reasoning steps.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes **Maze**, a geometry-driven test-time scaling framework for LLM reasoning. It replaces multi-sample Best-of-N inference with a single or few LLM calls by selecting an ordered few-shot prompt path based on pre-learned layer-wise energy metrics. Maze conceptualizes reasoning as traversal over a multi-layer latent manifold of exemplars, aiming to achieve comparable accuracy to multi-call ensembles with much lower computational cost.

### Strengths
- Conceptually interesting: reframes test-time scaling as *geometric path planning*.  
- Provides a principled attempt to link LLM hidden-space geometry with reasoning efficiency.  
- Experiments are comprehensive, exploring the effects of model scale, exemplar count, and path-pool size.  
- Theoretically motivated with width–depth trade-off analysis.

### Weaknesses
- The method’s practical improvement is modest, especially on strong LLMs.  
- The definition of "layers" and exemplar construction is abstract and somewhat unclear.  
- Scaling Maze to larger models might significantly increase computational overhead.  
- The “width–depth” analogy, though elegant, may not fully hold in practice since Maze operates across distinct tasks rather than multiple reasoning trajectories on the same problem.

### Questions
- Are exemplars restricted to the same domain or problem type?  
- How does the learned metric generalize to unseen tasks or exemplars?  
- What is the theoretical or empirical behavior when scaling to >10B models?

### Soundness
3

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
This paper introduces Maze, a novel test-time scaling (TTS) framework that reframes LLM reasoning as a pathfinding problem over a learned latent manifold. Instead of sampling multiple reasoning chains (e.g., self-consistency or Best-of-N), Maze pre-computes a metric space over exemplars and selects a single (or few) high-energy path(s) to form a few-shot prompt. The approach is inspired by ideas from metric geometry and physics, and it aims to achieve the performance of multi-call TTS methods like Best-of-N with significantly fewer LLM calls (often just one). The authors provide theoretical justification (width-depth equivalence) and empirical validation across several reasoning benchmarks (MMLU-Pro, GPQA, GSM8K, MATH-500), showing competitive or superior performance with 60–80% reduction in inference cost.

### Strengths
### 1. Novel and Principled Framework
The idea of modeling LLM reasoning as navigation through a learned geometric space is original and intellectually compelling. The Maze framework introduces a new paradigm for test-time scaling that goes beyond simple ensemble voting or prompt tuning.
### 2. Efficiency and Interpretability
By externalizing the scoring process and avoiding LLM-in-the-loop ranking, Maze significantly reduces inference latency and cost. The geometric interpretation also offers a level of interpretability not typically found in black-box TTS methods.

### Weaknesses
### 1. Limited Theoretical Justification for Generalization
While Theorem 1 provides a theoretical foundation under strong assumptions (low-rank, Lipschitz, margin), it is unclear how well these assumptions hold in practice across diverse tasks and models. The width-depth equivalence hypothesis is intriguing but remains largely heuristic without stronger formal grounding.

### 2. Training Overhead and Reproducibility Concerns
While inference is efficient, the offline training process involves generating thousands of contrastive pairs and tuning several learnable mappings. The paper lacks detailed ablation on training cost, sensitivity to hyperparameters (e.g., metric dimension, MLP size), and reproducibility details (e.g., training time, hardware, random seed stability).

### Questions
see weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a test-time scaling method for LLMs leveraging the signals from hidden representations of LLMs' internal layers. The proposed method is physics-inspired and learnable.

### Strengths
* This paper proposes a physics-inspired test-time scaling method for LLMs based on LLMs' internal representation.

### Weaknesses
* The primary weakness is the proposed method's strong and impractical assumption on the accessibility of LLM weights, which severely diverges from the typical black-box setting in LLM TTS literature, limiting its generalizability.
* Second, the paper fails to adequately contextualize the work by ignoring highly relevant previous methods that require weaker assumptions and provides no critical comparison to them, leaving the method's advantage unsubstantiated. Notably, many previous work have discussed the width-depth trade-off in this context [1-3].
* Finally, the experimental rigor is sub-standard as it does not meet current field expectations, specifically by lacking a comprehensive performance-resource (number of output tokens) curve necessary to evaluate the system's efficiency.

[1] Welleck, Sean, et al. "From decoding to meta-generation: Inference-time algorithms for large language models." arXiv preprint arXiv:2406.16838 (2024).

[2] Snell, Charlie Victor, et al. "Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning." The Thirteenth International Conference on Learning Representations. 2025.

[3] Wang, Junlin, et al. "Think deep, think fast: Investigating efficiency of verifier-free inference-time-scaling methods." arXiv preprint arXiv:2504.14047 (2025).

### Questions
See above.

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Maze, a physics inspired reasoning paradigm that conceptualizes LLM reasoning as finding a path in the latent manifold representing exemplars and domain knowledge. During inference, Maze identifies the optimal paths referring to a sequence exemplars that are mostly related in terms of certain metrics, guiding the LLM to correct answers. Experimentally, the method achieves similar performance as Best-of-N while significantly reduces budgets, showing that this geometry inspired paradigm is efficient for test-time scaling.

### Strengths
The physics-inspired method is interesting and principled, leveraging geometry to guide the LLM reasoning. The learned manifolds and metrics are LLM-free, making it broadly applicable. The training and inference computation is within an acceptable range.

### Weaknesses
Major:

* Can you try on more state-of-the-art models, such as GPT-5, Gemini-2.5-Pro, Claude-4.0, and other members in Qwen or Deepseek families?
* Can you also include the AIME dataset, which is a standard benchmark in recent LLM reasoning literature?
* Did you try out larger or smaller hidden dimension and metric dimension? Did you try out other mass mappings other than linear projection, and other fusion mappings apart from the simple MLP? How do these settings and other hyperparameters affect the performance?

Minor: 

* The writing could be improved. There should be more precise and concise text descriptions of the proposed method, and lacks algorithmic description.
* The theory part seems a little unclear to me. Theorem 1 needs strong assumptions, while other results seem to be kind of irrelevant to the main conclusions.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
3
