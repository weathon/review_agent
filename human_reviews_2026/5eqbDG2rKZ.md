# ALPS: Adaptive LLM Pruning via Gradient Search in Learned Representation Space

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Deploying Large Language Models (LLMs) at the edge is crucial for data privacy and offline operation, yet their massive parameter count poses significant resource challenges. While existing methods rely on discrete-space heuristics to search for pruning configurations, we introduce a fundamentally different approach: reformulating the search for optimal LLM pruning configurations as gradient optimization in a learned continuous representation space. Our method, ALPS (Adaptive Layer Pruning via Search), embeds discrete pruning configurations into a continuous space where efficient gradient-based optimization becomes possible, then decodes optimal representations back to implementable discrete pruning schemes. This encoder-evaluator-decoder architecture automatically learns from collected “pruning-score" data pairs, eliminating manual tuning while jointly optimizing for model performance, latency, and energy consumption in a deployment-specific manner. Extensive experiments across Llama-7B, Llama2-7B, Llama2-13B, and Vicuna-7B demonstrate ALPS's superiority, achieving up to 34.1% energy reduction and 33.5\% lower latency while maintaining over 91% of original performance. At high pruning ratios (50%), ALPS consistently outperforms state-of-the-art methods in both perplexity and downstream task accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a novel pruning configuration search method that maps discrete heuristics into a continuous space. This approach directly optimizes the pruning configuration with respect to both model performance and latency. Experimental results demonstrate that the method achieves superior performance in terms of latency, perplexity, and downstream task accuracy.

### Strengths
I like the idea presented in this paper. While the concept of relaxing discrete heuristics into a continuous space is well known in the machine learning community, this work provides a well-designed and likely first practical adaptation of that idea for pruning configuration optimization. In addition, the paper introduces a thoughtfully constructed data generation pipeline for training the configuration model.

The paper is clearly written and presents comprehensive experiments that evaluate performance from multiple perspectives, including next-token prediction accuracy, downstream task performance, and latency.

### Weaknesses
The motivation and method descriptions in this paper are well-written, but I have several concerns regarding the experimental section:
- The experimental setup (e.g., models, tasks) feels somewhat outdated. I would expect evaluations on at least LLaMA3.4 or the latest Qwen model families. This is important because the behavior of pruning methods on models like LLaMA-2 can differ significantly from SOTA models due to their much richer pretraining corpora (beyond datasets like WikiText).
- Similarly, the selection of baseline methods is mostly limited to those before 2024. While I haven’t kept up with every recent pruning paper, there are at least methods like SVD-LLM and ModeGPT that achieve much better performance and should be considered in the comparisons.
- I also feel the authors should include more comparisons with global sparsity allocation methods. Since the proposed method primarily focuses on optimizing pruning configurations, it would be useful to relate it to approaches addressing similar global sparsity allocation challenges—such as those discussed in Appendix B.10 of ModeGPT.
- Lastly, I’m curious about the pruning time required for the proposed method. I expect it may take more time, which is acceptable for an inference-speedup target. However, providing this comparison would help readers better understand the trade-offs and decide when to adopt this method in practice.

### Questions
- In line 49, I'm considering if it's good to say other solutions cannot adapt to specific hardware constraints, we can do this by adjusting the pruning ratio, right?
- How to choose the dimension d during the encoding?
- If more training data were available for learning the pruning configurations, do you expect further improvements in performance? Have you tested the sensitivity of the method to data size?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces Alps, an adaptive pruning framework for large language models (LLMs) that reframes the discrete search for pruning configurations as a continuous optimization problem in a learned latent space. An encoder–evaluator–decoder pipeline is trained using “pruning configuration–performance score” pairs obtained from heuristic methods. Gradient ascent within this learned space identifies promising pruning representations, which are decoded into executable pruning configurations. The authors report reductions in inference latency and energy consumption while maintaining accuracy across several LLM benchmarks.

While the idea of continuous relaxation for pruning configuration search is intriguing, the overall contribution is limited by heavy dependence on heuristic data, lack of theoretical justification for the latent-space mapping, and insufficient evaluation on real hardware platforms or against strong baselines.

### Strengths
+ Conceptual Novelty: Reformulating discrete pruning as differentiable optimization is an appealing and underexplored direction. The proposed encoder–decoder pipeline represents a step toward automating pruning configuration search.

+ Multi-Objective Framing: The method jointly considers model accuracy, latency, and energy efficiency, aligning with real deployment goals beyond parameter count.

+ Breadth of Evaluation: Experiments span multiple LLMs and pruning ratios, with consistent improvements over naïve baselines.

### Weaknesses
+ Dependence on Heuristic Data (Critical): Although Alps claims to overcome heuristic pruning, its foundation still rests on heuristic-derived “configuration–score” pairs. The framework effectively learns a regression over existing heuristic outcomes rather than discovering new principles. The performance and generality of Alps therefore hinge entirely on the diversity and quality of this pre-collected data, which the paper neither quantifies nor ablates.

+ Lack of Theoretical Clarity: The latent continuous representation is presented as a black box. The paper provides no analysis of the embedding’s geometry, smoothness, or correlation with true pruning effectiveness. Without such evidence, it remains unclear whether the gradient-based optimization is meaningful or merely interpolating between existing heuristic points.

+ Insufficient Efficiency and Cost Analysis: The authors emphasize latency and energy savings but omit end-to-end computational cost — both for data collection and model training. The claimed “efficiency” gains are potentially offset by this large upfront overhead. Furthermore, all results are reported on a single GPU (A40), which cannot substantiate claims about edge deployment or hardware adaptability.

+ Missing Baseline Comparisons: Alps is compared mainly against heuristic methods, but not against recent learned or reinforcement-based pruning frameworks (e.g., AutoCompress, AdaPruner). Without such comparisons, it is difficult to assess the true competitiveness or scalability of the proposed approach.

+ Interpretability and Reproducibility Concerns: The continuous space and optimization trajectory are opaque. The lack of interpretability makes the system difficult to trust or debug, especially since small representation shifts could yield vastly different pruning masks. Code and data availability are also not explicitly stated, raising questions about reproducibility.

### Questions
In Lines 44–47, you argue that latency and energy are more critical than model size. However, memory constraints often dominate on real edge devices. Could you clarify this trade-off, ideally with supporting empirical evidence?

The efficiency results are limited to an NVIDIA A40 GPU. How does Alps generalize across architectures with different compute–memory trade-offs (e.g., mobile NPUs, Jetson, or CPU-based inference)?

What is the total computational cost (in GPU-hours) of training the encoder–evaluator–decoder model, and how does it compare to conventional pruning search methods in both wall-clock time and energy usage?

Have you performed any ablation to evaluate the sensitivity of Alps to the quality or quantity of the heuristic “configuration–score” data?

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
The paper proposes a framework for pruning LLMS, which reformulates the pruning configuration optimization as a continuous optimization problem and solves the problem by gradient-based optimization method.

### Strengths
1.	The formulation is novel, which relaxes the LLM pruning problem by using discrete pruning ratio of each layer.
2.	Instead of only focusing on utility, the paper also includes efficiency and latency in the optimization problem.

### Weaknesses
1.	The validity of formulation remains questionable. From my point of view, representing a layer by a single pruning ratio is over-simplified. More evidence is needed to prove the simplification works.
2.	The paper claims high efficiency of the proposed method, but the process of collection pruning–score pairs consumes far more energy than other methods.
3.	The continuity and reasonability of the representation space is confused.  The representation space is learned purely from discrete samples of traditional heuristics. Based on that, the gradient optimizer operates on the representation space, with no guarantee that it faithfully reflects the true pruning landscape.

### Questions
1. Could you please provide more evidence about the alignment between predicted and the true performance?

2. Could you further explain the choice of using LSTM? From my point of view, the pruning configuration sequence is not temporal; a simple MLP or transformer encoder may also work.

3. As one of the main concern is the efficiency, and the number of heuristic points is highly related to the total cost. Could you please provide how sensitive is ALPS to the number of heuristic configurations used for training? 

4. I am quite curious about the stability of Gradient optimization. Does the gradient search in the learned space diverge or produce invalid configurations?

### Soundness
2

### Presentation
2

### Contribution
2
