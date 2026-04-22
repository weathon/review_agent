# Efficient Multimodal Spatial Reasoning via Dynamic and Asymmetric Routing

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 8, 6, 4

## Abstract
Recently, visualization-of-thought (VoT) has unlocked new opportunities for complex spatial reasoning in multimodal large language models (MLLMs) by complementing verbal reasoning with visual thinking.
However, the autoregressive accumulation of lengthy and redundant tokens substantially increases computation and memory costs.
In this paper, we present a new efficient framework for multimodal spatial reasoning, named *DARE*, designed to adaptively prune multimodal tokens across different network depths, reasoning hops, and modalities. 
First, *DARE* devises an intra- and inter-hop-aware differentiable retention mechanism to dynamically estimate token importance both within each reasoning step and across successive hops. 
Recognizing that deeper network layers encode visual cues into verbal streams, *DARE* introduces an asymmetric compression strategy that prunes tokens according to modality-specific redundancy and semantic importance.
Furthermore, *DARE* incorporates a progressive KV-cache retention policy aligned with cross-modal fusion dynamics, further reducing memory overhead during autoregressive reasoning.
Our method delivers substantial reductions in computation and memory footprint, averaging a 40.37\% reduction in FLOPs and 46.07\% reduction in KV caches usage, 
while consistently preserving or even improving reasoning performance across seven multimodal spatial reasoning benchmarks, and further generalizing to broader multimodal reasoning tasks, 
establishing a scalable and robust recipe for efficient multimodal reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces DARE (Dynamic and Asymmetric Routing), a new framework designed to address the significant computational and memory costs associated with multi-hop reasoning in Multimodal Large Language Models (MLLMs). It proposes a dynamic token pruning mechanism. Through experiments on seven spatial reasoning benchmarks, the authors demonstrate that DARE achieves substantial reductions in FLOPs and KV-cache usage.

### Strengths
The paper provides a clear motivation by outlining the computational efficiency challenges in multi-hop multimodal reasoning. Furthermore, the work is presented with a coherent structure, making the methodology and experimental sections straightforward to follow.

### Weaknesses
Regarding the experiments on dynamic spatial reasoning, I am genuinely interested in understanding how the results in Table 2 were obtained. As someone familiar with unified MLLMs and interleaved multimodal reasoning, I noticed that the datasets mentioned—MAZE, MINIBEHAVIOR, and FROZENLAKE—do not appear to be publicly available, at least to the best of my knowledge. This raises some concerns about the reproducibility and authenticity of the reported results, as independent verification is currently not feasible. I believe that providing access to the source code and implementation details of this part would help clarify how the experiments were conducted. To adhere to the double-blind review policy, an anonymous link to a code repository would be very helpful.

### Questions
Please see the weaknesses above.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new framework that enhances the efficiency of MLLMs during spatial-reasoning tasks by adaptively pruning multimodal tokens across network depths, reasoning hops, and modalities. DARE integrates a layer- and hop-aware differentiable retention mechanism with an asymmetric compression strategy to selectively preserve essential tokens. Experiments on multiple multimodal spatial reasoning benchmarks show that DARE significantly reduces FLOPs and memory overhead while maintaining performance.

I have reviewed this paper for NeurIPS 2025. After careful check, I noticed that the authors have incorporated the reviewers' suggestions and addressed their concerns in this version. For example, the comparison between DARE and MoD (Appendix D.4: Comparison with Heuristic Token Retention and Routing Methods) and the experiments across models of different sizes (Scalability to Larger and Smaller Models, Section G.1). I think this version is clear and well-written, meeting the standard for acceptance.

### Strengths
1. The motivation is solid, and the method is well-motivated. The authors tackle a key efficiency bottleneck in multi-hop multimodal spatial reasoning. The approach is clearly formulated and effectively presented.

2. The main experiments are comprehensive, covering seven spatial-reasoning benchmarks, general VQA, hallucination detection, and detailed ablation studies.

3. The authors provide thorough discussion and analysis, including comparisons between DARE and MoD, interpretation of retention rates, the necessity of learning-based routing, and how DARE scales as model size increases. Overall, the paper offers a complete and clear discussion of the DARE method and its position within the broader research landscape.

### Weaknesses
No obvious weaknesses.

### Questions
After careful review, the authors have addressed all the questions I raised in my previous review, and I have no further questions.

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
This paper introduces DARE, a novel framework designed to improve the efficiency of multi-hop multimodal spatial reasoning by addressing the high computational costs of existing methods. It achieves this through several key contributions: a modality-aware token routing mechanism with dedicated routers at each transformer layer to score token importance; an intra- and inter-hop aware retention strategy that uses Gumbel-Softmax to learn dynamic pruning ratios across both network layers and reasoning steps; and a KV-cache policy that reduces memory overhead by aligning with token pruning decisions. The authors report significant efficiency gains while maintaining or improving performance.

### Strengths
1. DARE shows compelling performance, consistently achieving comparable or superior accuracy to its baselines while being significantly more efficient.
2. The framework demonstrates impressive generalization by showing effectiveness on two different interleaved reasoning architectures: one based on visual token referencing (VolCano) and another on mental image generation (Anole).
3. The paper is supported by a comprehensive set of experiments and ablation studies that strongly validate the proposed design choices. The authors' effort in providing this level of detail, including a well-organized appendix, is very impressive.

### Weaknesses
1. **Impact of New Hyperparameters on Applicability:** The proposed method introduces several new hyperparameters (e.g., target retention ratios $p_{target}$, hard pruning threshold $\epsilon$, prefix size $\kappa$). While the paper includes helpful ablation studies, finding the optimal set of these hyperparameters for a new model or task could be a non-trivial and expensive process, potentially limiting the method's plug-and-play applicability.

2. **Fixed Number of Reasoning Hops**: The framework appears to operate with a pre-defined maximum number of reasoning hops ($H$). Real-world problems may require a variable number of reasoning steps. It is unclear how DARE would adapt to tasks that require more or fewer hops than what it was trained for, which may limit its flexibility in more open-ended scenarios.

### Questions
1. How do the router's importance scores change when the model is given a different task? For instance, does a visual search task yield a different tendency compared to a dynamic navigation task (like MAZE)?

2. The term "hop" is central to the paper but is not explicitly defined. Is a "hop" defined as a single autoregressive generation step for an intermediate thought (which could be a mix of visual and textual tokens), as suggested by Figure 2? A precise definition would be helpful.

3. In Table 1, DARE-LH shows a particularly large accuracy improvement over the VolCano baseline on EmbSpatial compared to other compositional tasks like VSR. Is there an intuition for why the proposed method is especially effective for this specific benchmark?

4. Table 3 reports results for DARE-L but not DARE-LH on the General VQA and Hallucination benchmarks. Is this because these tasks are typically solved in a single reasoning step (i.e., one hop), making the inter-hop (-LH) mechanism not applicable?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces DARE, a novel and well-motivated framework for efficient multimodal spatial reasoning. By dynamically and asymmetrically pruning tokens across layers and reasoning hops, the method achieves reductions in computation and memory while often improving task performance. The paper is supported with comprehensive analysis and evaluation with comparisons over different system variants.

### Strengths
1. The paper is well-motivated. The paper excels at identifying and motivating two nuanced challenges in multi-hop multimodal reasoning: the dynamic, shifting importance of tokens across both network depth and reasoning steps, and the asymmetric redundancy patterns between visual and textual data. 
2. Interesting technical design. I really like the use of a differentiable, end-to-end learnable routing mechanism compared to fixed heuristics. Furthermore, the asymmetric compression strategy and the explicit focus on reducing KV-cache usage are critical for autoregressive models.
3. Comprehensive empirical evaluation. The authors validate DARE across an extensive and diverse set of benchmarks, comparing it against diverse baselines. The results are good, showing that DARE not only meets its efficiency goals in terms of FLOPs and KV cache reduction but also even improves accuracy, demonstrating the method's effectiveness.

### Weaknesses
1. Reproducibility and reliability of the results: I'm concerned with the experimental results in Table 2, especially with MVoT and VoT. To be more specific, I would appreciate it if the authors can clarify:
* Computational resources: can Anole 7B be trained with 40 GB GPUs for how long? Could the authors provide training logs for clarification purposes?
* Experiment results: given that there is no public available datasets and model checkpoints MVoT uses in the original paper, and the reported numbers in Table 2 strictly aligns with the results in MVoT paper, plus the experimental settings (number of epochs, types of GPUs, numbers of GPUs) are not same as in the original paper, I would appreciate the authors provide more details regarding this. (I tried to look into the supplementary materials in the zip file, but didn't find the training script)
2. Methodological complexity and hyperparameters. While effective, DARE is a complex system with numerous interacting components and hyperparameters (target retention ratios, pruning thresholds, Gumbel temperature, etc.). The paper could benefit from a clearer discussion of the tuning effort required and the generalizability of the chosen hyperparameter values to different model architectures or domains.

### Questions
See as above in weaknesses. I'm happy to adjust the scores if the author can address the concerns in the weaknesses.

Typo: L037 MoT should be MVoT

### Soundness
2

### Presentation
3

### Contribution
2
