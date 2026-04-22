# BLOB-Q: Boosting Low Bit ViT Quantization via Global Optimization on Model Distortion

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
In this paper, we present a novel Mixed-Precision Post Training Quantization (PTQ) approach for Vision Transformers (ViTs). Our approach aims to minimize the output distortion caused by quantization, and thus can maximally maintain the accuracy of ViT models even quantized to low bit widths. Different with prior works which typically optimize the output error of current layer (layer distortion), when performing quantization, our approach directly minimizes the output error of the last layer of the model (model distortion). As model distortion is highly related to accuracy, our approach can maximally maintain the accuracy even when quantized to low bit widths. We formulate the quantization of ViTs as a model distortion optimization problem, given the constraint of size. By solving the optimization problem, the optimal bit allocation across layers, i.e., the optimal bit width of each layer, can be obtained, with minimized model distortion. Directly solving the optimization problem is an NP-hard problem. We propose to adopt the second-order term of the Taylor series expansion to approximate model distortion, where an important additivity property can be derived under the approximation. Utilizing the second-order additivity property, the optimization problem can be decomposed into sub-problems and solved efficiently in an iterative manner. Specifically, we propose a dynamic programming algorithm to solve the optimization problem and efficiently find the globally optimal solution with only linear time complexity. Extensive experiments on six ViT models demonstrate the effectiveness of our approach. Results show that our approach significantly improves state-of-the-art and can further reduce the size of ViT models to 4 bits to 6 bits without hurting accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a Mixed-Precision Post-Training Quantization (PTQ) framework for Vision Transformers (ViTs). Unlike previous approaches that minimize layer-wise reconstruction error, the method directly minimizes the final-layer output error (i.e., model-level distortion). The authors formulate the problem as a global optimization task and introduce a dynamic programming algorithm that finds the globally optimal bit allocation in linear time. Experimental results demonstrate consistent improvements over baseline methods.

### Strengths
The formulation based on second-order approximation and the additivity property provides a sound theoretical foundation for global distortion minimization.
The proposed dynamic programming solver is computationally efficient, achieving linear scaling and outperforming prior heuristic search methods.
Theoretical analysis is thorough and well-supported by derivations in both the main text and the supplementary material.
The overall framework is efficient and well-motivated.

### Weaknesses
The comparative analysis is limited. Most reported baselines use uniform-bit quantization, making the improvement less surprising. The few mixed-precision baselines included are outdated and insufficient to establish competitiveness against recent methods.

The fine-grained bit allocation across both weights and activations (as shown in Figures 7–8) may introduce nontrivial implementation overhead and complicate hardware deployment compared to uniform-bit schemes.

The assumption of inter-layer independence—and particularly the independence between activation and weight errors—is questionable. Figure 3 implies negligible inter-layer dependency, which seems unrealistic, especially within transformer blocks where correlations between layers of same block are expected. The figure also omits quantitative measures of dependency magnitude. It remains unclear whether the proposed method would remain effective without this assumption.

### Questions
Please see Weakness

### Soundness
2

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
The paper proposes output-based model distortion as a metric of layerwise quantization impact on the output and additivity where the output distortion is a sum of layerwise quantization impacts. 
Based on this, the paper presents a dynamic programming based solution to determine per-layer precision while meeting the given bit constraint.

### Strengths
Compared with most of quantization methods which focus on layerwise distortion minimization or training loss minimization, the presented idea of output distortion minimization looks effective according to experimental results.
Additivity property, which I think is not new but very well studied in detail in the paper, is exploited in dynamic programming to explore the entire space of bit allocation.
Ablation studies (in the main paper and appendices) are quite extensive.

### Weaknesses
Appendix D.6 reports a preliminary experiment on language model.
It would be nice if more detailed and quantitative analysis were provided.

### Questions
The proposed method looks quite effective on ViTs.
How would it be applied to language models?
In other words, how can we address the basic assumptions of the proposed method, zero mean, inter-layer independence, linearity (to ignore higher order terms), ...?

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
3

### Summary
This paper introduces a Mixed-Precision Post-Training Quantization (PTQ) framework for Vision Transformers that minimizes final-layer output error rather than layer-wise reconstruction error. The authors formulate bit allocation as a global optimization problem and develop a dynamic programming algorithm that finds the optimal solution in linear time.

### Strengths
- The paper is well-written
- Theoretical justification.
- The overall framework is efficient and well-motivated.

### Weaknesses
- Figure 3 in the appendix only indicates that the dependency of quantization error across layers is smaller than the squared quantization error of each layer. However, it is not clear whether the product of quantization errors between any two layers is significantly smaller or approximates zero. The fact that layers are quantized independently does not intuitively imply that the product of errors equals zero.

### Questions
- Tables 14 and 15 demonstrate that the mixed-precision approach is faster and consumes less memory than the uniform-bit approach. How is this possible? What is the reason behind this?

- Is it possible to replace the bit-width allocation of stronger mixed-precision methods (e.g., EMQ/OMPQ/...) to evaluate how much stronger the proposed mixed-precision approach is?

### Soundness
2

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
4

### Summary
This paper proposes a Mixed-Precision Post Training Quantization (PTQ) method for Vision Transformers (ViTs) that minimizes model distortion (the final layer's output error) instead of layer-wise distortion to better preserve accuracy. By approximating model distortion using a second-order Taylor expansion and exploiting its additivity, a dynamic programming algorithm efficiently finds the optimal bit allocation, achieving 4–6 bit quantization without accuracy loss across multiple ViT models.

### Strengths
1. The writing and theoretical derivation of this work are reasonable and easy to follow.
2. The problem formulation and solution both take computational efficiency into account.
3. Experiments demonstrate the effectiveness of the proposed method.

### Weaknesses
1. The proposed method is not entirely new, similar ideas have been explored in prior works. For example, model output reconstruction instead of layer-wise reconstruction has already been proposed; the second-order optimization in Section 3.1 and the approximation method in Section 4 have also been widely used. If I have misunderstood, the authors are welcome to clarify.
2. The compared methods are somewhat outdated, limited to baselines from 2023 or earlier. Moreover, the comparison might be unfair, as most existing methods perform uniform-precision quantization.
3. Some mathematical derivations could be improved. For instance, in Property 1 (Equation 4), a more rigorous expression should be 
E <= E_W + E_A. This can be easily proven and makes more sense than the current “Second-Order Additivity” assumption. It would not affect the subsequent derivation or conclusions, as it can be regarded as optimizing the upper bound of the error.

### Questions
0. Please first address the questions in the “weaknesses.”
1. It would be helpful to further summarize the technical contributions. At present, most components appear to have been proposed before, so it would be good to highlight what is newly introduced or improved.
2. When solving the mixed-precision quantization problem, could the authors elaborate on the advantages of using dynamic programming? More commonly used alternatives today maybe are Pareto or linear programming methods.
3. The concept of Model-level Optimization is unclear. For example, although BLOB-Q formulates a global optimization problem, it also decomposes it into subproblems for solving. Similarly, HAWQ models the Hessian of the task loss with respect to model outputs, so why is that not considered “Model-level Optimization”?

Minor comments:
1. In Related Works, the subheadings “PTQ” and “MPQ” do not need to be abbreviated again since they have already been defined earlier.
2. In Section 3, there is only one subsection (3.1). Consider adding an additional subsection for better structure.

### Soundness
2

### Presentation
3

### Contribution
2
