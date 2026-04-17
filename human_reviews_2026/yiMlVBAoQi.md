# Efficient Quantization of Mixture-of-Experts with Theoretical Generalization Guarantees

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Sparse Mixture-of-Experts (MoE) allows scaling of language and vision models efficiently by activating only a small subset of experts per input. While this reduces computation, the large number of parameters still incurs substantial memory overhead during inference. Post-training quantization has been explored to address this issue. Because uniform quantization suffers from significant accuracy loss at low bit-widths, mixed-precision methods have been recently explored; however, they often require substantial computation for bit-width allocation and overlook the varying sensitivity of model performance to the quantization of different experts. We propose a theoretically grounded expert-wise mixed-precision strategy that assigns bit-width to each expert primarily based on their *change in router’s* $l_2$ *norm* during training. Experts with smaller changes are shown to capture less frequent but critical features, and model performance is more sensitive to the quantization of these experts, thus requiring higher precision. Furthermore, to avoid allocating experts to lower precision that inject high quantization noise, experts with large *maximum intra-neuron variance* are also allocated higher precision. Experiments on large-scale MoE models, including Switch Transformer and Mixtral, show that our method achieves higher accuracy than existing approaches, while also reducing inference cost and incurring only negligible overhead for bit-width assignment.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel quantization method for Mixture-of-Experts (MoE) models that dynamically selects the precision for each expert based on changes in the $l_2$ norm of its weights. Theoretical analysis is provided for a toy MoE architecture, and numerical experiments demonstrate that the proposed approach outperforms heuristic methods in low-bit quantization scenarios.

### Strengths
1. Theoretical results are provided, and the method design is well aligned with the theoretical justification.  
2. Numerical results demonstrate that the proposed method achieves performance gains over baseline approaches, delivering comparable or even superior performance to heuristic methods and techniques that rely on calibration sets.

### Weaknesses
1. The theoretical analysis is based on a highly simplified MoE architecture focused on binary classification tasks. The gap between the theory and empirical results is not sufficiently addressed. In particular:  
   (1) Can the insights from Theorem 4.4 be used to *adaptively* determine quantization precision—rather than relying on predefined bit levels?  
   (2) Do activations in practical MoE models align with the conclusions of Theorem 4.3? If not, what are the key discrepancies, and how do they impact the method’s effectiveness?

2. Several important experimental details are missing:  
   (1) In Figure 2, the “activation weights” baseline—which specific method  does it correspond to?  
   (2) How are the hyperparameters (e.g., $\zeta$) selected? Is their choice task- or model-dependent, and how sensitive is the final performance to variations in $\zeta$?

### Questions
See weakness

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
2

### Summary
This paper presents an expert-wise mixed-precision quantization method for MoE models. It allocates higher bit-widths to experts that are more sensitive to quantization, identified through the router’s L2 norm change and neuron variance. The paper provides theoretical support showing that experts learning rare but important features require higher precision. Experiments show that the method achieves well performance than existing quantization methods with lower average bit-width and minimal overhead.

### Strengths
1.	The paper proposes a novel and efficient expert-wise mixed-precision quantization strategy based on router norm change.
2.	The paper provides theoretical analysis explaining why experts learning rare features are more sensitive to quantization errors.
3.	Achieves well performance and inference efficiency under the same average bit budget.

### Weaknesses
1.	When only pretrained router norms are used without fine-tuning, if the pretraining corpus and downstream data distributions differ, can the norm-based ranking still reliably identify “rare but important” experts?
2.	Can the claim that experts learning rare tokens exhibit weaker activations and smaller router norm changes be directly verified through visualization or statistical analysis on real LLM corpora?
3.	Would lightly retraining the low-bit experts further reduce the average bit-width or improve model robustness?

### Questions
see the questions

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
This paper proposes a novel expert-wise mixed-precision quantization strategy for Mixture-of-Experts (MoE) models, aiming to reduce their substantial memory footprint without significant performance degradation. The core contribution is a theoretically-grounded, two-stage heuristic for assigning bit-widths to experts. First, experts are ranked based on the change in their router's L2 norm during training $\Delta^T= || w^T||_2 - || w^0||_2$, the paper's theory suggests that experts with a smaller norm change are more sensitive to quantization and thus require higher precision. Second, this ranking is adjusted by promoting experts with high maximum intra-neuron variance to higher ranks to mitigate quantization noise. The authors provide a theoretical analysis for a simplified two-layer MoE model to justify their primary metric. They validate their approach empirically on large-scale models, including Switch Transformer and Mixtral (8x7B and 8x22B), demonstrating superior performance compared to several baselines, including uniform quantization and prior expert-wise methods. A key advantage highlighted is the negligible computational overhead of this bit-assignment method compared to more expensive SOTA approaches like PMQ.

### Strengths
The proposed method in this paper for bit-width assignment is computationally trivial, requiring a simple sort based on router norms. This stands in stark contrast to computationally expensive SOTA methods like PMQ, which require extensive calibration. This is a major practical advantage.

The method demonstrates strong performance on challenging benchmarks with large-scale Mixtral models, often outperforming existing expert-wise and non-expert-wise quantization methods.

### Weaknesses
1. For the key experiments on pre-trained models, the method abandons its primary metric (change in norm) for a surrogate (final norm). This switch is poorly justified and severs the link to the paper's own theoretical analysis. 

2. The `MaxVar` reordering step is not theoretically motivated and feels like an engineered solution to patch deficiencies in the primary metric. The lack of an ablation study makes it impossible to disentangle its effect, obscuring the true source of the performance gains.

### Questions
Roughly same as weakness:

- The use of the final router L2 norm as a surrogate for the change in norm is the most critical unsupported step in the paper. Can you provide any empirical evidence, for instance on a smaller model that can be fine-tuned, that these two metrics produce a similar expert ranking?
- The proposed theory posits that experts with smaller router norm changes learn "less frequent but critical features." Can the authors show what types of tokens or inputs are processed by the high-precision vs. low-precision experts in a real-world task, does this align with your "critical but infrequent" hypothesis?

- Please provide an ablation study that evaluates the performance of your method under three conditions: (i) using only the router norm ordering, (ii) using only the MaxVar ordering, and (iii) the proposed combination. This will help understand where the gains are coming from.

### Soundness
2

### Presentation
2

### Contribution
2
