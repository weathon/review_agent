# Rethinking Distance Metric Generalization in Neural Combinatorial Optimization for Vehicle Routing Problems

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Neural combinatorial optimization (NCO) has emerged as a promising approach for solving the vehicle routing problem (VRP). However, its ability to generalize across diverse instances is a key challenge for practical applications. Current research on generalization primarily focuses on problem scale and node distribution. The distance metric between nodes, such as 2D Euclidean distance or geographical distance, is also an important characteristic of VRP instances. Unfortunately, existing NCO methods typically use a single distance metric for both training and testing, neglecting the diversity of distance metrics. To fill this gap, this paper systematically investigates the impact of distance metrics. First, we introduce a benchmarking framework that supports multiple distance metrics and evaluates model generalization across them. Experimental results reveal that models trained on instances with a single distance metric perform poorly on instances with different metrics. This suggests that variations in distance metrics pose a significant challenge to model generalization. Second, we examine several training data configurations and find that jointly training on data with diverse distance metrics significantly improves model generalization across different metrics. Moreover, by integrating our proposed method for distance metric generalization with prior advances for problem scale and node distribution generalization, the performance of NCO models on various real-world VRP instances is substantially improved.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript proposes a Multi-Metric training strategy.

### Strengths
This manuscript is well-structured.

### Weaknesses
**W1 Limited innovation:** Although the authors proposed a Multi-Metric training strategy, it is relatively simple. Moreover, joint training approaches of this type have already been widely adopted in other neural combinatorial optimization studies [1-3].

**W2 Insufficient experiments:**  The experimental evaluation is limited to applying MDMMT on MatNet, which constrains the evidence for the method’s generality.



[1] Multi-Task Learning for Routing Problem with Cross-Problem Zero-Shot Generalization. KDD, 2024.

[2] CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention. ICML, 2025.

[3] RouteFinder: Towards Foundation Models for Vehicle Routing Problems. TMLR, 2025.

### Questions
See Weaknesses.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates distance metric generalization in vehicle routing problems (VRPs). It introduces a benchmarking framework covering eight distinct distance metrics to evaluate the generalization ability of neural combinatorial optimization (NCO) models. Furthermore, it proposes a simple multi-task training approach as a solution for this setting. However, this paper still has several notable issues, and in its current form, the contributions may not yet be sufficient for acceptance at a top-tier ML conference.

### Strengths
* This paper investigates distance metric generalization, a key challenge that limits the applicability of NCO approaches in real-world scenarios. The proposed generalization setting deserves more attention from the NCO community.
* The writing of the abstract and introduction is clear and well-structured.

### Weaknesses
* This paper studies distance metric generalization in VRPs. However, a recent work [1] has already explored this setting. The authors should discuss and highlight the differences and contributions beyond prior work.
* Section 3 presents a benchmark for distance metric generalization, but it only includes one model (MatNet) and one training setting (2D Euclidean). The benchmark would be more comprehensive if it included additional approaches and settings, such as those mentioned in Section 2 `Distance-Matrix-Based`.
* It is rather straightforward that multi-metric training can enhance model generalization across distance metrics. The proposed method therefore offers limited technical novelty, and deeper insights or analyses are needed.
* The paper title refers to VRPs, yet the experiments are limited to TSP. Additional problem variants (e.g., CVRP) and stronger baselines should be included to validate generality.
* Beyond the abstract and introduction, the writing of the remaining sections requires improvement for clarity and flow.

[1] Lifelong Learner: Discovering Versatile Neural Solvers for Vehicle Routing Problems.

### Questions
* Why are only three distance metrics considered for multi-metric training (MMT) in Section 4?
* Why not considered asymmetric distance?
* Although the distance-matrix-based input format is more general, it incurs an $O(N^2)$ space complexity. What is the computational overhead when training or testing on large-scale instances as reported in Table 4?

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper addresses the issue of limited distance metric generalization in neural combinatorial optimization (NCO) models applied to the Vehicle Routing Problem (VRP). It highlights the dependency of current NCO methods on a single distance metric (e.g., 2D Euclidean distance), which restricts their ability to generalize to different distance metrics encountered in real-world scenarios (e.g., 3D Euclidean, geographical distances). The authors propose a benchmarking framework that supports eight different distance metrics to evaluate NCO models. They introduce a "Multi-Metric Training (MMT)" strategy, which significantly improves generalization across various distance metrics. Experimental results on VRPLIB instances show the effectiveness of MMT.

### Strengths
1. This work is relevant for the practical deployment of NCO models in real-world VRP scenarios, where diverse distance metrics are common.
2. The methodology is clearly explained, and the paper is easy to follow.

### Weaknesses
1. The MMT strategy is based on only four distance metrics. The authors do not explain why these particular metrics were selected and how well they represent the diversity of real-world metrics. Additional distance metrics or a more diverse selection should be included.
2. The analysis of *why* different distance metrics affect model performance feels a bit superficial. The paper shows that single-metric training leads to poor generalization but doesn’t really explore what properties of distance metrics (e.g., symmetry, triangle inequality strength) cause this. Some theoretical or interpretive discussion would make the contribution stronger.  
3. The MMT approach is quite straightforward — essentially an application of standard multi-task training — without much adaptation to the specific characteristics of distance metrics. 
4. The author should consider other state-of-the-art distance-matrix-based models as baselines instead of MatNet (2021), as discussed in Related Work.

### Questions
1. The paper defines a “Maximum Distance” metric, but it’s unclear where such a metric would appear in real-world routing scenarios. Could the authors give an example of its practical use?  
2. How were the metrics chosen for MMT training? Were other metrics (e.g., Manhattan, Maximum, obstructed) tested and found less effective?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents an empirical study on the generalization of Neural Combinatorial Optimization (NCO) models across different distance metrics for Vehicle Routing Problems. The authors first demonstrate that models trained on a single metric (e.g., 2D Euclidean) fail to generalize to unseen metrics, as shown in Table 1. To address this, the paper introduces a new benchmark framework encompassing eight distinct distance metrics. The authors also propose a method, "Multi-Metric Training" (MMT), which involves jointly training the model on instances from multiple different metrics. The final proposed model, "MDMMT," combines this approach with existing techniques for distribution generalization to achieve improved performance on real-world VRP instances.

### Strengths
This work addresses an important and under-explored aspect of generalization in NCO models. The paper's primary strength is its clear empirical demonstration that standard models overfit to a single distance metric, resulting in a significant performance collapse on unseen metrics (Table 1). The proposed benchmarking framework is a useful contribution to the community for evaluating this specific dimension of robustness.

### Weaknesses
The paper's central weakness is the profound lack of methodological novelty. The proposed "Multi-Metric Training (MMT)"  is presented as a novel method, but it is functionally identical to the standard, well-known data augmentation strategy of training on a more diverse dataset. The paper's own text highlights this contradiction: it frames "Multi-Distribution Training (MDT)"  (training on diverse node distributions) as prior art , while presenting MMT (training on diverse metrics) as its own contribution. This is a distinction without a difference; both are simple data augmentation strategies, and the core "insight" that training on diverse data improves generalization on that data is not a sufficient contribution for a top-tier methods conference.   

The motivation for the benchmark itself is underdeveloped. While the authors provide real-world applications for 3D Euclidean and Geographical distances , they fail to provide any practical justification for the inclusion of other key metrics, such as 2D/3D Manhattan or 2D/3D Maximum distances. Without this context, these additions appear arbitrary, undermining the practical significance of the benchmark. This supports the concern that experiments are being conducted for their own sake rather than to address clearly defined, practical needs.   

The paper's contribution appears incremental when viewed in the context of recent SOTA "generalist" NCO solvers. Works like UniCO (Pan et al., 2025)  are also leveraging matrix-based inputs  to unify entirely different problem types (e.g., TSP, HCP, SAT) , which is a far more significant generalization challenge than generalizing across different instantiations of a distance matrix.   

Furthermore, the "real-world" experiment in Section 5 introduces significant confounding variables. To handle large-scale instances (up to 10k nodes), the paper embeds the NCO model into an iterative 'destroy and re-generate' heuristic framework. The final results in Table 3  show the performance of this combined system, making it impossible to decouple the gains from the MDMMT training strategy itself versus the gains from the powerful iterative search framework. A more rigorous design would have included a head-to-head comparison on medium-scale real-world instances without this heuristic wrapper to isolate the model's true performance.   

Finally, the design of the MMT training set is arbitrary and lacks supporting ablation studies. The authors select {2D Euclidean, 3D Euclidean, Geographical} for MMT training  and show it generalizes to unseen metrics like 2D Manhattan. However, the paper provides no analysis as to why this specific combination is effective or if it is optimal. It is unknown if another combination (e.g., {2D Euc, 2D Manhattan, Obstructed}) would also generalize to 3D and Geographical metrics. This lack of analysis makes the specific MMT configuration feel arbitrary and its success on unseen metrics insufficiently explained.

Pan, W(2025). "UniCO: On Unified Combinatorial Optimization via Problem Reduction to Matrix-Encoded General TSP." In The Thirteenth International Conference on Learning Representations (ICLR).

### Questions
Can you clarify the methodological novelty of MMT? Given that training on diverse data to improve generalization is a standard technique, and that the paper itself treats the analogous "Multi-Distribution Training" (MDT) as prior art , what is the specific, novel contribution of MMT beyond being a simple data augmentation strategy?   

What are the specific, real-world VRP applications for the Manhattan and Maximum distance metrics that motivated their inclusion in the benchmark?

How does your approach to "metric generalization" compare to more ambitious "generalist" solvers like UniCO (Pan et al., 2025) , which aim to generalize across different problem types using a similar matrix-based input format?

### Soundness
2

### Presentation
2

### Contribution
2
