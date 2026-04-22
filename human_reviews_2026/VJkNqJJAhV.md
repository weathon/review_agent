# FAME: Formal Abstract Minimal Explanation for Neural Networks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
We propose $\textbf{FAME}$ (Formal Abstract Minimal Explanations), a new class of abductive explanations grounded in abstract interpretation. FAME is the first method to scale to large neural networks while reducing explanation size. Our main contribution is the design of dedicated perturbation domains that eliminate the need for traversal order. FAME progressively shrinks these domains and leverages LiRPA-based bounds to discard irrelevant features, ultimately converging to a $\textbf{formal abstract minimal explanation}$.  To assess explanation quality, we introduce a procedure that measures the worst-case distance between an abstract minimal explanation and a true minimal explanation. This procedure combines adversarial attacks with an optional $VERI{\large X}+$ refinement step. We benchmark FAME against $VERI{\large X}+$ and demonstrate consistent gains in both explanation size and runtime on medium- to large-scale neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes FAME (Formal Abstract Minimal Explanations), a framework for generating formally verified minimal explanations for neural network predictions. The method combines abstract interpretation with LiRPA (Linear Relaxation based Perturbation Analysis) to efficiently identify the minimal subset of input features responsible for a model’s decision. FAME removes the traversal-order dependency in prior formal explanation methods through an abstraction-based parallel mechanism that eliminates multiple irrelevant features simultaneously. Experiments on MNIST and GTSRB show that FAME produces more compact explanations and up to 25× faster runtime compared to VERIX+, supporting its theoretical claims.

### Strengths
1. The paper introduces a novel and theoretically grounded framework, FAME (Formal Abstract Minimal Explanations), that advances formal explainable AI by combining abstract interpretation with LiRPA for efficient and provably minimal explanations.
2. It effectively removes the traversal-order dependency that limits prior formal XAI methods and achieves substantial improvements in explanation compactness and runtime on benchmark datasets.

### Weaknesses
1. The paper introduces formal minimal explanations as a novel interpretability construct, yet the theoretical operationalization of “interpretability” remains vague. The framework lacks a precise articulation of how its formal minimality relates to established interpretive criteria, such as faithfulness or epistemic transparency, which weakens the clarity of its conceptual contribution.
2. The evaluation is limited to MNIST and GTSRB, which are too simple to validate FAME’s scalability or general effectiveness. More representative benchmarks such as CIFAR-10 for visual interpretability or COMPAS for tabular reasoning would better demonstrate the framework’s robustness across domains.
3. Although the paper asserts improved explanation quality, the experiments report only efficiency-related metrics (runtime and explanation size). Without fidelity- or stability-based evaluation, the claimed enhancement in explanation quality is not empirically supported.
4. The iterative optimization in Abstract Batch Freeing is described as a key component of the framework, but its contribution has not been empirically isolated. No ablation or comparative results are provided to verify whether this step improves efficiency or explanation compactness. Without such analysis, the practical impact of this mechanism remains speculative.

### Questions
1. I do not fully understand the validation setup described in Section 7.2, where the authors claim that FAME achieves formally minimal explanations but provide no direct comparison against exhaustive search or exact verification results. What metric was used to assess proximity to the true minimal set, and how do the authors justify that this indirect evaluation sufficiently demonstrates minimality?
2. The paper reports in Section 6 that FAME relies on LiRPA-derived abstract bounds to identify irrelevant features. However, recent work has shown that local linear relaxations may miss globally relevant feature interactions (see Lu et al., 2024, ICML — EiG-Search: Generating Edge-Induced Subgraphs for GNN Explanation in Linear Time). Could such limitations affect the completeness of FAME’s explanations, and have the authors considered adaptive or hierarchical abstraction domains to mitigate this risk?

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
4

### Summary
This paper introduces FAME, a new framework for computing formal abductive explanations for neural networks. The core of the method formulates 'batch freeing' as a multidimensional knapsack problem, which is then solved efficiently using a greedy heuristic. Experiments on MNIST and GTSRB models (both FC and CNN) demonstrate that FAME consistently outperforms the VERIX+ baseline.

### Strengths
* The paper shows a novel approach to breaking the 'sequential bottleneck' in formal XAI.
* Using LiRPA-based abstract interpretation (which is GPU-accelerated), instead of CPU-bound solvers, is practical. This can speed up and improve scalability.
* The formulation of batch freeing as a knapsack problem is a clever and effective

### Weaknesses
### 1. Data
The experiments are limited to MNIST and GTSRB. These are relatively small-scale and low-dimensional datasets. The paper's claim to "scale to large neural networks" is not fully substantiated by this evidence. The scalability of FAME on high-dimensional data (e.g., CIFAR-100 or ImageNet) is a major open question.

### 2. Model
Similarly, the models used (small FC and CNNs) are not "large-scale" by modern standards (e.g., ResNets, Transformers). The quality of LiRPA bounds is known to degrade with network depth. The paper does not investigate how the "gap" between the abstract explanation and the true minimal explanation scales with model depth. If this gap becomes too large, the abstract-first approach may lose its advantage, as the final refinement step would become computationally dominant.


### 3. Metric
The "distance to minimality" (Section 6) is a good concept, but the paper does not fully explore the failure modes. The paper notes in Section 4.2 that if bounds are too loose (e.g., for complex models or large epsilons), the abstract method may fail to free any features. This "failure mode" is not empirically characterized.

### Minor
* format: It seems the paper format is not the ICLR format (e.g., references)
* Suggest improving clarity and making the explanation less dense for readers to get the idea. Also, I can’t get the idea from Figure 1 at first; maybe improving this can be helpful for the readers.

### Questions
* Could the authors provide results on a more complex dataset (e.g., CIFAR-10) and a deeper architecture (e.g., a small ResNet) to more robustly demonstrate scalability?
* What percentage of the total runtime does the "optional VERIX+ refinement step" account for, on average, across the reported experiments?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
Formal explainability paper that presents FAME, a method to generate *abductive* explanations for neural networks. Novelty is in the adoption of LiRPA bounds to discard irrelevant features and in not relying on ordered features, thus achieving faster results.

### Strengths
- Formally computing explanations with guarantees for neural networks is a critical and important research problem, with room for novel contribution.
- The approach is described in coherent manner, and paper storytelling is linear.

### Weaknesses
- Coming from a different community, I found hard to familiarize with the jargon, as some key concepts are left without definition (e.g. abstract interpretation, LiRPA). They may be second nature for the authors, but gently introducing them would help clarify and broaden up the audience.
- The paper could use some examples, to ease the understanding from a broader audience (e.g. when introducing the verification task, 99-102, or by providing additional examples properties to verify $P$ - other than adversarial robustness).
- (minor): definition identifiers in the main corpus do not match (e.g. line 162, 219). Citation style not compliant with ICLR guidelines.
- Evaluation covers only vision scenarios (MNIST, GTSRB) and does not cover tabular data (a critical use case for certified explanations).
- Experimental results w.r.t baseline: neither explanation size or response time are dramatically smaller (i.e. better) than VERIX+.

### Questions
- Could you clarify what is the proposed delta over LiRPA?
- How can FAME support discrete features?
- Is VERIX+ the only baseline you can compare against?

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
3

### Summary
The paper introduces FAME (Formal Abstract Minimal Explanations), a scalable framework for generating abductive explanations in neural networks. FAME leverages abstract interpretation and dedicated perturbation domains to eliminate the need for traversal order in formal explanations. It also introduces a procedure for measuring the quality of these explanations by comparing them to true minimal explanations. The authors demonstrate that FAME consistently outperforms the state-of-the-art (VERIX+) in terms of explanation size and runtime, particularly for medium- and large-scale networks. This framework is designed to handle the scalability challenges of formal explainability methods for neural networks and is made publicly available for further research.

### Strengths
1. The paper presents a novel approach to formal explanations, eliminating the traditional traversal order bottleneck. This is a significant step forward in making formal XAI methods scalable to large networks.

2. The proposed FAME framework is sound, and the experiments demonstrate its potential to scale well with complex networks.

3. The writing is generally clear, and the method is presented in an organized way. The diagrams and figures help illustrate key points effectively.

### Weaknesses
**Traversal Order Issue:** The paper claims to eliminate traversal relationships, but the algorithm still performs multiple rounds of feature selection, which can be considered a form of traversal. This method needs further clarification to truly eliminate traversal relationships.

**Impact of Feature Order:** The paper underestimates the significant impact that the order of feature selection has on the results. A more thorough discussion and empirical validation are required. Specifically, when removing feature A, feature B may still play a role, and vice versa. This interdependency should be addressed.

**Attack Boundaries:** The approach only provides a lower bound or instance-based gap for attack robustness, making it difficult to achieve a global worst-case upper bound. This limitation needs clearer discussion.

**Greedy Method Guarantee:** The greedy approach used in the paper lacks an approximation guarantee, which limits its theoretical robustness. An approximation guarantee would strengthen the argument for its practicality.

**Simplicity of Dataset:** The dataset used is relatively simple. Most explainability techniques are validated on more complex datasets like ImageNet. It would enhance the credibility of the method if it were tested on more challenging datasets.

**Complexity Analysis:** The paper lacks a detailed analysis of the algorithm's complexity, especially concerning GPU acceleration and distributed verification. A more detailed discussion of these aspects would provide a clearer understanding of the scalability of the method.

**Hyperparameter Sensitivity:** The paper does not sufficiently address the sensitivity of the results to hyperparameters. A more comprehensive analysis of this sensitivity would help understand the stability and robustness of the approach.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2
