# DR-CFGNN: A Completion-Aware Framework for Counterfactual Explainability in Graph Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 2

## Abstract
In this study, we propose a novel framework for counterfactual explainability in graph neural networks (GNNs). To the best of our knowledge, this is the first generic, model-agnostic method for local-level GNN explainability that considers both edge removal and edge assertion. The approach takes advantage of the progress achieved in factual explainability, coupling it with an encoder-decoder deep learning model to learn valid and robust graph expansions. In addition to standard benchmark datasets, we evaluate our method on a new variant of a popular synthetic dataset to study how explainability is influenced by data incompleteness, a common characteristic of real-world graph data. A multi-faceted experimental analysis with both established metrics from relevant literature and novel ones aimed at assessing the validity and the quality of explanations, demonstrates the advancement that our proposed approach brings to state-of-the-art baselines.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a generic, model-agnostic method for local-level GNN explainability that considers both edge removal and edge assertion. The approach takes advantage of the advanced explanation objectives through an ensemble design. 
Experiments show that the proposed method not only achieves state-of-the-art performance on standard benchmark datasets but also generates more robust explanations when data incompleteness occurs.

### Strengths
- This paper proposes a novel ensemble framework for counterfactual explainability in graph neural networks (GNNs).
- The study of the robustness of counterfactual explanations based on noise is interesting.
- The paper overall is easy to follow.

### Weaknesses
1. The method proposed in the paper resembles an ensemble explainer, as it integrates candidate explanations from multiple explanatory objectives. Reporting the best-performing one among these will always yield results no worse than the best explanations from a single objective, and this approach can intuitively improve performance across metrics. However, each of these objectives can be found in existing works such as [1], so the contribution to explanatory methods is not strong.  

2. Regarding the paper’s contributions to the evaluation system, they mainly come from the integration of existing metrics. Although the authors are the first to evaluate the impact of counterfactuals in dealing with incomplete graph data through a noisy-based metric, the relevant theories and implementation methodologies also originate from existing works such as [2]. As one of the key contributions claimed in the paper, this noisy-based metric lacks discussion and analysis on the necessity of its introduction for GNN explanation. This necessity determines whether it can be adopted as a new general metric for subsequent GNN explanation works.  

3. Eq. (3), Eq. (4), and Eq. (5) are confusing. The right-hand side of the equal sign represents the edited adjacency matrix, yet the graph label does not appear in the equations. I am confused about how it optimizes the objective described in the context.  

In summary, when considering the paper’s contributions to explanatory methods and evaluation methods in isolation, they are not strong. Therefore, it is more necessary to supplement analyses that previous works have not conducted, such as discussing the necessity of introducing the new metric, and explaining why an ensemble is necessary and why a unified objective cannot be used to extract explanations in an end-to-end manner. However, the paper lacks such analyses and fails to provide new impactful insights.

- [1] Joint factual and counterfactual explanations for top-k gnn-based recommendations.
- [2] Robust counterfactual explanations in machine learning: a survey.

### Questions
Please refer to the above weakness section for suggestions and questions.

### Soundness
3

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
The paper proposes a method for constructing counterfactual graphs through edge addition and edge removal, leveraging a graph classification model and a factual graph explainer. 

To identify minimal edge removals, the method first extracts a subgraph that preserves the original classification result using a factual graph explainer, then exhaustively searches candidate subgraphs by removing edges from this factual subgraph to disrupt the pattern.

For edge addition, the approach classifies graphs with incomplete motifs and those with complete motifs into two categories, and trains a GNN to predict the edges required to complete the motif so that the graph’s class label changes accordingly.

The main contribution of this paper lies in exploring the potential of decoupling edge addition and edge removal, and in analyzing their respective roles in generating counterfactual examples.

### Strengths
1. The paper proposes a new framework for identifying counterfactual examples in post-hoc graph neural network (GNN) analysis.

2. It explores the potential of separating edge addition and edge removal, and analyzes their respective effects on discovering counterfactual examples.

3. The proposed model demonstrates advantages in specific scenarios, such as recognizing graph incompleteness.

### Weaknesses
1. The paper lacks methodological novelty. The edge removal component essentially performs an exhaustive search based on the results of existing factual GNN explainers, while the edge addition component closely resembles traditional link prediction methods. Overall, the contribution is insufficient to meet the novelty standards expected for ICLR.

2. Several ambiguous descriptions appear in key sections, making the paper difficult to follow. More detailed comments are provided in the following section.

3. The reported performance improvement over CFX primarily arises from the experimental setting involving incomplete motifs, which is specifically tailored to favor the proposed edge addition approach. Therefore, this advantage cannot be considered a general one.

### Questions
In Definition 3.1, the graph is inconsistently represented as ${A, X}$ and ${M, F}$. Please clarify whether these notations represent different formulations.
In Section 4.1, the loss function is expressed as a matrix, which is an unconventional formulation in machine learning. It would be helpful to include an explanation or derivation of how this matrix loss is computed.
In Equation (6), $L_{+/-}$ appears to contain the same term as $L_{+}$ or $L_{-}$. Readers may find it confusing why these terms need to appear again in the final loss formulation without a clarification addressing this overlap.
The feature mask introduced in the edge-removal method and the feature noise used in the experiments do not seem directly related to the proposed approaches. Please clarify their roles and how they contribute to the main methodology.
In Section 5.3, it is unclear why $r = k = 0$ is considered a trivial case. Although the total number of edges remains unchanged, different edge combinations could still alter or disrupt structural patterns in the graph.
In Table 1, no explanation size is reported for the CFE models. Since explanation size and PN are typically considered together in a trade-off, including explanation size would allow for a fairer and more complete comparison.
In RQ3, the paper notes that adding noise could result in invalid molecular graphs, yet similar concerns are not raised for textual or semantic noise in sentimental analysis. Please clarify why these two types of noise are treated differently.

### Soundness
2

### Presentation
2

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
The paper proposes a framework for local counterfactual explainability that jointly considers edge removals and additions. The experiments indicate that the proposed method outperforms the baselines.

### Strengths
- The paper is well-motivated, and the problem is clearly defined.
- The method is explained clearly.

### Weaknesses
- The presentation of tables and figures is difficult to read; some tables are very small and not visually clear. The overall visual presentation needs improvement.
- Novelty appears limited; the paper does not convincingly position its methodological contribution.
- The number of baselines is limited, focusing on only one baseline and its variants.
- The algorithm’s computational complexity is not discussed, and running times are not reported.
- The graphs used in experiments are small; please provide, preferably in a table (Appendix is fine), the average number of nodes and edges for each dataset.
- The shared code includes a requirements.txt that references local paths and lacks version specifications, which hinders reproducibility.

### Questions
- A random algorithm outperforms the main algorithm on one of the most important metrics. Can you elaborate on why this occurs? Is it due to small graph sizes? If so, why should one prefer the ML-based approach over a random baseline?
- Can you provide a general requirements.txt with pinned package versions and the Python version? Also, please share details of your runtime environment (e.g., CPU/GPU, memory, and machine specifications) to support reproducibility.
- Can you report the algorithm’s time complexity and empirical running times?

Additional suggestions for readability and improvements:
- Please refer explicitly to the table for RQ4 in Section 5.4.
- Consider enlarging tables, adopting consistent formatting, and adding visual cues (e.g., row/column grouping, margins, fonts) to improve readability.
- Strengthen the novelty narrative by clearly contrasting your approach with existing methods and articulating the specific scenarios where your framework offers unique advantages.
Expand the baseline set to include diverse and stronger comparators to better contextualize performance.

### Soundness
2

### Presentation
2

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
This paper proposes a counterfactual explanation framework for graph neural networks that aims to identify important subgraphs and provide interpretability through a decomposition-and-masking procedure. While the motivation is sound, the paper suffers from weak novelty justification, and insufficient experimental rigor. Moreover, several key methodological and presentation issues significantly hinder readability and the evaluation of contributions.

### Strengths
1.	The paper addresses the important topic of counterfactual explanations in graph models, which is relevant and timely.
2.	The experimental section provides some visualization and ablation, which indicates implementation effort.

### Weaknesses
1. Weak and Redundant Challenge Definition
The listed “challenges” are not convincing.
*	The first two challenges both focus on subgraph reasoning or graph expansion, which have already been extensively explored in prior works such as GCFExplainer. Therefore, the motivation for re-stating these as novel challenges is weak.
*	Furthermore, the paper does not analyze the limitations of existing “graph expansion” methods (e.g., GCFExplainer), nor explain what unique challenge this work addresses beyond them.
*	The fourth challenge mentions that “most models implement costly search methods,” yet the proposed paper provides no discussion or quantitative analysis regarding computational complexity or runtime cost, which makes this challenge invalid.
2. Problem Formulation Unclear:  The problem formulation section is confusing and not well aligned with existing definitions of counterfactual explanation. The authors cite Guo et al. and CFExplainer (the baseline), but these works already provide a formal and widely accepted definition of the counterfactual explanation task on graphs. Since the current paper does not define a new problem, the formulation should directly follow this established setting instead of introducing vague “decomposition” and “mask” operations, which belong to the method design, not the problem formulation.
3. Unclear Method Innovation: Due to the challenges are vague and overlap with existing work, it is difficult to understand the design motivation of the proposed framework. The method seems to combine existing decomposition and masking steps without a clear theoretical or algorithmic novelty. The paper does not demonstrate why these design choices are necessary or how they overcome specific weaknesses of prior methods.
4. Baseline Selection Insufficient: The experiments only compare with CFExplainer, which is inadequate. At minimum, the authors should include Grad-CAM, or GCFExplainer, the relative methods mentioned in your paper or the paper of CFExplainer, as additional baselines. Without these comparisons, the claimed improvement lacks credibility.
5. Missing Discussion of Hyperparameters: Key hyperparameter choices are not justified. For example, in RQ3, the perturbation ratio is fixed to 4%, but the paper provides no explanation for this choice, nor any sensitivity analysis. Such details are essential to assess the robustness and fairness of the method.
6. Experimental Presentation and Formatting Issues: 
The presentation quality is poor and significantly affects readability:
*	Several figures (e.g., Figure 2) contain multiple subplots in a single image, making the contents unreadable due to small font and low resolution.
*	Table formatting is inconsistent across the paper, with misaligned columns and missing captions.
*	Figure labels and axis annotations are too small to interpret.
*	Overall, the visual quality and layout do not meet publication standards.

### Questions
Please address the weaknesses.

### Soundness
1

### Presentation
1

### Contribution
2
