## Human Reviewer 1

### Summary
This paper proposes a Forest-based Graph Learning (FGL) framework to address the trade-off between global receptive fields and computational efficiency in graph neural networks. The authors reinterpret graph information propagation as a transmission process across multiple spanning trees (a forest) to efficiently achieve long-range information aggregation. Methodologically, the paper designs a tree sampler based on homogeneity estimation and a linear-time tree aggregator, achieving Transformer-like global interaction capabilities while maintaining linear complexity. Experimental results demonstrate that FGL outperforms existing GNN and Graph Transformer models on multiple semi-supervised node classification tasks while maintaining significant training efficiency.

### Strengths
1. This is an interesting paper. The methodology of this paper represents a significant innovation compared to GNN and GT.
2. I really like the design of Tree Aggregator and it's quite interesting. The paper also provides ample theoretical analysis.
3. The paper is well-written and easy to follow. The experimental results demonstrate the superiority of this method.

### Weaknesses
1. Given that the paper emphasizes the efficiency of this method, experiments on large-scale graph datasets may be warranted. In my view, the experimental data presented in the paper is insufficiently large.
2. Compared to GNN and GT, the expressive power advantage of FGN remains unknown. I look forward to the authors providing further insights on this aspect in their paper.

### Questions
Does FGL struggle with relatively dense graphs, resulting in more edges and information being discarded in the generated tree compared to the original graph?

### Soundness
3

### Presentation
3

### Contribution
4

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a Forest-Based Graph Learning paradigm for semi-supervised node classification, addressing the trade-off between cost-effectiveness and global receptive field in existing GNNs and Graph Transformers. FGL models information propagation as transport over a forest of spanning trees, generated via a homophily-guided sampler. It incorporates a linear-time tree aggregator for efficient long-range interaction and a tree fuser to merge multi-tree knowledge. Experiments on show FGL outperforms state-of-the-art methods (e.g., 11.9% gain over GCNII, 16.1% over DIFFormer) with linear complexity and faster runtime.

### Strengths
1.Breaks the local-global trade-off by leveraging spanning trees, enabling efficient global coverage with low structural cost.

2. Linear time/space complexity per epoch, with 2–5× speedup over GTs (e.g., DIFFormer) and deep GNNs (e.g., GCNII).

3. Excels on both homophilous (Cora, Pubmed) and heterophilous (Actor, Cornell) graphs, with robust generalization under label scarcity.

### Weaknesses
1.Using a forest to address homophily–heterophily problems is not a particularly novel idea, as many path-based methods [1] also aim to expand the receptive field. The idea of employing trees follows a similar intuition, broadening the range to enhance the model’s ability to capture global structural knowledge.

2.In line 252, there seems to be a definition error — what does Fa(u) and g(u) represent, and how do they satisfy Properties (I) and (II)? If both letters H and S denote embeddings, it would be better to use a consistent notation, otherwise it becomes confusing. I suggest the authors clarify this theorem more explicitly.

3.My main concern with this method lies in the semi-supervised setting: how to accurately characterize homophily becomes a major issue. The paper uses “the cross-entropy loss with targets... We train the local graph attention by minimizing...” to obtain pseudo labels. However, if the model performs poorly, the predicted homophily may deviate significantly from the true one, leading to a tendency toward over-smoothing.

4.In Section 4.1, what does c represent? If Y is a one-hot label, then finding k nearest neighbors in the representation space seems meaningless.

5.The paper lacks a discussion of related work on heterophilous graphs, and there is also a lack of baselines specifically designed for heterophilous GNNs

### Questions
See weaknesses

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
5

---

## Human Reviewer 3

### Summary
The paper propose the forest-based graph learning paradigm, which achieve the efficient long-range information propagation. The core idea is to reinterpret graph message passing as transport over spanning trees.

### Strengths
Novel conceptual paradigm. The proposed Tree Aggregator is a meaningful and technically novel formulation that enables composable and reversible message passing on trees, achieving efficient long-range propagation with linear complexity.

Well writing. 

The experiment section is solid and broad. The evaluation includes both homophilous and heterophilous benchmarks with extensive baselines, ablations, and hyperparameter analysis. The empirical performance (especially on Wisconsin, Texas) is notably strong.

### Weaknesses
While the paper emphasizes “forest-based” modeling, the connection to existing tree decomposition / hierarchical GNN approaches(For example, [1]) is not clearly distinguished.

Overly complex pipeline: The four-stage pipeline (preprocessing → tree sampling → tree aggregator → tree fuser) introduces multiple training stages and parameters (β₁, β₂, γ, Kᴸ, N_T), yet lacks analysis on training stability and sensitivity.


[1] Fast and Effective GNN Training with Linearized Random Spanning Trees

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper proposes Forest-Based Graph Learning (FGL) to reconcile global receptive fields with efficiency in GNNs. Instead of deep stacking or quadratic global attention, FGL models message passing as transport over a small forest of spanning trees: (i) build an augmented graph via pseudo-label kNN; (ii) sample trees using a homophily-guided, weighted Wilson method; (iii) run a two-pass, linear-time tree aggregator (bottom-up then top-down) on each tree; and (iv) fuse normalized per-tree embeddings with a lightweight local module via a residual. Theoretically, better homophily scoring increases expected tree homophily and approaches a structural upper bound.  FGL reports competitive or superior accuracy and favorable runtime on node classification benchmarks.

### Strengths
1. The paper proposes a novel graph learning framework—Forest-Based Graph Learning (FGL) that combines global structural modeling with local feature propagation by viewing graphs through a forest of spanning trees to capture hierarchical node dependencies. 

2. The technical quality is strong: the objectives and their links to smoothing and spectral properties are clearly derived, and extensive experiments across benchmarks. 

3. The writing is clear and well-structured.

### Weaknesses
1. The approach largely reads as a straightforward assembly of existing components: pseudo-label kNN augmentation (standard self-training/graph augmentation), weighted Wilson sampling driven by attention scores (classic LERW with learned edge weights), two-pass tree DP (bottom-up/top-down rerooting), and RowNorm + mean + residual fusion. There seems to be little apparent redesign of the modules or new learning objectives beyond composition.

2. The paper primarily evaluates on semi-supervised node classification, leaving unclear whether the forest paradigm generalizes to broader settings (graph classification/regression, link prediction, inductive and dynamic graphs, heterophily extremes, robustness to noise, and scalability in billion-edge graphs). Broader tasks and protocols would better establish practical generality.

3. The method introduces multiple hyperparameters ($\beta_1, \beta_2, \gamma$ etc.) that require careful tuning for different scenarios. The hyper-parameter part implies that performance may be deeply dependent on these parameters, which could make the method difficult to deploy in practice, potentially limiting its practical applicability.

4. The paper does not thoroughly examine when the forest paradigm might underperform. For instance, what happens when node features are noisy or weakly correlated with the graph topology? Does the method perform well on highly dense graphs where local tree-structured neighborhoods may not be able to effectively capture the graph’s global dependency patterns?

### Questions
Please address the concerns raised in the weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
5