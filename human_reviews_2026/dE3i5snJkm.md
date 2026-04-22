# GLLP: Graph Learning from Label Proportions

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Learning from Label Proportion (LLP) is a weakly supervised learning paradigm in which only aggregated label proportions over collections of instances (i.e., bags) are provided, rather than individual labels. This allows classification while preserving privacy or reducing annotation costs. Existing LLP methods, however, have been largely restricted to i.i.d. tabular or image data. No solution currently addresses graphs, where instances are inherently interdependent through network structure. In this paper, we generalize LLP to the graph domain and study the problem of node classification with label proportions, where only distributional supervision is available for node bags, and the goal is to infer labels for all nodes in the graph. We argue that the lack of node-level supervision is the main challenge for LLP on graphs, and that existing methods based on i.i.d. assumptions fail to exploit topological correlations. To overcome this, we propose GLLP(Graph Learning from Label Proportions), a framework that leverages Optimal Transport (OT) with a homophily-aware cost to generate soft pseudo-labels for individual nodes. These pseudo-labels provide stronger supervision signals for training Graph Neural Networks. We further establish theoretical guarantees showing the alignment of our cost function with the node classification objective. Extensive experiments on six homophilic graph benchmarks demonstrate that GLLP consistently outperforms existing LLP baselines and variants. Code and benchmark datasets is released in: https://anonymous.4open.science/r/GLLP-2C9E/README.md.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper studies the problem of learning from label proportion (LLP) for graph data. In particular, the paper devises a very interesting method GLLP, which decomposes the LLP problems into two: one is to explore optima transfer (OT) to derive soft pseudo-labels for nodes based on a bag-level label distribution and the node-level prediction from GNN models; the other is to train the GNN model based on the derived pseudo-labels. GLLP has alternations between these two steps, pseudo-label generation and GNN training, which is rather effective to iteratively refine node-level predictions and remain consistent with bag-level proportions. Experimental evaluation gives clear evidence to demonstrate the strength of the GLLP over other different proposals.

### Strengths
-- The paper addresses an important problem of LLP for graph data.
-- It presents a first proposal for the research problem and the main idea is quite appealing to me.
-- The paper is well-written and the experimental results give strong support for the proposed method.

### Weaknesses
-- The theoretical analysis is Section 3.3 is quite interesting to me, but the proof in the appendix is rather difficult for me to understand. 
-- There are quite some typos in the paper: (page 3, line 125; page 4, line 194; page 5, line 228, line 236, line 248, line 254, line 260; page 6, line 299, etc. also a few typos on page 12).

### Questions
1. What is exactly the temperature \tau? In the experiments, it is simply set at 2. Why? How does its value impact the results?
2. In general, can you make the proof on page 12 more detailed?

Especially, how to expand Eq. 12 into Eqs. 26 and 27? I also don't understand the last sentence "Leveraging the assumption in Eq. 15, we can obtain the final conclusion". I thought Eq. 15 is what you aim to prove.
 
Once these are resolved, I am willing to improve my overall rating.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Graph Learning from Label Proportions (GLLP), extending the Learning from Label Proportions (LLP) paradigm to graph-structured data for node classification tasks. The authors highlight that the bag-level supervision yields weak signals. To address this, they propose to leverage optimal transport and graph penalty terms, with soft pseudo-labels.

### Strengths
1. The paper study an underexplored LLP problem in graph domain.
2. The authors propose to generate pseudo-label based on graph structure inductive bias.
3. The writing is easy to understand.

### Weaknesses
* While the introduction emphasizes LLP's applicability in scenarios where node-level labels are infeasible or undesirable, the experiments rely on standard homophilic benchmarks like citation networks (e.g., Cora, CiteSeer), Amazon product graphs, and WikiCS, which do not inherently reflect these constraints. Labels are readily available in these datasets, potentially undermining the method's real-world validation.
* The proposed method generates pseudo-labels based solely on bag-level proportions and graph structure. Intuitively, for a given bag and its proportion, there is a vast solution space of possible node label assignments that are consistent with both the proportion and the graph's homophily. The paper would be strengthened by a deeper investigation into this ambiguity, such as an analysis of the stability of the pseudo-labels or the sensitivity of the final results to different initializations, which is currently lacking.
* Figure 3, intended to illustrate the Optimal Transport process, is not sufficiently detailed to enhance reader understanding.
* For Theorem 1, the critical homophily assumption is not formally defined in the main text. The statement "Under the homophily assumption..." is vague. 
* The experimental setup follows prior LLP works by using random sampling to create bags. However, in a graph context, random sampling can be detrimental as it may arbitrarily fracture local community structures. 
* Only evaluate a single GCN backbone.
* The code is not available.

### Questions
* More evaluation on data with consistent and reasonable constraints.
* Detailed theorem presentation.
* Reproducible issues.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studied a new problem, i.e. LLP, in graph domain, where we only know the label distributions over each bag of nodes, rather than node-level labels. The authors proposed two-level losses, the first is bag-level KL loss, and the second is node-level supervison, got by OT. Experiments showed good performances of the technique.

### Strengths
1. The authors involved an interesting problem, Learning from Lable Proportion, into graph domain.
2. The theoretical and experimental stuff supports their techniques.
3. The organization is clear and easy to follow.

### Weaknesses
1. Learning from Label proportion sounds more like an industrial scenario. For users' privacy, we have to mask sensitive labels. Also in introduction, the authors mentioned online advertising. But all experments were done on non-industrial datasets. The used datasets, like Cora, Citeseer, never have the requirements of masking labels. So, I suggest the authors to try their techniques on some industrial cases.

2. The authors used optimal transport to get pssudo-labels. OP has a high cost O(N^2). I concern about scalability and efficiency if the authors compute OT every epoch.

### Questions
As in Eq. (9), the cost matrix C is obtained based on logits, i.e., the GNN output. But if GNN is not well-trained, especially at the early stage, the quality of the cost matrix could be low. Why not try to decompose bag-level label proportions into node-level pesudo-signals, and use these signals, rather than logits, to construct cost matrix?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces GLLP, a framework that extends Learning from Label Proportions (LLP) to graph-structured data, where nodes are interdependent. The method employs Optimal Transport with a homophily-aware cost to generate soft pseudo-labels for nodes, enabling effective node classification under distributional supervision. Theoretical analysis and experiments demonstrate that GLLP outperforms existing LLP baselines.

### Strengths
1. Applying LLP to graph structures is a novel contribution.

2. The proposed method is consistent with the theoretical analysis.

### Weaknesses
1. My main concern is the lack of real-world scenarios for the proposed graph LLP problem. The paper does not present convincing real-world applications, and the experiments are conducted only on synthetic datasets.

2. For the proposed graph LLP setting, the paper should discuss more carefully the role of edges — both between different bags and within each bag. Without such analysis, simply applying the LLP framework to graph data is not particularly meaningful.

3. Using only synthetic datasets in the experiments is insufficient to validate the practicality and relevance of the proposed graph LLP setting.

### Questions
on weakness

### Soundness
2

### Presentation
3

### Contribution
2
