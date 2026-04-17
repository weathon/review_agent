# Wasserstein Hypergraph Neural Network

- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
The ability to model relational information using machine learning has driven advancements across various domains, from medicine to social science. While graph representation learning has become mainstream over the past decade, representing higher-order relationships through hypergraphs is rapidly gaining momentum. In the last few years, numerous hypergraph neural networks have emerged, most of them falling under a two-stage, set-based framework. The messages are sent from nodes to edges and then from edges to nodes. However, most of the advancement still takes inspiration from the graph counterpart, often simplifying the aggregations to basic pooling operations. In this paper we are introducing Wasserstein Hypergraph Neural Network, a model that treats the nodes and hyperedge neighbourhood as distributions and aggregate the information using Sliced Wasserstein Pooling. Unlike conventional aggregators such as mean or sum, which only capture first-order statistics, our approach has the ability to preserve geometric properties like the shape and spread of distributions. This enables the learned embeddings to reflect how easily one hyperedge distribution can be transformed into another, following principles of optimal transport. Experimental results demonstrate that applying Wasserstein pooling in a hypergraph setting significantly benefits node classification tasks, achieving top performance on several real-world datasets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces the Wasserstein Hypergraph Neural Network, an architecture designed to overcome a key limitation in existing hypergraph neural networks. WHNN proposes a conceptual shift by viewing these neighborhoods as empirical samples from underlying probability distributions. To capture their geometric properties, the model employs Sliced Wasserstein Pooling as its core aggregation mechanism. This design creates embeddings where the Euclidean distance approximates the Sliced Wasserstein distance between the original distributions, thus preserving structural information like shape, spread, and density. Implemented within a standard two-stage message-passing framework, the authors demonstrate that WHNN outperforms traditional aggregation methods and achieves state-of-the-art performance on multiple node classification benchmark datasets.

### Strengths
Principled Aggregator Choice: By adopting the distributional perspective, the choice of Sliced Wasserstein Pooling as an aggregator is well-justified. It endows the model with a powerful inductive bias to capture the shape, spread, and density of feature distributions within hyperedges, a capability that mean-based aggregators lack.   

Thorough Empirical Validation: The authors conduct a rigorous evaluation across ten datasets, comparing against eight strong baselines. The inclusion of comprehensive ablation studies effectively isolates the contribution of the Wasserstein aggregator, providing convincing evidence for the paper's central claims

### Weaknesses
1.	Theoretical Grounding not thoroughly discussed: the paper critiques sum-based aggregators like Deep Sets. Deep Sets is known to be a universal approximator for permutation-invariant functions defined on sets. The authors provide no such theoretical analysis for WHNN. They rely on the properties of SWP established in prior work, stating that their embedding approximates the SW distance, but they do not analyze what this approximation implies for the expressive power of the full HGNN model. This leaves several important theoretical questions unanswered. Does WHNN retain universal approximation capabilities? How does the error introduced by the SWP approximation propagate through multiple layers of message passing? Is the model guaranteed to distinguish between any two non-isomorphic hypergraph neighborhoods - is the aggregation function injective? The literature on SWP embeddings has explored properties like injectivity and bi-Lipschitz continuity, but the authors do not connect these foundational concepts to their hypergraph model. Without this theoretical grounding, the paper's contribution feels more like an engineering result rather than a fundamental advance in understanding hypergraph representation learning.

2.	Potentially Overstated Claims and Interpretation of Results: The paper claims to achieve "top performance" and "top results" across all datasets. While technically true based on the mean accuracy reported in Table 2, this language can be misleading when the margin of victory is small and falls within the standard deviation of a competing method. For example, on the NTU2012 dataset, WHNN-MLP scores 90.87 $\pm$ 1.59, which is statistically very close to ED-HNN's 89.48 $\pm$ 1.87. A more measured tone would be appropriate. More concerning is the interpretation of the ablation studies. The authors initially motivate SWP as a way to reduce the learning burden on the encoder. However, their own results in Figure 3 show that SWP's performance also significantly improves when paired with the more complex SAB encoder. This suggests the initial premise may be a false dichotomy. A more nuanced conclusion, better supported by the data, is that powerful encoders and expressive aggregators are complementary, and their combination is what yields state-of-the-art results.

3.	Omission of Computational Cost and Practicality Analysis: The paper's central claim revolves around achieving superior classification accuracy, but it entirely neglects to discuss the associated computational costs. The literature on SWP confirms its computational overhead, with a complexity of $O(LM \log M)$ for a set of size $M$ and $L$ slices, and often requires a large number of slices ($L$) to be effective. The paper's own complexity analysis in Appendix D confirms this super-linear dependency on neighborhood size.

### Questions
1. Check the first point of Weaknesses, some theoretical proofs should be shown in the paper.
2. The Wasserstein aggregator appears significantly more computationally intensive than the simple summation used in baselines like ED-HNN. Could the authors please provide empirical performance metrics for WHNN? This would allow for a fair assessment of the accuracy-efficiency trade-off.
3. Given that the performance gains are sometimes modest (e.g., $\approx 0.4\%$ absolute improvement over ED-HNN on Cora), how do the authors justify the additional complexity of SWP for practical applications where inference speed or training cost are critical constraints?
4. The ablation studies show that both the baseline aggregators (Deep Set, PMA) and the proposed SWP aggregator benefit from a more powerful encoder. This seems to suggest that the initial motivation—that simpler aggregators uniquely "move the complexity... to the initial encoding"-may be an oversimplification. Could the authors please clarify or provide a more nuanced interpretation of these results? It appears a more general conclusion is that better encoders are complementary to better aggregators.
5. Wasserstein distance is extensively used in AI and machine learning. Why this WHNN adds new values?

### Soundness
3

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
This paper introduces the Wasserstein Hypergraph Neural Network, which models nodes and hyperedge neighborhoods as distributions and aggregates them using Sliced Wasserstein Pooling. Unlike traditional mean or sum aggregators, this approach preserves geometric properties of distributions, capturing richer higher-order relationships. Experiments show that it significantly improves node classification performance on real-world hypergraph datasets.

### Strengths
1. Although the paper addresses a traditional GNN problem, it takes a very interesting perspective. Instead of using conventional aggregators like mean or sum, the authors represent the nodes within a hyperedge as a distribution. This approach better captures the internal structure and higher-order relationships of hyperedges.
2. The writing of the paper is very logical and well-structured. Although it involves some theoretical concepts, the explanations are clear and easy to follow.
3. This paper presents an innovative application of optimal transport theory in the context of GNNs.

### Weaknesses
The main drawback of the paper is the lack of complexity analysis and runtime comparisons. Since the optimal transport algorithm is generally not cheap computationally, this raises some concerns about the efficiency of the proposed method.

### Questions
1. This part feels vague in terms of its advantages—could it be presented in a more systematic way? 

“For example, a hypergraph containing two clusters of nodes suggests a bimodal underlying distribution. On the other hand, a hyperedge where nodes are close in the feature space denotes a unimodal probability distribution, suggesting a homophilic behaviour. A hyperedge in which nodes have similar representations indicates a low-variance distribution, while a hyperedge with diverse nodes suggests a more uniform distribution (see Figure 1).”

2. Why was optimal transport chosen to address this problem? Could other methods for measuring distances between distributions not work as well?

3. Please include an analysis of time complexity and a comparison of the algorithm’s runtime efficiency.

P.S. While I am generally favorable toward this paper, as I find the idea of modeling hyperedges as distributions very appealing, I would be more cautious and might consider a weak accept if the authors cannot demonstrate acceptable computational efficiency.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Wasserstein Hypergraph Neural Network (HNN). Whereas most existing HNNs use sum-based pooling to smooth information during message passing, the proposed method uses Wasserstein aggregation to preserve geometric information of node feature distribution within each hyperedge. The proposed method outperforms the baseline HNNs in several node classification benchmark datasets.

### Strengths
The key strengths are two-fold:
- The key idea is simple, intuitive, and solid. Existing set pooling functions may struggle to capture the full geometry of set-structured inputs, and using a more expressive pooling function to improve HNNs is a natural approach. 
- The approach is also somewhat original to hypergraph learning. Intuitively, node feature distribution within each hyperedge could be distinct and informative for hypergraph learning.

### Weaknesses
I have five primary concerns:
- [limited novelty]: The key technical innovation is adapting Sliced Wasserstein Pooling (SWP), an existing module published in 2021, for HNNs. While the adaptation makes sense and seems to be performant, without a new technical innovation, the paper does not meet the standards of a top-tier conference.
- [thin content]: The experiments and analyses are thin. The authors used only 7 benchmark datasets with 8 (old) baseline HNNs. The method is evaluated only based on node classification performance. All the benchmark datasets have high label-homophily. No large dataset was tested. No formal or empirical analysis of the proposed method was provided. In fact, in a 9-page paper, the authors spent 3 pages solely on related work and background, which I do not think is generally acceptable.
- [weak method justification]: I do not agree with the authors' claim that the superior performance of the proposed method is due to 'capturing geometric information' of node features within each hyperedge. In most of the benchmark datasets used, the average hyperedge size is very small, e.g., 3-4 in Cora and Citeseer. Moreover, their input feature vectors are mostly sparse, multi-hot vectors. That is, I do not think such informative 'geometry' can be robustly identified in the benchmark datasets used. In fact, the authors did not provide any empirical analysis of the 'geometry' in the benchmark datasets. This drawback leaves the authors' key claim 'that this geometric information is highly relevant for hypergraph learning' (line 069) only speculative.
- [unfair hyperparameters]: The authors used self-loop addition as part of the tuned hyperparameters for each dataset. This is atypical and unfair. All the baseline methods, while they could, did not have the self-loop addition as their hyperparameter. This makes the empirical superiority of the proposed method suspicious.
- [copyright]: The baseline performance table is copied from the AllSet paper. This is okay, as long as the authors clearly mention that in the paper. However, I could not find it anywhere. This is a research ethics issue. Please revise properly. 

I have other, less significant concerns:
- [missing details]: Details about the synthetic case in Figure 1 are missing, making it unclear whether the case is illustrative or based on formal analysis. 
- [unclear writings]: Some writings are unclear. e.g., I do not understand 'This way, the hyperedges are not only characterised by the combination of their elements, but by the regions of the space where their elements are situated. The nodes became prototypes of the hyperedge behaviour.' Moreover, the authors write 'more sensitive to the geometric structure of the hyperedge' (line 228-229). However, if I understood correctly, they are not sensitive to the structure 'of hyperedge' but of 'node feature distribution within a hyperedge'. Please clarify.
- [unclear figures]: Some figures are quite confusing. E.g., in Fig. 2, what do node colors mean? What does the bar plot aim to depict? What do the contours aim to visualize? Adding legends, labels, and annotations may help.
- [minor errors]: Algorithm 1's line 4 is empty

Overall, in the current version, the paper is more like a workshop paper, with a simple, new, and promising idea and preliminary empirical outcomes.

### Questions
See weakness

### Soundness
3

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Paper is interested in modeling neural networks for Hypergraphs: a generalization of graphs where any edge can connect zero-or-more nodes. Prior neural networks on hypergraphs follow message-passing paradigm, where each edge updates itself as an *average* of (a transformation of) its incident (endpoint) node features, then each node updates itself with an *average* of its incident hyperedges. This *average* (or sum) pooling captures only a simple statistic: the average. The paper pictorially shows why average is insufficient for representing a node. Suppose a graduate course with student average of 80, where most students receive a grade around the average. Suppose course with bimodal grades (centered at 90, and at 70). While both courses have a similar average, i.e. modeled the same with most (earlier) models, they should be represented differently. Paper rather tries to represent nodes "using distribution of edges" (and vice-versa: represent edges using distribution of nodes). Distance between 2 distributions is quantified using Wasserstein (W) distance. While W is intensive to compute for distributions over multiple dimensions, it is much easier to compute it for probability distributions over one (scalar) variable. (as sum of distances of inverse of CDF), allowing one to approximate the distance between 2 distributions (over higher dimension) by taking many random direction lines, and projecting the distributions upon these lines i.e. break-down the problem into many 1D W-distances, and average them, to approximate the distance in higher dimensions. Further, rather than calculating the approximate distance (even in the 1D case), paper mentions that they can project onto a latent space where euclidean-distances are trained to approximate W-distances.

Sorry for the long description :)

All-in-all -- I like the paper a lot. However, there are two main things that need to be addressed for acceptance consideration (IMO):
1- Clarity around the math / algorithm (see Weaknesses).
2- Novelty. There are Wasserstein GNNs (https://arxiv.org/abs/2102.03450)

### Strengths
* Paper is well-written. I learned a bunch, as perhaps apparent from the above summary. It provides both good mathematical understanding (step-by-step walkthrough), as well as demonstration figures in main paper and Appendix.
* The paper utilizes different grounds-up concepts, which is "refreshing" in LLM era.
* Good experimental results

### Weaknesses
## main weakness
The main weakness is in the clarity of the method. Not in terms of the write-up/motivation/high-level description (I think those are clear), but in terms of actually implementing or replicating this method. The math/algorithm section has loose ends, and personally, while I appreciate this work, I am unable to implement it from reading this paper alone. Great papers, on the other hand, I am able to implement them by reading only those papers. Let's try to improve this paper from "Good" to "Great" in order to have it appropriate for ICLR.

Specifically,

* what is the "interpolation" mentioned in Step 2 (Page 7) or line 12 in Alg.2. It seems that it will discard entries if there are too many, or invent new entries if there are only a few of them. I would guess that inventing could imply taking average of existing points?

* Line 5 of Alg2: Does it expand the dimensionality of X?
* Line 7 of Alg2: what are the vectors (reference points) sorted by? by their CDF value?
* Line8 onwards of Alg2: what is the lowercase symbol $s$? Of course, upper-case $S$ is the set of neighbors -- does lower-case $s$ correspond to the center (node or edge) for which its neighbros are recorded in upper-case $S$? If so, perhaps consider looping $(s, S) \in \mathcal{N}.items()$ -- or something similar. We are lucky that most ML folks are able to read pythonic-pseudocode.
* Line 18 of Alg2: is $W$ vector of scalars? In general, **please add sizes of all tensors**. Does this line sum-over a dimension? (weighted sum, with weights present in vector $W$)
* Alg1 line 7 or Step 1 on page 7 -- what is the reference distribution $q$? does it contain some anchor (randomly-sampled) nodes/edges? Do they have to go through the encoder (Line 11)?


## Novelty

Have you seen: https://arxiv.org/pdf/2102.03450 ? In what way does your method differ? Is it only in the application (Hypergraphs VS graphs)? or is the entire construction different?


## minor weakness:
* Typo: Perhaps replace "inner integral" with "integral" on line 201? I get that this becomes "inner" once plugged-into Eq.2 but perhaps the text is confusing as word "inner" shows up before Eq.2

### Questions
* Are the nodes of the hyper-edges ordered (i.e. order of nodes matter, as a generalization of directed graphs)? The math does not seem ordered. But what if the data is ordered? How can one extend the model to handle this data? E.g., (<Student> studied <program> at <university>)
* Please answer the questions in weaknesses, importantly, by updating the paper.

### Soundness
3

### Presentation
3

### Contribution
3
