# Is Attention All You Need for Temporal Link Prediction? A Lightweight Alternative via Learnable Positional Encoding and MLPs

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
Link prediction is of key importance in many real-world applications like social network analysis and recommender systems. To leverage the expressive power for achieving SOTA performance, many recent works adapt the attention mechanism to the structured data for link prediction, in which dense or relational attention is often unaffordable on large-scale structured data. Moreover, in a realistic setting, the time-evolving topological and feature information can raise more challenging questions about the efficiency and effectiveness of attention mechanisms. In spite of the expressive power, we discern that the attention mechanism may not always be as irreplaceable as expected for temporal graph representation learning, at least not for temporal link prediction tasks. Formally, we discover that some deliberately-designed simple positional encoding can enable MLPs to exploit attributed graph information to achieve SOTA performance than complex graph transformers. Hence, we propose a simple temporal link prediction model, named SimpleTLP. In detail, for SimpleTLP, we first propose to adapt Fourier Transform on temporal graphs for learning informative positional encoding, then we (1) prove this learning scheme can make positional encoding preserve the temporal graph topology from the spatial-temporal spectral viewpoint, (2) verify MLPs can fully exploit the expressiveness and reach and even surpass Transformers on that encoding, (3) change different initial positional encoding inputs to show robustness, (4) analyze the theoretical complexity and obtain less empirical running time than SOTA baselines, and (5) demonstrate its temporal link prediction out-performance in a comprehensive way on 13 classic datasets and with 10 algorithms in both transductive and inductive settings using 3 different sampling strategies. Also, SimpleTLP obtains the leading performance in the large-scale TGB benchmark (the newest TGB 2.0).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper is about temporal link prediction, which is an interesting topic. The authors discover that some deliberately-designed simple positional encoding can enable MLPs to exploit attributed graph information to achieve SOTA performance than complex graph transformers. The paper is well written and well organized. However, there are several concerns in the current version of the paper that addressing them will increase the quality of this paper.

### Strengths
1 Cutting-edge research directions.

2 Clear writing logic.

3 Sufficient experimental results.

### Weaknesses
1 The authors may consider improving some statements in the abstract, especially in the contribution section.  Some contributions only state the process without stating the results (e.g. ‘‘verify the roles of ...’’ without saying what role they played), which is difficult to convey effective information in the abstract.

2 In my understanding, dynamic graphs can be divided into two categories, namely CTDG (Continuous-Time Dynamic Graph) and DTDG (Discrete-Time Dynamic Graph). CTDG corresponds to the concept of “temporal graph”, while the graph with static snapshots mentioned in the paper is actually DTDG. The author should further investigate related work and use professional terms accurately.

3 For this kind of dynamic graph learning based on static snapshots, will it be limited by the number of nodes? One thing that makes me curious is that the author mainly compares small-scale datasets, but also obtains good results on the large-scale datasets of TGB. So how does this learning model based on adjacency matrix adapt to large-scale datasets?

### Questions
As above.

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
This paper introduces a positional encoding for graphs that integrates the graph structure, enabling it to be used with MLP models to outperform graph Transformers in temporal link prediction. The extensive experimental results demonstrate the effectiveness of the proposed positional embedding, even surpassing graph Transformers. This finding could significantly impact future research directions. However, the meaning of Theorem 1 requires clearer explanation, as detailed below.

### Strengths
1. The paper conducts extensive experiments to evaluate the performance, including experiments with multiple baseline methods and experimental settings.
2. Across multiple evaluation scenarios, although the method does not achieve the best results in every case, considering the number of datasets and baselines, as well as the significant improvements in the majority of scenarios, the overall experimental performance of this method is strong. Its notably lower average rank compared to other methods further supports this conclusion.
3. Table 3 is impressive, as it offers new insights that may inspire a shift in understanding for future research in the transformer-GNN field. This could significantly influence subsequent work in this area that the Transformer block not being necessary, where the well-designed positional encoding may be more important than people think.

### Weaknesses
1. The logical relationships in the introduction could be further developed. The introduction does not adequately discuss the significance of positional encoding for the transformer-based temporal GNN methods, nor does it clarify what can be learned. Consequently, the motivation for adding positional encoding to MLPs to level the approach against transformer-based methods presents a conceptual leap that requires further justification.
2. The meaning of Theorem 1 in Section 4 is unclear. First, it is uncertain whether the assumption that the temporal graph \( G \) is 'slowly changing' is valid or meaningful. Even if valid, this assumption greatly restricts the applicability of the method. Second, Equation 14 presents a relationship between the estimates of \( p \) at times \( t \) and \( t'' \); however, its connection to the model’s approximation or learning capability for \( p \) remains unknown. Overall, the significance of this theorem is unclear.

### Questions
1. How is equation (3) designed? What is the relationship of $p^t_u$ and $\tilde{p}^t_u$ during this design for the purpose of prediction and update?
2. What is the relationship of equation (3) and equation (8)? is that mean after the timestamp t is revealed, the positional encoding of (3) is updated following equation (8)?
3. In section 3.2, why is node encoding and link encoding modeled separately? Each interaction involving node \(u\) contains both the node features of \(u'\) and the edge features between \(u\) and \(u'\). What is the rationale for modeling these two aspects as separate variables and what would be the benefit of this approach? 
4. Please explain the theorem 1 mentioned in weakness 2.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The author focuses on link prediction tasks in temporal graphs, which have widespread applications in real-world systems. The authors propose a lightweight method that does not incorporate an attention mechanism: they first learn a positional encoding through discrete fourier transform and then apply a multi-layer perceptron (MLP) to encode these encodings for performance improvements. Additionally, the authors construct a loss function to supervise the positional encoding learning process. Extensive experimental results highlight the effectiveness of their proposed method, providing a thorough illustration of the model.

### Strengths
The paper reveals several strengths that deserve recognition:

* **Interesting Motivation.** The motivation for learning a positional encoding for the target node in temporal graphs, which can capture structural information, is intriguing.
* **Extensive Experiments.** The authors conduct a wide range of experiments to evaluate the effectiveness of their model, particularly across various link prediction settings, metrics, and negative sampling strategies.
* **Well-Written.** The main body of the paper is relatively well-organized and easy to read.

### Weaknesses
However, there are several weaknesses in the paper:

* **Ambiguous Motivation.** The authors claim to “design a module that is better than graph transformers” (Lines 64-65). However, why do you achieve this by finding a learnable positional encoding? Are there existing methods for positional encoding in static graphs? What makes static graph positional encoding insufficient for dynamic graphs? How does your proposed method differ from those used in static graphs?

* **Concerns about ‘Lightweight.’** The authors assert that their model is lightweight due to the absence of an attention mechanism, but **technically**, their model seems to be not appear lightweight. For example, the introduction of $L$ in positional encoding seems to introduce significant additional computation. From Table 18, $L$ appears to be quite large, potentially exceeding the number of samples per node. For a given node, the time complexity of the proposed method for generating its representation is $\mathcal{O}(Ln)$, where $n$ is the number of sampled neighbors. If an attention mechanism were used, the model complexity would be $\mathcal{O}(n^2)$, yet unfortunately, $L > n.$
* **Redundant Experiments.** Although the authors have designed numerous experiments to validate their model's effectiveness, these experiments do not address my key concerns. For instance, a deeper analysis of the important parameters $L$, $\alpha_{neg}$ in Equation 12 and $\alpha_{pe}$ in Equation 13 would be beneficial. Additionally, visualizations of the positional encodings or a case study could enhance the persuasiveness of your motivation.
* **Concerns about Efficiency.** The authors introduce an auxiliary loss to learn the positional encoding, which could significantly impact the model's convergence speed. However, the authors state in Table 10 that their model has a relatively fast convergence speed. What explains this discrepancy? The trade-off between effectiveness and efficiency (the selection of $L$) should be analyzed in the paper.
* Since PINT [1] also improves link prediction performance in temporal graphs through positional encoding, incorporating PINT—whether through empirical analyses or comparative discussions—could enhance your representation.

[1] Provably Expressive Temporal Graph Networks, NeurIPS 2022.

### Questions
Q1: How does your approach build upon or differ from existing methods for learnable positional encoding in static graphs?

Q2: Can you clarify the rationale behind your model's claim of being lightweight, considering the additional computations introduced by $L$?

Q3: What specific insights do you expect to gain from a deeper analysis of the parameters?

Q4: Can you explain the impact of the auxiliary loss on the convergence speed of your model, and how this relates to the overall efficiency of the method?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The authors propose learnable position encoding (LPE) approach and node-link-positional encoding approach: the prior applies discrete Fourier transform on the (spatial) positional embeddings which are essentially the eigenvectors of graph Laplacian matrix for every graph snapshots, while the latter aggregates the information of node features and edge features in the neighborhood. Empirical results show the SOTA performance of the proposed SimpleTLP model.

### Strengths
1. This paper is well written and the experiments are solid. 

2. The theory looks sound, but I did not check it very carefully.

3. The research problem is interesting. When it comes to the information aggregation in the neighborhood, either Mean-pooling or MLP is used.

### Weaknesses
1. The idea of positional encoding and discrete Fourier transform is not new in the field of dynamic graph learning. This paper applies DFT only to the positional embeddings {$\mathbf{p}_u^t$}, rather than to the final representations {$\mathbf{h}_u^t$}. I am curious about the benefits of doing so. 

2. In the viewpoint of graph signal processing, the eigenvectors of graph Laplacian matrix represents the basis of low-frequency signals on the graph [1]. When the graph structures evolve over time, the basis would probably change, then it is not clear to me what kind of information the filter in Eq. (1) can learn. Please remind that the high-frequency noise in for example $\mathcal{G}^{t_i}$ might correspond to low-frequency signal in another snapshot $\mathcal{G}^t_j$ [2].

3. It seems that p_u^t  in Eq.(3) encodes the spatial positional information, h_{u,N}^t encodes the node features, and h_{u,E}^t encodes the edge features and the time intervals between recent interactions. I am not sure whether  h_{u,N}^t is capable of learning sequential patterns in the neighborhood. This is very important for personalized recommendation. To be specific, the sequence order of historical items (i.e., 1-hop neighboring nodes) for a target user (i.e., query node) will reflect her/his future behaviors.

4. Theorem 1 states that the positional embeddings would similar to each other when temporal graph is slowly changing, then one raising concern is whether DFT is necessary.  Personally, I believe that it would be more interesting to see that LPE has the ability to learn abrupt changes.

Minor comments are summarized as follows:

1. It would be better to recall the definitions of $f_T$ and $\mathbf{e}$ in line 247.

2. It seems that the computational complexity of eigenvalue-decomposition is not mentioned in complexity analysis. As far as I know, it is quite expensive to achieve orthogonalized (approximate) eigenvectors. This would largely limit the application of the proposed model. (I read D.6 which has addressed part of this issue.)


[1]  Sampling in paley-wiener spaces on combinatorial graphs. Transactions of the American Mathematical Society.

[2] Sparse representation on graphs by tight wavelet frames and applications. Applied and Computational Harmonic Analysis.

### Questions
Please see the comments above. Thanks.

### Soundness
3

### Presentation
4

### Contribution
3
