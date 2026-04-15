# Filtered Semantic Search via Vector Arithmetic

- Decision: Reject
- Scores: 5, 3, 3

## Abstract
How can we retrieve search results that are both semantically relevant and satisfy certain filter criteria? Modern day semantic search engines are increasingly reliant on vector-based search, yet the ability to restrict vector search to a fixed set of filter criteria remains an interesting problem with no known satisfactory solution. In this note, we leverage the rich emergent structure of vector embeddings of pre-trained search transformers to offer a simple solution. Our method involves learning, for each filter, a vector direction in the space of vector embeddings, and adding it to the query vector at run-time to perform a search constrained by that filter criteria. Our technique is broadly applicable to any finite set of semantically meaningful filters, compute-efficient in that it does not require modifying or rebuilding an existing $k$-NN index over document vector embeddings, lightweight in that it adds negligible latency, and widely compatible in that it can be utilized with any transformer model and $k$-NN algorithm. We also establish, subject to mild assumptions, an upper bound on the probability that our method errantly retrieves irrelevant results, and reveal new empirical insights about the geometry of transformer embeddings. In experiments, we find that our method, on average, yields more than a 21% boost over the baseline (measured in terms of nDCG@10) across three different transformer models and datasets.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper is on (vector) embedding-based retrieval. Suppose we have a pre-trained (text) embedding model. Suppose we have a document corpus and assume that each document has a label (such as the \textit{color} or \textit{brand} of a product). Given a (query, label), the problem is to fetch documents which are relevant to the query and with the specified label. 

The authors work in the setting where they intend to use a generic embedding model to embed the document corpus. The key proposal is to learn $d$-dimensional representations for labels (where documents are also represented in $d$-dimensional Euclidean space). The authors assume that the set of labels is small — much smaller than the corpus size. Therefore, the number of additional learned parameters is small (compared to the number of parameters in the frozen embedding model).

This is how they learn the linear probe $\nu_f$ for label f. Each document $d_i$ is assumed to have a label $y_i$. They learn a $d$-dimensional vector $\nu_f$ for each label value $f$ such that $\nu_f \cdot v_{d_i}$ is close to $\mathbf{1}\{y_i = f\}$. They minimize the squared loss $\sum_{i=1}^T (\nu_f \cdot v_{d_i} - \mathbf{1}\{y_i = f\})^2$ where $T$ is the training set size. 

Now given a query and label $(q, f)$, they perform nearest-neighbor search in the document corpus for the search vector
$$
v_{q+f} + \lambda \nu_f
$$
after unit l2-normalization, where $\lambda$ is tuned by cross-validation. Note that the document embeddings are not modified, it is only the search vector that is modified, by the procedure. So crucially, this procedure can be applied for multiple label sets (filter sets) with the same document embedding corpus.

The proposed method performs significantly better than natural baselines, with no consequential increase in serving cost.


==== UPDATE AFTER THE REBUTTAL =====

There are papers like https://openreview.net/pdf?id=wLFXTAWa5V (pointed to by Reviewer BEgY) which address the same problem of filtered ANNS but with experiments on much larger datasets than those in the submission. In light of this, I believe that the authors need to demonstrate that their method scales. I am reducing my rating now, but encourage the authors to resubmit to a good conference after scaling up experiments and comparing with the methods in literature.

### Strengths
* The retrieval quality improvements in Table 2 are significant.

* The method is straight-forward to implement. (This is a strength.)

* Intuitively it makes sense to rank documents by the weighted sum of two scores: (i) $v_{q+f} \cdot d$: a general query document dot-product (ii) $\nu_f \cdot d$: which is trained to be close to 1 if the document is likely to have label $f$.

### Weaknesses
* The datasets in experiments are small (Table 1). Unfortunately, I do not have pointers to larger datasets.

### Questions
* In Table 2, the performance of sgpt seems to be worse than the mpnet for colors and brands; even though from the model sizes (with similar training) one would expect the opposite.

* Further, the baseline numbers are similar between the three base models. What explains this?
How is the hold-out set prepared for experiments documented in Table 3?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper investigates how do efficient, vector-based nearest neighbor search given various filter criteria. The authors theoretically and empirically explore a linear probing-based approach. They show strong empirical gains using the proposed method compared to a particular baselines.

### Strengths
The paper explores a nuanced and import problem in nearest neighbor search, namely search for nearest neighbors meeting some filter set criteria. I believe that this is problem can have very good impact on both academic and industry research communities. 

The strengths of the paper include:  
* Well performing, simple, scalable approach to filtered-based nearest neighbor search
* Well presented methodological approach, clearly demonstrated through intuitive examples
* Grounding in theoretical statements

### Weaknesses
I am an advocate for the setting of the paper as well as the simplicity of the approach. 

However, I feel that the paper falls short in a few key aspects:

* First, given the simplicity of methodology of the approach (which I generally support), I would have expected more analysis in comparison to Filtered-DiskANN and other alternative approaches. Understanding why and when to use which approach is key for having impact, especially with practitioners. I think the authors could have more clearly outlined pros & cons between their approach and Filtered-DiskANN and further shown a more complete setting of experiments to demonstrate this. 
* Further, I am misguided, but I feel that the method is on the straightforward side, e.g., it is what many researchers may have wanted to try for this problem. In some ways that is a kudos for the authors for trying this, but in others, I would have expected much more explanation about the approach as it relates to classical methods like PCA, etc.
* However, my major concern about the work is that, the datasets (Table 1) are much to small by modern standards for us to get a sense of how the method scales. 

Minor:
* No conclusion
* Various issues with citation parens
* There is certainly work earlier than 2009 for IR (line 116)

### Questions
* Can you say more about how you expect the method to scale w/ larger corpora?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors introduce the problem of retrieving items for queries with preference filters. They propose a simple yet effective solution of representing the resultant filtered query representation as weighted sum of the corresponding vectors. Additionally, they provide a theoretical analysis of the performance guarantees of their method under specific distributions of relevant and non-relevant items.

### Strengths
- Faceted search is an interesting and important problem, and the novelty of this work lies in its approach to addressing it.
- The authors present a straightforward solution and set up an experimental framework to test their method's effectiveness in solving the filtered search problem.

### Weaknesses
- The evaluation domains are quite limited, and some of the curated data is synthetic in nature.
- Using vector summation to produce a composite representation is a common approach. Given this, I feel the paper doesn’t present a novel contribution in representation learning. However, with its unique problem formulation and with additional analysis, along with broader testing across domains, it could still be a fit for data mining and information retrieval conferences.
- If the assumptions regarding the relative distribution of documents and query representations do not hold, the method may fail to perform effectively. Given the existing literature on the anisotropic behavior in embeddings, I am cautious about overlooking this concern.
- The theoretical claims appear to be self-evident. The authors assume that relevant and irrelevant documents follow specific distributions, then design the representation as the distribution mean with a separability threshold based on variance. Naturally, this setup ensures separability; thus, the analysis seems to add little value. It would be more compelling if the authors showed that the method performs within an ϵ-optimal range even when data distributions deviate from these assumptions.

### Questions
The weakness themselves can be considered as questions.

**Additional Questions and Suggestions:**

- Of the following three contributions, which do the authors consider most central to their work?
    - Identifying the problem of filtered search and providing a benchmark for it
    - Proposing a non-trivial method to solve the filtered search problem
    - Offering theoretical guarantees for the proposed method
    
    The paper touches on aspects of each of these contributions, but none are explored in depth. I suggest enhancing one or more of these areas and restructuring the paper with a clearer, more focused motivation. While focusing on all of them is a good idea, but the paper reads very incomplete without going into depth in some of them.
    
- Do the authors believe similar ideas could extend to other domains? The current filters seem somewhat synthetic—additional experiments on real-world data would strengthen the work.
- Could you test alternative, simple baselines?
    - For instance, multiply the scores of the query and filter for each document, such that
        
        p(q+f,d)=p(q,d)×p(f,d).
        
    - Another approach could involve converting  q+f into natural language queries using an LLM, then feeding the output text to the encoder to generate the query + filter representation

### Soundness
2

### Presentation
2

### Contribution
2
