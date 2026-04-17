# Native Logical and Hierarchical Representations with Subspace Embeddings

- Decision: Reject
- Scores: 4, 8, 2, 8

## Abstract
Traditional embeddings represent datapoints as vectors, which makes similarity easy to compute but limits how well they capture hierarchy, asymmetry and compositional reasoning. We propose a fundamentally different approach: representing concepts as learnable linear subspaces. By spanning multiple dimensions, subspaces can model broader concepts with higher-dimensional regions and nest more specific concepts within them. This geometry naturally captures generality through dimension, hierarchy through inclusion, and enables an emergent structure for logical composition, where conjunction, disjunction, and negation are mapped to linear operations. To make this paradigm trainable, we introduce a differentiable parameterization via soft projection matrices, allowing the effective dimension of each subspace to be learned end-to-end. We validate our approach on hierarchical and natural language inference benchmarks. Our method not only achieves state-of-the-art performance but also provides a more interpretable, geometrically-grounded model of entailment. Remarkably, the ability to perform logical composition with the learned concepts arises naturally from standard training objectives, without any direct supervision.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors proposed a fundamental shift in embedding methodology: instead of using standard vector representations, one should represent concepts as a linear subspace. They claim this constitutes a paradigm shift in embedding. Specifically, their approach constructs a subspace embedding using orthogonal projections. The authors then attempted to demonstrate that these resulting subspace embeddings exhibit emergent properties regarding logical operations, backing this significant claim with empirical geometrical analysis and experiments.

The motivation behind this paper stems from the fact that dense vector embeddings in Euclidean spaces lack in terms of capturing the directionality and hierarchical relationship among the embedded objects. Furthermore, vector spaces lack any native operators for logical conjunction and negation.  

This paper proposes an alternative where instead of mapping a concept to a single vector, it is embedded as a linear subspace of $\mathbb{R}^d$. In this framework, generality and specificity of concepts get captured through subspace dimensions. The paper defines all the necessary mathematical framework for this proposed new approach. Lastly, they demonstrate the efficacy of the proposed approach for three different tasks - WordNet reconstruction, WordNet Link prediction, and SNLI. 

While motivation is fully justified and the proposed approach is also novel but my fear is that almost identical conceptual framework was proposed in a series of papers related to Quantum Embeddings couple of years back. I would like authors to have a careful look at these because the ideas proposed in these papers is quite closely matching what is proposed here. 

- https://papers.nips.cc/paper_files/paper/2019/file/cb12d7f933e7d102c52231bf62b8a678-Paper.pdf
- https://proceedings.neurips.cc/paper_files/paper/2020/file/b87039703fe79778e9f140b78621d7fb-Paper.pdf

Authors needs to highlight a very clear and strong differentiation as well as justification about their work in light of the above prior art.

### Strengths
- The motivation behind the paper and proposed idea is novel but unfortunately, almost identical idea was proposed as part of Quantum Embedding work couple of years back. 
- The idea of soft projection operator is nice and gives an handle to manage non-differentiability of projection operator.
- The metric NIS defined to quantify the subspaces similarity is also nice.

### Weaknesses
1. I find it hard to agree with the claim that the proposed method—which uses orthogonal projections to derive subspace representations of language—is novel or a paradigm shift. The authors seem to have neglected a proper literature survey on hierarchical representation across language and knowledge bases. It's a well-established technique in the field to use orthogonal projections and their properties for building hierarchical language representations. See the works of 1) Garg et. al, NeurIPS 2019 on quantum embedding of knowledge for reasoning (https://papers.nips.cc/paper_files/paper/2019/file/cb12d7f933e7d102c52231bf62b8a678-Paper.pdf) , 2) Srivastava et.al NeurIPS 2020 on inductive quantum embedding (https://proceedings.neurips.cc/paper_files/paper/2020/file/b87039703fe79778e9f140b78621d7fb-Paper.pdf).  

2. Authors discussed the properties about projection and its smooth approximation. However, I failed to understand clearly how they are constructing the subspace representations. May be the subsection “Subspace Projection Head” is not written clearly. What is the input, what are all the intermediate steps, and what is the output?

3. My initial interpretation is that the authors take a sequence representation from a transformer model, enclose the resulting vectors within an unbounding box, and define this as the input sequence's subspace. If this is the case, I do not understand how the system is learning an orthogonal (smooth) projection matrix for an input statement from the ground up. This learning process is essential to the claimed properties but is missing from my understanding of their proposed construction.

4. I believe the experiments are not exhaustive enough to convincingly demonstrate that their method surpasses previous state-of-the-art results. To strengthen their claims, they should take the following steps: First, to prove that the LLM's representation is not hierarchical, they must compare their results with the latest LLM. Second, they should perform more complex reasoning tasks to persuasively demonstrate the emergent properties of logical operations. Third, they should test their methods on reasoning tasks that utilize preferably a large size knowledge bases such as Wikidata (18 billion triples) or OpenStreetMap.

### Questions
- The section on Subspace Projection Head (SPH) is crucial section but the details are relatively less clear. This section can be elaborated a bit more in my view for conveying the idea bit more clearly. My understanding is that every piece of text would be first converted to matrix H which is then converted into matrix X whose columns span the corresponding subspace?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents a new alternative to the traditional approach of representing data points as embeddings in a vector space. Specifically, the main idea is to embed concepts as linear learnable subspaces. This allows a much richer representation of concepts such as hierarchy and can natively represent logical operations such as conjunction, disjunction, etc as linear operations. The key technical contribution is the learning method to enable subspace representation of concepts where instead of fixing the embedding dimensions a set of  vectors of varying importances are learned through soft projection making subspaces flexible to select vectors as needed for their representation. To do this, they define a soft projection approximation where the dimensionality changes smoothly and is thus differentiable to be learned through gradient-based methods. The approach is also extended to transformer models and end-to-end learning for reconstruction, link prediction and NLI. A comprehensive set of experiments are performed on these tasks comparing them with state of the art methods in each.

### Strengths
+ Novel approach for representation learning, the idea of subspace learning naturally fits into more interpretable logical operations compared to existing approach. Given the generality of the formulation, this type of learning could turn out to be highly significant in several applications.
+ Comprehensive empirical validation that shows the generality of the approach in various tasks. The proposed method outperforms state of the art models in word net reconstruction, link prediction and NLI. The results also show the ability of the approach to represent meaningful hierarchies and logic compositionally using multimodal retrieval.

Overall, this seems to be strong paper with a novel, well-defined formalism that is general purpose and empirical results that strongly validate the claims.

### Weaknesses
Weakness
- Not a weakness as such but the paper does not spell out what and if there are limitations of the new representation. Scalability is one possible limitation perhaps (section 6 talks about this briefly) compared to standard embeddings. But in general, it would be nice to know about the trade-offs being made to achieve learning that is more semantically rich.

### Questions
Can you comment on the choice of baselines in the NLI tasks, are they considered state-of-the-art. The performance improvements in NLI seemed lower than the others. I would assume due to the nature of the subspaces it should be much better in answering logical queries such as entailment. Do you have some comments on this?

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to embed concepts as linear subspace, and introduces algebraic computation to embed set-theoretic relations between concepts, namely, intersection, union, and negation. Authors introduce a smooth relaxation of orthogonal projections to learn both subspace orientation and dimension through gradient descent, and experimented with well-known datasets, e.g., WordNet, NLI benchmarks, achieving state-of-art performances.

### Strengths
The proposed embedding method increases the interpretability of logical operations and concept structures in neural networks. The traditional semantics of logic and sets is explicitly represented in geometric entities.

### Weaknesses
This version of the proposed work still suffers from both technical and theoretical clarities. It seems that authors mix entities and propositions, and this will introduce problems. For example,  authors embed “man on a boat” as a concept and embedded as a single direction, then, map it to x1 and x2, where x1 might represent a“man on a boat that is fishing” while x2 might represent “man on a boat that is not fishing”. In this way, the concept “man on a boat” is represented by the subspace span(x1, x2). This raises the uniqueness problem. The “man” on a boat can also sit, stand, sleep, jump, …. 

Authors target precisely embedding set-theoretic and logical concepts; they shall evaluate whether they can introduce determinacy and rigour into neural networks, but in experiments, they slip back into popular statistical evaluation and only achieved state-of-the-art performance.

### Questions
What can be called “concept”? 

If the concept “man on a boat”, why not represent it as a topological relation ("on") between two the “man” subspace and a “boat” subspace?

How can we use the proposed embedding method to represent distance and orientation relations between objects?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes representing linguistic concepts as learnable linear subspaces rather than fixed-dimensional vectors. Specifically, the authors argue that this paradigm enables one to naturally capture concept complexity through subspace dimensionality, inter-concept hierarchies through subspace inclusion, and multi-concept logical operations (conjunction, disjunction, negation) through simple algebraic manipulations. More importantly, this work introduces an efficient way of learning these subspaces in a differentiable manner. The proposed representation learning approach is evaluated, both quantitatively and qualitatively, on several NLP tasks. These experiments strongly suggest that the proposed paradigm outperforms competing state-of-the-art representation learning approaches while offering distinct new benefits such as naturally capturing inter-concept relationships and complexities.

### Strengths
Thank you so much for submitting this work! I enjoyed reading this paper and learned a great deal from it. Below are what I believe are this paper’s main strengths:

1. **[Originality, Critical]** The proposed paradigm (i.e., representing concepts as subspaces rather than fixed vectors), the used learning algorithm (i.e., a mechanism for learning different rank subspaces from the data), and the properties that derive from using this paradigm and learning process (i.e., naturally composable and hierarchical representations) are all, to the best of my knowledge, novel and interesting proposals. Because of this, I strongly believe the sheer originality of this paper is very strong.
2. **[Significance, Critical]** This work introduces an elegant geometric framework that naturally enables representations to capture hierarchies (inclusion), complexity (dimension), and logical compositions (i.e., via linear operations). These are all highly desirable properties that, as the authors correctly argue, are known to be problematic/lacking in existing representation learning frameworks. Therefore, I believe that the main ideas proposed in this paper have the potential to have a high impact across various areas in AI. In particular, I find the proposed differentiable mechanism for learning distinct ranks for different subspaces to be extremely interesting and potentially applicable to a wide range of representation learning tasks.
3. **[Quality, Major]**  The evaluation of the proposed method is extensive, spanning everything from traditional representation learning tasks to qualitative analyses and more interesting experiments that showcase novel features that emerge from the proposed paradigm. Moreover, the results suggest that the proposed methodology improves upon the SotA performance on key representation learning tasks (e.g., WORDNET reconstruction) while it provides a more interpretable, geometrically-grounded model of entailment. All of these are very strong forms of high-quality experiments that strongly suggest that the proposed methodology "works" as intended.
4. **[Clarity, Major]** The paper is very well-written and carefully motivates every key decision. Moreover, the supplementary material includes the code used for this paper, and the appendix contains concise and clear proofs of critical theoretical results/considerations that appear in this work.

### Weaknesses
In contrast, I believe the following are some of this work’s limitations:

1. **[Significance, Major]** Against standard good scientific practices, the empirical quantitative experiments in this work do not include any error bars. This makes it difficult for one to judge the significance of any observed differences and sets a negative precedent for not following good experimental conventions.
2. **[Quality and Clarity, Minor]** It is unclear how certain hyperparameters like $ \lambda $ are selected and what their impact is on the observed results. 
3. **[Quality and Clarity, Minor]** Although there is a short discussion on the effectiveness of the proposed paradigm, it is unclear how this manifests itself in practice in terms of quantities that are practically important (e.g., memory consumption, wall-clock training times, etc.).
4. **[Quality, Minor]** There is no discussion of any limitations of the proposed framework anywhere in the paper.

### Questions
Balancing the contributions and strengths of this paper with respect to the weaknesses listed above, I am leaning towards accepting this work. This is because I believe this is a well-written piece of work that proposes a very interesting, elegant, and practical paradigm, whose impact could be significant. Nevertheless, it is worth mentioning that my main area of research is not specifically in language representation learning, so it is possible I might’ve missed some important distinctions/previous works that this paper failed to consider (hence my relatively low confidence). That being said, I have included some questions below that could help improve the quality of this manuscript. If these concerns are properly addressed, particularly my concerns regarding the lack of error bars in the quantitative experiments, I would be more than happy to consider updating my score.

1. **[Major]** Could you please provide error bars for the empirical quantitative results? Are the improvements in SE's results significant?
2. **[Major]** How was the hyperparameter $\lambda$ chosen? What effect does this hyperparameter have on the observed results and stability of the training procedure?
3. **[Minor]** How does the learning time and memory usage of SE/SPH compare to that of competing baselines (in practical terms, such as wall-clock times and peak memory consumption rather than asymptotically)?
4. **[Minor]** Could you please describe some key limitations of SE? If possible, I would strongly recommend adding these limitations to the paper.
5. **[Minor]** Why is $\mathbb{R}^{10}$ used for the Euclidean baseline for the Link Prediction experiments rather than $\mathbb{R}^{128}$ to match the maximum rank of SE? Is that a fair comparison?

### Minor Suggestions and Typos

1. **[Potential Typo, Nit]** There are missing parentheses in the citations in lines 76 and 376.

### Soundness
3

### Presentation
4

### Contribution
4
