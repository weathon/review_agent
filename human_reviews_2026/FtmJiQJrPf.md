# INSES: Intelligent Navigation and Similarity Enhanced Search for Knowledge Graph Reasoning

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
GraphRAG is increasingly adopted for converting unstructured corpora into graph structure, enabling relational, multi-hop reasoning beyond chunk-level retrieval. Most systems then reason over these graphs with classic graph algorithms. However, such traversal, tied to static connectivity and 'connected triple' paths, frequently misses latent semantic links in real-world knowledge graphs (KG) that are noisy, sparse, or incomplete. To address this gap, we introduce INSES (Intelligent Navigation and Similarity Enhanced Search), a dynamic graph-reasoning framework that couples LLM-guided navigation, which prunes noise and steers triple selection with embedding-based similarity expansion to recover hidden links and bridge gaps beyond explicit edges, turning search from a purely structural walk into semantics-aware multi-hop reasoning. Additionally, since GraphRAG style search generally incurs higher complexity than naïve RAG, we complement INSES with a lightweight router that sends simple queries to naïve RAG and escalates complex multi-hop cases to INSES, balancing efficiency and reasoning depth. Across multiple QA benchmarks, INSES consistently outperforms SOTA RAG and GraphRAG baselines. Results highlight complementary strengths of coarse-grained text retrieval for easy cases and fine-grained triple reasoning for harder ones. On the MINE benckmark, INSES remains robust across KGs produced by KGGEN, GraphRAG, and OpenIE, improving accuracy by 5\%, 10\%, and 27\%. This work opens the door to adaptive, router-backed KG reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Intelligent Navigation and Similarity Enhanced Search (INSES), a dynamic graph reasoning framework designed to improve GraphRAG-style pipelines. INSES combines three ideas:
(1) LLM-guided navigation that prunes irrelevant edges and selects promising triples,
(2) similarity-based expansion that adds edges between semantically similar nodes to mitigate graph incompleteness, and
(3) a router that dispatches simple queries to lightweight text-RAG and complex ones to INSES for efficiency.

This paper evaluate INSES on several multi-hop QA benchmarks (MuSiQue, 2Wiki, HotpotQA) and on the MINE dataset, showing modest but consistent gains over existing text- and graph-based RAG methods.

### Strengths
1. The paper identifies a limitation of existing GraphRAG methods that strict reasoning reliance on the explicit graph edges.
2. The combination of LLM-guided traversal, embedding-based expansion, and routing is modular and intuitive.
3. Multiple datasets, detailed prompts, and case studies are provided, improving clarity and reproducibility.

### Weaknesses
1. The performance gains are small (1-2 % improvement) without multi-run averages or standard deviations are reported. It is unclear whether the improvements exceed the stochastic variance inherent in LLM outputs, making it difficult to judge statistical significance.
2. There is a lack of computational and complexity analysis. The similarity-based expansion appears to require pairwise similarity checks among candidate nodes, which is $O(N^2)$ in the worst case. The paper provides no runtime/complexity analysis to show whether these extra computations are justified by the previous mentioned small accuracy gains. I think that could also be the reason that the author needs to propose a router based method to reduce the runtime overhead provided by INSES in simple query.
3. LLM-guided graph traversal and similarity-augmented edge expansion have both extensively appeared in prior work (for example, Think-on-Graph, link prediction in KG, etc). The contribution is mainly an engineering integration rather than a new theoretical or algorithmic insight.
4. While the MINE experiment in the last subsection attempts to validate INSES’s ability to handle incomplete KGs, the setup effectively evaluates link prediction rather than reasoning accuracy. Moreover, the main benchmarks (MuSiQue, 2Wiki, HotpotQA) do not reflect the noisy or aliased KG conditions motivating the method (Graph-RAG constructed noisy KG). This weakens the causal link between the stated motivation and the demonstrated improvements.

### Questions
See the weakness before.

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
The paper proposes a method for query answering (QA) on knowledge graphs that uses an LLM to navigate the graph and find an answer. Crucially, the method is thought to be integrated with techniques that automatically construct a knowledge graph from text. For this reason, there might be nodes in the graph with different labels but referring to the same concept, which would make the task of QA more difficult. Instead of trying to detect such nodes and merge them during graph generation, the authors instead propose using embedding similarities during inference in order to "jump" to nodes that would otherwise not be connected. This in turn is claimed to improved retrieval performances for QA.

### Strengths
I have appreciated the presentation of the paper, with a clear writing, self-explanatory figures, and a detailed description of both the proposed method and the experimental setting. Although I am not entirely confident, it seems that the many recent relevant works have been discussed thoroughly in Section 2 and in the rest of the paper.

I believe the contributions of the paper are substantial in this area. In particular, I think this paper provides a different perspective where, instead of trying to improve the construction of the graph index, an improved retrieval system is exploited. However, I believe there are some ablation aspects that should be addressed (see weaknesses).

### Weaknesses
The paper claims that pruning allows the method to discard triples that are not useful, which in turn helps removing noisy information (L217-218). The pruning is carried out by the LLM Navigator component, whose ablation w.r.t. other components in shown in Table 3. While I have appreciated the current ablation study, I believe it still misses an important baseline where no pruning is performed. My conjecture is that a huge portion of the improvement over the a Naive RAG baseline is due to (1) the presence of the graph index and (2) the similarity enhance part combined with the naive routing mechanism.

Since asking the LLM to navigate the triples quickly becomes very expensive with the average nodes degree, I believe the authors should perform ablation w.r.t. the LLM-based navigator. Furthermore, comparing with Naive RAG instead of a Graph RAG method does not enable me to disentangle the presence of the graph index and all the other proposed components. In other words, I believe a Graph RAG + Similarity Enhance baseline is missing from Table 3 (+ optionally also the Router).

I am willing to raise my overall score if the authors can provide some results supporting the need of the LLM-based navigator.

### Questions
- What method has been used to build the graph index for obtaining the results shown in Table 2 for INSES + Router?

See also my points in the Weaknesses part.

### Soundness
2

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes a novel Graph-RAG system that aims to fix the shortcomings of previous methods, in that they mostly explore the Knowledge Graphs (KGs) via the pre-existing, static edges. These approaches thus under-capture cross-entity relations, as real-world KGs are often incomplete. The proposed method, INSES, bridges this gap by augmenting the neighborhood with potential missing links predicted from the highest entity similarity matches. INSES also employs an LLM to select which links may be most relevant to the given queries at each exploration step, effectively pruning the search space. To resolve the computational efficiency issue, the authors further propose a router algorithm that only invokes the expensive multi-hop INSES search if the given query seems to be sufficiently complex. Overall, the authors empirically demonstrate the superior performance of INSES on a variety of real-world KG datasets, and through ablation studies, show that the embedding similarity enhancement technique indeed introduces the most performance gain.

### Strengths
1. The paper's introduction, related work, and experiment sections are well-written, rendering the insights and ideas behind the proposed method easy to understand.

2. The experimental part is extensive and comprehensive, with INSES benchmarking against a variety of baseline methods on many real-world, challenging KG-based QA benchmarks, showing the robustness of the proposed method. The ablation studies are also targeted, giving convincing evidence that the similarity-based missing-link enhancement contributes the most performance gain.

### Weaknesses
Most of my concerns regarding weaknesses are related to the Methodology section, which seems to be too high-level and lacking critical technical details.

W1. Throughout the paper, the success of the proposed method hinges on pre-existing embeddings computed for the entities. However, little attention is given (at least before the Experiment section) to discussing how these entity embeddings are computed, and what properties they should satisfy in order for INSES to work reliably. Are these embeddings purely semantical, based solely on the natural language descriptions and types of the entities? Or are they purely structural, computed only from the KG structures? Or are they a hybrid and a mix of both sources of information?

My understanding from the context (which may be very incorrect) is that the entity embeddings used throughout the paper are purely semantical, obtained via textual embedding methods. If that is true, one critical question needs to be asked: why not use more structural-aware KG embeddings? There is a gigantic corpus of literature on KG embedding methods developed for predicting missing links in KGs. These methods can take in both semantic and KG structural information, which intuitively would better suit the purpose of INSES compared to purely semantic-based embeddings.


W2. In section 3.4: Similarity-Based Expansion and Augmentation, the set of entities with the most similar embeddings, $V_{\text{sim}}$ is added to the set of entities to be explored and expanded next. However, in many real-world KGs, the entities with the most similar semantic embeddings to the current entity are very likely to be merely aliases, likely providing little to no extra information that is beneficial to answering the given query. The procedure thus may waste a lot of iterations on visiting or expanding the aliases, as the total number of allowed iterations is only 6. Thus, why not have the LLM navigator make a judgment on the entities in $V_{\text{sim}}$ as well, to make a decision whether these similar entities are also relevant to the queries?

W3. In Section 3.5, the discussion of the Router design, which should be another integral and significant part of the proposed method, is too high-level and lacks critical technical details. How does the Router decide whether a query is simple or complex? How does it assess its own confidence? How reliable is the estimated complexity and confidence? The paper would benefit if more important technical details were discussed in the main text.


Some other concerns regarding the mathematical writing in the Methodology section:

W4. The equation (4) is quite confusing to read. Is it suppose to say something like this?

$V_{\text{sim}}(u) =  \\{ v \in V \setminus \\{ u \\} \mid \cos ( \phi(u), \phi(v) ) \geq \tau_{\text{sim}} \\} ,$

which simply states that $V_{\text{sim}}(u)$ is the set of entities (apart from $u$) whose embedding cosine similarity with $u$ is above a threshold $\tau_{\text{sim}}$.

In addition, a LaTex math formatting typo on Line 264: The $V_{current}$ should be formatted as $V_{\text{current}}$.

### Questions
Q1. (Related to W1) Can the authors elaborate on the design of the entity embeddings? What kinds of information should be contained in the entity embeddings for INSES to work properly? Would a hybrid embedding incorporating both semantic and structural information be even more helpful for INSES and boost its performance?

Q2. (Related to W2) Can the authors elaborate on how too many aliases may affect the embedding-similarity-enhancement component? Is it possible for INSES to waste too much iteration budget on expanding aliases?

Q3. In Section 3.1, Definition 1, why define the triplet as $(u, \lambda_E (e), v)$ rather than $(\lambda_V(u), \lambda_E (e), \lambda_V(v))$? More generally, what is the purpose of $\lambda_V$? Since it does not seem that $\lambda_V$ is mentioned or used in any other part of the paper.

Q4. In Section 3.3, is there a restriction on how many candidate endpoints are allowed to be selected by the LLM navigator?

### Soundness
3

### Presentation
2

### Contribution
3
