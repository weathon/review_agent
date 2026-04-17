# Graphlets as Building Blocks for Structural Vocabulary in Graph Foundation Models

- Decision: Reject
- Scores: 2, 4, 4, 4

## Abstract
Foundation models excel at language, where sentences become tokens, and vision, where images become pixels, because both reduce to discrete symbols on a shared, fixed grid. Knowledge Graphs share the discreteness, but not the geometry. Their entities and relations are discrete symbols, yet their arrangement is relational and lacks a common, fixed grid. Knowledge Graphs (KGs) share the discreteness, but not the geometry.
They form irregular, non-Euclidean topologies whose local neighborhoods differ from graph to graph. Therefore, Graph Foundation Models (GFMs) rely on identifying structural invariances to produce transferable representations. Without a universal token set, GFMs are limited in their ability to transfer representations across unseen KGs. We close this gap by treating graphlets, small connected graphs, as structural tokens that recur in heterogeneous KGs. In this paper, We introduce a model-agnostic framework based on a vocabulary of graphlets that mines a KG between relations via pattern matching. In particular, we considered closed and open 2- and 3-path, and star graphlets, to obtain robust invariances. The framework is evaluated on 51 KGs from a wide range of domains, for zero-shot inductive and transductive link prediction. Experiments show that adding simple graphlets to the vocabulary yields models that outperform prior GFMs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces ULTRA+, a knowledge graph foundation model that aims to improve ULTRA and MOTIF by replacing SPMM with pattern-matching with SPARQL and by reducing the high-arity pattern to positional-binary edges to use a single relation graph to compute the relation invariant. They further highlight the importance of open and closed paths when considered as graphlets, and with additional graphlets consideration, such as star-shaped graphs. Empirically, they report a marginal gain over ULTRA and MOTIF on zero-shot link prediction across 51 KGs.

### Strengths
- Pattern matching with SPARQL is a practical method that augments MOTIF and ULTRA, and it naturally supports open/closed distinctions, which is nice to see as this has been implemented beyond SPMM, which are quite limited.
- Binary version and summarization with positional binary edges are novel and a clean engineering trick.

### Weaknesses
**Major**:
- **Limited theoretical contribution**: The theorems are stated without proofs, which makes it hard to assess the core claims. In particular, a formal treatment of separation power (e.g., a WL-style characterization as in [1]) would substantially strengthen the work. 
- **Open vs. closed paths claim**: MOTIF can theoretically distinguish open vs. closed paths if the motif family includes the relevant closed patterns (e.g., cycles). In that sense, the architectural difference emphasized for ULTRA+ appears to be a direct corollary to Theorem 6.4 in [1]. 
- **Runtime/efficiency claims**: The paper states that SPARQL-based construction is “computationally less demanding” than the SPMM kernel, but neither a complexity analysis nor an empirical runtime/memory study is provided.
- **Scope of architectural novelty**: Beyond the relation-graph construction, ULTRA+ appears architecturally the same as ULTRA. 
- **Limited empirical evaluation** The Author does not empirically evaluate nor compare with TRIX [2], which can also implicitly count the homomorphisms. KG-ICL[3] is also not considered, which serves as a strong baseline. Additionally, it would be nice to see how each model variation can catch up with further fine-tuning, as the current gains over the existing model are small with no error bar or significance tests.
- **Explanation over empirical evaluation is unsubstantiated**: Notice that in the zero-shot setting, there is technically no difference between transductive and inductive dataset splits since the KGFM model does not observe the relation types or their distribution, regardless (they only know from the pre-training mix). Thus, the explanation of why ULTRA+ is worse than MOTIF on the transductive dataset is not justified. The author should instead discuss what the actual differences between these classes of datasets are, e.g., regarding graph statistics, to further justify their claim.

**Minor**: 
- **Small Bug in codebase**: During training, the author first computes the relation representation from the relation encoder and then applies edge dropout in the training mode. This edge dropout during training might potentially change the relation graph constructions and thus yield different relation representations. 
- **Presentation**: There are noticeable typos and inconsistencies in the paper; figures and tables are not properly adjusted.

[1] Huang, Xingyue, et al. "How Expressive are Knowledge Graph Foundation Models?." arXiv preprint arXiv:2502.13339 (2025). 

[2] Zhang, Yucheng, et al. "TRIX: A more expressive model for zero-shot domain transfer in knowledge graphs." arXiv preprint arXiv:2502.19512 (2025). 

[3] Cui, Yuanning, Zequn Sun, and Wei Hu. "A prompt-based knowledge graph foundation model for universal in-context reasoning." Advances in Neural Information Processing Systems 37 (2024): 7095-7124.

### Questions
- How does the notion of graphlet in ULTRA+ differ precisely from motifs in MOTIF?

- Can the author formally compare the separation power of positional-binary constructions with MOTIF or TRIX (e.g., via WL-style arguments or expressivity hierarchies)?

- Will the author include proofs for the stated theorems?

- Can the author provide runtime/memory analysis for SPARQL-based construction vs. SPMM (theoretical + empirical timing)?

- Does the author have an ablation isolating the effect of counting on relation graphs?

- Within the MOTIF framework, can the author add a triangle/cycle motif baseline to directly test the open vs. closed claim, and compare to ULTRA+ both empirically and theoretically?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors propose using graphlets as structural tokens to establish a shared vocabulary for Graph Foundation Models (GFMs).
Then, the authors introduce a model-agnostic framework that extracts and encodes graphlets (2- and 3-paths, closed triangles, and star structures) to capture structural invariances across heterogeneous KGs. This graphlet-based vocabulary enables zero-shot generalization across unseen graphs. Evaluations on 51 diverse KGs show that incorporating graphlets as structural tokens significantly enhances performance for both inductive and transductive link prediction tasks.

### Strengths
I like the idea of treating graphlets as a structural vocabulary that parallels the tokenization principle in language models, offering a clean conceptual bridge between discrete and relational domains. 
Using graphlets provides interpretable subgraph-level structures that can be intuitively linked to semantic or relational motifs in KGs.
The framework directly targets a key limitation in current Graph Foundation Models - their difficulty in transferring across unseen graphs.

### Weaknesses
- The use of graphlets as structural primitives is not entirely new; prior works in network science and graph representation learning have explored motif-based or subgraph-based encodings.

- The paper does not discuss the expressive power of the proposed graphlet-based vocabulary in relation to established graph isomorphism tests, such as the Weisfeiler–Lehman (WL) hierarchy. It remains unclear whether incorporating graphlets enhances the representational capacity beyond standard GNNs.  

- Extracting graphlets at scale (especially 3-paths or larger motifs) can be computationally expensive for large or dense graphs.

### Questions
- Does the approach capture semantic relations beyond structural similarity? For instance, can similar structures with different relational meanings be disambiguated?

- How does the proposed framework compare against motif-based GNNs or subgraph isomorphism-based methods in terms of both efficiency and generalization?

- How does the proposed graphlet-based structural vocabulary relate to the Weisfeiler–Lehman (WL) in terms of expressive power?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper introduces a model-agnostic framework Ultra+ that builds a vocabulary of small graphlets to address the challenge of creating transferable representations in Graph Foundation Models (GFMs) for Knowledge Graphs (KGs), which lack a universal geometric structure. This sounds like a meaningful and promising research question, both for GFMs and KGs. However, the paper still faces some issues that need further improvement and clarification.

### Strengths
The problem studied in this paper is highly important and offers insightful implications for applying graph foundation models to knowledge graphs. Apart from some minor details, the overall writing of the thesis is professional. The experiments appear sufficiently comprehensive and generally sound, both in terms of dataset selection and comparisons with baseline methods. This paper seems to have sufficient theoretical support. There are insights into the improvement of the Ultra framework. Such research is worthy of appreciation and recognition.

### Weaknesses
From the perspective of graph foundation model frameworks, Ultra+ is an extension of the existing Ultra framework, and its novelty appears somewhat limited. I'm not very clear whether the graph foundation model framework used in knowledge graphs is relatively similar and unified, but it is clear that innovation at the model level is not the main contribution of this article. 

While the paper provides very detailed definitions, it only includes two theorems, which are insufficient to robustly support the core argument. It would be beneficial to rigorously demonstrate the superiority of Ultra+ from perspectives such as expressive power, similar to what was done in paper [1].

Theoretically-driven improvements are certainly appreciated, and some theoretical ideas may already be integrated into the main text. However, the contributions should be emphasized more clearly. At present, it is difficult to fully grasp the theoretical innovations and the sources of Ultra+'s advantages as a new framework.

Some details need to be improved and clarified:

The description of graph foundation models on line 114 is outdated. It is now recognized that large language models represent a key branch of graph foundation models, extending beyond the scope of pre-trained GNNs[2, 3].

Lines 165-166: The “product function” notation $\eta \cdot \rho \cdot \eta$ is non-standard and can be confusing.

Are the concepts of graphlets and motifs first introduced in this paper? If not, they should be appropriately cited.

As a theorem, Theorem 3.2 requires a proof or a citation to the original work where it was first proposed. Similarly for Theorem 4.3.

Subfigures (d3) and (e3) in Figure 2 are identical.

“the query triple q(h, ?)” on line 294 and “the query (h, q, ?)” on line 303 are inconsistent. This discrepancy is confusing. The notation for a query should be unified and clearly defined.

A key claimed advantage of Ultra+ on lines 328-329 is its ability to discriminate between closed and open paths, unlike Motif. This point would be significantly strengthened by providing a concrete example illustrating this discrimination and linking it to theoretical results about expressive power.

[1]Xingyue Huang et al. How Expressive are Knowledge Graph Foundation Models? ICML, 2025.

[2]Jiawei Liu et al. Graph foundation models: Concepts, opportunities and challenges. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

[3]Zehong Wang et al. Graph foundation models: A comprehensive survey. arXiv preprint arXiv:2505.15116, 2025.

### Questions
"Ultra+ extends this approach by incorporating a richer set of graphlet-based pattern," how large is the size of this Graphlets in the specific implementation? If it is less than 5 as shown in Figure 2, how to "capture more complex and higher-order interactions between relations"

Will expanding the size of Graphlets expand the range of structural vocabulary and further enhance the generalization performance of GFMs?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a Graph Foundation Model (GFM) framework based on graphlet structural vocabulary, designed for knowledge graph reasoning and zero-shot link prediction tasks. The authors argue that existing GFMs (such as Ultra and Motif) are limited in their ability to capture complex structural patterns, particularly due to their neglect of closed paths and higher-order structures. To address this, Ultra+ introduces a rich graphlet-based vocabulary that includes 2-path, 3-path, and star-shaped motifs to capture more robust structural invariances. Experiments conducted on 51 knowledge graph datasets demonstrate that Ultra+ consistently outperforms Ultra and Motif in both inductive and zero-shot reasoning settings.

### Strengths
Overall, the paper demonstrates strong innovation, with rigorous theoretical definitions, comprehensive experimental design, and good reproducibility, offering a new perspective on the role of structural vocabulary in GFMs.

**S1.** This paper presents an innovative framework Ultra+ which enhances graph structural modeling capability by introducing a graphlet-based structural vocabulary.


**S2.** On methodology, Ultra+ adopts a two-stage message passing mechanism that decouples relation graph learning from entity embedding learning, thereby achieving strong inductive and zero-shot generalization capabilities on unseen entities and relations.


**S3.** The experimental section covers more than 50 knowledge graph datasets, validating the model’s generality and superiority in inductive reasoning and zero-shot link prediction, and demonstrating consistent and robust improvements over models such as Ultra and Motif.

### Weaknesses
**W1.**  The improvements over Ultra and Motif remain somewhat ambiguous, lacking a clear mechanistic explanation—particularly regarding why the introduction of closed paths and higher-order graphlets leads to theoretical and performance gains. Moreover, the paper does not clearly justify why only closed and open 2-paths, 3-paths, and star-shaped graphlets are considered to achieve robust invariance.

**W2.**  GFMs should not be limited to KG; the paper lacks a discussion on the potential significance and generalizability of the Ultra+ framework when applied to other graph datasets, such as social networks or molecular graphs.

**W3.** The paper does not clearly explain how the proposed structural vocabulary is sampled, nor does it provide the corresponding complexity analysis.

**W4.** The paper lacks a notation table, and the numerous mathematical symbols used throughout, along with the unclear connections and roles between definitions and theorems, make the paper difficult to follow. It is recommended to include clarifying explanations or detailed proofs to improve readability.

**W5.** The experiments are not sufficiently comprehensive and lack comparative evaluations with the latest Graph Foundation Model baselines.

### Questions
See the weakness above.

### Soundness
3

### Presentation
2

### Contribution
2
