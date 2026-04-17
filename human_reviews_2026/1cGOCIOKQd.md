# Flock: A Knowledge Graph Foundation Model via Learning on Random Walks

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
We study the problem of zero-shot link prediction on knowledge graphs (KGs), which requires models to generalize to novel entities and novel relations. Knowledge graph foundation models (KGFMs) address this task by enforcing equivariance over both nodes and relations, which enables them to learn structural properties of nodes and relations that transfer to novel KGs with similar structure. However, the conventional notion of deterministic equivariance inherently limits the expressive power of KGFMs, as it prevents them from distinguishing relations that are structurally similar but semantically distinct. To overcome this limitation, we propose to leverage probabilistic node-relation equivariance, which preserves equivariance in distribution while using structured randomness to break symmetries at inference time. Building on this principle, we present Flock, a KGFM that iteratively samples random walks, encodes them into sequences, embeds them with a sequence model, and aggregates node and relation representations through learned pooling. Flock respects probabilistic node-relation equivariance and, crucially, is a universal approximator for isomorphism-invariant link-level functions over KGs. Empirically, Flock perfectly solves our new diagnostic dataset Petals on which current KGFMs fail, and achieves state-of-the-art performance on entity and relation prediction tasks across 54 KGs from diverse domains. Code is available at https://github.com/jw9730/flock.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces FLOCK, a novel knowledge graph foundation model for zero-shot link prediction. The authors identify that existing knowledge graph foundation models (KGFMs) rely on deterministic node-relation equivariance, which limits their ability to distinguish between structurally similar but semantically different relations. To address this issue, the authors introduce the concept of probabilistic node-relation equivariance and design FLOCK based on this principle. FLOCK iteratively samples random walks from KGs, encodes them into sequences, embeds them with a sequence model, and aggregates node and relation representations through learned pooling.

### Strengths
1. The paper's concept of probabilistic node-relation equivariance extends the theoretical framework of knowledge graph foundation models and provides detailed proofs showing FLOCK is a universal approximator for link-level functions.

2. The authors clearly identify limitations of existing KGFMs, particularly in handling structurally similar but semantically different relations, and provide an intuitive Star Wars character relationship example (Figure 1).

3. The paper presents extensive evaluations on 54 KGs across different domains, including both entity prediction and relation prediction tasks, in both zero-shot and fine-tuned settings, with FLOCK outperforming state-of-the-art models in most cases.

### Weaknesses
1. While the authors highlight FLOCK's significant advantage over ULTRA and TRIX on the Metafam dataset in lines 288-290, Table 10's fine-tuning results show all three models achieving nearly 1.000 MRR, contrasting with the significant differences in the zero-shot setting. This inconsistency is not explained.

2. While Section 4.2 provides a thorough theoretical analysis, the paper doesn't adequately discuss how these theoretical advantages directly explain specific performance differences observed in experiments, particularly across different types of knowledge graphs.

3. On certain datasets (e.g., WK-50, Table 8 line 1570+), FLOCK performs worse than ULTRA and TRIX, contradicting the overall conclusions, but these anomalies aren't analyzed or explained.

### Questions
1. Given the random walk coverage issues, have you explored more targeted random walk strategies rather than uniform random walks? For example, could biased random walks based on graph structure or query information improve model performance?

2. In the transductive setting, FLOCK underperforms baseline models, which seems to contradict the model's theoretical advantages. Could you provide a deeper analysis of this phenomenon and how this issue might be resolved while maintaining the model's expressiveness?

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
This work argues that existing Knowledge Graph Foundation Models (KGFMs) are limited by deterministic node-relation equivariance, preventing them from distinguishing structurally isomorphic yet semantically distinct relations. They propose probabilistic node-relation equivariance as a relaxation and introduce FLOCK, a KGFM based on sampling and encoding anonymized random walks using sequence models, without using traditional message-passing. FLOCK is claimed to respect probabilistic equivariance and be a universal approximator for link-invariant functions. They also design a diagnostic dataset (PETALS) to showcase the limitations of prior work, and showcase empirical results suggesting state-of-the-art performance on the standard benchmark of the domain with 54 KGs. Overall, I think this work makes a valuable contribution, and I would be happy to increase my score if my concerns are addressed.

### Strengths
1. **Problem:** Identifies and demonstrates a clear limitation (expressivity with respect to structurally isomorphic relations) of KGFMs based on strict equivariance using a new synthetic dataset (PETALS).
2. **Elegant solution:** Proposes probabilistic node-relation equivariance as a potentially more expressive alternative inductive bias and introduces FLOCK, a novel non-message-passing KGFM architecture based on random walks and sequence models.
3. **Theoretical backing:** Provides theoretical analysis regarding universality and probabilistic invariance (though practical relevance is debatable).
4. **Empirical Backing:** Shows strong empirical performance on a wide range of KGs, particularly in relation prediction and on the PETALS set.

### Weaknesses
1. **Scalability:** The reliance on random walks raises concerns for large graphs, both in terms of sampling time and ensuring adequate graph coverage. The proposed test-time adaptation is heuristic. Efficiency comparisons (Table 6) show FLOCK can be much slower than baselines.
2. **Practicality:** The theoretical benefits (universality, distinguishing specific isomorphic cases) might not translate into significant practical advantages across all common KG structures and tasks, especially given the variance introduced by stochasticity. Although the results across the standard benchmark suggested improvement, prior work [1, 2] has shown that this benchmark has potential issues. FLOCK's pipeline (sampling, recording, sequence processing, consensus) is also more complex than standard message passing approaches, potentially hindering adoption.
3. **Ablation Gaps:** More detailed ablation (like Figure 4b) on the random walk strategy (like length, different samplers) and sequence model architecture would be beneficial.

---

*[1] Harry Shomer, Jay Revolinsky, & Jiliang Tang (2025). Towards Better Benchmark Datasets for Inductive Knowledge Graph Completion. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD).*

*[2] Arvindh Arun, Sumit Kumar, Mojtaba Nayyeri, Bo Xiong, Ponnurangam Kumaraguru, Antonio Vergari, & Steffen Staab. (2025). SEMMA: A Semantic Aware Knowledge Graph Foundation Model. In The Thirtieth Conference on Empirical Methods in Natural Language Processing (EMNLP).*

### Questions
1. Beyond PETALS, can you provide examples from the benchmark KGs where the ability to distinguish structurally isomorphic relations proved crucial for FLOCK's performance advantage? How common is this structural ambiguity in real KGs?
2. The universality proof sketch relies on walks covering all edges. How does the approximation quality degrade in practice when walks inevitably provide only partial coverage on large graphs? Does this partial coverage limit the types of functions FLOCK can effectively learn?
3. Given the computational overhead of sampling and ensembling, under what specific conditions (graph size, structure, task requirements) does FLOCK offer a clear practical advantage over potentially faster, deterministic KGFMs like TRIX or ULTRA, especially if these were to be augmented with techniques to increase expressivity (subgraph features or textual semantics like [2])? SEMMA [2] approaches the same problem of “losing the ability to distinguish between two entities with opposite relationships”, but by directly encoding the textual features into ULTRA. This at least warrants a discussion.
4. How was the sequence model architecture (biGRU + RMSNorm + SwiGLU) chosen? Did you experiment with other paradigms (like Transformers or SSMs), and how did they perform in terms of accuracy and efficiency?
5. Could the proposed probabilistic equivariance be achieved through methods other than random walks (like stochastic message passing, randomized node features)? How does the choice of random walks specifically contribute? Appendix F is great, but more than naive noise injection would make it stronger.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The work proposes a novel knowledge graph foundation model (KGFM) architecture, FLOCK, with stochastic equivariant node and relation representations. The stochastic equivariance property endows the model with better expressivity without sacrificing zero-shot generalizability to novel KGs with unseen node and relation types. The authors achieve this via an innovative adoption of random walks that anonymize the identities of nodes and relation types, thus keeping the model informed of solely the structural roles that the visited nodes and relation types play in the random walks. In the experimental part, the paper then demonstrates the performance of FLOCK on a wide range of both synthetic and real-world KGs, showing superior performance to the baselines and the capability to distinguish isomorphic relation types for the link prediction tasks on the synthetic PETALS dataset.

### Strengths
1. The paper correctly identifies that most of the existing KGFMs employs deterministic node-relation equivariance, which creates an expressivity bottleneck. The paper then follows the well-established, theoretically sound notion of stochastic/probabilistic equivariance. The introduction of random-walk based methodology to tackle the KG representation is novel, and the proposed model design is both intuitive and well-motivated.

2. The paper creates the new PETALS synthetic dataset to cleverly demonstrate the unique advantage of FLOCK at distinguishing isomorphic relation types. This group of experiments is well-designed, providing targeted empirical validation. The rest of the experiments are also diverse and comprehensive, with FLOCK showing consistently superior performance.

### Weaknesses
W1. **My major concern is the paper's novelty claim of "the first stochastic KGFM"**. As identified in one of the previous work (Gao et al. 2023), the InGram model (Lee et al., 2023) can be viewed as a stochastic node-relation equivariant KG model, because it injects random initial node and relation type embeddings, thus creating node and relation embeddings that are equivariant in distribution. Based on this observation, Gao et al. then proposed an improvement, named DEq-InGram, that further stabilizes the stochastic equivariant embeddings of the original InGram, showing significant improvement in performance. Thus, I do not believe that this work may claim that FLOCK is the "first stochastic KGFM." Could the author then clarify this point? If DEq-InGram was indeed a predecessor to FLOCK on the line of stochastic KGFMs, could the author further elaborate how FLOCK could be uniquely advantageous compared to DEq-InGram?


W2. **Missing error bars on the empirical results**. In all experiment tables in the main text, only the average numbers are reported. The error bars (e.g. standard deviation numbers) are missing. It would be informative to see the comparison of variance of model performance, particularly for stochastic equivariant models, because the stochasticity nature of the model usually results in higher variance of model prediction accuracy. 

W3. **Clarity of writing and lack of critical technical details in methodology**. The writing on the random walks section (line 219 - 229) is a bit confusing to read and could be improved. For instance, the paragraph on line 226 seems to repeat the same message with those discussed in the preceding paragraph. Furthermore, in this preceding paragraph that starts on line 219, it is not clearly explained why $3n$ walks are needed. Could the authors clarify why are there $3n$ walks?

Later in the sequence processor section (line 257 - 269), it is critical for the readers to understand how does the "structural indices" such as 1, 2, 3, for nodes and $\alpha$, $\beta$ from equation (5) and (6) translates to which neural embeddings in the GRU model from equation (7). This part is critical because later, the analysis of the equivariance-ness of the entire pipeline hinges on it. It would be great if more writing effort is devoted here to clarify and elaborate how the structural indices interacts with the GRU model.


W4. **Ablation on the Consensus Protocol**. In the consensus protocol section, the authors made the following claim regarding the approach of taking averages of the proposals:

> The drawback is that uninformative proposals from e.g. dangling regions of walks are not directly suppressed, and can affect the state updates.

This claim seems to be a conjecture at this stage, without theoretical analysis or empirical evidence. Would it be possible to provide further evidence to support this claim, such as some sort of ablation studies?


W5. **Lack of proof insight or proof sketch**. Although this is not necessarily a critical concern or weakness, it would be great if the theoretical section 4.2 have some discussions on the insights or ideas or proof sketch to illustrate intuitively, why does the claims and propositions are mathematically correct. Currently, it reads like the section 4.2 is simply an information dump of related theoretical results without much insights into their correctness.


W6. **Notations clarity**: On line 154

> Let $\omega$ be a function assigning to each KG $G = (V, E, R) \in \mathbb{K}_{n, m}$ ...

The notation $\mathbb{K}_{n, m}$ is first used before being properly defined and introduced (it was later introduced in Line 158).


References:

[1] Gao, Jianfei, Yangze Zhou, Jincheng Zhou, and Bruno Ribeiro. "Double equivariance for inductive link prediction for both new nodes and new relation types." arXiv preprint arXiv:2302.01313 (2023).

[2] Lee, Jaejun, Chanyoung Chung, and Joyce Jiyoung Whang. "InGram: Inductive knowledge graph embedding via relation graphs." In International conference on machine learning, pp. 18796-18809. PMLR, 2023.

### Questions
Please see the Weaknesses section for my questions and concerns.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
An architecture for zero-shot entity and relation prediction. FLOCK adopts a probabilistic approach to the equivariance assumption, as opposed to the conventional deterministic interpretation. The method relies on random walks sampling and direct sequence encoding.

### Strengths
- Research problem/limitation of SoTA KGFMs is well explained and clearly justified.
- Novelty: FLOCK is the first to adopt a stochastic take on node-relation equivariance. Extending the notion of probabilistic invariance to KGs is more than reasonable.
- Comprehensive evaluation campaign. Up-to-date baselines choice from literature. Protocol shows fair comparison w.r.t. to baselines.
- Relation prediction results clearly stronger than conventional KGFMs.
- Writing is clear and narrative flows well and is coherent.
- Related work is comprehensive and up to date.

### Weaknesses
- Scalability of random walks on large KGs (the authors are well aware and they have discussed this in the conclusions)
- I could not find an overall training time and inference time comparison against conventional KGFMs such as ULTRA, TRIX, FLOCK. It is not entirely clear to which extent the adoption of random walks is slower than message passing? (apologies if I have missed)
- Results: entity prediction results only marginally better than prior art.
- (minor) Figure 2 could help a better caption, to clarify color coding and actions taken at each step.
- Some examples would help clarify, e.g. "They reveal nodes $v_s$ and relations $r_s$ specific to each KG which obstructs transferability to unseen KGs." (line 239)

### Questions
- What is the recommended interval of $l$ values for random walks, keeping in mind impact on predictive power and training time? 
- Have you considered using a transformer architecture for the sequence processor?
- line 236: Can you clarify what do you mean by "to keep GPU memory usage in a range"?

### Soundness
4

### Presentation
4

### Contribution
4
