# HYPER: A Foundation Model for Inductive Link Prediction with Knowledge Hypergraphs

- Decision: Accept (Poster)
- Scores: 2, 4, 6, 8

## Abstract
Inductive link prediction with knowledge hypergraphs is the task of predicting missing hyperedges involving completely *novel entities* (i.e., nodes unseen during training). Existing methods for inductive link prediction with knowledge hypergraphs assume a fixed relational vocabulary and, as a result, cannot generalize to knowledge hypergraphs with *novel relation types* (i.e., relations unseen during training). Inspired by knowledge graph foundation models, we propose HYPER as a foundation model for link prediction, which can generalize to *any knowledge hypergraph*, including novel entities and novel relations. Importantly, HYPER can learn and transfer across different relation types of *varying arities*, by encoding the entities of each hyperedge along with their respective positions in the hyperedge. To evaluate HYPER, we construct 16 new inductive datasets from existing knowledge hypergraphs, covering a diverse range of relation types of varying arities. Empirically, HYPER consistently outperforms all existing methods in both node-only and node-and-relation inductive settings, showing strong generalization to unseen, higher-arity relational structures.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Proposes a method for inductive link prediction on knowledge hypergraphs (HKG). The approach closely follows ULTRA (for regular KGs), but modifies the representation of arity information in the relation graph (via an MLP position encoder) and replaces NBFNet (KGs) by HCNet (HGKs). Performs an experimental study on fully-inductive and node-inductive HKG link prediction. I like the paper, but I am not convinced that the experimental study is fair (see W2), and I am thus hesitant to recommend acceptance.

### Strengths
S1. Simple, convincing approach

S2. Relevant problem

S3. Experimental results are comprehensive and promising

### Weaknesses
W1. Low novelty. The proposed method is ultimately a rather straightforward modification of ULTRA. I do not count this against the paper too much though, because it is the natural approach and exploring and evaluating this approach provides value.

W2. Comparison to ULTRA not convincing. After reading the abstract of this paper, I immediately thought why not use ULTRA + reification. The authors then actually did this but in a way that is not convincing. To me, a key contribution of this paper is a fair and solid evaluation of the proposed method and alternatives, but I am not convinced that this has been done. This is for the following reasons:

1. The reification approach appears problematic. All relations become nodes, and only max-arity relations of form hasEntity-k(hyperedge, entity) + one hasRelationType relation are used. This severely limits the relation modelling capabilities of ULTRA. Instead, the natural reification is to relation-k(hyperedge, entity) and drop the hasRelationType relation. I feel that this approach needs to be explored, as it is what first comes to mind and as it appears much more promising.

2. ULTRA has not been retrained on the reified data (whereas the proposed method is). Since ULTRA has never seen reified relations, I'd not expect it to work well. A fair comparison would do the reification above and then train ULTRA on the same graphs (both KGs and HGs) as the proposed method. Only then I'd consider the study insightful and convincing.

W3. Comparison to HCNet not convincing. The authors seem to use random relation embeddings, as HCNet does not support new relations. I fail to see the value of this experiment; it clearly cannot lead to useful results. The experiment should either be dropped (ok with me) or perhaps (i) transductive results added and (ii) for the inductive setting, fold-in the new relations into HCNet.

### Questions
See W2.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a foundation model named HYPER for inductive link prediction. By constructing a relation graph G_rel and encoding relation representations on it, HYPER utilizes these learned representations to derive entity representations on the original knowledge hypergraph, enabling zero-shot generalization to knowledge hypergraphs of any arity, including new entities and new relations.

### Strengths
The paper proposes a framework that enables zero-shot generalization to knowledge hypergraphs of arbitrary arity, including novel nodes and relations, at test time.

The idea of encoding hyperedges using positional interaction is innovative.

The paper is clearly written with precise definitions, illustrative figures for concepts like relation graphs and reification, and provides code.

### Weaknesses
On L39, "The generality knowledge hypergraphs" should be corrected to "The generality of knowledge hypergraphs". The expression on L320-321 may also be problematic.

The pre-trained ULTRA model used for comparison, as described in Table 12, has not been pre-trained on any hypergraph, whereas HYPER(4HG) and HYPER(3KG+2HG) have both been trained on hypergraphs, which may lead to unfairness in the comparative experiment.

The idea of "constructing a relation graph $G_{rel}$ to learn relation embeddings and then using relation embeddings to learn entity embeddings" is similar to ULTRA, with only an extension of positional interaction encoding based on the concept of hyperedges, which may lack sufficient novelty.

### Questions
During the reification process, the positional information between nodes in a hyperedge is transformed into distinctions on edges of the form hasEntityi(edge id, ui). Does this lead to a loss of positional information?

HYPER's pre-training dataset is too small, potentially leading to overfitting to the relational structures of specific graphs. Could HYPER be trained with a larger pre-training dataset to assess its generalization ability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a foundation model (HYPER) for predicting missing hyperedges in knowledge hypergraphs that contain unseen entities and unseen relation types. HYPER could generalize to arbitrary-arity relations through a relation graph encoder that captures positional interactions between relations. The paper also introduces 16 new inductive benchmark datasets derived from existing knowledge hypergraphs.

### Strengths
1.	HYPER is a general framework that can transfer learned relational patterns across different relation types and arities. This is the first foundation model that supports zero-shot generalization on knowledge hypergraphs of arbitrary arity.
2.	16 new inductive benchmark datasets derived from existing knowledge hypergraphs are constructed for evaluation.
3. Empirical investigation over the positional interaction encoding scheme

### Weaknesses
1.	The positional interaction encoding $\mathrm{EncPI}((a,b)) = \mathrm{MLP}([p_a \| p_b])$ may violate symmetry and lacks proof of equivariance or smooth extrapolation to unseen arities; this weakens the theoretical basis for HYPER’s claimed generalization.

2.	While HYPER demonstrates strong zero-shot generalization across diverse hypergraph benchmarks, how practical is it for large-scale real-world knowledge systems—given that the number of positional interactions grows quadratically with relation arity and may impose significant computational and memory costs during training and inference?

### Questions
Refer to the Weakness section.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes and architecture for inductive link prediction with unseen entities as well as unseen relations over knowledge hypergraphs  via knowledge hypergraph foundational models. The authors show the feasibility of the approach over link prediction task with standard evaluation metrics and propose a set of new datasets.

### Strengths
- The paper introduces a problem which is very much mainstream in the field of knowledge hypergraphs and address an important gap.
- The paper is very well written and easy to read.
- The authors performed thorough experimentation with the comparative results along with the analysis.

### Weaknesses
- the authors could cite another relevant paper which is performing inductive link prediction over knowledge graphs considering features related to the relations [1].
- The role of encoding positional encoding could be explained with the help of an example to show its importance for the proposed approach. 
- It so far seems like a combination of approaches, the authors should highlight the main theoretical contribution in the paper.
- Table 3 show the MRR results where HCNET outperforms HYPER. Also in many cases the improvements are only marginal. Can authors shed the light over these kind of results since there can be a possibility that the results of the baselines might improve with extensive hyper-parameter optimization. 



[1] https://dl.acm.org/doi/10.1145/3579051.3579066

### Questions
See the weaknesses of the approach.

### Soundness
3

### Presentation
4

### Contribution
3
