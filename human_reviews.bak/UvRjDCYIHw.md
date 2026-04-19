# Double Equivariance for Inductive Link Prediction for Both New Nodes and New Relation Types

- Decision: Reject
- Scores: 6, 8, 3, 5

## Abstract
The task of inductive link prediction in discrete attributed multigraphs (e.g., knowledge graphs, multilayer networks, heterogeneous networks, etc.) generally focuses on test predictions with solely new nodes but not both new nodes and new relation types. In this work, we formally define the task of predicting (completely) new nodes and new relation types in test as a doubly inductive link prediction task and introduce a theoretical framework for the solution. We start by defining the concept of double permutation-equivariant representations that are equivariant to permutations of both node identities and edge relation types. We then propose a general blueprint to design neural architectures that impose a structural representation of relations that can inductively generalize from training nodes and relations to arbitrarily new test nodes and relations without the need for adaptation, side information, or retraining. We also introduce the concept of distributionally double equivariant positional embeddings designed to perform the same task. Finally, we empirically demonstrate the capability of the two proposed models on a set of novel real-world benchmarks, showcasing relative performance gains of up to 41.40% on predicting new relations types compared to baselines.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper investigates inductive link prediction for both new entities and new relations. It proposes an inductive structural double equivariant architecture that decomposes a knowledge graph into subgraphs containing different relations and encodes and aggregates them in the same way to eliminate the use of relation embeddings. The paper also constructs two datasets based on OpenEA and Wikidata5M. Extensive experimental results demonstrate the strong performance of ISDEA.

### Strengths
S1. This paper addresses an important and challenging task.

S2. It introduces a novel framework that avoids reliance on relationship embeddings.

S3. Good reproducibility - the paper provides code and detailed experimental settings.

### Weaknesses
W1. The proposed framework requires significant preprocessing and expensive encoding costs. This may be attributed to three factors: preprocessing costs, encoding for each relation, and separate scoring for each candidate entity.

W2. The 1 vs. 50 evaluation poses a risk as negative samples obtained from negative sampling are mostly easily distinguishable. This setup may not be sufficient to cover real-world scenarios.

W3. While ISDEA appears suitable for relation prediction, its performance on node prediction is not very good.

### Questions
Q1. As shown in Table 1(b), ISDEA's performance is not good and, in some datasets, even receives the lowest scores. Can you explain the reasons for this?

Q2. I am concerned about the efficiency of the proposed framework. Could you report training and inference times on some datasets?

Q3. It should be clarified that the multilingual KGs in the OpenEA library share the same schema. So many of the relations in these KGs overlap.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a theoretical framework for inductive link prediction over multi-relational graphs (knowledge graphs) where both entities and relations are unseen at test time. The framework includes the concepts of double permutation equivariance (to node permutation and edge type permutation) and its slight relaxation of distributionally double equivariance (to incorporate another existing model into the framework). Further, the authors introduce the first GNN implementation of the proposed framework – ISDEA as a double equivariant model, and DEq-InGram as a distributionally-double equivariant version of InGram. Experimentally, the authors devise a handful of new datasets and run experiments on relation prediction $(i, ?, k)$ and node prediction $(i, r, ?)$ tasks.

### Strengths
**S1.** Overall, I think it is a solid work that lays important theoretical foundations for the hardest of inductive link prediction tasks - dealing with both new entities and relations at test time requires more effort beyond learning relation embeddings. This is highly relevant for modern graph learning tasks, especially in low-data regimes without input node features.

**S2.** The experimental agenda is convincing - a handful of newly proposed datasets with relation prediction and entity prediction tasks. Perhaps the experimental section could have been even stronger if the evaluation was performed on all nodes/relations in the inference graph instead of 50 random negatives, but the authors acknowledge it is the scalability issues of the ISDEA model (not the framework in general) that are likely to be addressed in the future work.

### Weaknesses
The following ones are not the critical weaknesses but rather several discussion points I’d invite the authors to elaborate on: 

**W1.** The formalization in Section 2 assumes the existence of bijections (nodes-to-nodes, relations-to-relations) in training and test graphs. Basically, the framework posits the double equivariance only when training and test graphs have exactly the same number of nodes and edge types - which practically does not happen very often. On the other hand, the constructed datasets PediaTypes and WikiTopics all have different numbers of nodes and relations at training and test time (so there is no bijection possible). Could you please comment on the seeming discrepancy between the theory and what is measured in the experiments? 

**W2.** Section 5.2: “_relatively easier task of node prediction_” - I do not quite agree with this statement. The results might suggest the task is easier simply because you take 50 random negatives among _thousands_ of nodes in the inference graph, so those negatives are likely to be _easy_ negatives. On the other hand, the number of relations in the datasets is 50-150 in PediaTypes and <50 in WikiTopics, so the negative relation samples are likely to be harder. It was found that evaluation on small number of negative entities overestimates the performance, so I would hypothesize the numbers (and task impression) would change when the architecture would scale to ranking all nodes in the inference graph.

### Questions
**Q1.** What are the input features to standard GNN architectures reported in the experiments under GraphConv / GAT / GIN? Initialization of nodes with all ones or with random vectors? 

**Q2.** Since DEq-InGram is distributionally double equivariant (by means of averaging several runs with different random relation vectors initializations), would averaging NBFNet results across several runs with random relation initialization count as distributionally double equivariant as well?

**Q3.** The distributionally double equivariant idea posits equivariance in expectation, of which the easiest implementation is averaging over several runs (if we talk about drawing samples of relation vectors). Drawing parallels to group-equivariant CNNs, it is possible to achieve equivariance via augmentations such as frame averaging. I wonder if any such “augmentation” or frame averaging is possible within the double equivariance framework. If so, it might be a good idea to clearly state in the paper that distributionally-double equivariance is different from frame averaging

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the task of  "doubly inductive link prediction'", where the objective is to be able to make inductive prediction on both novel nodes and novel relation types, which are not encountered during training. This is a highly challenging task, especially because the authors do not allow the use of any additional context regarding the unknown relations.   The authors propose a general framework ISDEA to generate "double permutation-equivariant" representations and further explore ways to augment the existing InGram architecture with "distributionally double equivariant positional embeddings". Two new real-world datasets are proposed for benchmarking "doubly inductive link prediction" and experiments are carried out to validate the theoretical findings.

### Strengths
- **Problem and setup**: Inductive link prediction is a very important task and authors generalise this task to also predict novel relation types. The paper provides an approach for modeling equivariant representations of nodes and relations. 
- **Motivation and study**: A clear motivation, including the study of different architectures.
- **Benchmarking**: New benchmarking datasets are introduced and assessed against prior methods, establishing a new context.

### Weaknesses
- **Presentation and formal writing**: The writing of the paper is problematic and concepts are often unclear:
  - The text is very repetitive and contains many redundancies (i.e., the contribution of the paper is highlighted three times in the first page with paraphrased sentences), but when it comes to formal definitions, it does not make a rigorous treatment (see below).
  - Figure 1: This is crowded and does not explain much to me: why are the relations typed using conjunctions at this point? How does the logical description given in the beginning of page 3 in any way correspond to this figure?
  - Multigraph: Authors seem to suggest a multigraph is more general than a knowledge graph. It is unclear to me what authors specifically mean by this? If they mean a directed, multi-relational graph then this is nothing more than a knowledge graph. Heterogenous networks are special instances with single relation types allowed between nodes etc.
  - Doubly inductive: The naming is somewhat problematic, because the inductive prediction is either on the relation or on one of the entities at a time, but not both according to Def 1. 
   - Isomorphic triplets: The definition of multigraph isomorphism and triplet isomorphism is a very odd one. I have no idea why, e.g., (Hans, Grand $\land$ Father, Bob) in train and (Hanna, Granny $\land$ Mother) should be considered isomorphic (and at this point we still do not know the role of logical conjunction in defining the relations). This is essential because everything builds on this notion of "isomorphism" which is completely unjustified.
  - The paper is very hard to parse in general: in many cases, the statements of the results appear ambiguous to me, including the ones in the appendix.

- **New architectures**: The new architectures introduced in the paper appear to be somewhat incremental. IDSEA is a variant of DSS-GNN operating on relation-induced subgraphs, whereas DEq-InGram is a simple modification of InGram with bagging.

- **Empirical findings**: IDSEA seems to perform consistently worse than DEq-InGram in the task node prediction on PediaTypes, which does not seem to match what the theory suggests and is not being discussed in the paper. 

- **Train and test distribution**: The paper predominantly focuses on scenarios where the train and test graphs share a similar distribution. However, there exists a range of tasks involving unseen nodes and relations where the distribution significantly differs between the training and testing phases.  Further experimental validation on these tasks is required.

### Questions
Please refer to my review for clarifications and some more questions  here:

- In the experiments, why do the authors not compare with standard relational GNNs such as RGCN, CompGCN, NBFNets, etc?

- What are the differences between ISDEA, DEq-InGram, and InGram in terms of their runtime?

- Since both DEq-InGram and InGram produce distributionally double equivariant representations, why is there a substantial performance gap between these models on both datasets?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper aims to address the so-called doubly inductive link prediction task, where both new nodes and new relation types can be found solely in test time. To this end, author proposes two different models, ISDEA and DEq-InGram, which all abides by the equivariance requirement. Finally, experiment results show the new method beats baseline empirically.

### Strengths
The result of the paper seems sound, the author provides the reader with many theorems and proofs for its theory and they seem plausible to me.

The experiments are good, including many baselines, and the empirical result shows that the new method is in general better (though it falls behind the baseline in some settings).

### Weaknesses
The design of ISDEA is very straightforward, however, it is purely brutal force and has very high complexity. I have checked the statistics of the dataset used for experiment evaluation and found these two newly crafted datasets are significantly smaller than commonly used datasets, like FB15k or even its subset FB15k-237. I believe one major motivation for the setting for inductive learning is to allow for scalability towards a larger knowledge graph, yet the model design seems to be in the opposite direction.

### Questions
What is the largest knowledge graph that can be computed by ISDEA, for example, with GPU memory of 32 GB?


Can the isomorphism requirement be reduced to some WL test to reduce the complexity yet maintain decent empirical results?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
2 fair
