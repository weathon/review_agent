# GNN Explanations that do not Explain and How to find Them

- Decision: Accept (Poster)
- Scores: 6, 4, 8

## Abstract
Explanations provided by Self-explainable Graph Neural Networks (SE-GNNs) are fundamental for understanding the model's inner workings and for identifying potential misuse of sensitive attributes.
Although recent works have highlighted that these explanations can be suboptimal and potentially misleading, a characterization of their failure cases is unavailable.
In this work, we identify a critical failure of SE-GNN explanations: *explanations can be unambiguously unrelated to how the SE-GNNs infer labels.*
We show that, on the one hand, many SE-GNNs can achieve optimal true risk while producing these degenerate explanations, and on the other, most faithfulness metrics can fail to identify these failure modes.
Our empirical analysis reveals that degenerate explanations can be maliciously planted (allowing an attacker to hide the use of sensitive attributes) and can also emerge naturally, highlighting the need for reliable auditing.
To address this, we introduce a novel faithfulness metric that reliably marks degenerate explanations as unfaithful, in both malicious and natural settings.
Our code is available on [GitHub](https://github.com/steveazzolin/gnn_deg_expl).

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
While GNNs have received much research attentions, explanations on GNNs are becoming important. There are in general two families of studies on GNN explainations: (1) post-hoc explanations, and (2) self-explainable GNNs. This paper focuses on the latter one, and the authors made three important contributions: (1) they indicate that even self-explainable GNNs cannot be trusted. They show both therectially and empirically that the explanations (in the form of nodes, edges, attributes, or subgraphs) given by such self-explainable GNNs could be different from the real elements that lead to the prediction. In other words, this could be a misalignment between the explanations and predictions; (2) they show that such misalignments could be manipulated by attackers, and they can also arise naturally; (3) they show that existing faithfulness metrics often fail to capture such misalignments, and they propose a new metric to mitigate such issues. Overall, I think this paper is in good quality, ready for publication.

### Strengths
(1) The probelm is well defined;

(2) The paper is well organised;

(3) Most statements/claims are supported empirically with experiments;

(4) This paper provides strong insights on limitations of exitsing self-explainable GNNs;

### Weaknesses
(1) The theoretical analysis focuses on using isolation nodes as explanations, which may not be practical; as far as I know, most GNN explataions provide subgraphs (rather than isolated nodes) as explanations;

(2) This paper focuses on class classification, and node classification is not discussed;

### Questions
No questions, please see the week points.

### Soundness
3

### Presentation
3

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
This paper identifies a critical failure mode in Self-Explainable Graph Neural Networks (SE-GNNs): they can produce degenerate explanations that are completely unrelated to how the model actually makes predictions, while still achieving optimal predictive performance. The authors theoretically prove that under mild conditions, SE-GNNs can output such unfaithful explanations (e.g., highlighting non-discriminative "anchor set" nodes) and still achieve optimal true risk. Empirically demonstrate that these explanations can be maliciously planted to hide the use of sensitive attributes and existing faithfulness metrics often fail to detect such unfaithful explanations.  The paper then propose a new faithfulness metric, EST (Extension Sufficiency Test), which more reliably flags degenerate explanations by testing all possible supergraphs of the explanation.

### Strengths
1. Identifies a fundamental failure mode with serious implications for trustworthy AI in graph domains. The insight that explanations can serve as label-encoding channels is important for SE-GNN practitioners.

2. Theorem 1 formalizes when optimal risk coincides with degenerate expla- nations for SE-GNNs. The proof technique (constructing explicit e and g pairs) is interesting. Extension to Theorem 2 connecting EST with formal explanation notions strengthens the contribution.

3. Empirical validation on multiple datasets (synthetic RBGV, real-world MUTAG/MNISTsp/SST2P) and multiple architectures spanning different SE-GNN families in both adversarial (Section 4) and natural (Section 6) settings.

### Weaknesses
1. Theorem 1 requires hard explanation extractors, excluding soft/continuous scores common in practice, and it assumes |R| > 0, which limits generality. Also, the anchor set definition (single-node subgraphs in all graphs) is restrictive (in the discussion, it mentions generalizations but no formal treatment)

2. Attack requires training access (strong assumption), though authors acknowledge this fits MLaaS scenarios. The stopping criteria (appendix D.1.2) are manually tuned per dataset which is unclear how to set these systematically. Also, tere is insufficient analysis of when/why attacks fail for Table 2.

3. Fail to cite and discuss closely relevant recent works [A-C] (see questions)

### Questions
1. Given that Yu et al. (2019) formally characterized degeneration in rationalization models (where explanations encode labels via trivial patterns) and proposed complement predictors as a solution, how is your Theorem 1 different beyond adapting notation to graphs? (here my question is about motivation and your novelty). Can you explicitly show what new theoretical insight SE-GNNs provide over the rationalization literature?

2. You demonstrate a serious problem (degenerate explanations) and provide a detection metric (EST), but there is no empirically validated prevention strategy. Why other existing solutions from rationalization (complement predictors and shared encoders) or neuro-symbolic systems (like reasoning shortcut mitigation) don’t work for SE-GNNs?

3. Your theorem requires hard explanation extractors ({0,1} or {r,1}) and single-node anchor sets, but most practical SE-GNNs use soft scores and more complex patterns. How restrictive are these assumptions?

4. On the discussions with relevant work

- RQ1: On the Brittleness of Explanations & Malicious Manipulation (RQ1) 

Connection to Ref [A] (Li et al., ICML'24): The work shows explanations can be maliciously planted. Ref [A] demonstrates that GNN explanations are fragile, meaning small perturbations to the graph structure can drastically alter the explanation. This suggests that the problem of unreliable explanations is even more widespread: they can be not only deliberately fabricated but also easily and unintentionally broken. 

How does the phenomenon of maliciously planted degenerate explanations (your work) relate to the general brittleness of explanations under minor structural perturbations (Ref [A])? Could the attack you propose be seen as a structured, large-scale exploitation of this inherent fragility?


- RQ2: On Causal Metrics for Faithfulness

Connection to Ref [B] (Behnam et al., ECCV'24): The work shows existing faithfulness metrics fail. Ref [B] proposes a causal-effect-based metric, arguing that it is less susceptible to spurious correlations and can more reliably capture the true reasoning of the model. This presents a directly relevant alternative to your proposed EST metric.

How does your proposed EST metric compare to causal-effect-based metrics like the one in Ref [B]? Could a causal framework provide a more principled foundation for defining and measuring faithfulness, potentially avoiding the need for the perturbation-based approach of EST? Please discuss the potential advantages and limitations of both perturbation-based and causal-based faithfulness metrics in the context of your findings.

- RQ3: On Provably Robust Explanations

Connection to Ref [C] (Li et al., ICLR'25): The work identifies a problem (degenerate explanations), and Ref [C] offers a potential solution for a related issue. Ref [C] aims to build SE-GNNs whose explanations are provably robust to graph perturbations, ensuring consistency. While their goal is robustness against adversarial graph perturbations, the underlying principle—enforcing stability in the explanation—might also prevent the model from latching onto arbitrary, degenerate subgraphs.

Could the techniques for building provably robust explanations (Ref [C]) be adapted or extended to formally prevent the emergence of degenerate explanations identified in your Theorem 1? In other words, can robustness constraints guide the model toward explanations that are not only stable but also inherently faithful?

[A] Li et al. Graph Neural Network Explanations are Fragile. ICML’24

[B] Behnam et al. Graph Neural Network Causal Explanation via Neural Causal Models. ECCV’24

[C] Li et al. PROVABLY ROBUST EXPLAINABLE GRAPH NEURAL NETWORKS AGAINST GRAPH PERTURBATION ATTACKS. ICLR’25

**I would consider raising the score if the questions are sufficiently addressed during the rebuttal**

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper raise a possible explanation for self-explainable GNN(SE-GNN)'s unfaithfulness, that the explanation extractor may extract subgraphs with bijective mapping to the real groundtruth explanation to achieve well classification performance, yet only providing no-related explanation subgraphs. The author validate this view by training a group of attacked models which are humanly controlled on specific false explanation, while achieving good classification performance. Based on such insights, the author further propose a new sample-based metric namely EST on evaluating unfaithful explanations of SE-GNNs, and validate its effectiveness through experiments on both good true-negative and near-zero false-negative.

### Strengths
**1.** This paper is well presented, which makes me enjoy reading it. The flow of questions raised, illustration of the problem, proper designed figures and complete experiments make the motivation and findings of the paper well expressed. 

**2.** The raised explanation for SE-GNN's unfaithfulness in Theorem 1 is interesting and convincing. It assumes that some common anchor nodes serves as a bridge between real pattern and classification attention is novel, which also makes sense to me. The example 1 on the with figure illustrated is also well designed, which is simple and obvious for understanding.

**3.** The experiments are complete and sound, which support mostly all claims in the paper. To validate the existence of such problem the author provide (1) the attacked models' performance (Table 2), (2) attacked models' rejection rate (Table 3) and (3) natural models' rejection rate (Table 4), which show that such unfaithfulness could exist in normal performing mode and may be potential reason for unfaithfulness of natural model.  To validate the effectiveness of the proposed EST, both the well rejection ratio of attacked models (Table 3) and the near zero rejection rate for designated explanations (Table 10) are provided to show this EST is a reliable metric. The provided Figure 2 also aligns with the supposed sample amount requirement of EST and also shows its effectiveness. It's also good to see every experiment column in Table 4 is assigned with a illustration figure.

### Weaknesses
**1.** The design of the new metric is somewhat limited in novelty. The difference between the proposed metric and other metrics is mainly just  enlarging the choice range of samples, i.e., including both nodes and edges for randomness. This metric is designed to approximate all the subgraphs, however, just assumed as the sample amount increases they are equivalent, which may increase high complexity. While this may not be a critic point since evaluation complexity is not a so serious problem, and explanation subgraphs are usually small. 

**A minor typo**: Fig.2 UST may suppose to be EST

### Questions
1. Would this raised faithful problem also happens in post-hoc explainer on raw GNN? I think this shift between original important patterns and patterns used for classification may also appears in raw GNN models, e.g., for a star graph it only recognizes the most high degree node.  And could this metric be also applicable for them?

### Soundness
4

### Presentation
4

### Contribution
3
