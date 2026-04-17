# LogicXGNN:  Grounded Logical Rules for Explaining Graph Neural Networks

- Decision: Accept (Poster)
- Scores: 4, 8, 6, 8

## Abstract
Existing rule-based explanations for Graph Neural Networks (GNNs) provide global interpretability but often optimize and assess fidelity in an intermediate, uninterpretable concept space, overlooking the grounding quality of the final subgraph explanations for end users. This gap yields explanations that may appear faithful yet be unreliable in practice. To this end, we propose LogicXGNN, a post hoc framework that constructs logical rules over reliable predicates explicitly designed to capture the GNN's message-passing structure, thereby ensuring effective grounding. We further introduce data-grounded fidelity ($Fid_D$), a realistic metric that evaluates explanations in their final-graph form, along with complementary utility metrics such as coverage and validity. Across extensive experiments, LogicXGNN improves $Fid_D$ by over 20% on average relative to state-of-the-art methods while being 10-100 times faster. With strong scalability and utility performance, LogicXGNN produces explanations that are faithful to the model's logic and reliably grounded in observable data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper tackles post-hoc rule-based model-level GNN explainabiltiy, aiming to ground the extracted global rules to human understandable data-level. They propose LOGICXGNN, which constructs logical rules directly grounded in data, ensuring that explanations correspond to real subgraphs. It introduces a new metric, data-grounded fidelity (FidD), which measures how well explanations match the GNN’s outputs in the actual input space, not in an abstract one.

### Strengths
1. Quantitative results are consistently strong. Strong empirical performance on fidelity, coverage, stability, validity and efficiency.
2. Well motivated. The paper identifies a valid and underexplored problem. Clear identification of an overlooked issue in prior rule-based explainers. 
3. The framework is complete and reproducible.

### Weaknesses
1. **Missing clear discussion and comparison on an important baseline.** The paper briefly mentions GCNeuron (Xuanyuan et al., AAAI 2023) in the related work section, categorizing it as a “concept-based global explanation”. However, it does not clearly articulate the methodological differences or justify the absence of a direct comparison. Given that GCNeuron is also a rule-based, model-level GNN explainer that produces logical rules to characterize model behavior, a clearer discussion of methodological distinctions, or at least a brief justification for not comparing, would strengthen the completeness of the evaluation.
2. **Overly domain-specific evaluation. Lack of validation on simple synthetic tasks.**
The paper claims to propose a general and scalable global rule-based GNN explainer, but almost all examples and visualizations are from molecular datasets (Mutagenicity, BBBP, NCI1). This makes the work look domain-specific rather than general. The authors did not include enough simple synthetic datasets where the ground-truth reasoning patterns are known. On such datasets, an effective rule-based method should ideally achieve near 100% data-grounded fidelity, clearly showing that it can capture the model’s true decision rules. Without this type of controlled experiment, it remains unclear whether the proposed rules genuinely reflect model reasoning or are just fitting chemical regularities in the data.
3. **Excessive complexity of grounded rules without well-clarifications.** While the proposed orbit-based grounding mechanism improves formal precision, it significantly increases the structural and logical complexity of the resulting rules. The grounded rules involve multi-level orbit decomposition, nested logical conjunctions, and numerical thresholds, which make them difficult for non-expert users to interpret. Compared to prior rule-based explainers such as GLGExplainer and GraphTrail that generate concise and human-readable logic, the grounded rules here are overly abstract and cumbersome. This undermines one of the central goals of grounding in this paper, which was explicitly framed as aiming for "human-understandable" explanations.
4. **Insufficient justification for the rule-based paradigm.** Although the paper emphasizes the importance of rule grounding, it does not convincingly demonstrate why a rule-based approach is preferable to other established global explanation paradigms, such as generation-based (XGNN, GNNInterpreter) or subgraph-based (TreeX) methods. There is no human study, no qualitative comparison of interpretability, and no synthetic benchmark explicitly designed to evaluate rule quality. As a result, the claimed advantage of rule-based grounding remains conceptually appealing but empirically unproven.
5. **Questionable novelty of the proposed evaluation metric.** The proposed “Data-grounded Fidelity (Fid$^D$)” metric appears conceptually aligned with the instance-level fidelity evaluation already used in prior work such as TreeX, where global explanations are mapped back to instances to check whether they reproduce the original GNN’s predictions. The paper reframes this idea under a new term “data-grounded fidelity”, but this seems more like a terminological reframing than a fundamentally new metric design. Moreover, TreeX also provides motif-level global explanations evaluated on instances, making it a highly relevant baseline. However, the authors neither compare with TreeX nor acknowledge this conceptual overlap. This omission weakens the claimed novelty of Fid$^D$ and leaves the evaluation incomplete. 
6. **Limited technical novelty.** The proposed framework largely combines existing ideas: constructing logical rules from learned predicates (as in GLGExplainer, GCNeuron and GraphTrail), grounding them to data (as in motif-based explainers like TreeX and PAGE), and evaluating instance-level fidelity (similar to TreeX).

### Questions
n/a

### Soundness
1

### Presentation
1

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
The paper addresses a real limitation in current global explainability methods for GNNs: they derive logical rules in a latent or concept space, and only afterwards associate these latent concepts with illustrative (sub)graphs; as a result, the generated example (sub)graphs often fail to correspond to real structures in the dataset (in molecular datasets, some are even chemically invalid). 
The author propose a multi-step framework that learns logical rules whose predicates are directly grounded in subgraphs observable in the input data. The method is compared against two relatively recent baselines (GLGExplainer and GraphTrail), which the authors re-evaluate by substituting their latent concepts with the representative subgraphs provided by those methods. A new evaluation metric called data-grounded fidelity is introduced to assess how well the logical explanations reproduce the model’s behavior on real graphs rather than on latent representations. 

Overall, the motivation is clear and the addressed problem appears real and relevant. However, the paper is occasionally difficult to follow, mainly because the pipeline uses several dense steps, sometimes similar (e.g. different decision trees for different purposes). The provided implementation also raises minor concerns: the released code explicitly skips the grounded part of the pipeline for all datasets except BBBP, Mutagenicity, and NCI1, and it appears not to run correctly for IMDB-BINARY.

### Strengths
The paper identifies a genuine limitation of current global explainability methods for GNNs: the lack of grounding of logical explanations in real graph instances. The proposed framework offers a systematic and sound solution. The proposed metric is reasonable. The presentation of the baseline results seems rigorous: tha appendix states that the authors of the original methods were consulted to verify the correctness of the reproduction.

### Weaknesses
The proposed procedure is quite convoluted, and the absence of ablation studies makes it difficult to understand which components of the pipeline are essential and which could be simplified. For similar reasons, the methododgical explanation is hard to follow: the paper requires several readings to fully grasp the role of each step (e.g., the multiple decision trees used for different purposes). The experimental comparison includes only two baselines.

### Questions
1. The released code explicitly skips the grounded part of the pipeline for all datasets excepts BBBP, Mutagenicity, and NCI1. Could the authors clarify whether this was intentional and whether the grounded evaluation can be extended to other datasets (e.g., IMDB-BINARY doesn’t seem to work)?
2. The procedure is difficult to follow. Have the authors considered simplifying the exposition, for example through a schematic overview?
3. An ablation study could help isolate which steps in the pipeline are most important. In addition, testing against more baselines would strengthen the empirical evidence.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a framework to extract logic rules as explanation structures for GNN outputs. The method encodes node receptive fields using a WL kernel-based hash function and trains a decision tree to generate regulations that consistently classify the model's predictions. Conjunctive logic rules are extracted as grounding rules to link predicates with the input feature space. An experimental study verified the methods' efficiency, fidelity, and scalability.

### Strengths
S1. There is novelty in WL hashing-based structural encoding and decision tree for generating logic rules. 
S2. The overall method is justified with a cost analysis. 
S3. Results support the major claims.

### Weaknesses
W1. The link between the quality of the generated rules and their closeness to real-world evidence remains unevaluated. 
W2. The time-cost analysis lacks a more rigorous elaboration. Important steps seem overly simplified or omitted. 
W3. More baselines are needed, such as motif-based GNN explanation, which can directly generate explanations as subgraphs.

### Questions
D1. A main challenge is how to ensure the explanation structures are better grounded in genuine real-world evidence, while the process yields rules based on GNNs that still seem to focus on the models' faithfulness. Have any human experts or authorities assisted in evaluating the generated rules?  How will such a measure be quantified, if possible? 

D2. The time cost omits several sources of overhead, such as training decision trees; a more complete analysis is not in place. 

D3. Is the method model-specific? Meaning: if one changes to another test set or another model, does the method need to be restarted from scratch, even when the graph is not changed? How may the method respond to larger-scale analysis?

D4. There is a lack of in-depth analysis of how likely the rules are to be redundant or logically entailed by others—a missed opportunity for optimization? 

D5. Other approaches, including motif-based GNN explanation, directly output graph patterns or subgraphs, which can be readily converted to conjunctive triple patterns or rules. Representative work needs to be compared with.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes a novel approach to post-hoc explanations of graph classification, based on concepts. Differently from previous methods, however, it grounds the explanations into actual patterns in the data by means of Weisfeiler–Lehman (WL) graph hashing.

### Strengths
- The paper is well written and presented. 
- The methodology is not particularly complex, but it allows creating a very effective method
- Section 3.4 reports a very nice Analysis subsection, theoretically describing the computational complexity of the method and its applicability to different gnn architectures
- The experiments show a clear advantage of the proposed method against a couple of state-of the-art baselines

### Weaknesses
General weaknesses:
- The paper does not have a limitation paragraph, which is now considered almost mandatory in top-level conferences
- The scope of the proposed method is focused to explanations regarding graph classification only (similarly to the baselines). It would be very interesting if the authors could show or even just describe whether the proposed method could be applied to node classification, or if any methodological assumptions fail to hold in that case. 
- Also in several parts there are mentions of graph tasks, suggesting that different types of tasks have been considered, which does not seem the case: only multiple instances of the graph classification task have been tested. I would recommend rephrasing as it is currently misleading.
- One of the main result of the paper is that previous baselines provide explanations that are not grounded. However, in the main paper (I saw them in the appendix) there are no mentions regarding how the baselines have been reproduced. Without at least a footnote saying how the explanations for the baselines have been extracted, Figure 1 results too strong and may rise critiques.


Specific weaknesses:
- The sentence "As a result, LOGICXGNN not only generates a rich set of representative subgraphs but also learns generalizable grounding rules for each predicate, addressing unreliable grounding in existing methods." is not very clear I would suggest rephrasing.
- $p_j$ and $P$ are not properly defined

### Questions
My main question is regarding the applicability of the proposed method to node classification task:
- is it feasible to consider the same framework also in this case? 

My guess is that the hashing could be re-used similarly but possibly also the decision trees to select the patterns over the embeddings and the one over the activation matrix.

### Soundness
4

### Presentation
4

### Contribution
4
