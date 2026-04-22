# GCSGNN: Towards Global Counterfactual-Based Self-Explainable Graph Neural Networks

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Graph Neural Networks (GNNs) exhibit superior performance in various graph-based tasks, ranging from scene graph generation to drug discovery. However, they operate as black-box models due to the lack of access to their rationale for a specific prediction. To enhance the transparency of GNNs, graph counterfactual explanation (GCE) identifies the minimal modifications to the input graph that cause the GNN to change its prediction to a different class. Current GCE methods face two major challenges: (1) they adopt a post-hoc explanation paradigm by separately training an explainer model for a trained GNN. This sequential optimization
process yields suboptimal explanations since the GNN training process is not exposed to the explainer. (2) Current methods are primarily local-level approaches, which means that they generate explanations for each input sample individually. As a result, they cannot capture the shared prediction rationales that generalize across the entire input data distribution. 
To address these two challenges, we propose a novel Global Counterfactual-based Self-explainable GNN (GCSGNN) framework. GCSGNN can simultaneously act as a GNN, providing predictions on input samples, and an explainer, generating explanations for its predictions. Furthermore, GCSGNN is trained to identify common patterns in the GNN embeddings across input samples, enabling it to learn global (i.e., model-level) explanations. Extensive qualitative and quantitative analysis across various datasets demonstrates that our GCSGNN achieves outstanding performance against the baseline methods. Our code can be found at https://anonymous.4open.science/r/gcsgnn.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors of this work start by pointing out critical issues with current graph counterfactual explanations. Namely, since current existing GCE methods are post hoc, there is an external explanation model that is detached from the GNN they are seeking to train. They point out that this can cause the generation of inconsistent explanations. Another issue is the limited scope of GCE methods; most methods take an individual input graph and generate an explanation per instance. They point out that this methodology fails to diagnose global patterns consistent throughout the distribution of graphs. Their method tries to find global patterns that are more consistent and stable by exploring the embedding space. Specifically, they find the channels (dimensions) in the embedding space that influence the graph prediction the most. They find counterfactual embeddings that are vectors that have the same dimension as the number of important channels. They then convert the counterfactual sub-embedding into a subgraph that acts as the counterfactual explanation. They describe their counterfactual explanation in the form of a tuple called counterfactual graph embedding edits (CGEE) which consist of the position of the important channels/dimension of the embedding space and the counterfactual sub-embedding vector. They introduce the notion of coverage where given a GNN the coverage of a set of CGEE is the portion of input graphs such that when applying one tuple in the set of CGEE produces a valid counterfactual (change of label w.r.t. the GNN). The authors also prove that their method achieves better explanations (by maximizing the MI between label and prediction for both the original GNN and counterfactual explanation) than any post-hoc methods. This proof is straightforward observation that they employ an unconstrained optimization in comparison to a constrained optimization over the parameter space that post-hoc methods use. Their framework takes learnable matrices that learn important channels in the embedding space and counterfactual sub-embeddings, and minimizes the global-level GCE loss function. They also train an encoder and decoder according to a reconstruction loss. They conduct experiments on several baselines on graph counterfactual interpretability. They assess w.r.t. their metric of coverage and proximity (graph edit distance to the CFE) the other baselines to see how methods fare. Finally, they assess other factors such as parameter analysis, a case study, and runtime analysis.

### Strengths
S1. The paper is well written. From a scan of the literature it seems they are filling a gap in the literature utilizing several existing frameworks/methodologies to provide an improvement to existing graph interpretability works, particularly in counterfactual explanations.


S2. Methodology is reasonable and initial implementation decisions make sense. 


S3. The results in some experiments suggest that the method is more effective than existing post-hoc methods while also being more effective than other global CFE methods.

### Weaknesses
W1. The idea of exploring the embedding space and exploring critical channels is a somewhat novel angle. The one issue with this is the reliability/stability of going from embedding space to graph space. Ideally, the mapping from important sub-embeddings to critical subgraphs that act as the CFE should be exact, however in practice this is not usually the case. The authors do not give any guarantees or even assurances on the reliability of sub-embeddings being mapped to important subgraphs. The authors should explore this weakness or at the very least justify this point.
W2. The authors conduct experiments on several baselines. However, they have missed a critical baseline on counterfactual graph explanations [1]. If this work is not relevant to their work they should still justify their choice of excluding it since it is a global graph counterfactual method. 
[1] Bajaj, Mohit, et al. "Robust counterfactual explanations on graph neural networks." Advances in neural information processing systems 34 (2021): 5644-5655.

### Questions
1.) Can you explain why the reconstruction loss ensures that sub-embeddings can be mapped reliably to important subgraphs for CFE?
2.) As per W2 [1] was left out and if it intentional can you justify the reasoning of excluding it from this work, otherwise I do believe this is a relevant baseline and should at least be mentioned in the work as it is quite relevant.

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
5

### Summary
This paper tackles the challenges of post-hoc explainer misalignment and the inefficiency of 
local-level-only explanations in GNNs . The authors propose GCSGNN, a self-explainable GNN 
that jointly learns to predict and explain its own predictions . Its main contribution is a framework
for learning global counterfactual explanations by identifying and applying shared edit rules 
(CGEEs) directly in the graph embedding space.

### Strengths
- The paper combines the GNN and its explainer into a single model that trains together. Unlike post-hoc methods, it helps capture the model's dynamics better and provides predictions and explanations simultaneously.
- The model is designed to find global explanations that apply to multiple graphs in the dataset, one graph at a time, like the local methods.
- The experiments demonstrate that the model achieves significant time and performance improvements over the baselines

### Weaknesses
- Limitations of Binary Classification: The framework is specifically designed for binary 
classification. The counterfactual loss only maximizes the probability of a single, fixed class 1. This design limits the method's applicability, and the paper makes no attempt to discuss its use for multi-class classification scenarios.
- Lack of proximity optimization: The paper's methodology is disconnected from the main goal of GCE, which is to find "minimal 
modifications". The final objective function contains no loss term that explicitly minimizes the 
graph edit distance or penalizes large counterfactuals. The model is only trained to find a valid 
label change, not a minimal one, which ignores a core principle of GCE.
- Ambiguous formulation: The paper doesn’t provide a precise mathematical definition for "proximity”. It is vaguely 
described as a "squared sum" on one-hot vectors 2, but this omits the full details of a proper 
GED calculation, such as costs for insertions and deletions. This affects the reproducibility of the method.
- Baseline Comparisons: The experimental results in Table 1 show that most baselines perform at 0.00 coverage, time 
out (TOO), or are n/a in proximity. This is highly suspicious and suggests an unfair experimental
setup. The baselines were significantly modified from their original (e.g., factual) purpose, and 
all post-hoc methods are unfairly evaluated on explaining the GCSGNN model itself. This 
comparison makes the claimed superiority of GCSGNN unreliable.
- Unclear ablation study (Figure 4): The ablation study in Figure 4 presents large improvement percentages (e.g., +2785.0%). This 
is because the "ablation" is not a meaningful removal of a component, but rather a comparison 
against a broken model where components, such as the encoder or generator, are fixed at their 
random initialization. This only proves that a trained component is better than a random one, 
which is a uninformative and trivial claim.
- Unclear theoretical analysis: The entire proof relies on a critical assumption: that their "proxy method" of editing the 
embedding space is a valid approximation for the true, complex problem of editing the graph's 
structure. This strong assumption isn’t supported, which affects the whole analysis considerably.

### Questions
- Although the authors provide the source code, the details of the experimental setups 
are not clear enough. For example, what is the base model that the post-hoc methods 
optimize? Does it constitute the GCSGNN model’s encoder and predictor? In that case, 
the comparisons are unfair. How do authors set hyperparameters for the baselines?
- The paper claims that global methods are better than local explainer, as they provide 
explanations for multiple graphs of a dataset and offer a general insight. However, these 
two approaches address different problems, as local methods can provide fine-grained 
and sample-specific explanations, which are more useful in many cases. Are there any 
other points to consider when acknowledging global methods over local ones? 
Additionally, the paper claims that counterfactual methods are superior to factual 
methods, but they do not provide sufficient evidence to support this claim.

### Soundness
2

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
5

### Summary
This paper proposes GCSGNN, a self-explainable graph neural network that jointly learns prediction and counterfactual explanation. Instead of finding instance-specific structural edits, GCSGNN learns a small set of global counterfactual graph embedding edits (CGEEs)—shared latent transformations that can flip the prediction for many graphs simultaneously. The model consists of four jointly trained modules (encoder, counterfactual generator, decoder, and predictor) and optimizes classification, counterfactual, and reconstruction losses. Experiments on molecular and image-graph datasets show that GCSGNN achieves higher counterfactual coverage and lower generation time than post-hoc counterfactual baselines such as CF-GNNExplainer and GCFExplainer.

### Strengths
- Clear motivation and formulation. The paper clearly articulates the limitations of existing post-hoc counterfactual GNN explainers and motivates the need for global, self-explainable counterfactual reasoning.

- Novel conceptual idea. Learning shared counterfactual edit templates in latent embedding space is an original and elegant approach to discover global reasoning patterns.

- End-to-end design. The unified architecture jointly trains the explainer and predictor, avoiding costly post-hoc optimization and enabling fast inference.

- Strong empirical results. The method consistently outperforms prior counterfactual explainers in terms of coverage and efficiency on multiple datasets.

### Weaknesses
- Limited connection to global explanation literature. The paper primarily compares to post-hoc counterfactual explainers but does not discuss related model-level explanation methods such as XGNN and GNNInterpreter, which also aim to extract global reasoning patterns. Positioning GCSGNN relative to these works would clarify its contribution to the broader explainability landscape.

- Interpretability of latent edits. Counterfactuals are generated by manipulating embedding channels, yet the semantic meaning of these edits in terms of node or edge structure remains unclear. More examples or visualizations are needed to demonstrate that CGEEs correspond to meaningful graph modifications.

- Lack of discussion on decision-boundary understanding. Although counterfactual generation implicitly explores the decision boundary, the paper does not analyze or visualize how embedding edits relate to the classifier’s boundary. A clearer discussion of boundary behavior, potentially referencing works like GNNBoundary, would strengthen the conceptual grounding.

- Limited metric diversity. Evaluation is largely restricted to coverage and proximity; including metrics such as fidelity, sparsity, or diversity would provide a more comprehensive assessment of interpretability.

### Questions
- Can the authors provide more concrete examples or visualizations to show how specific CGEEs correspond to meaningful node- or edge-level changes in the graph?
- Since counterfactual generation inherently explores boundary regions, can the authors analyze how the learned edits interact with or traverse the decision boundary of the classifier?
- Would additional interpretability metrics, such as fidelity, sparsity, or diversity of CGEEs, yield more nuanced insights into the model’s explanations?
- How sensitive are the results to the number and dimension of CGEEs (k, dₛ)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors provide a method for the generation of global graph counterfactuals (GCSGNN) based on as opposed to the most often occurring instance-level/local counterfactual explanations. A major selling point of the paper is, that the the method is self-explainable, that is, one does not need to carry out post-hoc analysis to obtain model explanations. The method is evaluated on a collection of common bench-mark datasets.

### Strengths
The modelling problem of finding global counterfactual explanations is very interesting, and an area still under development. The general idea of the paper is well-conveyed, and the code is made publicly available already at this time. The proposed method shows good performance compared to baselines on the results reported by the authors.

### Weaknesses
- I am slightly confused about whether the scope of the paper is only to generate global or also local/instance level counterfactual explanations? The ability to generate global counterfactual explanations is highlighted as a main contribution, but at the same time, the ability of the model to generate local GVE's is highlighted (e.g. line 133-134).
- The theoretical analysis in section 3 amounts to pointing out, that the GCSGNN can achieve a superior mutual information between the label and the predicted factual/counterfactual since optimization occurs over a larger parameter space. This is frankly not surprising. The analysis carried out is centered around a post-hoc counterfactual generation interpretation of GSCGNN, and does not give a clear argument in favor of self-explainable methods. Besides, the authors do not consider whether the proposed joint optimization procedure can have a negative impact on the classification performance. Lastly the analysis I recommend that the authors reconsider the role of this section, and in particular tighten the mathematical rigor. 
- The authors do not at all consider the fact that the same graph can have different representations (i.e. be the same up to isomorphism). 
- Experiments and evaluation: As I understand it counterfactuals are the decoded graphs of the counterfactual embeddings; I find this slightly misleading as an encoding of the generated counterfactual is not necessarily going to the same as the counterfactual embedding. This can have potentially large consequences for the evaluation of the model in terms of coverage depending on whether the validity of a counterfactual is computed with respect to the counterfactual embedding or the encoding of the decoded counterfactual.

### Questions
- "Explainer Misalignment" is considered a major challenge for GNNs. But is this really the case? One could argue that training a post hoc explainer is preferable, as one would then be able to generate explanations in cases where the training procedure of the model we wish to explain is not known or not controlled.
- Line 110: Is the encoder permutation invariant?
- Line 125-127: Which training objective? Please be explicit if possible.
- The lines 87-92 and lines 127-131 are almost exactly the same. I suggest that the authors make this more concise to minimize redundancy.
- Line 131: How are the global counterfactuals obtained from the counterfactual subembeddings? In my understanding the subembeddings are not used to obtain a global GCEs, but are rather combined with the embeddings of a specific input graph to produce an instance level, local GCE.
- Typo in equation one: "$\textbraceleft t \ldots \textbraceright \text{ for some s} \in  \mathcal{S}$" should be "$\textbraceleft \ldots \text{ for some s } \in  \mathcal{S}\textbraceright$".
- On line 110-111 the method is said to consist of the models $f_p$ and $f_e$, however, at line 171-172 the method additionally has the element $f_c$ denoting the counterfactual explainer. Later in section 4.1 "Model overview" a decoder $f_d$ is introduced. I urge the authors to be consistent in the model setup.
- Line 179-180: Is this assumption reasonable? And how does it impact the analysis if it is not?
- Line 182-183: It is stated that "GCSGNN aims to find the set of parameters that maximizes both the mutual information between the label and f". Please elaborate on this connection between the mutual information and the training objective as it is not evident from the text.
- Table 1: Many of the methods reported perform extremely poorly (e.g. 0.00 coverage on the Aids dataset). Why do they perform so poorly?
- Can you provide samples and illustrations of the global counterfactuals produced, and the graphs which are sampled from the models? I do not see any samples reported in the paper.

### Soundness
2

### Presentation
3

### Contribution
2
