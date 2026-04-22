# CaSBRE: Causality-inspired Semi-supervised Biomedical Relation Extraction

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 2, 6

## Abstract
Biomedical interaction relations, such as chemical-protein interactions (CPIs) and gene-disease associations (GDAs), are crucial for advancing drug discovery and clinical treatments. However, the vast diversity of biomedical entities and the limited availability of labeled data pose significant challenges to accurately modeling these interactions using traditional supervised learning approaches. These methods often overfit to spurious feature-label correlations in the scarce labeled relations, leading to poor generalization to unseen biomedical entities. To overcome these challenges, we introduce CaSBRE, a causality-inspired semi-supervised learning framework designed to disentangle and mitigate the impact of such spurious correlations. CaSBRE includes two core components: (i) Feature Disentanglement, which separates causal from spurious features by identifying and exploiting discrepancies between their correlations in labeled and unlabeled data; and (ii) Do-calculus Interaction Inference, which marginalizes the influence of spurious features on relation predictions. Through extensive experiments on CPI and GDA tasks, we demonstrate that CaSBRE substantially outperforms state-of-the-art methods, particularly in generalizing to previously unseen biomedical entities, thereby providing a robust and scalable solution for biomedical relation extraction.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper CaSBRE (Causality-Inspired Semi-Supervised Biomedical Relation Extraction) proposes a causality-driven framework to improve biomedical relation extraction, such as chemical–protein and gene–disease interactions, under limited labeled data. Traditional semi-supervised methods often overfit to dataset-specific biases, so CaSBRE introduces a Feature Disentanglement Network to separate causal from spurious features by detecting inconsistencies between labeled and unlabeled data, and a do-calculus inference module to marginalize out spurious effects during prediction. Tested on DrugBank (CPI) and DisGeNET (GDA) datasets, CaSBRE significantly outperforms existing supervised and semi-supervised baselines, especially when generalizing to unseen biomedical entities. This demonstrates that causality-inspired disentanglement and inference can substantially enhance robustness and generalization in biomedical relation learning.

### Strengths
1. The paper accurately identifies two core challenges in current biomedical relation extraction, which are *training set distribution bias (domain bias)* and *label scarcity*.
2. The proposed framework is intuitive and easy to reproduce, without relying on large external modules or complex graph structures.

3. The writing follows a coherent logical structure, and the conclusion section provides a thoughtful discussion of the method’s limitations.

### Weaknesses
1. **Weak causal foundations**

   The framework lacks an explicit causal graph (DAG), structural causal model (SCM), or counterfactual formulation. The “do-calculus” step is only a heuristic averaging procedure rather than a theoretically identifiable intervention. The key assumption $P_l(C|S) \neq P_u(C|S)$ is also not empirically verified. The approach is more like a statistical learning framework wrapped by causality.

2. **Lack of interpretability in feature disentanglement**

   The paper claims to disentangle causal (C) and spurious (S) features, but provides no semantic or biological interpretation of these components. Figure 2 only shows point-cloud separation without explaining what the dimensions mean or whether (C) aligns with known biochemical properties.

3. **Heuristic implementation of do-calculus**

   The “intervention” simply samples $S \sim N(0,1)$ and averages predictions. This prior is a little arbitrary. The paper does not discuss how the sampling number $K$ or prior choice affects stability or performance. This step behaves more like stochastic ensembling than formal causal inference.

4. **Limited experiments**

   Experiments are restricted to two datasets (CPI and GDA), and the baselines are earlier approaches, rather than the most recent. The ablation only removes $L_{sc}$ or do-calculus, without comparing against alternative disentanglement or SSL strategies. The generality of the conclusions is therefore limited.

5. **Lack of biological interpretability**

   While performance gains are shown, the paper does not analyze whether the learned causal features correspond to known biological mechanisms. No case studies or example-level explanations are provided, which weakens credibility for domain adoption.

### Questions
1. **Empirical support for the $P_l(C|S) \neq P_u(C|S)$ assumption**

   Can the authors provide quantitative or visual evidence showing that the conditional distributions indeed differ?

2. **Rationale for the do-calculus sampling prior**

   Why is $S \sim N(0,1)$ chosen? Have the authors attempted to estimate $P(S)$ from the data or tested the sensitivity to the prior and the number of samples $K$?

3. **Interpretation of the disentangled features**

   What do $C$ and $S$ correspond to biologically? Can the authors visualize attention maps, motifs, or clusters that support the claim of causal versus spurious components?

4. **Comparison with alternative semi-supervised methods**

   How does CaSBRE compare with modern SSL approaches such as consistency regularization, pseudo-labeling, or contrastive learning? Is the improvement due to the “causal” idea or simply an effect of domain regularization?

5. **Terminology accuracy**

   Given the lack of formal causal identifiability, would it be more precise to describe the method as a “distributionally robust semi-supervised framework” rather than “causality-inspired”?

6. **Generalization and extensibility**

   Can the framework be extended to multi-relational (n-ary) or multimodal biomedical knowledge graphs? The paper briefly mentions transfer and active learning as future work; elaborating a concrete path would strengthen the contribution.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces CaSBRE, a causality-inspired semi-supervised framework for biomedical relation extraction. It aims to disentangle and mitigate the impact of such spurious correlations by separating causal features from spurious ones. The core idea is to exploit the discrepancy in S-C correlations between labeled and unlabeled data as a supervisory signal for disentanglement. For inference, it employs a do-calculus interaction inference strategy to marginalize the influence of spurious features, thereby improving generalization to unseen entities.

### Strengths
- The paper proposes a novel feature disentanglement strategy that leverages the assumption that S-C correlations in labeled data may not generalize to unlabeled data.   
- The empirical results are strong and consistent, showing that CaSBRE substantially outperforms state-of-the-art methods in the most challenging OOD settings.

### Weaknesses
- The paper lacks related work of causality-inspired representation learning for OOD generalization.
  - Arjovsky et al. "Invariant risk minimization." arXiv preprint arXiv:1907.02893 (2019).
  - Lv et al. "Causality inspired representation learning for domain generalization." Proceedings of CVPR. 2022.  
- The ablation study in Table 3 is unclear; the configuration of CaSBRE (baseline) is not specified.
- The paper lacks a qualitative analysis of the learned spurious features (S), leaving it unclear what semantic information or dataset biases are actually being captured.   
- The ablation study shows that feature disentanglement alone improves performance, but the paper does not provide a rationale for why this improvement occurs over the baseline.   
- The paper does not justify the choice of marginalizing S via do-calculus over simpler alternatives, such as discarding S entirely and using only C for prediction.

### Questions
- Could you clarify the precise configurations for "CaSBRE (baseline)" ?
- How does your SSL-based approach for achieving invariance compare to other OOD generalization methods?
- Have you performed any qualitative analysis to interpret what semantic information or dataset biases the learned spurious features (S) are capturing?
- The ablation study shows that feature disentanglement alone (w/o do-calculus) improves performance over the baseline. What is the authors' hypothesis for this performance gain?
- Could you justify the choice of marginalizing S via do-calculus for inference, as opposed to a simpler alternative such as discarding S and using only C for prediction?
- Could you comment on the stability and convergence of the training procedure (Algorithm 1), particularly its sensitivity to initialization or hyperparameter choices?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a novel pipeline for relation extraction between entities (e.g., gene-disease, chemical interactions) with the goal of moving beyond spurious correlations to identify more causal relationships. The core contribution is a two-step approach: first, a Variational Autoencoder (VAE) is used to learn representations that aim to disentangle causal factors from spurious ones. Second, during inference, a marginalization step inspired by the backdoor criterion is applied to control for confounding bias. The authors claim this method improves the robustness and accuracy of relation extraction by focusing on underlying causal mechanisms. To validate their approach, they test the pipeline on several comprehensive datasets, presenting competitive results for most cases.

### Strengths
- Comprehensive Evaluations: The paper is supported by comprehensive evaluations, including ablation studies on the model's components and helpful visualizations, which add credibility to the empirical results.

- Causality-Inspired Approach: Tackling problems with huge hypothesis spaces and limited labeled data, such as relation extraction, is a promising direction, and framing the problem from a causal perspective is a usually valuable .

### Weaknesses
The paper's weaknesses can be categorized into three main areas: text and notations, novelty, and engagement with related work.

## Related Work

This is the main drawback of the paper.
* **Lack of Causal Inference Context:** For a paper centered on a causality-inspired method, the related work section is surprisingly sparse on the topic. It fails to situate the work within the vast and long-explored field of distinguishing spurious correlations from causal relations. There is no introduction to the specific definitions of causality being used or a comparison to alternative methods.
* **Misleading Focus and Claims:** The related work focuses on SSL methods for causal discovery but without sufficient citations. The claim that "Most current methods rely on supervised methods" is inaccurate (see for example the large body of literature on topics like GRN and PPI inference uses unsupervised methods on batch data.)
* **Superficial Treatment of Causal Concepts:** Key concepts like the backdoor criterion are mentioned casually, without stating the actual criteria or acknowledging the significant challenges and assumptions (e.g., causal sufficiency) required for its valid application, which are often not met in real-world data. The vast literature on the limitations of causal learning is not addressed at all.

**(c) Novelty**

* **Limited Methodological Novelty:** While the goal is commendable, the novelty of the approach is limited. The pipeline combines existing components (VAE, backdoor-inspired marginalization), but there is a major lack of a deep, rigorous dive into these methods and their known limitations.
* **Unsupported Claims of Novelty:** The paper claims the "introduction of a benchmark dataset" as a contribution, yet the datasets mentioned (chemical-protein, gene-disease) are widely used benchmarks in the community. The authors need to clarify what makes their specific formulation a novel contribution.
* **Overstated Framework Novelty:** The claim of proposing the "first general semi-supervised framework for causal interaction learning" is a significant overstatement. Causality-inspired methods are prevalent in related biomedical fields, such as GRN inference, with many existing frameworks that could be considered precedents.

## Text and Notations

* **Clarity and Structure:** The text is difficult to follow. While the motivation and preliminaries are relatively well-written, the core details of the proposed framework are repeatedly mentioned in a high-level manner up to page 4, making it hard to grasp the specifics. The main figure, which should provide a clear overview, is not introduced until page 4 and is accompanied by minimal explanation.
* **Repetition:** Repetition is a recurring issue throughout the paper (see for example the Introduction and Section 2.2).
* **Unclear Notations:** Several key notations remain ambiguous. For instance, in lines 091-095, the origin and meaning of `C` are confusing, being referred to as derived from both labeled and unlabeled data. The exact meaning of `P(C|S)` is not defined: Is `C` a subset of `S`? Are `C` and `S` distinct sets of correlations? Is the analysis performed on univariate relations or in a multivariate context? These ambiguities hinder a full understanding of the method.

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Prediction of biomedical interaction relations such as chemical-protein interactions and gene-disease associations is important for biomedical research. However, due to the scarcity and bias of annotated interaction relation, it is hard for the model to capture the generalizable features for interaction relation prediction. This paper proposes a semi-supervised learning framework to disentangle the spurious correlations from real interactions and perform additional do-calculus inference to further reduce the impact of spurious correlations on the interaction relation prediction. The experimental results show that the proposed CaSBRE method outperforms other supervised and semi-supervised models in many settings, especially in setting with unseen biomedical entities.

### Strengths
(1) The paper is well written and well organized. It provides clear definition of the problem, clear explanation of motivation, and enough background knowledge to make the paper easy to follow.

(2) The CaSBRE method proposed in the paper is formalized rigorously and technically sound. The feature disentanglement network and causal interaction inference are described with both mathematical formalizations and corresponding algorithms.

(3) The paper conducts ablation study and addition experiments to verify the learning result of the proposed model is consistent with the design objective, which strengthens the paper quality.

### Weaknesses
There are still some inconsistencies in the use of mathematical symbols. It would be great to make the symbols defined in equations and the ones referred in the context the same.

### Questions
(1) In the definition of $Err^l_{\mathcal{SC}}$ and $Err^u_{\mathcal{SC}}$, whether the $2$ in the superscript means square or should be placed to subscript to represent L2-norm.

(2) Please make it more clear that how $S^{k_x}_x$ and  $S^{k_w}_x$ are sampled from $X$ and $W$.

(3) The current formalization of Algorithm 1 will lead to used of undefined symbols $\mathbf{De}_{x_u}$,

$\mathbf{De}_{w_u}$,

$\mathbf{En}_{x_u}$,

and $\mathbf{En}_{w_u}$. Please clarify that to make the Algorithm 1 strict.

### Soundness
3

### Presentation
3

### Contribution
3
