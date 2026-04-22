# I2Mole: Interaction-aware Invariant Molecular Learning For Generalizable Property Prediction

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 4

## Abstract
Molecular interactions are a common phenomenon in physical chemistry field, which could produce unexpected biochemical properties harmful to humans, such as drug-drug interactions. Machine learning has the potential to deliver rapid and accurate predictions. However, the complexity of molecular structures and the diversity of molecular interactions could undermine model prediction accuracy and hinder generalizability. In this context, identifying core invariant substructures (\textit{i.e.}, rationales) has become essential for enhancing interpretability and generalization. Despite notable efforts, existing models often neglect the molecular pairs’ modeling, leading to insufficient capture of interaction relationships. To address these limitations, we propose a novel framework, \textbf{I}nteraction-aware \textbf{I}nvariant \textbf{Mole}cular learning (I2Mole), for generalizable property prediction. I2Mole meticulously models atomic interactions such as hydrogen bonds by initially establishing indiscriminate connections between intermolecular atoms, which are subsequently refined using an improved graph information bottleneck theory tailored for merged graphs. To further enhance model generalization, we construct an environment codebook by environment subgraph of the merged graph. This approach not only could provide noise source for optimizing mutual information but also preserve the integrity of chemical semantic information. By comprehensively leveraging the information inherent in the merged graph, our model accurately captures core substructures and significantly enhances generalization capabilities. Extensive experimental validation demonstrates the efficacy and generalizability of I2Mole. The implementation code is available.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
I2Mole predicts drug–drug interactions by constructing a dense merged molecular graph that connects every atom in one drug to all atoms in its partner, followed by attention-guided pruning to retain only the most informative inter-molecular edges. A Graph Information Bottleneck (GIB) module then compresses the pruned graph into an invariant “core substructure”, while a vector-quantized environment codebook captures contextual variations across molecular pairs. Trained end-to-end, I2Mole jointly optimizes node embeddings, attention-based pruning, information compression, and codebook assignments. Experiments show strong performance across various DDI benchmarks, especially in out-of-distribution settings.

### Strengths
1.	The approach introduced in this study is well-structured, with clear methodology.
2.	The experimental section is comprehensive and solid, encompassing benchmark evaluations, interpretability analyses, and detailed examinations of the codebook module.
3.	Using GIB and invariance learning to model inter-molecular interactions makes the model more interpretable. 
4.	Compared to many existing approaches, the proposed method shows stronger performance and better generalization across three DDI prediction benchmarks.

### Weaknesses
1.	The work does not report results on widely used DDI benchmarks, such as DrugBank and TWOSIDES. 
2.	The evaluation is scoped to DDI only: results are limited to DDI, with no experiments on other interaction tasks. 
3.	How much do you expect external physicochemical factors such as pH or temperature to alter the significance of your results？
4.	Including an analysis of the physicochemical mechanisms underlying the modeled interactions could be added further validate the effectiveness of the proposed model.
5.	The manuscript contains a few typographical errors, which do not affect the overall quality of the work. For instance, Howevew->However ，REALTED WORK -> RELATED WORK.

### Questions
See Weaknesses.

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
The paper proposes the I2Mole framework, aiming to improve model generalization and interpretability in drug–drug interaction (DDI) prediction scenarios. The framework constructs merged molecular graphs, applies an enhanced graph information bottleneck, and develops a molecular environment codebook to model atomic-level interactions and extract core substructures. By performing invariant learning on these substructures, the method enhances the model’s capability to generalize across different data distributions.

### Strengths
1. The experiments are relatively comprehensive, and the reported results demonstrate strong performance.
2. The paper provides extensive visualizations, which are beneficial for achieving a clear understanding of the proposed methodological design.

### Weaknesses
1. The scientific significance of the problem addressed in the manuscript is not clearly and specifically articulated. The lack of generalization in previous methods within this domain is a common issue across many fields; however, it would be preferable if the manuscript could provide a more concrete and well-defined scientific contribution.
2. The manuscript lacks strong observational analysis. Most of the nine observations presented are general descriptions without in-depth or insightful examination. Highlighting these observations in bold as if they were substantial analyses appears inappropriate.
3. There are typographical errors in the manuscript, such as the misspelling “realted work” on page 15.
4. In the initial modeling stage, the authors establish models for every possible atom pair between the two molecules, and then perform filtering to characterize hydrogen bonds and van der Waals forces. I would like to know whether the authors have reliable justification ensuring the correctness of the hydrogen bond modeling (as this likely involves some specific filtering steps).
5. Regarding the “molecular environment codebook” mentioned in the manuscript, I would like to know what exactly is meant by “molecular environment” in this context, and what its precise definition is within the field of chemistry.

### Questions
See the weaknesses.

### Soundness
2

### Presentation
1

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
This paper proposes I2Mole, a DDI-focused framework that builds a merged intermolecular graph with atom–atom links, applies a GIB-style subgraph extractor, and augments invariance via a learned environment codebook (VQ) for robust generalization. On three DDI benchmarks, the method reports consistent gains in transductive, inductive, and domain-shift settings, with ablations supporting the roles of intermolecular interactions, GIB, and the codebook.

### Strengths
- The paper’s formulation—merged graph + GIB-based rationale + VQ environment codebook—is clearly specified theoretically and experimentally ablated. 

- Results span transductive/inductive/domain-generalization regimes with competitive numbers and reasonable sensitivity analyses.

### Weaknesses
- Please streamline the paper’s scope to drug–drug interaction prediction. The architecture (pairwise merged graph, DDI benchmarks) is DDI-specific; claims about “molecular property prediction” (generally single-molecule, regression) or broader molecualr interaction tasks (e.g., DTI) feel overstated without evidence.

- Many DDIs arise from shared or convergent pathways rather than direct atom–atom contacts. This is not well captured by purely structural intermolecular edges.

- Although the authors state I2Mole “models van der Waals and hydrogen bonds,” Eq. 5 effectively creates fully connected cross-molecule edges and top-x% pruning; without 3D interatomic geometry, this is not an explicit physical interaction model. This is an another overstated claim in this study.

- The top-x% selection for intermolecular edges is non-differentiable; it would be suggested to use a differentiable sorting relaxation (e.g., NeuralSort/SoftSort) to avoid potential gradient issues, or justify why straight-through works here.

- Several tables/figures appear inconsistent (e.g., optimal I2Mole performances on ZhangDDI differ between sensitivity and ablation/comparison tables). Please reconcile and ensure one canonical set of results across Tables 4–5/9/10/12/13 and the main comparison tables.

- For interpretability, Figure 5’s element-level analysis is coarse. Consider adding maximum common substructure (MCS) exemplars per environment to indicate whether the codeboook can actualy learn and clustering chemical structures or not.

### Questions
- There are two “moderately” words on p.2 line 99 (typo).

- What exact pair embedding is used for the t-SNE plots?

- Which dataset underlies Figure 5a?

- Please list the bond/edge features used for intermolecular connections.

- Model names are missing in Tables 12 and 13—please restore.

- How exactly are scaffold and size splits implemented?

- For interpretability, specify the matching criteria for “DDI-level, perpetrator-level, frequent functional group,” and how key substructures are selected.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes I2Mole, an interaction-aware, invariant molecular relational learning framework for predicting properties of molecular pairs (with a focus on drug–drug interactions).

The key ideas are:
1. build a merged graph that explicitly connects atoms across the two molecules and run intra- and inter-molecular message passing to model atomic interactions
2. extend the Graph Information Bottleneck (GIB) idea to extract an invariant rationale (core subgraph) from the merged graph via controlled noise injection
3. learn a discretized environment codebook (via vector quantization) that represents non-core environmental contexts and is used to encourage invariance / domain robustness.

The authors evaluate on three DDI datasets (ZhangDDI, DeepDDI, ChChMiner) and report consistent gains over baselines in transductive, inductive and domain-shift experiments.

### Strengths
* Well-motivated idea. Explicitly modeling inter-molecular atomic interactions addresses a real limitation of existing single-molecule GNNs in pairwise tasks.
* Novel combination of techniques. Extending GIB with a learned VQ codebook for structured noise injection is conceptually interesting and empirically effective.
* Thorough evaluation. The experiments are comprehensive, including multiple datasets, settings, ablations, and statistical significance checks. The fact that performance gains are consistently observed on all settings is especially impressive.

### Weaknesses
* Marginal performance improvement. The proposed model is relatively heavy compared with baselines, and some gains (especially on larger datasets) are modest. It’s unclear whether improvements stem mainly from architectural bias or model capacity.
* Presentation quality. The paper is dense and difficult to follow in several parts. Important design choices are buried in the appendix or described briefly. A clearer, higher-level walkthrough and cleaner notation would greatly improve readability. There are also some typos and grammatical errors throughout the text, such as 98-99 "moderately and moderately", 182-183 "$u_i^{\'}$", etc.

### Questions
* Could you add some kind of error metrics (such as the standard deviation)  to the reported performance numbers, by training and inferencing the models multiple times with different seeds?
* Could you include parameter-matched baseline comparisons or lighter model variants to better assess the cost–performance tradeoff?
* There are many typos and grammatical errors throughout the text. A careful proofreading could help.
* Please fix the styles for citations (do use `\cite`, `\citet` and `citep` wisely).

### Soundness
3

### Presentation
2

### Contribution
2
