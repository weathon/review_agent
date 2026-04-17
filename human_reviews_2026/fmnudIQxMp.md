# DT-BEHRT: Disease Trajectory-aware Transformer for Interpretable Patient Representation Learning

- Decision: Reject
- Scores: 4, 4, 4, 4, 4, 2

## Abstract
The growing adoption of electronic health record (EHR) systems has provided unprecedented opportunities for predictive modeling to guide clinical decision making. Structured EHRs contain longitudinal observations of patients across hospital visits, where each visit is represented by a set of medical codes. While sequence-based, graph-based, and graph-enhanced sequence approaches have been developed to capture rich code interactions over time or within the same visits, they often overlook the inherent heterogeneous roles of medical codes arising from distinct clinical characteristics and contexts. To this end, in this study we propose the Disease Trajectory-aware Transformer for EHR (DT-BEHRT), a graph-enhanced sequential architecture that disentangles disease trajectories by explicitly modeling diagnosis-centric interactions within organ systems and capturing asynchronous progression patterns. To further enhance the representation robustness, we design a tailored pre-training methodology that combines trajectory-level code masking with ontology-informed ancestor prediction, promoting semantic alignment across multiple modeling modules. Extensive experiments on multiple benchmark datasets demonstrate that DT-BEHRT achieves strong predictive performance and provides interpretable patient representations that align with clinicians’ disease-centered reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies the problem of electronic health records modeling and introduces a new pretrained framework including a tailored architecture and two pretrained learning objectives. The proposed architecture explicitly encodes both the hierarchical ontology of diagnosis codes and the temporal dynamics across patient visits through specialized modules. Extensive experiments on three downstream clinical prediction tasks across two benchmark datasets demonstrate the effectiveness of the proposed approach.

### Strengths
- The framework leverages multiple inductive biases inherent in EHR data, including the hierarchical ontology of diagnoses, the unordered nature of intra-visit medical codes, and the temporal dynamics across visits.
- The experimental evaluation is comprehensive, including multiple downstream tasks, promising performance, and complete ablation studies.

### Weaknesses
- The core contribution of this paper lies in explicitly modeling diagnosis codes rather than treating all medical codes uniformly through a shared attention mechanism. This improvement appears engineering-driven, and I am not entirely sure whether the novelty reaches the methodological bar expected at ICLR.
- From an architectural perspective, the model seems to lack a dedicated mechanism for handling long sequences. Concatenating all medical codes across all visits into a single token sequence can introduce significant computational overhead, especially considering that ancestor-level tokens for diagnoses are additionally inserted.
- Regarding the disease progression module, most cases in MIMIC involve only a single visit. Therefore, the resulting heterogeneous graph degenerates into a star-shaped structure (a visit node connected to its diagnosis nodes). Multi-layer GAT propagation under such a structure may lead to over-smoothing and potentially reduce to a simple attention pooling over diagnosis codes. Since diagnostic tokens already participate in full self-attention within the sequence encoder, the practical benefit of this component under single-visit settings remains uncertain.
- The motivation behind using ancestor prediction as a pre-training objective is not clear. My understanding is that the hierarchical ontology provides a prior that codes within the same ancestor category share semantic proximity, but predicting ancestor nodes does not really increase the learning difficulty. In principle, if the model can correctly predict a diagnosis code, it should also be able to infer its ancestor. This is also reflected in the ablation results.
- As for the presentation, it would be great to simplify the notation and refine the visualizations. The current version is not sufficiently intuitive.

### Questions
See weakness.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
Outcome prediction from longitudinal electronic health records is a well established direction. Many models try to use past hospital visits to predict clinically important outcomes such as mortality, prolonged length of stay, readmission and future phenotypes.
This paper proposes DT-BEHRT, which learns patient representations by explicitly separating two kinds of structure in the record. First, it aggregates diagnosis codes by clinical system level for example cardiovascular or respiratory systems. Second, it builds a temporal progression graph over visits to model how disease states evolve over time. The final patient embedding combines both the system level view and the progression view for downstream prediction tasks such as mortality, readmission and phenotyping.

### Strengths
The presentation is generally clear and well structured.

The paper evaluates on several standard clinical prediction tasks and includes ablations that try to isolate the contribution of the proposed modules, suggesting that different components benefit different tasks.

### Weaknesses
I am mainly skeptical about the interpretability claims. The paper treats interpretable patient representation as one of its main selling points, but interpretability is not actually used as a rigorous design objective, nor is it quantitatively evaluated. The paper still does not answer basic questions such as: how are the explanations generated, and how is the causal link between the explanation and the prediction verified. The authors state that the architecture mirrors clinician reasoning, but I find this interpretation unconvincing. How is that demonstrated. Was any clinician asked to assess it.

In addition, the discussion of interpretability in the main text is almost entirely limited to the single case visualization in Section 4.4, rather than being developed systematically in the method. Many prior works have already used attention or weight visualizations to argue interpretability. That line of argument is neither new nor convincing here, because attention is fundamentally just highlighting features the model considers important, which is something any model must do internally anyway. Also, the case study presents only one patient example, which is not persuasive.

There are missing details in the experimental setup. For example, the definition of the readmission task is not clearly specified. It is not stated over what future horizon readmission is being predicted. Likewise, mortality is not defined in terms of prediction window or observation window. These choices control the actual clinical meaning of the task and strongly affect how results should be interpreted. The paper also does not clearly state whether all baselines are trained and evaluated under exactly the same task definitions and cohort construction.

Section 4.3.2 states that DT BEHRT consistently achieves the best performance on both the full cohort and the subgroup of patients with three or more hospital visits. However, Table 2 shows that for many specific phenotype categories, ExBEHRT and HEART actually outperform DT BEHRT. This suggests that the reported improvements may involve a fair amount of outcome specific variability rather than a uniform advantage.

Both datasets used in the study are MIMIC III and MIMIC IV. These are standard public critical care datasets, but they are collected from the same hospital system, and there is known overlap in patient population and clinical practice patterns between them even though they cover different time periods. The paper treats results on MIMIC III and MIMIC IV as if they were evidence of generalization across settings, but in reality they are two cohorts from essentially the same site. The added value of reporting both without any external validation set from a different institution is therefore limited.

### Questions
See weaknesses.

### Soundness
3

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
4

### Summary
This paper proposes DT-BEHRT, a graph-enhanced sequential model for structured EHR. The method distinguishes the roles of medical code types, aggregates diagnosis codes at the organ-/system-level via Disease Aggregation (DA), models cross-visit disease evolution via a heterogeneous graph Disease Progression (DP), and combines them with a pretraining scheme (GCMP + ACP) to align representations across modules. The approach is motivated by limitations of purely sequential models and of graph-only models in handling intra-visit code order and cross-visit dependencies, and it targets general outcome prediction and phenotyping.

### Strengths
S1. The paper proposes DT-BEHRT, a graph-enhanced sequential model for EHR. It separates code roles, aggregates diagnoses at organ/system level with DA, and models cross-visit evolution with a heterogeneous DP graph. It adds GCMP and ACP pretraining to align modules. The goal and setup are clear in the Introduction and Methods.
S2. The method is sound. SR gives contextualized tokens. DA uses ICD-9 chapters plus masked attention and decorrelation. DP builds a visit–diagnosis graph with forward time edges and GAT. PR pools DA/DP with sequence guidance. GCMP/ACP provide structure-aware pretraining.
S3. The case study shows clear interpretability signals.  The behavior matches clinical reasoning.

### Weaknesses
W1 The paper claims that prior work ignores heterogeneity. However, HEART already models heterogeneous relations and connects visits as a graph. The paper does not specify what type of heterogeneity HEART fails to capture. It also does not show how DA and DP address that specific gap. A targeted, side-by-side comparison is needed.

W2 The contributions list “robustness of patient representations.” Robustness is not defined in the Introduction. Robustness is not analyzed in the experiments. This creates a mismatch between claims and evidence.

W3 In phenotyping with all patients, performance fluctuates. None of DT-BEHRT, ExBEHRT, or HEART is consistently first. The paper still claims it “consistently achieves the best performance.” This is too strong. Please report more metrics, such as AUROC, micro-AUPRC, F1, and calibration, and moderate the claim.

W4 Ablations show that removing pretraining makes DT-BEHRT weaker than HEART and ExBEHRT on some tasks. This suggests pretraining drives a large share of the gains. Please add a control that uses only GCMP and ACP without DA/DP. Also compare carefully against the strong baseline HEART. Since HEART has no pretraining, discuss whether HEART is structurally stronger when DT-BEHRT is also trained without pretraining.

W5 If robustness is a key contribution, the paper should add a clear robustness section. Define robustness precisely. Add targeted tests, such as higher masking rates, event dropout, subpopulation shifts, temporal splits, or label noise, with appropriate metrics.

### Questions
Please see the weaknesses.

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
This paper introduces DT-BEHRT, a graph-enhanced transformer for EHR data. It disentangles disease trajectories by explicitly modeling diagnosis-centric interactions within organ systems (DA module) and their temporal progression (DP module). A tailored pre-training strategy aligns these components, achieving strong predictive performance and interpretability.

### Strengths
* **Clinically-Aligned Architecture:** The model's DA and DP modules are designed to mirror clinical reasoning, enhancing interpretability.
* **Novel Pre-training:** The Ancestor Code Prediction (ACP) task effectively aligns the model's different modules with ontology information.
* **Strong Empirical Results:** DT-BEHRT outperforms baselines, especially on complex phenotyping and readmission tasks.
* **Targeted Ablation:** Ablation studies demonstrate the distinct contributions of the DA and DP modules to different tasks.

### Weaknesses
* **Ontology Dependence:** The Disease Aggregation (DA) module is explicitly tied to the ICD-9 ontology, which may not be adaptable.
* **Fixed Aggregation Threshold:** DA tokens are activated by a fixed hyperparameter $k$, and the impact of this choice isn't explored.
* **Incomplete Pre-train Ablation:** The ablation study does not isolate the effect of the DA token decorrelation loss ($l_{cov}$).
* **Simplistic Code Roles:** The model simplifies code roles, treating diagnoses as interactive while other codes are less so, which may be inaccurate.
* **Relatively old baselines:** Other the HEART model, the rest of the baselines seem relatively old.

### Questions
- Why ICD-9 was used? It seems an old format in 2025. 
- What’s the rationale for the chosen list of disease? How much these would be generalizable.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a novel framework, 'DT-BEHRT (Disease Trajectory-aware Transformer),' for learning interpretable patient representations from Electronic Health Record (EHR) data. The model integrates graph-based modules into a BERT-style architecture to explicitly model a patient's disease trajectory.

The core components are as follows:
(1) Disease Aggregation (DA) Module: Groups diagnosis codes based on ICD-9 chapters (organ/system level) to capture high-level semantic patterns beyond individual codes.
(2) Disease Progression (DP) Module: Uses a heterogeneous graph to model the temporal order between visits and disease development trends.
(3) Specialized Pre-training: Combines trajectory-level code masking (GCMP) with ontology-based ancestor code prediction (ACP) to enhance semantic alignment across the different modules.

The authors claim that in experiments using the MIMIC-III and MIMIC-IV datasets, DT-BEHRT demonstrated superior predictive performance compared to existing SOTA models. They also assert, through case studies, that the model provides interpretability similar to a clinician's reasoning process.

### Strengths
1. Demonstrated Modular Contribution: The framework consists of multiple modules (DA, DP) and a new pre-training task (ACP). A key strength is the use of an Ablation Study (Table 3 in the paper) to clearly demonstrate how much each component contributes to the model's performance improvement.

2. Clinical Interpretability: The model doesn't just aim for higher performance; it attempts to link the rationales for its predictions to clinical reasoning (e.g., problems in a specific organ system, temporal progression of a disease) via the DA and DP modules. This is well-visualized in the Case Study.

### Weaknesses
1. Limited Dataset Validation: The datasets used for the experiment are limited to MIMIC-III and MIMIC-IV, which is insufficient to prove the model's generalization performance. EHR data has inherent biases depending on the hospital system, country, and ethnicity. Therefore, external validation on other large-scale ICU datasets (e.g., eICU, HiRID, UMCdb) is essential.

2. Lack of Prediction Task Diversity: The variety of prediction tasks performed is insufficient to claim that the proposed framework is 'generally superior.' While LOS, Mortality, PLOS, and Phenotype prediction were used, many other clinically important tasks exist.
Specifically, Phenotype prediction is ultimately the prediction of grouped diagnosis codes, making it not fundamentally different from a diagnosis code prediction task.
Therefore, performance validation on more diverse and challenging tasks, such as sepsis prediction, acute kidney injury (AKI) prediction, or lab value prediction, is needed.

3. Insufficient Comparative Analysis with Baselines: The ablation study only shows that the modules within the proposed model are useful; it does not clearly explain why these modules are superior to the approaches used by other baseline models.
Many baselines (e.g., HEART, G-BERT) also use a combination of graph and sequence information. However, the paper's Methodology section lacks a detailed comparison of how DT-BEHRT's DA and DP modules specifically differ from existing methods and in what respects they are superior.
For example, the paper should clarify whether 'Disease Aggregation (DA)' was considered at all in previous baselines, and if so, how DT-BEHRT's approach differs. This comparative analysis needs to be strengthened.

### Questions
.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 6

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents DT-BEHRT, a disease-trajectory-aware Transformer model for learning interpretable patient representations from EHR data. It models disease evolution across visits through a modular design consisting of four modules. Moreover. the authors introduce two pretraining objectives. Experiments show consistent improvements over strong baselines on four (3+1) prediction tasks, while maintaining clinical interpretability.

### Strengths
S1. The paper conducts experiments on MIMICs across multiple tasks. Beyond quantitative metrics, the authors include patient-level case studies that qualitatively analyze model explanations. 

S2. Each module in DT-BEHRT, i.e., sequence, aggregation, progression, and patient representation, is clearly motivated from a clinician’s perspective, reflecting real-world medical reasoning.  

S3. The introduction of two pretraining tasks allows the model to fully leverage EHR data across visits and disease hierarchies. 

S4. The model’s performance on diverse phenotype prediction tasks, spanning both acute and chronic conditions, shows consistent and significant improvements.

### Weaknesses
W1. The paper integrates design patterns from both Transformer-based and graph-based models, resulting in an architecture that appears more incremental. The combination of SR–DA–PR resembles conventional Transformer stacks, with the main variation being the use of ancestor node embeddings and customized losses. Similarly, the SR–DP–PR path largely parallels prior graph-based pipelines that model interactions between disease, visit, and patient nodes. 

W2. The proposed Global Code Masking and Ancestor Code Prediction tasks closely follow prior pretraining designs from G-BERT [2]. Even though authors state, “we introduce the novel ACP task (Line 411)”, such method is highly (completely) overlapped with Sherbet published in 2023, which define an even more elegant objective. 

W3. The baseline selections prefer more on Transformer-based baselines, omitting several representative recent graph -based/graph-transformer approaches like GraphCare [1], GCT [6], RAM-EHR [4], or GT-BEHRT [3].  

W4. All experiments focus on simple classifications (binary or multi-class). The absence of multi-label tasks such as drug recommendation or diagnosis prediction limits the assessment of the model’s scalability to more complex EHR applications. 

W5. While ablation results are provided, they do not test alternative pretraining schemes from prior work. Comparing the proposed objectives against those of G-BERT [2] or Sherbet [5] would more clearly justify the necessity and effectiveness of the new pretraining design. 

W6. The model employs multi-head self-attention in the sequence encoder and graph attention propagation across both disease and visit nodes, which substantially increases computational overhead. Moreover, jointly learn multiple embeddings further raises concerns about the model’s practicality in clinical practices. The concern why most previous works avoid using both large-scale graphs and transformer together is related to efficiency problem, but DT-BEHRT combines both without discussing potential optimization or scalability issues. 

--- 

References 

[1] GraphCare (ICLR-2023) 

[2] G-Bert (IJCAI-2019) 

[3] GT-BEHRT (ICLR-2023) 

[4] RAM-EHR (ACL 2024) 

[5] Sherbet (IEEE Transactions on Cybernetics, 2023) 

[6] GCT (AAAI-2020)

### Questions
Please check the weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
2
