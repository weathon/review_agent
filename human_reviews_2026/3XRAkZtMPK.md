# Temporally Detailed Hypergraph Neural ODE for Disease Progression Modeling

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6, 4

## Abstract
Disease progression modeling aims to characterize and predict how a patient's disease complications worsen over time based on longitudinal electronic health records (EHRs). For diseases such as type 2 diabetes, accurate progression modeling can enhance patient sub-phenotyping and inform effective and timely interventions. However, the problem is challenging due to the need to learn continuous-time progression dynamics from irregularly sampled clinical events amid patient heterogeneity (e.g., different progression rates and pathways). 
Existing mechanistic and data-driven methods either lack adaptability to learn from real-world data or fail to capture complex continuous-time dynamics on progression trajectories. To address these limitations, we propose Temporally Detailed Hypergraph Neural Ordinary Differential Equation (TD-HNODE), which represents disease progression on clinically recognized trajectories as a temporally detailed hypergraph and learns the continuous-time progression dynamics via a neural ODE framework. TD-HNODE contains a learnable TD-Hypergraph Laplacian that captures the interdependency of disease complication markers within both intra- and inter-progression trajectories. Experiments on two real-world clinical datasets demonstrate that TD-HNODE outperforms multiple baselines in modeling the progression of type 2 diabetes and related cardiovascular diseases.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This study introduces a novel method that combines Neural ODE and GNN–based for interpretable trajectory generation.
This work uses hypergraph neural network to capture relational dependencies among variables and uses neural ODE to learn continuous dynamics.
The authors further provide theoretical support ensuring representational validity.
Experiments demonstrate that the method achieves accurate trajectory reconstruction and offers interpretable latent dynamics.

### Strengths
1. It combines Neural ODEs and GNNs to effectively capture both temporal and spatial dependencies. 

2. It improves over the traditional fixed hyperedge weight matrix by using a learnable hyperedge weight matrix. 

3. By modeling continuous dynamics and graph relations, the model offers interpretable latent representations and visualizable trajectories that help in analyzing its validity. 

4. The experimental section shows superior reconstruction and forecasting accuracy compared with baseline methods.

5. It provides visualization for the generated trajectories.

### Weaknesses
1. This paper lacks a comprehensive investigation of Transformer based model for irregularly-sampled event sequence.

For Transformer models, it should include but not limited: 

[1] Learning the natural history of human disease with generative transformers. Nature 2025. 

For linear Transformer, it can be considered as discreted ODE, for instance:

[2] TrajGPT: Irregular Time-Series Representation Learning of Health Trajectory. IEEE J-BHI 2025. 

I just list few recent works. You could find more related works which should be included in the literature review.

2. Although this paper claims that it can generate interpretable trajectory, it has limited data analysis about the generated trajectories. It should include more case studies about how a patient's health progress over time. It could also include population-level analysis about the subphenotypes or combritidy. 

3. It lacks interpretation and visualization of the learned token embedding (like HP, AF). We do not know whether model learns meaingful embedding or not in the graph-based model.

### Questions
Extending the weakness 2. 

1.  While the motivations mentions subpheontypes, did the author try to analyze it? whether it is connected the different progression speed in Fig.4. 

2. This work focuses on T2D and selects features to analyze it. Did the author analyze the correlation or combritidy between T2D and other features? whether T2D will also contribute to other diseases (like heart failure)

3. While the background mentions medication and talks about "timely treatment", it does not have any thing about the interactions between disease progression and medications? we should see medication helps the disease recovery? 

4. Some symptoms are irreversible. I am wondering whether the medictions can help stabalize it? It is interesting that the authors mention irreversible symptoms but it does not make analysis in the result section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces TD-HNODE, a disease progression model that integrates clinical knowledge into a temporally detailed hypergraph combined with Neural ODE. Each node represents a disease complication marker, and each hyperedge is a predefined clinically validated progression trajectory. For disease modeling, the authors propose a time-adaptive Laplacian that governs continuous-time diffusion of latent marker states, comprising an attention-based incidence matrix for patient-specific, time-aware weighting of markers and a learnable hyperedge weight matrix. Experiments on two EHR datasets (University Hospital and MIMIC-IV) show that TD-HNODE improves accuracy, recall, and F1 compared with strong baselines (T-LSTM, ContiFormer, TGNE, HyperTime, CODE-RNN). Ablations support the contributions of both adaptive incidence $H_p$ and learnable weights $W_p$, and a case study suggests that the model’s latent embeddings reveal sub-phenotypes of diabetic progression.

### Strengths
1. The topic is important. Continuous-time modeling of chronic disease trajectories is an important and emerging topic. 
2. Interpretability might be good, as the hyperedges align with known clinical pathways. This might serve as a foundation for explanation and clinical validation. 
3. Consistent improvements on two EHR datasets, particularly in recall (clinically critical for early detection).

### Weaknesses
1. Over-complex and arguably unnatural construction.
While the idea of embedding medical knowledge into continuous-time dynamics is valuable, the resulting architecture feels heavily engineered. TD-HNODE stacks many modeling layers—curated trajectories $\rightarrow$ hypergraph $\rightarrow$  attention-based incidence $\rightarrow$  dense inter-trajectory weighting $\rightarrow$  ODE integration—each adding parameters without clear generative justification. It is difficult to discern whether the model captures meaningful structure or merely benefits from large capacity.
2. Everything is learnable, risking loss of inductive bias. Almost all structural components ($H_p$, $W_p$, time encodings) are trainable.
This undermines the “knowledge-infused” motivation: the learned Laplacian may diverge from curated pathways, reducing interpretability.
Regularization toward clinical priors or partial parameter freezing would help maintain domain grounding.
3. Bias compounding across multiple submodules.
The architecture effectively stacks a model on top of another (attention $\rightarrow$ self-attention $\rightarrow$ pooling $\rightarrow$ ODE $\rightarrow$ decoder). Each stage may introduce its own bias, and the composition could amplify rather than mitigate error.
It remains unclear which layer drives performance versus redundancy.
4. Questionable scalability of the hypergraph construction.
The temporally detailed hypergraph may require combining all observed markers across trajectories, potentially leading to a combinatorial explosion of hyperedges. The paper does not quantify computational or memory costs beyond brief remarks in the appendix.
Recomputing dense, time-varying $\tilde{L}(t)$ could be infeasible for large EHRs.
5. Clarity and naturalness of formulation. The path from clinical intuition to mathematical formulation is hard to follow. Although mathematically consistent, the paper could better motivate why this sequence of modeling choices is natural or necessary.

### Questions
1. Please detail how predefined trajectories were constructed (clinical source, inter-rater agreement, # of trajectories, examples). 
2. Since hyperedges are set-valued, how is order encoded beyond attention? 
3. Why not construct trajectories from actual temporally detailed sequences (e.g., frequent pattern mining, partial order mining), then regularize by clinical priors? 
4. Classical hypergraph Laplacians rely on diagonal W to ensure symmetry and PSD. With dense $W_p$, $\tilde L$ may not be symmetric/PSD unless extra constraints hold. Non-PSD can destabilize diffusion and the ODE. Please prove conditions under which $\tilde L(t)$ is symmetric/PSD with dense $W_p$. 
5. Please specify the exact initialization: $S(t_1)=\mathrm{Enc}([x(t_1), y(t_1)])$

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
TD-HNODE is a neat and technically sound synthesis -- temporally detailed hypergraph structure inside a neural ODE which is complemented with empirical improvements and ablations that attribute gains to both the attention based incidence and learnable trajectory weights.

Overall, I find the paper easy to read, well-written, and reasonably reproducible with the code provided as supplementary material. Further, it addresses a practical and relevant clinical modelling gap. The main caveats are clinical validity of the "irreversible" and pathway assumptions, narrow evaluation metrics for deployment, and baseline coverage -- which I would like the reviewers to clarify further.

### Strengths
Overall, the paper is well-motivated and tackles a problem of high relevance; namely addressing continuous-time disease progression with irregular visits and aligns modeling with clinically recognized pathways.

Summarising the positives below,

- Methodological novelty with clear mechanics. TD-HNODE combines a Neural ODE with a temporally detailed hypergraph: an attention-based, time-aware incidence matrix and a learnable hyperedge weight matrix to form a TD-Hypergraph Laplacian that drives ODE dynamics. This is a neat way to infuse high-order, pathway-level structure into continious modelling.
- Good experimentation including two real-world EHR datasets with patient-wise splits; additional cardiovascular disease experiments suggest some generality beyond diabetes which could further enhance the method applicability.
- Clear gains vs strong baselines + informative ablations. TD-HNODE tops Accuracy/Recall/F1 across T-LSTM, ContiFormer, TGNE, etc.
- Code is supplied in the supplementary material which greatly enhances reproducibility and dissemination of the work.

### Weaknesses
On the weaknesses for the paper can be summarised below,

- Labeling/trajectory assumptions may be clinically brittle. All markers are treated as irreversible first-occurrence events; including lab states like "HbA1c High/Low", "Poor Lipid/BP". This monotonicity simplifies modeling but risks mischaracterizing fluctuating conditions and inflating cumulative positives which can inflate results.
- Metric story is narrow for a clinical setting. Results emphasize Recall/Accuracy and macro F1; there’s no AUROC/AUPRC, calibration, decision-curve or time-to-event analysis. This is rather important in this setting as there are class imbalances and early-warning goals. Overall precision remains relatively low (e.g., 31.8% on MIMIC-IV), so false-positive indidence remains unclear.
- Baseline customization may under-serve competitors. Graph baselines are forced into pairwise graphs (breaking hyperedges), while discrete-time hypergraph baselines are snapshot-based. That’s fair to compare paradigms, but the paper should also include at least one continuous-time hypergraph variant.

### Questions
The questions that I have for the authors are described in the weaknesses of the paper that I mentioned, further I would like the authors to clarify the following,

- Tables report mean/std but not CIs/significance tests, and there’s no cross-site validation - why is that?
- There is also no analysis of shift when varying hypergraph definitions, and limited reporting on class prevalence which is key for clinical claims?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the prediction of disease progression from patient encounter data, incorporating risk factors such as medications, laboratory test results, and vital signs. The authors propose TD-HNODE, which combines a neural ODE and a hypergraph neural network, where Learnable TD-Hypergraph Laplacian plays a key role to make the model more data-driven and adaptable to patient-specific disease trajectories while maintaining its clinically verified nature. The proposed method was evaluated on two real-world EHR datasets and consistently outperformed baselines.

### Strengths
-- Learnable TD-Hypergraph Laplacian is a reasonable enhancement for the combination of neural ODE and hypergraph neural network, where Attention-based Indicence Matrix adjusts the degree of attributions of v in e flexibly according to the context, and Learnable Hyperedge Weight Matrix captures data-driven similarities between trajectories.

-- Experimental results on multiple datasets demonstrated the effectiveness of the proposed method.

-- The case study is interesting and practically important.

### Weaknesses
-- Clarity issues:
* In l.100, what is k?
* In l.147, The authors mentioned "we use the terms ‘hyperedge’, ‘pathway’, and ‘trajectory’ interchangeably", but this makes descriptions confusing. For example, p_j and e_j look the same, so we should use only one of them consistently throughout the paper.
* In Eq.2, LHS is better to be f(t,S(t),x(t);\Theta) not dS(t)/dt.
* In Eq.3, I is not defined.
* In the descriptions starting from l.242, e is used like an index of edge, but it was j until then. The index for the edge should be j, and the edge itself is denoted as e for consistency.
* In l.265, the temporally detailed hyperedge should have u on superscript.
* In l.269, e may not be the index for O and F, maybe. For indexing, only j is enough.

### Questions
Nothing.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the authors propose TD-HNODE that models disease progression along clinically recognized trajectories by constructing a temporally detailed hypergraph and capturing continuous-time progression dynamics through a neural ODE framework.

### Strengths
1. This provides a novel modeling method for EHR.

2. The figure is clearly diagrammed, and the notation table enhances the readability.

3. Both open-source and closed-source datasets are evaluated, validating its practice.

### Weaknesses
1. It is limited to evaluating the proposed method on only one category of EHR, i.e., type 2 diabetes. Other medical scenarios, e.g., Alzheimer’s disease, Parkinson’s disease, and chronic kidney disease (CKD), mentioned by the authors, are ignored. 

2. The evaluation lacks soundness. Firstly, it is encouraged to involve the doctors' diagnosis by comparison. Secondly, the ODE steps need to extend to evaluate the robustness of TD-HNODE.

3. The writing is quite informal, e.g., we use the terms ‘hyperedge’, ‘pathway’, and ‘trajectory’ interchangeably.

4. The crucial baseline that models graph ODE, is ignored, e.g., HOPE[1].

5. The induction of equation 11 is absent. Why can an ODE capture this dynamic Laplacian matrix? 

6. The related work is not explicitly illustrated.

[1] Luo, Xiao, et al. "Hope: High-order graph ode for modeling interacting dynamics." International conference on machine learning. PMLR, 2023.

### Questions
See the weakness.

### Soundness
2

### Presentation
3

### Contribution
3
