# MoVE: Synergistic Integration of Temporal and Cross-Variable Experts for Efficient Multivariate Time Series Forecasting

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
Multivariate time-series forecasting presents the dual challenge of modeling intricate temporal dynamics and complex cross-variable dependencies. Prevailing approaches often prioritize one aspect at the expense of the other, leading to suboptimal performance. To address this limitation, we introduce MoVE, a novel framework that synergistically integrates temporal and cross-variable modeling within a unified architecture. MoVE employs two specialized experts. A Temporal Expert for capturing long-range dependencies and a lightweight Cross-Variable Expert for modeling robust cross-variable interactions. By decoupling these components within a Mixture-of-Experts framework and optimizing them collaboratively, MoVE dynamically adapts to diverse forecasting scenarios. Extensive experiments demonstrate that our framework achieves superior performance, establishing a new paradigm for effective multivariate timer series forecasting.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes the MoVE framework for efficient multivariate time series forecasting. MoVE decouples temporal modeling from cross-variable modeling into two expert modules: a Temporal Expert based on CycleNet for periodic modeling, and a Cross-Variable Expert (RCVTE) for cross-variable modeling. These two experts are dynamically gated and fused to adapt to different types of time series. The authors also introduce Period-Aware Local Curriculum Learning (PALCL) and Robust Cross-Variable Attention (RCVA) to enhance training stability and generalization capabilities. Experiments on multiple benchmark datasets validate the performance of MoVE, achieving new state-of-the-art results on the ETTh1, Electricity, and Solar datasets.

### Strengths
1. By decoupling temporal modeling from cross-variable modeling through the MoE framework, the model's interpretability and adaptability are improved.
2. The introduction of PALCL and RCVA effectively alleviates the attention collapse problem, enhancing training stability.

### Weaknesses
1. The experimental results only achieved first place in 7 out of 16 cases, which is not very good.
2. Figure 1 is very schematic and does not well illustrate the overall architecture of the model.
3. MoVE has limited innovation, with an architecture that is very similar to the SST[1]. Moreover, there is a lack of relevant performance comparisons.
4. The core modules of MoVE (such as RCVTE and PALCL) are, to some extent, combinations and improvements of existing methods (CycleNet, iTransformer), lacking entirely original modeling mechanisms.
[1] SST: Multi-Scale Hybrid Mamba-Transformer Experts for Long-Short Range Time Series Forecasting

### Questions
See the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a novel framework, MoVE, which synergistically integrates temporal and cross-variable modelling within a unified architecture. MoVE employs two specialised experts: a temporal expert to capture remote dependencies, and a lightweight cross-variable expert to model robust cross-variable interactions. By decoupling these components within a hybrid expert framework and optimising them collaboratively, MoVE dynamically adapts to diverse forecasting scenarios. The authors experimentally validate the effectiveness of the approach.

### Strengths
The motivation is clear. The paper contends that prevalent methods often prioritise one aspect at the expense of another, resulting in suboptimal performance. To address this limitation, the paper introduces MoVE, a novel framework that synergistically integrates temporal and cross-variable modelling within a unified architecture.

### Weaknesses
1) The authors assert in the abstract that multivariate time series forecasting faces the dual challenge of modelling complex temporal dynamics and intricate cross-variate dependencies. Popular approaches often prioritise one aspect at the expense of the other, resulting in suboptimal performance. Nevertheless, numerous methods have already addressed the simultaneous modelling of temporal and variate dependencies, such as [1,2]. The innovation of the authors' approach is limited.

[1] TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables [NeurIPS24]

[2] TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable-and Time-Aware Hyper-state [ICML25]

2) Poor visualization. The authors provide no visualizations in the paper, preventing readers from observing advantages over other methods. Furthermore, ablation studies also lack visualization. 

3）The architecture diagram (i.e. Figure 1) appears rather unappealing, and the caption lacks a narrative description of the method's workflow.

4) The content of the paper is insufficient. The abstract is merely half a page long, and much of the relevant prior work has not been mentioned. Furthermore, the entire manuscript barely reaches eight pages. The details of many methods have not been described clearly.

### Questions
Please refer to the weaknesses.

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
4

### Summary
This paper proposes a novel framework called MoVE (Mixture of Various Experts), designed to address the dual challenge of simultaneously capturing complex temporal dynamics and cross-variable dependencies in multivariate time series forecasting. The core innovation of MoVE lies in its mixture-of-experts architecture, which decouples temporal modeling and cross-variable modeling into two specialized expert modules: one for capturing long-term dependencies through a temporal expert (based on CycleNet), and another cross-variable expert (the newly proposed RCVTE module). Experimental results on multiple benchmark datasets demonstrate the superior performance of MoVE, accompanied by an extensive ablation study.

### Strengths
Significance: Effectively integrating temporal modeling with cross-variable dependency learning is a well-recognized core challenge in multivariate time series forecasting. This paper directly tackles this issue.

Originality: In contrast to existing studies that adopt static fusion methods for capturing periodic patterns and cross-variable dependencies, this paper introduces a mixture-of-experts architecture that adaptively integrates both aspects. This approach is clear in its methodology and offers a novel research perspective for addressing challenges in this field.

### Weaknesses
The cross-variable dependencies modeling: The Recurrent Cross-Variable Transformer Encoder (RCVTE), mentioned multiple times in the paper, appears to be merely a module that uses a simple cross-attention mechanism to capture the dependencies between variables, with some modifications to the attention computation formula. However, many prior works (such as Crossformer in the baseline) also utilize basic cross-attention mechanisms to model inter-variable dependencies. Given that the paper emphasizes this module as one of its core innovations, I believe it would benefit from further clarification of the advantages of this module and its unique contributions compared to previous studies.

The temporal dimension modeling: The paper introduces two specific experts (the temporal modeling expert and the inter-variable relationships modeling expert), but seems to focus primarily on the latter, with little discussion on the former. The paper only briefly mentions "a temporal expert inspired by CycleNet" without providing a detailed explanation of the specific structure of this expert.

### Questions
1. Periodic Dependency: The proposed model seems to heavily rely on the recurrent, noise-free periodic vector extracted by CycleNet. Is there a risk that the model's predictions may become overly dependent on CycleNet? Additionally, could the model potentially become too reliant on the periodicity of the data itself?

2. Gated Network Weights: In the Cross-Variable Expert section, the model constructs four experts and outlines the role of each. Could the authors provide examples of the weights (g_1, g_2, g_3, g_4) of these experts in different scenarios, such as when the historical sequence exhibits periodicity or when there is no clear periodicity?

3. Curriculum Learning: The paper introduces Periodicity-Aware Local Curriculum Learning (PALCL). Could the authors further elaborate on the motivation or necessity of employing a curriculum learning strategy? Additionally, would it be possible to include an experimental group that does not use the curriculum learning training strategy in the ablation study to demonstrate the actual effectiveness of this approach?

4. Network Scale: Could the authors provide more details on the scale of the model’s network parameters, such as the total number of experts? For all the datasets involved in the experiments, does the Recurrent Cross-Variable Transformer Encoder consist of only one layer? Furthermore, when the number of channels is large (e.g., 862 channels in the Traffic dataset), does calculating channel attention result in substantial storage consumption?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes MoVE, a Mixture-of-Experts style architecture that attempts to combine a “temporal expert” (based on CycleNet) and a lightweight “cross-variable expert” (RCVTE) with a learned gating network and a Periodicity-Aware Local Curriculum Learning (PALCL) training protocol.

### Strengths
The high-level idea of combining separate modules that target temporal periodicity and cross-variable coupling is reasonable and aligns with prior modular/MoE thinking.

### Weaknesses
1.	The proposed system is largely an engineering assembly of existing pieces (CycleNet temporal module, a single-layer transformer for cross-variable interaction, a simple linear gating network, RevIN). The paper provides no compelling new modeling principle beyond “put these together and gate them.”
2.	The authors state broad superiority, but inspection shows the model achieves the best result on only four datasets/tasks
3.	Table 5 shows a limited ablation (only on six datasets) and omits a crucial ablation: the effect of PALCL itself. Given that PALCL is one of the principal methodological claims, the absence of an experiment that toggles PALCL on/off (with otherwise identical settings) is a serious oversight.
4.	The detailed architecture of Recurrent Cross-Variable Transformer Encoder (RCVTE) are not clear, the architected should be plotted.

### Questions
Provide an ablation that evaluates the model with and without PALCL
Replace RCVTE with simpler alternatives and report accuracy/computer.
Add parameter counts, FLOPs, and measured inference latency and VRAM usage to allow cost-benefit comparison.

### Soundness
2

### Presentation
1

### Contribution
2
