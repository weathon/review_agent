# STAR: Boosting Time Series Foundation Models for Anomaly Detection  Through State-aware Adapter

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 6, 4, 2

## Abstract
While Time Series Foundation Models (TSFMs) have demonstrated remarkable success in Multivariate Time Series Anomaly Detection (MTSAD), in real-world scenarios, many time series comprise not only \textit{numerical variables} such as temperature and flow, but also numerous discrete *state variables* that describe the system status, such as valve on/off or day of the week.
Existing TSFMs often overlook the distinct categorical nature of state variables and their critical role as conditions, and typically treat them uniformly with numerical variables. This inappropriate modeling approach prevents the model from fully leveraging state information and even leads to a significant degradation in detection performance after state variables are integrated.
To address this critical limitation, this paper proposes a novel **ST**ate-aware **A**dapte**R** (STAR). STAR is a plug-and-play module designed to enhance the capability of TSFMs in modeling and leveraging state variables during the fine-tuning stage. Specifically, STAR comprises three core innovative components: (1) *Identity-guided State Encoder* effectively captures the complex categorical semantics of state variables through a learnable *State Memory*. (2) *Conditional Bottleneck Adapter* dynamically generates low-rank adaptation parameters conditioned on the current state, thereby flexibly injecting the influence of state variables into the backbone model. (3) *Numeral-State Matching* module effectively detects anomalies inherent to the state variables themselves. Extensive experiments conducted on real-world datasets demonstrate that STAR can improve the performance of existing TSFMs on MTSAD.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes a novel approach called STAR (STate-aware AdapteR) to enhance Time Series Foundation Models (TSFMs) for Multivariate Time Series Anomaly Detection, particularly in real-world industrial scenarios where time series consist of both numerical variables and discrete state variables that describe system statuses. Specifically, STAR introduces three core components: first, the Identity-guided State Encoder, which captures the complex semantics of state variables through a learnable state memory; second, the Conditional Bottleneck Adapter, which dynamically generates low-rank adaptation parameters based on the current state, allowing for flexible incorporation of state information into the TSFM; and third, the Numeral-State Matching module, which uses contrastive learning to improve anomaly detection for both numerical and state variables. Experimental results on several real-world datasets demonstrate that STAR significantly improves the performance of existing TSFMs in anomaly detection.

### Strengths
S1. The paper addresses a significant gap in existing Time Series Foundation Models (TSFMs) by recognizing the importance of state variables, which are often overlooked or treated uniformly with numerical variables. By introducing STAR (STate-aware AdapteR), the authors propose a novel method that more effectively handles categorical state variables and their conditional influence on numerical variables, improving anomaly detection in complex real-world scenarios.

S2. STAR is designed as a plug-and-play module that can be seamlessly integrated into existing TSFMs during the fine-tuning stage. This makes it a practical solution for a wide range of applications without the need for completely reworking existing models.

S3. The paper demonstrates the effectiveness of STAR through extensive experiments on multiple real-world datasets with state variables. The results consistently show that STAR improves the performance of TSFMs, particularly in complex scenarios where state variables are essential.

### Weaknesses
W1. While the paper includes an ablation study to evaluate the performance of different modules in STAR, it lacks an experiment that isolates the Identity-guided State Encoder (ID-SE) and replaces it with a simpler model like an MLP. Analyzing this would help to better understand the specific contribution of the ID-SE and whether its complexity is justified or if a simpler approach could yield comparable results, which could provide deeper insights into the trade-offs in model design.

W2. The paper does not mention some contrastive-based methods such as  TFMAE[1] and CTAD[2].

W3. There is a minor typographical error in line 270, where "R, B" is used, but it should be "R, D." Such typographical mistakes can cause confusion, especially in technical papers. Ensuring the accuracy of notation and consistently following the same terminology would enhance the paper’s clarity and professionalism.

[1] Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection

[2] Contrastive Time-Series Anomaly Detection

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the difficulty of the TSFM model in effectively handling discrete state values during anomaly detection. It proposes an efficient method for representing discrete state–type numerical features and using them as conditions in a conditional information bottleneck module to fine-tune the TSFM model.

### Strengths
1. This paper addresses an important practical problem encountered in time-series anomaly detection models — how to effectively handle discrete state variables.

2. Overall, the paper is logically coherent, clearly presented, and well-structured.

3. Experimental results demonstrate that the proposed method effectively improves the accuracy of anomaly detection in the TSFM model.

4. The design that employs an MoE-like structure for state representation, along with the integration of LoRA and the conditional information bottleneck within the conditional network, exhibits a certain degree of novelty.

### Weaknesses
1.The description of matrix operations such as aggregating, averaging, and variance computation along a specific dimension is unclear. Readers must infer which dimension the operation applies to based on the data shape, which makes the paper somewhat difficult to follow.

2.When applying the conditional information bottleneck, the loss function does not include any mutual information optimization term related to this network. This raises concerns about whether the method can reliably ensure that the conditional information bottleneck compresses irrelevant information while preserving relevant information.

3.A key but insufficiently discussed concept in the paper is the distinction between variable identities and state identities. The authors present this distinction as one of the core innovations and contributions, yet they do not clearly define what state identities and variable identities mean. Since these are not standard terms in the field, they require detailed clarification.

### Questions
1. Could you please discuss why the conditional information bottleneck can effectively preserve the relevant information while compress irrelevant ones without the mutual information loss term?

2. Could you please discuss what the variable identities and state identities mean?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes STAR (STate-aware AdapteR), a plug-and-play module to enhance TSFMs for multivariate time series anomaly detection involving both numerical and discrete state variables. STAR includes (1) an identity-guided state encoder for categorical semantics, (2) a conditional bottleneck adapter for state-conditioned adaptation, and (3) a numeral–state matching module for state-related anomaly detection. Experiments on real-world datasets show that STAR improves TSFM performance.

### Strengths
The paper presents a clear and original perspective by emphasizing the overlooked distinction between numerical and discrete state variables in multivariate time series anomaly detection. This problem formulation is both valid and practically significant, as many real-world industrial systems contain mixed-variable types. The proposed STAR module is a well-motivated and technically sound solution that effectively integrates state-aware adaptation into existing TSFMs. The experimental evaluation is comprehensive and convincing, covering multiple real-world datasets and various TSFM backbones, consistently demonstrating performance improvements.

### Weaknesses
1. The model complexity deserves further discussion. Compared with lightweight adaptation methods such as LoRA, the proposed STAR introduces additional components (state encoder, conditional adapter, and matching module), which may increase the computational cost and implementation complexity.
2. The issue of handling numerical versus discrete state variables, while valid, might represent a relatively minor challenge within TSFM-based anomaly detection. Theoretically, a sufficiently generalizable TSFM could already accommodate such heterogeneity without explicit modeling. From the experimental results, the performance gains are not always substantial, particularly for combinations such as Moment + SMAP and Timer + SMAP. Therefore, whether such a design trade-off is justified in real-world deployment scenarios warrants further consideration.
3. The experimental implementation details are somewhat incomplete. For instance, in line 329, the settings and balancing strategies for the hyperparameters λ₁ and λ₂ are not clearly explained. A more detailed discussion would enhance the reproducibility and interpretability of the results.

### Questions
Does the proposed method require explicit input type information, i.e., predefined knowledge of which channels are numerical variables and which are discrete state variables? If so, how sensitive is the performance to potential misclassification or ambiguity of variable types?

### Soundness
2

### Presentation
3

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
The paper addresses a limitation of existing Time Series Foundation Models (TSFMs), which typically overlook the state variables associated with time series data. The authors propose integrating the learning of state variables into TSFMs and explore several architectural designs to optimise this integration.

### Strengths
•	The paper provides a clear rationale for why careful model design is necessary to effectively leverage state variables.

•	Experimental evidence supports the authors’ claim that naively unifying state-variable modelling within TSFMs does not improve performance.

•	The adapted STAR approach demonstrates performance gains, supporting the value of the proposed design.

### Weaknesses
•	The novelty of some technical components is limited. Several methods are described as being inspired by prior work (e.g., LoRA and Shentu et al., 2024), but the precise distinctions from these works are not clearly explained.

•	Certain technical details remain ambiguous. For instance, how do dynamic masks work in your model?

•	Some variables in the equations are undefined, making parts of the formulation difficult to follow.

•	The paper lacks runtime or complexity analysis. Since the CB Adapter generates a new matrix R and vector D for each patch, quantifying the associated computational overhead would make the efficiency claims more convincing.

### Questions
- In Eq. (1) (lines 213–215), not all notations are defined. For example, it is unclear whether i denotes the variable index or the time index (assumed to be the variable index).

- In Eq. (5) (lines 234–235), the second denominator avg(E_sel) may be a typo and should possibly read avg(E_imp) instead.

- How is the dynamic mask incorporated into the proposed method? The current description only provides formulas without an intuitive explanation. A short textual or diagrammatic illustration would help.

- Both Figure 2 (lines 162–182) and Section 3.2 (lines 265–289) show the CB Adapter applied to a generic weight matrix W_0, but it is unclear to which specific layers this adapter is attached. Please clarify where within the model architecture the CB Adapter is inserted.

- Appendix B.2 presents a t-SNE example. Including additional visualisations, for example, adapter activations or differences in anomaly scores across states, would provide deeper insight into how state information influences model behaviour.

### Soundness
2

### Presentation
2

### Contribution
2
