# PDFormer: Progressive Dual-Head Transformer for Behavioral Choice Prediction

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 6, 4, 8

## Abstract
Many applications require joint prediction of interdependent behavioral choices, yet existing models often treat each choice independently (e.g., through parallel prediction heads), overlooking the influence of one on the other. In this work, we propose Progressive Dual-Head Transformer (PDHFormer), a novel framework that performs two-step prediction: the model first estimates one choice and then conditions the second on this upstream estimate through an explicit head-to-head pathway. A shared encoder captures the common structure of two prediction tasks, while the dual-head module explicitly reflect cross-choice dependence. A gated residual mechanism integrated into the embedding layer and the dual-head modules further improves the training stability and the prediction performance.
Extensive experiments on an urban mobility behavioral choice dataset and a real-world manufacturing dataset demonstrate that PDHFormer consistently outperforms state-of-the-art machine learning models, deep tabular models, as well as parallel-head Transformer variants across multiple metrics. Moreover, our ablation study confirms that both  the proposed progressive dual-head and gated residual mechanism are key contributors to the observed gains in different prediction tasks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes PDFormer, a Transformer-based architecture for tabular behavioral choice prediction.
The model treats each feature as a token, applies self-attention across features, and uses a “progressive dual-head” design where the first head predicts an initial choice and the second head refines it conditioned on the first output.
Experiments on several tabular datasets reportedly show improved classification performance over traditional machine learning and tabular deep models.

### Strengths
Implementation appears complete and reproducible.
Empirical results on tabular benchmarks are consistent and show moderate improvement over baselines like TabNet and TabTransformer.

### Weaknesses
### **Weaknesses**

1. **Conceptual confusion in data formulation.**  
   Lines 134–135 state that each *row* represents a sample, and Eq. (6) shows only one feature vector \(x_i\), yet the predictor produces a prediction for every sample \(i\).  
   It is unclear how a single FFN operating on one vector yields a matrix of per-sample predictions.  
   In standard choice modeling, one sample corresponds to a *choice set* containing multiple items, each with its own feature vector.  
   The current formulation collapses the set structure entirely, making the modeling setup ambiguous and inconsistent with discrete-choice principles.

2. **Feature-as-token attention is semantically inappropriate.**  
   The proposed Transformer treats heterogeneous features (e.g., “price,” “region,” “time”) as exchangeable tokens and applies shared QKV projections.  
   Such feature-wise attention lacks theoretical meaning for capturing context or halo effects among *items*.  
   If the authors intend each row to represent one option, then the attention should be *item-as-token*, modeling dependencies across alternatives within a choice set, not across feature dimensions.

3. **Missing comparisons with relevant sequential choice models.**  
   Although the paper claims to address *sequential choice modeling*, it only compares against generic tabular classifiers (e.g., TabNet, TabTransformer, CatBoost).  
   No baselines from the actual sequential-choice literature are included—making it impossible to assess whether PDFormer truly advances the state of the art in modeling context-dependent decision sequences.

4. **Lack of novelty.**  
   Architecturally, PDFormer is largely a minor variant of TabTransformer with two additional feed-forward heads.  
   Beyond repackaging the feature-as-token encoder and introducing a simple conditioning mechanism, there is no substantial methodological innovation or new theoretical insight.

5. **Unclear dataset description and experimental design.**  
   The paper does not clearly describe the datasets used for evaluation—What are the features?
   Without this information, it is impossible to determine whether the proposed task setup corresponds to a valid behavioral-choice problem or a standard classification benchmark.  
   This lack of clarity further weakens the paper’s claims regarding modeling “sequential” or “context-dependent” choices.

### Questions
See Weaknesses

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces PDFormer (Progressive Dual-Head Transformer), a novel deep learning framework for the joint prediction of interdependent behavioral choices. It moves beyond standard parallel Multi-Task Learning by employing a progressive dual-head predictor that explicitly conditions the second choice ($c_2$) on the output of the first ($c_1$) through a head-to-head pathway. The architecture integrates a gated residual mechanism for stability and uses a composite loss with regularization. PDFormer achieves consistent, significant improvements over strong baselines on real-world urban mobility and manufacturing choice prediction tasks, validating the benefit of modeling directional dependency.

### Strengths
PDFormer exhibits high originality by introducing a directional multi-task approach, a necessary step beyond parallel-head architectures for truly interdependent choices. The quality of the work is evident in the model's consistent and superior performance across multiple metrics compared to a wide array of strong baselines. The paper's clarity is excellent, featuring a well-motivated design and a clear description of the gated residual mechanism and loss regularization. The inclusion of SHAP analysis further adds to the work's significance by providing valuable interpretability regarding feature importance.

### Weaknesses
The work is limited by its strong reliance on knowing the dependency order; an ablation study reversing the task order ($c_2 \rightarrow c_1$) is missing and would strengthen the claim of modeling a directional dependency. The current Dual-Head design restricts immediate application to scenarios with only two interdependent tasks; the paper should discuss the technical challenges and architecture needed for $N>2$ tasks in a dependency chain. Finally, while the manufacturing dataset is a key contribution to model robustness, the paper only presents classification results for the urban mobility data; the full quantitative results for the manufacturing dataset (including R2/PCC for regression targets) should be included in the main text.

### Questions
The authors should clarify the technical details of the head-to-head connector: what specific signal (logits, probabilities, or intermediate embedding) is passed from Head 1, and what is the dimensionality of this concatenated input for Head 2? Please include an ablation study on task order to experimentally confirm that the assumed directional dependency is indeed optimal for prediction. For broader applicability, please discuss how the progressive architecture can be extended to a chain of $N>2$ tasks and the associated challenges like error propagation. To fully support the model's claimed robustness, please provide the complete results for the manufacturing dataset in the main paper, including appropriate regression metrics if applicable.

### Soundness
4

### Presentation
4

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
This paper introduces Progressive Dual-Head Transformer (PDFormer), a Transformer-based architecture designed to predict two correlated categorical decision variables. The core idea is to move away from standard multi-task approaches that predict choices in parallel (implicitly assuming independence) and instead adopt a "progressive" two-step prediction. The model first predicts an initial choice (Choice 1). The output representation from this prediction is then explicitly passed via a "head-to-head connector" to a second prediction head, which then predicts the subsequent choice (Choice 2) to capture interdependent decision
patterns. The architecture also incorporates a gated residual mechanism to stabilize training. The authors demonstrate the effectiveness of their model on two real-world datasets, an urban mobility dataset (ride-hailing decisions) and a manufacturing dataset, showing consistent performance gains over a comprehensive set of machine learning and deep learning baselines.

### Strengths
1. The paper is well-written and easy to follow. Figure 1 provides a clear and detailed overview of the PDFormer architecture.

2. The experiment design is comprehensive. The empirical evaluation is a strength of this work. The reports of experiment results are clear. The ablation study as well as the test of the reversed prediction order provide strong evidence that modeling the true underlying causal sequence of decisions is crucial.

### Weaknesses
1. Novelty is limited. Other papers (e.g., [1-5]) also propose similar Dual-Transformer models. What are the key differences? The paper could be strengthened by more clearly positioning its contribution in the design of novel models.

2. Two datasets are not enough. More datasets could be added.

3. Generalizability issue. The paper focuses exclusively on the case of two sequential choices ($N=2$). The discussion of a longer sequence of choices ($N>2$) would strengthen the paper's contribution to border applications.

4. The improvement compared with baselines is modest. While PDFormer consistently ranks as the top model, the margin of improvement over the best baseline is modest for some key metrics. For instance, in Table 1, the ACC for Choice 2 improves from 0.8410 (TabICL) to 0.8539 (PDFormer).

[1] Dual Vision Transformer

[2] Dual Transformer for Point Cloud Analysis

[3] Dual Transformer Encoder Model for Medical Image Classification

[4] DTSyn: a dual-transformer-based neural network to predict synergistic drug combinations

[5] Analysing the Behaviour of Tree-Based Neural Networks in Regression Tasks

### Questions
1. No Reproducibility Statement. As the author guide, “authors are strongly encouraged to include a paragraph-long Reproducibility Statement at the end of the main text (before references) to discuss the efforts that have been made to ensure reproducibility”.

2. Could you provide some analysis on the computational cost (e.g., training time, inference latency) of PDFormer compared to a parallel-head Transformer baseline?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies how to find a transformer-like framework that can predict joint choices where these two decisions are correlated. Existing methods have either not-so-good performance or do not include the interaction between two choices. A new architecture is proposed to fill this gap, and various experiments have shown its efficacy.

### Strengths
Very nice written paper with solid results and clear presentation. The identified research question is clear, and the experiments are sufficient.

### Weaknesses
- The problem is well-established, and the model architecture is a variant of the parallel version.
- Minor weakness mainly in writing. Please refer to the questions section.

### Questions
- The title/name of the transformer makes me think of this paper: https://arxiv.org/pdf/2301.07945, which has the same name but a different model. Perhaps you may consider a slight twist to identify your model better?
- in line 147-160, it might be better to write each component in a consistent manner ("the" and "an" were both used)
- What is the reason to choose GELU activation?
- I also wonder how this framework can be applied to multiple joint choices  (more than 2).

### Soundness
3

### Presentation
4

### Contribution
3
