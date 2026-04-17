# TimeSeg: An Information-Theoretic Segment-Wise Explainer for Time-Series Predictions

- Decision: Accept (Poster)
- Scores: 6, 6, 4

## Abstract
Explaining predictions of black-box time-series models remains a challenging problem due to the dynamically evolving patterns within individual sequences and their complex temporal dependencies. Unfortunately, existing explanation methods largely focus on point-wise explanations, which fail to capture broader temporal context, while methods that attempt to highlight interpretable temporal patterns (e.g., achieved by incorporating a regularizer or fixed-length patches) often lack principled definitions of meaningful segments. This limitation frequently leads to fragmented and confusing explanations for end users.  As such, the notion of segment-wise explanations has remained underexplored, with little consensus on what constitutes an *interpretable* segment or how such segments should be identified. To bridge this gap, we define segment-wise explanation for black-box time-series models as the task of selecting contiguous subsequences that maximize their joint mutual information with the target prediction. Building on this formulation, we propose TimeSeg, a novel information-theoretic framework that employs reinforcement learning to sequentially identify predictive temporal segments at a per-instance level.  By doing so, TimeSeg produces segment-wise explanations that capture holistic temporal patterns rather than fragmented points, providing class-predictive patterns in a human-interpretable manner. Extensive experiments on both synthetic and real‑world datasets demonstrate that TimeSeg produces more coherent and human-understandable explanations, while achieving performance that matches or surpasses existing methods on downstream tasks using the identified segments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces TimeSeg, an explanation method to highlight segments in a (univariate) time series that contributes highly to the prediction. The method is in contrast of the previous approaches that highlight individual observations in a time series. The method leverage the use of mutual information and reinforcement learning.

### Strengths
1) The paper is written quite well. The flow and logic are mostly clear and notations are good as well.
2) The premise of the paper is clear and reasonable.
3) The results of TimeSeg seems to be great. The selection of metrics rae pretty straightforward and standard.

### Weaknesses
1) There are some parts related to RL towards the end of section 4 that may not be clear to non-RL-experts, like me. The overall summary of the method is not present in the main text. (Sections in the Appendix provides some details make it easier to understand.) Details are outlined in the "Questions" box below.
2) (This is not necessarily a weakness) While the premise of the paper is good, as shown in the abstract, "existing explanations focus on point-wise explainations, we introduce segment-wise explanations.", there is a lack of "interpreting" that segment - the method on "highlight" the segment. The paper introduces an method of explanation, with its output similar to previous approach. Thus this paper does not aim for higher level interpretations of the predictions. This means that the paper is not really "ground-breaking".

### Questions
As I am not familiar with RL at all, the questions I have may be a little basic. I would like to first summarize my understanding of the method (outlined in Section 4 and the Appendix)

I. Our goal is to find segments $s_{1:K}$ so that they "explain" the predictions. To do this, we maximize MI

II. Decompose MI in CMI and get equation 3. We can optimize equation 3 using RL framework and Problem is to optimize eq 3 in a RL framework using equation 5, with the reward in equation 4. Problem the sampling step is $\pi_\phi$ is not differentiable. Also $\pi_\phi$ are segments are therefore we only need to provide the "start" and the "end" for each segment. Thus $\pi_\phi$ can be decomposed into $\pi_{\phi^s}$ and $\pi_{\phi_e}$.

III. We can directly compute the gradient w.r.t $\phi$ and optimize (5), but the gradient variance is high. Thus, we need another loss term, equation (25) to reduces gradient variances, by introducing another network $V_\psi$ to estimate expected cumulative return.

IV. However, the new policy can diverge far away. So we need another loss term equation (26) instead. The total loss is a combination of (5), (25) and (26).

V. Now we have 2 networks, $\pi_\phi$ and $\pi_\psi$. It's time to train these networks. We jointly optimize these 2 during the training stage (as I understand because each loss has an expectation on x, and the number of steps are described in Appendix A2.)

VI. To generate explanations using inference data, we fix the 2 networks as they are trained. We then use equation 8) for a stopping criteria on whether new segments should be added.

For clarification, I wonder if the above description are correct or not.

Questions.
Q1. At inference time (if there is a difference between training vs inference), how do we select the next segment? Is it just sample from $\pi_\phi$ once? Or do we just do argmax? Or do we sample several times to see what which segment yields the highest reward?

Q2. As mentioned in L147, the segments are non-overlapping. But in the method section, there is no such constraints (in case I did not miss it). The only constraints are $t^s <= t^e$. So do we combine non overlapping segments?

Q3. I think A big part of Section 4.4 can be shortened, as it just describes a very trivial mechanism. The total loss, the training procedure and inference procedure can be described more clearly in the method section instead. (In case there is a separation between training and inference time)

Q4. (In case there is a separation between training and inference time,) how do we validate when the training is done? Do we validate on a separate validation set?

Q5. How does it apply to multivariate time series? Are there some rough run-time numbers?

Minor
1. The "s" is reused and it is slightly confusing. It can mean the segment index, or the "start" of the segment. It should not create confusion if the user is reading this carefully though. Changing this is optional.
2. In Table 2, what is the rationale of the highlighting? For example, Epilepsy, WinIT has 1.99 suff. while TimeSeg has 1.94 suff.. If we only look at the mean, TimeSeg should be bolded and WinIT should be underlined.
3. In Table 3, the errors should be subscripted (as in Table 2), for consistency.

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
5

### Summary
The paper presents TimeSeg, an information-theoretic framework that explains black-box time-series models by selecting contiguous temporal segments that best capture information relevant to predictions. Using reinforcement learning, it sequentially identifies predictive segments for each instance. This approach provides interpretable, segment-wise explanations that reveal holistic, class-predictive temporal patterns.

### Strengths
The paper addresses an interesting problem with a novel and well-conceived approach. It is clearly written and well-organized. The experiments are comprehensive, covering diverse datasets and evaluation metrics.

### Weaknesses
The paper overlooks relevant prior work. Specifically, it omits discussion of Theissler, A., et al. (2022). Explainable AI for time series classification: a review, taxonomy and research directions. IEEE Access, 10, 100700–100724, which provides a comprehensive survey on explainable AI for time series. It also fails to reference Spinnato, F., et al. (2023). Understanding any time series classifier with a subsequence-based explainer. ACM Transactions on Knowledge Discovery from Data, 18(2), 1–34, which presents a directly related method. Furthermore, the authors set τ = 0.3 and Kmax = 5 without explaining the rationale or analyzing the sensitivity of results to these parameters. Finally, the experiments are restricted to a single black-box model (CNN), overlooking state-of-the-art time-series classifiers such as ROCKET (Dempster, A., et. al (2020). ROCKET: exceptionally fast and accurate time series classification using random convolutional kernels. Data Mining and Knowledge Discovery, 34(5), 1454-1495.), MiniROCKET (Dempster, A., et. al.. (2021, August). Minirocket: A very fast (almost) deterministic transform for time series classification. In Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining (pp. 248-257).), and MR-HYDRA (Dempster, A., et. al. (2023). Hydra: Competing convolutional kernels for fast and accurate time series classification. Data Mining and Knowledge Discovery, 37(5), 1779-1805.). The evaluation should include multiple black-box models to better demonstrate the robustness and model-agnostic nature of the proposed approach.

### Questions
Questions can be derived from the above weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes an information-theoretic framework named TimeSeg for segment-wise explanation of black-box time-series models. The core motivation stems from the limitation of existing approaches, which mostly provide point-wise or fixed-patch explanations and lack a principled definition of meaningful temporal segments. TimeSeg aims to identify a set of continuous temporal segments that maximize the mutual information with the model’s prediction outcomes. The authors formalize this as a segment selection problem that seeks to maximize the joint mutual information between the selected segments and the predictive output. Since direct optimization is intractable, the problem is reformulated under a reinforcement learning framework to approximate the optimal segmentation policy. Specifically, a policy network predicts the start and end indices of informative segments and applies a sparsity regularization to control the explanation length. The policy is trained using Proximal Policy Optimization to ensure stable learning. Experiments conducted on multiple synthetic and real-world time-series datasets demonstrate that the proposed method produces more human-aligned and coherent explanations while preserving predictive performance.

### Strengths
The paper provides a clear theoretical formulation, defining segment-wise explanation for black-box time-series models as an optimization problem that seeks continuous subsequences maximizing the joint mutual information with the target prediction.

The authors transform the sequential decision process of maximizing conditional mutual information into a reinforcement learning task, enabling efficient exploration of the optimal explanatory segments within a discrete and non-differentiable selection space.

The proposed method operates effectively under a strict black-box setting, without requiring access to internal gradients or model parameters, demonstrating strong practical applicability. Moreover, the generated explanatory segments are coherent and cognitively aligned with human reasoning, showing high interpretability and real-world utility.

### Weaknesses
The evaluation metrics adopted in the current experiments lack sufficient persuasive power and fail to comprehensively reflect the proposed method’s advantages and degree of improvement.

All experiments are conducted solely on univariate time-series datasets, leaving the method’s applicability and robustness in multivariate scenarios unverified.

Moreover, the evaluation is performed exclusively on the Temporal Convolutional Network , without validation on other black-box architectures such as RNNs or Transformers, which limits the evidence for the method’s generality.

In addition, the paper does not report the computational cost associated with the reinforcement learning process, nor does it provide a systematic analysis of the algorithm’s efficiency, stability, or convergence behavior. The reproducibility details are relatively fragmented, and key hyperparameters (e.g., PPO parameters, learning rate) are not clearly summarized in the main text, which hinders reproducibility and engineering-level assessment of the proposed approach.

### Questions
**On the Reinforcement Learning Training Process**

a. Reinforcement learning typically requires extensive interactions between the agent and the environment (i.e., the black-box model) to sample training data. However, the paper does not report the computational cost of training and inference, nor does it provide an analysis of the algorithm’s efficiency or convergence.

b. The paper sets the maximum number of segments to $K_{\text{max}} = 5$. Could this constraint limit the number of agent–environment interactions and thus affect training performance? Please include a sensitivity analysis regarding this hyperparameter.

c. Please clarify whether the reinforcement learning training exhibits stability and consistency under different random seeds.

**On Experimental Evaluation and Comparisons**

a. Although the paper compares against methods such as TimeX++, the datasets and evaluation metrics differ from those used in the original TimeX++ experiments. Why not adopt the same evaluation metrics and experimental settings to ensure fair comparison? It is recommended to include results under consistent metrics such as AUPRC, AUP, and AUR to enhance persuasiveness.

b. All experiments are conducted exclusively on the TCN architecture. Please further validate the explainability and generalization of TimeSeg on other black-box models such as RNNs and Transformers.

c. The paper does not include TimeX as a baseline, even though it represents a key prior work in time-series explainability. The absence of this comparison reduces the completeness and competitiveness of the experimental conclusions.

**On Methodological Applicability**

The proposed method is evaluated only on univariate time-series datasets. Please elaborate on the framework’s applicability and potential extensions to multivariate time-series scenarios.

### Soundness
2

### Presentation
3

### Contribution
3
