# Learning What Matters: Toward Causally-Grounded Foundations for Contextual Time-Series Forecasting

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Time-series evolution is driven by intrinsic dynamics and heterogeneous contextual factors whose relevance can be spurious or unstable across settings, which limits correlation-driven designs that impose fixed structures on contextual inputs. We propose a causally grounded foundation model for context-aware forecasting that augments a generative pre-trained transformer with a context-aware attention module to adaptively integrate external signals. From an information-theoretic perspective, a query-modulation mechanism conditions temporal queries on global context summaries, reducing redundancy and sharpening focus, while a neural structural-equation component injects inductive bias toward genuinely influential features. To align attribution with behavior, counterfactual perturbations enforce consistency between structural importance and predictive responses; an entropy-based regularizer further encourages sparse, interpretable attributions. The overall design targets accuracy, robustness to distributional shifts, and explanatory clarity in the presence of noisy or weakly relevant context. Extensive experiments on diverse real-world benchmarks demonstrate consistent gains over strong baselines in point forecasting and calibration, alongside clearer and more stable explanations of context contributions.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes a transformer-based time-series forecasting model that can also handle heterogeneous external input channels as contexts.  They depend on cross-attention over the external input channels, and that would be a routine model architecture.  The paper's main contribution is to introduce three modifications to the vanilla cross attention: modulate the query vector with a term derived from a summary of the context, introduce a global channel-specific bias, and introduce a loss term to promote selection of only useful channels.  Their method is compared with several prior methods on multiple datasets, and shown to provide modest gains.

### Strengths
S1.  Empirical comparison on a large variety of datasets show consistent gains over prior methods.

S2.  The problem tackled is well-motivated and at the frontier of interesting problems to solve in time-series forecasting.

S3.  The intuition of selecting globally consistent sparse set of external channels makes a lot of sense.

### Weaknesses
W1.  The paper is selling the proposed method under the causal modeling banner, but finally what is implemented is just simple global feature selection.  

W2. In so doing, the clarity of the presentation has been seriously compromised.  It took me a lot of time to cut through the clutter of ungrounded terms like "semantic selectivity", "structural attribution", and "behavioral validation" to understand that all they are doing is selecting a minimal set of channels globally.    Along a similar vein, they present vague connections to information-bottleneck  with Equation (3), which they claim is implemented in a "soft parameteric" way using equation (4).  There is no attempt at establishing a connection between the two equations.  The whole of that section has been terminology-washed.    Later we encounter "neural structural equation module" without any citation, and without any justification of how the learned network is a structural causal equation.  Just naming an equation or variable as such does not make it causal.

W3.  The use of Equation (8) to promote sparsity in selection of channels seems unconventional.  Feature selection traditionally in ML has been handled by regularizers, for example, an L1 penalty on weights of selected channels.  The loss term they introduce where they mask the Top-K features and minimize contrastive loss seems like an over-kill.  Also, TopK is not differentiable.  How is this handled during end-to-end training?  There might be un-intended side-effects where some other parameters may get the wrong gradient with loss terms of opposite signs.  A comparison with simple global L1 regularization is needed.  

W4: Proposition 1 meant to be a formal claim but there are vaguely defined terms like "assuming predictive stability
 and behavioral degradation under intervention".   I do not understand what is the point being made in Theorem 2.

W5: The gains beyond vanilla cross attention is very modest as per Table 3.

### Questions
Q1:  Is there a typo in Eq (8).  Don't you want the masked loss to be greater than the unmasked (factual) loss?

Q2:  In Table 1, are the different baselines run with the same number of network parameters, or are the numbers obtained from previously published numbers?

Q3: Are the ablation studies performed in the full-shot or zero-shot mode? It will be interesting to see if the gains beyond vanilla cross attention persists even in zero-shot mode.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes a causally grounded foundation model for context-aware time series forecasting. The model introduces a context-aware transformer enhanced with a causal inductive bias through query modulation and neural structural equation modeling. It further incorporates a counterfactual validation step to align predictive behavior with causal attributions. The framework aims to improve robustness and interpretability by identifying causally relevant contextual features rather than relying solely on correlation.

### Strengths
- The paper demonstrates careful experimental design and a clear effort to connect causal reasoning with practical forecasting performance.

- The inclusion of theoretical analysis (Proposition 1 and Theorem 2) is a positive aspect that adds conceptual depth, even though parts remain high-level.

- The experiments are systematic and well executed, with diverse datasets, strong baselines, and thorough ablations.

### Weaknesses
### **Problem & Motivation**

- The framing of “causally grounded” forecasting is interesting and timely, but the title and claims might be misleading. The method does not actually perform causal inference or guarantee causal identifiability; rather, it regularizes attention weights heuristically. Removing “causal” from the title could make the contribution more accurate.

- It is unclear how causality is actually achieved. The mechanism that ensures the model distinguishes causal factors from spurious correlations is not well defined.

- While the paper uses the term “foundation model,” it’s not evident that large-scale pretraining or generalization at scale is performed; model size and training regime are not reported.

---

### **Methodology**

- Despite the high-level motivation, the actual novelty is moderate, relying mainly on cross-attention and regularization-based extensions of standard transformer components.

- The model’s attention mechanism assumes that the causal bias term $B$ highlights salient contextual factors, but there is no formal guarantee or validation showing that this term reliably corresponds to causal importance.

- The mathematical details of Proposition 1 and Theorem 2 are insufficiently explained. It’s not clear what they aim to prove—are they about identifiability, stability, or information preservation? Proofs or even intuitive derivations are missing.

- The paper could better clarify where the causal bias module is inserted (which layers, how often, and with what impact). A layer-wise analysis would help understand its contribution.

- There is no discussion of what happens when context is unrelated to the time series. Does the model ignore it or overfit to noise? This is a critical question for real-world deployment.

---

### **Experiments**

- The experimental design is robust and thorough, using multiple datasets and strong baselines. However, OOD (out-of-distribution) evaluations are limited. If the model is claimed to be causally grounded, testing under domain or context shift is essential.

- The results are strong overall, but the zero-shot generalization claims would be more convincing with ablations showing how model scale and layer depth affect transferability.

### Questions
Please refer to the weaknesses.

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
3

### Summary
This paper designs an SEM module for discovering globally important features and trains the model using a loss function guided by counterfactual reasoning.

### Strengths
1. An effective multimodal data processing solution that combines time series and text data.

### Weaknesses
1. Despite the author's repeated emphasis on causality, the method proposed in this paper cannot discover strict causal laws and is only used to discover stable and usable features. I suggest the author carefully revise the text to avoid misunderstandings. Otherwise, causal effect tools like DAG need to be considered.

2. Lack of verification of false correlation variables. The intution of the paper is to identify real causal variables, but there is no relevant support for this point. It is recommended to conduct verification on a synthetic dataset.

3. Additional computational burden: Compared to general training paradigms, counterfactual based training will introduce at least twice the additional inference time.

### Questions
1. The selection mechanism of top-k is a non differentiable and rigid choice, which will greatly affect the computational burden. Can Gumbel Softmax and other alternatives replace it？

2. Can the SEM module handle the joint effects between multiple variables?

3. Is there a potential conflict between the two parallel mechanisms that affect attention in the model? For example, in a certain sample, query modulation may suggest reducing attention to a certain feature, but global causal bias forces an increase in attention to that feature. How does the model solve this contradiction? Will this lead to unstable training?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a causally grounded foundation model for context-aware time series forecasting by adaptively integrating internal and external contexts.

### Strengths
1. Integration of both internal and external context.
2. Clear paper writing and enough experiments across multiple datasets.

### Weaknesses
- Novelty is not high. ContextAdapt Block is the most outstanding innovation in model architecture; it still employs cross-attention mechanisms to figure out the alignment between context and history target.
- No code provided.

### Questions
1. What do you mean by global context summary $g_c$? You have $g_c=\phi(Z_c)$ in Figure 1 and $g_c=f(Z_c)$ in Eq. 4, is it a typo? What is that function $\phi$ or $f$? Why do you need $g_c$ rather than $Z_c$?
2. Is context equivalent to a causal relationship? Since you try to figure semantic selectivity between the history target and the context, do you think the context has a causal relationship with history target? To me, the causal relationship may be more obvious between history and the future.
3. How do the external covariates look like and how did you align them with the history target?
4. How do you think the counterfactual loss could be helpful to training the model if removing access to the most salient context channels?
5. Typo for " Semantic Slectivity" in Figure 1.

### Soundness
3

### Presentation
3

### Contribution
3
