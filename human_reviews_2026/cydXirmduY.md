# Emergent World Representations in OpenVLA

- Avg Score: 2.00
- Decision: Reject
- Scores: 2, 2, 2

## Abstract
Vision Language Action models (VLAs) exhibit complex control behaviors without explicitly modeling environmental dynamics. However, it remains unclear whether VLAs implicitly learn world models, a hallmark of model-based RL. We propose an experimental methodology using embedding arithmetic on state representations to probe whether OpenVLA, the current state of the art in VLAs, contains latent knowledge of state transitions. Specifically, we measure the difference between embeddings of sequential environment states and test whether this transition vector is recoverable from intermediate model activations. Using linear and non-linear probes trained on activations across layers, we find statistically significant predictive ability on state transitions exceeding baselines (embeddings), indicating that OpenVLA encodes an internal world model (as opposed to the probes learning the state transitions). We investigate the predictive ability of an earlier checkpoint of OpenVLA and uncover hints that the world model emerges as training progresses. Finally, we outline a pipeline leveraging Sparse Autoencoders (SAEs) to analyze OpenVLA's world model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies Vision Language Action models (VLAs) which are trained with policy-based reinforcement learning (RL) methods that do not explicitly model environmental dynamics. The research question is whether VLAs instead do implicitly learn world models. The authors use an experimental approach to investigate whether trained models gather latent knowledge of state transitions. Evaluation results are performed on OpenVLA.

### Strengths
S1 The considered problem is interesting and relevant.

S2 The paper is overall well-written.

### Weaknesses
W1 Regrading the research question, the paper does not provide analytical results.

W2 It remains unclear whether the observations made hold beyond OpenVLA.

W3 The evaluation is too small, contains only few examples and is hence not fully convincing. The question whether trained models gather latent knowledge of state transitions is not clearly answered and the use of the analysis remains unclear (at least to me).

### Questions
Do you expect the same results for VLAs different from OpenVLA?

Is it possible to consider larger problem instances?

Is Theorem 1 supposed to be a key contribution? How does it effectively relate to the numerical observations made?

Is it possible to actively use the located latent world model for state predictions?

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
The paper aims to study the emergence of implicit world models in pre-trained OpenVLA. Using frozen representations and intermediate activations, they train linear and MLP probes on representations of different layers of the model to predict actions. The central claim is that autoregressive next-token prediction training induces representations consistent with “world models".

### Strengths
1. The use of standard linear and MLP probes and similarity analyses on frozen embeddings is well explained and replicable in principle.

### Weaknesses
1. The work largely documents known consequences of autoregressive temporal modeling, i.e., predicting future tokens forces latent spaces to encode local dynamics. If the goal of the paper was to provide a mechanistic interoperability theory of this phenomenon, it would have been valuable, but the current probing approach and corresponding results do not shed any light on the mechanics. The claimed emergence of “world representations” appears to be a direct and unsurprising byproduct of temporal modeling.

2. Following up on my point above, there is no controlled experiment showing or discussing necessity or sufficiency of autoregression for structure emergence. The fact that $R^2$ of internal activations for predicting actions is lower compared to $R^2$ of embeddings for predicting actions, does not show any causality. It is expected that last layer reprs are richer in content compared to internal reprs.

3. The paper claims (even in abstract) that OpenVLA is pretrained using policy-based RL. This is not correct. There is no RL involved in training OpenVLA and it is trained via imitation learning.

4. Dividing RL into model-based and policy-based is not correct. Model-based methods also learn a policy. Do the authors mean model-free RL by "policy-based"?

5. Many of the files in the anonymous repo are missing (I get "The requested file is not found.")

### Questions
1. What distinguishes your findings from standard properties of temporal predictive models?

2. Would a non-autoregressive masked-video model show similar results as a VLA?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper studies whether the large Vision-Language-Action model OpenVLA exhibits an emergent world model, by training linear and nonlinear probes on intermediate activations to predict embedding-based state transition vectors. The authors report that certain layers (e.g., 15, 22, 30) yield statistically significant predictive ability, suggesting that OpenVLA implicitly encodes world dynamics.

### Strengths
* The question of emergent “world models” in policy-based RL systems is interesting and timely.
* The methodology (probing activations with linear and MLP regressors) connects recent interpretability work in language models to robotics.
* The paper is clearly written and includes theoretical motivation using Koopman operators.

### Weaknesses
1. *Unclear research question and experimental motivation.* It is never clearly explained why only specific layers (7, 15, 22, 30) are selected for probing. The rationale for these discrete sampling points is missing, and without a systematic sweep or justification, the conclusions about “middle layers” encoding world models are not convincing.

2. *Ambiguous test definition.* The precise experimental test used to claim an emergent world model is difficult to follow. It remains unclear what exactly is being predicted (embedding deltas? latent transitions?) and how statistical significance across layers and datasets translates into evidence for an internal model, as opposed to trivial temporal correlations in embeddings.

3. *Methodological opacity.* The description of the regression probes lacks detail about data partitioning, normalization, and hyperparameter tuning. It is also not fully clear how permutation tests are performed or interpreted.

4. *Interpretation of results.* The claim that “world models emerge” appears overstated relative to the presented evidence. The reported $R^2$ values are modest, and there is no ablation verifying that probes are not learning the dynamics directly.

### Questions
1. What is the exact hypothesis being tested — that OpenVLA contains an implicit model of the environment, or merely that its embeddings exhibit temporal coherence?

2. Why are only layers 7, 15, 22, and 30 selected? Are these equidistant checkpoints? Is there empirical or theoretical motivation for these choices? Did you examine whether neighboring layers yield similar behavior?

3. What is the rationale for examining specific horizons K = 1, 3, 10, 30? How were these values chosen, and how sensitive are the results to K?

### Soundness
3

### Presentation
2

### Contribution
1
