# Preferential dynamic modeling with forward-backward smoothing

- Decision: Reject
- Scores: 6, 2, 6, 6

## Abstract
Estimating a secondary signal (e.g., behavior) from neural activity over time is central to both causal online decoding and non-causal offline inference in neuroscience. Existing two-signal latent state-space modeling methods typically either support causal prediction of the secondary signal from the primary signal or non-causal inference (smoothing), but rarely both; here we extend one analytical linear method (PSID) and one nonlinear deep learning method (DPAD) beyond causal prediction to also support non-causal inference, yielding a more universally applicable family of methods. We provide theoretical derivations extending PSID to enable optimal filtering and optimal smoothing of the secondary signal. We show that, in the PSID setting, the presence of a secondary signal increases identifiability. This allows us to uniquely learn the quantities needed for the optimal Kalman update via a reduced-rank regression step that augments the standard SVD-based PSID algorithm, yielding our first contribution, PSID with filtering. We next design a forward-backward construction for smoothing, yielding our second contribution, PSID with smoothing. For nonlinear prioritized modeling, we extend DPAD to a bidirectional variant that combines forward and backward hidden states at readout to perform smoothing, yielding our third contribution, DPAD with smoothing. In simulations, we validate that PSID with filtering and smoothing reach ideal performance. In non-human primate motor cortex data, PSID with smoothing consistently improves over PSID with filtering, which improves over one-step-ahead prediction with standard PSID. Finally, we test DPAD with smoothing on three Neural Latents Benchmark (NLB) datasets, where it achieves the top behavior-decoding result on at least one dataset and near-top performance in behavior decoding and held-out neural prediction on all three. Together, these methods form a family with wide-ranging applications, from causal online decoding to offline inference, in both linear and nonlinear settings.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper extends PSID (linear) and DPAD (nonlinear) to support both filtering and smoothing of a secondary signal from a primary time series, thereby bridging causal and non-causal inference in two-signal system identification. The authors provide analytical derivations for the linear case and introduce a bidirectional RNN variant for the nonlinear case. Experiments on simulations, primate motor cortex data, and the Neural Latents Benchmark (NLB) demonstrate performance competitive with top-ranking methods such as Ctrl-TNDM and LangevinFlow, achieving top behavior-decoding accuracy on the MC_RTT dataset.

### Strengths
- Well written, clearly structured, and easy to follow.
    
- Strong empirical validation across multiple datasets and settings, with consistent improvements over relevant baselines.

### Weaknesses
- Theoretical contributions are presented in a textbook style, making it hard to separate novel insights from background material. Appendix e.g. lacks formal structuring (e.g., propositions or theorems) that would clarify assumptions and originality.

### Questions
- Can the authors summarize their key theoretical contributions more formally (e.g., in a compact proposition) to distinguish them from standard Kalman filtering results?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this work, the paper proposes the framework extensions of two existing dual-signal latent state-space modeling methods. The first is a linear method named Preferential Subspace Identification (PSID), and the second one is the non-linear Dynamic Prioritized Analysis of Dynamics (DPAD). The goal of this proposed framework is to enable optimal filtering and smoothing of a secondary signal e.g., behavior, $z_k from the primary neural activities. As for PSID, the authors use the Reduced Rank Regression (RRR) step to do the filtering extension. They also extend the PSID with forward-backward smoothing. Then for the DPAD, they use the bidirectional RNN (Bi-RNN) deep model architecture to implement the smoothing. By comparison, the original PSID and DPAD methods are focused on one-step-ahead prediction.

For the experiment parts, the authors validate their proposed framework with simulation data and the widely-adopted NLB benchmark.

### Strengths
1. The paper writing flow is concrete and easy-to-follow.
2. The paper provides a comprehensive framework addressing the filtering and smoothing tasks in two wide-adopted models, linear (PSID) and nonlinear (DPAD). These problems are actually quite important in computational neural data modeling.

### Weaknesses
1. The overall framework in the submitted manuscripts is like a combination of the incremental model enhancement of two existing methods, which is pretty heuristic and intuitive. There is even no name of the proposed framework.
2. Besides the a bit engineering method combinations, there seems to be no real algorithm novelty to the neural and behavior analysis community.
3. Some components of the proposed framework, like bi-directional RNNs, are actually a bit old-fashioned. The framework overall is actually looks like a theoretically unvalidated construction.

### Questions
I have no more questions, other concerns please see my weaknesses section.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper develops methods for filtering and smoothing two-signal SSMs, i.e., where observable $y\_k$ and $z\_k$ depend on latent $x\_k$ and the goal is to predict or infer $z$ from $y$. The proposed methods extend linear (PSID) and nonlinear (DPAD) methods to smoothing using forward-backward approaches. For PSID, they also add a method for system identification for optimal filtering that learns only the part of the $x,y$ dynamics that is identifiable from $z$.

### Strengths
Strong theoretical grounding.
Good experimental results on synthetic data and on neural decoding tasks against established baselines.

### Weaknesses
The paper claims the extensions of PSID have certain optimality properties but this is not proved. It is claimed that solving eq 7 (setting aside sampling error from a finite training set) yields the optimal filter for predicting $z_k$ from $y\_{1:k}$, but this is not quite proved. A similar question applies to PSID with smoothing, but for that algorithm there is also the more primary question of whether it yields the optimal (i.e., Bayesian or minimum-variance unbiased) estimate of $z\_k$ when the system parameters are known. I expect this would be easy to prove and would be important to add to the paper.

### Questions
I work a lot with SSMs and Kalman-type methods but had not seen models where the system and observation noise are dependent. Can you add a brief explanation or motivating example for why we should expect such dependence?

I was unfamiliar with DPAD and wonder whether it could benefit from tracking uncertainty ($P$). If I understand correctly the only latent state in the RNN is $x_k$, which is a mean estimate, whereas optimal filtering also requires covariance. Can the model be extended in this way, or does it somehow track uncertainty as is?

eq 12a: $A$ should be $I$

Duplicate \labels for eqs 19 and 25?

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
2

### Summary
This paper extends PSID (linear) and DPAD (nonlinear) methods to support optimal filtering and smoothing of a secondary signal (e.g., behavior) from a primary neural signal. The authors derive theoretical updates, introduce a forward-backward smoothing framework, and validate the methods on both simulated and real neural datasets, showing improved decoding accuracy over existing baselines.

### Strengths
* Clear problem formulation for prediction, filtering, and smoothing in two-signal settings.
* Solid theoretical derivation for extending PSID to filtering via reduced-rank regression.
* Comprehensive evaluations on simulations, primate data, and NLB benchmarks.
* Unified framework covering both linear and nonlinear models for causal and non-causal inference.

### Weaknesses
* There seems to be no discussion on computational cost or real-time applicability.
* The current abstract is a bit long.

### Questions
* Whether the forward and backward RNNs can be replaced with GRU or LSTM, since they could provide more smoothness.
* What's the definition of $\\boldsymbol \\epsilon_k$? Does it have to be Gaussian?

### Soundness
3

### Presentation
2

### Contribution
3
