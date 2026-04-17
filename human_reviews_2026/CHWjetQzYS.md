# Closed-form uncertainty quantification of deep residual neural networks

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
We study the problem of propagating the mean and covariance of a general multivariate Gaussian distribution through a deep (residual) neural network using layer by-layer moment matching. We close a longstanding gap by deriving exact moment matching for the probit, GeLU, ReLU (as a limit of GeLU), Heaviside (as
a limit of probit), and sine activation functions; for both feedforward and generalized residual layers. On random networks, we find orders-of-magnitude improvements in the KL divergence error metric, up to a millionfold, over popular
alternatives. On real data, we find competitive statistical calibration for inference
under epistemic uncertainty in the input. On a variational Bayes network, we show
that our method attains hundredfold improvements in KL divergence from Monte
Carlo ground truth over a state-of-the-art deterministic inference method. We also
give an a priori error bound and a preliminary analysis of stochastic feedforward
neurons, which have recently attracted general interest.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The authors aim to address the problem of propagating the mean and covariance of a general multivariate Gaussian distribution through a deep (residual) neural network using layer-by-layer moment matching. Technically, they derive analytical representations for the first two moments when using sine and probit activations in the neural network. Theoretical guanrantees and adversarial examples are provided for further analysis. Experimentally, they find competitive results on statistical calibration for inference under noisy or missing features on real data.

### Strengths
- The problem to be addressed is relevant for uncertainty estimation based on the first and second moment propagation.

- The idea is novel in the sense of deriving analytical solution for neural networks with sine or prohit activation functions.

- Theorectical guarantees have been dervied and analyzed for attribution of sources of error in this approach.

- The analysis of adversarial examples can shed light in future investigation in this direction.

- The carefully desgined experiments are comprehensive in terms of the tasks and statistical significance.

### Weaknesses
- The major concern regarding this idea is the scalibility and generalizability of the specific activation functions on which this work is built. As sin and prohit activation functions are less widely used than others such as relu, etc., the analytical repsentations derived in this work seems limited in this sense.

- The presentation flow is somehow hard to follow, especailly the section of method. I would suggest the authors to shorten the method part and merge the problem formulation, theorectical guarantees into this part to improve the coherence. 

- Though it is insightful for the readers to provide theorectical guarantees and adversarial examples. The actual implication of them for the main claim in the paper is not very clear. Therefore these two parts could be reduced to avoid distraction to the readers.

- There are two suggestions for the experimental part:
 1. The result presentation layout can be improved, for example, Fig1 and 2 can be merged to avoid occupaying the whole page. 
 2. The proposed solution would be more convincing to be evaluated on more challenging tasks with larger network architectures and datasets, such as image classification.

### Questions
See above

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
3

### Summary
This paper proposes an analytic method (“Yana”) for propagating Gaussian input uncertainty through deep residual neural networks. By deriving closed-form expressions for the mean and covariance of sine and probit activations under Gaussian input, the method allows exact moment matching at each layer. Experiments on synthetic random networks and two small real datasets (California Housing regression, Taiwanese Bankruptcy classification) show that the proposed method can achieve orders-of-magnitude improvements in approximation accuracy over linearization, mean-field, and unscented transform baselines. The paper also includes theoretical error bounds and provides code for reproducibility.

### Strengths
- **Novelty**: Closed-form propagation formulas for probit and sine activations are mathematically elegant and, to my knowledge, new.  
- **Theoretical contribution**: Includes detailed derivations, moment identities, and a theoretical error bound.  
- **Synthetic evaluation**: Stress tests across 72 random networks, showing clear gains over other analytic approximations.  
- **Reproducibility**: Code and detailed appendices are provided.

### Weaknesses
1. **Ambiguity of contribution**  
   The paper presents itself as proposing a *new method for uncertainty quantification*. If that is the claim, the evaluation seems inadequate, since there are no comparisons to standard UQ baselines such as heteroscedastic neural networks [5], ensembles [6], dropout-based Bayesian approximations [2], or fully Bayesian neural networks [1].  
   If instead the true contribution is a *new analytic Gaussian propagation approximation*, then the chosen baselines (linearization, mean-field, unscented transform) are more reasonable, but the framing should be explicit about this narrower scope.  
   As written, the paper makes claims about introducing a novel UQ method, while in practice it only proposes a new way of propagating Gaussian input distributions through a network. Moreover, all reported baselines are themselves approximation schemes of this propagation task, rather than established UQ methods, which makes the positioning and evaluation misaligned.


2. **Missing ground-truth baseline** A clear benchmark should be included, a simple Monte Carlo sampling: draw many samples from the Gaussian input distribution, run them through the network, and empirically compute the output mean and covariance. This baseline is straightforward, works with any activation, and—given enough samples—provides a gold standard approximation. While more costly due to multiple forward passes, it would contextualize the claimed accuracy improvements of the proposed analytic method. Its absence makes it difficult to judge whether the gains are practically meaningful compared to this simple and applicable approach.

3. **Computational cost considerations**  
   The method scales as \(O(n^3)\) in layer width (due to covariance propagation), as noted by the authors, similar to linearization methods but higher than a single forward pass (\(O(n^2)\)). While cheaper than Monte Carlo with many samples (\(O(Mn^2)\)), the trade-off is unclear in practice. Runtime comparisons against Monte Carlo sampling vs the proposed method would clarify the actual efficiency benefits.  

4. **Limited scope of experiments**  
   - Real datasets are small (≤20k samples, ≤95 features).  
   - Networks are shallow (≤3 layers, width ≤200).  
   - Evaluations are therefore small for modern NNs, not representative of modern deep learning architectures (e.g., ResNets, Transformers).  

5. **Narrow focus on input Gaussian uncertainty**  
The method appears to model only *aleatoric* uncertainty, while ignoring *epistemic* uncertainty. It is unclear why a practitioner would choose this approach over standard aleatoric uncertainty-aware models—such as heteroscedastic neural networks [5]—where the network learns to predict both the mean and the variance (as opposed to only the mean in standard regression tasks).



6. **Restrictive activations**  
   Exact formulas are derived only for sine and probit activations, which are rarely used in modern practice compared to ReLU or GELU. This strongly limits applicability.


---

### **References**
[1] Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). *Weight Uncertainty in Neural Networks.* ICML.  
[2] Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.* ICML.  
[5] Nix, D. A., & Weigend, A. S. (1994). *Estimating the Mean and Variance of the Target Probability Distribution.* IEEE International Conference on Neural Networks (ICNN).  
[6] Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.  
[7] Wright, O., Nakahira, Y., & Moura, J. M. F. (2024). *An Analytic Solution to Covariance Propagation in Neural Networks.* AISTATS.

### Questions
1. Can the method be generalized or approximated for ReLU/GELU activations?  
2. How does the method compare in runtime and accuracy to Monte Carlo sampling?  
3. Is the contribution intended as a general UQ method or a propagation approximation?  
4. How would the method scale to modern architectures with hundreds of layers?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a deterministic method for propagating input uncertainty through deep neural networks by computing closed-form expressions for mean and covariance transformations under specific activations (probit and sine) that admit analytic bivariate Gaussian moments. The method enables full-covariance propagation through deep and residual layers without resorting to Monte Carlo or linearization.

### Strengths
- The core mathematical derivations seem to be correct and technically consistent.
- Exact propagation of mean and covariance for specific activations is neat and removes the need for local linearization.

### Weaknesses
- In my opinion, the paper lacks a coherent narrative, with abrupt paragraph breaks, redundant notation, and sections that feel disconnected. Key ideas are scattered rather than developed progressively, making the reading experience not smooth.
- Current experiments are limited, with no demonstration on modern architectures, applications (e.g., uncertainty quantification), or interesting benchmarks. As of now, these experiments are not up to the standard of a conference like ICLR.
- The paper proposes a method that only works in sin and probit activations. These are not used in practice, so people will end up reverting to methods that already exist in the literature.
- The bound in section 4 is underwhelming.
- The authors do not discuss two recent works that seem to be relevant and actually discuss useful applications of why someone would want to propagate uncertainty across networks (including residual networks). Authors should cite these works [1, 2] and compare them in more serious uncertainty quantification tasks.

[1] Post-Hoc Uncertainty Quantification in Pre-Trained Neural Networks via Activation-Level Gaussian Processes. Bergna et al., 2025.
[2] Streamlining Prediction in Bayesian Deep Learning. Li et al., 2025.

### Questions
- Why would people use this method (Y_ana) in practice if they use common activation functions? Won't people revert to approximations than seem to work well in practice and exist already? [1, 2]

### Soundness
2

### Presentation
1

### Contribution
1
