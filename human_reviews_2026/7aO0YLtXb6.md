# From Muon to Gluon: Bridging Theory and Practice of LMO-based Optimizers for LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Recent developments in deep learning optimization have brought about radically new algorithms based on the Linear Minimization Oracle (LMO) framework, such as 𝖬𝗎𝗈𝗇 and 𝖲𝖼𝗂𝗈𝗇. After over a decade of 𝖠𝖽𝖺𝗆's dominance, these LMO-based methods are emerging as viable replacements, offering several practical advantages such as improved memory efficiency, better hyperparameter transferability, and most importantly, superior empirical performance on large-scale tasks, including LLM training. However, a significant gap remains between their practical use and our current theoretical understanding: prior analyses (1) overlook the layer-wise LMO application of these optimizers in practice, and (2) rely on an unrealistic smoothness assumption, leading to impractically small stepsizes. To address both, we propose a new LMO-based framework called 𝖦𝗅𝗎𝗈𝗇, capturing prior theoretically analyzed methods as special cases, and introduce a new refined generalized smoothness model that captures the layer-wise geometry of neural networks, matches the layer-wise practical implementation of 𝖬𝗎𝗈𝗇 and 𝖲𝖼𝗂𝗈𝗇, and leads to state-of-the-art convergence guarantees. Our experiments with NanoGPT and CNN confirm that our assumption holds along the optimization trajectory, ultimately closing the gap between theory and practice.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Gluon, a generalized linear minimization oracle (LMO) optimization method that unifies Muon, Scion, and SGD as specific instances. The authors define an extension of the concept of L-smoothness of the loss landscape, and derive convergence bounds for Gluon based on the smoothness parameters. By assuming the bounds are tight, one can estimate the optimal learning rates after observing a training run, and these closely match those found through hyperparameter searches. This suggests that the generalized smoothness formulation provides a more accurate description of the loss landscape than classical L-smoothness, and offers a principled alternative to ad hoc tuning of per-layer step sizes.

### Strengths
- Gluon unifies existing theory surrounding other LMO optimizers, and is able to provide generalized bounds that specialize into known bounds for specific cases.
- Gluon offers a way to estimate the best possible learning rates from observation of a finished training run.

### Weaknesses
- Learning rate estimation from observing a training run hinges on a crucial assumption that bound (4) is saturated, which I do not find to be plausible in practice.
- Not very much empirical evidence is given to show that the modified smoothness model accurately describes the loss landscape.
- Not very much empirical evidence is given to show that the modified smoothness model of the loss landscape, which the analysis is based upon, is a better description than classical L-smoothness, for deep learning tasks. Some learning rates match across theory and experiment, but that's it.
- A training run must be done first, at the same scale, in order to estimate the best possible learning rates. Other methods relying on scaling laws only require having a previous run at a smaller scale in order to predict the best learning rates at a larger scale.
- To obtain learning rates from a past training run, one must estimate the smoothness parameters by presuming saturation of Inequality (4). Even if we ignore the accuracy of the saturation assumption, the method in Equation (10) used to estimate terms of Inequality (4) is not very sound. Saturation of Inequality (4) requires the value calculated by Equation (10) to be as high as possible, whereas Equation (10) is likely to produce a value as low as possible, since learning trajectories tend to proceed down the smoothest directions of a loss landscape on most iterations. The bound (4) will probably end up too loose and the smoothness parameters recovered by presuming tightness will be unreliable.

### Questions
- Is there any more evidence that the theory matches experiment? For example to test the validity of Assumption 1: after you estimate $L_i^0$ and $L_i^1$ by experiment, can you adverserially try to pick $X$ and $Y$ to break Assumption 1, but only ever manage to saturate the bound and never break it, showing that the assumption is accurate and faithful to the shape of the loss landscape?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Gluon, a unifying, layer-wise LMO-based optimizer framework within which Muon and Scion are special cases. They claim to bridge the gap between theory and practice by proposing a layer-wise (L_0,L_1)-smoothness assumption that allows them to prove convergence in the stochastic optimization setting using step sizes that do not shrink with the number of training iterations. The authors claim that this is more realistic than previous work for two reasons: (a) trajectory smoothness varies significantly from layer to layer when training neural networks, and (b) learning rates that are inversely proportional to the number of steps quickly become too small to be realistic in practice. Empirically, the paper validates the smoothness model and step-size structure with NanoGPT-124M trained on FineWeb and a CNN trained on CIFAR-10, showing that their proposed smoothness model tracks closely with the observed trajectory smoothness. They also find that per-layer radii used in Scion correlate with their theoretically suggested magnitudes.

### Strengths
- The paper is well written. It is easy to understand the contributions of the paper, and their relation to previous work is made clear.
- The introduction of layer-wise smoothness is an improvement upon existing work. 
- The introduction of realistic step sizes in the proof is a clear improvement. 
- I appreciate the empirical validation of assumption 1 in figures (1) and (2).
- While I have limited experience with proving convergence, I did read through the math in the paper and did not notice any mistakes. However, I did not check the proofs.

### Weaknesses
- While I appreciate that the paper bridges the theory-practice gap with more realistic assumptions than previous work, I still struggle to understand how the current theory can “lead to guiding practical choices”, touted as the main goal of theory on line 164. Could the authors elaborate on why this is the case?
- The experiments are small-scale, which is not generally a concern for a theoretical paper that doesn't make claims about empirical performance. However, you do claim to bridge the theory-practice gap with more realistic assumptions, but you only validate these at a smaller scale and only for a subset of layers in the network(Figs 1,2,5,6,7,8,9,10). I think the paper would be significantly stronger if you did something along the lines of the following:


(a) Validate your layer-wise smoothness assumptions hold, at least at the beginning of training, across a few larger model scales (maybe 500M and 1B parameters) to get an idea of the trend.


(b) Provided aggregate statistics across all layers in the model (e.g., to get a summary for the main paper). For example, you could show curves reporting how many layers satisfy the assumption across the entire model instead of reporting a single layer.

### Questions
- What is small k in Table 1? I assume this was omitted because k is included in the radius and momentum formula. I still think it would be clearer to provide intuition as to what it controls. 
- Decoupled weight decay is critical for achieving strong performance [1,2], but it doesn’t feature in your framework or existing convergence analysis, from my understanding. Do your rates still apply when using decoupled weight decay?
- Line 472 possible typo: finetuning → HP search 



[1][Decoupled Weight Decay Regularization]

[2][Muon is Scalable for LLM Training]

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
This paper claims that prior theoretical analyses are flawed because: 1) they overlook the practical "layer-wise" implementation of these optimizers, using a simplified "global" update, and 2) they rely on the standard L-smoothness assumption, which is unrealistic for deep learning and leads to theoretically impractical stepsizes.
To solve this, the authors propose Gluon, a generalized layer-wise LMO framework, and a layer-wise $(L^0, L^1)$-smoothness assumption. Under this assumption, they provide convergence guarantees. The key result is that their deterministic analysis yields an adaptive layer-wise stepsize. Their experiments on NanoGPT and a CNN validate that their smoothness assumption is a good empirical fit and that their theoretically-derived stepsizes match the relative magnitudes of layer-wise learning rates found in Scion.

### Strengths
1. This paper proposes a layer-wise $(L^0,L^1)$-smooth method for Muon/Scion type of optimizers.
2. The empirical validation in the paper shows that the proposed smoothness model is a close fit for the "trajectory smoothness" observed during training.
3. It's interesting to see that the theory predicts the layer-wise stepsize ratios that align with the previous work found through tuning.

### Weaknesses
1. The paper claims that its layer-wise analysis as a key novelty over Scion, since Scion uses a 'global' update. I think it's a misunderstanding. If I understand correctly, Scion is also a layer-wise framework. It achieves per-layer control by per-layer radius for scaling individual layer norms.
2. The paper's novel adaptive, layer-wise stepsize is derived only for the deterministic setting (Theorem 1). For the stochastic analysis (Theorem 2), the paper reverts to a standard non-adaptive, decaying stepsize.
3. Although the authors claim that Gluon is an _adaptive_ optimizer, I'm not sure if it's true. The "adaptive" stepsize in line 293 from Theorem 1, is impractical because it requires a priori knowledge of the $L^0_i$ and $L^1_i$ constants for every layer. It looks like this paper finds these constants through running the experiment at first and then fit them into another run. Could Gluon adjust $t_i^k$ during training to achieve real adaptive method? (Correct me if my understanding is wrong.) 
4. I notice that there is another paper [1] also analyzes a $(L^0,L^1)$-smoothness model. It may be not very accurate to say "all existing analyses... are built on the classical $L$-smoothness assumption". Could the authors discuss the difference between this paper?

[1] Thomas Pethick, Wanyun Xie, Mete Erdogan, Kimon Antonakopoulos, Tony Silveti-Falls, and Volkan Cevher. "Generalized Gradient Norm Clipping & Non-Euclidean $(L_0, L_1)$-Smoothness." arXiv preprint arXiv:2506.01913 (2025).

### Questions
1. Is it possible to extend both theory and experiments to the version with weight decay?
2. I notice that the authors use a constant learning rate. Is this necessary for getting approximate $L^0_i$ and $L^1_i$ or other empirical results?

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
The paper proposes Gluon, a layer-wise Linear Minimization Oracle (LMO) optimizer unifying Muon/Scion–style updates. It replaces global LMOs with per-layer LMOs and analyzes convergence under a new layer-wise (L_0, L_1)-smoothness assumption. In the deterministic case it yields adaptive per-layer step radii and O(1/\sqrt{K})-stationarity, and in the stochastic case with momentum it gives O(1/K^{1/4}). Experiments on NanoGPT-124M (FineWeb) and a CNN on CIFAR-10 claim that the assumption tracks measured “trajectory smoothness,” and that theory-suggested layerwise learning-rate scales align with tuned Scion settings.

### Strengths
1. Methodology is motivated and presented clearly: how Muon/Scion arise as Gluon special cases and why layer-wise LMOs matter computationally. The Algorithm 1 description and per-layer ball LMOs are easy to follow.

2. Convergence analysis for Gluon is provided, and the proofs are easy to follow in the appendix.

### Weaknesses
1. The experimental parts need to be strengthened: all experiments presented in section 5.1 and 5.2 are performed on quite small-scale model. Since the vanilla muon is claiming better performance than AdamW when training on LLMs, I think there should at least be some experiments for larger models to demonstrate the effectiveness of Gluon, other than just CNN on CIFAR10.

2. Evaluation of methods focuses on validation loss curves. I would expect a more comprehensive evaluation on memory overhead / running time, to understand the tradeoff between performance, memory, and time for Gluon.

3. The convergence in Theorem 2 requires specific decay laws $t_{k,i}\propto k^{-3/4}$, $\beta_k = 1- 1/\sqrt{(k+1)}. However, I think the advantage of the proposed gluon lies in the adaptive per-layer radii that are practically attractive. 

4. I suggest quantifying inexact LMO effects: add ablations varying SVD truncation rank / sign approximations and report the loss in performance and any stability changes. Also, to make the theoretical parts more solid, there could be some analysis over error-aware descent.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2
