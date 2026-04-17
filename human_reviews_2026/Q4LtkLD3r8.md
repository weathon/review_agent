# Random Feature Spiking Neural Networks

- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Spiking Neural Networks (\textit{SNN}s) as Machine Learning (\textit{ML}) models have recently received a lot of attention as a potentially more energy-efficient alternative to conventional Artificial Neural Networks. The non-differentiability and sparsity of the spiking mechanism can make these models very difficult to train with algorithms based on propagating gradients through the spiking non-linearity. We address this problem by adapting the paradigm of Random Feature Methods (\textit{RFM}s) from Artificial Neural Networks (\textit{ANN}s) to Spike Response Model (\textit{SRM}) \textit{SNN}s. This approach allows training of \textit{SNN}s without approximation of the spike function gradient. Concretely, we propose a novel data-driven, fast, high-performance, and interpretable algorithm for end-to-end training of \textit{SNN}s inspired by the \textit{SWIM} algorithm for \textit{RFM}-\textit{ANN}s, which we coin \textit{S-SWIM}. We provide a thorough theoretical discussion and supplementary numerical experiments showing that \textit{S-SWIM} can reach high accuracies on time series forecasting as a standalone strategy and serve as an effective initialisation strategy before gradient-based training. Additional ablation studies show that our proposed method performs better than random sampling of network weights.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper proposes S-SWIM, a gradient-free method to train SNNs for time series forecasting. Experimental results have shown its effectiveness.

### Strengths
The S-SWIM method is novel for SNN training. Especially, it does not need gradient backpropagation.

### Weaknesses
1. This paper is difficult to understand. I recommend the authors to draw an illustration figure of the S-SWIM method to help the readers better understand it.
2. The paper is written like a technical report rather than a paper. For a paper, the authors should propose the scientific problem this paper aims to address, then introduce how the method addresses this problem. The whole part of Section 3 introduces the technical details without holistic intuition. Also the experimental settings (baselines, training method details, etc.) are vague in Section 4.
3. The baselines in Table 1 are not cited. I am not familiar with these works, as far as I know, Spikformer and QKFormer are designed for vision tasks, how do the authors use them in time series forecasting tasks? In addition, it seems that works in 'SNNs for Time Series Forecasting' in Related work are not included in this table.

### Questions
1. There lacks the explanation for Eq (1)(2)(3). I recommend the authors to introduce $\phi$ and $\eta$ in Eq (2)(3) (spike response kernel and reset kernel). Besides, what is PSPK, RfKin line 133? A full name should be given. Where do the spike-cost $c$ appear in the equations?
2. In line 157, it seems that $\tau_{max}$ is not continuous at $O=H$. This design is a bit weird, does it have any intuitions?
3. In Eq. (6)(7), what is $-3 \cdot s_t$ in the normal distribution? Normal distributions only have two parameters in common.
4. Authors have said that S-SWIM is much faster than SGD. Where is the experimental data to support this conclusion?

### Soundness
2

### Presentation
1

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
This paper proposes a Random Feature Spiking (RFS) method to enable efficient supervised learning in Spiking Neural Networks (SNNs) by treating membrane dynamics as random projections. The authors offer theoretical guarantees showing consistency to SGD-based weight updates and demonstrate preliminary experimental evaluation.

### Strengths
1. The paper provides sufficient theoretical justification for the proposed method.
2. A promising solution for enabling global weight updates in SNNs similar to SGD.

### Weaknesses
1. Since the paper states conceptual relationships to SWIM, but does not empirically or algorithmically contrast against it, it remains unclear that whether temporal parameters meaningfully impact learning quality, and whether RFS is fundamentally more scalable or simply a variant of existing methods.
2. Experimental evaluation is insufficient to validate claims
The paper contains only one dataset and one architecture configuration, with missing details such as:
number of synapses / learned parameters in neurons,
Spike-based computation metrics (SynOps / NeuOps).
3. Writing clarity and notation consistency need improvement
Numerous grammatical mistakes, unclear expressions, and inconsistent mathematical notation negatively affect readability. Some expressions lack necessary subscripts or arguments, making it harder to follow derivations.
4. Presentation misses a comprehensive contextual comparison
The related-work section briefly lists methods but does not sufficiently articulate technical distinctions from:
Local learning rules like e-prop [1]
Sparse BP [2,3]
Current efficient training methods like gradient estimation [4] and integer spikes [5]
5. Conclusion overstates findings relative to evidence
The claims of “better credit assignment” remain theoretical without adequate practical demonstration.

[1] Bellec, G., Scherr, F., Subramoney, A. et al. A solution to the learning dilemma for recurrent networks of spiking neurons. Nat Commun 11, 3625 (2020).
[2] Meng, Qingyan, et al. "Towards memory-and time-efficient backpropagation for training spiking neural networks." Proceedings of the IEEE/CVF international conference on computer vision. 2023.
[3] Perez-Nieves, Nicolas, and Dan Goodman. "Sparse spiking gradient descent." Advances in Neural Information Processing Systems 34 (2021): 11795-11808.
[4] Meng, Qingyan, et al. "Training high-performance low-latency spiking neural networks by differentiation on spike representation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
[5] Luo, Xinhao, et al. "Integer-valued training and spike-driven inference spiking neural network for high-performance and energy-efficient object detection." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
see in weakness

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
2

### Summary
This paper proposes a gradient-free training method for spiking neural networks (SNNs) based on the spike response model (SRM), leveraging the random feature method. It introduces a sampling strategy for data points, a weight design scheme for temporal parameters, and an approach for handling output delays.

### Strengths
**S1.** This paper addresses an important topic — exploring how to train spiking neural networks (SNNs) without relying on gradient-based methods.

**S2.** The incorporation of the random feature method into SNN training is novel, and the experimental results demonstrate its effectiveness to a certain extent.

### Weaknesses
**W1.** The paper introduces a large number of notations, which makes it somewhat difficult to follow. In particular, Definition 3.1 employs multiple types of brackets—{}, () and []—without clear distinction (e.g.,$\phi{x}, f(t), and \zeta[v]$), which can be confusing. Moreover, the use of the symbol “!=” should be clarified. Since I am an emergency reviewer with limited time to thoroughly read the paper, this issue significantly affects the readability and comprehension of the technical content.

### Questions
**Q1.** According to Figure 1, why do S-SWIM-trained networks fine-tuned with SGD (i.e., SGD-1) require less training time than the S-SWIM-trained networks themselves in some cases?

**Q2.** What is the architecture of the network trained using S-SWIM? Since S-SWIM models each neuron individually, the training time may increase linearly with the number of neurons. Could the authors clarify this relationship?

**Q3.** How sensitive is S-SWIM to the number of random features?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes S-SWIM, a gradient-free training algorithm for Spike Response Model (SRM) spiking neural networks based on Random Feature Methods (RFMs) and inspired by the SWIM algorithm previously developed for ANNs. The key idea is to construct hidden-layer weights and temporal parameters (delays, kernel widths) in a data-driven way so that pairs of inputs with similar signals but dissimilar targets are separated in function space, and then to solve a linear problem for the output layer. This avoids backpropagating gradients through the non-differentiable spike function and thus avoids surrogate gradients entirely.

Experiments focus on time-series forecasting on four standard datasets (METR-LA, PEMS-BAY, Solar, Electricity) with multiple prediction horizons. The authors compare: (a) S-SWIM alone, (b) surrogate-gradient SGD (Adam), and (c) S-SWIM used as initialization followed by SGD, and report RSE metrics and training times. S-SWIM often matches or outperforms pure SGD on these forecasting tasks and is reported to be one to three orders of magnitude faster in training time.

### Strengths
The paper cleanly adapts data-driven random feature sampling from SWIM to a Spike Response Model SNN setting, including temporal parameters (delays, kernel supports) as part of the random feature construction. This is more principled than purely random ELM-style initialisation and fits well with the SNN temporal structure. 

S-SWIM provides a surrogate-gradient-free alternative, addressing a real pain point in SNN training (biased surrogates, gradient instability, heavy compute/memory cost). All learning is done via sampling, linear algebra, and correlation analysis over spike trains and kernel responses.

### Weaknesses
All experiments are on tabular / numerical time-series datasets (METR-LA, PEMS-BAY, Solar, Electricity) with RSE metrics. 

There are no experiments on canonical SNN benchmarks such as neuromorphic image datasets (e.g., DVS variants), frame-based image classification (MNIST/CIFAR-like), audio/speech spiking tasks, or text/event-driven tasks.

Given the broad claims (“general fast training algorithm for SNNs”, “potentially more energy-efficient alternative to ANNs”) and the heavy emphasis on the generality of the SRM parameterization, the fact that only one task family (time-series forecasting) is actually tested makes the empirical validation feel incomplete. A reader cannot tell whether S-SWIM meaningfully helps in other common SNN application domains such as vision or speech.

The paper focuses on a relatively specific shallow feed-forward architecture (the “shallow network case”) and does not empirically validate S-SWIM on deeper networks, even though the method is claimed to be applicable to them.

There is no evaluation on classification tasks or tasks with spike-valued outputs, although the theory discusses them. This accentuates the feeling that the method is only demonstrated in the easiest setting for S-SWIM: continuous regression with L2 loss.

### Questions
Overall, this is a technically solid and well-written paper with a genuinely interesting idea: bringing modern data-driven random-feature sampling into the SNN world to obtain a fully gradient-free training scheme with good training-time savings on time-series forecasting benchmarks. The mathematical treatment and modular algorithm design are clear strengths.

However, from a conference-acceptance point of view, the experimental validation feels too narrow. The method is only demonstrated on numerical time-series forecasting, with no experiments on images, videos, text, or speech, and no demonstration on classification or neuromorphic benchmarks. Given the broad claims and ambitious framing, this limited evaluation undermines the impact.

### Soundness
3

### Presentation
2

### Contribution
1
