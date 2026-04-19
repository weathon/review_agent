# Effective Learning with Node Perturbation in Multi-Layer Neural Networks

- Decision: Reject
- Scores: 6, 6, 3

## Abstract
Backpropagation (BP) remains the dominant and most successful method for training parameters of deep neural network models.
However, BP relies on two computationally distinct phases, does not provide a satisfactory explanation of biological learning, and can be challenging to apply for training of networks with discontinuities or noisy node dynamics.
By comparison, node perturbation (NP) proposes learning by the injection of noise into network activations, and subsequent measurement of the induced loss change. NP relies on two forward (inference) passes, does not make use of network derivatives, and has been proposed as a model for learning in biological systems.
However, standard NP is highly data inefficient and unstable due to its unguided noise-based search process.
In this work, we investigate different formulations of NP and relate it to the concept of directional derivatives as well as combining it with a decorrelating mechanism for layer-wise inputs.
We find that a closer alignment with directional derivatives together with input decorrelation at every layer strongly enhances performance of NP learning with large improvements in parameter convergence and much higher performance on the test data, approaching that of BP.
Furthermore, our novel formulation allows for application to noisy systems in which the noise process itself is inaccessible.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
In this paper, the authors extend an existing framework to train multi-layer neural networks without using backpropagation (BP). Node perturbation (NP) relies on injecting noise in the network layers, measuring the in the loss differential between the clean and noisy pass, and computing a layer-wise update which relies on the noise vector, the pre-synaptic activity and the loss differential. The authors improve on the NP framework with three main contributions:
* In the traditional NP approach, the effect of the layer $\ell$ noise $\epsilon_\ell$ on the downstream layers is unaccounted for. By computing the directional derivative of the loss on the noise injected at layer $\ell$ ($\nabla_v \mathcal{L}$), they can more precisely target the updates to layer $\ell$. This method, referred to as iterative node perturbation, requires $L+1$ forward passes, and relies on access to the noise vector.
* Next, the authors propose activity-based noise perturbation (ANP). This approach relies on the assumption that all the layers are independent, which requires measuring the state difference between clean and noisy passes instead of the noise injected. This requires only two passes and does not require access to the noise signal, but rather its effect on the network.
* Lastly, using an existing trainable decorrelation procedure, they show improved performance of their proposed algorithms by decorrelating the inputs to each layer.

These variations of NP are tested fully connected and convolutional networks on CIFAR-10 and Tiny ImageNet and compared to BP.

### Strengths
I believe the paper is well structured and well written, the results are interesting and overall well defended. In detail:
* The methods section is clear and straightforward. I appreciated the incremental structure that starts from Node Perturbation ad adds on the newly proposed variations explaining well the contribution of each piece.
* The results prove the claims of the authors regarding how each variation of NP compares, and I appreciate using Tiny ImageNet as a benchmark which is more challenging of what is usually found in these types of papers.
* I found section 3.4 very interesting, as most methods I know of rely on having non-noisy systems. I think removing the assumption of having a clean and noisy pass, and assuming all passes are noisy makes it a very interesting algorithm.

### Weaknesses
I am giving this paper a 6 because of the following weaknesses I found, but I would be happy to re-evaluate if these concerns are addressed. My main concerns are:
* The world bioplausible is mentioned throughout the papers, but details on how these algorithms could be implemented in biological neural networks are not provided. I believe the paper still stands without needing to justify it as bioplausible, so I believe that either removing the bioplausibility aspect (and exchange it with more details on possible hardware implementation) or providing more details about the bioplausibility would be better alternatives.
* In the last years, many alternatives to back propagation that rely on multiple forward passes have been proposed. In particular, looking at equation (6) in the paper. For example, Dellaferrera and Kreiman, 2022, proposes to use the differential between a clean and "noisy pass" to train the network, where the noisy pass relies on a perturbation of the activity computed using the loss differential. Since there are many other algorithms like this (for example Hinton 2022, Kohan et al. 2018, Frenkel et al 2021), I believe a comparison in the methods section would be beneficial.
* The figures position could be improved to make the paper more readable. For example Figure 1 could be pushed up in the paper as it summarizes the contributions of the paper, Figure 4 could be moved in section 3.3.
* I think figure 2 could be improved. For example figures 2 right does not add anything that cannot be explained in words.

### Questions
These questions/suggestions mostly add on top of what I mentioned in the weaknesses:
* What happens to the update angle during the training. Do you observe a better alignment close to convergence? If this is interesting it could be added to Figure 2.
* I am pretty surprised about the results with a fully connected network trained with BP, as it is pretty common to see these networks obtaining >60% accuracy on CIFAR-10. Could you elaborate more on the choice of hyperparameters?
* I believe is always important to include a code repository in these papers as it helps the other researchers in the field and makes the experiments easier to reproduce.

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
Backpropagation (BP) is the standard for training deep neural networks but is criticized for its lack of biological plausibility and computational complexity due to separate forward and backward phases. Node Perturbation (NP) offers an alternative approach by injecting noise into hidden layer activities and using the resulting change in the loss function to guide weight updates. However, traditional NP is inefficient, unstable, and requires precise noise control, limiting its practical utility and biological relevance.

This study extends NP by introducing more robust formulations. It reframes NP using the concept of directional derivatives, leading to an iterative approach (INP) that better aligns with BP in terms of gradient estimation. Additionally, the paper presents an activity-based variant (ANP) that estimates the gradient using differences between clean and noisy activations, thus bypassing the need for precise noise measurement. A key contribution is integrating a layer-wise input decorrelation mechanism, which mitigates the bias in NP updates and accelerates convergence.

Numerical experiments demonstrate that these modified NP algorithms, particularly when combined with input decorrelation, significantly enhance performance compared to standard NP and, in some cases, approach BP-level accuracy. The study also shows that these methods can be extended to noisy systems where the noise process is not directly observable, making them applicable to both neuromorphic computing and potential models of biological learning.

### Strengths
**Biological Motivation**: The exploration of node perturbation (NP) as an alternative to backpropagation is compelling due to its alignment with plausible biological mechanisms, negating the need for backward passes and allowing learning from reward signals. Previous work on NP suffers from poor performance and reliance on specific and accurate noise control. This work improved on previous studies by offering significant improvements that could potentially make NP a competitive framework.
- **Innovative Formulations**: The introduction of iterative node perturbation (INP) and activity-based node perturbation (ANP) adds theoretical depth, notably linking perturbation approaches with directional derivatives and improving the stability of NP in noisy environments. In particular, the authors show that the loss gradient can be computed without precise control over the noise process. This solution is elegant and informative.
- **Decorrelation Mechanism**: The incorporation of input decorrelation as an unsupervised learning mechanism demonstrates clear improvements in convergence speed, adding practical value to NP and its variants, while maintaining biological plausibility.

### Weaknesses
1.  **Scalability Limitations**: While the paper suggests that scaling to larger problems could be addressed through parallelization, this solution conflicts with the biological motivation emphasized throughout the text. The authors should reconcile this discrepancy by exploring biologically feasible alternatives or clarifying the practical biological implications.  Notably, this approach is clearly relevant for neuromorphic computing, particularly since the noise perturbations in this framework can be arbitrarily small.
   
2. **Gradient Approximation**: Theoretical analyses (e.g., Equation 4) focus on mean gradients. Still, the role of noise variance and the number of noisy samples in the stability and efficiency of gradient estimates are underexplored. Since the authors emphasize the framework's efficiency, claiming high performance can be achieved with a small number of noisy passes, the typical, rather than the mean loss gradient, should be analyzed. 

3. **Decorrelation Analysis**: Adding an unsupervised learning rule for input decorrelation in each layer is an intriguing and potentially beneficial approach. However, the paper’s analysis of this aspect is insufficient, both numerically and theoretically. The improvement observed from decorrelating inputs, as demonstrated in Figure 3, is unsurprising. In a single-layer architecture, this step functions similarly to an additional linear transformation or data preprocessing step, which is expected to yield performance gains. This effect diminishes the novelty of the finding.
   Furthermore, while Figure 4 shows notable improvements in BP training accuracy with decorrelation, the minimal test accuracy gains suggest overfitting, indicating that the method primarily accelerates convergence without enhancing generalization. This point needs further exploration to determine the trade-offs between train and test performance. Moreover, Figure 4 highlights that input decorrelation does not enhance—and may even degrade—performance in convolutional networks. This discrepancy calls for a more thorough investigation into the conditions under which decorrelation aids or hinders performance. The authors should address these limitations and clarify whether decorrelation consistently benefits deeper and more complex architectures, or if its effectiveness is limited to simpler cases.
   
4. **Learning in the deep hidden layers**: Figure 2 indicates that the gradient alignment in the output weights closely matches that of BP, which may be due to the low-dimensional nature of the output space. This raises a concern that the observed performance, which falls short of BP’s, could be primarily driven by the readout weights, potentially bolstered by the unsupervised decorrelation step applied to the layer activities. This implies that the NP algorithms may contribute minimally to learning in the deeper layers. 
   To address this issue, the authors should provide evidence that ANP/INP enhances learning throughout the network, rather than merely acting as a support for the final readout layer. Specifically, they should demonstrate that these algorithms outperform a simpler baseline approach involving unsupervised learning in the hidden layers followed by SGD or NP for training the output layer.
### Minor Comments
- **Appendix Clarifications**: The derivations in Appendix C do not add significant new insights beyond the main text and should be expanded to include formal proofs that strengthen the theoretical claims made in the main manuscript.
- **Clarification of Sample Averaging**: The use of a noise direction vector $v$ for each sample should be clearly articulated to explain how this averaging over noise directions ensures accurate gradient approximation.

### Questions
1. **Variance and Convergence of the Loss Gradient**: Could the authors provide an analysis of the variance in the estimated loss gradient as a function of the number of samples? Additionally, what is the typical convergence rate of the loss with an increasing number of samples? This data would offer valuable insights into the scalability and efficiency of the proposed methods.

2. **Impact of the Decorrelation Step**: Can the authors confirm that the observed performance improvements are not solely driven by the decorrelation rule? From Figure 4, it is unclear whether INP and ANP contribute significantly without the decorrelation step. What results would be obtained if only the decorrelation step was implemented, followed by NP applied solely to train the readouts?

While I have additional questions, addressing these principal concerns would be pivotal in reconsidering the rating of this work.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
In this manuscript, the authors propose an improved algorithm for node perturbation (NP) with two key modifications compared to the standard NP. First, the weight update is calculated using the total change in activity at each hidden node instead of the direct perturbation at that node. Secondly, the algorithm incorporates a decorrelation step at each layer to minimize noise correlations. The authors demonstrate numerically that decorrelation robustly enhances the performance of NP in deep neural networks trained on the CIFAR-10 dataset. They also show that using the total change in activity for updates outperforms the vanilla NP in convolutional neural networks.

### Strengths
The manuscript is clearly written and well-motivated. Moreover, the numerical experiments convincingly demonstrate that decorrelation improves the performance of NP.

### Weaknesses
Section 2.1.2, especially L127-L128, provides an impression that INP is required to make the update unbiased. However, the vanilla node perturbation is known to converge to the true backpropagation at the infinite sample limit if the noise added to each layer is independent of each other (Fiete et al., 2007; Hiratani et al., 2022). 

On the contrary, the ANP update rule is biased against back-propagation. This can be demonstrated in a one-hidden layer non-linear network $y = W_2 f (W_1 x)$ with a loss function L(y). 
Given perturbations $h v_1$, $h v_2$, at $h \to 0$ limit, the ANP update for the second layer is
$$\begin{eqnarray}
\Delta W_2 
&=& \frac{1}{h} \langle (L(\tilde{y}) - L(y)) (v_2 + W_2 [f'(W_1 x) \odot v_1] ) f(W_1 x)^T \rangle 
\nonumber \\\\
&=&   \langle (v_2 + W_2 [f' (W_1 x) \odot v_1] ) (v_2 + W_2 [f' (W_1 x) \odot v_1] )^T \rangle \frac{\partial L(y)}{\partial y}   f (W_1 x)^T 
\nonumber \\\\
&=& (I + W_2 \text{diag} [ f'(W_1 x)^2 ] W_2^T) \frac{\partial L(y)}{\partial y}  f(W_1 x)^T
\end{eqnarray}$$
Because the true gradient is 
$\frac{\partial L(y)}{\partial W_2} = \frac{\partial L(y)}{\partial y} f(W_1 x)^T$,
the ANP update rule above is biased against the true gradient, implying that the claims made in section 2.1 are inaccurate. 

Given this bias, it is unclear why ANP achieves better alignment with backpropagation in Figure 2. Nevertheless, biased updates can sometimes facilitate faster learning, as discussed in Song, Millidge, et al., Nature Neuroscience, 2024. Therefore, the results shown in the bottom panel of Figure 4 are potentially interesting.

Regarding comparison with BP, empirical results are presented in a somewhat misleading way. A key limitation of NP is its need for a low learning rate, which means performance comparisons with BP should be conducted across a wide range of learning rates.

### Questions
Could you elaborate on the implementation details in convolutional networks, particularly regarding how weight sharing was handled?

### Soundness
2

### Presentation
3

### Contribution
2
