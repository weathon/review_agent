# Toward Practical Equilibrium Propagation: Brain-inspired Recurrent Neural Network with Feedback Regulation and Residual Connections

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 2, 6

## Abstract
Brain-like intelligent systems need brain-like learning methods. Equilibrium Propagation (EP) is a biologically plausible learning framework with strong potential for brain-inspired computing hardware. However, existing implementations of EP suffer from instability and prohibitively high computational costs. Inspired by the structure and dynamics of the brain, we propose a biologically plausible Feedback-regulated REsidual recurrent neural network (FRE-RNN) and study its learning performance in EP framework. Feedback regulation enables rapid convergence by attenuating feedback signals and reducing the disturbance of feedback path to feedforward path. The improvement in convergence property reduces the computational cost and training time of EP by orders of magnitude, delivering performance on par with backpropagation (BP) in benchmark tasks. Meanwhile, residual connections with brain-inspired topologies help alleviate the vanishing gradient problem that arises when feedback pathways are weak in deep RNNs. Our approach substantially enhances the applicability and practicality of EP. The techniques developed here also offer guidance to implementing in-situ learning in physical neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a feedback-regulated residual recurrent neural network (FRE-RNN) to address the instability, slow convergence, and high costs of Equilibrium Propagation, a biologically plausible alternative to backpropagation that uses RNN dynamics for credit assignment. By scaling down feedback strength to accelerate settling and adding residual connections to counter vanishing gradients in deep networks, the approach draws from brain-like cortical modulation and recursive topologies. Evaluated on MNIST and CIFAR-10, FRE-RNN achieves BP-comparable accuracy with orders of magnitude speedups and enhanced stability.

### Strengths
- FRE-RNN shows practical improvements to EP while matching backpropagation accuracy on MNIST.
- The paper includes systematic ablations on hyperparameters such as T, K, β_i, α_i, and network depth. Comparisons in Table 1 and Figure 6 illustrate key trade-offs. 
- This work advances energy-efficient computing suitable for neuromorphic hardwares.

### Weaknesses
- Individual components like feedback scaling and residual connections are not novel, and the biological framing is rather superficial.
- Evaluations are limited to MNIST and CIFAR-10 datasets. There are no tests on sequential or natural language processing (NLP) tasks. Can the authors demonstrate the proposed method on at least one sequential processing / NLP tasks?

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes an improved network architecture for training with equilibrium propagation. This new architecture includes scaling down of feedback signals to improve convergence speed and, to prevent vanishing gradients caused by the weaker feedback, residual connections in both forward and backward direction.

The contributions in this paper are indeed important steps in making equilibrium propagation more practically feasible, as it's applicability is often hampered by excessively long training times and the restriction to small networks.
However, in the current state I can not recommend acceptance of this paper, for two main reasons. First, the presentation of the methods and results in the paper lacks clarity in many places (see details below in the "Questions" segment), which hampers the understanding and the applicability of the presented methods for others. Second, the experimental evidence is limited and a crucial sanity-check/ablation study is missing (see details below in the "Weaknesses" segment).

If these concerns are sufficiently addressed in the rebuttal, I am willing to increase my score of the paper.

### Strengths
- Making the dynamics in EP converge faster by scaling down the feedback signals is an important contribution, as the slow training times are a big drawback of the standard EP methods.
- Adding the residual connection is a good and, as far as I am aware, novel idea. It directly tackles the problem of vanishing gradients/error signals in lower layers, which is made worse by the weak feedback used to speed up convergence. Therefore, the paper immediately provides a solution to a drawback of its proposed method.

### Weaknesses
Experimental evidence:
- Networks with one hidden layer can solve MNIST. If a network with more hidden layers is trained on MNIST (or in fact even if the single hidden layer is wide enough), it can happen that only the weights of the top layer need to adjust to produce very high accuracies (> 95%). Therefore, just achieving high MNIST accuracies in a big (> 1 layer and > 200 neurons) does not actually prove that meaningful error signals are reaching the lower layers (which is the point of an algorithm aiming to approximate backprop). 
- I suggest the following (in my opinion crucial) ablation study to remove this doubt:
	- Train a network normally (like you did) and observe the resulting distributions of the weights in each layer.
	- Initialize a new network randomly according to those distributions.
	- Freeze the weights in the lower layers and train the others.
	- Compare accuracies. If accuracies reached in this ablated model are significantly lower than in the fully trained case, we can be convinced that meaningful error signals reach the lower layers.
- The used networks have too many layers, more than are needed to solve the task. This is illustrated by the fact that also in the BP networks the most shallow ones have the best accuracy (even though they have the fewest parameters). Due to this heavy overparameterization I am not sure if any real conclusions can be drawn from these results, because the lower layers were never needed to actually solve the tasks: For example in Sec. 4.3 the paper concludes  "Here, we found that due to gradient differences across different layers induced by weak feedback, a 3-hidden-layer RNN at βi = 0.01 (Table 1, ‘ours (tanh)’) learns well with a uniform learning rate", i.e. the weight updates for the lowest layers do not need to be amplified by a higher learning rate. At the same time, the next section states that with increasing depth, gradients are weaker and weaker due to the attenuation via beta. An explanation in which the two statements do not contradict is: it did not matter if the lower layers only received a very weak error signal, because they were unnecessary in the first place.
- The paper claims in prominent places (second sentence of the abstract, last point in the contributions and final sentence of the paper) to be a strong candidate for applicability on neuromorphic hardware and to be biologically plausible. In order to be able to claim this, in my opinion, it is necessary to prove that this algorithm deals well with noise (which is prevalent in both biological and neuromorphic systems). In particular it needs to be shown, that weight quantization and/or noise on weights as well as time-varying noise on the state variables s do not substantially impact functionality.

Literature:

- Weak feedback has been a prominent feature in other bio-plausible backpropagation algorithms, this should be acknowledged [1, 2, 3]

[1] Sacramento, João, et al. "Dendritic cortical microcircuits approximate the backpropagation algorithm." _Advances in neural information processing systems_ 31 (2018).

[2] Haider, Paul, et al. "Latent equilibrium: A unified learning theory for arbitrarily fast computation with arbitrarily slow neurons." _Advances in neural information processing systems_ 34 (2021): 17839-17851.

[3] Meulemans, Alexander, et al. "Credit assignment in neural networks through deep feedback control." _Advances in Neural Information Processing Systems_ 34 (2021): 4674-4687.

### Questions
Areas of the paper lacking clarity:

- Fig 1, Eq 2 and surrounding text contradicting each other:
	- The text around Eq 2, the surrounding inline equations and the referenced Fig 1 describe a 2-hidden-layer setting (explicitly mentioned in the text) but Eq 2 does not match that, it seems to describe a single layer setting. This is confusing.
	- The quantity alpha which is part of the model depicted in Fig 1 is missing in the Eq 2, it is not clear for which of the forward weights this scaling is really applied and for which it is not?
- Recurrency in the network (Eqn 1, 2 vs Alg. 1):
	- Eqs 1 + 2 feature an explicit recurrency within the layer. In Eq 2 this is expressed via the parameter W. However, Alg. 1 and Fig 1 do not feature those weights.
	- What about the learning of the layer-internal weights W? Are they not adjusted? How are they chosen if they are fixed?
	- Do your networks contain layer-internal recurrent weights W or not?
- Fig 1b: the depicted network structure is not helpful to actually understand the CNN-based setup. Could there be a more extensive illustration, e.g. in the appendix?
- Fig 2:
	- What is the value denoted by color? The state variable s? If yes of which layer? A label on the color bar would help.
	- If two color bars for left and right columns would be used, the actual values would be better visible (the two columns have different ranges of values, so if each gets a colorbar more details are visible).
	- Caption: "The hidden layer neurons are numbered from input to output." What does that mean? How can they be arranged, it's a recurrently connected set of neurons? What makes one closer to the input and another closer to the output?
- Choices/description of experimental setup unclear:
	- l. 198-200: How exactly is the comparison with BP and FA done? Is the same network architecture trained with BP or FA (i.e. BP through time) or is a vanilla ANN of the same feed-forward sizes used?
	- Section 3.3: Why can't you have AGT for symmetric setups or regularly spaced residuals like in Fig3a for asymmetric setups?
	- Why do the numbers in Fig 5 not match the numbers in Tab 1?
	- Did the setups in Fig 4 + 5 have skip connections?
	- Tab1: Why are there no BP results for the 2HL case?
	- Tab2: According to the statement that T = 10 x N_hidden was used, the convergence times set for these results is much higher than anywhere else in the paper. Why? Also, this should be at least mentioned as a qualifier in the text.
- Fig 6:
	- What is the network size (layer widths not given)?
	- What is K?
	- Caption: "By default T = 10 * N_hidden" Does that really mean that you are comparing networks of different sizes here? And that in a - d) the red network has a hidden layer that consists of 1 neuron? 
- Comparison of accuracies in Figures 4 and 6:
	- When accuracies are plotted for multiple different models it is very difficult to actually see the final accuracies (and their differences) in the plots. It would be much easier to see if errors instead of accuracies were plotted and the y-axis would be on a log-scale.
	- Fig 6: Additionally, it would be much easier to interpret, if the plots shared the same y-axis range.

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Equilibrium propagation is a biologically plausible method for credit assignment in deep neural networks. It operates through two phases, each requiring convergence to a fixed point. However, this relaxation process becomes exponentially slower as network depth increases. This paper addresses this problem by weakening feedback connections and introducing skip connections. The authors then demonstrate empirically that these modifications reduce convergence time and thus improve performance.

### Strengths
**Innovative approach to the relaxation time problem.** The authors address the increased relaxation time in deeper networks indirectly by effectively reducing the depth over which error signals propagate. After several multiplications by the $\beta_i$ coefficients that scale the feedback weights, negligible signal remains. The residual connections provide an intuitive mechanism for ensuring all layers receive teaching signals.

### Weaknesses
My main concerns are that the results are difficult to trust and the paper lacks proper analysis of what the method is actually doing.

**Insufficient analysis of the method's mechanisms.** The proposed architectural modifications can be understood as a way to learn only layers that have direct impact on the output (similar to reservoir computing). This interpretation raises several critical questions: How far in the hierarchy are accurate error signals propagated? How would the proposed method behave if only the last layer or last few layers were trained? Answering these questions is essential for scientific understanding of the proposed modification and is currently absent. As a side note, the proposed modifications are rather independent to EP, analyzing their effects on other biologically plausible learning algorithms would strengthen the analysis.

**Results are difficult to trust.** This stems from both presentation issues and technical inaccuracies:

- The dynamics of Equation (2) are not energy-based dynamics and, more importantly, do not converge to stationary points of the energy function. Consequently, the EP update will only be an approximation. This limitation is not mentioned anywhere in the paper.
- The EP framework (which receives minimal introduction) provides simple and elegant ways to describe the corresponding learning algorithm, yet the authors complicate it unnecessarily. For instance, they fix the error to its forward pass value; while correct, this adds unnecessary complexity. Algorithm 1 occupies more than half a page describing something explained concisely in the main text. Using the same symbol ($\beta$) for both feedback strength and nudging strength creates confusion. While these could be dismissed as cosmetic details, collectively they create an impression of confusion that undermines the paper.
- The typical baseline for EP is recurrent backpropagation, not standard backpropagation, since the network is recurrent. Backpropagation can only be applied to a feedforward version of the model, which is generally non-trivial to define. Consequently, the empirical comparisons feel like comparing apples to oranges.

**Additional remarks:**

- Figure 2 requires more statistically robust analysis.
- It is unclear why the authors sweep over forward connection strengths in Section 4.1, as this appears unrelated to their method.
- Figure 6 does not convincingly demonstrate that "Larger β_i requires more iterations for the RNN to reach fixed point."
- Empirical results are overly descriptive and sometimes lack proper interpretation. For example, the takeaway from Section 4.3 is unclear.

In my opinion, the paper is not currently ready for acceptance. It requires another extensive round of improvements to both the results themselves and their presentation.

### Questions
--

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces FRE-RNN, a biologically inspired recurrent neural network designed to make Equilibrium Propagation (EP) practical for modern deep learning. EP is a local, brain-inspired learning rule but has traditionally suffered from slow convergence and instability in deeper architectures. To address this, the authors introduce feedback regulation, scaling the feedback weights to less than 1 to stabilize dynamics and accelerate convergence. They further incorporate residual and random skip connections to improve gradient flow and enable deeper recurrent structures. Experiments on MNIST demonstrate that FRE-RNN achieves significantly faster runtimes than standard EP while maintaining accuracy comparable to backpropagation, and that it can successfully train deeper RNNs than previous equilibrium-based methods.

### Strengths
This paper presents an interesting and well-motivated step toward developing biologically inspired yet computationally practical learning systems. The proposed FRE-RNN introduces effective modifications, such as feedback regulation and residual connections, that significantly improve the efficiency and stability of Equilibrium Propagation models. The authors demonstrate that their approach achieves at least an order-of-magnitude speedup in convergence while maintaining accuracy comparable to backpropagation-based networks. The experiments are clear, well-structured, and provide convincing evidence that these improvements make equilibrium-based RNNs more scalable to deeper architectures. Overall, the paper is clearly written, conceptually sound, and provides strong justification for the proposed modifications.

### Weaknesses
While the proposed framework demonstrates clear improvements over prior equilibrium propagation methods, it seems to only work well on relatively simple datasets. The approach performs well on MNIST but struggles on CIFAR-10, suggesting that it may not yet generalize effectively to more complex, real-world scenarios. Although the paper emphasizes the potential of bridging neuroscience and machine learning, it would benefit from a deeper discussion on the practical implications and real-world applicability of such biologically inspired models, especially given their current performance limitations. In addition, the experiments focus on image data, which is may not be the best fit for RNN-based architectures. Demonstrating results on temporal or sequential modalities, such as text, would have strengthened the case for the proposed approach. Finally, while the paper is generally well written, certain sections, particularly Section 4.1, could be improved for clarity and flow, as some results are presented in a non-linear order.

### Questions
1.	If the proposed model, which aims to closely mimic biological mechanisms, performs poorly on more complex datasets, what practical advantage does it offer over less biologically inspired models that achieve much higher accuracy?
2.	For the results shown in Figure 5, were the reported trends averaged over multiple runs, or are they based on a single experiment? Can the authors conclusively state that lower beta values and higher alpha values consistently lead to better performance across architectures and datasets?

### Soundness
2

### Presentation
3

### Contribution
3
