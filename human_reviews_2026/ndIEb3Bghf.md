# CharFxReg : Characteristic Function based Regularisation

- Avg Score: 1.33
- Decision: Reject
- Scores: 0, 2, 2

## Abstract
Regularization plays a crucial role in neural network training by preventing overfitting and improving generalization. In this paper, we introduce a novel regularization technique grounded in the properties of characteristic functions, leveraging assumptions from decomposable distributions and the central limit theorem. Rather than replacing traditional regularization methods such as L2 or dropout, our approach is designed to supplement them, providing a contextual delta of generalization. We demonstrate that integrating this method into standard architectures improves performance on benchmark datasets by preserving essential distributional properties and mitigating the risk of overfitting. This characteristic function-based regularization offers a new perspective in the direction of distribution-aware learning in machine learning models.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The submission proposes a regularization method that involves matching the distribution of the (standardized) sum of instances of the output random variable of a predictive neural network model with that of a standard Normal, using the characteristic function representations. This is motivated by the Central Limit Theorem, which states that sums of a random variable converge to a Gaussian distribution.

Experiments show some improvements in generalization with respect to unregularized and ElasticNet models for a range of datasets and neural networks.

### Strengths
The key idea is intuitively sensible, and novel, to my knowledge.

The experiments are conducted over a wide range of datasets and models.

### Weaknesses
The primary weaknesses of the submission lies in adequately grounding the contributions in existing literature and in a robust experimental validation of the method.

  -- Regularization methods have been proposed in the past that also directly operate on the outputs of neural networks instead of weights, and the paper would be markedly strengthened if principled connections could be drawn to these earlier works, since it is likely that some of the existing methods have the same effect of "pulling in" overconfident predictive distributions: examples are label-smoothing, predictive-entropy regularization, spectral decoupling. Additionally, these are also more relevant to compare against, rather than solely ElasticNet.

 -- The experiments appear exhaustive at first sight, however the baseline numbers for almost all methods are quite lagging behind relative to where current state-of-the-art results are. On the one hand, one can argue that regularization effects are better demonstrated with room for improvement, but on the other, it seems pointless to showcase improvements on outdated baselines. Moreover, the performances are more often than not lagging behind the only point of comparison for regularizers, ElasticNet, which is hardly the most common regularizer unanimously used for such a wide range of models and datasets.

### Questions
No additional questions

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a regularization technique based on the theory of characteristic function. The basic idea is to apply Lyapunov Central Limit Theorem on the model output (deep features in this paper). The characteristic function is derived to converge to the standard Gaussian distribution by applying the theorem. Then the paper proposes to explicitly drive the empirical output distribution estimated by training examples to the standard Gaussian, and uses their distance as a regularization term. Multiple experiments are conducted to evaluate the proposed method over diverse data modalities and models.  From the experimental results, the proposed method shows clear advantages in the Generalization Efficiency Score (GES) metric, but doesn’t seem to improve existing regularization techniques such as ElasticNet in terms of test accuracy.

### Strengths
1. Diverse datasets and models are used in the experiments. 
2. The proposed method achieves best results in terms of GES.

### Weaknesses
1. The rationale of the regularization method is not quite clear. Since the characteristic function $\phi_{\mathcal{D}}$ already has the expected convergence to $\phi_{\mathcal{N}(0,1)}$ when n is large, why should we employ an additional regularization term constraining their distribution distance?
2. Implementation details are not described clearly, especially how the empirical term $\phi_{D}$ is calculated in model training. From Eq (7), it seems to also rely on $p_i$. Please provide source code or detailed pseudo code to clarify the steps and important hyperparameters.
3. The proposed method doesn’t show better experimental results in terms of test accuracy. The GES metric is reasonable, but test accuracy is still the most essential metric for most real applications.  
4. Page 5 claims that ImageNet and ImageNet-21K are used for evaluation but results are not reported. 
5. Figure 2 may not be the most appropriate way to present the corresponding result, since the X-axis (Dataset Index) does not represent a continuous or ordered sequence. Using a line chart implies temporal or sequential relationships between adjacent points, which is not the case here.

### Questions
1. What is the time and memory complexity of the proposed method? 
2. In section 3.1 the authors explain the motivation of the work is from “the observation that, in many classification settings, the output of a neural network (especially under sigmoid or softmax activation) can be interpreted as a sequence of Bernoulli trials”. Is there any concrete evidence for this?

### Soundness
2

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes "characteristic-function-based regularisation", which (1) models the outputs of a neural network by a chain of Bernoulli variables and (2) regularizes its characteristic function to be that of the unit normal distribution, based on the Lyapunov Central Limit Theorem of independent random variables. The proposed method is evaluated on a range of classification benchmarks that span the text, audio, and image domains.

### Strengths
Using characteristic functions to regularize neural networks is a novel idea to my knowledge.

### Weaknesses
The main weakness of the paper is that it does not really provide enough implementation details for the proposed method to be assessed. From my understanding, the paper treats the output activations of the neural networks as a weighted sum of independent Bernoulli variables. However, in Axiom 1 and Definition 3, it seems that $\mathscr{N}$ refers to "_data points_" sampled from the data distribution rather than network outputs, which is misleading. Moreover, it is not explained whether such treatment is applied to each output dimension (e.g., the classification probs of each class) independently or to the sum of all dimensions. It is neither clear to me how exactly the regularization term is calculated. Please see Questions for more details.

The motivation of the paper is also not very clear to me. Why could the output activations of neural networks be treated as a weighted combination of Bernoulli variables (except that they are indeed in the range $[0, 1]$)?

Finally, I do not think the current empirical results are significant enough to support the claim that the proposed method could improve generalization. Many of the gains on the proposed "generalization metrics" are simply due to the _decrease_ on train acc rather than the increase on test acc, which is meaningless in practice.

### Questions
- What does $\mathscr{N}$ exactly correspond to in a neural network?
- If $\mathscr{N}$ should asymptotically be unit Gaussian, why not directly regularize the distribution of $\mathscr{N}$ (e.g., by regularizing its moments) instead of resorting to its characteristic function?
- Why the output activations of a neural network should be modeled as a weighted combination of Bernoulli variables? What is the semantics of each Bernoulli variable?
- Appendix B: What are $u_k$ and $k$?

### Soundness
1

### Presentation
1

### Contribution
2
