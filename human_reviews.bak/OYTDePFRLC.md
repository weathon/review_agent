# Forward Learning with Differential Privacy

- Decision: Reject
- Scores: 3, 6, 5, 6, 6

## Abstract
Differential privacy (DP) in deep learning is a critical concern as it ensures the confidentiality of training data while maintaining model utility. Existing DP training algorithms provide privacy guarantees by clipping and then injecting external noise into sample gradients computed by the backpropagation algorithm. Different from backpropagation, forward-learning algorithms based on perturbation inherently add noise during the forward pass and utilize randomness to estimate the gradients. Although these algorithms are non-privatized, the introduction of noise during the forward pass indirectly provides internal randomness protection to the model parameters and their gradients, suggesting the potential for naturally providing differential privacy. In this paper, we propose a privatized forward-learning algorithm, Differential Private Unified Likelihood Ratio (DP-ULR), and demonstrate its differential privacy guarantees.  DP-ULR features a novel batch sampling operation with rejection, of which we provide theoretical analysis in conjunction with classic differential privacy mechanisms. DP-ULR is also underpinned by a theoretically guided privacy controller that dynamically adjusts noise levels to manage privacy costs in each training step. Our experiments indicate that DP-ULR achieves competitive performance compared to traditional differential privacy training algorithms based on backpropagation, maintaining nearly the same privacy loss limits.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper introduces a new algorithm, the Differential Private Unified Likelihood Ratio method (DP-ULR), which enhances privacy protection in deep learning by utilizing forward-learning techniques instead of traditional backpropagation. DP-ULR integrates noise during the forward pass. Theoretical analysis and experiments claim to show that DP-ULR achieves better performance than DP-SGD.

### Strengths
1. This paper proposes a novel approach to DP training other than the popular DP-SGD
2. The paper conducts reasonable theoretical analysis and experiments.

### Weaknesses
I think the proposed approach is interesting but I have some questions about it.

1. Misleading DP-SGD baseline: It is common sense that the larger the batch size is, the better performance we can achieve with DP-SGD. See Dormann et al. (https://arxiv.org/pdf/2110.06255), Figure 2 of De et al. (https://arxiv.org/pdf/2204.13650) and Figure 4(b) of Yu et al. (https://arxiv.org/pdf/2306.08842). Why do the results in Table 1 of this paper give the opposite conclusion? Let's put your proposed method aside, and just look at the results of DP-SGD, could authors explain why the performance drops for large batch size when the noise level is the same? Could it be the learning rate set improperly? 

2. The results presented in Table 1 reveal that the performance gap between DP-ULR and DP-SGD varies and is relatively small in some instances. To draw more reliable conclusions, it would be beneficial to report the average accuracy across multiple runs, along with the standard deviation. Additionally, the performance of MLP on MNIST appears notably low, even at reduced noise levels.

3. For the entire experiment, it makes more sense to utilize more popular models such as ResNet-18 or Wide-ResNet-16-4 or 40-4 which is widely used in literature such as De et al. Also, it is very important to show whether the proposed method can be applied to the complex model which is more complex than ResNet-5 used in the paper.  BTW, it is not very clear from Figure 4 what is the final test accuracy for DP-SGD and DP-ULR. I can see they are close to each other. Does it mean that the proposed method can not achieve better utility than DP-SGD when dealing with a large model?

### Questions
Please see the weakness.

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
4

### Summary
In this paper, the authors theoretically analyze the feasibility of DP with noise introduced by forward learning and propose a ULR variant, DP-ULR. The paper presents the novel batch sampling operation with rejection, which is used in DP-ULR. The paper also analyzes the performance and privacy guarantee of DP-ULR in both theoretical and experimental terms.

### Strengths
- The authors discovered that forward learning's inherent randomness can be utilized to achieve DP, which is an enlightening combination of noise introduced by both DP and forward learning. 

- The paper introduces a novel sampling-with-rejection technique for DP-ULR, which provides a quadratic privacy amplification. Realizing such a technique may introduce impairment on the privacy cost, the authors propose an assumption for DP-ULR and design experiments to demonstrate that such an impairment is minimal. 

- The paper is well written, clearly presenting the motivation for combining DP and forward learning. Besides, the theoretical analysis part is presented in conjunction with traditional methods, which makes it easy for readers to follow.

### Weaknesses
- The authors mention that DP-ULR inherits certain advantages over backpropagation-based DP-SGD, such as suitability for non-differentiable or black-box settings, high parallelizability, and efficient pipeline design. However, the paper lacks experiments on non-differentiable or black-box settings to support these ideas. Even though current hardware or software implementation may limit the efficiency of DP-ULR, the authors do not provide an asymptotic complexity analysis of DP-ULR either.

### Questions
- How does DP-ULR perform on non-differentiable modules or in black-box settings as mentioned in the abstract?
- Could the authors provide an asymptotic complexity analysis of DP-ULR?
- The authors mention that non-isotropy may introduce redundant noise to DP-ULR, which may impair the accuracy of the gradient estimation. The reviewer wonders about the proportional relationship between this impairment and the number of model parameters. For example, experiments comparing DP-ULR and DP-SGD on different model sizes could be added to the paper to illustrate this relationship. It would help to clarify whether DP-ULR's performance degrades relative to DP-SGD as models become larger.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
Forward-learning algorithm is a kind of non-DP algorithm that updates the model parameters without back-propagation, which has advantage on parallelizability and black-box settings.  The paper introduces DP-ULR which attempts to ensure differential privacy (DP) in forward-learning by utilizing randomness in the forward pass. The method leverages a unique batch sampling approach with rejection and integrates a privacy controller to dynamically adjust noise for privacy-cost management. Their experimental results indicate the proposed method shares a comparable performance as the DPSGD algortihm.

### Strengths
- The research problem is clearly motivated by the limitation of DP-SGD, such as its infeasibility of black-box modules and differentiable requirement.
- The paper is well-written, with clear demonstration and algorithm details.
- The results are comparable to DPSGD, but without the need to perform back-propagation.

### Weaknesses
- **Insufficient differential privacy protection**: This work aims to provide a final $(\epsilon, \delta)$-DP, which by definition should bound the influence of a sample in the dataset, including both feature \(x\) and label \(y\). However, in Line 199 of Algorithm 1, for each sample $(x_i, y_i) \in B_t$ in a batch, only the output derived from the input $x_i$ is perturbed while $y_i$ is used with its ground-truth value. Thus, it might be unconvincing that the proposed algorithm guarantees DP.

- **Data-dependent noise magnitude**: As equation (8) shows, the noise magnitude is determined by calculating the minimum spectrum of a matrix, which is non-privatized. Therefore, calculating the sigma itself violates privacy, and the overall differential privacy guarantee might be problematic.

- **The efficiency advantage is unclear compared to ZeroDP**: Starting from Line 278, the algorithm requires one more forward pass to compute the Jacobian matrix and loss without any perturbation. Thus, in total, there are two forwards for every sample. There are several works that privatize loss values obtained in two forward passes in zeroth-order optimization for achieving DP guarantee on protecting every sample. If this work also requires two forward passes, what is the advantage compared to the line of ZeroDP works? In Section 3.4, the authors discuss the efficiency advantage compared to DP-SGD. However, there lacks discussion to compare the advantage to other forward-based DP algorithms, such as the following references:
  - Differentially Private Zeroth-Order Methods for Scalable Large Language Model Finetuning
  - Faster Differentially Private Convex Optimization via Second-Order Method
  - Private Fine-tuning of Large Language Models with Zeroth-Order Optimization

### Questions
- How does this compare with the line of work on ZeroDP?
- Why does using the ground truth $y_i$ still yield a sample-level DP guarantee?
- Does using the non-privatized minimum spectrum of a matrix to estimate $\sigma$ violate privacy?

### Soundness
2

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
The paper proposed DP-URL algorithm, which is an improvement of DP-SGD by dynamically adjusting the injected noise into the model.

### Strengths
The proposed algorithm improved the accuracy of DP-SGD without sacrificing the privacy, especially in the large batch-size applications.

### Weaknesses
The runtime and complexity of the proposed algorithm are not well discussed. Even though DP-SGD does not require gradient clipping, it requires the computation of the Jacobian matrices, whose complexity seems to be cubic if I understand correctly.

### Questions
What is the complexity of one step in DP-URL? 

What is the runtime of DP-URL in comparison with DP-SGD?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces the Differential Private Unified Likelihood Ratio (DP-ULR) method,to achieve differential privacy in deep learning through forward learning. Unlike traditional backpropagation-based methods, DP-ULR leverages the inherent randomness of forward learning to estimate gradients.

### Strengths
1. This work studies an important problem with some new insights. 

2. Extensive theoretical analysis is provided.

3. The paper is well written.

### Weaknesses
1. The assumption of this work may not always hold in practice. First, is the number of repeat time K (which is similar to the number of training batches in DP-SGD) related to privacy cost? If so, we cannot assume it to be too large. In other words, the assumption stated in lines 237-238 is not hold. Second, the privacy cost also relies on the assumption of a full-rank covariance matrix of gradient proxies. If these assumptions do not hold, the privacy guarantees may be compromised.

2. DP-ULR is highly sensitive to the choice of parameters such as the noise standard deviation, batch size, and the rejection threshold. However, it is unclear how to select these parameters appropriately. This sensitivity can make it challenging to find the optimal settings for different datasets and models, potentially requiring extensive hyperparameter tuning. 

3. As mentioned in the first paragraph of page 2, the original ULR mothed (Jiang et al., 2023) incorporates noise addition into ULR algorithm. Therefore, it appears that the contribution of this work is primarily focuses on the analysis of DP properties. The authors should clearly outline their contributes, highlighting the differences from the original ULR algorithm.

### Questions
1. In line 198, should be description be “Add noise to the l-th module’s input” (i.e., “output”-> “input”) ?

2. Please also refer to Weaknesses section for more questions.

### Soundness
3

### Presentation
3

### Contribution
3
