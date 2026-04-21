# LoKO: Low-Rank Kalman Optimizer for Online Fine-Tuning of Large Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 5, 6, 6, 3

## Abstract
Training large models with millions or even billions of parameters from scratch incurs substantial computational costs. Parameter Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), address this challenge by adapting only a reduced number of parameters to specific tasks with gradient-based optimizers. In this paper, we cast PEFT as an optimal filtering/state estimation problem and present Low-Rank Kalman Optimizer (LoKO) to estimate the optimal trainable parameters in an online manner. We leverage the low-rank decomposition in LoRA to significantly reduce matrix sizes in Kalman iterations and further capitalize on a diagonal approximation of the covariance matrix to effectively decrease computational complexity from quadratic to linear in the number of trainable parameters. Moreover, we discovered that the initialization of the covariance matrix within the Kalman algorithm and the accurate estimation of the observation noise covariance are the keys in this formulation, and we propose robust approaches that work well across a vast range of well-established computer vision and language models. Our results show that LoKO converges with fewer iterations and yields better performance models compared to commonly used optimizers with LoRA in both image classifications and language tasks. 
Our study opens up the possibility of leveraging the Kalman filter as an effective optimizer for the online fine-tuning of large models.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes an innovative Kalman filter-based low-rank optimization algorithm (referred to as LoKO) aimed at online fine-tuning of large-scale models.

The algorithm initially employs the LoRA algorithm to effectively reduce the number of trainable parameters through low-rank decomposition.
Subsequently, it adopts a diagonal approximation of the covariance matrix P and integrates Exponential Moving Average (EMA) techniques to estimate the covariance matrix R of the observation noise.

The implementation of these strategies significantly lowers the computational complexity of LoKO. Moreover, experimental results across
multiple computer vision and language models demonstrate that its performance is on par with existing advanced methods.

### Strengths
This paper innovatively proposes using the Kalman Filter algorithm combined with LoRA for online fine-tuning of large-scale models, and
reduces the computational complexity of fine-tuning through approximate solution methods.

The experiments also demonstrate that the proposed method achieves performance comparable to existing gradient descent-based algorithms across multiple models and datasets.

Finally, the paper investigates the sensitivity of the final performance to the initialization range of hyperparameters through experiments, thereby proving the robustness of the proposed method.

### Weaknesses
The LoKO method assumes that the model's trainable parameters remain unchanged during the observation process, meaning that the generation process of the observed variables remains consistent throughout the entire observation period. This assumption is often difficult to satisfy in online scenarios and results in the loss of the model's dynamic adaptation capability.

Although the authors observed through experiments that the matrix tends to become diagonal during training, they did not provide a
corresponding analysis. 

In Table 3, the LoKO method consistently underperforms compared to the baseline methods LoRA/AdamW, but the authors did not provide any related analysis.

One significant contribution of this paper is the proposal of a method for online fine-tuning of large-scale models. However, the largest model tested in the experiments, RoBERTa-large, has only 335M parameters. Therefore, further testing on even larger models is needed to validate the effectiveness of their method.

Finally, the paper lacks a comparison of the runtime efficiency between LoKO and the baseline methods.

### Questions
In Table 1, the authors provide the tested models and their corresponding LoRA layers along with the trainable parameters. How do the remaining non-trainable parameters be obtained?

In Table 4, the authors provide the diagonal initialization bounds for parameter P for different datasets and models. How are these bounds determined? Is there a more general selection criterion?

### Soundness
3

### Presentation
3

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
Previous works show that the recursive Extended Kalman Filter (EKF) algorithm can optimize relatively small models with performance surpassing its gradient-based counterparts. However, it is challenging to optimize neural networks with EKF because the size of the crucial covariance matrix in EKF can grow quadratically with the number of model parameters in training. In this paper, the core idea is to have EKF work with the LORA so that only a few parameters are involved in the EKF optimization. The proposed approach LoKO shows promising results on computer vision and language modeling benchmarks, such as MNIST, CIFAR 10/100, ImageNet 100, SST-2, COLA, MRPC.

### Strengths
LoKO seems to be an interesting approach that successfully combine the EKF and the LORA, this provides a good new perspective to approach the online gradient-descent method.

This paper provides a well structured experiments across computer vision and NLP. The results in general are not state-of-the-art, but they are decent to show the promising results of the LoKO.

The proof to show that the proposed approach equation is equivalent to that in the vanilla EKF algorithm seems to be sound.

The paper is well structured and well written. The logical flow from problem definition through methodology to empirical validation helps readers grasp the paper’s objectives and contributions effectively.

The proposed LoKO seems to provide high accuracy with lower computational footprint. This seems to be interesting to some practical applications where the computational efficiency is important.

### Weaknesses
The evaluation is limited. This paper has done quite a few evaluation on the proposed LoKO method. However, i find none of those datasets are well suited for the LoKO. In general, the model full fine-tuning can do better than that from the LoRA fine-tuning. The LoKO would be suboptimal compared to the model full-fining as long as one can afford the GPU computes. The proposed experiments do not cover the dynamic streaming type of data, which would be able to show the strength of the proposed LoKO.

The proposed LoKO seems to suggest to require one to initialize the covariance matrix, and noise estimation parameters. However, it is unclear how sensitive of those parameters to the final model performance. There are no studies to reveal the sensitive of the proposed approach.

There are multiple LoRA and its variants for PEFT. The experiments do not cover the comparison between the proposed one against the other PEFT methods such as QLoRA, QA-LoRA, LoftQ. It would be interesting and useful to see how the proposed LoKO compared against the other LoRA and its variants.

The experiments do not cover the study on the memory efficiency and what's the limitations of the proposed LoKO. It is important to understand the limits and boundary of the LoKO.

### Questions
1. What's the limitation of the proposed approach? It would be great to have more throughly memory usage and efficiency analysis and experiments, given the computational efficiency is one of the advantages of the proposed approach. 

2. How to initialize the LoKO? how sensitive of the model is? It would be great to have more studies on those.

3. How does the proposed LoKO compare against the standard LoRA and other LoRA variants such as QLoRA?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces LoKO, a novel optimization algorithm that integrates the Extended Kalman Filter (EKF) with Low-Rank Adaptation (LoRA) for the online fine-tuning of large pre-trained models. By reframing parameter-efficient fine-tuning (PEFT) as a state estimation problem, the authors leverage the EKF to estimate optimal trainable parameters in an online manner. The use of LoRA's low-rank decomposition reduces the number of trainable parameters, and a diagonal approximation of the covariance matrix further decreases computational complexity from quadratic to linear. An Exponential Moving Average (EMA) approach is proposed for estimating the observation noise covariance without additional computational cost. Empirical evaluations on various computer vision and natural language processing tasks demonstrate that LoKO achieves competitive or superior performance compared to traditional gradient-based optimizers like AdamW and AdaGrad.

### Strengths
Casting PEFT as a state estimation problem using the Kalman filter is a novel and insightful approach. The combination of LoRA with EKF addresses the scalability issues traditionally associated with Kalman filters in large-scale models. The diagonal approximation of the covariance matrix and EMA-based estimation significantly reduce computational complexity, making the approach practical for large models.

The paper provides a clear and well-founded mathematical formulation of the proposed method. The derivation of the simplified Kalman update equations is thorough, and practical implementation considerations are discussed. The thoughtful discussion on covariance matrix initialization and its impact on performance demonstrates a deep understanding of the method's nuances.

The method is evaluated across multiple domains, including image and text classification tasks, demonstrating its versatility. Thorough comparisons with mainstream optimizers (AdamW and AdaGrad) show that LoKO often achieves better convergence and performance. Detailed analyses on the effects of covariance initialization and the forgetting factor provide valuable insights into the method's behavior.

### Weaknesses
The paper lacks a formal convergence analysis of the proposed algorithm. Without theoretical guarantees, it's unclear under what conditions LoKO is expected to perform reliably.
The justification for the diagonal approximation of the covariance matrix is primarily empirical. A theoretical analysis or conditions under which this approximation holds would strengthen the contribution.
The theoretical implications of using EMA for observation noise covariance estimation are not fully explored. Understanding its impact on convergence and stability is important.
The experiments do not include comparisons with other recent PEFT methods such as QLoRA or AdaLoRA. Including these would better position LoKO within the current landscape. Evaluations are conducted over a single epoch, which may not capture the long-term behavior and stability of the optimizer. The focus on online fine-tuning with batch size 1 raises questions about the method's applicability and performance in standard offline settings with larger batch sizes. In some language tasks, LoKO performs slightly worse than AdamW. Investigating these cases could reveal insights into the method's limitations.

While the paper claims reduced computational complexity, it does not provide empirical runtime measurements or resource usage comparisons to substantiate these claims. There is limited discussion on potential trade-offs between computational cost and optimization performance compared to other methods.

### Questions
Please address the comments in the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
Low-Rank Kalman Optimizer (LoKO) is a novel optimizer for Parameter-Efficient Fine-Tuning (PEFT) aimed specifically at LoRA. LoKO frames fine-tuning as an optimal filtering problem, where trainable parameters are estimated as state variables updated dynamically with new observations. By leveraging Kalman’s recursive estimation, LoKO achieves efficient convergence with fewer updates. Additionally, the optimizer incorporates a diagonal approximation for the covariance matrix, which reduces computational complexity from quadratic to linear in the number of parameters.

### Strengths
LoKO shows that Kalman optimizers work similarly well with LoRA as they do with bigger weight matrices in a neural network with better and faster convergence and efficient online learning.

### Weaknesses
"Novelty" The novelty of the work seems hard to gauge and should be explicitly noted in the paper. The use of EKF as an alternative optimizing strategy has already been pursued and its efficacy for faster convergence has been demonstrated in previous works some of which are cited in the paper. The diagonal approximation (amongst other approximation attempts as in the cited, Chang et al.) for the covariance matrix to reduce computations has also been utilized in a number of previous works regarding Kalman filtering. The novelty in this work seems to be in the application to the finetuning of neural networks using a specific low rank adapter, LoRA. There are no new findings or theoretical insights proposed other than what is expected of training a subset of neural weights (LoRA) with a Kalman optimizer. Unfortunately, this does not seem to be adequate especially given the limited diversity of experiments (explained below) and applicability to a single type of low rank adapter.

The gaussian assumption for the neural parameters and noise are unsupported by any empirical results in the paper.

"Diagonal Approximation" LoKO relies on a diagonal approximation of the covariance matrix to reduce computational costs. While effective, this approximation limits the optimizer's ability to capture parameter correlations fully, which could lead to suboptimal updates for tasks with high-dimensional dependencies​. Experiments on complex tasks like text-to-image generation or others should be utilized to ablate this aspect of the method.

In addition, the sensitivity to initialization of the covariance matrix requires significant ablation across different architectures and tasks to investigate its failings and properties, and whether it varies across different modalities.

"Noise Approximation" The exponential moving average (EMA) approach for noise covariance estimation may be unstable, particularly when new data distributions diverge significantly from the initial training data. It is necessary to test LoKO in OOD generalization tasks to ablate this approximation strategy.

Given this work relates to optimization specific to PeFT methods, its necessary to show its efficacy on different low rank adapters to show similar results and properties can be observed from them. Comparisons with DoRA, X-LoRA, etc. would be great additions to the work.

### Questions
- What is the novelty of the work beyond application to LoRA?
- Is this applicable beyond LoRA to other methods?
- Does the diagonal approximation cause problems in harder tasks like text-to-image generation?
- Is OOD generalization adversely affected due to the noise approximation?

### Soundness
3

### Presentation
2

### Contribution
2
