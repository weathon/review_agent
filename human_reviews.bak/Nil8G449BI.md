# Block-local learning with probabilistic latent representations

- Decision: Reject
- Scores: 5, 5, 6, 6

## Abstract
The ubiquitous backpropagation algorithm requires sequential updates through the network introducing a locking problem. In addition, backpropagation relies on the transpose of forward weight matrices to compute updates, introducing a weight transport problem across the network. Locking and weight transport are problems because they prevent efficient parallelization and horizontal scaling of the training process. We propose a new method to address both these problems and scale up the training of large models. Our method works by dividing a deep neural network into blocks and introduces a feedback network that propagates the information from the targets backwards to provide auxiliary local losses. Forward and backward propagation can operate in parallel and with different sets of weights, addressing the problems of locking and  weight transport. Our approach derives from a statistical interpretation of training that treats output activations of network blocks as parameters of probability distributions. The resulting learning framework uses these parameters to evaluate the agreement between forward and backward information. Error backpropagation is then performed locally within each block, leading to "block-local" learning. Several previously proposed alternatives to error backpropagation emerge as special cases of our model. We present results on a variety of tasks and architectures, demonstrating state-of-the-art performance using block-local learning. These results provide a new principled framework for training networks in a distributed setting.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, a new training method is proposed for neural network architectures. Also, the authors provide a novel theoretical framework to analyze deep neural networks as parameters of probability distributions. Based on it, a new training method is proposed, with theory and numerical experiments on classification tasks.

### Strengths
The theoretical analysis to interpret DNN for probability distribution viewpoint is interesting. Also, the paper is well organized and well written to prove the new concept and method.

### Weaknesses
Please see the questions in the following part. There are some details not clear enough. Also, the numerical experiment results are not satisfying.

### Questions
Here are some questions for this paper:
Q: How to the principle or theory to select split blocks for each architecture? 
Q: How to deal with models with multiple branches? Is it possible to split the network for each branch? 
Q: What is the reason for the large performance gap on Cifar10 and ImageNet tasks, compare with BP method? Are there using same data preprocessing and augmentations? 
Q: What is the training time and convergence speed compared with BP method?

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
his paper presents two ideas.
The first is the bidirectional (forward and backward) propagation over a set of synaptic weights.
A part of the network propagates a signal in the forward direction.
The other part propagates the signal in the backward direction. 
The second is local blocking or breaking down of a deep network into smaller blocks. 
This promotes parallel processing.

### Strengths
This paper is well-organized.

### Weaknesses
1). The main contributions or claims of this paper seem minimal relative to existing work in this area.

Here are my reasons:

The bidirectional propagation overlaps with the work presented in the paper titled "Bidirectional Backpropagation".
It uses a set of synaptic weights for forward and backward propagation.
This generalizes to the case of using two separate networks.
The two separate network case is a special case that uses a deterministic dropout along the forward and backward propagation. 


They also present deep-neural blocking as a method for breaking down deep-neural networks into blocks of smaller networks.
They used the multiplication theorem for probability to factor the complete likelihood to a product of block likelihoods.
You can find this in their paper on "Bidirectional backpropagation for high-capacity blocking networks".
Equations (1) and (2) of their paper show the likelihood factorization.
This is similar to what you have listed as one of the main contributions.
You can also check equation (133) in the paper on "Noise can speed backpropagation learning and deep bidirectional pretraining".

Finally, their paper on "Bidirectional backpropagation for high-capacity blocking networks" combines bidirectional backpropagation with blocking.
This is a combination of bidirectional propagation and deep-neural blocking.

### Questions
Please can you identify the distinction(s) between your work and the prior work listed above on bidirectional backpropagation and neural blocking?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a probabilistic framework for block/layer-wise learning, such that a network encodes (parameters of) conditional distributions between latent variables that sequentially go from input to output.

### Strengths
1. As far as I know, the idea is novel and is reasonably backed by the variational methods literature. 
2. The idea works on simple tasks, and doesn't fail (although doesn't excel) on ImageNet, which is a good sign.

### Weaknesses
1. I found the paper rather hard to read. I think the main issue is that the network architecture and its specific computations are unusual, but they're never presented in one place. It takes from page 4 to page 7 to introduce the full architecture/losses. This is fine, since there are many non-trivial steps, but having them in one place would help a lot (perhaps in an extended Tab. 1, which by the way should be labelled as an algorithm). 

2. The following claim from the abstract: "demonstrating state-of-the-art performance using block-local learning" is not correct -- ~54% top1 accuracy on ImageNet from Tab. 2 with a ResNet50 is not very good. It's nice to see that the model doesn't fail, but it's not SoTA at all (even AlexNet reaches 56% top1 accuracy). 

3. It's also worth noting that block/layer-wise learning can perform as well as backprop, and with a similar idea of using small backward error networks. See Fig. 1 and overall results in [Belilovsky et al., 2019] (cited in this paper). It's not clear if the performance difference is due to inherent problems with the probabilistic interpretation in this paper or some other reasons.

### Questions
**Comments**:
1. First of all, the authors should use the ICLR style (currently the typeface is wrong and citations are not highlighted) and fit the paper into 9 pages. 
2. Top-3 for MNIST/CIFAR10 is not a standard metric. Moreover, given 10 classes and the simplicity of the datasets, anything but top-1 is mostly meaningless.
3. In S1.3, I think the first expectation should be just an integral over $z_k$. Expectation adds $p(z_k)$ which shouldn’t be there.
4. Citation issue: Jimenez Rezende et al. (2016) citation accidentally includes the author’s middle name.
5. Eqs. S12-13 and later: should there be brackets starting after the first sum?

**Clarification questions**

2. Eq. 7 should have $\beta_k$ instead of $q_k$, right?

3. This bit on page 6 is confusing:
> furthermore, the loss is local with respect to learning, i.e. it doesn’t require global signals to be communicated to each block. In this sense, our approach differs from previous contrastive methods that need to distinguish between positive and negative samples. In our approach, any sample that passes through a block can be used directly for weight updating and is treated in the same way

The paper was about supervised learning up until now, right?

**Conceptual questions**:

1. What’s the actual speed-up compared to backprop? With a naive implementation that doesn't account for unlocked updates, it should be as expensive. I guess it might be possible to put different blocks on different devices so the backward pass in earlier blocks will happen early on, but that wouldn’t work on a single GPU (right?) and also wouldn’t be easy to implement in PyTorch. How was it implemented by the authors?

2. Judging by Fig. 2, posterior bootstrapping introduces backward locking. Is that correct?




-----------
**Post-rebuttal**: since my questions were addressed, but the weaknesses like performance are still there, I'm increasing the score from 5 to 6. (Note to authors: sorry, I accidentally didn't change the rating when changing the review at the end of the discussion period. Corrected now.)

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a probabilistic interpretation of the input-output mapping defined by a neural network, together with a variational inference method used to optimize the log-likelihood.
The variational loss is an upper bound on the actual log-likelihood, and thus optimizing this variational loss to zero guarantees the optimization of the log-likelihood.
The variational loss can be decomposed into block local loss terms, making the optimization of this surrogate objective amenable to parallelization, removing both forward and backward locking of the back-propagation algorithm.
It also solves the weight transport problem as the feedback from labels, usually computed with back-propagation, is provided by a separate neural network.
The resulting algorithm shows promising performances on standard classification benchmarks, with accuracy being rather low with respect to vanilla gradient descent implemented with the back-propagation algorithm.
The proposed method being however well-posed and grounded in variational inference, where practical issues are well documented, it is likely that further improvements could be achieved by optimizing all the hyperparameters such as the design of the feedback network, the expressivity of the exponential family considered as an intermediate representation, or simply the optimization hyperparameters such as the learning rate or the type of optimizer used.

The paper allocates a long portion of the main body to the description of the proposed method and only shows a brief summary of the experiments performed, with experimental details and some additional experiments (i.e. ablation study) deferred in the appendix.

### Strengths
1) The probabilistic interpretation of input-output mapping allows the derivation of a variational inference method to tackle the optimization of the model weights. The reviewer is not aware of such formulation in the literature but is however not an expert in this particular field.
2) The proposed method is broadly applicable to most feedforward architectures used in deep learning applications.
3) The results on classification benchmarks are on par with other local learning methods without extensive hyperparameter tuning.

### Weaknesses
1) The proposed probabilistic interpretation of the input-output mapping appears new and is a bit hard to follow at first glance.
2) It is difficult to understand how the variational distribution $q$ is actually defined. Only backward messages $\beta_k = q(y | z_k)$ and it is unclear how the full posterior $q(z_k | x, y)$ is defined. My best guess is that it is implicitly defined through the bayes rule by combining the messages $\beta_k$ for all $1 \leq k \leq N$.
3) Some notations are only implicitly defined, such as $q_k$, or not defined at all such as $a_{kj}$ and $z_{kj}$.
4) It is difficult to understand in which case the variational posterior could recover the true posterior. In other words, it is unclear which messages $\beta_k$ would lead to a perfect reconstruction of the true posterior, as well as how to compute them.
5) There are no guidelines on how to design the different components of the proposed algorithm. For example, there is no ablation study on the expressivity of the EF distribution used as a latent representation, nor on the expressivity of the feedback network.
6) [Minor comment] Equation S2 of Section S1.2 in the supplement material defines an upper bound of the log-likelihood and refers to equation (1) of the main text. However, equation (1) of the main text refers to the gradient of the log-likelihood.

### Questions
1) The distribution $q$ is only defined implicitly through the definition of backward messages $\beta_k = q(y | z_k)$. Could the author either confirm that it is only implicitly defined because we only need to compute messages $\beta_k = q(y | z_k)$ or specify the definition of the full posterior $q(z_k| x, y)$?
2) In equation (5), I don’t understand what the index $j$ stands for. Are $\alpha_{kj}$ and $\beta_{kj}$ parameters of the distribution $\alpha_k$ and $\beta_k$ or actual distributions in their own right?
3) Does equation (6) make use of the assumption that $p_k$ and $\alpha_k$ are Gaussian or from a distribution in the exponential family?
4) Could the author clarify the approximation gap between the true log-likelihood and their variational surrogate? In particular, when does the variational loss recover the true log-likelihood?
5) Could the authors give some intuition on the design of the latent representation and the expressivity needed in the feedback network?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good
