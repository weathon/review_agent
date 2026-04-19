# Leveraging Low Rank Structure in The Lazy Regime

- Decision: Reject
- Scores: 3, 5, 3, 3

## Abstract
Understanding the training dynamics of neural networks has gained much interest in the scientific community. The dynamics of training over-parameterized models is characterized by the lazy regime in which networks exhibit near-linear behavior and minimal parameter changes. In addition, it has been argued that the Jacobian of large neural models has a low-rank structure. In this paper, we focus on the opportunities laid out by the combination of low-rankness and laziness of large neural models. Specifically, we provide a scalable way to measure the extent of laziness, evaluated via the rate of change of the model Jacobian, as well as a scalable method to verify low-rankness of the model Jacobian without storing the entire Jacobian. Taking advantages of both laziness and low-rankness, we design a scalable training algorithm for over-parameterized models that performs backpropagation-free gradient descend training. In particular, this algorithm is of lower computation and storage requirements in cases of massive parameter sharing, as is the case of many state-of-the-art neural architectures. Empirical results confirm the scalability and effectiveness of our approach, opening new pathways for exploring novel learning strategies in neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The authors propose a backpropagation-free training algorithm that relies on the lazyness of the network and the low-rankness of its jacobian. More precisely: thanks to the lazyness assumption, the jacobian $J_t$ does not evolve in time, making it possible to store the one at initialization. Given its size, by leveraging the low-rank assumption a low-rank decomposition of it is stored instead. The authors moreover proposed a scalable way to test the lazyness/low-rankness hypothesis before training.

### Strengths
The work is very well presented and the proposed approach is an interesting alternative to backpropagation when the authors' assumption hold.

### Weaknesses
The authors' claims are not supported by an adequate experimental section. In particular, given that the paper proposes a novel training pipeline, I believe it is necessary for the authors to test it and show results on it.

### Questions
1) I have serious concerns about the low-rankness assumption. While it may hold at initialization, how can I quantify of how much the "effective rank" would change during classical training (even if close to the lazy regime)? In the case of it increasing, then it is possible that this training procedure may actually underperform with respect to training with backpropagation. For this reason, I believe it is crucial for the authors to show results comparing their training procedure with a classical backpropagation-based one (both to show scalability and the difference in performance).
2) It is not clear to me how to select $\mathcal R$, and how much its size would impact the final training results;
3) Why not update the Jacobian also after a fixed number of iterations? I guess this would introduce a tradeoff between performance and cost, but with no experiment it is difficult to say.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors propose a memory efficient and backprop-free training method for neural networks. The approach considers the jacobian of very wide nets, computed in a preprossesing step and subsequently low-rank factorized. 

Based on the assumption of lazy ntk regimes, the authors use the factorized precomputed jacobian to train the network without backprop.

### Strengths
- The authors take their time to review theory on ntk, especially useful for readers unfamiliar with the subject. 
- A relatively efficient method to compute the low-rank gradients is proposed, without naive "full-size" SVD.

### Weaknesses
- The main weakness of this paper are the strong assumptions upon which the method is build. Lazy Learning and NTK regime are very specific scenarios, and it is unclear in which scenarios the proposed method actually works
- In this regard, the numerical evaluation is lacking. The authors demonstrate the method only on fully-connected layers of varying widths, and only on the MNIST dataset. The achieved accuracies are not excatly SOTA. The methods purpose is unlocking resource efficient learning (assumably on large models and datasets), however the experimental setup does not reflect this. At least a transformer should be considered, or a well-used computer vision model on typical NLP and Vision benchmarks to get a grasp of the expected performance.

### Questions
- Is the Jacobian the combined Jacobian of all network parameters, or is the notation to be viewed seperately for all layers? 
- Do you have an method to verify cheaply during training whether the assumptions for using the fixed low-rank jacobian still hold? Can you re-compute the Jacobian if neccessary with constrained cost?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes to leverage the linear dynamics and the low-rank structure of the Jacobian (of the output of a model w.r.t. the parameters) to build a training algorithm that is computationally cheap and backpropagation-free. This algorithm is meant to work when training neural networks (NN) in the so-called "lazy regime".

### Strengths
# Clarity

The paper is easy to read.

### Weaknesses
# Quality

## About the lazy-training regime

According to [1] (cited in the paper), lazy training is very likely a phenomenon that we want to avoid, since it is associated to poor generalization. So, it is not clear at all that it is useful to leverage this phenomenon to increase training speed.

## Linear dynamics

The whole algorithm, and specifically its property of being "backpropagation-free", is based on the fact that the training dynamics of a NN is basically linear. Before attempting a low-rank approximation of the Jacobian, it is necessary to check if the algorithm defined in Eqn (5) works without approximation (it can be done on small NNs trained on small datasets). Even there, it would be interesting to know if it is possible to leverage the linear dynamics to reduce training cost.

## Training experiments

In the abstract, the authors claim to propose a training algorithm. Thus, any reader would expect to find a series of experiments, testing various architectures of NNs on various datasets (even small ones). Actually, I would be very surprising and worth publishing if the authors show that their learning rule (almost linear dynamics of training) is sufficient to train NNs, even small ones on easy tasks.

But, without such experimental evidence, it is intuitively very hard to believe that the learning rule defined in Eqn (5) would be usable tob train a NN from start to end. Moreover, Alg. 1 does not mention any update of J.

# References

[1] *On lazy training in differentiable programming*, Chizat et al., 2019.

### Questions
How does perform the proposed algorithm without low-rank approximation?

How does it perform in various tasks? (show training curves and comparisons with SGD, etc.)

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper studies over-parameterized neural networks in the lazy regime. The main contributions are listed below: 

**(1)** A *scalable* method for estimating the Jacobian’s rank is proposed.

**(2)** An *scalable* approach to quantify network laziness is proposed.

**(3)** A backpropagation-free learning algorithm that leverages low-rank structures for improved scalability and efficiency.

### Strengths
The paper explores combining NTK theory with low-rank approximations to enhance training efficiency in neural networks potentially.

### Weaknesses
**Limited novelty of Algorithm 1:** The algorithm mainly applies dimensionality reduction and exploits a constant Jacobian assumption during training, combining established ideas without introducing significant innovations. It reuses existing techniques rather than offering a novel approach to efficient training.

**Complexity and Practical Implementation Concerns:** The algorithm’s eigenvector computation steps may still incur high computational costs, especially in large networks. Its efficiency on high-dimensional tasks is  unclear to me.

**Limited empirical evidence:** Given the paper's empirical focus, the experiments are insufficient; authors only test on MNIST without exploring complex, large-scale tasks that would better validate the method’s scalability and real-world effectiveness.

**Limited Experimental Setup Details:** Key details on dataset preprocessing, model configurations, and training conditions are missing.

**Unclear general applicability:** The paper claims applicability to architectures like CNNs and transformers, but lacks empirical support. No experiments on complex, parameter-sharing models are provided, making generalization to these architectures uncertain. Moreover, I do not think laziness assumption holds for more complex architectures.

**Writing:** The paper is unclear, with poor notation and room for significant improvement in clarity and precision. There are also typos, which further affect readability. (See Ln 375 & 334 for instance)

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
2
