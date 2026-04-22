# Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss

- Avg Score: 6.67
- Decision: Accept (Oral)
- Scores: 6, 8, 6

## Abstract
Mixture-of-Experts (MoE) models lack explicit constraints to ensure the router's decisions align well with the experts' capabilities, which ultimately limits model performance. To address this, we propose expert-router coupling (ERC) loss, a lightweight auxiliary loss that tightly couples the router's decisions with expert capabilities. Our approach treats each expert's router embedding as a proxy token for the tokens assigned to that expert, and feeds perturbed router embeddings through the experts to obtain intermediate activations. The ERC loss enforces two constraints on these activations: (1) Each expert must exhibit higher activation for its own proxy token than for the proxy tokens of any other expert. (2) Each proxy token must elicit stronger activation from its corresponding expert than from any other expert. These constraints jointly ensure that each router embedding faithfully represents its corresponding expert's capability, while each expert specializes in processing the tokens actually routed to it. The ERC loss is computationally efficient, operating only on $n^2$ activations, where $n$ is the number of experts. This represents a fixed cost independent of batch size, unlike prior coupling methods that scale with the number of tokens (often millions per batch). Through pre-training MoE-LLMs ranging from 3B to 15B parameters and extensive analysis on trillions of tokens, we demonstrate the effectiveness of the ERC loss. Moreover, the ERC loss offers flexible control and quantitative tracking of expert specialization levels during training, providing valuable insights into MoEs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes expert-router coupling loss, a lightweight auxiliary loss that couples expert capabilities (activation norm) and the router’s decisions. The authors claim that this loss encourages each expert and each proxy token to match with each other, improving performance.

### Strengths
1. The ERC loss is computationally cheap.
2. The experiments show MoE gets considerable gain from this ERC loss.
3. Much analysis and ablation are provided.

### Weaknesses
1. You might need to compare with Router Orthogonalization Loss in https://yiyan.baidu.com/blog/publication/ERNIE_Technical_Report.pdf, since your loss is somewhat similar to ||(RW_g)TRWg - I||_F, if you assume W_g^TW_g\approx I, it is similar to ||R^TR - I||_F.
2. It seems that this ERC loss can be optimized to 0 when RMS(R) -> 0 or RMS(W_g) -> 0, so will it only serve like weight decay?

### Questions
1. Are both R and W_g optimized by ERC loss (rather than R only)?
2. Can you directly calculate the expectation of ERC loss under \delta and optimize the expectation directly?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, to mitigate performance degradation caused by the representation mismatch between the router and experts in MoE architectures, each row of the router matrix is treated as a representative vector for the vectors processed by an expert. A constraint is introduced in the form of a loss function, ensuring that when values in the vicinity of this representative are input to the corresponding expert, the activation is maximized. This constraint helps ensure that the expert selected by the router based on the actual input is the one most efficiently activated by that input. Experimental results show that the proposed method consistently outperforms the standard MoE setup, with few exceptions.

### Strengths
Many recent MoE papers focus on improving routing. Methods exist to resolve the mismatch between routing and experts, such as adding an auxiliary loss to teach desired properties in expert specialization, or modifying the model architecture, like AoE (which is also adopted as a baseline in this paper). The proposed method belongs to the former category; it achieves its goals regarding expert specialization by simply adding a simple constraint, without modifying the conventional model architecture at all. The proposed method can leverage existing software assets as-is, for example, by being integrated into existing training toolkits. The proposed method is an extremely lightweight loss function, and it is believed to have no or very small practical impact on training speed when introduced into the MoE training process. Personally, I am impressed that the authors conceived of this method, and I would like to try it in our own training framework.

It is noteworthy that the proposed method enables the router and experts to have an explicit geometric correspondence, succeeding in achieving a similar effect to AoE without requiring significant architectural modifications. It may also facilitate the visual analysis of the model's internals.

### Weaknesses
The experiments only validate the method on a single, very small-scale model instance. It has not been demonstrated whether the method is effective across the wide variety of MoE architectures. Since the experiments involve expensive pre-training, it is understandable that validating on various settings must be forgone due to cost, but it is true that the information provided feels somewhat insufficient.

The method includes randomness, which may be a source of training instability, although as shown in 4.4 (2), the fact that this randomness contributes to generalization appears to be valid.

A new hyperparameter, $\alpha$, which is difficult to tune intuitively, is introduced. Given the current lack of experimentation across a wide range of model instances, applying the settings used in the paper directly to other experiments is considered to carry a certain amount of risk.

### Questions
Regarding the randomness: could a method be devised to make the training behavior more theoretically consistent and predictable? For example, could a derivative algorithm be considered, such as marginalizing $R[i] \odot \delta_i$ over $\delta_i$, or handling the noise as a distribution (without sampling)?

The proposed method can be seen as a form of contrastive learning between the router and expert features, and therefore it seems relatively natural to consider leveraging techniques from contrastive learning. Are there any thoughts on this at this time?

### Soundness
3

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
2

### Summary
This paper proposes a new auxiliary loss for MoE training which better couples the experts and the routers and promotes expert specialization. The idea is to think of the router rows as cluster centers and to generate cluster data from the centers via random perturbations. The data from a given cluster should induce higher activation norm in the corresponding expert relative to the other experts and to data from other clusters. This is enforced by a soft hinge penalty. Adding the auxiliary loss adds modest overhead during training and does not affect inference. The authors find that the penalty improves downstream metrics over a baseline vanilla MoE and is comparable to the more expensive AoE method.

### Strengths
The method is simple, intuitive, and clearly presented. The experiments and evaluations are thorough and the auxiliary loss does appear to improve performance.

### Weaknesses
The activation metric is not scale invariant, the auxiliary loss can be decreased in a non-meaningful manner simply by scaling up $W_g^i$.

The auxiliary appears to make the gradient dense across experts since activations norms for each token are computed across experts.

### Questions
In addition to clarification of the weaknesses, I have the following questions:

Is it possible that $\alpha > 1$ can perform even better? At what $\alpha$ will you recover vanilla MoE?

What about using post SwiGLU activations or $W_o$?

Do the norms $\lVert R[i] \rVert$ stay comparable across $i$? If this is not true, then there seems to be a mismatch between Euclidean distance based clustering and inner-product based routing.

### Soundness
4

### Presentation
4

### Contribution
3
