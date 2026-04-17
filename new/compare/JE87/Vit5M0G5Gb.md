# Review

## Summary
This paper studies the simplicity bias of neural networks trained with gradient descent. Simplicity bias refers to the phenomenon that the complexity of the model increases over the course of training. The authors present a theoretical framework that explains this simplicity bias in terms of saddle-to-saddle dynamics for a general class of neural networks, including fully-connected, convolutional, and attention-based architectures. They show that linear networks learn solutions of increasing rank, ReLU networks learn solutions with an increasing number of kinks, convolutional networks learn solutions with an increasing number of convolutional kernels, and self-attention models learn solutions with an increasing number of attention heads. The authors analyze fixed points, invariant manifolds, and the dynamics of gradient descent learning, showing that saddle-to-saddle dynamics operates by iteratively evolving near an invariant manifold, approaching a saddle, and switching to another invariant manifold. They disentangle data-induced and initialization-induced saddle-to-saddle dynamics, finding that the former leads to low-rank weights while the latter to sparse weights. The theory predicts the effects of data distribution and weight initialization on the duration and number of plateaus in learning. Overall, the paper offers a framework for understanding when and why gradient descent progressively learns increasingly complex solutions.

## Soundness
3

## Presentation
3

## Contribution
3

## Strengths
The paper is well written and easy to follow. The authors present a novel theoretical framework that explains the simplicity bias of neural networks trained with gradient descent. The simplicity bias is a well-known phenomenon that has been observed across different architectures, tasks, and training paradigms. However, there is a lack of a unifying theory that can explain this phenomenon. The authors fill this gap by providing a theoretical framework that can explain the simplicity bias for a wide range of architectures, including linear networks, ReLU networks, convolutional networks, quadratic networks, and linear self-attention models. This framework offers a principled link between stages and simplicity, showing that earlier stages in learning are simpler. The authors also define an operative notion of simplicity based on the number of effective units in the architecture, such as hidden neurons, convolutional kernels, or attention heads. The framework shows that fixed points in the loss landscape are recursively embedded, with fixed points of smaller networks embedded in saddle points of larger networks. Saddle points are connected by invariant manifolds along which a larger network behaves like a smaller one, preserving simplicity along the connecting trajectories. The link between saddle-to-saddle dynamics and simplicity arises from the interplay of the saddle hierarchy and timescale separation. The authors disentangle data-induced and initialization-induced timescale separation, showing that the former leads to low-rank weights while the latter leads to sparse weights. Overall, the theory provides a unified perspective on the simplicity bias across different architectures and predicts when non-stage-like behavior will arise.

## Weaknesses
The authors focus on one layer in the network and assume that the activation function is homogeneous or linear. These assumptions simplify the analysis but may not fully capture the complexity of real-world neural networks. The authors consider the gradient flow dynamics of the networks, which is a simplification of the more complex stochastic gradient descent algorithm used in practice. The analysis is based on the assumption of small learning rate, which may not hold in some cases. The authors consider the two-layer case and extend the results to deep networks, but the analysis of deep networks is limited. The authors assume that the output is a scalar, which is not always the case in real-world applications. The authors provide a definition of simplicity in terms of the number of effective units, but this definition may be too simplistic and may not capture other aspects of complexity, such as the smoothness or regularity of the learned functions. The authors focus on the simplicity bias in terms of the number of effective units, but there may be other biases, such as the bias towards learning low-rank or sparse solutions, that are not fully explored in this work.

## Questions
How does the proposed framework compare to other existing theories that attempt to explain the simplicity bias? Are there any limitations or assumptions in the proposed framework that are not present in other theories?

The authors consider the gradient flow dynamics, which is a simplification of the more complex stochastic gradient descent algorithm used in practice. How well do the results presented in this paper translate to the more realistic setting of stochastic gradient descent?

The authors consider the two-layer case and extend the results to deep networks, but the analysis of deep networks is limited. How well do the results presented in this paper apply to deep networks in practice?

The authors assume that the output is a scalar, which is not always the case in real-world applications. How well do the results presented in this paper apply to cases where the output is not a scalar?

The authors provide a definition of simplicity in terms of the number of effective units, but this definition may be too simplistic and may not capture other aspects of complexity, such as the smoothness or regularity of the learned functions. How do the authors think the definition of simplicity should be extended to capture these other aspects of complexity?

The authors focus on the simplicity bias in terms of the number of effective units, but there may be other biases, such as the bias towards learning low-rank or sparse solutions, that are not fully explored in this work. How do the authors think the proposed framework should be extended to include these other biases?

The authors assume that the activation function is homogeneous or linear, which simplifies the analysis but may not fully capture the complexity of real-world neural networks. How do the authors think the proposed framework should be extended to handle more general activation functions?

The authors consider the saddle-to-saddle dynamics, but there may be other types of dynamics that are not fully explored in this work. How do the authors think the proposed framework should be extended to include other types of dynamics?

The authors focus on one layer in the network, but in practice, neural networks consist of multiple layers. How do the authors think the proposed framework should be extended to handle multiple layers?

The authors assume that the learning rate is small, which may not hold in some cases. How do the authors think the proposed framework should be extended to handle larger learning rates?

The authors consider the gradient flow dynamics, which is a simplification of the more complex stochastic gradient descent algorithm used in practice. How do the authors think the proposed framework should be extended to include the effects of batch normalization and other normalization techniques?

The authors consider the two-layer case and extend the results to deep networks, but the analysis of deep networks is limited. How do the authors think the proposed framework should be extended to handle deep networks in practice?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
8

## Confidence
4