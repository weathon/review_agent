# Review

## Summary
The authors propose a method to infer the dynamics of a physical system using a Hamiltonian Neural Network (HNN). They claim that the proposed method preserves the geometric structure governing the system's dynamics, leading to improved performance compared to traditional approaches like PINNs. The method is tested on a 2D heat transfer system, showing superior results in predicting the system's behavior compared to baseline methods.

## Soundness
2

## Presentation
2

## Contribution
1

## Strengths
The paper is well-structured, with a clear presentation of the proposed method and the experimental setup. The authors provide a detailed explanation of their approach, including the use of Riemannian optimization for learning the system matrix. The experiments are thoroughly described, and the results are presented in a clear and concise manner.

## Weaknesses
1. The novelty of the proposed method is limited. The use of Hamiltonian Neural Networks (HNNs) for modeling physical systems is not new. The paper does not clearly differentiate the proposed method from existing HNN-based approaches, such as [1] and [2].
2. The experimental evaluation is insufficient. The authors compare their method only against traditional time series models like LSTM and XGBoost, without considering any of the numerous existing methods for modeling physical systems using neural networks. A comparison with state-of-the-art methods like Physics-Informed Neural Networks (PINNs) [3], Neural ODEs [4], and Hamiltonian Neural Networks [1] is necessary to validate the effectiveness of the proposed approach.
3. The authors claim that their method preserves the geometric structure of the system's dynamics, but they do not provide any theoretical analysis to support this claim. A more rigorous theoretical treatment of how the proposed method preserves geometric structure is needed to substantiate this claim.
4. The paper does not discuss the limitations of the proposed method. It is unclear under what conditions the method might fail or perform suboptimally. A discussion of the method's limitations would provide a more balanced view of its applicability.

[1] Greydanus, S., Dzamba, M., & Yosinski, J. (2019). Hamiltonian neural networks. Advances in neural information processing systems, 32.

[2] David, M., & Méhats, F. (2023). Symplectic learning for hamiltonian neural networks. Journal of Computational Physics, 494, 112495.

[3] Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of computational physics, 378, 686-707.

[4] Chen, R. T., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2019). Neural ordinary differential equations. Advances in neural information processing systems, 32.

## Questions
1. How does the proposed method differ from existing HNN-based approaches? What are the advantages of the proposed method over these existing approaches?
2. Why were state-of-the-art methods like PINNs, Neural ODEs, and Hamiltonian Neural Networks not included in the experimental comparison? How does the proposed method compare to these methods?
3. Can you provide a more rigorous theoretical analysis of how the proposed method preserves the geometric structure of the system's dynamics?
4. What are the limitations of the proposed method? Under what conditions might it fail or perform suboptimally?

## Flag For Ethics Review
No ethics review needed.

## Details Of Ethics Concerns


## Rating
3

## Confidence
5