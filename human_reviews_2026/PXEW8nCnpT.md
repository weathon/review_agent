# Projected Neural Additive Models as Universal Approximators

- Decision: Reject
- Scores: 4, 2, 4, 4

## Abstract
This article proves that any continuous multi-variable function can be approximated arbitrarily close by a linear combination of single-variable functions of the inputs in a projected space. Using a set of independent neural networks to parameterize these feature functions of the projected inputs, we introduce their linear combination as the projected neural additive model (PNAM): an extension of the neural additive model (NAM) (cf. Agarwal et al. (2021)) that now enables universal approximation. While the couplings of the input variables bestow the PNAM with the universal approximation property, they could diminish the interpretability intrinsic to the NAM. As such, we propose regularization and post hoc techniques to promote sparse solutions and enhance the interpretability of the PNAM. The single-variable characteristic of the bases also allows us to convert them into symbolic equations and dramatically reduces the number of required parameters. We provide results from numerical experiments on invariants in knot theory, phase field fracture mechanics, and the MNIST benchmark to illustrate the expressivity and interpretability of the PNAM.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Projected Neural Additive Models (PNAM), an extension of Neural Additive Models (NAM) that achieves universal approximation capability by incorporating a linear transformation of inputs before processing them through independent single-variable neural networks. The authors establish the theoretical foundation for PNAM's universal approximation property using the Stone-Weierstrass theorem and propose regularization and post-processing techniques to enhance interpretability. Through experiments on mathematical knot invariants and phase field simulations, they demonstrate PNAM's competitive performance against MLPs and NAMs while maintaining better interpretability. The work is significant as it provides a balance between expressivity and interpretability, particularly beneficial for scientific domains where understanding model behavior is crucial.

### Strengths
The paper presents several notable strengths. First, it provides a rigorous theoretical foundation by formally proving PNAM's universal approximation property using the Stone-Weierstrass theorem, addressing a key limitation of the original NAM. Second, the proposed architecture elegantly combines the interpretability of additive models with enhanced expressivity through input projection, representing a meaningful advancement in interpretable neural network design. Third, the comprehensive regularization framework—including weight decay, function value constraints, input coupling penalties, and sparsity promotion—effectively addresses the trade-off between model complexity and interpretability. Fourth, the post-processing techniques for feature importance ranking, parameter pruning, and symbolic regression conversion offer valuable tools for enhancing model transparency. Finally, the experimental evaluation on two distinct domains (knot theory and phase field fracture) demonstrates the model's versatility and provides convincing evidence of its competitive performance against relevant baselines.

### Weaknesses
1) While the paper presents a compelling approach, there are several aspects that warrant further consideration. First, the theoretical foundation primarily focuses on the universal approximation property, but lacks analysis of the optimization process. Specifically, there is no discussion of convergence guarantees for the Adam optimizer when applied to PNAM's architecture, nor is there an analysis of the convexity properties of the loss function with the proposed regularization terms. Additionally, the impact of the projection dimension M on the optimization landscape and convergence speed remains unexplored.

2) The experimental setup could benefit from more comprehensive details. The selection of regularization weights (w1-w5) is not adequately justified, as they are simply set to fixed values without sensitivity analysis. Similarly, the criteria for choosing key hyperparameters such as the projection dimension M and network architecture are not well explained. Moreover, the paper lacks information on computational requirements, training times, and resource consumption, which are important for assessing practical feasibility.

3) The scale and diversity of the experimental data raise some concerns. While the knot theory dataset is substantial, it only has 17 input dimensions, which may not represent high-dimensional challenges. The phase field dataset, on the other hand, is extremely small (only 96 samples), which may lead to overfitting and limit generalization. Furthermore, the absence of experiments on truly high-dimensional, large-scale datasets or standard machine learning benchmarks makes it difficult to evaluate PNAM's performance in more realistic settings.

4) The comparison with alternative approaches is somewhat limited. The paper primarily benchmarks against MLP, NAM, and KAN, but omits comparisons with other interpretable neural network methods and traditional statistical approaches like Generalized Additive Models (GAMs). Additionally, the comparison with symbolic regression methods is insufficient, especially given the emphasis on converting PNAM to symbolic form.

5) There is a lack of systematic parameter sensitivity analysis. The paper does not explore how the projection dimension M affects model performance and interpretability in depth, nor does it analyze the sensitivity to regularization weights, despite their critical role in balancing accuracy and interpretability. The impact of network architecture choices (depth, width) on performance is also not adequately addressed. Moreover, the evaluation of the quality and reliability of the symbolic expressions obtained through post-processing is limited, and there is no analysis of PNAM's computational complexity during training and inference.

### Questions
Please refer to the above questions.

It's a fascinating insight for NAMs, while the proof seems largely similar to generalized additive models.
I would be grateful and willing to raise my scores if the authors would solve my above concerns.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This study introduces a method to enhance the expressivity of neural additive models (NAMs) while allowing for interpretability. Specifically, it proves that the projected neural additive model (PNAM), which extends NAMs by applying a learnable linear transformation to the features before they enter the additive structure, achieves the universal approximation property. To address the reduced interpretability that arises from feature coupling after the linear transformation, the autors introduce regularization strategies that promote sparsity and penalize unnecessary interactions, enabling ranking and pruning of features and transformations. Finally, symbolic regression converts the pruned MLPs into mathematical expressions, further enhancing interpretability.

### Strengths
● Theoretical support for universal approximation. The paper makes a theoretical contribution by establishing the universal approximation property of PNAM, which had not been previously proven.

● Efforts to enhance interpretability. The framework employs both regularization and post-hoc techniques to improve interpretability by encouraging the use of only the necessary linear transformations of features. Moreover, converting the pruned MLPs into mathematical expressions enhances interpretability while reducing computational complexity.

### Weaknesses
● Reduced interpretability due to feature coupling. The learnable linear transformation entangles multiple features. This makes it difficult to attribute model behavior to individual inputs and limits interpretability.

● Computational complexity. The additional optimization required for regularization tuning and symbolic regression may introduce significant computational overhead.

● Dependence on hyperparameter choices. The performance and sparsity outcomes appear sensitive to the selection of projection dimension and regularization weights, requiring extensive tuning.

● Lack of synthetic experiments for interpretability validation. Synthetic experiments with known ground truth would enable a clearer quantitative assessment of interpretability, but such evaluation is currently missing.

● Limited evaluation of pruning and symbolic regression. Given the performance gap between the symbolic expression and the original MLP, it would strengthen the contribution to clearly justify why the derived symbolic form can still be regarded as interpretable and reliable.

### Questions
● Could you clarify the column descriptions in Table 3? In particular, it would be helpful to explain the distinction between the “reported” and “ours” values in the evaluation of test accuracy, the reason for the large discrepancy between them, and the precise definition of “Total acc.”

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This work attempts to provide a potential proof of the universal approximation property of PNAM through Theorem 2.1. Furthermore, by incorporating regularization constraints to enforce sparsity, as shown in Equation (9), the authors propose a possible approach to enhance the interpretability of PNAM.

### Strengths
1. It is helpful to provide a high-level overview or intuitive sketch before presenting the formal proof of Theorem 2.1.

2. The overall progression of the proof from Lemma A.2 to Lemma A.3 and then to Theorem A.1 is reasonable and clear, except for a few concerns I have raised (see Questions).

### Weaknesses
1. In the introduction, you should provide more details on how you define the interpretability of neural networks in your work. Does it refer to the significance of different input variables, the weighting parameters within the network, or another aspect?

2. In Section 2.1, which introduces the PNAM, more details should be provided about its underlying advantages and mechanisms. For example, why is a NAM needed beyond a standard neural network? How does it contribute to interpretability — by reducing the number of connections in a conventional NN, or in another way? In addition, how does PNAM enhance the expressiveness of NAM? Does the linear transformation from x to z play a beneficial role without compromising interpretability?

3. The mapping form of the PAM, as shown in Lemma A.1, appears to be more constrained than that of a conventional neural network, which could potentially reduce the hypothesis space for pattern recognition. How do you justify that the possible loss in expressiveness compared to a conventional NN is negligible or does not significantly affect performance? For example, can you justify why PNAM or NAM would not be more prone to underfitting compared to conventional neural networks? Otherwise, it is unclear why these new architectures are necessary, especially if their expressiveness is limited—even with a proof of universal approximation.

4. The notation of deg in Lemma A.2 should be explained to readers.

5. Even though the parameters may be non-unique, PNAM restricts the possible expressiveness to a subspace (as organized in Equation (A.3)) compared with a conventional neural network. How do you justify that the optimal expressiveness indeed lies within this subspace defined by PNAM?

### Questions
1. In Equation (1), $\epsilon$ represents the polynomial approximation error. Can it be reasonably treated as noise (for example, assumed to follow a Gaussian distribution)?

2. Regarding Equation (7), the proof of Theorem 2.1 does not appear rigorous. What justifies the assumption that the $\epsilon$ serving as an upper bound in Equation (7) is the same as the $\epsilon$ in Equation (5)? The essential condition for this relationship to hold is that the general form $F_2(X)$ in Equation (5) and the specific form of $F_2(X)$ in Equation (7)  is in the same continuous functional space.  Specifically, for conventional neural networks, this existence holds because they span the full polynomial space. However, in this work, you have constrained the representation to a subspace; therefore, how do you prove that the existence result still holds under this restriction?

3. Should interpretability help reduce uncertainty? However, with respect to Equation (9), it does not appear to guarantee this key property, since we can not guarantee the "significance" values are not wide in range. In that case, can the model still be considered interpretable?

4. What are the variables related to $\mathcal{L}_P$?

5. A key condition connecting Lemma A.2 and Lemma A.3 is the statement that ``Lemma A.3 can be proved by expanding $F(x_1, x_2)$, which produces the same set of monomials as the product of $f_1(x_1)$ and $f_2(x_2)$ in Lemma A.2.'' However, how do you justify that this condition always holds, or at least holds to some extent? Otherwise, certain forms of $F(x_1, x_2)$ may lack the necessary expressibility, making Theorem A.1 inapplicable.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper reviews Projected Neural Additive Models (PNAM) and targets to prove that this family can approximate any continuous target when given enough directions and flexible components. They also add practical tools for interpretability: encouraging the projection to be orthogonal and sparse, supporting monotonic/convex shape constraints on the one-dimensional components, and post-training symbolic compression of those components into compact expressions. Experiments on structured scientific problems demonstrate that PNAM outperforms a standard NAM and competes with other baselines at a similar capacity.

### Strengths
The math formulation and demonstration are easy to follow, also aligning with the claimed approximation capability.

The added practical interpretability tools (orthogonality/sparsity, shape constraints, symbolic compression) make it helpful to implement.

### Weaknesses
1. For Theorem 2.1, one clarification is needed: Is the universal approximation applied to an arbitrary continuous function?

2. From NAM to PNAM, it seems projection is what enables universal approximation here. If so, it may be clear and intuitive to show the mechanism, e.g., the projection mixes features and he 1-D parts then capture interactions that plain NAM cannot.

3. What’s the cost to get universality, like computation, memory, and sample complexity? As there is no free lunch.

4. While the claimed proof logic flow is easy to follow, the general idea needs some clarification. Why can this NN be universal at all? As formulated, the model abandons flexible raw-variable coupling, which is limiting. Is the point that the projection picks up the coupling among features into z, and then you keep separate 1-D MLP parameterizations on those directions? 

5. Interpretability of NN, and NN for symbolic regression are mentioned as motivation. It's helpful to clarify where PNAM sits relative to other interpretable NN families like NALU (neural arithmetic units) and AI Feynman / SINDy / DSR, which are also stated in the paper. PNAM offers interpretable 1-D parts of a high-dimensional approximation, while NALU has exact function recovery?

### Questions
Please see Weakness.

### Soundness
3

### Presentation
3

### Contribution
2
