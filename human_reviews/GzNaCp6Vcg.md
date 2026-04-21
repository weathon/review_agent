# PINNACLE: PINN Adaptive ColLocation and Experimental points selection

- Avg Score: 7.50
- Decision: Accept (spotlight)
- Scores: 8, 8, 8, 6

## Abstract
Physics-Informed Neural Networks (PINNs), which incorporate PDEs as soft constraints, train with a composite loss function that contains multiple training point types: different types of collocation points chosen during training to enforce each PDE and initial/boundary conditions, and experimental points which are usually costly to obtain via experiments or simulations. Training PINNs using this loss function is challenging as it typically requires selecting large numbers of points of different types, each with different training dynamics. Unlike past works that focused on the selection of either collocation or experimental points, this work introduces PINN Adaptive ColLocation and Experimental points selection (PINNACLE), the first algorithm that jointly optimizes the selection of all training point types, while automatically adjusting the proportion of collocation point types as training progresses. PINNACLE uses information on the interactions among training point types, which had not been considered before, based on an analysis of PINN training dynamics via the Neural Tangent Kernel (NTK). We theoretically show that the criterion used by PINNACLE is related to the PINN generalization error, and empirically demonstrate that PINNACLE is able to outperform existing point selection methods for forward, inverse, and transfer learning problems.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work looks into adaptive experimental design for physics informed neural networks (PINNs) incorporating PDEs as constraints. The authors evaluate the design points from various types including experiments and initial and boundary conditions by computational constraint. They are then put together into the proposed PINN Adaptive Collocation and Experimental points selection, or PINNACLE, framework, which considers the interactions between the type of points selected. The authors theoretically demonstrate the relation of the framework to the generalization error, and the performance of PINNACLE compared to other design methods.

### Strengths
1. Experimental design for physics-informed neural networks (PINN) is a very interesting problem, and incorporating the PDE constraints and the initial and boundary conditions into the system makes the objective of the work quite attractive and carry substantial weight.
2. The notion of augmentation of points from various sources with the computational budget constraint is a unique idea that combines domain knowledge with experimental design strategies, which I think is of value.
3. The work expressed the contribution and introduced the proposed framework clearly with illustrations that are quite helpful for understanding.

### Weaknesses
1. Not necessarily a weakness, but I am quite interested in the selection procedure of $Z_{pool}$ from $Z$ as introduced in Algorithm 1 and whether the authors have considered some structured designs that may give it more advantage than random sampling.
2. Similar question regarding the selection procedure from subset $Z$ of $Z_{pool}$. I see two different design approaches are used here, but I wonder whether the authors have considered additional options, say weighted space-filling designs. Just a thought.

### Questions
1. My understanding is that the constraints by initial conditions or boundary conditions are only considered as "soft" constraints here due to how the objective function is set up. I guess for most situations its impact on the results would not be significant, or in other words, the approximation is going to be quite good. However, I wonder whether there could be cases where even a minor violation of the limiting conditions will cause significant distruptions to the system, and if so, how the authors propose to tackle that.
2. This follows my previous question. I see the limiting conditions are introduced as regularization components into the objective function, with the corresponding regularizer coefficients described as having a certain value. I wonder whether the authors have considered how to best place values on these regularizers according to the physical system, and whether that would have a substantial influence on the final result. I presume different limiting conditions will have varying impact on the system, so I am curious on whether the selection of coefficients may be incorporated into the framework.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a method for adaptively selecting collocation points, which is necessary for training physics-informed neural networks. In particular, by treating the points related to experiments, partial-differential-equation constraints, and initial-boundary-value constraints in a unified manner, the proposed method automatically adjusts what proportion of collocation points should be allocated to each of them. The proposed method is based on a newly introduced criterion which is derived by using the theory of neural tangent kernels. The relationship between this criterion and generalization error has also been shown.

### Strengths
The choice of collocation points for PINNs is an important issue that has a significant impact on performance. In particular, this paper deals with collocation points in a unified way, independent of the type of loss functions associated with them. As far as I know, this is certainly a new approach and seems to be very promising.

In addition, the newly introduced criterion is derived with a theoretical basis and is highly reliable. This is just my impression but I suppose that the theorem that shows the relationship with the generalization error is itself valuable.

### Weaknesses
The strength of this paper seems to be that collocation points for initial boundary values and those for PDEs can be treated in a unified way, but a method of training networks without collocation points for initial boundary values is also proposed. When such a method is employed, the proposed method may lose a certain extent of significance.

### Questions
As mentioned above, for the initial boundary values, a method of designing neural networks to satisfy them has been proposed, e.g. in the following paper. 

Lagaris, I. E., Likas, A., and Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. IEEE Transactions on Neural Networks, 9(5):987–1000.

What is the significance of the proposed method when such a method is employed?

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes the PINNACLE, which is based on PINN network to address PDE problem. Specifically, proposed algorithm jointly optimizes the selection of all training points type and the authors demonstrate theoretically that criterion used by PINNACLE is related to generalization error of PINN.

### Strengths
1. The theoretical results are interesting. The author connects the proposed convergence degree notion with the generalization error bound.  and demonstrates how to approximate the optimal set for convergence degree.

2. The experimental results are great. Compared with other baselines, proposed methods introduce observable improvements.

### Weaknesses
1. For the K-MEANS++ method, the authors are encouraged to provide the time consumption comparison with other baselines. I am not sure if K-MENAS++ will introduce much extra computation cost.

2. For the K-MEANS++ method, the author claims that "this method select points with high convergence degrees". The authors are expected to provide more explanation why this method increase the convergence degree. It is the same for SAMPLING method. The authors are expected to explain how to "select a point which is proportional to $\hat{\alpha}(z)$ and how it improves the convergence degree" explicitly.  

3. For the experiments, the authors are encouraged to provide the results mean error with running time rather than only steps. Furthermore, since PINNACLE achieves great results in 1D Advection and 1D Burger settings, it will be interesting to explore with more settings, e.g., 1D Wave equation and 1D KdV equation.

### Questions
Check the weakness.

### Soundness
3 good

### Presentation
3 good

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
Joint selection of experimental points and collocation points in PINN training is proposed in this paper to achieve better training results for PINNs. A point selection algorithm that aims to select a training set that maximizes the convergence degree is proposed, which can lead to lower generalization bounds.

### Strengths
The paper contains a substantial amount of mathematical proofs. The experimental results are excellent, and the algorithm exhibits significantly higher accuracy compared to other algorithms.

### Weaknesses
The algorithm seems straightforward, and the presentation can be more concise.

### Questions
1. In Algorithm 1, how to select SAMPLING or K-Means sampling strategy?
2. Is the calculation of NTK used to compute the convergence degree?
3.The runtime comparison is not provided.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
