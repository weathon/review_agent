# Architectural Insights for efficient Physics-Informed Neural Network optimization

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 3, 3

## Abstract
Physics-informed neural networks (PINNs) offer a promising avenue for tackling both forward and inverse problems in partial differential equations (PDEs) by combining deep learning with fundamental physics principles. Despite their remarkable empirical success, PINNs have garnered a reputation for their notorious training challenges across a spectrum of PDEs. In this work, we delve into the intricacies of PINN optimization from a neural architecture perspective. Leveraging the Neural Tangent Kernel (NTK), our study reveals that Gaussian activations surpass several alternate activations when it comes to effectively training PINNs. Building on insights from numerical linear algebra, we introduce a preconditioned neural architecture, showcasing how such tailored architectures enhance the optimization process. Our theoretical findings are substantiated through rigorous validation against established PDEs within the scientific literature.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The manuscript studies optimization issues in the context of Physically Informed Neural Networks, specifically ones with Gaussian activation functions. The paper contains several related results.

1. Capitalizing on previous results showing that the lowest NTK eigenvalue correlates with convergence--- the authors derive a lower bound on the lowest NTK eigenvalue for the specific case of Gaussian activations. They argue that it increases favourably with width thereby allowing convergence for smaller networks. 

2. They then turn to study the conditioning number (ratio of highest to lowest eigenvalue) of a relevant part of the Loss Hessian and suggest a method of improving this number by evening out (equilibrating)  the rows of weight matrices. 

3. They perform several experiments supporting their statements that Gaussian PINNs and more so Equilibrated Gaussian PINNs are optimal for performance as well as better conditioned.

### Strengths
The paper applies recent results related to NTK to an important relatively fresh domain, that of PINNs. 

The theoretical effort is grounded and closely tied with experiments. 

It provides a novel PINN normalization scheme which may lead to improved performance.

### Weaknesses
The work is somewhat of a mixed bag of results. We have what is hopefully an exact result on the scaling of the lowest eigenvalue, a heuristic discussion of conditioning numbers of the Hessian, and experiments which partially support apparently known claims (that G-PINNs are better) and some which support the results on conditioning numbers and EG-PINNs. Due to this lack of cohesiveness, I feel that, to some extent, each of these three results should stand on its own. 

For example, the conditioning number is huge for all three architectures in Fig. 4. so can one really trust it to be an indicator of performance? If so should I expect EG-PINN performance to be 10^5 better than the rest (it isn't...)? If I try and rely on section 4. I encounter many heuristics and if I want to base this on the first result, then that first section discusses the lowest eigenvalue rather than the conditioning number and furthermore does that for a different matrix. 

More specifically I think potential weaknesses are the following
1. More numerics is required to support the claim that EG-PINN is a novel competitive architecture. For instance, if this was a claim on an image classifier, showing superior results on CIFAR-10, Fashion-MNIST, and ImageNet would have been the bare minimum for making such a claim. 

2. As NTK spectral is typically highly multiscale (and very loosely related to the Hessian), I'm not surprised by the huge conditioning numbers the authors find. However, I find it hard to see how such huge numbers can affect experiments. More probable in my mind, is that only the high Hessian modes are relevant and perhaps those are correlated with the lowest ones. To make this a bit more concrete say that we study standard NTK dynamics and that this conditioning number is the NTK conditioning number. This would mean the lowest eigenmodes of the target would be learned 10^30 times slower than the highest ones. Moreover, the learning rate is typically adjusted w.r.t. to the higher modes as they produce the majority of the gradients. Hence a realistic learning rate is one where the highest modes are learned in O(1) epochs hence the lowest ones in O(10^30) epochs... To summarize a better link is required between the heuristic claims of Sec. 4. and the experimental results. 

3. Generally, for properly normalized neural networks, the NTK eigenvalues scale as N. Given that the authors take n_k > N, I can't see how the statement that \lambda_{min}(K) is asymptotically dominating n_k^2 can be true. It appears that the authors are following the notations of previous works but still, this needs to be clarified. 

4. Evidence for the scaling of point 3. shown in Fig. 1. is quite weak. To show power law behaviour, I'd expect to see a log-log plot with at least two orders of magnitude. Similarly 

5. The theory of the current work is not sharply dependent on the PINN setting. Why are the authors allowed to analyze just the boundary aspects of the PINN NTK? From my experience, and from the authors' data, the boundary behaviour is typically easier to capture than the bulk behaviour. Would they argue that Gaussian activation is also good for image classification?

### Questions
Please see previous weaknesses section.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper shows that the smallest eigenvalue of the NTK of a PINN's data loss can be bounded from below if the activation functions are Gaussian, and that the condition number can be reduced by preconditioning of the initial weight matrices (row equilibration). These architectural changes -- equilibration and Gaussian activations -- are then shown to improve training success and test performance.

### Strengths
The paper treats an interesting and timely topic, as the training of PINNs is still not fully understood. From this perspective, the results in the paper are important contributions to a better understanding of PINNs. At a first glance, the experiments seem to support the theory presented in the paper.

### Weaknesses
The paper has, in my opinion, not matured sufficiently to merit publication at an A* venue, due to the following weaknesses:
* The experiments seem to rely on a setting different from the theory. While the theoretical results in Th. 3.1 require that the layer width $n_k$ is larger than $N$ (the number of collocation points, I assume), the experiments use comparably small layer widths (128 neurons). On the one hand, the theory is thus applicable to only very narrow settings rarely used in practice. On the other hand, it is then less clear how the theory and experimental evidence are connected, since the theorem does not hold for the experimental setting. From that perspective, I would be interested in seeing results also for PDEs exhibiting periodic behavior, to investigate whether the Gaussian activation still outperforms the sine activation.
* For the experiments, an ablation study is missing. If I understood correctly, 5.1 and 5.2 do not employ equilibration but only Gaussian activations, while 5.3 uses both, but does not show results for equilibration only (i.e., without Gaussian activations). I would appreciate seeing results (e.g., in 5.1) for the G-PINN, EG-PINN, and a PINN that uses tanh activations but preconditioned weight matrices.
* The results are not fully convincing. E.g., in Fig. 1 the quartic increase is not evident and would require much larger $N$. Also, the curve for tanh seems to be constant, while theory suggests quadratic behavior. Can you explain this discrepancy?
* The notation is not fully explained or clear (see below).
* The experimental setup is not fully clear: E.g., in Sec. 5.2, it is not clear how the initial and boundary conditions were chosen and if simulation data was available for training or not.
* Some statements in the paper are redundant, e.g., the last sentence on page 2, the first paragraph on page 4. The last paragraph of Sec. 4.1 is not fully clear. The font size in Fig. 2 is too small.

### Questions
See above for the most critical questions. Additional questions:
* Does $N$ in Th. 3.1 refer to $N_b$ or to $N_r$? 
* How is the $k$ in Th. 3.1 related to the number $L$ of layers? 
* What is the $\beta_k$ in Th. 3.1?
* In Prop. 4.2, how does equilibration affect the first term on the right-hand side of the inequality?
* Is there a connection between Gaussian activations and equilibration, or are these two independent ingredients?
* In Table 3, does the IC error and PDE error refer to test or training data?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper improves PINN in (1) Gaussian activations and (2) an architecture that conditions neural weights.

### Strengths
The paper is well-motivated and did rigorous theoretical analysis for both the Gaussian activations and the architecture that conditions neural weights..

### Weaknesses
Activation functions have been well-studied in the PINN literature: https://arxiv.org/abs/2209.02681.
Nowadays, proposing new activations does not seem to be novel.

For the model structure part, adding more neurons is now to boost PINN's performance. It is good to compare all model structures based on the same number of parameters. Also, some better models' inferences are more costly. The authors should take the computational costs into account.

### Questions
Please explain your contribution: how is your model better than the survey on activations functions in PINNs: https://arxiv.org/abs/2209.02681?

How expensive/efficient is your model structure? Can your model outperform others under the same number of parameters.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper investigates a new architectural design for accelerating training processes of PINNs. There are largely two components proposed: (i) Gaussian activation functions and (ii) preconditioning of weight matrices in layers. The new design is largely inspired from the neural tangent kernel perspective of interpreting PINNs and the goal is to design architectures that increase the magnitude of eigenvalues of the NTK for faster convergence. The paper tests the proposed architecture on several benchmark partial differential equations.

### Strengths
- The main manuscript is well-written and easy to follow. Although the whole appendix has not been carefully examined, several parts of Appendix look correct, which gives the impression that the rest of the parts would be also credible. 

- The idea of preconditioning the layer weights is interesting and two potential ideas were proposed (Jacobi and row equilibrated preconditioners), where both of them are diagonal matrices. 

- The paper provides experimental results comparing the results with other baselines.

### Weaknesses
- One of the main findings in the previous work (Wang, et al, 2022, JCP) is that the discrepancy between the eigenvalues of NTKs computed from the data matching loss $L_b$ and the residual loss $L_r$ should be small in order to achieve faster convergence. Loss re-weighting shown in the paper (Eq. (8)) is to mitigate such imbalance between $L_b$ and $L_r$.

- In the paper, the proposed architectural change is only to improve minimization of boundary loss $L_b$ (as written in the fourth paragraph of page 4 and also in Theorem 3.1), which gives some concerns how this would affect the balance between $L_b$ and $L_u$. Would there be any theoretical explanation in this regard? 

Concerns on practicality:
- As the method is derived from the NTK perspective, the users might have to assume that the width is larger than a certain value. On the other hand, as shown in the original PINNs, some empirical observations saying that making the neural network wider (e.g., larger than 50) would not provide much performance improvements. 

- Regarding the wall time: based on the number of epochs, it does seem that the proposed algorithm is better. However, considering that the additional matrix-vector products are added to the computations, it might not be the case that the proposed method performs better in terms of efficiency measured in wall-time (or achievable accuracy given a wall time). Would there be any studies in this regard?

Experimental section is weak. Regarding this please see the questions below.

The presentation has some room to be improved

  - for example, Figure 3, the colorbar range can be fixed. Current, with the eyeball norm, every heatmap looks the same. Figure 4, the table inside does seem to require some height and width scaling. 

  - Burger's => Burgers' 

  - the authors may not follow the guidelines when it comes to referencing papers (\cite or \citep or else).

### Questions
Please refer to the question above.

Additional questions are:

-  could the authors comment on the efficiency or the scalability of the proposed method? For varying width and depth, the performance could be varying and some systematical assessment would be needed to really see the proposed method works better. 

- how is the reported values are chosen? For example, in Table 3, could the authors can provide information on how were those numbers selected? Were the values generated by the single run of each model? were there hyper-parameter sweeps for baselines (width, depth, etc)? Is that particular specification used the experiments enough to assess the models' performance? (what are the justifications of that choice?)

- what is the definition of train error? is it the entire loss measured at training data instances? Also, in some tables, there are train error and in other tables, there are L2 train error. Are they different? How is the Rel. test error is defined?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good
