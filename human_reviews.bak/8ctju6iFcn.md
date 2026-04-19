# Certified Training with Branch-and-Bound: A Case Study on Lyapunov-stable Neural Control

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 5, 5

## Abstract
We study the problem of learning Lyapunov-stable neural controllers which provably satisfy the Lyapunov asymptotic stability condition within a region-of-attraction.  Compared to previous works which commonly used counterexample guided training on this task, we develop a new and generally formulated certified training framework named CT-BaB, and we optimize for differentiable verified bounds, to produce verification-friendly models. In order to handle the relatively large region-of-interest, we propose a novel framework of training-time branch-and-bound to dynamically maintain a training dataset of subregions throughout training, such that the hardest subregions are iteratively split into smaller ones whose verified bounds can be computed more tightly to ease the training. We demonstrate that our new training framework can produce models which can be more efficiently verified at test time. On the largest 2D quadrotor dynamical system, verification for our model is more than 5X faster compared to the baseline, while our size of region-of-attraction is 16X larger than the baseline.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper aims to produce Lyapunov-stable neural controllers. The novel aspect of the work is using a branch-and-bound approach during training to generate samples. The authors evaluate their method using case studies on several dynamical systems, demonstrating a significant reduction in verification time and an expansion of the region of attraction (ROA) compared to existing methods.

### Strengths
The experiment result shows the proposed approach outperforms the previous work.

### Weaknesses
This work has the following major issues.

1. The background and related work introduction is not proper.
    - 1.1. The discussion of a whole branch of work on barrier function/control barrier function-based controller training is largely missing or in a misleading way. Barrier function and control barrier function techniques for safety are very similar to Lyapunov function techniques for stability in terms of their formulation in control theory. There have been extensive works on neural barrier function learning or barrier/policy joint learning [1,2,3]. The authors misinterpret these works by claiming that they, including [2], "did not provide formal guarantees". In fact, most of these works [1,2] explicitly mention they have an additional verification step to ensure the correctness of their approaches (Page 10 in [1] and Page 6 in [2]). It raises concerns about the credibility of this work. Please properly discuss these works.
    - 1.2. The local/global robustness verification is not quite relevant to this work, and the discussion should be eliminated from the related work. Instead, the verification of neural-network controlled systems should be discussed, as they are in the same track of this work, e.g., [4,5,6]. 

2. The branch-and-bound section is not very clear.
    - 2.1. Is the training dataset $\mathbb{D}$ a set of points or regions? What does it mean by $(\underline{x},\underline{x})$ in Line 269?
    - 2.2. Does the proposed approach need to verify that $g(x)\leq 0$ for every region? If so, then how does this work scale to high-dimensional systems?

3. Empirical comparison with STOAs is expected.
    - 3.1. ROA considered in this work is also very relevant to the invariant set in control theory. Thus, this work is expected to compare with STOAs in this domain, e.g., [7].  

[1] Zhao, Hengjun, Xia Zeng, Taolue Chen, Zhiming Liu, and Jim Woodcock. "Learning Safe Neural Network Controllers with Barrier Certificates." arXiv preprint arXiv:2009.09826 (2020).

[2] Jin, Wanxin, Zhaoran Wang, Zhuoran Yang, and Shaoshuai Mou. "Neural certificates for safe control policies." arXiv preprint arXiv:2006.08465 (2020).

[3] Wang, Yixuan, Simon Sinong Zhan, Ruochen Jiao, Zhilu Wang, Wanxin Jin, Zhuoran Yang, Zhaoran Wang, Chao Huang, and Qi Zhu. "Enforcing Hard Constraints with Soft Barriers: Safe Reinforcement Learning in Unknown Stochastic Environments." arXiv preprint arXiv:2209.15090 (2022).

[4] Ivanov, Radoslav, Taylor Carpenter, James Weimer, Rajeev Alur, George Pappas, and Insup Lee. "Verisig 2.0: Verification of neural network controllers using taylor model preconditioning." In International Conference on Computer Aided Verification, pp. 249-262. Cham: Springer International Publishing, 2021.

[5] Huang, Chao, Jiameng Fan, Xin Chen, Wenchao Li, and Qi Zhu. "Polar: A polynomial arithmetic framework for verifying neural-network controlled systems." In International Symposium on Automated Technology for Verification and Analysis, pp. 414-430. Cham: Springer International Publishing, 2022.

[6] Teuber, Samuel, Stefan Mitsch, and André Platzer. "Provably Safe Neural Network Controllers via Differential Dynamic Logic." arXiv preprint arXiv:2402.10998 (2024).

[7] Harapanahalli, Akash, and Samuel Coogan. "Certified Robust Invariant Polytope Training in Neural Controlled ODEs." arXiv preprint arXiv:2408.01273 (2024).

### Questions
Please see the weakness.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper aims to train safe neural network controller in dynamic systems in discrete time and continuous action space. The safety is formally verified using lyapunov function (stability) and existing formal NN verifier. The training process consists of recursively splitting the considered input domain (region-of-interest) and learning through backpropagation to fullfill two conditions on each subset: (1) lyapunov stability and (2) controller should not steer outside of input domain. This iteratively increases the region for which stabilty is shown (region of attraction (ROA) due to lyapunov function), which the paper aims to maximize. The approach is demonstrated on three low-dimensional benchmarks.

### Strengths
- The paper is very well written, and one can follow along with all arguments and choices nicely.
- The topic of formally verifying the safety of neural network controller is highly relevant and suits ICLR
- The approach is novel and demonstrated on three benchmarks with improvements on related work.
- Even though the approach is simple, it is still effective in achieving its goal.

### Weaknesses
Major points
- Approach is based on branch-and-bound, which suffers from the curse of dimensionality and thus might not be applicable in high-dimensional systems. This is also visible in Tab. 3 where the final data set size is already much larger for the 6-dimensional quadrotor benchmark (which probably also increased the runtime). Appendix A also states that they stopped splitting after 5000 training steps for this benchmark. This limitation is mentioned in Sec. 5 but a discussion about the theoretical complexity and directions to overcome this limitation would be helpful.
- The paper only considers the safety property of stability, i.e., the actor should steer to some (predefined?) equilibrium state  (see question below for other safety properties that could be discussed).
- The results in Sec. 4 are only single-dimensional: Over how many seeds are the runs averaged? Please provide stand deviations where applicable, e.g., the area of ROA.

Minor points:
- Eq (2) assumes perfect knowledge of the system with no disturbance (e.g., sensor noise).
- The paper considers only discrete time and continuous action space, which could be highlighted more clearly.
- Spelling / Grammar mistakes: e.g., line 063: "stability condition needs to *be* verified", line 168: "lower bound" should be "upper bound" (?), line 182: "the the", line 183: "an NN" should be "a NN", ...
- Experiments are only done on low-dimensional systems. This limitation is addressed in the paper.
- Convergence to a single point within the region of interest is not well motivated (see question below)
- Appendix A: The considered networks are rather small.
- no repeatability package

Other points that did not directly influence the score
- First contribution: "relatively global guarantee" could be specified more precisely.
- Tab. 1: (re-)explain all variables, e.g., d, n_u are not explained properly.
- Fig. 1: What does the color mean? Is it your Lyapunov function?

### Questions
- Using your dynamic splitting, are there certain areas that get splitted more often than other areas? How do these areas differ? E.g., are there more splits at V(x) = 0 or similar? Would be nice to get more insights there. Maybe one can visualize the splitted subsets using a heat map or similar to see in which area the most splits occured.
- Does the number of regions converge to some maximum or do they continuously get split during training? 
- Why did you decide to stop splitting after 5000 training steps for the quadrotor benchmark?
- Would it be useful to merge some regions at some point again?
- Is it sensible to assume a single equilibrium state x*? How is x* determined as x* has to be known to evaluate the equations given sin Sec. 3.4? The authors could discuss the implications of this assumption and how the method needs to be adapted for multiple x*.
- Would your approach also work for more equilibrium states / if the system does not converge but should not violate a safety property where the actor should stay outside of some unsafe region after some point in time?
- End of Sec. 3.1: What holds for points x \in B \  S ? 
- Addition of adversarial attacks in training objective (6): Why is this necessary? How would training change if this was not included? Would it still work or is it necessary "to get started"? Maybe add to ablation study.
- Sec. 3.3: The initial splits appear a bit random: Would it not be better to start off with the entire region of interest and only refine into subsets where necessary?
- Is it sensible to test each dimension before deciding where to split? This might increase the training time (especially if higher-dimensional systems are considered). Would another heuristic make more sence, e.g., sensitivity?
- Lines 306-309: Isn't this already penalized by the second condition in (4)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes a new certified training framework for generating neural networks with relative global guarantees. By introducing a training-time branch-and-bound method that dynamically maintains a training dataset, the most difficult sub-regions are iteratively divided into smaller sub-regions. The verification boundaries of these sub-regions can be computed more tightly to simplify training, addressing the challenges of certified training in relatively large input regions.

### Strengths
This paper is well-written and easy to follow with clear logic and a well-structured layout. The comparative experimental results, along with accompanying visualizations, demonstrate the effectiveness of the proposed method, and the necessity of the training dataset is highlighted through ablation experiments.

### Weaknesses
Although this paper presents valuable insights, there are still several areas that need improvement.

It is well-known that the synthesis speed of Lyapunov functions for low-dimensional nonlinear systems is very fast, especially with learning methods based on SMT solvers and counterexample-guided approaches. These methods not only provide formal correctness guarantees (in contrast to the simulation testing used in this paper) but also leverage efficient neural network architectures, demonstrating strong learning capabilities. However, the certified training framework proposed in this paper, based on the branch-and-bound idea, lacks theoretical support and does not discuss the soundness of the proposed method.

The experimental section of this paper provides only limited comparisons with the related work of Yang et al., 2024, making it difficult to assess the reliability of the experimental results. The authors should also compare their method with other approaches to highlight the contributions of this work. Additionally, we would like to see more examples involving high-dimensional systems to demonstrate the efficiency of the proposed method.

Due to the limited length of the article, the discussion of related work should be appropriately integrated into the introduction. It is necessary to introduce the relevant theories and background on system stability and Lyapunov functions. The paper focuses on the work of Yang et al.; we suggest that the authors provide a brief explanation to highlight the differences between their method and the new approach presented in this paper.

### Questions
I would like to emphasize that this paper is well-written and clear. But there are some questions and uncertainties that I hope the authors can kindly address.

1.	Learning-based methods often lack interpretability. Although a Lyapunov function is obtained through learning, there may be some errors. Does the learned Lyapunov function truly satisfy the stability conditions? Could the authors provide an explanation for this? 

2.	The authors state in the paper, "Empirically, the neural controllers generated by the training framework in this work can be verified to satisfy the Lyapunov conditions, with a larger region of attraction (ROA), and the Lyapunov conditions can be verified more effectively during testing." Generally, the consistency between experimental results and theoretical foundations provides assurance for the validity of the method. I do not understand why the method's efficiency is indicated solely based on empirical evidence, especially since these experiments are limited and conducted in low dimensions.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a verification framework for certified training based on BaB techniques, and conducts a case study on the neural Lyapunov control task. Unlike adversarial training techniques widely used in the literature, this paper obtains a relatively global output guarantee without using time consuming verifiers such as SMT, MIP, etc.

### Strengths
1. Proposed method verifies the condition $g_\theta(x)\leq 0$ for $x\in\mathcal{B}$ instead of adversarial examples.
2. The experiments on Mujoco environments, particularly the significant reduction in verification time and the expansion of the region of attraction (ROA), provide strong evidence for the framework's effectiveness. Achieving a 5X faster verification time and a 16X larger ROA for the 2D quadrotor system demonstrates impactful results.

### Weaknesses
1. Line 215: typo “on the entire”.
2. Relaxation of the activation function is mainly RELU based. It would be more interesting to see some other activation functions.
3. It is claimed in the introduction that “this approach supports the random initialization” in lines 324 to 330. It would be great to have some experiments with different randomized initialization and the initialization impacts on verification and ROA calculations.
4. In line 307, it would be great to explain more why replacing margin $\rho$ to $\rho+\epsilon$ will prevent the controller going out of ROI.
5. The setting and Lyapunov synthesis condition is exactly the same as [1]. The only difference is that the certification uses BaB framework instead of adversarial robustness, which makes the contribution not too significant, since BaB framework for neural network verification is also pretty common (e.g., [7,8]), especially for RELU network. It would be great to see some variations using different reachability tools other than $\alpha-\beta$ CROWN, such as in Sherlock [2], nnenum [3], etc.
6. Also it would be interesting to compare the current setup with some existing NNCS (Neural Network Control System) verification tools, such as NNV [4], Polar-Express [5], CORA [6], etc.
7. Though not limited to this method, all the similar methods seem to have a bottleneck on dimensionality.

References:
1. Yang, Lujie, et al. "Lyapunov-stable Neural Control for State and Output Feedback: A Novel Formulation." Forty-first International Conference on Machine Learning.
2. Dutta, S., Chen, X., Jha, S., Sankaranarayanan, S., Tiwari, A.: Sherlock-a tool for verification of neural network feedback systems: demo abstract. In: Proceedings of the 22nd ACM International Conference on Hybrid Systems: Computation and Control. pp. 262–263 (2019)
3. Bak, Stanley. "nnenum: Verification of relu neural networks with optimized abstraction refinement." NASA formal methods symposium. Cham: Springer International Publishing, 2021.
4. Lopez, D.M., Choi, S.W., Tran, H.D., Johnson, T.T.: Nnv 2.0: the neural network verification tool. In: International Conference on Computer Aided Verification. pp. 397–412. Springer (2023)
5. Wang, Y., Zhou, W., Fan, J., Wang, Z., Li, J., Chen, X., Huang, C., Li, W., Zhu, Q.: Polar-express: Efficient and precise formal reachability analysis of neural-network controlled systems. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems (2023)
6. Althoff, Matthias, and Niklas Kochdumper. "CORA 2016 manual." TU Munich 85748 (2016).
7. Bunel, Rudy, et al. "Branch and bound for piecewise linear neural network verification." Journal of Machine Learning Research 21.42 (2020): 1-39.
8. Wang, Shiqi, et al. "Beta-crown: Efficient bound propagation with per-neuron split constraints for neural network robustness verification." Advances in Neural Information Processing Systems 34 (2021): 29909-29921.

### Questions
See above in weaknesses

### Soundness
3

### Presentation
3

### Contribution
2
