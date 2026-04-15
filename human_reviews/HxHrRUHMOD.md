# Accurate Differential Operators for Neural Fields

- Decision: Reject
- Scores: 3, 8, 6, 3

## Abstract
Neural fields have become widely used in various fields, from shape representation to neural rendering, and for solving partial differential equations (PDEs). With the advent of hybrid neural field representations like Instant NGP that leverage small MLPs and explicit representations, these models train quickly and can fit large scenes. Yet in many applications like rendering and simulation, hybrid neural fields can cause noticeable and unreasonable artifacts. This is because they do not yield accurate spatial derivatives needed for these downstream applications. In this work, we propose two ways to circumvent these challenges. Our first approach is a post hoc operator that uses local polynomial-fitting to obtain more accurate derivatives from pre-trained hybrid neural fields. Additionally, we also propose a self-supervised fine-tuning approach that refines the neural field to yield accurate derivatives directly while preserving the initial signal. We show the application of our method on rendering, collision simulation, and solving PDEs. We observe that using our approach yields more accurate derivatives, reducing artifacts and leading to more accurate simulations in downstream applications.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a post-processing method to improve the estimation of first- and second-order derivatives on learned hybrid neural fields (e.g., Müller et al., 2022a), a.k.a. implicit neural representations (INRs). By applying the local polynomial regression, the estimates of these derivatives are smoothed w.r.t the size of the local sampling domain and the number of samples in the local domain. An auxiliary approach is also proposed to fine-tune hybrid neural fields by biassing the estimates via autograd toward the previously proposed smoothed operator.

### Strengths
- Quality: The paper offers a range of practical downstream applications to illustrate the utility of the approach for hybrid neural fields. While it may not yield surprising results for most proposed tasks, it does exhibit improvements compared to pre-trained hybrid neural fields.
- Clarity: The approach is lucidly presented, drawing motivation from empirical observations and classical techniques in shape analysis.
- Significance: The approach may be beneficial for downstream applications requiring certain types of neural field architectures.

### Weaknesses
- It seems that the paper's claim regarding motivation from empirical observations might be somewhat overstated, particularly when applying it to **all** types of neural fields. The problem of noise overfitting is notably more prevalent in neural fields that heavily emphasize local details but might not be as prominent in other more global architectures like SIREN or Multiplicative Filter Networks. The authors should consider providing additional evidence, particularly in the context of these architectures, to support their claim for all neural fields.
- Novelty: Novelty: The proposed method can be applied to any continuous function, but its contribution from a machine learning perspective is limited apart from adding the fine-tuning loss.
- It seems that the authors modified the neural field architecture used in Chen et al., (2023a) for the application of PDE simulation (sec. 5.3). Will the claim of avoiding prediction error explosion still hold with the original architecture?
- When working with signals that inherently contain high-frequency features, as often encountered in applications like Physics-informed approaches (such as Chen et al., 2023a) for fluid dynamics, it may not be appropriate to employ unified smoothing parameters across the entire domain.
- Minor issues: 
  - Please double-check the citations in your paper. There are some instances of:
    - Duplicated citations (e.g., "Chen et al., 2022c" and "Chen et al., 2023a", "Müller et al., 2022a" and "Müller et al., 2022b", "Tancik et al., 2022a" and "Tancik et al., 2022b").
    - Incorrect formatting (Sara Fridovich-Keil and Alex Yu et al., 2022).

### Questions
- Does the prediction error explode using the original architecture with SIREN as in Chen et al., (2023a)?
- In Fig. 7, the grid solver gives non-zero loss at the initial time. Is there any issue in the simulator?

### Soundness
3 good

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper addresses a critical issue in neural fields, namely, the gradient problem, which holds immense significance in computer vision (e.g., 3D reconstruction in NeRF; consider the neuralAngelo paper), geometry processing, and solving PDEs. The authors propose a method to accurately compute differential operators, including gradients, Hessians, and Laplacians, for hybrid grid-based neural fields. This approach effectively mitigates high-frequency noise issues associated with previous hybrid neural field techniques, particularly on high-resolution grids, by locally fitting a low-degree polynomial and computing derivative operators on the fitted polynomial function. The author's approach outperforms both auto-differentiation-based and finite difference-based methods.

### Strengths
Clear Problem Formulation: The authors adeptly address the critical gradient problem in neural fields, making a significant contribution to the field and potentially enabling broader adoption of neural fields over traditional representations like meshes and point clouds.

Novel Approach: The proposed polynomial fitting approach is both straightforward and highly effective, tackling issues related to high-frequency noise and artifacts.

Experimental Validation: The authors thoroughly validate their approach across various neural field applications, including rendering, collision simulations, and PDE solving.

Hyperparameter Discussion: The paper provides a very nice discussion of hyperparameters, particularly the parameters 'k' and 'sigma.' The comparison with MLS methods, which also involve hyperparameters (spread), also adds value to the discussion.

Comparison with Previous Methods: The authors conduct a comprehensive comparison with baselines using a variety of datasets, such as the FamousShape dataset

### Weaknesses
The primary concern lies in the scalability of this method. Most of the presented results pertain to normal-scale setups. While these results are suitable for the paper's scope, questions arise regarding the method's ability to handle gigascale scenarios, especially with the availability of gigascale NeRF implementations.

### Questions
It would be beneficial if the authors could provide more insights into the scalability of their method and its potential applicability to gigascale NeRF scenarios.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper the authors tackle issues that arise with hybrid (grid-based) neural fields in the computation of differential operators. They emphasize that automatic differentiations leads to inaccurate and noisy derivatives with a trained hybrid neural field $F$.  Then, they introduce a local-polynomial to approximate the SDF field in the neighborhood of a query point, in order to take the differential operators of the local polynomial instead of that of $F$. This approach can be useful at test time, but requires computation at each iteration. To tackle this issue, the authors propose a fine-tuning that is supervised with smoothed gradients while preserving the represented signal.
They perform some experiments to validate the effectiveness of the local-polynomial and of the fine-tuning, and then show that this increased accuracy in derivative computations has a positive impact on several downstream tasks (rendering, collision simulations, 2d advection).

### Strengths
The paper is really well written and enjoyable to read. The angle of the paper is original and the local-polynomial fitting sounds like a simple yet effective idea to smooth out the derivatives of a neural field. The authors experimentally validate their polynomial-based differential operators on the FamousShape dataset, on which they outperform autodiff and finite difference across all metrics. Subsequently, they experimentally confirmed that fine-tuning with polynomial fitting lead to better results than with finite differences.

### Weaknesses
All the experiments were conducted with instant-ngp, a particular implementation of a hybrid neural field. Therefore, it is difficult to understand if the problem is systematic across all methods of this class. It would also have been interesting to compare the method with non-grid-based neural fields, such as SIREN, Fourier Features, MFN, etc. 

Similarly, apart from the PDE simulation section - which seems to be unrelated to the rest of the paper - all the experiments were done with SDF data. It would strengthen the paper if the method would also apply to other modalities.

### Questions
For post hoc operators, how do you find the best value of $\sigma$ ? What is the criterion to choose the $\sigma$ that yields the best autodiff gradients ?

Would the method work with different neural field architectures ? Did you try other grid-based neural fields ? Would it be also useful to apply this method for non-grid-based neural fields ?

Did you try your method on other datasets and with other modalities ?

How do you solve the polynomial fitting ? How long does it take for one query ?

What is the training pipeline for fine-tuning ? How long does it take ?

Could you provide more details on the PDE simulation section ? Can you compare your result with standard INR ?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors address the important problem of inaccurate derivatives computed via automatic differentiation (autodiff) from hybrid neural fields. The authors identify high-frequency noise amplification during derivative computation as the core issue. They propose two solutions - a  polynomial fitting operator to smooth the field before differentiation, and a method to fine-tune the field to align autodiff derivatives with more accurate estimates. Experiments demonstrate 4x and 20x reductions in gradient and curvature errors respectively over autodiff.

### Strengths
- Identifies a significant limitation of hybrid neural fields that could impede their use in downstream applications requiring accurate derivatives.
- Proposes two concrete solutions that are justified from both theoretical and experimental perspectives.
- Achieves impressive quantitative improvements over baselines.
- Demonstrates reduced artifacts and improved results when using the proposed methods in example applications like rendering, simulation, and PDE solving.

### Weaknesses
- **Lack of rigorous literature review**: It seems that the authors have ignored some of the very relevant papers in this domain. The problem of inaccurate and noisy gradients is a well known problem and several people have proposed alternative solutions: (1) DiGS : Divergence guided shape implicit neural representation for unoriented point clouds (2) Given the high cost of dense point cloud capture, can we use sparse point clouds?​ (3) Use a local region prior to generalize to various unseen local reconstruction targets​ (4) NeuFENet: Neural Finite Element Solutions with Theoretical Bounds for Parametric PDEs (5) mechanoChemML: A software library for machine learning in computational materials physics. These are just a few examples of how people try to use other approaches to compute the derivative than, the approach that authors take. While the authors may and probably will try to defend the paper saying that these papers are not relevant, I would argue that they are. For example: [2] uses the nearest neighbors idea to compute SDF and update the derivative (using AD of course). [4,5] use Lagrange polynomials from Finite Elements to evaluate the gradients for backprop, [1] solves the AD noise problem by adding an additional loss for having zero divergence. Each of them are valid alternative approaches to what the authors of this paper are planning to do. A good paper would try to compare with them and ensure that their approach is truly necessary.

- **Computational cost** - The method seems to be completely based on a pre-trained network. Which means, the model is essentially being trained twice. 

- **moving target problem** - if the authors use the pre-trained model as a reference, do they assume that pre-trained model is good enough to approximate the properties? Do they consider, updating the weights of the pre-trained model after say 1000 iterations? (just like it is done in DQN-type algorithms?). This could be a great concern if the noisy gradients dont allow the model to converge to a solution at all.

- **Instant-NGP** - It seems that all the experiments from the author are using Instant-NGP as their baseline and working further from there. Naturally, there are issues with Instant-NGP that the multi-resolution hash encoding is not well understood in terms of derivatives. Some recent papers provide some mathematically sound hash encodings for AD (In Neus or Neus++ paper I think). Therefore, doing the experiments using some non-hash-encoding-based methods like SIREN or Nerf (although time-consuming, would be better).

-**Comparisons** - comparison with Finite-Difference (which is already built-in inside Instant NGP) and AD is like a typical strawman approach where you choose several approaches and rigorously test them is missing. I would like to see more than this.


-

### Questions
See the weaknesses above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
