# Special Unitary Parameterized Estimators of Rotation

- Decision: Accept (Poster)
- Scores: 8, 8, 8, 10

## Abstract
This paper revisits the topic of rotation estimation through the lens of special unitary matrices. We begin by reformulating Wahba’s problem using $SU(2)$ to derive multiple solutions that yield linear constraints on corresponding quaternion parameters. We then explore applications of these constraints by formulating efficient methods for related problems. Finally, from this theoretical foundation, we propose two novel continuous representations for learning rotations in neural networks. Extensive experiments validate the effectiveness of the proposed methods.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper, building on the Wahba problem, proposes two novel representations for learning rotations in neural networks. The first, 2vec, closely resembles the widely used GSO implementation, which has become a standard in neural network training. This representation retains all the desirable properties of the GSO approach, including smoothness and continuity. The second, QuadMobius, introduces a 16-dimensional parameterization for rotations, offering an alternative approach for representing rotational transformations in neural networks.

### Strengths
Solid paper - good visualization. 
well motivated and good structure to follow along. 
coherent notation throughout the paper

### Weaknesses
Experimental results show, as expected, no drastic benefit. 
Discussion of timing results in the main paper can be emphasized more.

### Questions
no questions

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper develops three new representations of rotations based on unitary matrices.

Section 1 introduces Wahba's problem (aka estimate a rotation matrix from vector observations) and provides a brief overview on recent works. 

Section 2 first discusses how to measure distances in complex projective space and uses the derived distance metric to formulate Wahba's problem in terms of unitary rotations. This allows the problem to be solved in terms of a set of linear constraints on unitary matrix parameters / quaternions (Section 2.1). Section 2.2 approximates the solution to Wahba's problem by estimating an optimal Möbius transform (a special case of unitary matrix). Section 2.3 derives another solution in terms of linear constraints on quaternions parameters.

Section 3 discusses how the aforementioned solution to Wahba's problem extend to related rotation estimation tasks.

Section 4 proposes to novel formulation of rotations: 2-vec (6D vector that is mapped to an optimal rotation using unitary matrices) and QuadMobius (16D vector is mapped to a Möbius transform and subsequently rotation matrix). QuadMobius is computed either via the SVD or through an algebraic method.

Section 5 compares the proposed rotation representations to other representations (Euler, Quaternions, SVD) on various rotation estimation experiments.

### Strengths
Learning with 3D rotations affects numerous applications of machine learning, such that **providing three potentially state-of-the-art rotation representations** is a significant achievement for the ICLR community. The mathematical proofs are elegant and the experiments provide sufficient evidence on the utility of the derived representations. The code appears sound and will be (hopefully) made open-source.

### Weaknesses
The writing style of the paper is excellent. However, I personally found the paper's structure to be confusing. 

In particular, important prerequisite for understanding the paper are placed in the appendix, while other seemingly unrelated topics are discussed in the main paper:
1) The mathematical tools for this paper are given in Appendix A. Here, I would have liked a 2D illustration of the projective mapping. Also the notation in Equation 30 is a bit confusing. Perhaps, given the projection in (28) and (30) different symbols. Also an overview on the transformations between spaces (SO(3), S^2, CP1, S^3/Quaternion space, Möbius transform space) and the constraints in each space could be insightful.
2) In Section 1, the reader is introduced to Wahba's problem and possible routes to solving it using unitary rotations. Some proofs are given in the section while others are moved into Appendix B.
3) Section 3 talks about solving weighted and unweighted version of Wahba's problem. However, the derived equations are not used in the rest of the paper / experiments, such that I wondered what the purpose of this section was. It seemed this section could have been a paper of its own.
4) Section 4 is somewhat comprehensive, but unfortunately the authors only very briefly mention (around line 350) that the derived representations relate to prior work on learning with rotations. I would have preferred if the authors would have moved derivations to the appendix and shorten/remove Section 3 to instead talk in depth about connections to related work.

### Questions
- Line 426 - Is this really an  "Unsupervised learning task"? Seems quite supervised to me as joint locations are given.

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes new parameterization methods for 3D rotation estimators. The method is based on operation in the complex number domain, and 3 methods are proposed including 2-vector, QMAlg and QMSVD. The proposed method is benchmarked on Wahba's problem, point cloud pose estimation, inverse Kinematics and camera pose estimation.

### Strengths
- The parameterization of rotation estimator is a challenging and practically important topic in machine learning.
- The paper introduce new theoretical results and their proofs.
- The experimental results are extensive and the results are mostly better than the baselines.

### Weaknesses
- The paper is math heavy and may not be easy for general reader to get a gist of the approach.

### Questions
- Do the proposed approaches solve the discontinuity and double cover problems of 3D rotation parameterization?
- Do the baseline methods (e.g., GS) also solve these two problems? The GS baseline seems to work well for point cloud data and camera pose data, I am curious whether the importance of solving discontinuity or/and double cover issues are shown in the results.
- What are the numbers of output elements need for the proposed methods to parametrize a rotation?
- Figure 1 provides some intuitive illustration of the method. For QuadMobius, is there an interpretation for the fact that projecting to intermediate points can improve the results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This work presents a comprehensive and theoretically grounded exploration of 3D rotation estimation, introducing Wahba's problem's solution via the special unitary group SU(2). This work proposes two continuous representations for learning rotations in deep neural networks:
1. 2-vec: different from Gram-Schmidt's greedy approach, 2-vec maps these vectors to an optimal rotation in the sense of Wahba's problem.
2. QuadMobius:  A 16D representation based on the paper's Möbius transformation approximation. It maps the network output to a Hermitian matrix, solves for its smallest eigenvector, and then projects onto SU(2) to get the final rotation.
The authors demonstrate these contributions through extensive experiments, including 3D shape alignment, unsupervised inverse kinematics, and real-world camera pose estimation.

### Strengths
1. The reformulation of Wahba’s problem via SU(2) presents a rigorous mathematical derivation. Translating theoretical derivation to practical representation (2-vec and QuadMobius) of deep rotatiton estimation is novel and intriguing.
2. The paper tests its representations on a diverse set of tasks on ModelNet10-SO3, Inverse Kinematics,  and Cambridge Landmarks, providing strong evidence of their robustness and general effectiveness of the proposed representation.

### Weaknesses
1. In Appendix F, the author tried to directly predit an SU(2) form but it works poorly due to the double cover issue like quaternions. It raises the question that the true improvement stems from the intermediate Möbius transformation rather than the property from SU(2). The paper might want to clarify its narrative to highlight the key contributing factor.
2. The QuadMobius has noticeably slower inference speed, shown in Tab. 7, but I believe this is a minor issue as running the neural network inference would dominate the slight increase of arithmetic computation from QuadMobius.

### Questions
I suggest the authors highlight the intuition why SU(2) derived rotation form performs better early on in the paper. Though the theoretical in-depth derivation is presented beautifully, it's unclear whether the true source of improvement of QuadMobius is from the properties of SU(2) or Mobius transformation. I hope the authors could clarify this.

### Soundness
3

### Presentation
4

### Contribution
4
