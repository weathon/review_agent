# A Primer on SO(3) Action Representations in Deep Reinforcement Learning

- Decision: Accept (Poster)
- Scores: 4, 6, 6

## Abstract
Many robotic control tasks require policies to act on orientations, yet the geometry of SO(3) makes this nontrivial. Because SO(3) admits no global, smooth, minimal parameterization, common representations such as Euler angles, quaternions, rotation matrices, and Lie algebra coordinates introduce distinct constraints and failure modes. While these trade-offs are well studied for supervised learning, their implications for actions in reinforcement learning remain unclear. We systematically evaluate SO(3) action representations across three standard continuous control algorithms, PPO, SAC, and TD3, under dense and sparse rewards. We compare how representations shape exploration, interact with entropy regularization, and affect training stability through empirical studies and analyze the implications of different projections for obtaining valid rotations from Euclidean network outputs. Across a suite of robotics benchmarks, we quantify the practical impact of these choices and distill simple, implementation-ready guidelines for selecting and using rotation actions. Our results highlight that representation-induced geometry strongly influences exploration and optimization and show that representing actions as tangent vectors in the local frame yields the most reliable results across algorithms. The project webpage and code are available at amacati.github.io/so3_primer.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Authors investigated the importance of 3D rotation representations in the context of reinforcement learning. Authors further discussed 4 rotation representations, i.e., rotation matrix, quaternion, Euler angle, and Lie algebra, in absolute and relative coordinates. Extensive simulation experiments are performed on drone control as well as manipulation problems. Authors concluded that on average, the relative Lie algebra is the best performant representation for RL applications. To explain the advantage of relative Lie algebra, various hypothesis are presented.

### Strengths
Despite previous studies on rotation representations in the context of supervised learning, this paper explores rotation representations in the RL setup, which is an important topic but is ignored by previous works. This paper provided extensive comparisons on different RL benchmarks, ranging from drop control to manipulation. This paper also compared different RL algorithms, including stochastic and deterministic algorithms. The hypothesis point interesting explanation of the advantages of relative representations.

### Weaknesses
One important baseline seems messing: 6D rotation representation [1], where the neural network first outputs two 3D vectors, then uses Gram-Schmidt to normalize these vectors and thus reconstructs the rotation matrix. How would this baseline, and perhaps it’s relative version, performs in the RL tasks?

[1] On the Continuity of Rotation Representations in Neural Networks. Yi Zhou, Connelly Barnes, Jingwan Lu, Jimei Yang, Hao Li.

### Questions
Table 1 is very helpful for explaining different rotation representations. Could authors add similar table to summarize the representations on Table 2? E.g., R stands for rotation matrix, \delta R stands for relative rotation matrix.

How does Figure 2 visualize SO(3) rotation? Figure 2 seems visualized a S2 sphere, which only has 2 DoFs, but SO(3) has 3 DoFs.

Why Figure 3 only compared relative rotation representations? What do absolution representations perform?

For Figure 4, an additional bar that shows the average of each representation would be helpful for overall comparison.

### Soundness
3

### Presentation
2

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
The paper analysis how SO(3) rotation actions should be parameterized to facilitate reinforcement learning using PPO, SAC, and TD3. In particular, the paper analyzes how using the changes in rotation as action representation could benefit learning. First the paper discusses the different rotation representations and how to predict changes in rotation from deterministic / stochastic policy networks. In a simply toy example, the authors show that indeed the different rotation representation affect learning in particular for sparse reward settings. Afterwards, the authors discuss different hypothesis on why certain rotation representations show better performance. In particular, the authors consider the effect of smoothness on learning, interplay of rotation projections and exploration / entropy regularization, and action scaling. Finally, the paper provides extensive experimental benchmarks that underline the importance of rotation representations for reinforcement learning in SO(3).

### Strengths
The paper looks at a critical issue for the reinforcement learning community that so far has been mostly overlooked. The paper is well structured, clearly written, and the experiments are extensive and thorough. Considering how rotation projection affects exploration in RL is quite clever. I generally enjoyed reading the paper and consider it an excellent fit for ICLR.

### Weaknesses
**Comment 1: Each delta rotation representation requires special treatment to perform well.**  

In Section 2.4, the authors outline how the Euclidean network outputs are mapped onto the rotation manifolds. For deterministic networks, the action are simply projected onto the respective manifold via a differentiable projection layer. For stochastic policies (as far as I can guess from the very brief description), the network mean is projected onto a rotation manifold, then points are sampled around this projected vector in Euclidean space, and these points are also projected onto the rotation manifold. **These approaches on obtaining rotations from deterministic/stochastic policies  are the basis of all of the subsequent discussion and therefore quite important.** 

The problem is that neural networks are designed around zero-centered activations which with standard parameter initialization output zero-centered output distributions. This becomes a problem, if we want to fairly compare rotation action representation using rotation changes. The **unit rotation for lie algebra rotations and Euler angles is a zero-vector**. However, the unit rotation for quaternions and rotation matrices equals one or several unit vectors. In turn, if a network has to directly predict changes in quaternions or rotation matrices, then it has to first learn the unit rotation operation while for lie algebra rotations this is clearly not the case. In the paper (Table 2), the authors conclude that using changes in rotation is only sensible for the Lie Algebra rotation and Euler angles while predicting changes of quaternions and rotation matrices shows drastic drops in performance. I bet this is because of the aforementioned difference in the representation's respective unit rotation vectors. 

I also suspect that Lie Algebra changes in rotation could be particularly well suited for modelling rotation actions. However, if setup correctly, I am positive that modeling changes in rotation via quaternions and rotation matrices would not show such drastic decreases in performance as reported by the authors if setup "correctly". As the author's paper acts as a primer for the RL community, not carefully evaluating how changes in rotations need to be parameterized could lead to rushed conclusions.

To provide a fair comparison between different representations of changes in rotation, I suggest you **model changes in quaternions / rotation matrices by adding the network output to the unit  rotation and then project the resulting vector onto the rotation manifold**. For stochastic policies, you also could add the network mean to the unit rotation and then sample in Euclidean space around this mean.

The authors have now the following courses of action:

1) Make a convincing argument why my above assessment is flawed (e.g. you did exactly the approach I have described above). If convincing, I increase my rating to an accept.
2) Extend the discussion and experiments in Section 2 and 3 to consider the above proposed approach to predicting changes in Quaternions / Rotation matrices. If done thoroughly, I increase my rating to a strong accept.

### Questions
Please see Comment 1 above.

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies how the representation of rotation in actions affects RL training. The authors compare different types of SO(3) action representation in a range of RL environments, including an ideal rotation control environment and some more realistic robotic benchmarks. From the results, the authors recommend using tangent vectors for practitioners.

### Strengths
1. The SO(3) action representation is a commonly encountered problem by RL practitioners, especially in the field of robotic control. This work provides valuable guidance for the design of the action space and environments.

2. The impact of different rotation representations is studied in a diverse set of benchmarks, and all the hypotheses are accompanied by detailed analysis. They all make the conclusions convincing.

### Weaknesses
1. The considered scope of this work is somewhat limited. Only the simple MLP network + Gaussian noise as policy representation is studied. Other popular design choices, such as discretizing actions into bins and learning a categorical distribution over the discrete action space could also be investigated.

2. The insights/takeaways from experiments could be better presented. In Section 3.3, the authors state several hypotheses but the conclusions are not highlighted. Since the results are quite mixed for different representations, rewards and algorithms, I think the presentation of the analysis could be improved.

### Questions
Please refer to the weaknesses part.

### Soundness
3

### Presentation
2

### Contribution
3
