# RATE-DISTORTION OPTIMIZED PRAGMATIC COMMUNICATION FOR COLLABORATIVE PERCEPTION

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Collaborative perception emphasizes enhancing environmental understanding by enabling multiple agents to share visual information with limited bandwidth resources. While prior work has explored the empirical trade-off between task performance and communication volume, a significant gap remains in the theoretical foundation. To fill this gap, we draw on information theory and introduce a pragmatic rate-distortion theory for multi-agent collaboration, specifically formulated to analyze performance-communication trade-off in goal-oriented multi-agent systems. This theory concretizes two key conditions for designing optimal communication strategies: supplying pragmatically relevant information and transmitting redundancy-less messages. Guided by these two conditions, we propose RDcomm,
a communication-efficient collaborative perception framework that introduces two key innovations: i) task entropy discrete coding, which assigns features with task-relevant codeword-lengths to maximize the efficiency in supplying pragmatic information; ii) mutual-information-driven message selection, which utilizes mutual information neural estimation to approach the optimal redundancy-less condition. Experiments on 3D detection and BEV segmentation show that RDcomm achieves state-of-the-art accuracy on datasets DAIR-V2X, OPV2V, V2XSeq, and V2V4Real, while reducing communication volume by up to 108×. Our code is available at
https://github.com/gjliu9/RDcomm.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper tackles how to maximize downstream task performance under bandwidth constraints. The authors develop a Pragmatic Rate–Distortion Theory, deriving two core conditions—pragmatic relevance and redundancy minimization—to guide the trade-off between task performance and communication volume. Building on this, they propose RDcomm, which achieves SOTA results in both performance and communication efficiency.

### Strengths
- The work tightly links collaborative communication objectives to downstream task risk, derives two actionable conditions, and uses them to design the modules—enabling a quantitative analysis of the performance–communication trade-off in multi-agent systems.
- The overall logic is smooth. The implementation is concise and easy to apply.
- The empirical results substantiate the theoretical rationale and validate the effectiveness of the proposed method across extensive experiments.

### Weaknesses
- The equality conditions require $H(Z \mid Y) = 0、I(Z; X_r) = 0$. The former implies the message is deterministic given the task, and the latter requires complete independence from the receiver’s observation—both are hard to satisfy in practical perception. The paper lacks a metric or empirical assessment of how closely these conditions are met.
- The paper devotes little attention to the expansion and coordination of multi-agent systems. The method centers on removing redundancy, with insufficient characterization of synergistic gains among three or more agents.

### Questions
- In equation (9), if $\delta > I(Y;X_s \mid X_r)$, the right-hand side becomes negative. The paper does not clarify a truncation or other handling for this case.
- Equation (10) interprets $I(Y;X_{s};X_[r])$ as “redundancy with respect to (Y)”, but interaction information can be positive or negative; negative values often indicate synergy rather than redundancy. Related discussion is missing.
- The receiver employs UNet for smoothing/dilation, which changes the spatial support of effective information, which seems at odds with the premise of strictly controlling redundancy.

### Suggestions
In multi-agent collaboration, explicitly distinguish redundancy vs. synergy, and explore selection/coding mechanisms that leverage synergy rather than only pruning redundancy. Also report how complexity scales with the number of agents.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes RDcomm, a communication-efficient collaborative perception framework underpinned by a pragmatic rate-distortion theory. This theory posits that optimal communication strategies should satisfy two key conditions: (i) transmitting only pragmatically relevant information for the task; and (ii) avoiding the transmission of inter-agent redundant messages. Guided by these principles, RDcomm introduces two core innovations: (i) task entropy discrete coding, which assigns variable-length codewords to features based on their task relevance to enhance encoding efficiency; and (ii) mutual-information-driven message selection, which leverages neural mutual information estimation to eliminate redundant content. Experiments on 3D object detection and BEV semantic segmentation tasks demonstrate that RDcomm achieves state-of-the-art perception performance, while reducing communication volume by up to 108x.

### Strengths
1. Solid Theoretical Foundation: The work introduces a pragmatic rate-distortion theory specifically for multi-agent collaboration, providing a robust theoretical basis for task-oriented communication design.

2. Clear Design Motivation: Task entropy coding and mutual information-driven selection directly address the theoretical conditions of "pragmatic relevance" and "redundancy-less communication," indicating a well-motivated design.

3. Dual Superiority in Performance and Communication Efficiency: RDcomm achieves leading performance in both communication efficiency and perception accuracy across various modalities (LiDAR/camera) and multiple tasks.

### Weaknesses
1. Real-Time Latency in Two-Round Communication: RDcomm adopts a two-round communication mechanism (first sending a coarse abstract, then the receiver evaluates redundancy, and finally refined messages are sent). Although the abstract message constitutes only about 10% of the total communication volume, this mechanism introduces end-to-end latency and incurs synchronization and coordination overhead in real-time sensitive scenarios like vehicle-to-everything (V2X) communication. The paper lacks an analysis of the actual impact of this latency on perception performance, which is crucial for real-world deployment.

2. Robustness of Mutual Information Estimation: The method relies on a GAN-style neural mutual information estimator to determine message redundancy. However, mutual information estimation in high-dimensional feature spaces is susceptible to sampling bias and training instability. The paper lacks a quantitative analysis of how estimation errors affect the quality of message selection (e.g., a comparison with different estimators, correlation between estimation error and task performance degradation), which weakens the guarantee of robustness in complex real-world scenarios.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper investigates the trade-off between task performance and communication volume for collaborative perception from an information-theoretic perspective. They formulate a pragmatic rate–distortion theory, deriving the optimal bit rate for message transmission and two necessary conditions for optimal compression: pragmatic-relevant and redundancy-less. They propose task entropy discrete coding and mutual-information-driven message selection for communication-efficient collaborative perception.

### Strengths
1. This paper provides a theoretical analysis of the trade-off between task performance and communication volume in collaborative perception.
2. This paper contains very detailed formula derivations and proofs.

### Weaknesses
1. Experiments were conducted only on two relatively early datasets, lacking validation on the latest datasets such as V2X-Seq and V2V4Real.
2. The training process of the entire method is complex and requires three independent stages of training.
3. There is a lack of analysis and comparison regarding dimensions critical for practical deployment, such as model inference time and parameter count.
4. The experimental section lacks comprehensive comparative methods, particularly omitting some recent communication-efficient collaborative perception approaches.
5. The overall presentation of the paper is not very reader-friendly. It is recommended to add some figures to enhance the clarity of the explanations.

### Questions
1. Can the method proposed in the paper be generalized to other collaborative perception scenarios, such as drone swarms?
2. Is the proposed model robust to temporal asynchrony, spatial misalignment, and noise interference among multiple agents?
3. Is there further analysis and ablation on the thresholds and hyperparameters involved in the paper?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a theoretically grounded framework for optimizing communication efficiency in multi-agent collaborative perception. It introduces a pragmatic rate–distortion theory that formalizes the trade-off between task performance and communication bandwidth in collaborative systems such as autonomous vehicles or multi-robot teams.

### Strengths
S1: This paper provides a theoretical foundation by extending rate–distortion theory to multi-agent collaboration with clear optimality conditions.

S2: I like the idea of proposed in task entropy discrete encoding which enables variable length of codes.

### Weaknesses
W1: The mutual-information–based selection shares similar ideas with “What Makes Good Collaborative Views? Contrastive Mutual Information Maximization for Multi-Agent Perception [AAAI 2024].” I strongly recommend that the authors clarify the conceptual and methodological differences, and maybe justify the advantages through experiments.

### Questions
Please see the above weakness and that's my biggest concern of novelty. I will raise my score if my concerns are well addressed.

### Soundness
3

### Presentation
3

### Contribution
2
