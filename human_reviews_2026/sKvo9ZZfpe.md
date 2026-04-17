# RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization

- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
The practical deployment of Neural Combinatorial Optimization (NCO) for Vehicle Routing Problems (VRPs) is hindered by a critical sim-to-real gap. This gap stems not only from training on oversimplified Euclidean data but also from node-based architectures incapable of handling the node-and-edge-based features with correlated asymmetric cost matrices, such as those for real-world distance and duration. We introduce RRNCO, a novel architecture specifically designed to address these complexities. RRNCO's novelty lies in two key innovations. First, its Adaptive Node Embedding (ANE) efficiently fuses spatial coordinates with real-world distance features using a learned contextual gating mechanism. Second, its Neural Adaptive Bias (NAB) is the first mechanism to jointly model asymmetric distance, duration, and directional angles, enabling it to capture complex, realistic routing constraints. Moreover, we introduce a new  VRP benchmark grounded in real-world data crucial for bridging this sim-to-real gap,  featuring asymmetric distance and duration matrices from 100 diverse cities, enabling the training and validation of NCO solvers on tasks that are more representative of practical settings. Experiments demonstrate that RRNCO achieves state-of-the-art performance on this benchmark, significantly advancing the practical applicability of neural solvers for real-world logistics. Our code, dataset, and pretrained models are available at https://github.com/ai4co/real-routing-nco.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates the application of neural combinatorial optimization solvers to real-world problem instances. Specifically, the authors use a adaptive attention-free module from prior work to incorporate real-world travel time and distance features, in contrast to existing studies that rely on synthetic data with Euclidean distances. They further compile a comprehensive dataset covering problem instances from 100 cities, which serves as a testbed for evaluating the proposed method against established baselines.

### Strengths
I personally found this paper very exciting, as it presents (to the best of my knowledge) the first large-scale real-world dataset for benchmarking neural combinatorial solvers. This represents a valuable step forward, and the dataset has the potential to benefit many researchers in the community.

### Weaknesses
I do not have much to complain about this paper, below is a question that I am curious about. 

- Can the adaptive attention-free module handle features other than travel time and distances? What about contextual features that are shared among node pairs, for example, time of the day and weather?

### Questions
See above

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces RRNCO, a new neural combinatorial optimization (NCO) architecture targeting practical real-world vehicle routing problems (VRP). The paper also presents a new and comprehensive VRP dataset based on OpenStreetMap data from 100 cities, enabling both in-distribution and out-of-distribution evaluation.

### Strengths
1. The paper introduces a large, scalable, and diverse real-world VRP dataset that captures the essential complexities (asymmetry, real-world travel times) often ignored in current research. Making this dataset open-source will prove highly valuable to the research community.

2. The proposed ANE and NAB mechanisms are well-motivated, addressing the need to handle node-and-edge features natively, especially asymmetric matrices, which are more representative in real logistics.

3. This paper is clearly written.

### Weaknesses
1. Problem Definition: The definition of “real-world VRP” in the paper appears to focus almost exclusively on the presence of asymmetric cost matrices (distance, duration). While this is a major source of realism, real-world VRPs are also characterized by other features such as dynamic distance changes, rich operational constraints (e.g., skills, priorities, real-time updates), time-dependent travel times, etc. The current problem definition could be more comprehensive and better justified in context.

2. Comparison to OR-Tools in ATSP: In the ATSP experiments, the baseline comparisons do not include Google OR-Tools.

3. Computation Time Reporting: For some baselines (e.g., OR-Tools takes “several hours”), the exact computation protocol is unclear. 

4. Scope of Generalization: Experiments are done on the proposed real-world dataset with varied geographies, but it’s not clear if the model's performance holds for even more complex cases. Better to show results on estinblished dataset or benchmarks as well.

### Questions
1. On the Definition of "Real-World": Besides asymmetry, what other real-world features (as per logistics practice) does your approach not cover? How might your model adapt to additional complexities such as time-dependent or dynamic edge lengths?

2. On ATSP Baselines: Why was OR-Tools not included in the ATSP comparison, given that it supports ATSP solutions? Are there technical limitations, or was this a deliberate choice? Please elaborate.


3. On OR-Tools Computation Time: The reported computation time for OR-Tools on VRP benchmarks is shown as “7h” or “several hours.” How was this measured?

4. On Generalization to Other VRP Benchmarks: How easily can your RRNCO be adapted to incorporate other benchmarks?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the critical sim-to-real gap for Neural Combinatorial Optimization (NCO) in Vehicle Routing Problems (VRPs), which stems from oversimplified symmetric data and inadequate node-centric architectures. The authors make a dual contribution: 1)  introducing an Adaptive Node Embedding (ANE) and a Neural Adaptive Bias (NAB) to jointly model real-world asymmetric distance, duration, and angles, and 2) a new, open-source VRP benchmark with asymmetric data from 100 real-world cities. Experiments show RRNCO achieves state-of-the-art performance among NCO solvers on this more realistic benchmark.

### Strengths
1. This paper provides an open-source benchmark dataset. This is a major service to the NCO community, enabling the training and validation of NCO solvers on tasks that are more representative of practical settings.

2. The Adaptive Node Embedding (ANE) and Neural Adaptive Bias (NAB)  are effective for fusing multiple asymmetric edge features.

3. The authors compare against a wide range of strong baselines (both traditional and NCO). The evaluation on in-distribution, OOD-city, and OOD-cluster scenarios provides robust support for the authors' claims.

### Weaknesses
1. The NAB is a core contribution that models distance, duration, and an "angle matrix".However, the main paper does not define how this angle matrix is computed.

2. The paper fails to specify any of the runtime parameters, configurations, or computational budgets (e.g., time limits) used for these traditional solvers.

3. For dynamic features $D^t$ (defined in Appendix A), it never specifies what these features are for each problem type (ATSP, ACVRP, ACVRPTW). This is a critical missing detail for implementation.

4. In Decoder, the formulation in Line 710 only uses the last node's embedding, this is a significant deviation from standard NCO practice for problems like ATSP (which also use the first node's embedding for context). The paper provides no justification for this non-standard design, making it unclear if it's an intentional choice or an omission.

5. The study does not include a granular ablation on the inputs to NAB. Consequently, it is impossible to determine if the performance gain comes from the novel combination of all three features (D+T+$\Phi$), or if it is overwhelmingly driven by just the distance matrix (D) alone.

6. RRNCO's encoder is based on the AAFM from [1]  and its decoder on [2]. However, these two highly relevant works are not included as baselines in the comparison.

7. The overall quality of this paper is good, but its clarity would benefit from a final proofreading pass to correct some minor typos and inconsistencies: 
- In Line 260,  "adaptation" is spelled as "adaption" ; 
- The attention layers are referred to as "AAFM" in the text but "AFT Layers" in Table 4; 
- The "clipping hyperparameter $C$" from Equation 20 is not defined in the hyperparameter list (Table 4);
- Equation 10 is defined twice (lines 265 and 295);

### Questions
1. How is the "angle matrix" $\Phi$, a key input to the NAB (Section 4.1.2), defined and computed?

2. For the classical solver runtimes (LKH3, PyVRP) in Table 1, what was the CPU configuration? Did they utilize multi-core parallel processing, and if so, how many cores were used?

3. Figure 2 is ambiguous. Please clarify the encoder architecture: (1) Is my understanding correct that ANE is a one-time initial embedding step, not part of the attention stack? If so, I suggest revising Figure 2, Figure 2 ambiguously depicts ANE as part of the encoder stack and also omits the multi-layer stacking logic. (2) Why does the encoder diagram omit the Feed-Forward Network (FFN), while the decoder includes one? Does the encoder not require an FFN?

4. In Figure 4, when the NAB is removed, does the model revert to the original heuristic bias, or does it use a different attention mechanism entirely, such as a standard Multi-Head Attention (MHA) block?

References:

[1] Instance-conditioned adaptation for large-scale generalization of neural combinatorial optimization. arXiv preprint arXiv:2405.01906, 2024.

[2] Rethinking Light Decoder-based Solvers for Vehicle Routing Problems. ICLR, 2025.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes RRNCO, a neural combinatorial optimization architecture aimed to handle the asymmetry in vehicle routing problems. The contributions are twofold: (1) the Adaptive Node Embedding (ANE), which integrates spatial coordinates and asymmetric distance information through a contextual gating mechanism, and (2) the Neural Adaptive Bias (NAB), which jointly models distance, duration, and directional angle features. The authors also construct a new benchmark dataset with asymmetric real-world routing data from OpenStreetMap. Experiments on ATSP, ACVRP, and ACVRPTW demonstrate that RRNCO outperforms prior neural and classical solvers.

### Strengths
1. Clear motivation and relevance: the paper addresses a significant practical limitation of current NCO research: the reliance on symmetric, synthetic datasets. The focus on the sim-to-real gap is well-motivated and meaningful for practical deployment.
2. Valuable benchmark contribution: the creation of an asymmetric VRP dataset derived from 100 OpenStreetMap cities represents a substantial and reusable resource for future research. It enhances reproducibility and encourages broader progress in realistic NCO studies.

### Weaknesses
1. Limited theoretical novelty: both ANE and NAB are constructed from existing attention and gating mechanisms. While effective, they represent architectural refinements rather than fundamentally new theoretical concepts.
2. Insufficient evaluation for real-world robustness: the sim-to-real gap also arises from dynamic traffic conditions that alter travel times in real operations. The proposed model does not evaluate robustness under such time-dependent or stochastic variations. When congestion occurs after route generation, the model would likely suffer from the same limitations as prior works.

### Questions
1. Are there any theoretical analysis about of RRNCO (e.g., learnability, convergence, or generalization bound)?
2. Is it possible to evaluate the robustness of RRNCO with respect to the time-dependent or stochastic variations of traffic conditions?

### Soundness
2

### Presentation
2

### Contribution
2
