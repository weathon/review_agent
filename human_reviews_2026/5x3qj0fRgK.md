# Recurrent State Encoders for Efficient Neural Combinatorial Optimization

- Decision: Reject
- Scores: 2, 4, 4

## Abstract
The primary paradigm in Neural Combinatorial Optimization (NCO) are construction methods, where a neural network is trained to sequentially add one solution component at a time until a complete solution is constructed. We observe that the typical changes to the state between two steps are small, since usually only the node that gets added to the solution is removed from the state. An efficient model should be able to reuse computation done in prior steps. To that end, we propose to train a recurrent encoder that computes the state embeddings not only based on the state but also the embeddings of the prior state. We show that the recurrent encoder can achieve equivalent or better performance than a non-recurrent encoder even if it consists of $3\times$ fewer layers, thus significantly improving on latency. We demonstrate our findings on three different problems: the Traveling Salesman Problem (TSP), the Capacitated Vehicle Routing Problem (CVRP), and the Orienteering Problem (OP) and integrate the models into a large neighborhood search algorithm, to showcase the practical relevance of our findings.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a neural network architecture for combinatorial optimization. Its key contribution is a recurrent module that generates node embeddings not only from the current state but also from the embeddings produced in previous steps. The neural solver is trained to imitate the behavior of a conventional solver used to generate the training data. The authors further integrate the trained network into a neighborhood search metaheuristic, using it as an oracle to generate initial solutions and iteratively improve them. Computational results show that this integration achieves comparable or better solution quality with lower computation time on TSP and CVRP instances, although performance lags behind baseline methods on OP instances.

### Strengths
- Overall, the paper is well written, and the intuition for designing the recurrent module is clearly articulated.
    
- The computational study is also comprehensive, covering well-known combinatorial problems that are commonly used in related literature.

### Weaknesses
- **Absolute performance margin**. My main concern lies in the practical significance of the reported improvements over baseline methods. The authors claim a (relative) 1.8–2.8× speed-up in inference, which is impressive. However, as shown in Figure 1, the inference times for existing neural combinatorial solvers are already under one second for TSP and CVRP instances with up to 200 nodes. In most applications, such problems are not solved in real time, and even when they are, current methods appear sufficiently fast. The small absolute performance gain in latency therefore raises questions about the broader motivation for studying this problem.

- **Imitation learning instead of reinforcement learning.** In my opinion, the primary goal of a neural combinatorial optimization solver is to produce high-quality solutions rather than simply achieving faster inference, since neural approaches are already much faster than traditional optimization algorithms. To ensure strong solution quality, the mainstream approach in recent literature has been to use reinforcement learning, whereas imitation learning is inherently limited by the performance of the solver used to generate the training data (as the authors also acknowledge). The reliance on imitation learning in the current implementation makes the y-axis of all figures difficult to interpret. I recommend that the authors (1) evaluate the proposed architecture using reinforcement learning–based training methods, and (2) report the y-axis as the relative optimality gap to facilitate a clearer understanding of the solver’s performance.

- **Performance.** In Figure 1, the proposed method performs worse than the baseline on OP instances, which raises concerns about the robustness of the approach. I find the idea of using the trained model as an oracle within a large neighborhood search scheme interesting. However, the y-axis in Figure 2 is difficult to interpret, as it appears to be reported relative to a potentially weak baseline, and the observed improvement margins are quite small (for example, from 0.3\% to 0.1\%)

### Questions
- **Section 3.1.** I found the description regarding the path-TSP formulation in Section 3.1 a bit confusing. In particular, how do you reformulate an TSP as a sequence of shortest path problems?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper addresses inefficiencies in Neural Combinatorial Optimization (NCO), where traditional construction-based methods suffer from redundant computations—they fully recalculate state embeddings at each step despite minimal state changes (only one node is typically removed when adding a solution component). To solve this, this paper propose a recurrent encoder that enables lightweight updates: it reuses embeddings from the previous step, only adjusting for small state differences instead of recomputing from scratch.  
Experiments on the Traveling Salesman Problem (TSP), Capacitated Vehicle Routing Problem (CVRP), and Orienteering Problem (OP) show the recurrent encoder outperforms or matches non-recurrent encoders (with 3× fewer layers) in solution quality, while cutting latency by 1.8–4×. It also exhibits strong robustness and practical value—integrating it into a Large Neighborhood Search (LNS) algorithm improves both efficiency and solution quality for large-scale problems.

### Strengths
I think this paper, by combining the base encoder and the recurrent encoder, attempts to solve the problem of redundant computations in the model training and inference process, which is interesting and has novelty.
1. Proposes a recurrent encoder that reuses prior step computations and only updates for small state changes, significantly reducing latency (1.8–4× faster) compared to non-recurrent encoders.
2. Demonstrates strong robustness—maintains stable performance even when the key hyperparameter k (recurrent update frequency) is much larger than the trained value.
3. Achieves comparable or better solution quality than non-recurrent counterparts while using 3× fewer layers, enabling model lightweighting.
4. Proves practical value by integrating with the Large Neighborhood Search (LNS) algorithm, improving both efficiency and solution quality.

### Weaknesses
1. Lacks a diagram that depicts the entire system architecture, which would help readers quickly grasp the key innovations.
2. Relies on imitation learning trained with expert trajectories, which depend on high-quality solutions from classical solvers and lack the ability to autonomously explore better strategies like reinforcement learning.
3. Requires periodic calibration via the base encoder (controlled by hyperparameter k), introducing additional tuning overhead.
4. Has not been tested on a wider range of combinatorial optimization problems (e.g., BPP, PSSP), making its generalizability across different problem types unproven.
5. The elements contained in Figures 1 and 2 in this paper are too numerous, and it takes a long time to understand them.
6. Cann't download opensource files from https://anonymous.4open.science/r/2CB0.

### Questions
In practical applications, the accuracy of the algorithm is undoubtedly the top priority. If the accuracy deteriorates, the practicality will be greatly affected. The paper mentions that using a recurrent encoder will accumulate errors, so it is necessary to periodically use the base encoder to recalculate. Moreover, the time and memory consumption of these baseline models compared in the paper are acceptable, so I think the motivation for accelerating with the recurrent encoder is not very solid.  
Maybe the author could attempt to theoretically explain the benefits of using the concurrent encoder for the final model performance, rather than merely from the perspective of acceleration.   
And, although in the experimental results, the methods presented in this paper outperform the baseline methods in some results, but the comparison is not very fair and incomplete (for example, POMO did not use reinforcement learning, only comparing scenarios with a problem size of 100 for training and other sizes for testing, and the test dataset did not use popular datasets such as TSPlib and CVRPlib).

### Soundness
2

### Presentation
2

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
The paper proposes Recurrent State Encoders (RSE) for Neural Combinatorial Optimization (NCO), aiming to improve inference efficiency by leveraging temporal similarity between consecutive partial solutions.Instead of re-encoding the entire problem state at every construction step, RSE incrementally updates node embeddings using a recurrent encoder that combines the previous embeddings with the new partial solution features. A hyperparameter k controls how frequently the model performs full re-encoding.The method is evaluated on TSP, CVRP, and Orienteering Problem (OP) tasks, showing similar or slightly improved solution quality compared to baseline NCO models, while achieving up to 4× faster inference and maintaining robustness when integrated into a Large Neighborhood Search(LNS) framework.

### Strengths
1. The proposal to incrementally update embeddings using recurrent state memory is conceptually sound and well aligned with practical latency constraints in NCO applications.
2. Experiments span three canonical COPs (TSP, CVRP, OP) and multiple scales up to 1,000 nodes, comparing against a wide range of competitive baselines (e.g., BQ-NCO, LEHD, GLOP, POMO).

### Weaknesses
1. The proposed method is essentially an application of RNN-style computation reuse to NCO encoders, offering incremental rather than conceptual innovation.
2. No discussion on embedding stability, convergence, or representational drift during recurrent updates.
3. Models are trained solely via imitation learning; the effect of RL or self-improvement paradigms is not explored experimentally.
4. The paper reports performance improvements but offers little analysis of the underlying mechanisms driving them.

### Questions
1. How do the authors ensure that the recurrently updated embeddings remain stable and do not accumulate drift or noise over long decoding horizons, especially for large k or OOD instances?
2. Why are self-improvement models (e.g., Luo et al. 2024, Pirnay & Grimm 2024b) not compared directly, given that they represent the current state of the art in NCO efficiency?
3. Compared to a completely re-encoded transformer, does the representational power of recursive updates decrease? Is there any theoretical or experimental verification of the representational equivalence between the two?

### Soundness
3

### Presentation
3

### Contribution
2
