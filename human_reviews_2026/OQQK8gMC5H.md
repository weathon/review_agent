# DOPPLER: Dual-Policy Learning for Device Assignment in Asynchronous Dataflow Graphs

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 2

## Abstract
We study the problem of assigning operations in a dataflow graph to devices to minimize execution time in a work-conserving system, with emphasis on complex machine learning workloads. Prior learning-based approaches face three limitations: (1) reliance on bulk-synchronous frameworks that under-utilize devices, (2) learning a single placement policy without modeling the system dynamics, and (3) depending solely on reinforcement learning during pre-training while ignoring optimization during deployment. We propose Doppler, a three-stage framework with two policies—$\mathsf{SEL}$ for selecting operations and $\mathsf{PLC}$ for placing them on devices. Doppler consistently outperforms baselines by reducing execution time and improving sampling efficiency through faster per-episode training. Our results show that Doppler achieves up to 52.7\% lower execution times than the best baseline. The code is available at https://github.com/xinyuyao/Doppler.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper shows DOPPLER, a dual-policy reinforcement learning framework for optimizing device assignment in asynchronous. 
Traditional synchronous scheduling enforces global barriers and that leads to idle GPUs. 
The presented method addresses device placement in an asynchronous dataflow environment where operations can be executed dynamically. The framework does a three-stage pipeline: (1) imitation learning using heuristic supervision, (2) reinforcement learning in simulation, and (3) online fine-tuning during real deployment. Experimental results on dataflow graphs from matrix chains, feedforward networks, and Llama layers show up to 52.7% reduction in execution time relative to state-of-the-art baselines. The method also generalizes to new hardware and architectures with few-shot adaptation.

### Strengths
The strengths are:

The paper is well written and organized. It also shows ablation studies and pseudocode.

The work addresses device placement under asynchronous execution where small per-query improvements scale to substantial cost reductions at deployment scale. A dual-policy decomposition separates operation selection and device placement. Then they introduce a three-stage training pipeline that bridges imitation learning, simulation-based RL, and real-system adaptation.

The framework’s shows transferability and few-shot adaptability to new hardware that is relevant for real-world heterogeneous distributed inference and training systems.

### Weaknesses
The weaknesses are:

The paper is missing explanation or studies for convergence or sample complexity analysis of the dual-policy setup. Adding those could strengthen the understanding of stability during online adaptation. 

The experiments focus on up to 8 GPUs. It remains unclear how DOPPLER scales to large GPU clusters (256 GPUs) or heterogeneous systems with varying compute/memory characteristics.

The paper presents a 3 stage training, which seems rather complex. The incremental benefit of Stage III appears limited, and simpler two-stage alternatives could potentially achieve similar performance with reduced engineering overhead. The paper mentions a simulator used in Stage II but offers limited quantitative validation of how closely its performance correlates with real hardware results. This impacts the credibility of simulation-based pretraining.

### Questions
A few questions about the paper:

In the Stage III online phase, how is policy stability ensured when runtime-derived rewards could be noisy, is there a mechanism to prevent noisy updates?

What is the observed correlation between simulated runtime rewards and real execution times? This could validate the Stage II training phase and strengthen the paper

Would integrating model-parallel or operator-fusion heuristics into DOPPLER’s action space further improve placement quality?

### Soundness
2

### Presentation
3

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
### **Review Summary**

This paper proposes DOPPLER, a reinforcement learning (RL) framework for the device placement of dataflow graphs in an asynchronous, "work-conserving" (WC) system. The authors argue that existing bulk-synchronous systems under-utilise resources and that prior RL solutions use a single, inflexible placement policy. DOPPLER's core contributions are a **dual-policy architecture**—a selection policy (SEL) to choose the next node and a placement policy (PLC) to assign it—and a **three-stage training process** (Imitation Learning, RL in simulation, and RL on the real system). The method is shown to outperform heuristic and RL baselines in execution time across several ML workloads.

While the problem is highly relevant and the empirical results are strong, the paper's core contributions are not well-justified. The necessity of the dual-policy (SEL+PLC) over a single placement policy is poorly motivated and its benefits are conflated with other implementation choices. Furthermore, the 3-stage training pipeline relies on an online, real-world RL stage, which is impractical for the very production environments the paper targets, undermining the "cost-effective" claims.

### Strengths
### **Strengths**

1.  **Important Problem:** The paper correctly identifies a key inefficiency in modern ML systems: bulk-synchronous execution (e.g., all-reduce) leads to device idle time [cite: 69-72]. It convincingly argues that asynchronous, work-conserving (WC) systems offer significant speedups (Table 1) [cite: 161-163], but that optimizing for them is a more complex temporal problem [cite: 169-171].

2.  **Comprehensive Training Framework:** The three-stage training paradigm (Stage I: Imitation Learning, Stage II: Simulation-based RL, Stage III: Real-system RL) is a thorough approach [cite: 177-180, 487]. Figure 3 demonstrates that this curriculum is highly effective, using imitation and simulation for a strong warm start before the final, high-performance real-system tuning [cite: 731-734].

3.  **Strong Empirical Performance:** The final model, DOPPLER-SYS (which uses all 3 stages), demonstrates state-of-the-art performance. It achieves significant execution time reductions (up to 52.7%) compared to all baselines, including prior RL methods (PLACETO, GDP) and a strong, purpose-built heuristic (ENUMERATIVE OPTIMIZER) [cite: 619-621, 642-644].

4.  **Pragmatic Scalability:** The paper includes a critical implementation detail: running GNN message passing only *once per episode* (per graph) rather than *once per step* (per node) [cite: 481-485]. Appendix G.3 correctly identifies this as a massive efficiency gain (a 30x reduction in message-passing operations for a <1% performance trade-off), making the approach scalable and computationally viable [cite: 2331-2334, 2399-2400].

### Weaknesses
### **Weaknesses and Questions**

1.  **Weak Motivation for Dual Policy:** The paper's central contribution is the "dual-policy" (SEL+PLC) architecture [cite: 61, 174-176, 182]. However, the *reason* for factoring the policy this way is poorly justified. The paper never compares this two-policy agent against the most obvious and standard baseline: a *single* policy that directly outputs a `(node, device)` pair. The ablation in Table 3 [cite: 646-649] only compares DOPPLER to variants where one of the two policies is replaced by a heuristic; this does not prove that the dual-policy factorization itself is beneficial.

2.  **Misleading Scalability Claims:** The paper's impressive scaling results (Fig. 4) [cite: 737-739] are presented as a benefit of DOPPLER. However, the text (and Appendix G.3) reveals this speedup is due to an implementation choice (GNN-per-episode) that is *orthogonal* to the dual-policy architecture [cite: 481-485, 2331-2400]. The paper criticizes PLACETO for being "prohibitive" and "inefficient" because it runs the GNN per-step [cite: 482-483], but this is an apples-to-oranges comparison. A fair comparison would require re-implementing all baselines with the same "GNN-per-episode" optimization. As such, the scalability results do not support the *dual-policy architecture* itself.

3.  **Impracticality of Stage III (Real System RL):** The paper's best results rely on `DOPPLER-SYS`, which uses Stage III, an online RL stage on the "real hardware" [cite: 180, 552-554]. The paper claims this is "for free"[cite: 554], which is highly misleading. This stage requires deploying an *exploratory* (i.e., non-optimal) policy in a production environment, forcing users to suffer through the high-variance, low-performance assignments generated during exploration. Figure 3 ("0k-0k-8k" curve) shows exactly how slow and unstable this "from-scratch" online learning is [cite: 732-734]. This reliance on an impractical training step undermines the paper's applicability.

4.  **Simulator-Reality Gap:** The only reason the impractical Stage III is necessary is that the simulator in Stage II is not accurate enough. This is shown by the consistent performance gap between `DOPPLER-SIM` (sim-trained) and `DOPPLER-SYS` (real-trained) in Table 2[cite: 619, 643]. The paper explicitly confirms this gap in Appendix G.1, stating the simulator "tends to overestimate" and has "trouble differentiating between assignments" [cite: 2220-2221]. The 3-stage framework is thus not just a feature but a *crutch* to overcome a flawed simulator.

### Questions
none

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The authors introduce DOPPLER, a method to efficiently assign computational operations in dataflow graphs to GPU devices in work-conserving (asynchronous) systems. The key innovation is a dual-policy learning framework that separates the problem into selecting which operation to assign next and determining which device should execute it. DOPPLER employs a three-stage training approach combining imitation learning, simulation-based RL, and continuous real-system RL during deployment, to achieve significant execution time reductions over existing baselines using clever design choices to ensure both good performance and scalability.

### Strengths
- Very significant practical impact in performance gains over best baselines towards the dataflow graph assignment problem.
- Well clever design choices to enable both scalable training and improved performance 
	- Pretraining techniques to improve performance during the actual RL phase.
	- Clever message passing implementation to reduce training time with very negligible performance impact
- Strong generalization across hardware architectures

### Weaknesses
- Evaluation seems limited to relatively small graphs (~200 nodes). Real life workloads could be much bigger that could affect the linear scalability claim. In my own personal experience, scheduling larger graphs sometimes introduces additional complexity that sometimes is not captured on smaller graphs
- Dynamic Dataflow - I could imagine that dataflow may change dynamically during execution - especially with larger computational graphs. Is this a setting your algorithm can handle because from the writing this doesn't seem to be considered.

### Questions
- How does this work relate to the substantial body of research on learning-augmented scheduling? While your setting differs (multi GPU dataflow graphs vs. traditional job scheduling), the core problem structure appears similar:
	- Both involve sequential assignment decisions under uncertainty
	- Both use predictions/learning to schedule DAGs
	- Both must balance load and minimize communication/movement costs
	The learning augmented scheduling literature has developed concepts like consistency (performance with accurate predictions), robustness (worst case guarantees), and prediction error bounds. Could these frameworks provide theoretical grounding for DOPPLER? For example, can you characterize DOPPLER's consistency and robustness formally, or provide approximation guarantees based on simulator accuracy?
- How does your work scale to larger graphs? Is this a realistic setting for dataflow? And does your linear scalability still hold?
- How does this work for dynamic graphs (that is when the graph changes dynamically) - is this a possible setting for your problem?
- Why is an RL method needed for this problem versus approaches like Bayesian Optimization or other direct optimization methods?
- Hyperparameter selection? To deploy your algorithm in practice for heterogeneous workloads and hardware, how should one set hyperparameters?
- Why those specific neural network architectures? This is not a criticism - rather I'm seeking to understand why you made those design choices.

If some of my more important questions are addressed, I'm very inclined to increase my score. I think this will be a very impactful addition to the community speed up compute especially for ML jobs.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This study developed a reinforcement learning based approach (DOPPLER) to enable efficient assign operations in a dataflow graph. Experiments shows that the trained RL model could reduce the execution time in neural network operations with different network architectures.

### Strengths
The problem has a clear motivation, illustrated by a dataflow graph, in the introduction section for matrix multiplication in deep learning operations.

### Weaknesses
#1 It is not easy for readers to follow the logic of this paper. The paper fails to clearly describe the problem and methods.

#2 The abstract is too short to demonstrate the significance of DOPPLER. Basically, there are no quantitative results in abstract

#3 “Simulator” for system execution was used through the paper without any description. Please clarify. 

#4 Since section 3 was titled as problem definition, is section 2 just a background? More specifically, is Algorithm 1 an existing algorithm or the problem you aim to study in this paper? No citations were provided in Section 2.

#5 The paper does not provide sufficient technical details but rather forced everything to the appendix.

#6 GNN is too broad. What specific architecture did you use? Clarification and citation are needed in the main text.

#7 The rationale of three-stage training was not well justified. Any challenges or preliminary experiments that could support your proposed training mechanism?

#8 What reinforcement learning algorithm did you use?

#9 How does “RL for combinatorial optimization” in the related work section contribute to underpin the background of your work? E.g., you mentioned many works for TSP but does it help here?

#10 The “combinatorial structure” in line 197 was not clearly introduced in the main text.

### Questions
Please see weakness section.

### Soundness
1

### Presentation
1

### Contribution
2
