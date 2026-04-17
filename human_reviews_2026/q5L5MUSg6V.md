# Bridging Efficiency and Safety: Formal Verification of Neural Networks with Early Exits

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Ensuring the safety and efficiency of AI systems is a central goal of modern research. Formal verification provides guarantees of neural network robustness, while early exits improve inference efficiency by enabling intermediate predictions. Yet verifying networks with early exits introduces new challenges due to their conditional execution paths. 
In this work, we define a robustness property tailored to early exit architectures and show how off-the-shelf solvers can be used to assess it. 
We present a baseline algorithm, enhanced with an early stopping strategy and heuristic optimizations that maintain soundness and completeness. 
Experiments on multiple benchmarks validate our framework’s effectiveness and demonstrate the performance gains of the improved algorithm. 
Alongside the natural inference acceleration provided by early exits, we show that they also enhance verifiability, enabling more queries to be solved in less time compared to standard networks.
Together with a robustness analysis, we show how these metrics can help users navigate the inherent trade-off between accuracy and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper formalizes local robustness for networks with early exits (EEs) by redefining the robustness counterexample to account for conditional execution and multiple potential output layers. It proposes: (i) a baseline verification procedure (Alg. 1) that mirrors EE inference logic and is sound and complete given a sound/complete backend, and (ii) an optimized procedure (Alg. 2) with “break/continue” checks that can certify safety earlier and skip many inner-loop queries. The authors also give a trace-stability notion and show fixed-parameter tractability under this assumption (FPT in layer width × length of the stable trace). Experiments on MNIST/CIFAR (FC, LeNet, ResNet-18, VGG-16) show large runtime gains for SAFE cases (up to ~10×) and that adding EEs to standard models can make them easier to verify without large accuracy loss; they also analyze threshold/exit-placement trade-offs.

### Strengths
1. Precise redefinition of robustness with EEs and proofs of soundness/completeness for both algorithms; FPT analysis under trace stability gives theoretical insight. 


2. Alg. 2’s early break and continue checks skip redundant exit evaluations once safety or unsafety is determined, greatly speeding up SAFE certifications (e.g., VGG-16 cases that previously timed out).

3.  Adding EEs often accelerates verification (particularly SAFE instances) and exit placement/thresholds provide a tunable accuracy-latency-verifiability trade-off.

### Weaknesses
1. Because α/β-CROWN lacked nested AND encoding, queries were decomposed into multiple conjunctive calls; this may distort absolute runtimes and comparability to “vanilla” queries. A native-AND backend (e.g., Marabou) could change results. 


2. Scope restricted to local robustness and ReLU. Theory and claims lean on ReLU and local robustness; no formal handling of other properties (e.g., fairness/safety specs) or activations beyond brief discussion. 



3. Runs are on a single M3 machine with modest samples; many UNKNOWN arise from timeouts/loose bounds. More large-scale, multi-backend evidence (and variance/CI reporting) would strengthen the case. 


4. The “break/continue” conditions are intuitive but largely justified empirically; a tighter analysis of when these checks dominate (beyond the trace-stability assumption) would help.

### Questions
1. A recent paper, “Learning Minimal Neural Specifications” (Geng et al., NeuS 2025), investigates neural activation patterns (NAPs) and shows that only a small subset of neurons is often sufficient to characterize a model’s robust behavior. Your early-exit mechanism seems conceptually related—it also leverages intermediate representations that can certify outputs earlier. Do you see these two phenomena as reflecting the same underlying principle of redundant or concentrated information flow in deep networks? Could you discuss the connections to Geng et al.?

2. Beyond local robustness: How would you instantiate your framework to other properties (e.g., safety constraints or fairness predicates) within the EE setting? What changes are needed in the property encoding and in your early checks?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper explores how formal verification techniques can be extended to neural networks with early exits. The authors propose a robustness definition that accounts for conditional execution paths and introduce two verification algorithms: a baseline and an optimized version with early-stopping heuristics. They provide proofs of soundness and completeness and evaluate their approach on MNIST, CIFAR-10, and CIFAR-100 using various architectures.

### Strengths
* Tackles an underexplored intersection between neural networks with early exits and formal verification.
* Theoretical development, proofs, and algorithms are clear and internally consistent.
* Experiments are well executed across multiple datasets and architectures, supporting the main claims. 
* The presentation is clean and well structured, making it easy to follow the core ideas.

### Weaknesses
* The novelty is somewhat limited. The work mainly adapts existing verification methods to networks with conditional exits, which, while useful, feels like an incremental extension rather than a conceptual breakthrough.
* While the initial idea of applying formal verification to early-exit networks is well motivated, many of the subsequent algorithmic and analytical components follow naturally from that setup and feel more like straightforward extensions than new conceptual contributions.
* A large fraction of UNKNOWN results in Figure 1 indicates that the method still faces scalability and solver limitations.

### Questions
The experiments analyze the effect of the confidence threshold $T$ empirically, but could you comment on how sensitive the formal guarantees are to small variations in $T$? For instance, would a slight change in the threshold invalidate previously proven robustness properties?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces the framework for formal verification of neural networks with early exits (EE), addressing the gap between efficiency-oriented EE architectures and formal safety guarantees.  The authors formalize a robustness property tailored to early exit networks that accounts for conditional execution paths and multiple output layers.  They present a baseline verification algorithm along with two key optimizations: early stopping within the verification loop and heuristics to reduce redundant subqueries. 

The work is evaluated across multiple datasets (MNIST, CIFAR) and architectures (FC, CNN, ResNet, VGG), demonstrating that networks with early exits can be verified more efficiently than standard networks while maintaining soundness and completeness.

### Strengths
- The paper formulates robustness property that  handles the conditional execution logic inherent in EE networks.  This addresses the challenge of multiple potential output layers and conditional branching in verification.
- The theoretical analysis establishes that the problem is Fixed Parameter Tractable (FPT) under trace stability assumptions.  This helps provide  insights into when EE verification becomes useful.
- Results show that adding early exits improves verifiability compared to standard networks.

### Weaknesses
- The method only improves runtime of solvable cases, i.e., if the underlying verifier cannot verify a property given sufficient time, neither can the EE approach. From a practical standpoint, verification is typically a one-time cost that can run for days, limiting the real-world utility of runtime improvements. That said, EE would be beneficial from competitions where runtime matters.

- The algorithm scales poorly in the worst case, requiring E·(C-1)+1 verification queries (E is #  exits and C # of classes). This can lead to significant overhead when EE conditions are not met, making verification slower than standard approaches.

- Theoretical guarantees rely heavily on the trace stability assumption, which may not hold in practice. The paper assumes this holds but provides limited analysis of fallback strategies. The FPT complexity result (Thm 2) becomes meaningless if trace stability fails.

- Evaluation should include additional verifiers, e.g., the top X from the DNN verification competition (VNN-COMPs). Most of them are opensource.   The paper also say Marabou supports the format naturally, why not evaluating on it?

- Using ABCrown on an Apple M3 is a very strange setup as ABCrown, as well as most top DNN verification tools, rely on GPU.  In other words, in a proper setup of running DNN verification tools on computer with good GPU, this approach might not provide much improvement.  Running it on M3 appears as if the paper intentionally slow down existing work to highlight their improvements.  

- The proposed algorithms do not handle UNKNOWN results from the underlying verifier, which appear frequently in the experimental evaluation.

### Questions
- Can early exits actually help verify hard instances that underlying verifiers cannot solve even with extended runtime (e.g., days), or does the method only provide speedups for already solvable cases?

- AB-Crown works best on GPUs, so evaluating on an Apple M3 might not fully disclose the contributions.  Could you run it on a computer with GPU to properly demonstrate the improvement made in this work? 

- Evaluating on a single verifier doesn't seem adequate.  There's no reason for not trying other verifiers, which support similar input/output formats as ABCrown. 

- UNSAFE results are typically found quickly by verifiers (Fig 2b), yet Fig 2a and 2c show significant runtime for UNSAFE cases in some benchmarks. What accounts for this discrepancy?

- How does the method perform when the trace stability assumption fails?

### Soundness
2

### Presentation
3

### Contribution
2
