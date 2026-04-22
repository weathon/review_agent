# Speculative Speculative Decoding

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 2

## Abstract
Autoregressive decoding is bottlenecked by its *sequential* nature. Speculative decoding has become a standard way to accelerate inference by using a fast draft model to predict upcoming tokens from a slower target model, and then verifying them *in parallel* with a single target model forward pass. However, speculative decoding itself relies on a *sequential* dependence between speculation and verification. We introduce *speculative speculative decoding* (SSD) to parallelize these operations. While a verification is ongoing, the draft model *predicts* likely verification outcomes and prepares speculations pre-emptively for them. If the actual verification outcome is then in the predicted set, a speculation can be returned immediately, eliminating drafting overhead entirely. We identify three key challenges presented by speculative speculative decoding, and suggest principled methods to solve each. The result is Saguaro, an optimized SSD algorithm. Our implementation is up to 2x faster than optimized speculative decoding baselines and up to 5x faster than autoregressive decoding with open source inference engines.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces ​Speculative Speculative Decoding (SSD)​, a novel framework that parallelizes the drafting and verification phases in speculative decoding. By pre-speculating likely verification outcomes during ongoing target model verification, SSD eliminates the sequential dependency between drafting and verification, achieving significant latency reductions.

### Strengths
1 The idea of speculating on speculationis both creative and impactful. By leveraging idle draft-model compute during verification, SSD addresses a fundamental bottleneck in existing speculative decoding methods. The theoretical framework (Algorithm 1) is elegant and generalizable.

2 The paper provides strong theoretical analysis.

3 Comprehensive Empirical Validation.

### Weaknesses
1 While SSD reduces latency, the parallel speculation across multiple branches increases draft-model FLOPs. The paper should quantify this overhead more explicitly.

2 Although comparisons to standard SD are thorough, benchmarking against recent parallel SD methods (e.g., SwiftSpec) or tree-based decoding (e.g., SpecInfer) would strengthen the claims. Does SSD outperform these methods in high-batch or high-temperature regimes?

### Questions
1 How does SAGUARO’s memory footprint scale with batch size and fan-out? Could cache eviction policies improve efficiency?

2 Could the pre-speculation idea be combined with approximate drafting (e.g., EAGLE-3) for further gains?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Speculative Speculative Decoding (SSD) — a new framework that parallelizes both speculation and verification in a distributed manner. While typical SD alternates executions between drafter and verifier, SSD removes this dependence and performs drafting and verification concurrently and asynchronously. In SSD, while verification for round T is ongoing, the draft model predicts the likely verification outcomes and pre-speculates corresponding continuations.

### Strengths
- This paper has novelty in terms of serving the draft and target models asynchronously.
- This paper has clear derivations and proofs (Theorem 7, 12, 17), which make the framework analytically sound.
- The author has conducted extensive benchmarks on multiple model families, detailed ablations (fan-out, sampling, fallback), and clear visualizations.

### Weaknesses
- The flow of the paper can be improved, e.g. Section 4 shows the results first and then explains the algorithms
- Can give a better flowchart of the workflow in Figure 1, e.g. label the step index
- This method requires an independent GPU for the speculator, this can increase the cost of serving and system complexity (e.g. heterogeneous GPUs)

### Questions
- While the deployment of LLMs typically runs on $2^n$ GPUs, SSD actually requires $2^{n} + 1$ GPUs. How can users deploy the SSD for serving efficiently? e.g. if using homogeneous GPUs, using a Hopper GPU for only the small draft model can be an overkill as much of the compute and memory are not saturated. If using heterogeneous GPUs (i.e. running the speculators on another type of GPU), how much impact will the communication between speculators and verifiers be?

### Soundness
3

### Presentation
1

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
This paper introduces Speculative Speculative Decoding (SSD), a framework that parallelizes the drafting and verification stages of speculative decoding by having the draft model predict likely verification outcomes in advance and pre-speculate for them. The authors propose SAGUARO, an optimized SSD algorithm that addresses three key challenges: accurately predicting verification outcomes, balancing cache hit rate with token acceptance rate, and handling cache misses efficiently. SAGUARO achieves up to 2x speedup over standard speculative decoding and up to 5x over autoregressive decoding on open-source inference engines.

### Strengths
1. The paper presents a principled and theoretically grounded approach to overlapping drafting and verification, effectively eliminating drafting latency through pre-speculation.

2. SAGUARO introduces practical and novel techniques—geometric fan-out allocation, cache-aware sampling, and adaptive fallback strategies—that collectively address core challenges in SSD.

3. The evaluation is comprehensive, covering diverse models, datasets, temperatures, and batch sizes, and shows significant end-to-end speedups while maintaining compatibility with existing acceleration methods.

### Weaknesses
1. The authors do not compare their method with existing SOTA methods (e.g., EAGLE2/3). They argue that their method is orthogonal to EAGLE and can be combined with it. However, the paper does not report the performance of their method when combined with EAGLE3. I would like to know what performance improvement could be achieved if they were combined.

2. The authors do not compare their method with existing Token Tree SPD methods, which I believe is a missing baseline. The proposed method (SSD) has the draft model send only a single speculative sequence to the verifier, and while the verifier validates that sequence, the draft model computes other possible branches in parallel. In contrast, Token Tree SPD methods have the draft model generate an entire "tree" structure at once, providing multiple candidate tokens at each position, and the verifier must validate all branches and nodes of this tree in a single forward pass. I consider these to be two similar solution approaches, and therefore the authors should compare these two types of methods. For example, they could replace the Token Tree SPD component in EAGLE3 with their SSD method and report whether the performance increases or decreases.

### Questions
1. I would like the authors to provide the wall time for their method. For example, while the target model is performing Verify, how many tokens can the draft model generate in that same amount of time?

2. I would like to know the number of accepted tokens for your method. Compared to EAGLE3, does your method accept more or fewer tokens per round?

I will adjust my score based on the author's rebuttal.

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
3

### Summary
This paper proposes to parallelize the draft and verifier models in speculative decoding by running them on separate hardware devices. The proposed approach demonstrates superior performance compared to traditional speculative decoding in the reported experiments.

### Strengths
1.	The proposed method outperforms standard speculative decoding (SD) in the reported experiments.
2.	Introducing parallelism between the draft and verifier models represents an interesting and promising direction for advancing speculative decoding.

### Weaknesses
1.	The algorithmic description is difficult to follow in the current paper organization. It is unclear how the proposed method preserves the original distribution, and while the paper provides extensive theoretical analysis, it offers limited explanation and intuition about the algorithm itself, particularly in Section 3.
2.	The paper lacks baseline comparisons against other speculative decoding (SD) approaches, making it hard to assess the relative performance and contribution of the proposed method.
3.	The paper does not include a dedicated experimental section—experiments appear only as a subsection—and there is insufficient detail regarding the setup, such as which models were evaluated and under what conditions.

### Questions
Could you improve the paper’s structure to make the algorithm easier for readers to follow?
Additionally, could you add more experiments to address the weaknesses noted in points 2 and 3, particularly by including comparisons with other speculative decoding methods and providing more detailed experimental setups?
Finally, could you clarify how your method preserves the original distribution, both conceptually and mathematically?

### Soundness
2

### Presentation
1

### Contribution
2
