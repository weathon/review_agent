# SCDBench: A Benchmark for LLM-Based Smart Contract Decompilers

- Decision: Reject
- Scores: 2, 4, 6, 6

## Abstract
Smart contracts are programs deployed on blockchains that manage digital assets and enable decentralized applications. While their bytecode is always accessible on-chain, more than 99% of Ethereum contracts lack verified source code, making decompilation essential for transparency and security analysis.

Traditional decompilers rely on program analysis to produce structured but low-level representations. Recent advances in large language models (LLMs) enable source-like output with higher readability and even recompilability. Yet systematic evaluation is missing: existing tools use narrow datasets and inconsistent metrics, hindering fair comparison and reproducibility.

We present the first systematic benchmark for smart contract decompilation. Our contributions are: (i) a diverse dataset of real-world contracts, filtered for redundancy and stratified by difficulty; (ii) a staged evaluation framework with metrics for format completeness, compilability, Application Binary Interface (ABI) recovery accuracy, and semantic equivalence; and (iii) baseline evaluations using a fine-tuned reference model, establishing a strong foundation for future research.

Our benchmark establishes a common ground for rigorous, reproducible evaluation and aims to accelerate the development of reliable smart contract decompilers for blockchain security and transparency.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper is timely, well-written, and conceptually valuable, addressing an important gap — a standardized benchmark for LLM-based smart-contract decompilation. The staged evaluation (format → compile → ABI → semantic equivalence) is thoughtful and reproducible, and the anonymous repository and openness policy meet reproducibility expectations.

### Strengths
1. Smart-contract transparency and LLM decompilation are both emerging and important; this benchmark fills a genuine gap.
2. The four-stage design (format → compile → ABI → semantics) is systematic and easy to reproduce.
3. Writing, organization, and graphics are polished.

### Weaknesses
1. Novelty clarity: what is truly “first”?
You claim “the first systematic benchmark” but do not run widely-used non-LLM decompilers (Gigahorse, Shrnkr, Heimdall-rs, Panoramix) as external baselines on your dataset. Without those, it’s hard to judge whether the benchmark fairly stresses existing approaches and how LLM decompilers compare head-to-head. Add a baseline suite covering at least one static and one symbolic/industry tool and report all four stages on them.

2. Dataset Scale and Representativeness
The final benchmark includes only 150 contracts, corresponding to 2,735 unique functions. Although the construction pipeline (deduplication → clustering → stratified sampling) is sound, this sample size is insufficient for a benchmark claiming to evaluate semantic fidelity across the heterogeneous Ethereum ecosystem. Specifically: (1) With just 50 samples per difficulty bin, the variance across compiler versions, patterns, and application domains cannot be captured reliably; (2) The dataset omits Vyper, Yul, proxy upgrade patterns, and non-Ethereum EVMs. The verified-source bias means many prevalent contract archetypes. 

3. Stage dependencies and denominators are unclear
Stage 3 (ABI) and Stage 4 (semantic equivalence) are defined as contingent on compilability, but Tables 5–6 report high scores in medium/hard while Table 4 shows very few compilable outputs (e.g., only 3/21 compiled in hard for the reference model). Always report N (eligible) / N (bin) alongside each metric, and show CIs or bootstrap intervals. Provide a small flow diagram per bin: “#total → #format-ok → #compiled → #ABI-scored → #fuzzed.”

4. Lack of Executable Source Code and Pipelines
The reference model fine-tuned from “Qwen3-4B-Instruct-2507” is not released in your repository, and 'weights' or 'checkpoints' are unavailable.

5. Fuzzing-based semantic equivalence needs much stronger methodology
Ten random inputs per function is insufficient to establish semantic equivalence for contracts with wide state spaces. Use coverage-guided fuzzing (e.g., Forge’s fuzz with coverage hooks) + boundary heuristics for calldata/value, different msg.sender, msg.value, reentrancies, and environment (block.timestamp/number). Include stateful sequences (multi-call traces), not only single invocations.

6. Missing strong baselines
Existing non-LLM decompilers like Gigahorse, Elipmoc, Heimdall-rs, Panoramix are cited but not quantitatively compared. Without them, the benchmark cannot demonstrate where LLMs truly add value.

### Questions
1. How do you guarantee that near-duplicate contracts or variants compiled from the same template do not appear across train/test splits? 

2. Have you performed a global similarity clustering to prevent leakage, or only per-dataset filtering?

3. Beyond bytecode length, did you confirm that the “easy/medium/hard” bins correlate with actual decompilation complexity (e.g., control-flow depth, compiler optimizations, proxy patterns)?

4. How well does SCDBench represent DeFi, NFT, and infrastructure contract types? Are all Solidity versions (0.4–0.8) and compiler settings proportionally covered?

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
This paper proposes SCD-Bench, a benchmark for evaluating the capability of large language models (LLMs) in de-compiling smart contracts to recover the original code. The authors curate a set of 150 smart contracts whose source code is publicly available to construct the benchmark. The authors then evaluate a base LLM with and without fine-tuning to demonstrate the effectiveness of the benchmark.

### Strengths
- The paper addresses an important problem, smart contract decompilation, which has not been extensively studied with LLMs. The problem is also well motivated (security and auditing of smart contracts).
- The writing is very well organized and easy to follow.
- The benchmark is well designed, with a clear methodology for selecting contracts and evaluating model performance.

### Weaknesses
- Limited number of models evaluated -- while it seems like the main focus of the works on the benchmark itself, a more extensive evaluation of different LLMs (coding/reasoning etc.) would strengthen the paper.
- The benchmark size is quite small (50 contracts per group). Can this really be considered as a representative benchmark for the task?
- There are some hyperparameter choices that are not fully justified -- e.g. 10 inputs? is there some sort of completeness guarantee? and why $k=50$ and why $\alpha=0.7$? More details on these choices would be helpful.

### Questions
- Was synthetic data considered for any part of the benchmark? Fine-tuning, additional evaluation data, etc.
- Why the choice Qwen-4B specifically? (other than context length) is there a reason?

### Soundness
3

### Presentation
3

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
This paper presents SCDBench, the first systematic benchmark for evaluating LLM-based smart contract decompilers. SCDBench consists of a curated dataset of 150 real-world Ethereum contracts and 2,735 unique functions with filtering redundant (e.g., template tokens like ERC20) and duplicate contracts. This paper also proposes a stated evaluation framework with metrics such as ABI accuracy and format completeness.

### Strengths
1. Important and interesting topics that provide insightful benchmarks for further research.
2. The benchmark smart contracts are meticulously selected. Duplication is a well-known issue in the world of smart contracts; I really appreciate the authors' efforts in removing duplicate and template contracts and providing a benchmark with high quality.
3. The benchmark also carries a solid baseline method based on finetuned LLMs.

### Weaknesses
1. The size of the dataset is somewhat limited. I would appreciate it if the authors could enlarge the dataset with more unique contracts in the future.

2. The evaluation mainly compares a single fine-tuned model to its base version.

### Questions
In general, this is a good benchmark paper with a rigorous and thoughtfully designed framework. I appreciate that the authors recognize the widespread duplication in real-world smart contracts and take concrete steps to remove redundant or trivial instances. I believe the paper introduced a high-quality and diverse dataset. My main concern lies with the baseline evaluation: the paper only reports results for a single fine-tuned LLM (Qwen3-4B-Instruct), without comparisons to other open-source models such as DeepSeek or Llama-based variants. Could the authors justify the reason for that?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces the first systematic benchmark for smart contract decompilation, including a dataset of 150 real-world contracts and a staged evaluation framework with metrics for format completeness, compilability. The paper conducts baseline evaluations using a fine-tuned reference model, establishing a strong foundation for future research.

### Strengths
**Originality**
The paper introduced the first systematic benchmark for smart contract decompilation, with design principles that balance realism, diversity, and analytical value.

**Quality**
The paper constructed the dataset with staged pipeline that consolidates near-duplicates into coherent families and then selects a representative yet diverse subset suitable for benchmarking. 

**Clarity**
The paper is well-written with description of benchmark construction method and characterstics. The benchmark is open-sourced on anonymous repo.

**Significance**
The paper proposed a systematic benchmark which can help researchers evaluate their approach of using LLM to decompile smart contracts.

### Weaknesses
- Dataset consisting of 150 smart contracts may still be small even though the authors try to get a representative yet diverse subset, there may be risk of overfitting.
- Evaluation only on a small finetuned LLM Qwen3-4b, so it's hard to know whether the benchmark can assess the capability of state-of-the-art LLMs.

### Questions
1. Why do you choose k = 50 to construct only 150 representative contracts of three different difficulty levels?
2. Why you choose to only evaluate on a small LLM instead of state-of-the-art LLMs?

### Soundness
3

### Presentation
3

### Contribution
3
