# A Problem-Oriented Perspective and Anchor Verification for Code Optimization

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Large Language Models (LLMs) have shown remarkable capabilities in solving various programming tasks, such as code generation. However, their potential for code optimization, particularly in performance enhancement, remains largely unexplored. This paper investigates the capabilities of LLMs in optimizing code for minimal execution time, addressing a critical gap in current research. The recently proposed code optimization methods constructs program optimization pairs based on iterative submissions from the same programmer for the same problem. However, this approach confines LLMs to local performance improvements, neglecting global algorithmic innovation. To overcome this limitation, we adopt a completely different perspective by reconstructing the optimization pairs into a problem-oriented approach. This allows for the integration of various ideas from multiple programmers tackling the same problem. Furthermore, we observe that code optimization presents greater challenges compared to code generation, often accompanied by "optimization tax". Recognizing the inherent trade-offs in correctness and efficiency, we introduce a novel anchor verification framework to mitigate this "optimization tax." Ultimately, the problem oriented perspective combined with anchor verification framework significantly enhances both the correct optimization ratio and speedup to new levels.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the task of code optimization using LLMs, specifically focusing on reducing execution time. The authors propose two core contributions:

1. Problem-Oriented (PCO): The authors point out that existing code optimization datasets are User-Oriented, constructed by collecting iterative submissions from the same programmer for the same problem to form optimization pairs. They argue that this approach constrains LLMs to local performance improvements and overlooks global algorithmic innovations. As an alternative, this paper proposes a "Problem-Oriented" perspective, which pools solutions from all programmers for the same problem and sorts them by execution time. This method constructs an optimization trajectory that encompasses a more diverse range of ideas and algorithmic strategies.

2. Anchor Verification Framework: The authors discovered that as LLMs pursue higher efficiency, the correctness of the optimized code tends to decrease, a phenomenon they term the "optimization tax." To mitigate this issue, they propose an "Anchor Verification" framework. Experimental results show that an LLM fine-tuned on the problem-oriented PCO dataset achieves significant improvements in both optimization rate and speedup ratio compared to one fine-tuned on the user-oriented PIE dataset. Furthermore, applying the Anchor Verification framework further enhances all metrics, especially the correctness rate , demonstrating its effectiveness in alleviating the "optimization tax."

### Strengths
1. The shift from a "user-oriented" to a "problem-oriented" perspective is a novel and intuitively sound idea. It directly addresses a clear limitation of prior work -- a single programmer's cognitive inertia. This new perspective not only better aligns with the collaborative nature of real-world code review and refactoring but also, as the authors demonstrate, genuinely produces more diverse data.

2. The paper does not merely propose PCO and present the results; it provides strong supporting evidence through "structural analysis," "semantic representation analysis," and "manual and LLM-based sampling analysis." These analyses collectively confirm that PCO optimization pairs indeed involve more global, structural algorithmic changes, which enhances the quality of the dataset.

3. The "optimization tax" is a real and critical challenge when applying LLMs in multi-constrained domains like code. The authors identify it as the primary performance bottleneck for the model after PCO fine-tuning. The "Anchor Verification" framework is an elegant solution that creatively leverages the "slow code" as a "golden standard"—a unique asset in this specific task scenario. This is more reliable than relying on the LLM itself to generate test cases (as in the "Direct Test Generation" method) or self-debugging, a fact confirmed by the experimental results.

### Weaknesses
1. I have some concerns about the generalizability of PCO. The core of PCO relies on obtaining multiple valid submissions for the same well-defined problem from multiple programmers. While this data is abundant on programming contest platforms, it is rare in real-world software engineering. In a typical project, it seems unlikely to find 10 programmers solving the exact same problem with 10 different algorithms. Could the authors discuss the application of this approach in such real-world scenarios?

2. The entire Anchor Verification framework is built on the core assumption that the "slow code" is "functionally correct." Although the paper mentions that the dataset filters out incorrect submissions, the specific process for this is not entirely clear to me. What is the concrete procedure for ensuring this functional correctness in practice? If this "anchor" itself contains undiscovered flaws, how would that impact the reliability of the verification framework?

3. The paper claims the overhead of Anchor Verification is "almost the same" as direct test generation because the main cost is the LLM API calls. However, this conceals the premise that the "slow code" must execute "fast enough." If the "slow code" is extremely slow, executing it even on a few test cases could represent an unacceptable overhead. The paper touches on this, but I would appreciate a more detailed discussion on this point.

### Questions
Please refer to the Weaknesses section. If authors can address my concerns, I would be grateful and will raise my score. :-)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work tackles the challenge of LLMs in generating optimized code. Since the core idea of LLM-based code performance optimization is to construct a good dataset of pairs of code before and after optimization, called optimization pairs, this work proposes an alternative approach to collect this type of benchmark. Instead of collecting optimization pairs as code snippets from a single software developer across multiple steps of optimization, this work constructed pairs of code from multiple programmers with the same coding problem. The second contribution of this work came from the observation that when optimizing the code, LLM models have to sacrifice correctness in many cases, called “optimization tax”. To solve the challenge, the authors propose an anchor verification framework to generate tests for output code by testing the input “slow” code. The experiment was done with the fine-tuning process performed on CodeLLama, DeepSeekCoder, and Qwen, and with the anchor verification performed on Qwen2.5-Coder, GPT-4o, and DeepSeek-V3. The accuracy shows that the fine-tuning process with a problem-oriented perspective can significantly optimize the input code compared to user-oriented data collection. The anchor verification framework shows improvement over the self-debugging and direct test generation approach in improving code correctness.

### Strengths
- The paper is well-written.
- The idea of a problem-oriented perspective is sound and clear. It can help LLMs to learn how to optimize code by different coding styles.
- Valid approach to ensure code correctness by anchor verification.

### Weaknesses
- While this work has considerable potential for improvement, a primary concern from the reviewer is that one drawback of problem-oriented data collection is that the dataset may contain pairs of code that are too dissimilar, despite solving the same coding problem. For example, the proposed strategy of data collection can produce code pairs that differ significantly in terms of variable declaration and readability level, which may increase the risk of confusing the LLMs during fine-tuning.
- While the dataset was collected from a coding interview, it’s very likely that there were an existing dataset that already contained test cases, such as XCodeEval [1]. I suggest that authors consider evaluating this dataset.
- In some problems like “implementing a torch model and extracting the loss function”, the test outputs, which are usually loss numbers from 0..1, it’s very likely that with the same input, outputs of the slow code and optimized code are slightly different. The proposed anchor verification framework does not handle this case.
- To evaluate the code correctness, there are other coding perspectives, such as readability or coding style [2]. The anchor verification framework didn’t consider these perspectives

1.XCodeEval: xCodeEval: A Large Scale Multilingual Multitask Benchmark for Code Understanding, Generation, Translation, and Retrieval
2. Aligning Large Language Models to Coding Preferences

### Questions
- Can you provide an ablation study to see the cases that the PCO dataset collection collects pairs of code with high graph edit distance and vice versa (show a case of Figure 2)?
- An algorithm to eliminate low-quality coding pairs for PCO is needed.
- Does your dataset contain coding problems with approximated output, such as [3]?

3. https://github.com/Exorust/TorchLeet/blob/main/torch/easy/rnn/RNN.ipynb

### Soundness
2

### Presentation
4

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
This paper introduces two primary contributions to the field of code optimization using Large Language Models (LLMs). First, it proposes a "problem-oriented" perspective for constructing optimization datasets, a departure from the existing "user-oriented" approach. Instead of creating optimization pairs from the iterative submissions of a single programmer (which often leads to local, incremental improvements), the proposed method sources solutions from multiple programmers for the same problem, creating more diverse and algorithmically significant optimization trajectories.
Second, the paper identifies and addresses the "optimization tax," a phenomenon where LLMs improve code efficiency at the cost of functional correctness. To mitigate this, it introduces the "anchor verification framework," which leverages the original, inefficient but correct code as a "gold-standard anchor." This anchor is used to generate a set of verified test cases, which then provide execution feedback to iteratively refine the LLM's optimized code, significantly improving its correctness. The authors conduct extensive experiments on multiple state-of-the-art code LLMs, demonstrating that both the problem-oriented perspective and the anchor verification framework lead to substantial improvements in optimization ratio, speedup, and correctness.

### Strengths
- Novel and Intuitive Core Ideas: The two central contributions are well-motivated and insightful. The shift from a "user-oriented" to a "problem-oriented" perspective is a simple yet powerful reframing that directly tackles the issue of limited diversity in optimization strategies. The "anchor verification framework" is a practical and clever solution to the critical problem of correctness in automated optimization.
- Comprehensive and Rigorous Evaluation: The paper's empirical evaluation is a significant strength. The authors don't just present final numbers; they validate their hypotheses through multi-dimensional analysis, including structural (Graph Edit Distance), semantic (t-SNE embeddings), and human-led analysis to demonstrate the superior diversity of the problem-oriented dataset.
Strong and Significant Empirical Gains: The experimental results are impressive and consistently support the paper's claims. The performance lift from the user-oriented (PIE) to the problem-oriented (PCO) fine-tuning is substantial across all metrics. Furthermore, the anchor verification framework provides an additional, significant boost, particularly in the crucial "Percent Correct" metric, validating its effectiveness.

### Weaknesses
- The experiments are conducted on the PIE and PCO datasets, which are derived from competitive programming problems on CodeNet. While this is a suitable domain for studying algorithmic optimization, it raises questions about the generalizability of the findings. Real-world software optimization often involves different challenges, such as I/O bottlenecks, memory management, API usage, and interaction with large codebases, which are not well-represented in this setting. The effectiveness of the proposed methods in these more common software engineering domains remains unevaluated.
- Reliance on Simulated Performance Metrics: The paper relies on the gem5 CPU simulator for benchmarking execution time and calculating speedup. While simulators offer a controlled and reproducible environment, their performance characteristics can diverge from real-world hardware, especially concerning complex interactions like caching, memory bandwidth, and modern CPU microarchitectures. The performance gains, while significant in simulation, might differ on actual hardware. Including even a small-scale study on physical hardware would strengthen the claims of practical performance improvement.

### Questions
see weakness

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper argues that current LLM code-optimization datasets (e.g., PIE) are user-oriented—built from each programmer’s iterative submissions—so they mostly capture small, local improvements. It proposes a problem-oriented construction (PCO): pool all (functionally correct) submissions for the same problem across users, sort by runtime, and form optimization pairs along this global trajectory.

### Strengths
* The problem-oriented pairing is simple, scalable (pair count grows sharply with users per problem), and empirically yields more global, algorithmic improvements, which translate to strong %OPT/SPEEDUP gains after finetuning. 

* Anchor Verification cleverly reuses the slow reference to build verified test cases, outperforming self-debugging or direct test-generation baselines and improving %OPT, SPEEDUP, and CORRECT simultaneously.

### Weaknesses
* Evaluation centers on C/C++ competitive-programming style tasks compiled with -O3 on gem5. It’s unclear how PCO/Anchor Verification transfer to multi-file projects, diverse languages, library/API-heavy code, or system-level constraints.

* To equalize counts, PCO keeps the top-speedup pairs per problem (to 78K), which may advantage PCO relative to PIE beyond perspective alone; more controls (e.g., random or stratified matching) would strengthen the causal claim.  

* EST@k relies on sampling with k up to 8 and T=0.7; variance across seeds, compute/token budgets for finetuning, and wall-clock overheads (including executor/sandbox) are only lightly discussed.

### Questions
* How does PCO perform on multi-file repos, build systems, or tasks requiring library calls and I/O patterns? Any results beyond single-file competitive problems or beyond C/C++

* If PCO pairs were randomly selected (or matched by structural/semantic distance) rather than top-speedup, do gains persist with similar magnitude? Can you run a controlled ablation to isolate perspective vs. selection effects?

* What are the GPU hours and runtime overheads for PCO finetuning and Anchor Verification (per iteration)? How does BEST@k scale in cost vs. benefit for practical deployment?

### Soundness
3

### Presentation
3

### Contribution
3
