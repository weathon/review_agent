# Adaptive Thinking: Large Language Models Know When to Think in Latent Space

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Recent advances in large language models (LLMs) test-time computing have introduced the capability to perform intermediate chain-of-thought (CoT) reasoning (thinking) before generating answers. While increasing the thinking budget yields smooth performance improvements at inference time, the relationship between LLM capability, query complexity, and optimal budget allocation remains poorly understood for achieving compute-optimal inference. To address this challenge, we utilize $\textit{self-consistency}$, the agreement among multiple reasoning paths, as a proxy for thinking necessity. We first identify that lower self-consistency indicates when queries require extended thinking to reach correct answers. Building on this insight, we introduce $\texttt{Sonata}$ (Self-Consistency-Guided Adapter for Thinking Allocation), a lightweight approach that adaptively allocates thinking budgets to optimize the performance-efficiency tradeoff. $\texttt{Sonata}$ includes an adapter trained offline on a calibration dataset to predict self-consistency directly from the last layer hidden representations during the query prefilling stage. This prediction then guides on-the-fly budget allocation before thinking. The adapter is general, transferable across diverse tasks once trained, and introduces $<1$$\textperthousand$ computational overhead during inference. Notably, Sonata is compatible with existing CoT compression methods, enabling further efficiency gains when managing thinking budgets across queries. Extensive experiments on multiple models (Qwen3-8B, Qwen3-32B, GPT-OSS-120B, Qwen3-235B-A22B) and benchmarks~(AIME25, GSM8K, MATH500, GPQA, LiveCodeBench) demonstrate that $\texttt{Sonata}$ achieves $20\\%$ to $60\\%$ reduction in thinking tokens while maintaining the same accuracy, or up to $2\\%$ improvement in accuracy with the same token cost.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper uses self-consistency—the agreement among multiple reasoning paths for the same problem—as an operational proxy for determining “whether long reasoning is needed.” Empirically, samples with lower self-consistency tend to require extended reasoning. Building on this observation, the paper proposes a highly lightweight gating scheme, Sonata: during the prefill stage, it predicts self-consistency directly from the hidden representation of the last layer and the last position, and accordingly makes an instant decision on whether and how much to think before decoding. The adapter is trained offline on a calibration set, has an online cost of less than one-thousandth, can be reused across tasks, and can be combined with existing CoT compression methods for further gains. Experiments across multiple models and benchmarks show that it can reduce reasoning tokens by 20%–60% without loss of accuracy, or improve accuracy by about 2% under the same token budget, thus advancing the Pareto frontier of “performance–efficiency.” Overall, the paper turns the decision of “whether and how much to think” from a heuristic into a learnable, deployable, and almost zero-overhead paradigm, balancing theoretical motivation and engineering practicality.

### Strengths
1. The problem addressed in this paper is crucial for the real-world deployment of large models. The Pareto frontier results clearly demonstrate that the approach achieves higher accuracy under the same compute budget or lower cost under the same accuracy, providing intuitive and persuasive evidence.

2. The paper innovatively observes that “self-consistency becomes more separable in deeper hidden spaces,” and accordingly predicts self-consistency using the hidden state of the last layer and last token to form a lightweight gating mechanism. The idea is novel and well supported by extensive experiments.

3. The method is simple and practical: the adapter is trained once offline via calibration using a two-layer MLP, and during inference only reads one hidden state in the prefill stage for gating—introducing almost zero additional overhead and being highly engineering-friendly.

### Weaknesses
1. The coverage of baselines could be further expanded. Although comparisons with two entropy-based baselines are sufficient, the study does not include some of the stronger recent uncertainty or calibration signals (e.g., multi-step verification-based early stopping, lightweight proxies based on draft diversity, or learned “thinking switch” controllers). The current conclusion mainly shows that “entropy signals are inadequate,” without directly competing against the strongest adaptive reasoning strategies.

2. In Figure 4, the authors mention that when $s<\tau$, the model enters the reasoning phase and adaptively allocates reasoning token budget. However, the main text does not explain how these different tiers are assigned, and the algorithmic framework only shows a binary classification. The authors should report the concrete procedure for adaptive budget allocation in detail.

### Questions
1. The paper only discusses using the hidden state of the “last layer and last token.” Would multi-position aggregation or multi-layer fusion bring further benefits? This limitation might affect robustness on semantically complex or multi-turn inputs.

2. The current method decides the reasoning length solely based on the state of the last token of the question. Could this be extended to dynamically determine, during long-form reasoning, whether the reasoning token budget can be shortened according to the proposed self-consistency measure?

### Soundness
4

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
The paper proposes Sonata, a lightweight plug-in designed to predict answer self-consistency from the last-layer hidden state of a large language model (LLM). The predicted score determines whether the model should enable long chain-of-thought (CoT) reasoning. Trained offline on 1,000 competition-math problems, the approach introduces less than 0.1% additional FLOPs while reducing CoT token usage by 20–60%, maintaining or slightly improving accuracy on four math and science benchmarks.

### Strengths
1. The proposed method is highly modular and can be seamlessly integrated into existing large language models without modifying or fine-tuning the base parameters. It is also compatible with current chain-of-thought (CoT) compression and optimization techniques, enabling straightforward adoption in practical systems. Moreover, the approach incurs negligible computational cost, requiring only a single MLP forward pass on already-computed hidden states. 
2. The authors provide empirical evidence that latency and memory overhead remain minimal across a wide range of model scales (from 8B to 235B parameters).
3. The paper is overall well-written and easy to follow.

### Weaknesses
1. The preliminary visualization experiment, which performs PCA clustering on the hidden state of the final token, is conducted only on a single mathematical benchmark. The study does not extend this analysis to other datasets or task types, such as commonsense reasoning or code generation. It remains unclear whether the observed separability is a general phenomenon or specific to mathematical domains.

2. The authors claim that the MLP module trained to predict self-consistency on math datasets can generalize across domains. However, among the four benchmarks used for evaluation, three are mathematical (AIME, MATH, GSM) and only one is scientific (GPQA). This experimental setup provides limited evidence for cross-domain generalization.

3. The paper fixes the decision threshold for enabling reasoning at τ = 0.3, but does not justify how this value is chosen. Providing illustrative cases that show differences in question difficulty or prediction confidence around this threshold would make the calibration choice more convincing.

4. The paper lacks a qualitative case study demonstrating the effect of Sonata on model behavior. For instance, it would be helpful to show concrete examples comparing model outputs before and after applying Sonata, to better understand how the method changes reasoning dynamics.

### Questions
No further questions beyond those discussed above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Sonata, a lightweight adapter that predicts whether and how much chain-of-thought “thinking” an LLM should use by estimating self-consistency directly from the last-layer hidden state during prefilling. Using predicted self-consistency as a proxy for reasoning difficulty, Sonata allocates thinking budgets via a simple threshold, enabling non-thinking for easy queries and CoT for hard ones with negligible overhead. Across multiple models (8B–235B) and benchmarks (AIME25, GSM8K, MATH-500, GPQA), it reduces thinking tokens by 20–60% at the same accuracy, or improves accuracy by up to 2% at the same cost, while also lowering latency. The adapter is task-agnostic once trained and is compatible with existing CoT compression/pruning methods.

### Strengths
- Minimal overhead with substantial efficiency gains: Adds virtually zero compute (<0.1% layer-equivalent), reduces thinking tokens by 20–60%, and cuts end-to-end latency by ~27–36% across models.

- Better performance–efficiency trade-off: Self-consistency–guided allocation outperforms fixed budgets and self-judging, maintaining accuracy or improving it (up to +2%) at the same cost.

- Generalizable and compatible: Once trained per model, the adapter transfers across tasks (e.g., math to GPQA) and integrates seamlessly with existing CoT compression/pruning and early-exit strategies.

### Weaknesses
- Missing strong baselines: Lacks systematic comparisons against recent dynamic budget/early-stop/CoT-pruning methods, so relative advantages are not fully established.

- Narrow task coverage: Evaluation focuses on math/scientific reasoning; absent tests on code generation, open-ended generation, and  long-context multi-step planning

- Coarse and potentially brittle control: Online decision is primarily binary via a single threshold

### Questions
How sensitive is Sonata to offline calibration choices (calibration set size and difficulty distribution, number of samples N per query, and verifier noise), and can similar performance be maintained under lower calibration cost?

### Soundness
2

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
4

### Summary
This paper addresses the critical challenge of compute-optimal inference for large language models (LLMs) with chain-of-thought (CoT) reasoning capabilities. A key limitation of existing methods is the lack of adaptive thinking budget allocation: fixed budgets either waste resources on simple queries or lead to errors on complex ones, while entropy-based proxies fail to capture intrinsic reasoning difficulty. The authors propose two core insights: (1) self-consistency (agreement among multiple reasoning paths for a query) is a reliable proxy for thinking necessity—low self-consistency indicates a need for extended reasoning, while high self-consistency signals minimal thinking is required; (2) self-consistency patterns are highly distinguishable in the LLM’s latent space (especially in deeper layers), enabling efficient prediction without expensive repeated sampling.

Building on these insights, the authors introduce Sonata (Self-Consistency-Guided Adapter for Thinking Allocation): a lightweight 2-layer MLP adapter trained offline on a calibration dataset to predict self-consistency directly from the last-layer hidden representations of queries during the prefilling stage. At inference, Sonata uses this prediction to dynamically allocate thinking budgets (e.g., skipping thinking for high self-consistency queries) with <1‰ computational overhead. Extensive experiments across 4 models (Qwen3-8B to Qwen3-235B-A22B) and 4 benchmarks (AIME25, GSM8K, MATH500, GPQA) show Sonata reduces thinking tokens by 20–60% while maintaining or improving accuracy (up to 2% gain). It also generalizes across tasks and is compatible with existing CoT compression methods.

### Strengths
- Experiments are comprehensive, covering diverse model sizes (8B to 235B parameters) and task types (mathematical, general reasoning), ensuring results are not limited to specific settings.

- Unlike black-box adaptive methods, Sonata’s reliance on self-consistency and latent space patterns provides clear interpretability—users can trace budget decisions to a measurable proxy (self-consistency) rather than opaque model outputs.

- Sonata’s minimal overhead and offline training make it easy to integrate into existing LLM pipelines. Compatibility with CoT compression enables further efficiency gains, enhancing real-world deployment value.

### Weaknesses
- Sonata requires a calibration dataset (1000 samples from OpenMathReasoning) for offline training. The paper does not explore how calibration set size or difficulty affects performance—e.g., would smaller datasets or datasets with mismatched difficulty (e.g., training on easy queries, testing on hard ones) degrade Sonata’s accuracy? This is critical for low-resource scenarios.

- The paper mentions a cluster of “intrinsically difficult queries” (lower-left of Figure 2) where even extended reasoning fails, but does not address how Sonata handles these. For example, could Sonata be modified to detect such queries and avoid wasting tokens on futile reasoning?

- The self-judge baseline uses a binary “YES/NO” prompt for thinking need, but more nuanced prompts (e.g., “rate difficulty 1–5”) might improve its performance. The paper’s current self-judge design could underestimate the potential of LLM self-assessment, weakening the comparison to Sonata.

### Questions
1. How does Sonata’s performance degrade when the calibration dataset is smaller (e.g., 100 samples) or has difficulty distributions mismatched to the test set (e.g., calibration on easy queries, test on hard queries)? For real-world use, users may not have access to large, well-matched calibration data—insights here would improve practicality.

2. For the “intrinsically difficult queries” (low self-consistency, low thinking gain), does Sonata still allocate thinking budgets? If so, could you extend Sonata to predict both self-consistency and thinking gain (e.g., via a multi-output adapter) to avoid wasting tokens on these queries?

3. The self-judge baseline uses a binary prompt, but prior work shows nuanced prompts improve LLM self-assessment. Have you tested Sonata against a self-judge baseline with difficulty-rating prompts (e.g., “rate how much thinking this query needs: 1–5”)?

### Soundness
3

### Presentation
3

### Contribution
2
