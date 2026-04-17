# TrimR: Verifier-based Training-Free Thinking Trimming for Efficient Test-Time Scaling

- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Large Reasoning Models (LRMs) demonstrate exceptional capability in tackling complex mathematical, logical, and coding tasks by leveraging extended Chain-of-Thought (CoT) reasoning. Test-time scaling methods—such as prolonging CoT with explicit token-level exploration—can push LRMs’ accuracy boundaries, but they incur significant decoding overhead. A key inefficiency source is LRMs often generate redundant thinking CoTs, which demonstrate clear structured overthinking and underthinking patterns. Inspired by human cognitive reasoning processes and numerical optimization theories, we propose TrimR, a verifier-based, training-free, efficient framework to trim reasoning and enhance test-time scaling, explicitly tailored for production-level deployment. Our method employs a lightweight, pretrained, instruction-tuned verifier to detect and truncate redundant intermediate thoughts of LRMs without any LRM or verifier fine-tuning. We present both the core algorithm and asynchronous online system engineered for high-throughput industrial applications. Empirical evaluations on Ascend NPUs and vLLM show that our framework delivers substantial gains in inference efficiency under large-batch workloads. In particular, on the four MATH500, AIME24/25, and GPQA benchmarks, the reasoning runtime of QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, and Pangu-R-38B is improved by up to 70% with negligible impact on accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The manuscript proposes TrimR, a training-free, verifier-based termination rule to shorten the “thinking” phase of large reasoning models without materially hurting accuracy. The verifier is a small model that checks for (i) the presence of a candidate answer and (ii) stabilization or equivalence across successive partial answers; when either signal indicates convergence or diminishing returns, TrimR stops the chain. The paper claims token savings and latency gains, and sketches an online serving setup that sparsifies verifier calls alongside a high-throughput LLM stack. While the idea is intuitive and pleasantly simple, the experimental scope and missing head-to-head baselines make the novelty feel narrower than it could be, especially given very recent work on token-budget prompting, minimal-draft reasoning, certainty-based early exit, and BoN accelerators.

### Strengths
From an originality viewpoint, using a binary, training-free verifier to detect convergence rather than to score full solution traces is a clean reframing. Compared with certainty indices or reward-model scoring, the authors’ proxy has the advantage of being cheap and model-agnostic, and it aligns with the community’s turn toward adaptive test-time compute. This direction is timely because recent methods like TALE (token-budget-aware prompting) and Chain-of-Draft (short, dense drafts instead of verbose CoT) show that much of today’s long chains are unnecessary; TrimR adds a different lever by detecting when to stop rather than prescribing how to speak.

Quality-wise, the paper’s engineering instinct to make the verifier calls sparse and to integrate with production-oriented stacks is sound. If realized properly, the approach is compatible with high-throughput serving systems such as vLLM (PagedAttention) and SGLang (RadixAttention, structured decoding), where shaving thinking tokens often translates directly into more QPS under the same memory budget.

The central mechanism is easy to follow and does not rely on delicate finetuning or proprietary signals. Given the rising concern about over- and under-thinking in o1/R1-style models, a simple stop-when-stable rule is appealing to practitioners who need something they can deploy tomorrow. In significance, a strong result here could matter beyond math: certainty-style early-exit (Certaindex) and BoN accelerators (Speculative Rejection) are quickly becoming standard ingredients in serving stacks; a verifier that outperforms them at the same accuracy would be valuable.

### Weaknesses
The experimental design underplays novelty by omitting the baselines that readers will expect. At minimum, the paper needs controlled, iso-setup comparisons against TALE/Token-Budget prompting and Chain-of-Draft to show that a convergence-verifier adds value on top of short-draft prompting. Without those, improvements could be attributed to any reasonable length control, not specifically to the verifier. Please add head-to-head results on AIME24/25 and MATH500 at matched accuracy deltas (±1%) and plot a token-efficiency frontier.

Early-exit and serving claims are not persuasive without Certaindex as a first-class baseline. Certaindex already signals when further compute is unlikely to help and reports real serving wins; readers will ask whether TrimR’s binary verifier is actually better at minimizing false stops and at maintaining throughput under identical batching/scheduler policies. A direct comparison (same hardware, same scheduler, same models) is necessary.

For BoN or multi-sample regimes, Speculative Rejection is designed to abort low-reward trajectories efficiently. If TrimR is claimed to be more compute-efficient than BoN accelerators, it should be compared inside the same BoN pipeline and reward model, with time-to-win and win-rate controlled. Otherwise the reader cannot separate the benefit of what is scored (reward-model vs. answer-equivalence) from the benefit of simply stopping earlier.

Verifier-design ablations are thin. Given the rich recent literature on PRMs and generative verifiers, two yes/no checks feel brittle without stress tests. The paper should show that equivalence detection is robust under paraphrase and reflection-token noise, and report mis-fire rates versus PRM/GenVerifier scoring when chains include small algebraic deviations or reordered steps. Right now, the evidence is not enough to argue for stability across models and prompting styles.

The benchmark mix is too math-centric and potentially contaminated. GPQA-Diamond is a good inclusion; however, several recent works show that AIME24 is contaminated for some model families and advocate live or freshly released streams. The paper would be much stronger with OlympiadBench for very long traces, MMLU-Pro for general reasoning, and MathArena to mitigate contamination concerns; a light GSM8K check is also useful to demonstrate that stopping early does not hurt easy-regime accuracy.

Finally, the claims about mitigating both overthinking and underthinking are only partially evidenced. Underthinking has recently been operationalized and measured; TrimR should adopt or adapt those metrics, and show improved depth-when-needed rather than only shorter chains on average. Otherwise, the method risks favoring brevity in cases where more thought is necessary.

### Questions
1. How does the verifier behave relative to certainty-style indices under identical serving conditions? A paired experiment replacing TrimR’s verifier with Certaindex inside the same stack (same batching, same schedulers) would clarify whether the proposed proxy is actually superior at the iso-accuracy frontier. Please report false-stop rates and area-under-compute-curve for both. 

2. Can the authors provide a frontier plot comparing TrimR against TALE and Chain-of-Draft on AIME24/25, MATH500, and GPQA-Diamond, where x-axis is tokens per solved instance and y-axis is accuracy? This would isolate whether TrimR’s how-to-stop signal gives better trade-offs than how-to-speak methods. 

3. Could the paper include a BoN study where TrimR replaces Speculative Rejection’s early abort, using the same reward model and candidate pool? If TrimR’s structure-aware convergence check is stronger, it should win on time-to-target-reward at equal sample budgets. 

4. Because over- and under-thinking are now measured with concrete diagnostics, can the authors evaluate TrimR on an underthinking-sensitive suite and report thought-switching metrics and depth utilization? A small ablation where reflection tokens are suppressed or noisy would further test robustness of the equivalence check. 

5. The serving section would be more convincing with a cross-platform replication. Please report throughput and latency on both vLLM and SGLang, and make QPS vs. latency curves available under identical prompts and batch sizes.

### Soundness
2

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
5

### Summary
This paper introduces TrimR, a novel, training-free framework designed to improve the inference efficiency of LRMs at test-time. The core idea is to use a lightweight, pre-trained verifier model to dynamically detect and prune redundant reasoning steps, categorized as "overthinking," "underthinking," and "repetition"—in the CoT process. By halting generation once a stable solution is reached or when the model enters a non-productive loop, TrimR aims to significantly reduce runtime and token usage with minimal impact on task accuracy. The authors conduct extensive experiments on mathematical and scientific reasoning benchmarks (MATH, AIME, GPQA) using several state-of-the-art LRMs, demonstrating substantial efficiency gains. They also present T4S, an online system designed for production-level deployment.

### Strengths
1.  The "training-free" nature of TrimR is its most compelling feature. Unlike methods that require costly fine-tuning or reinforcement learning, TrimR is non-invasive and preserves the LRM's original capabilities. This makes it immediately applicable to a wide range of pretrained models, a major advantage for real-world deployment. The use of a small, external verifier to classify answer existence and equivalence is a simple yet elegant approach that sidesteps the complexity and instability of training full-fledged Process or Outcome Reward Models (PRMs/ORMs)

2. The evaluation across multiple powerful LRMs (QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, Pangu-R-38B) and challenging, well-regarded benchmarks demonstrates the method's robustness. The inclusion of results on MoE models (Appendix N) further strengthens claims of architectural agnosticism.

3.  The authors rightly focus on metrics beyond simple accuracy, such as wall-clock runtime, Time Per Request (TPR), and total token consumption under a batched inference setting. These are the metrics that matter for production systems, and the reported gains (e.g., up to 70% runtime reduction) are impressive.​

4. The detailed description of the T4S online system (Sec 3.6, Fig 3) is a valuable contribution. Showing how TrimR integrates with modern inference frameworks like vLLM on industrial hardware (Ascend NPUs) moves the work from a purely academic concept to a practical engineering solution. This level of detail greatly enhances the paper's reproducibility and impact.​

### Weaknesses
1. The paper introduces a "Theoretical Foundation" (Section 1, page 2) by framing the reasoning process as a formal optimization problem:
$$ \min_{c} Infer_Cost(y_t) \quad s.t. \quad Perf(X, y_t) \ge Perf(X, y_{t^*}) - \epsilon $$
This formulation is presented as the mathematical underpinning of the method.
This formalization is entirely disconnected from the proposed algorithms. The paper never defines the performance function `Perf(X, y_t)`, the tolerance `\epsilon`, or shows how the compression rule `c` is derived from this objective. The actual method does not solve or approximate this optimization problem. Instead, it employs a set of disconnected, heuristic rules based on binary classifications from a verifier. The "foundation" is an analogy, not a mathematical basis, which is a significant overstatement of the method. 

2. The set of thoughts containing a solution is defined as `S = {s_i | F_v(p1, s_i) > 1}`. Since `F_v` is an indicator function, its output is {0, 1}, making the condition `> 1` impossible. This is likely a typo for `= 1`. More critically, the verifier for answer existence (`p1`) only checks if a thought contains a concluding statement, not if the reasoning leading to it is sound. A thought can contain a numerically correct but logically flawed answer, which TrimR would incorrectly identify as a valid convergence point.

3. The algorithm's counter for equivalent thoughts resets to zero upon any mismatch (`else reset on any mismatch count = 0`). This makes the termination condition exceptionally brittle. A valid reasoning sequence like `A -> A -> B -> A -> A` (where `A` is a correct answer and `B` is a brief, self-corrected error) would have its convergence counter reset by `B`, failing to trigger an early stop. This behavior is not robust to the noisy, non-linear nature of LRM reasoning. Furthermore, the choice of `M=2` (terminating after only two equivalent thoughts) is aggressive and its sensitivity is not analyzed. 

4. Section 3.3 states that underthinking is flagged if the model fails to produce "at least **three consecutive** answer-bearing chains that agree". However, **Algorithm 2** in Appendix B contains no logic to check this condition. The algorithm merely checks if the token budget (`t > R_thres * M`) and thought count (`num_thoughts > N_thres`) have been exceeded, without any reference to the history of answer agreements. This is a direct contradiction. The implemented algorithm is a simple budget check, not the convergence failure check described in the main text.

5. The entire framework is built on two key heuristics whose robustness is assumed rather than proven. The method's ability to segment reasoning chains depends entirely on a manually curated list of "reflection tokens" (e.g., `,...`, `Hmm, ...` in Appendix E). This approach is inherently model-specific and fragile. There is no guarantee that future or different LRMs will use these exact lexical cues. The paper provides no quantitative analysis of the segmentation quality (e.g., precision/recall of thought boundaries) or how the system would perform if a model reasoned without these explicit markers.

6. Section 3.1 states that the "intermediate thought answer" is extracted by taking the last `N_sent` sentences of a thought segment. Section 4.1 reveals this parameter is set to `N_sent=50`. An "answer" that is 50 sentences long is not a brief segment; it is likely the entire thought. 
This contradicts the paper's claims of using a lightweight verifier on "shorter but more informative" segments to save computation. This parameter choice is counter-intuitive and lacks any justification, suggesting either a typo or a fundamental flaw in the description of the answer extraction process

7. The paper repeatedly describes TrimR as "non-invasive", while the verifier operates externally, the "forceful prompt" (`I think I already thought for a long time... think...`) is an explicit and invasive intervention. It injects a specific string into the generation context to hijack the model's output flow and force it to emit the `</think>` token. This directly manipulates the generative process and contradicts the claim of being a purely non-invasive method that simply "halts" generation.

### Questions
1. Equations (existence/equivalence) omit tie-breaking and calibration details for binary decisions. please specify the exact scoring rule (e.g., logit for “Yes” token vs. sum of subword probabilities), thresholding, and how vocabulary/tokenization differences between Pangu-7B and Qwen2.5-7B-Instruct affect decision comparability

2. Section 3.3 states underthinking is flagged when there are not at least three consecutive agreeing, answer-bearing chains within budgets, but Appendix Algorithm 2 only checks $ t \ge R_{thres}\cdot M $ and $ {num\_thoughts}\ge N_{thres} $ without any consensus logic; please reconcile this inconsistency and provide the correct, consensus-aware pseudocode and results.

3. For Eqn. 2, define the precise notion of “logical or mathematical equivalence” operationalized by the verifier: is identity on normalized numeric answers sufficient, or must the steps be semantically equivalent; a formal criterion would reduce ambiguity and enable reproducibility across verifiers.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes TrimR, a training-free framework that enhances the inference efficiency of Large Reasoning Models (LRMs) by dynamically managing their reasoning process during test time. TrimR identifies and trims redundant reasoning (overthinking) and halts unproductive reasoning (underthinking) through a lightweight external verifier that checks answer existence and equivalence within intermediate reasoning segments. When redundant patterns are detected, the system adaptively terminates generation to reduce unnecessary computation. Experiments across multiple reasoning benchmarks demonstrate token reductions.

### Strengths
- The paper tackles the inefficiency of overthinking and underthinking in Large Reasoning Models (LRMs), which is a practically important issue for real-world deployment. The explicit handling of both forms of reasoning redundancy is well-motivated.
- The proposed method is prompt-based and requires no additional training or fine-tuning, making it straightforward to integrate into existing reasoning frameworks.
- Experiments across multiple LRMs and reasoning benchmarks demonstrate reductions in token usage with negligible loss in accuracy.

### Weaknesses
- The verifier may inherit biases from its own training data, which could influence the trimming behavior. It remains unclear whether the proposed method generalizes when switching to verifiers of different models or sizes. The authors should clarify whether prompt engineering is needed for different verifier models. And whether larger size verifier models could lead to better performance.
- The evaluation omits recent closed-source reasoning models such as Gemini 2.5-Pro and GPT-5. 
- It is unclear whether the R-threshold for early stopping is task-specific, dataset-specific and model-specific. A more systematic analysis of how this threshold influences the accuracy–efficiency trade-off across different scenarios would strengthen the paper.
- If a fixed “thinking budget” is specified in the initial prompt, how does it affect the reasoning behavior? The paper should analyze the performance difference compared to TrimR’s R-threshold.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses inefficiency in Large Reasoning Models (LRMs) that generate extended Chain-of-Thought reasoning. The authors identify two redundancy patterns: overthinking (redundant verification of correct solutions) and underthinking (oscillation without convergence on hard problems). They propose TrimR, a training-free framework that uses a lightweight 7B verifier model to detect redundancy by checking answer existence and equivalence in intermediate reasoning segments. When redundancy is detected, the system uses prompt-based early termination. The approach achieves up to 70% runtime reduction with negligible accuracy impact across mathematical and scientific reasoning benchmarks (MATH500, AIME24/25, GPQA) on models including QwQ-32B, DeepSeek-R1-Distill-Qwen-32B, and Pangu-R-38B. The paper includes both the core algorithm and an asynchronous online system designed for production deployment.

### Strengths
The paper makes several strong contributions. First, it provides thorough empirical analysis of redundancy patterns in LRMs with concrete statistics demonstrating overthinking and underthinking phenomena. Second, the approach is highly pragmatic—it's training-free, non-invasive to base models, and integrates seamlessly with existing inference frameworks. The use of lightweight 7B verifiers with simple binary classification tasks (answer existence and equivalence) is more practical than training complex reward models. The two-stage verification design cleverly minimizes false positives by first checking for answer existence before equivalence. Third, the experimental evaluation is comprehensive, covering multiple models and benchmarks with detailed ablation studies, consistently showing substantial efficiency gains (16-70% runtime reduction) with minimal accuracy loss (<2% in most cases). Finally, the system design addresses real deployment concerns including asynchronous operation, verifier-to-LRM ratios (8:1), and integration with production inference systems like vLLM.

### Weaknesses
Several limitations merit attention. First, the method introduces infrastructure complexity by requiring a separate 7B verifier model, though one verifier can serve multiple LRMs. Second, the verifier accuracy is only 87-88%, and while claimed to have "negligible effect," there's limited analysis of how verification errors compound across multiple sequential checks or lead to premature incorrect convergence. Third, the hyperparameters (M=2, Rthres=50%, Nthres=20) appear determined by informal inspection rather than systematic optimization. Fourth, the approach shows largest gains on weaker models (R1Q-32B: 67% reduction) that exhibit more redundancy, raising questions about whether it primarily benefits suboptimal models. Fifth, evaluation beyond mathematical/scientific reasoning is limited, with only brief code generation results. Finally, the theoretical framing as an optimization problem is somewhat superficial and doesn't strongly guide algorithm design—the connection between the formal optimization objective and the actual detection heuristics is loose.

### Questions
How does the verifier's 87-88% accuracy propagate through multiple sequential verification checks during long reasoning chains? Have you analyzed the cumulative error rate and specifically quantified how often false-positive convergence detection (claiming convergence on wrong answers) occurs versus beneficial early stopping?

### Soundness
3

### Presentation
3

### Contribution
3
