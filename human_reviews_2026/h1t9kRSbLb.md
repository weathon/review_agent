# R-Stitch: Dynamic Trajectory Stitching for Efficient Reasoning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6, 4

## Abstract
Chain-of-thought (CoT) enhances the problem-solving ability of large language models (LLMs) but incurs substantial inference cost due to long autoregressive trajectories. Existing acceleration strategies either shorten traces via early stopping or compression, or adopt speculative decoding with a smaller model. However, speculative decoding provides limited gains when model agreement is low and rigidly enforces token-level consistency, overlooking the observation that some smaller models, when correct, produce significantly more concise reasoning traces that could reduce inference length.
We introduce R-Stitch, a training-free hybrid decoding framework that leverages token-level entropy as an uncertainty proxy to delegate computation between a small language model (SLM) and an LLM. Our analysis shows that high-entropy tokens are more likely to induce errors, motivating an entropy-guided routing strategy that lets the SLM efficiently handle low-entropy tokens while delegating uncertain ones to the LLM, thereby avoiding full rollbacks and preserving answer quality.
We further extend this design with R-Stitch$^+$, which learns an adaptive routing policy to adjust the token budget dynamically beyond fixed thresholds. By jointly reducing per-token decoding complexity and the number of generated tokens, our method achieves substantial acceleration with negligible accuracy loss. Concretely, it attains peak speedups of $3.00\times$ on DeepSeek-R1-Distill-Qwen-7B, $3.85\times$ on 14B, and $4.10\times$ on QWQ-32B while maintaining accuracy comparable to full LLM decoding. Moreover, it naturally enables adaptive efficiency–accuracy trade-offs that can be tailored to diverse computational budgets without retraining.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes R-Stitch, a training-free hybrid decoding method that accelerates chain-of-thought reasoning by dynamically switching between a small language model and a large language model based on token entropy. Low-entropy tokens are handled by the SLM to save computation, while high-entropy tokens are delegated to the LLM to preserve accuracy. It shows that token entropy correlates with reasoning errors and uses it for token-level routing. In addition, the paper extends R-Stitch to R-Stitch+, which learns an adaptive switching policy via reinforcement learning with a latency-aware reward. Evaluation shows seepdup in generation with minimal accuracy loss on tested benchmarks.

### Strengths
- The paper tackles an important and timely problem: optimizing reasoning-time inference in large language models. As reasoning chains become longer and more expensive, improving efficiency without sacrificing accuracy is crucial for practical deployment.

- The use of entropy as a routing signal is well-motivated and empirically justified. Figure 3 effectively shows that incorrect outputs correlate with higher entropy, that most tokens have near-zero entropy, and that errors cluster around locally uncertain regions. This analysis builds a foundation for using entropy as a measure to guide dynamic switching.

- The approach leverages model-internal signals to optimize system-level behavior, is conceptually elegant.

### Weaknesses
- The novelty is limited. The method is conceptually incremental to speculative decoding, differing mainly in using entropy for routing rather than token-level agreement. Moreover, the connection between entropy, confidence, and uncertainty has been explored in prior work [1], yet this paper neither cites nor compares against them.

- The evaluation setting (batch size = 1) is unrealistic and potentially misleading. In real deployments, systems run with batch sizes > 1 to maximize GPU utilization, especially in frameworks like vLLM, which fully support batching. Running with batch = 1 may exaggerate latency gains, and the authors should justify this design choice.

- The evaluation scope is narrow. All experiments and entropy analyses are restricted to mathematical reasoning tasks. Since the paper claims general acceleration of chain-of-thought reasoning, results across other reasoning domains (e.g., code, logical reasoning, or QA) would make the findings more convincing.

- The paper ignores the memory overhead of maintaining both the LLM and SLM simultaneously. This dual-model setup increases GPU memory consumption and can reduce the number of requests processed in parallel. It’s possible that this constraint is why the experiments use batch size = 1, which should have been discussed explicitly.

- The method is not compared to alternative reasoning optimization approaches such as early termination or compression-based techniques, which also shorten reasoning traces. A fair evaluation should include or discuss these baselines to clarify the relative contribution.

- The approach’s robustness to entropy misprediction is unclear. If entropy fails to reflect uncertainty accurately (e.g., when the SLM is overconfident in incorrect outputs), the routing may degrade accuracy or fail to switch appropriately.

- The computational overhead of computing token entropy is not thoroughly analyzed. Although entropy can be obtained from logits, its per-token computation and switching logic could add nontrivial latency, which should be measured or discussed.

[1] Efficiently serving llm reasoning programs with certaindex. Fu, Yichao ; Chen, Junda ; Zhu, Siqi ; Fu, Zheyu ; Dai, Zhongdongming ; Zhuang, Yonghao ; Ma, Yian ; Qiao, Aurick ; Rosing, Tajana ; Stoica, Ion ; Zhang, Hao

### Questions
See the weaknesses section.

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
4

### Summary
This paper focuses on efficient generation for long-CoT LLM. Long-CoT LLM normally generated extensively long trajectory, which is very time consumed. Speculative decoding (SpecDec) serves as traditional solution to speed it up. I.e. only use a LLM to generate when the tokens generated from the SLM are rejected. However, SpecDec requires a strict distribution alignment between LLM and SLM, and might falsely reject tokens that are correct but not aligned with LLM. 

In this paper, the authors firstly observe that high entropy normally leave to incorrect generation, thus propose to use entropy itself as a hint for token rejection and acceptance, i.e. R-Stitch. If the entropy of a token from SLM is too high, the decoding is switched to LLM until meeting a new token with a low entropy, then the decoding is switched again to SLM. The authors further proposes another variant, R-Stich$^+$, that applies RL to train the model for better accuracy and efficiency gain.

Through extensive experiments, R-Stich and R-Stich$^+$ show comparable or better accuracy to the LLM and SpecDec, while significantly improving the speedup.

### Strengths
1. The observation of high entropy leading to incorrect trace is interesting, and well investigated. And the proposed method is weel aligned with the observation.
2. The experiments are thorough, with multiple LLMs and benchmarks, showing the benefits from R-Stitch.
3. The ablation study is well-designed, justifying the design choice.

### Weaknesses
1. The choice of SLM is not reasonable. L1-1.5B-Short is used as SLM, while the target model is DeepSeek-R1 family. As we know, SpecDec is efficient when both draft and target model's distribution is aligned. From Table 1, SpecDec's speedup is even worse than the target model alone, which is unreasonable. It's suggested to include new results with SLM from the same family.
2. Lack of baselines. Only two baselines are included here, LLM and SpecDec. It's suggested to include recent strong baselines for better justification, like:

[1] Reward-Guided Speculative Decoding for Efficient LLM Reasoning

[2] AdaEDL: Early Draft Stopping for Speculative Decoding of Large Language Models via an Entropy-based Lower Bound on Token Acceptance Probability

### Questions
### Suggestions
1. Better to highlight the best results in Table 1.

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
2

### Summary
Authors propose entropy-guided hybrid decoding which switches between a SLM and LLM during chain-of-thought reasoning:

1. They uses normalized token entropy H_t = -Σp_{t,i}log(p_{t,i})/log(V) as an uncertainty proxy
2. SLM to LLM when entropy > τ (high uncertainty)
   LLM to SLM when entropy <= τ (low uncertainty, save computation)
3.  Maintains separate caches with partial prefill to minimize switching overhead

They add a learned routing policy via RL:
- Lightweight router decides when to invoke LLM for high-entropy tokens
- R = r_acc - λ·r_acc·L (penalizes latency only when correct)
- Linear regression models T(N_inf, N_kv) to avoid profiling overhead during training
- Group-normalized advantages for policy gradient

## Here are my thoughts
The paper conflates two distinct phenomena:
- Model disagreement (what speculative decoding addresses)
- Verbosity differences (SLM produces shorter traces)

The core claim that "speculative decoding's rigid token-level consistency prevents using SLM's conciseness" is misleading?? Speculative decoding doesn't prevent conciseness, it enforces correctness. If the SLM produces a correct but shorter solution, speculative decoding will accept it. The issue is that SLMs are often wrong, not just verbose.

2. Unfair Experimental Comparisons?
- Speculative decoding baseline uses high-agreement pairs (Distill-7B + Distill-1.5B) while R-Stitch uses low-agreement pairs (LLM + L1-Short)
- Is this maube backwards? Speculative decoding should get the favorable pairing?
- Are these cherry-pick scenarios where speculative decoding fails while giving R-Stitch optimal conditions?

3. Lack of Correctness Guarantees:
- R-Stitch discards SLM tokens and overwrites with LLM when entropy is high
- This creates a correctness risk: What if the SLM was actually right but uncertain? The LLM might introduce errors
- It would be nice to have an analysis of cases where switching to LLM hurts accuracy
- Table 1 shows accuracy drops in many settings (e.g., 7B on AIME: 33.33->30.00 at τ=0.03)

4. Entropy Is a Weak Proxy?
The empirical analysis (Section 3.2.1) is maybe superficial?
- Figure 3a: "Incorrect answers have higher entropy" - but correlation does not equal causation
- No comparison to other uncertainty measures (variance, top-k probability gaps, etc.)
- Figure 3c: "Harmful tokens have higher preceding entropy" - but the effect size is tiny (~0.028 vs ~0.024)
- 10.65% of tokens exceed entropy 0.1 - does this means the routing decision is rarely invoked?

5. Speedup Claims
- Peak speedups (3.00x, 3.85x, 4.10x) come with significant accuracy drops
- At τ=0.02 (claimed "sweet spot"), accuracy often decreases 2-5 points
- Speculative decoding maintains accuracy by construction
- It would be nice to report Pareto frontiers 


7. Method
- Entropy-based routing is already used in early exit, mixture-of-experts, etc.
- Is the "stitching" framing marketing? This is just conditional execution?
- R-Stitch+ is standard REINFORCE with a domain-specific reward
- The related work section (A.5) acknowledges EAGLE, Griffin, Hydra do similar things but claims they "increase consistency" while R-Stitch "exploits inconsistency" - I might be wrong, but this feels like  a false dichotomy

9. Minor Issues
- No error bars or confidence intervals (use SE if you can)
- Small test sets (30 samples on AIME)
- Table 1 shows accuracy increasing from 66.27 (LLM) to 77.11 (R-Stitch τ=0.001) on AMC 8k - this suggests variance, not real improvement?

## Fundamental Question

Why not just use the LLM with early stopping? If the issue is that LLMs are verbose, methods like DEER (which the paper briefly mentions) achieve similar latency reductions without the complexity of dual models. The paper doesn't convincingly argue why heterogeneous model collaboration is necessary?


I'm recommending weak accept, but I can move my score up if you address my issues. Thanks.

### Strengths
Strengths are in the above review.

### Weaknesses
Weaknesses are in the above review.

### Questions
Why not just use the LLM with early stopping? If the issue is that LLMs are verbose, methods like DEER (which the paper briefly mentions) achieve similar latency reductions without the complexity of dual models. The paper doesn't convincingly argue why heterogeneous model collaboration is necessary?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes R-Stitch, a bidirectional, entropy-guided collaboration between a small language model (SLM) and a large language model (LLM) for chain-of-thought reasoning. Decoding starts on the SLM, switches to the LLM when the SLM’s token entropy exceeds a threshold, and switches back once the LLM becomes confident, which preserves the SLM’s concise spans while using the LLM for hard tokens. To keep switching efficient, the system maintains separate KV caches and performs partial prefill so each model only recomputes tokens generated since the last switch. The R-Stitch+ variant adds a lightweight router trained with a latency-aware reward that penalizes runtime only when the answer is correct and relies on a profiled latency estimator to avoid online timing. Implemented in vLLM and evaluated on five math benchmarks with 7B, 14B, and 32B models, the method reduces per-sample wall-clock latency on a single A100 at batch size one, delivering roughly 1.4×–3.0× speedups at 7B/14B and up to around 4× at 32B under 8k–16k budgets while maintaining accuracy close to full LLM decoding.

### Strengths
* The core algorithm is simple and training free in its base form, using a clear entropy threshold to switch between SLM and LLM in both directions so that the system exploits concise SLM spans without sacrificing reliability on high-uncertainty tokens.
* The method is well motivated by an empirical analysis showing that incorrect answers have higher token entropy and that most tokens are very low entropy, which justifies entropy as a routing signal.
* The systems design is thoughtful, with explicit KV-cache management and partial prefill that reuses past caches on each model to avoid redundant attention and reduce switching overhead.

### Weaknesses
* The router is only described as a “lightweight” module fed by hidden states; its architecture, parameter count, placement, and per-token overhead are not reported, so deployability and reproduction costs are unclear.
* All latency results use a single GPU with batch size one. The current implementation only supports batch size one because switching happens at the token level. Real-world throughput under concurrent traffic is unknown.
* The system runs two engines with separate KV caches. This increases VRAM usage and system complexity. Partial prefill reuse is described, but the paper does not quantify memory costs or switching overhead.
* Performance depends on the entropy threshold. The paper tunes the threshold by sweeping values across a grid. There is no automatic rule that transfers across datasets and model pairs, which raises generability concerns of real-world deployments.
* The overall router design will make the system hard to scale and deploy comparing with methods like Eagle-3 etc.

### Questions
* Please specify the router: architecture, parameter count, compute placement, input features, and measured per-token overhead (and its share of end-to-end latency).
* For R-Stitch+, how robust is the learned policy across domains and model sizes? Please include cross-domain transfer results.

### Soundness
2

### Presentation
3

### Contribution
2
