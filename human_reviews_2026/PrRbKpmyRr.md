# PAC Reasoning: Controlling the Performance Loss for Efficient Reasoning

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large reasoning models (LRMs) have achieved remarkable progress in complex problem-solving tasks. Despite this success, LRMs typically suffer from high computational costs during deployment, highlighting a need for efficient inference. A popular direction of efficiency improvement is to switch the LRM between thinking and nonthinking modes dynamically. However, such approaches often introduce additional reasoning errors and lack statistical guarantees for the performance loss, which are critical for high-stakes applications. In this work, we propose Probably Approximately Correct (PAC) reasoning that controls the performance loss under the user-specified performance loss tolerance. In particular, we construct an upper confidence bound on the performance loss, formulated as a monotone function of the uncertainty score, and subsequently determine a threshold for switching to the nonthinking model. Theoretically, using the threshold to switch between the thinking and nonthinking modes ensures bounded performance loss in a distribution-free manner. Our comprehensive experiments on reasoning benchmarks show that the proposed method can save computational budgets and control the user-specified performance loss.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper is very exciting, as it offers a theoretical discussion of the transition between thinking and non-thinking models. The experimental results demonstrate that the proposed solution, grounded in theoretical analysis, can effectively reduce inference token usage.

Overall, I believe this paper marks the beginning of a promising direction for efficient LLM reasoning. However, further exploration and validation are needed to solidify its contributions.

### Strengths
1. This paper discusses LLM reasoning from a theoretical perspective, which is highly valuable and can serve as a useful guide for future research in this area.
2. Both the assumptions and theoretical results are clearly presented and easy to follow, which will benefit the LLM reasoning community.

### Weaknesses
1. Definition 1 requires more careful discussion. The current formulation assumes that a non-thinking LLM differs from its thinking variant only with a very small probability. This assumption may be overly relaxed and overlooks a central challenge in this area: identifying the difficult problems that truly necessitate the use of a thinking LLM. The manuscript should explicitly address how such problems are distinguished. Additionally, in Line 141, the authors posit that the uncertainty score can represent the likelihood of disagreement between thinking and non-thinking LLMs. This claim requires further clarification. It is not reasonable to use a single uncertainty score in this way; rather, disagreement should be computed based on uncertainty scores from both LLMs.
2. The experimental results are relatively weak and should be strengthened by comparison with baseline methods [1] and more benchmark datasets [2]. The paper primarily proposes an efficient reasoning method with PAC guarantees. Therefore, the experiments should include cases where existing methods fail to guarantee reasoning performance, but the proposed method succeeds, thus substantiating the theoretical claims. Furthermore, since the paper aims to address LLM reasoning challenges, more applications should be considered, including common sense reasoning, more complex mathematical reasoning, and code generation tasks.
3. The assumption of having an i.i.d calibration set is hard to be satisfied in the real applications, since if we have this set why should we performance RL training to teach the model to switch between thinking and non-thinking mode?

**Reference**

[1] Stop Overthinking: A Survey on Efficient Reasoning for Large Language Models. Trans. Mach. Learn. Res. 2025.

[2] A Theoretical Study on Bridging Internal Probability and Self-Consistency for LLM Reasoning. NeurIPS 2025.

### Questions
Please refer to the questions  in the `weaknesses` section. Moreover,

1. How do the theoretical results presented in this paper extend to other efficient LLM reasoning methods that employ similar core ideas? Furthermore, how does the proposed method compare to these existing approaches in terms of efficiency and practical performance?
2. This method requires inferring all problems with the non-thinking LLMs in advance to obtain the uncertainty scores. Does this operation introduce significant additional computational costs, and do other comparable methods also require such a preliminary step?
3. The experimental LLMs is limited, as it only considers the 4B variant of the Qwen model. This restriction may not adequately verify the effectiveness of the proposed methods on broader ranges of LLMs. 
4. Recent LLMs are capable of controlling their reasoning efforts. Can the theoretical analysis presented in this paper be applied to such models?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces PAC reasoning, a simple yet effective and theoretically sound uncertainty-thresholding framework. It provides distribution-free guarantees on performance degradation while improving inference efficiency. By constructing UCBs over an uncertainty score and switching between “thinking” and “non-thinking” modes, the method achieves target error tolerances with substantial token savings across several reasoning benchmarks. Both theoretical analyses (e.g., empirical risk bounds) and empirical evaluations (e.g., ECP/STP metrics) further substantiate its effectiveness.

### Strengths
- The problem and intuition are clearly articulated. The paper formalizes the goal of controlling performance under an $(\varepsilon, \alpha)$-PAC efficient model, and successfully reduces the problem to a tractable yet powerful formulation based on uncertainty-driven switching between “thinking” and “non-thinking” modes.
- The theoretical foundation appears sound. The authors derive distribution-free risk guarantees under mild assumptions and further extend the analysis to both empirical-risk and transductive settings, significantly enhancing the generality of the results. Moreover, they present upper confidence bounds (UCBs) based on both the Central Limit Theorem and concentration inequalities, covering large- and small-sample regimes.
- The empirical results are also promising. The paper introduces metrics such as Expert Calling Percentage (ECP) and Saved Token Percentage (STP) to quantify computational efficiency, offering a practical perspective on the performance–cost trade-off. Experiments across diverse benchmarks (MATH-500, ZebraLogic, Arena-Hard) consistently demonstrate the effectiveness of PAC reasoning.
- Overall, the paper is well-written, clearly structured, and easy to follow.

### Weaknesses
- It is advantageous that the proposed framework does not rely on ground-truth labels, instead leveraging the “expert answers” provided by the thinking model. However, the performance of PAC reasoning is also limited and upper-bounded by that of the thinking model, especially on more challenging tasks.
- The theoretical guarantee hinges on the monotonic relationship between uncertainty and the risk function. In practice, if uncertainty scores are noisy or poorly calibrated (especially for verbalized uncertainty), this assumption may break.
- The calibration phase introduces additional expensive queries to the expert (thinking) model. The associated compute and latency overheads are not explicitly analyzed.
- Experiments mainly use Qwen3-4B (thinking/instruct). The generality of results across larger models, different architectures, or modalities remains unexplored.

### Questions
- The PAC guarantee is defined relative to the thinking model rather than ground truth; how does this affect reliability if the expert model itself is biased?
- How sensitive is the method to violations of the assumed monotonic relation between uncertainty and error, especially for verbalized uncertainty?
- What is the actual compute cost of the calibration phase, and under what conditions does it outweigh or offset the token savings during inference?

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
4

### Summary
This paper proposes PAC reasoning, a method to make large reasoning models more efficient by dynamically switching between expensive "thinking" and cheap "non-thinking" modes. The authors use a calibrated uncertainty threshold, which ensures the performance loss stays below a user-specified limit with a statistical guarantee, saving computation resources while provably controlling errors. They conduct experiments to demonstrate the effectiveness of the proposed method.

### Strengths
1) The source code is available.
2) The effectiveness of the proposed method is verified theoretically.
3) The experimental settings are clearly explained.

### Weaknesses
1) PAC reasoning leverages the uncertainty of LLMs which is not always reliable.
2) The authors may need yo compare the existing reasoning methods with the proposed PAC reasoning method in main experimental results (e.g., Figure 2).
3) In addition to Qwen3, more LLMs can be considered in experiments.
4) In the math experiments, the y-axis reports the error which essentially reflects the guarantee of the predicted answer. How about the performance metrices, such as pass@k.

### Questions
1) What does EL(u) mean in Eq.(2)? 
2) Why dose PAC reasoning use importance sampling to construct UCB and why are sampling weights $\pi_i$ all the same (0.5)? The authors should give more explanations about this.
3) In Table 3, can the introduction of an additional calibration set lead to unfair comparisons of naive methods?
4) When the uncertainty scores of LLMs are unreliable, PAC reasoning can be useless. Are there any solutions to this problem?

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
This paper addresses an increasingly practical problem in reasoning LLMs: slow, expensive "thinking" modes yield higher accuracy, while fast "non-thinking" modes are cheaper but less reliable. Existing approaches to dynamically switch between these modes often lack statistically guaranteed performance bounds. The authors propose PAC Reasoning, a framework that uses a high-quality thinking model, and a faster model, along with a calibration dataset to learn an uncertainty threshold. At inference time, if the uncertainty score from the fast model is below, the system uses the fast model's prediction, otherwise, it falls back to the thinking model. Experiments on MATH-500, ZebraLogic, and Arena-Hard show that PAC Reasoning can reduce token usage by 20–40% while keeping empirical performance loss within the targeted tolerance (e.g., $\varepsilon=0.08$). They also compare logits-based vs. verbalized uncertainty measures, finding the former more stable and effective.

### Strengths
- The problem formulation is clear and tractable. The authors explicitly frame the problem as "controlling performance loss under efficiency constraints," turning a heuristic area (adaptive reasoning) into a principled, quantifiable one.
- The theoretical guarantees are solid. The PAC-style result gives a distribution-free bound on performance loss, a property rarely seen in existing efficiency-switching methods.
- The approach is model-agnostic. The framework only requires two models (thinking & nonthinking) and a measurable uncertainty score. It’s conceptually simple and easy to apply to different architectures.
- The experiments span math, logic, and open-ended reasoning tasks, showing generality across task types.

### Weaknesses
- The theoretical guarantee only holds if the uncertainty score correlates well with the probability of disagreement between fast and slow models. In practice, this is a strong assumption, as the uncertainty of LLMs is often unreliable.
- Algorithm 1 requires running the expensive reasoning model on a calibration set to compute the empirical loss curve and its UCB. For large models, this cost may offset much of the efficiency gain.
- The method uses a single threshold for all inputs. Different task types or reasoning depths might require different thresholds. Extending to conditional or multi-threshold routing could yield better efficiency.

### Questions
1. How often must calibration be re-run in practice, and how large must the calibration set be to maintain valid coverage? Have you analyzed the trade-off between calibration cost and runtime efficiency gains?
2. Could the method be extended to online or drift-aware settings, where the threshold is updated over time (e.g., using anytime-valid or sequential testing techniques)?
3. Could this framework handle multi-level reasoning chains (e.g., fast model -> hybrid model -> long thinking model)? The current binary setup seems restrictive.

### Soundness
3

### Presentation
3

### Contribution
2
