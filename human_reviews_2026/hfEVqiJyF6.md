# EAT: Entropy After $\textlangle \tt /Think \textrangle$ for reasoning model early exiting

- Decision: Reject
- Scores: 6, 8, 2, 4

## Abstract
Large reasoning models show improved performance with longer chains of thought.
However, recent work has highlighted (qualitatively) their tendency to overthink, continuing to revise answers even after reaching the correct solution.
We quantitatively confirm this inefficiency by tracking Pass@1 for answers averaged over a large number of rollouts and find that the model often begins to always produce the correct answer early in the reasoning, making extra reasoning a waste of tokens.
To detect and prevent overthinking, we propose a simple and inexpensive novel signal---Entropy After `</Think>`
 (EAT)---for monitoring and deciding whether to exit reasoning early. 
By appending a stop thinking token (`</think>`) and monitoring the entropy of the following token as the model reasons, we obtain a trajectory that decreases and stabilizes when Pass@1 plateaus; thresholding its variance under an exponential moving average yields a practical stopping rule.
Importantly, our approach enables adaptively allocating compute based on the EAT trajectory, allowing us to spend compute in a more efficient way compared with fixing the token budget for all questions.
Empirically, on MATH-500 and AIME-2025, EAT reduces token usage by 13–21\% without harming accuracy, and it remains effective in black-box settings where logits from the reasoning model are not accessible, and EAT is computed with proxy models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a novel early exit method to address the overthinking problem in large-scale inference models. The authors first quantify this by tracking the Pass@1 metric, demonstrating that inference models often converge to the correct answer early in the inference chain, making subsequent token generation redundant. To address this, the paper proposes EAT (Entropy After </ think>) as a signal to monitor the inference process. This involves manually inserting </ think> tokens during inference and calculating the entropy of subsequent individual tokens to determine whether to exit early. Inference is terminated early when the variance of EAT, estimated via exponential moving average, falls below a threshold. Experiments show that this method can save 13-21% of token usage without sacrificing accuracy on datasets such as MATH-500 and AIME-2025, and it is also effective in black-box settings using a small proxy model to monitor a large inference model.

### Strengths
S1. The paper quantifies the "overthinking" phenomenon of the inference model through large-scale experiments (128 rollouts), which is clearly visible in Figure 1. The Pass@1 metric often saturates within the first 10-20% of the inference chain, yet the model continues to generate thousands of tokens. This finding is valuable because the output token cost of the inference model is far higher than the input token cost. In commercial APIs, output tokens are also typically more expensive. Therefore, the ability to adaptively allocate computational budgets has clear economic value and practical significance. The paper also promises to release over 20,000 GPU hours of inference trajectory data, a significant contribution to the community.

S2. EAT only needs to calculate the entropy of a single token after `<think>`. Compared to the baseline method, which requires generating a complete answer through rollout, EAT has a significant advantage of O(|R|) complexity, where |R| is the length of the reasoning chain, while the overhead of the rollout method is stochastic and increases linearly with the value of K. 

S3.The paper finds that EAT can be calculated using small surrogate models (e.g., 1.5B parameters), thereby enabling early exit control of large inference models (e.g., 70B parameters). This is very important in practical applications.

S4. The paper's experiments covered multiple dimensions: three different inference models (DeepSeek-Qwen-8B, Llama-70B, Qwen-4B), four different datasets (MATH-500, AIME-2025, GPQA multiple-choice and open-ended), and two EAT computation methods (white-box and black-box). Figure 4 systematically compares EAT with the token-based baseline, showing the performance-efficiency trade-off curves at different thresholds, which is more convincing than single-point comparisons.

### Weaknesses
W1. Although the paper proposes a correlation between EAT and information gain in Equation 6, this relationship is heuristic rather than rigorously derived. The core question is: why can the entropy of a single token reflect the certainty of the entire answer? The paper does not provide a theoretical analysis. Observing Figure 2, we see a relatively smooth monotonically decreasing trend only at the manually labeled "Conclusion" position, suggesting that EAT may not be an essential, robust signal, but rather happens to be related to certain specific model behaviors.

W2. Appendix G.3 of the paper candidly presents two types of failure cases: unsolvable problems (Figure 11) and Pass@1 descent problems (Figure 12). For unsolvable problems, EAT is always unstable, and the algorithm will exhaust all T tokens; for Pass@1 descent problems, EAT may stop at a suboptimal position or also exhaust all tokens. However, the paper does not propose any potential solutions or improvement strategies for these failure cases. In real-world applications, we don't know in advance which problems are solvable and which will cause a decrease in Pass@1. If the method fails in these situations, its practical value is reduced. The paper should design a mechanism to detect these situations.

W3. The paper only includes a "solvable subset" in the GPQA results shown in Figure 4, meaning they pre-filtered out problems with a final Pass@1 > 0.8 for evaluation. This operation is methodologically problematic: in real-world applications, we cannot know in advance which problems are solvable; the method must be evaluated on the complete dataset. Pre-filtering is equivalent to running the method in "easy mode," which will naturally result in better performance.

W4. The paper criticizes the #UA@K method in Figure 5 for its high overhead due to the need to generate K rollouts. However, this criticism is unfair because the paper's own evaluation metric, Pass@1 (Avg@128), requires 128 rollouts. This presents a contradiction: the authors criticize the overhead of the baseline method, but their own ground truth metric has the same overhead.

W5. Using `\n` directly as a symbol in formula (5) is informal; a special symbol for newline should be used. The layout of the multiple subplots in Figure 1 can be optimized; the current column arrangement makes the legend and labels somewhat crowded. While the term "overthinking" is descriptive, it may not be formal enough in academic writing; expressions such as "redundant reasoning" or "reasoning inefficiency" could be considered. The warm-up condition `n ≥ 4/α` in Algorithm 1 lacks explanation and theoretical basis. The labels "Line 16," "Line 55," etc., in Figure 2 should be given more context in the text to help readers understand why these positions are special. Some experimental details (such as why a temperature of 0.6 and a top-p of 0.95 were chosen) could provide more motivation.

### Questions
Q1. Could you provide a more in-depth theoretical analysis to explain why EAT can serve as an effective signal for early stopping?

Q2. For the two types of failure cases shown in Appendix G.3 (unsolvable problems and Pass@1 descent problems), can an auxiliary mechanism be designed to detect and handle them?

Q3. Is it possible to compare the EAT and #UA@K methods under a fixed computational budget (e.g., the same total FLOPs, GPU time, or API call overhead)?

Q4. Could you report the performance on the full GPQA-Diamond dataset, and not just on a solvable subset?

Q5. The paper primarily uses DeepSeek and Qwen family models. Can the effectiveness of EAT be validated on other model families (such as Meta's Llama family, Anthropic models, etc.)?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the next token distribution's entropy when the reasoning language model's thinking processes. The experiment results suggest a strong connection between the time when the model starts to overthink and the entropy goes down. Based on the observation, the authors introduce EAT to monitor the entropy signal, and stops thinking once it reaches a threshold. In experiemnts, EAT significantly reduces the token usage without accuracy drop, and can be extended to the black box models where the entropy cannot be accessed.

Compared to previous works on the reasoning model early exiting topic, EAT provides several advantages:
1. It does not require further training;
2. It does not introduce much redundant computation;
3. It supports black box models by computing EAT on a proxy model.

### Strengths
- The writing is easy to follow and well organized. The authors provide comprehensive experiments to naturally introduce the concept and effectiveness of EAT (and EMA of EAT), then perform experiments on using EMA of EAT for early exiting.
- The advantage of EAT (no need for further training, smaller redundant computation) is significant comparing to previous work.
- The authors provide evaluation of EAT on reducing reasoning tokens on several perspectives.

### Weaknesses
- The author provides several experiments to show the relation between EAT and the model's confidence. However, most of the examples in the paper are only for the positive cases where the task is easy: In Figure 1, the Pass@1 for the four sampled question all reaches 1. When the question itself is hard (Pass@1 after the whole thinking is less than 1 or even near 0), what is the relation between the EAT and Pass@1 / #Unique answers? Providing some data in this case may help improve the soundness of the motivation. The same question applies to Figure 2.
- The effectiveness of EAT against other early exiting methods (e.g. classifier based, sampling based) is not discussed in the experiments. Adding such baselines could help improve the significance of EAT.

### Questions
- When using a proxy model to estimate the EAT of a black box model, is it common that the computation of that proxy model can be even higher than the computation saved for the black box model?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper tackles overthinking in reasoning LLMs by proposing EAT (Entropy After \</think\>), a simple stopping signal that appends an end-of-reasoning marker (sometimes with an answer prefix) and measures next-token entropy. The key observation is that EAT stabilizes right around where Pass@1 accuracy plateaus, so they use an EMA + variance threshold to trigger early exit. On MATH-500, AIME-2025, and filtered GPQA, their offline simulations show about 6–22% token savings with minimal accuracy loss. They also show a smaller model can compute EAT as a proxy for a larger one. The signal is cheap and training-free, adapts to each instance, and the correlation with accuracy saturation is clear on R1/Qwen-style traces. However, the contribution is incremental because it remains within the same entropy/confidence-based halting paradigm, mainly shifting the probe to the token after an explicit stop and smoothing it, while omitting the most natural comparison point, single-pass answer-entropy halting. Also, the evaluation is post-hoc on single sampled traces, not live decoding. One other issue, which I trust to be a minor overlook from the authors is that some claims could be toned down. The abstract highlights 13–21% savings, but the actual results span 6–22% depending on task difficulty, only AIME hits about 21–22%.

### Strengths
- Clear Motivation and Problem specification
- Very simple, training-free method: one-token entropy after an explicit stop marker, plus EMA+variance sounds is easy to implement and cheap.
- Adaptive per-instance compute
- On AIME-2025 and similar structured tasks the method cuts about 13–22% tokens without hurting accuracy.

### Weaknesses
- The idea of using entropy for early stopping isn't new, but the paper doesn't compare against the most obvious alternatives, single-pass answer entropy or confidence-based stopping. These would be the most informative comparisons. This is an incremental contribution as it only adds EMA 
- All the gains come from clean, math-heavy problems (MATH-500, AIME-2025) with models that already output \</think\> and structured answers. We don't know if this works on messier traces, shorter reasoning, or different model formats.
- Efficiency gains are computed offline from pre-generated traces, not measured during actual decoding. Real-world savings might differ.
- Claims like "first quantitative demonstration" seem strong given recent related work 
- The method needs specific formatting (\</think\>, sometimes "Final answer:", certain newline patterns) and manual tuning (EMA warmup). Small changes in how the model structures output could break the signal.
- Savings do not generalise because on stronger models / harder datasets the reduction drops to about 6–11%, so the benefit isn’t consistent across settings.

### Questions
Refer to the weaknesses please and one more question:

Can you report how often your scheduled EAT checkpoints didn’t occur naturally (i.e. the model didn’t emit the expected \n\n / \</think\>), and what EAT did in those cases?

### Soundness
3

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
4

### Summary
The paper propose a novel early stopping mechanism by measuring the entropy after </think> (EAT) as a reliable confidence measure to guide the model early stopping. Specifically, EAT first measure the `[prefix]<think></think>` entropy, and monitor at every reasoning step  the entropy of `[prefix]<think>r1 r2 ... rn</think>` to obtain a its variance under an exponential moving average. If such value passes a threshold, the reasoning process can practically early stop with a good result. The evaluation and ablation shows good result of the method across different models and datasets.

### Strengths
1. The methodology (EAT) is sound. 
2. The paper evaluate EAT on open-ended questions and show appealing result with early stopping.
3. The writing is easy to follow.

### Weaknesses
1. The solution lacks novelty. The paper needs to make a stronger and thorough comparison between EAT and the other methods to demonstrate its novelty.
2. EAT may prematurely terminate reasoning when the model has not yet arrived at a correct answer. This could limit the model’s opportunity for self-correction and increase the risk of incorrect outputs.

### Questions
1. The paper claims to be "first quantitative demonstration that reasoning models frequently overthink, continuing to generate lengthy reasoning chains even after the prediction has stabilized", which is not quite true given work mentioned in related work such as [Fu et al.](https://openreview.net/forum?id=wpK4IMJfdX, https://arxiv.org/abs/2412.20993), Yong et al. (https://arxiv.org/pdf/2505.18237), and other quantitative works using entropy and similar methods to control the reasoning trajectory. I would suggest to strengthening the statement with tailored comparison against these works, or soften the statement. 
2. The paper choose to use exponential averaging and hyperparameter window size $\alpha$ and a predefiend threshold $\delta$. 
3. Does the method prematurely terminate reasoning when the model has not yet arrived at a correct answer? This could limit the model’s opportunity for self-correction and increase the risk of incorrect outputs. The paper would benefit from a more detailed discussion of such failure cases and how EAT behaves under different reasoning dynamics.
4. EAT does not stabilize when the question is unsolvable, meaning EAT will waste all the tokens, which seems to be a big limitation of the method. Is there any mitigation the authors want to propose?

### Soundness
3

### Presentation
3

### Contribution
2
