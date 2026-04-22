# QTALE: Quantization-Robust Token-Adaptive Layer Execution for LLMs

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Large language models (LLMs) demand substantial computational and memory resources, posing challenges for efficient deployment. Two complementary approaches have emerged to address these issues: token-adaptive layer execution, which reduces floating-point operations (FLOPs) by selectively bypassing layers, and quantization, which lowers memory footprint by reducing weight precision. However, naively integrating these techniques leads to additional accuracy degradation due to reduced redundancy in token-adaptive models.
We propose QTALE (Quantization-Robust Token-Adaptive Layer Execution for LLMs), a novel framework that enables seamless integration of token-adaptive execution with quantization while preserving accuracy. Conventional token-adaptive methods reduce redundancy in two ways: (1) by limiting the diversity of training paths explored during fine-tuning, and (2) by lowering the number of parameters actively involved in inference. To overcome these limitations, QTALE introduces two key components:
(1) a training strategy that ensures diverse execution paths are actively explored during fine-tuning, and
(2) a post-training mechanism that allows flexible adjustment of the execution ratio at inference to reintroduce redundancy when needed.
Experimental results show that QTALE enables seamless integration of token-adaptive layer execution with quantization, showing no noticeable accuracy difference, with the gap to quantization-only models kept below 0.5\% on CommonsenseQA benchmarks.
By combining token-adaptive execution for FLOPs reduction and quantization for memory savings, QTALE provides an effective solution for efficient LLM deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Prior works (D-LLM (Jiang et al., 2024)) proposed token-adaptive layer execution on LLMs where each layer is adaptively bypassed according to a "router" neural network's decision on each token's layer representation. This paper extends D-LLM by two tricks: 1) A entropy regularization loss is designed to promote execution path exploration during training; and 2) the execution of each layer is controlled by a tunable probability threshold $\theta$ after training.

### Strengths
- This paper addresses the problem of token-adaptive layer execution with quantization, exploring the tradeoff of inference efficiency and model accuracy on the dimension of quantization.

### Weaknesses
- The proposed method is mainly based on D-LLM with a several heuristic adjustment and does not provide significant improvement on performance.

### Questions
- Why the authors introduce $\log$ in Eq. (5) if the actual implementation does not involve taking the logarithm of router outputs? Here we should be specific whether the sampling of router decision is based on ${\rm softmax}(\hat g_{\ell})$ or ${\rm softmax}(\log(\hat g_{\ell}))$ in **the actual implementation**.
- Instead of an additional regularization loss used in Eq. (7), have the authors considered converting the router's logits into probabilities by computing $\text{softmax}(\hat g_{\ell})$ and then sample according to 
$$\hat b_{\ell} = \mathbb{1}(\arg \max_i (\log ({\rm softmax}(\hat g_{\ell})_i) + \pi_i)) \text{ for } \pi_i \sim {\rm Gumbel}(0,1)$$
which is similarly implemented in original Gumbel Softmax sampling trick by [a]? This should resolve the issue of the observed large magnitude logits in Figure 5.
- What are the values of $\theta$ adopted in Table 1 and 2? Are the layer execution rates in Table 3 corresponded to the models evaluated in Table 1 and 2?
- What dataset is used for the fine-tuning phase?

[a] Jang, Eric, Shixiang Gu, and Ben Poole. "Categorical reparameterization with gumbel-softmax." arXiv preprint arXiv:1611.01144 (2016).

### Soundness
2

### Presentation
2

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
The paper proposes QTALE, a framework for combining token-adaptive layer execution with low-bit quantization in LLMs. Token-adaptive methods reduce FLOPs by skipping layers per token, while quantization reduces memory cost by lowering precision. When applied together, they often harm accuracy because token-adaptive models already reduce redundancy. QTALE addresses this by introducing (1) a quantization-robust training strategy that preserves training-path diversity using entropy regularization on routing logits, and (2) an inference-time execution-ratio control mechanism that adjusts the global threshold for layer execution to reintroduce redundancy as needed. Experiments on LLaMA-2-7B and LLaMA-3.1-8B show that QTALE maintains accuracy close to quantized full-model baselines (within 0.5% on CommonsenseQA) while offering significant reductions in both FLOPs and memory footprint.

### Strengths
* The motivation is clear, addressing a practical but overlooked incompatibility between quantization and token-adaptive execution.
 * The proposed solution is conceptually simple yet grounded in a solid understanding of redundancy loss in adaptive models.
 * Strong empirical coverage across multiple datasets and model scales; results consistently show improved robustness under quantization compared to D-LLM.
 * Inference-time controllability via a single threshold provides flexibility for deployment-time efficiency–accuracy trade-offs.

### Weaknesses
* The method’s generality beyond AWQ quantization and LLaMA-based architectures remains untested.
 * The claim of being “quantization-robust” would be stronger if supported by additional analysis under more aggressive or diverse quantization regimes (e.g., 2-bit or mixed-precision settings), even if such configurations are mainly diagnostic rather than practical for deployment.

### Questions
* How sensitive is the method to the choice of entropy regularization weight and inference threshold? A more systematic study across a wider range of values could help clarify the robustness of these hyperparameters. The ablation studies only show binary options where the hyperparameters either "turned on" or off.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Integrating token-adaptive layer execution with quantization causes additional performance degradation on large language models.
The authors identify that the degradation primarily stems from reduced training-path and parameter redundancy.
To enhance training-path redundancy, they introduce Gumbel noise during training to inject randomness into the path generation process. 
Additionally, an entropy regularization loss is employed to narrow the logit gap between bypassed and executed layers, encouraging more balanced training across paths.
To improve parameter redundancy, the authors adopt a post-training calibration procedure that searches for an optimal threshold to determine whether a layer should be executed or bypassed.
Experimental results demonstrate that the proposed method effectively mitigates the performance gap between the token-adaptive quantized model and a quantization-only baseline, reducing it to less than 0.5%.

### Strengths
1. The paper is well-structured and clearly written. The authors effectively illustrate the key challenges through intuitive figures and provide corresponding solutions that are logically motivated and easy to follow.
2. The paper offers a thorough background discussion, particularly in Section 2.2, which helps readers who may not be familiar with token-adaptive layer execution to understand the motivation and technical context of the work.
3. The authors conduct detailed ablation experiments to validate both the training-time and post-training components of the proposed QTALE framework, providing solid empirical evidence for the effectiveness of their design choices.

### Weaknesses
1. The motivation for QTALE could be further elaborated. In Section 3.1, the authors identify reduced training-path redundancy as a property of token-adaptive models, but it remains unclear why this phenomenon specifically leads to accuracy degradation when integrating token-adaptive layer execution with quantization. Providing either a more detailed theoretical explanation or supporting empirical evidence would make the motivation more convincing.
2. The connection between the proposed QTALE framework and quantization could be clarified further. While QTALE modifies the token-adaptive execution mechanism, it does not appear to include design choices specifically tailored for quantization. It seems possible that QTALE could also benefit other compression methods (e.g., sparsification). It would improve the completeness of the paper if the authors could either (a) clarify the specific relationship between QTALE and quantization, or (b) provide supplementary experiments applying QTALE to other compression techniques to demonstrate its generality.
3. In Tables 2 and 3, the comparison between QTALE and D-LLM appears somewhat unfair. In particular, on the Alpaca dataset, QTALE achieves better performance than D-LLM; however, as shown in Table 3, QTALE requires approximately 33% more computational cost. This suggests that the improved performance may partially result from the increased computation rather than the method itself. It would strengthen the paper to include an additional experiment where D-LLM is evaluated under a higher ​$R_{target}$ setting to ensure a fair comparison.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

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
This paper addresses the challenge of combining token-adaptive layer execution with post-training quantization for efficient LLM inference. Prior work (e.g., D-LLM) has shown that token-adaptive execution can reduce FLOPs by skipping layers, while quantization reduces model size by lowering weight precision. However, naively applying both causes severe accuracy drops due to compounded loss of redundancy. The proposed QTALE framework introduces two methods to overcome this:
* An entropy regularizer is added to the router’s loss to keep the execute/bypass logits balanced. This maintains higher entropy in the routing decisions during fine-tuning, forcing a diversity of execution paths (akin to dropout) and keeping more layers “alive” in training. In effect, the router logits stay in a moderate range so that Gumbel noise (used during training) continues to flip decisions, preventing collapse to a fixed subset of layers.

* At inference time, QTALE replaces the hard argmax decision with a softmax-based threshold. A global threshold θ on the execute-class probability allows tuning the model’s overall execution ratio without retraining. By lowering θ (below 0.5), more layers are executed and parameter redundancy is reintroduced to absorb quantization noise. This threshold is calibrated via a small grid search.

Together, these components enable QTALE to match the accuracy of unquantized, full-layer models while enjoying the combined efficiency benefits. On benchmarks (CommonsenseQA suite, MMLU, Alpaca PPL) with 4-bit and 3-bit quantization, QTALE incurs negligible accuracy loss compared to quantization-only LLMs. For example, on CSQA with LLaMA2-7B at 3-bit, the full-model accuracy is 72.22%, D-LLM drops to 70.57%, but QTALE recovers it to 72.79%. In aggregate, QTALE keeps the accuracy gap vs. the quantized baseline under 0.5%, while halving the model FLOPs and size. Thus, QTALE successfully unifies token-adaptive execution and quantization without significant performance degradation.

### Strengths
The paper targets a timely issue: reducing LLM inference costs in both computation and memory. It correctly observes that token-adaptive skipping and low-bit quantization are complementary, but the naive combination fails. Addressing this gap is practically important.

The proposed methods are clearly described and sound. Introducing an entropy regularizer on router outputs to maintain path diversity is a principled idea (inspired by dropout/stochastic-depth). The inference-time threshold mechanism is simple yet effective for fine-grained control of redundancy. Both are novel in this context and well-justified by the analysis of D-LLM’s failure modes.

The authors evaluate two large models (LLaMA2-7B, LLaMA3.1-8B) and multiple benchmarks (seven CommonsenseQA tasks, MMLU, Alpaca). Both 4-bit and 3-bit quantization are tested. Results show that QTALE retains accuracy close to the quantized full models, whereas the naive D-LLM+quant approach drops noticeably. The paper reports both accuracy and perplexity metrics, and averages across tasks.

Table 4 and its discussion isolate the effects of each component. As noted, entropy regularization boosts path diversity (narrowing the logit gap) without changing the execution ratio, while threshold tuning directly increases executed layers. This ablation confirms that both ingredients are necessary to restore accuracy after quantization. The inclusion of detailed analysis (e.g., router logit histograms, flipping ratios) is a plus.

The paper quantifies reductions in model size and FLOPs. For example, with token-skipping, the model’s per-token FLOPs drop to ~0.5× of full execution for CSQA and MMLU. Quantization reduces the size from ~12–15 GB down to 3–4 GB. QTALE achieves a good trade-off: it nearly matches D-LLM’s FLOPs savings (about 50% relative to full) while enabling the smaller quantized memory footprint.

The paper is well-structured, starting with background on pruning, token-adaptivity, and quantization. Figures (e.g., showing layer execution ratios) help illustrate the problems of D-LLM. Equations and algorithmic details (Lentropy, threshold rule) are clearly given. The writing is generally clear and concise. Table captions and metric definitions are precise.

### Weaknesses
The ideas, while useful, are relatively incremental. Entropy or diversity regularization is a known trick (e.g., in stochastic-depth, dropout), and thresholding probabilities is conceptually simple. It would strengthen the paper if the authors discussed related techniques in conditional computation or prior gating papers more explicitly. As it stands, the novelty claim rests on applying these ideas to the quantization setting.

QTALE introduces extra hyperparameters (entropy weight λ₂ and threshold θ). The paper does not specify how λ₂ was chosen (only says “balances Lentropy”), which may impact reproducibility. Moreover, the threshold search requires a calibration dataset and a grid search at inference time, which adds complexity. This procedure is somewhat heuristic; alternative adaptive or learned thresholding schemes could be considered.

The efficiency gains are reported in terms of FLOPs and model size, but no actual latency or throughput measurements are provided. In practice, the routers themselves add some overhead per layer (even if small MLPs), and skipping layers may have irregular memory access patterns. It would be useful to see wall-clock speedups on hardware.

The fine-tuning setup follows the D-LLM paper, but exact hyperparameters (learning rate, batch size, λ₁, λ₂, number of epochs) are not reported. Without these details or the released code, replicating the results may be challenging. The reliance on a prior paper for training configs weakens reproducibility.

All experiments are in a zero-shot evaluation regime on question-answering and reasoning tasks. It is unclear how QTALE performs on other tasks (e.g. summarization, generation, or even dialogue). The paper also only tests two model sizes; exploring a very large model (e.g.,>10B) or a smaller model could help verify the approach’s generality.

There are a few typos (e.g., “Implementation” in Sec. 4.1) and some formatting quirks in tables (the alignment of columns in Table 3 is hard to parse). Clarifying these (and ensuring all abbreviations are defined on first use, e.g., “PTQ” for post-training quantization) would improve readability.

### Questions
Specify all tuning choices: e.g., the values of λ₁ (execution ratio loss) and λ₂ (entropy weight), learning rate, number of epochs, and batch sizes. Including these (perhaps in an appendix) or in released code would aid reproducibility.

Instead of grid search, consider a more principled way to set θ, such as learning it via a small auxiliary dataset or using a schedule based on model confidence. This could reduce the need for calibration.

Explore whether combining QTALE with quantization-aware training (QAT) further improves robustness. The current work applies PTQ (AWQ). Allowing the model to see quantization noise during fine-tuning might synergize with the proposed method.

Test on additional domains (e.g., translation, summarization) to demonstrate generality. Also, if possible, include end-to-end latency benchmarks on GPU/TPU to confirm that the theoretical FLOPs savings translate into real speedups.

It may be insightful to compare against other efficiency methods: for instance, early exit techniques or structured pruning combined with quantization. This would contextualize QTALE’s benefits relative to alternatives.

### Soundness
3

### Presentation
3

### Contribution
2
