# AdaMuon: Adaptive Muon Optimizer

- Avg Score: 4.80
- Decision: Reject
- Scores: 4, 4, 6, 4, 6

## Abstract
We propose AdaMuon, a novel optimizer that combines element-wise adaptivity with orthogonal updates for large-scale neural network training. AdaMuon incorporates two tightly coupled mechanisms: (1) an element-wise second momentum estimator applied to orthogonalized update directions, and (2) a sign-stabilized orthogonal update, where the momentum is first sign-transformed before orthogonalization. These two components jointly enable variance-adaptive scaling while maintaining stable update geometry.
In addition, AdaMuon employs an RMS-aligned rescaling strategy to match the root-mean-square update magnitude to Adam, allowing direct reuse of existing learning rate schedules without extra tuning. Experiments demonstrate that AdaMuon not only maintains stability but can surpass Adam by more than 40% training efficiency in large-scale scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a new variation of the Muon optimizer with two tweaks: (1) Applying an element-wise sign operation before orthogonalization; (2) Applying an Adam/RMSProp like elementwise normalization of the updates. The authors argue intuitively for the benefit of this and then perform an experimental comparison on LLM pre-training at a fairly good model scale and training duration. The results indicate a decent speedup over both Adam and Muon.

### Strengths
* The problem studied (efficient optimizers for pre-training) is interesting and valuable to the community.
* The experiments are done sufficiently large scale, not just toy settings.
* The paper presentation, writing, and figures are clear.

### Weaknesses
* While the experiments are done on a decent scale, they are not very well-controlled. Hyperparameter configurations are somewhat arbitrary and comparing optimizers without thorough tuning of each varient is not very informative.
* The conceptual arguments for the necessity of both tweaks feel hand-wavy and are not convincing to me. 
* Certain experimental details are not given which makes it even harder to evaluate the soundness and conclusions of the experiments.
* The paper does not discuss any downsides to their approach. Yet there are definately some, for example the additional memory needed for second moment buffer. The addition of epsilon and beta2 are also extra hyperparameters. These can be shared with Adam when it is used for other hyperparameters, but when e.g. Lion is used this is no longer the case. Finally there is likely some (minor) runetime overhead from the additional operations that should at least be mentioned.

### Questions
Experimental configuration:
* What learning rate schedule is used? Is the learning rate kept constant after the warmup or is it decayed?
* What is the exact variant of Muon used for the baseline? Is it the Kimi-variant or another one?
* What batch size is used for the experiments?

Suggestions for strengthening the arguments of the paper:
* Sweeps over different learning rates. This could be done for shorter training runs (around 20 tokens/parameter). Ideally the learning rate for the Adam parameters should be swept separately and kept constant during the sweep for the Muon learning rate. 
* Ideally the experiments should be done for two different batch sizes, one large and one small. Certain operations like sign can have very different impacts for small and large batch sizes.
* It is not clear at all whether we should normalize updates at a scalar level. Some papers like Adam-mini (https://arxiv.org/abs/2406.16793) argues this is not the case. It would be nice if the justification for the method was much stronger, ideally backed up by either mathematical arguments or measurements that clearly demonstrate the problem you are trying to address and the effectiveness of the solution.

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
3

### Summary
This work expands the Muon Optimizer with the concept of variance-adaptive scaling, using an element-wise second momentum estimator applied to orthogonalized directions and a sign-stabilized orthogonal update, where momentum is sign-transformed before orthogonalization.

### Strengths
- The authors address a highly relevant and impactful topic. The motivation for their proposed approach is clear and well-articulated.
- The method is validated through extensive experimentation. This includes large-scale evaluations (e.g., 7B parameter models trained on 250B tokens) and thorough ablation studies on methodological decisions.

### Weaknesses
- The author’s claim of 40% training efficiency is based on training loss. However, it is unclear whether this gain translates to downstream tasks. This makes it difficult to assess the practical impact of the method.
- The main body is missing key details about the experimental setup. For example, the authors refer to a "warmup-stable policy" without defining the term.

### Questions
- Have the authors considered evaluating the AdaMuon optimizer result at 29.6% training efficiency on downstream tasks? This would provide valuable insight into the practical use of the training efficiency gains.
- Could the authors please clarify the specific learning rate schedule referred to as the “warmup-stable policy”? Does this refer to WSD (Hägele et al. 2024), or is it a schedule that omits the decay phase entirely? If the decay phase was omitted, investigating the optimizer's behavior during this phase seems like a valuable experiment, as this is often when noticeable performance improvements are observed.

### Soundness
2

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
3

### Summary
The paper proposes AdaMuon, an optimizer that augments Muon’s geometry-preserving orthogonal updates with element-wise adaptivity. Concretely: (i) it makes an estimation of a second moment on the orthogonalized direction, (ii) it stabilizes orthogonalization through applying an element-wise Sign transform to the momentum before the polar step (with a theorem arguing Sign is the canonical choice), and (iii) it RMS-aligns the update so its magnitude matches Adam’s ( ≈0.2), facilitating reuse of standard LR schedules. Extensive experiments on GPT-2 (125M–1.5B) and Qwen2.5 (1.5B & 7B) report token-efficiency and time-to-target speedups over Muon optimizer and AdamW optimizer, with up to ~30–40% efficiency gains vs Adam and small yet consistent gains vs the Muon optimizer.

### Strengths
1. Combining sign-stabilized orthogonal directions with element-wise EMA and RMS alignment is practical and simple; Alg. 1 is relatively easy to implement.

2. Thm. 1 provides a clean rationale for the sign transform due to the unique admissible element-wise map under natural constraints; the appendices provide a convergence discussion with assumptions stated explicitly.

3. On GPT-2 (4 scales) and Qwen2.5 (1.5B/7B) models, AdaMuon optimizer outperforms AdamW optimizer and slightly exceeds Muon optimizer in the aspect of token efficiency; the ablation studies also manifest that both sign and variance matter.

4. Alignment of RMS is a nice practical detail that enables drop-in learning-rate schedules.

### Weaknesses
1. Comparisons are limited to the AdamW optimizer and the Muon optimizer, despite many relevant contenders ( for example, Adafactor/Shampoo/Lion optimizers). This paper explicitly argues that these two baselines “suffice”.

2. There exists little analysis on the choice of RMS target (0.2), ε, β, or the number of Newton–Schulz steps; the argument for omitting bias correction hinges on RMS alignment canceling multiplicative bias, but early-phase, non-constant bias or heavy-tailed coordinates might violate the simplification. 

3. The convergence depends on alignment assumptions that are plausible but not established empirically; no finite-width stochastic analysis is provided.

4. The per-step time is a little higher than that of the AdamW or Muon optimizer, though wall-clock to target still improves; it is better for the authors to clarify kernel efficiency and implementation details.

### Questions
Could the authors supplement strong preconditioned or adaptive baseline methods (Adafactor, Shampoo/SM3, Sophia, Lion) and report both token-to-target and downstream evaluation results?

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
This paper introduces AdaMuon, a Muon variant for large-scale neural network training that aims to combine the geometric stability of the Muon optimizer with the element-wise adaptivity of Adam. The core idea is to enhance Muon's orthogonal updates with coordinate-wise variance scaling. To achieve this, AdaMuon incorporates a sign operation before polar decomposition, an element-wise second moment estimator similar to Adam's, and RMS-aligned rescaling that normalizes the final update magnitude to match Adam's. Experiments are conducted on GPT-2 and Qwen 2.5 models to evaluate the proposed method.

### Strengths
1. The work targets the development of more efficient and stable optimizers for training massive foundation models, which is a significant problem.
2. Combining Muon and Adam is an interesting research direction, and the experimental results are promising.

### Weaknesses
1. There is a disconnect between the stated motivation for the sign operation (stabilizing early-stage training) and the empirical results from the ablation study (Figure 3), which show the performance gap appearing in later stages. This questions the authors' understanding of why their own method works and weakens the overall narrative.
2. The paper presents AdaMuon as a principled combination, but the sign operation is a hard, non-linear transformation that fundamentally alters the information that both Adam (gradient magnitudes) and Muon (momentum matrix structure) rely on. This makes the method appear more like a heuristic rather than a seamless integration, and the theoretical justification feels insufficient to cover this drastic step.
3. The paper's claims of superiority over baselines rest on a very limited set of learning rates (6e-4 and 1e-3). A robust comparison requires a more comprehensive learning rate sweep for all optimizers.
4. The downstream performance of AdaMuon is not consistently or significantly better than the simpler Muon (Table 2). This makes the trade-off between its added complexity and its benefits less compelling.

### Questions
See above.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes AdaMuon, a new optimizer that combines the geometry-aware stability of Muon with the adaptive scaling ability of variance-based methods like Adam. It introduces sign-stabilized orthogonal updates and an element-wise second-momentum estimator to achieve both stable and efficient training. Experiments on GPT-2 and Qwen2.5 show clear improvements in both training speed and final performance.

### Strengths
1. It achieves better performance than Adam and Muon on GPT-2 and Qwen2.5, provides efficiency gains while maintaining convergence quality.
2.It demonstrates stable training behavior across different model scales.

### Weaknesses
1. The scalability of AdaMuon to even larger foundation models (e.g., 70B+) remains untested.
2. The experiments are insufficient, with missing hyperparameter analysis.
3. The effectiveness of accumulating the second-momentum term is not validated by comparing it with other options.

### Questions
1. How does AdaMuon perform on extremely large-scale models or longer training schedules?
2. Can AdaMuon adapt well to sparse or modular architectures such as mixture-of-experts?
3. How robust is the method to hyperparameter choices, and does it require specific tuning strategies to achieve the reported gains?

### Soundness
3

### Presentation
3

### Contribution
3
