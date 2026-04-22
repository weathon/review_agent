# Robust Training of Neural Networks at Arbitrary Precision and Sparsity

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 2, 6

## Abstract
The discontinuous operations inherent in quantization and sparsification introduce a long-standing obstacle to backpropagation, particularly in ultra-low precision and sparse regimes. While the community has long viewed quantization as unfriendly to gradient descent due to its lack of smoothness, we pinpoint—for the first time—that the key issue is the absence of a proper gradient path that allows training to learn robustness to quantization noise. The standard Straight-Through Estimator (STE) exacerbates this with its well-understood mismatch: a quantization-aware forward pass but oblivious backward pass, leading to unmanaged error and instability. We solve this by explicitly modeling quantization as additive noise, making the full forward-backward path well-defined without heuristic gradient estimation. As one natural solution, we introduce a denoising dequantization transform derived from a principled ridge regression objective, creating an explicit, corrective gradient path that makes learning robust to the noise STE bypasses. We extend this to sparsification by treating it as a special form of quantization that zeros out small values. Our unified framework trains models at arbitrary precisions and sparsity levels with off-the-shelf recipes, enabling stable A1W1 and sub-1-bit networks where others falter. It yields state-of-the-art results, mapping efficiency frontiers for modern LLMs and providing a theoretically grounded path to hyper-efficient neural networks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper locates the root cause of QAT instability in the STE’s "quantization-oblivious" backward pass and replaces it with a non-STE framework. The core method is a denoising dequantization transform derived from a ridge-regression objective, which injects an explicit corrective gradient path for quantization error and extends naturally to sparsification. The authors also introduce a shortcut for efficient affine-quantized matmul. The framework supports training across a wide range of precisions, including fully binary (A1W1) and sparse sub-1-bit models, with standard recipes.

### Strengths
1. A ridge-regularized dequantization transform creates an error-aware gradient path, stabilizing training at ultra-low precision and treating sparsification as quantization within one framework.
2. The approach is compatible with standard autograd and includes a shortcut formula that makes affine-quantized matmul computationally viable in practice.
3. At low precisions, the method achieves a new storage–energy Pareto frontier (demonstrated on Gemma-1B), offering clear, actionable trade-off guidance for model design.

### Weaknesses
1. Efficiency claims rely on a hardware-agnostic energy proxy rather than end-to-end measurements (latency/throughput/energy) on real accelerators.

2. Using affine+SCQ for the proposed method versus symmetric implementations for BitNet/ParetoQ can over-attribute gains to the estimator; require matched schemes (linear vs. linear) to isolate estimator effects.

### Questions
1. Can you clarify whether, in the extreme A1W1 regime, memory bandwidth remains the dominant bottleneck or whether the additional “shortcut” terms shift execution to a compute-bound regime—and, if so, whether these computations can be overlapped (e.g., via pipelining or operator fusion) with the main integer GEMM/memory traffic to hide their latency?
2. Can you provide ablations using linear (symmetric) quantization only, where your method and all baselines (STE, BitNet, ParetoQ) use the same quantization scheme and SCQ block size, so the estimator’s contribution is isolated from affine/SCQ effects?
3. Can you report how your method scales beyond 1B and 4B (e.g., ≥7B/13B/30B) and what advantage—if any—it holds over BitNet and related A1W1 approaches?

### Soundness
3

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
Quantization aware training (QAT) is a widely used method to prepare models for quantization. To overcome the rounding operation during backward pass, QAT uses straight through estimator (STE) which bypasses the gradients in the backward pass outside of the rounding operation. This work, characterizes this feature as the backward pass being quantization oblivious. To overcome this, the authors present a workaround that exposes the backward pass to the quantization error using a combination of ridge regression and affine quantization. Experiments on a wide array of models, and bit regimes shows superior performance of the proposed method when compared to plain STE-based QAT.

### Strengths
* This paper studies an interesting problem that is commonly taken as a given -- of using STE in QAT. 
* Interesting approach to alleviate STE gradient bypassing that allows error-aware gradient.
* Sparsification as a form of quantization?
* Impressive results on a wide-range of experiments; shows a clear benefit over QAT with STE.

### Weaknesses
* **QAT and error minimization:** The fundamental hypothesis of this work is that during the backward pass, QAT is oblivious to the quantization error. While this is possible, it has been shown by now that STE creates a different type of dynamic that results in weight oscillations. And these oscillations have factors that are trying to compensate the errors due to QAT [1,2]. Given this, what do the authors make of these explanations? And how does it alter their hypothesis, or not? If not, what is their argument?

* **STE beyond 2 bits:** Even if one were to attribute STE for poor performance in extremely low-bit width, how is this not manifested to the same degree in higher bit regimes. What is the explanation? Is STE more problematic only in extremely low-bit regimes? 

* **Ternary quantization with STE:** There are several works that use STE for ternary quantization [3,4]. The critique that STE cannot be used for extremely low-bit regimes does not hold up. 

* **Presentation clarity:** The ideas, experiments, and results are quite compelling in this work. However, the presentation is unclear in many places. There are vague statements, unsubstantiated by evidence (discussions around biological neurons, intelligence), and presentation of results makes it difficult to parse them. Many of the interesting results are in the Appendix (Fig. 3, Table 1) whereas the main results in Fig. 1 and Fig. 2 are illegible, with no clear captions, legends, axes labels. This is unfortunate as it dilutes the impact of otherwise nice contribution. I would suggest improving these aspects.   

* L-67: Very vague statement with exaggerated claims. It is by now common knowledge that large, quantized models outperform smaller ones. And also the claim of biological intelligence is extremely misplaced. 

### Other comments

* Reference to Figure 1-a in L-37 is not useful as none of the concepts are fully introduced; consider dropping this reference or elaborating the caption so that it can independently explain the concepts in the figure.

* L-38: What are the heuristic-based modifications authors are pointing to? No references to back this up. 

* L-54: Strange sentence; perhaps missing a preposition somewhere. Did the authors mean "full potential of the theoretically..."

### References

1. Wenshøj, Jonathan, Bob Pepin, and Raghavendra Selvan. "Oscillations Make Neural Networks Robust to Quantization." arXiv preprint arXiv:2502.00490 (2025).
2. Xie, Weiying, et al. "Allowing Oscillation Quantization: Overcoming Solution Space Limitation in Low Bit-Width Quantization." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
3. Choi, Jungwook, et al. "PACT: Parameterized Clipping Activation for Quantized Neural Networks." (2018).
4. Wang, Jinheng, et al. "1-bit ai infra: Part 1.1, fast and lossless bitnet b1. 58 inference on cpus." arXiv preprint arXiv:2410.16144 (2024).

### Questions
See weaknesses above.

### Soundness
3

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes an improved method for STE, where the backward pass is surrogated by that of naive linear mapping. Specifically, the authors claim that STE ignore the impact of quantization error during the backward pass, but only take into account of it for the forward pass, which is claimed to be the main cause of instability involved in training quantized models. The authors propose an algorithm based on a denoising dequantization transform derived from ridge regression, which is shown to be related to normalization layer, partially explain the reason of stablization. The experssion for the proposed quantization method is simplified to enable an efficient implementation with negligiblely introduced extra computation, only of the order of rank-1 corrections. The proposed method could unify binarization/quantization with sparsification, and experiemnts shows some improvements.

### Strengths
- The proposed method of using ridge regression to derive the quantization backward is novel.
- The analysis and method could unify binarization/quantization with sparsification.

### Weaknesses
- The characters in the figures are too small and very difficult to read.
- The final results are too difficult to read from Figure 2 and there are no table for comparison to illustrate the claimed advantage compared to previous results.
- The authors claim that in STE, the forward pass is affected by quantization error while the backward pass is not (line 042 in the original paper). However, during the backward pass, the weight and activations involved are also quantized, so it is also different from full-precision model. This statement could be improved and should be corrected.
- For asymmetric data distribution, as discussion in Section 2.2, the usual quantization process is to directly map it to some given quantized range, say from the full-precision interval of [0,1], to values of {0, 1/4, 2/4, 3/4} for 2-bit quantization, which does not introduce bias issue. Another way is to first shift and scale it to make it symmetric and apply the symmetric quantization, and finally rescale and shift back. For weight quantization, unbiased quantization is important as it impacts the training dynamics and is crucial for convergence, but for activation quantization, which is usually the asymmetric part, the symmetry requirement is not necessary and would not impact the training dynamics. Regarding training dynamics and symmetric weights, the authors could check previous work, such as [1].
- In equation 3, the authors should explain what the values with bar above them are.
- In Section 3.3.2, for the first part, for STE, the input to the dequantization of STE also includes the quantization error. For the second part, it is not clear or proved why the local derivative is an explicit function of the error. For example, suppose in equation 3, the final function after simplification become a linear function of q. The derivative would then be independent on q or delta, i.e. the quantization error.
- In equation 5 and 6, the authors should explain what n represents.
- Writing problems: The sentence at line 366 in the original paper is not complete nor correct in grammar. The sentence “This low resources setting quickly highlights the fragility of standard models.” at line 368 in the original paper is redundant and should be deleted.


[1] Ben Poole, et al., Exponential expressivity in deep neural networks through transient chaos. NeurIPS 2016.

### Questions
- At line 100 of the original paper, it states that unmanaged error corrupts the learning signal, leading to training divergence. Could the author provide some further demonstrations to verify this statement, e.g., gradually introducing such quantization error in back-propagation, i.e., making it as a linear combination of the vanilla full-precision gradient (which include the quantization error) and the one from STE (which does not include it), to demonstrate that the error indeed causes the divergence issue?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper tries to address the drawbacks of using Straight through estimator in QAT by proposing a denoising dequantization transform which models quantization error in both fprop and backprop. The proposed transform acts a normalization layer stabilizing gradients.
The efficacy of the proposed technique is measured on low bitwidth and sparse networks, and the emperical results on nanoGPT, Gemma-1B/4B models show the pareto frontier for storage and energy and demonstrates the outperformance compared to literature work on STE, BitNet etc.

### Strengths
1. The paper shows stable training of A1W1 and sparse sub-1-bit models where prior literature methods diverge.
2. Demonstrates results on nanoGPT and Gemma-1B/4B, showing pareto efficiency on storage and energy frontiers.
3. The method is simple, and needs no special hyperparameter tuning.

### Weaknesses
1. Lacks ablation on the sensitivity of λ (regularization) across architectures.
2. Do you more experiments with longer running fine tuning or pre-training to ensure that convergence and generalization does not suffer?

### Questions
1.How sensitive is training stability and convergence to λ across architectures and datasets?
2. Does the model remain robust during methods such as fine-tuning (dense or using LoRA)?

### Soundness
3

### Presentation
3

### Contribution
3
