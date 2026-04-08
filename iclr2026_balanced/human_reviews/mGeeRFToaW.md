## Human Reviewer 1

### Summary
The paper proposes Quantized Zeroth-Order Optimization (QZO) for fine-tuning quantized neural networks—primarily LLMs—without backpropagation. The key idea is to apply SPSA-style zeroth-order perturbations to the continuous quantization scales while keeping discrete quantized weights fixed, thereby eliminating gradients and optimizer states and drastically reducing memory. A simple Directional Derivative Clipping (DDC) further stabilizes training by reducing gradient-estimate variance.

### Strengths
1. Q-SPSA operates on Δ, keeping integer weights fixed; avoids backprop and optimizer states.

2. DDC has theoretical backing and empirical impact (NaNs avoided; improved stability).

3. ~18× memory saving; single-GPU (24 GB) fine-tuning for large models; reduced trainable params/FLOPs vs MeZO.

4. Works across 4-bit (GPTQ) and 2-bit (AQLM) quantization; preliminary coverage of diffusion.

Overall, I think this is a good paper.

### Weaknesses
1. The comparison set omits strong practical baselines such as LoRA/QLoRA/AdaLoRA on the same datasets/models/bit-widths. Since QZO and PEFT are orthogonal (QZO tunes scales; PEFT adds low-rank adapters), including them would contextualize accuracy-vs-memory/latency trade-offs more fairly than only MeZO/SGD. (Tables report MeZO and full FT but not PEFT.)

2. The method’s success is acknowledged to depend on PTQ quality, but there’s no systematic study across quantizers (e.g., AWQ/SmoothQuant vs GPTQ/AQLM) at fixed bit-widths to isolate the effect of QZO when quantization error changes.

3. Supervised experiments use small subsampled splits (1k train per task), which can inflate variance and may not reflect real fine-tuning regimes; larger or diverse instruction-tuning sets would strengthen claims.

### Questions
1. Can QZO be combined with LoRA/QLoRA to improve accuracy while keeping memory low?

2. How sensitive is QZO to ε and learning rate across bit-widths (2/3/4-bit)? Any principled schedule for C beyond static thresholds?

3. Two forward passes per step suggest similar step-costs to MeZO; what are end-to-end time-to-accuracy comparisons under matched hardware?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 2

### Summary
The paper aims to reduce GPU memory usage for LLMs by employing zeroth-order optimization on quantized neural networks, a method referred to as QZO. The key idea behind QZO is to approximate gradients by perturbing the continuous parameters involved in the quantization process while keeping the model weights quantized. The authors demonstrate that QZO achieves substantial reductions in GPU memory consumption while maintaining strong performance compared to prior methods and baselines that require significantly more memory.

### Strengths
- This is a particularly relevant problem, as researchers without access to high-end GPUs often face significant challenges in conducting LLM research due to the models’ large memory requirements.
- The paper is well written, the proposed solution is elegant, and the experiments are carefully designed to evaluate different components of QZO, such as clipping.

### Weaknesses
- Using clipping for variance reduction may make the proposed method sensitive to hyperparameter choices. As shown in Figure 3, the clipping parameter has a substantial impact on accuracy, and it is unclear how this parameter should be selected beyond trial and error.

### Questions
- In RL, gradient estimates often have high variance since they are computed over trajectories, for instance under the current policy in the on-policy setting. A common approach to reduce variance, which tends to be less brittle than the clipping used in PPO, is to introduce a KL-based regularization term. Could the objective be modified to incorporate such a regularization mechanism instead of relying on clipping?
- While QZO achieves substantial memory reduction, it would be helpful to see plots showing the full trajectory of the loss and test accuracy. How quickly do these approaches converge compared to the baselines?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper introduces Quantized Zeroth-Order Optimization (QZO), a straightforward yet powerful method that estimates gradients by perturbing the continuous quantization scale and enhances training stability through a directional derivative clipping technique.

### Strengths
1. This work proposes Quantized Zeroth-order Optimization (QZO), which significantly reduces training memory usage.
2. The optimization of quantized weights requires no de-quantization or re-quantization.
3. The memory optimization is effective — for example, fine-tuning Llama-2-13B on a single 24GB GPU.

### Weaknesses
1. In the Introduction, the paper lacks a clear distinction between full-parameter fine-tuning and parameter-efficient fine-tuning. The current discussion implicitly assumes full-parameter fine-tuning, which is insufficient.
2. The experiments are also inadequate. Comparisons with parameter-efficient fine-tuning methods (e.g., LoRA, QLoRA) are missing, both in terms of memory consumption and fine-tuning performance. As a result, the paper does not demonstrate the trade-off between performance and efficiency, nor does it convincingly show the practical utility of the proposed method.
3. As shown in Table 1, the proposed method suffers from a notable performance drop during training.

### Questions
0. Please first review the items listed under Weakness.
1. Is there a comparison with the Adam optimizer? The gap between the zero-shot optimizer and Adam remains unclear.
3. In addition to the final evaluation results, providing training loss curves would help to more intuitively compare training convergence and stability.

### Soundness
3

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 4

### Summary
The paper studies memory-efficient fine-tuning of LLMs using quantization and zeroth-order optimization. As the size of LLMs grows quickly, the memory cost of fine-tuning becomes expensive. Zeroth-order optimization has shown significant improvement in saving memory for LLMs fine-tuning. The paper further studies how to even reduce it by quantization and proposes quantized zeroth-order optimization (QZO). Theoretical analysis is provided to show that the proposed method reduces variance in gradient estimation. Empirical experiments on fine-tuning LLMs on downstream tasks show great savings in fine-tuning memory, while preserving similar performance.

### Strengths
The paper is clearly written and has nice structure. The main claims are verified with either theory or experiments. The studied topic is well-motivated, and the problem setting is interesting and relevant to the community. By designing memory-efficent methods for LLMs fine-tuning, more and more people that do not necessarily have large computation resources can benefit from the rapid development and advancements of LLMs.

### Weaknesses
I believe there is a large body of related works that are not properly discussed, especially on fine-tuning with zeroth-order methods and quantization. 

1. The study of quantization for zeroth-order fine-tuning of LLMs is not new; see [1,2,3]. The paper should properly discuss what are the major differences and contributions compared to these related works. It is hard to evaluate the novelty and contributions of this work as such discussions are missing. Indeed, the settings and algorithms in this work seem to be very similar to these previous papers. Therefore, I am confused and thus require additional explanations from the authors.

2. There are also other ways to further save memory compared to MeZO, such as sparse MeZO [4,5]. This could also serve as a valid baseline to compare memory consumption and model performance.

3. This is less relevant. The clipped method used in the algorithm is also popular and introduced in the context of private and zeroth-order fine-tuning; see [6,7].

The experiments are a bit limited.

1. The paper considers OPT and Llama, with model size 7B, 8B, and 13B. It is not clear whether the method also works for other model family, and whether the algorithm scales to even larger models, as fine-tuning 13B model does not really reach the memory bottleneck.

2. Comparisons with LoRA and QLoRA are missing. LoRA and QLoRA with gradient checkpoint and gradient accumulation should also significantly reduce the memory while keeping similar performance. I am not sure how quantized zeroth-order optimization compares to that.

3. The considered downstreams tasks are a bit limited. Only 5 different tasks are used.
 
[1] ZOQO: Zero-Order Quantized Optimization. ICASSP 2025. (arXiv:2501.06736)

[2] QuZO: Quantized Zeroth-Order Fine-Tuning for Large Language Models. 2025. (arXiv:2502.12346)

[3] Stepping Forward on the Last Mile. NeurIPS 2024. (arXiv:2411.04036)

[4] Sparse MeZO: Less Parameters for Better Performance in Zeroth-Order LLM Fine-Tuning. 2024. (arXiv: 2402.15751)

[5] Zeroth-Order Fine-Tuning of LLMs with Extreme Sparsity. 2024. (arXiv:2406.02913)

[6] DPZero: Private Fine-Tuning of Language Models without Backpropagation. ICML 2024. (arXiv:2310.09639)

[7] Private Fine-tuning of Large Language Models with Zeroth-order Optimization. TMLR 2024. (arXiv:2401.04343)

### Questions
Questions are already mentioned in weaknesses.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
2

### Confidence
4

---

## Human Reviewer 5

### Summary
The large model size of LLM poses a challenge to computational resources. This work aims to reduce the memory cost of all the weights, gradients and optimizer states through one algorithm. The authors propose an algorithm QZO. Specifically, it uses the quantized model to reduce the memory of weights, and proposes a novel technique Q-SPSA that only perturbs the continuous quantization scale to compute the zeroth-order gradient with low memory cost. In addition, QZO adopts the clipping on the obtained zeroth-order gradient to increase the stability of training. A theoretical analysis is provided to verify its effectiveness. 

In the experiments on several tasks, benefiting from reducing the memory cost of weights, gradients and optimizer states, QZO significantly reduces the memory usage compared to other baselines. Meanwhile, the gap of performances between it and full fine-tuning is not large, even higher than full fine-tuning in some cases. The ablation study shows the effects of gradient clipping.

The main contribution of this work is that it proposes an algorithm that could update the quantized weight with the zeroth-order gradient. This idea is verified in the experiments: the memory cost is around 1/3 of that of the SOTA algorithm.

### Strengths
Previous works either reduce the memory of weights by quantization, or reduce the memory in the optimization step by the zo gradient. This work designs the Q-SPSA method to apply the zo gradient on the quantized weights. The originality and novelty are good. The experimental results reveal that this is an effective method for memory-efficient LLM training, indicating that this is a significant work. The motivation of this work is clear, and the description for the algorithm is easy to follow.

### Weaknesses
This work does not provide any convergence guarantee for the proposed algorithm. The experiments focus on fine-tuning the model on downstream tasks and do not include any pretraining tasks. 

Typo: the parenthesis in equation (3) seems to be incorrect.

### Questions
The experimental results in Table 1 show that QZO always outperforms baselines in the generalization task, but is comparable to or even obviously worse than baselines in the classification tasks. Could you give some explanations for this, i.e. why QZO performs so well in the generalization task?

Table 2 shows that the trainable parameters of QZO are significantly less than baselines. The authors explain that “This is because QZO only fine-tunes the continuous quantization scale”. I think QZO just reduces the precision of the trainable parameters, further reducing the flops. But the number of trainable parameters should be unchanged, as each element $w$ has a corresponding $\Delta$ to compute its zo gradient. Could you respond to this issue?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3