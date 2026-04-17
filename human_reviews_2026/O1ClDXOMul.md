# FOSL: A Foldable Sparse-and-Low-Rank Method for Efficient LLM Pre-training

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
We propose FOSL, a foldable, sparse-and-low-rank reparameterization for efficient pre-training that decouples compute from width. Each linear/FFN/attention projection is rewritten as two cooperating paths: a low-rank path that injects expressive features via a compact adapter, and a folded sparse path that computes only a subset of output channels and synthesizes the remainder as virtual channels by reusing computed ones. A lightweight, variance-preserving rescaling keeps activations stable when channels are reused multiple times. This design delivers the benefits of a narrower network internally while maintaining full-width activations at the interface, avoiding the representation bottlenecks of hard pruning and complementing low-rank-only approaches. We evaluate FOSL for LLM pre-training across model scales from 60M to 7B parameters. Our experiments demonstrate that FOSL matches or surpasses full-rank models, while substantially reducing memory and computation costs.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper investigates efficient LLM pretraining by parameterizing linear layers with low-rank and sparse matrices. In particular, to overcome the limitations of unstructured or fixed channel-wise sparsity, it proposes a foldable method that randomly samples channels as base channels and reuses them to recover the original activation width. To ensure stability, it also introduces a variance-preserving rescaling scheme when channels are reused multiple times.

Experiments are conducted on 60M-, 130M-, 350M-, and 1.3B-parameter models with up to 10B training tokens, showing effectiveness in perplexity and memory usage. The paper also validates the method by fine-tuning RoBERTa models on the GLUE benchmarks.

### Strengths
* The method is very interesting. It uses low-rank and sparse matrices for efficient linear layers. Specifically, it identifies limitations of prior work in utilizing sparse matrices—either being hardware-unfriendly or relying on fixed channel-wise sparsity. Therefore, the paper proposes a foldable approach that randomly samples channels and reuses them for the remaining channels, matching the original activation width.

* The paper carefully considers its method design, including variance correction.

* The paper compares against different methods and shows its utility in both pretraining and fine-tuning.

### Weaknesses
* Given that the paper focuses on efficient pretraining (as stated in the title), in addition to FLOPs and memory, could the authors also report latency statistics? Pretraining latency is also a very important efficiency metric.

* I have doubts about the benchmarking. While the paper claims that its experimental setup follows prior work such as SLTrain and LOST, I checked those papers and found that the full-rank baselines differ (e.g., 16.52 vs. 15.56 for the 1.3B model in prior work). Could the authors confirm that all results in Table 2 (full-rank baselines, the proposed method, and prior methods) were trained under the same settings? Meanwhile, when comparing the proposed method with the full-rank baselines in Table 2, can the authors explain why the proposed method achieves even better perplexity (e.g., ~2 ppl lower with the 1.3B model and ~0.5 ppl lower with the 350M model)?

* While the paper targets 7B results as the next step, two important experiments are still missing. First, report downstream task performance using the 1.3B pretrained models; this would help us understand accuracy differences rather than only perplexity. Second, examine the overtraining regime. The 1.3B model trained on 13B tokens even does not reach the token budget suggested by scaling-law studies, and it is unclear whether the gap between the full-rank and the proposed method would decrease or increase as training approaches convergence. It is also unclear whether the proposed method encounters issues in an overtraining regime. Given cost constraints, could the authors present results for a 130M model trained on 100B tokens?

* There are some writing issues. For example, the paper claims, “We evaluate FOSL for LLM pretraining across model scales from 60M to 7B parameters,” but no 7B results are reported. For Eq. (6) and the three mixing methods, which γ are you referring to in Eq. (6)?

### Questions
Please see the weaknesses above. I find the method interesting; however, the experiments and writing prevent me from giving a higher score.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors of this manuscript propose FOSL, a foldable, sparse-and-low-rank reparameterization for efficient LLM pre-training. The method decomposes each projection into two paths: a standard low-rank adapter path and a novel "folded sparse path." This folded path computes only a subset of the output channels and then synthesizes the remaining "virtual channels" by reusing the computed ones. To maintain stability, a lightweight, variance-preserving rescaling is applied. The goal is to match or exceed full-rank model performance while significantly reducing computation and memory costs.

### Strengths
1- The paper is well-written, and the core concept of a "foldable" path that reuses channels is an intuitive and interesting approach to efficiency.

2- The reported perplexity results for the 1B model appear significant, outperforming the full-rank baseline and other competing methods shown in Table 2.

3- The method thoughtfully includes variance correction to stabilize the activation statistics, addressing a potential issue with channel reuse.

### Weaknesses
1- The related work and comparisons are incomplete. Previous work on sparse and low-rank pretraining, such as FST [1] and SLoPe [2], are missing from the discussion and experimental comparison. Without this, it is difficult to situate FOSL's contribution relative to the state-of-the-art.

2- The abstract claims the method scales to 7B models, but the experiments in the main paper only present results up to 1B parameters. This appears to be a typo or unsubstantiated claim, as the 7B results are not shown.

3- The perplexity results in Table 2 are counterintuitive. Most of the efficient training methods, including FOSL, achieve a lower perplexity than the full-rank 1B model (14.69 for FOSL vs. 16.52 for Full-Rank). This strongly suggests that the hyperparameters for the dense baseline were not properly tuned, which calls the validity of the reported improvements into question.

4- The training dynamics are vague. The paper does not provide any training loss graphs, which would offer crucial insight into the stability and convergence behavior of FOSL compared to the baselines.

5- The comparisons in Table 2 are not fair. FOSL (730M parameters) is directly compared against LORO, CoLA, and LOST (all 609M parameters). This parameter disparity makes the perplexity comparison less meaningful. A more rigorous comparison would adjust the ranks to ensure all models have a similar parameter count.

6- The paper makes claims about other methods without providing evidence. For example, the authors claim that LOST has "significant overhead due to computing SVDs" but do not show any profiling data to prove that SVD computation is actually a training bottleneck.

7- The paper lacks empirical efficiency measurements. All reported efficiency gains are based on parameter counts and estimated memory. There are no measurements of actual wall-clock speedup (e.g., tokens/sec) or measured memory reduction during training.

8- It is unclear if FOSL requires a warm start from a full-rank checkpoint. If it does, comparing only final perplexity is misleading, as the majority of loss reduction happens in the initial training phase. The authors should report the loss/perplexity improvement after switching to the efficient training method (PPL(dense) - PPL(final)) and compare that delta.

---

[1] Hu et al., FST: Fast Sparse Training of Transformers, ICML 2024

[2] Mozaffari et al., SLoPe: Double-Pruned Sparse Plus Lazy Low-rank Adapter Pretraining of LLMs, ICLR 2025

### Questions
1- How does FOSL compare in terms of perplexity, throughput, and measured memory usage against recent sparse and low-rank pretraining methods like FST [1] and SLoPe [2]?

2- Can the authors justify the high perplexity of the 1B full-rank baseline? Were its hyperparameters adequately tuned, and could training loss curves be provided for all 1B models to compare their training dynamics?

3- Is the claim of scaling to 7B models in the abstract a typo, or can the authors provide those results?

4- Can the authors provide a fairer comparison in Table 2 by adjusting the model ranks to ensure FOSL has a comparable parameter count to LORO, CoLA, and LOST?

5- Can the authors provide profiling data to substantiate the claim that the SVD computation in LOST is a significant training bottleneck?

6- What are the measured (not estimated) wall-clock speedups and memory reductions of FOSL during training compared to the full-rank baseline?

7- Does FOSL require a warm start with full-rank training? If so, what is the perplexity improvement from the point of switching (PPL at switching - final PPL), and how does this compare to a dense model trained for the same duration?

---

[1] Hu et al., FST: Fast Sparse Training of Transformers, ICML 2024

[2] Mozaffari et al., SLoPe: Double-Pruned Sparse Plus Lazy Low-rank Adapter Pretraining of LLMs, ICLR 2025

### Soundness
3

### Presentation
2

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
The paper proposes FOSL, a training-time reparameterization that combines a small low-rank adapter with a folded sparse path that computes a base subset of channels and duplicates them, using a diagonal variance correction to stabilize reused channels. The goal is to preserve full-width interfaces while paying the compute of a narrower layer. The authors experiment with pretraining LLaMA 60M-1B on C4, and finetuning Roberta-Base on GLUE.

### Strengths
* Simple idea, easy to implement: The folded path is just a narrow GEMM plus an index-based duplication with a diagonal rescale which is easy to incorporate to standard training.
* Clear efficiency intent, important goal.
* The authors provide a nice overview of SL-Train and other methods that they build on. Sections 2-3 describing these works and the proposed method are easy to follow.

### Weaknesses
* Gaps in the related work. In L60 the authors mention “More related work is covered in Appendix C”, yet Appendix C largely repeats the same set of papers with minimal rephrasing.  
The main text organizes prior work into three directions (1) low-rank/LoRA-style, (2) gradient-projection/optimizer-memory, and (3) sparse+low-rank (plus folding).  
However, a major missing category is activation compression such as CompAct[1], VeLora[2],  HOSVD [3] and many others, which directly reduce activation memory while preserving the integrity of the forward pass. These methods pursue the same objectives emphasized by the paper (L61-62) yet are neither discussed nor shown in Figure 1.

* There are inconsistencies between the hyper parameters used for FOSL and other methods when comparing in the experiments, making it unclear if the improvement is due to these inconsistencies, or the proposed method working. Especially as the method is very similar to previous work with the addition of the folding technique.
* Some results are overstated, and sometimes falsely so. For instance in Table 5 : 
L420 “Even small ranks (e.g.,  r=32) remain competitive due to the complementary capacity supplied by the folded sparse path.” with respect to what? What is considered competitive? What is the degradation without the sparse supplement? Additionally emphasizing that the performance improves when r=512 or 256 is simply misleading. With W=AB, if you use these high ranks, the number of parameters is larger than the original W in the baseline - hence the improvement in perplexity.
* The abstract and Appendix text mention an evaluation for a 7B model, even mentioning “5 days” training time, yet no 7B results are included in the experiments.

References:
* [1] CompAct: Compressed Activations for Memory‑Efficient LLM Training – Shamshoum et al., NAACL 2025, arXiv:2410.15352v1. 

* [2] VeLORA: Memory Efficient Training using Rank‑1 Sub‑space Activations – Miles et al., NeurIPS 2024 Poster, algorithm for compressing activations into 1-D subspace. 
* [3] Nguyen, L. T., Quélennec, A., Tartaglione, E., Tardieu, S., & Nguyen, V. T. (2024). Activation Map Compression through Tensor Decomposition for Deep Learning. arXiv preprint arXiv:2411.06346.

### Questions
1. Table (1) seems to have a few incorrect claims. For instance in LoRA\ReLoRA, although the trainable parameters are O(r(m+d)), the Fwd FLOPs are certainly not O(r(m+d)), as the multiplication with the original matrix W_0 is still executed.  Additionally, the fact that the Fwd FLOPs and Bwd FLOPs are independent of the sequence length and batch size is wholly inaccurate as these dimensions are certainly not negligible usually with respect to the m and d. Can the authors clarify and provide a table with correct accurate comparisons?
2. Why compare FOSL with 0.9 sparsity rate with SLTrain and LOST of 0.99 rate? How can we isolate the benefit to the folding rather than simple less sparsity \ more parameters? Looking at figure 3, the performance of FOSL get significantly worse from 0.9 to 0.99. With a perplexity around 33 for LLaMA-60M, and 27 for LLaMA-130M. These numbers are significantly worse than what is presented in the table.
3. Are the results for FOSL averaged over a few runs?. Can the authors provide the std for the results? 
4. L474 “Variance correction assumes weak cross channel correlations. Heavy-tailed or strongly dependent activations may reduce exactness.” Can the authors please elaborate on why they think this is the case? My intuition is, there must be some dependency in the activations - otherwise there wouldn’t be work compressing\pruning them. A plot showing this would help explain why the variance correction would work.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a new low-rank pre-training method to reduce computational FLOPs and reduce training time/cost. The proposed method, FOSL, is inspired by LoRA and uses a low-rank adapter to learn expressive features along with a folded sparse path that only computes a fraction of the channel. The rest of the channel is predicted using that computed channel to maintain the layer while reducing training FLOPS. Since reusing channels to predict other channels affects the layer statistics/variance, the authors propose variance correction to improve training dynamics. The method is evaluated on small-scale LLMs.

### Strengths
1. The core idea is intuitive. The method leverages previous insights that many channels in LLM layers are redundant and uses that to only compute a subset of channels using FOSL. 

2. The method consistently outperforms baseline methods with different model sizes and datasets (Table 2), showing the efficacy of the method.

3. The paper is well-written and easy to follow. The background and related work are well explained, making it easy for readers to understand the proposed method. 

4. Ablations are provided to explain how the hyper-parameters (mixing coefficient) were selected.

### Weaknesses
1. Results provided are limited. A more extensive evaluation on the zero-shot downstream dataset should be conducted to show that the trained model can be used for downstream tasks. 

2. The experiment in section 4.2 is not convincing. The results are mixed, and FOSL has more parameters than LoRA (with the same rank), making direct comparison difficult. 

3. In Table 3, the ablation on the mixing coefficient shows that learning the mixing coefficient gives similar performance as using a static coefficient. Do you have any insights?

### Questions
1. For predicting remaining channels, have you compared using/learning linear combinations of the computed channels? How does that compare with learning the duplication matrix?

### Soundness
3

### Presentation
3

### Contribution
3
