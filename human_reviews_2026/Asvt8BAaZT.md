# When Can You Get Away with Low Memory Adam?

- Decision: Reject
- Scores: 2, 4, 4, 4, 2

## Abstract
Adam is the go-to optimizer for training modern machine learning models, but it requires additional memory to maintain the moving averages of the gradients and their squares. While various low-memory optimizers have been proposed that sometimes match the performance of Adam, their lack of reliability has left Adam as the default choice. In this work, we apply a simple layer-wise Signal-to-Noise Ratio (SNR) analysis to quantify when second-moment tensors can be effectively replaced by their means across different dimensions. Our SNR analysis reveals how architecture, training hyperparameters, and dataset properties impact compressibility along Adam's trajectory, naturally leading to \emph{SlimAdam}, a memory-efficient Adam variant. \emph{SlimAdam} compresses the second moments along dimensions with high SNR when feasible, and leaves when compression would be detrimental. Through experiments across a diverse set of architectures and training scenarios, we show that \emph{SlimAdam} matches Adam's performance and stability while saving up to 98% of total second moments.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a layer-wise Signal-to-Noise Ratio (SNR) analysis to determine when the second-moment tensors in optimization algorithms (e.g., Adam) can be compressed by replacing them with their dimensional means. Given that SNR is computed as $\text{mean} / \text{variance}$, it serves as a natural metric for this purpose: a high SNR indicates that a tensor can be effectively approximated by its mean without significant performance loss. This approach provides a practical, quantitative guide for compressing optimizer states and offers evidence that Adam may not always require full second-moment information.

### Strengths
This work proposes to apply SNR as a metric to guide the compression of second-moment tensors with their means in LLM training. It offers a threshold-based criteria to determine when and how such mean compression can be applied across different architectural components of LLMs (e.g., query, key, value, and MLP layers). Furthermore, this work empircially investigates several factors influencing compressibility, including learning rate, data distribution, and initialization.

### Weaknesses
1. The main motivation of this work is to establish a metric for guiding dimension-wise mean compression of second-moment tensors and provide SlimAdam. However, this goal appears to overlap with Adam-mini [1], which not only implemenst a compression method based on block-wise mean values but also provides insights based on Hessian structure to explain why the full second-moment may be unnecessary and to guide how to compress. The authors should more clearly delineate their contributions and explicitly contrast their approach with the insights and methods provided by Adam-mini.

2. SNR is a natural choice for quantifying the viability of mean compression, given its formula of $\text{mean} / \text{variance}$. The paper does not sufficiently justify why it is superior to other plausible metrics. For instance, measures based on the L2-norm or KL-divergence of the error introduced by compression, or just variance, could be more direct and computationally efficient. The author should demonstrate the unique advantages of SNR over other alternatives through theoretical analysis or empirical comparision. 

3. For mean compression, it is clear that higher SNR correlates with better compressibility. However, the method remains dependent on an empirically set threshold (e.g., $\alpha=1$) to make compression decisions. This dependency not only limits the generality of the method by introducing a potentially sensitive hyperparameter across different scenarios but also raises the question of whether other metrics (e.g., L2-norm, KL-divergence, or variance) could perform just as effectively with a similarly tuned threshold.

4. The utility of SNR seems limited to mean compression and may not extend to or guide other compression paradigms (e.g., low-rank factorization, quantization).

[1] Zhang, Yushun, et al. "Adam-mini: Use fewer learning rates to gain more." arXiv preprint arXiv:2406.16793 (2024).

### Questions
1. The choice of SNR is intuitive for mean replacement, but why is it superior to other direct measures of compression error, such as the L2-norm or KL-divergence between the original and compressed tensor? Could the authors provide either (a) an empirical ablation study comparing the compression guidance performance of SNR against these other metrics, or (b) a theoretical argument for why SNR is an optimal or more robust criterion?

2. If the performance of the method is similar when using a simple threshold on other metrics (e.g., compress if $\text{variance} < X$, compress if $\text{L2-norm of error introduced by compression}/\text{L2-norm of target tensor}$), does this suggest the core insight is about identifying low-variance parameters rather than the unique information provided by SNR? What is the specific advantage of the SNR ratio over just using the variance or standard deviation?

3. The threshold $\alpha=1$ is presented as a critical value for making compression decisions. How was this value determined? Is it robust across different model architectures, layers, and tasks? Could the authors show sensitivity analyses for this threshold to demonstrate its generality?

4. The presentation could be improved for better clarity and reproducibility. For example, using pseudocode rather than plain text would help readers better understand the algorithm's workflow.

### Soundness
3

### Presentation
2

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
This paper studies the signal-to-noise ratio of Adam’s second-moment tensors for every layer along the input-channel (column) and output-channel (row) directions, finding instances where entries exhibit low variance relative to their mean and can safely share statistics during training; using these SNR profiles, it introduces SlimAdam, which collapses second moments only on the high-SNR direction of each layer, cutting memory while preserving Adam-level convergence and accuracy.

### Strengths
The paper tackles a well-motivated problem—the large memory footprint of Adam’s second-moment matrices—and clearly pinpoints instances where collapsing each layer’s parameters to a single scalar shrinks an matrix to just or values, cutting memory use while retaining Adam-level accuracy and stability.

### Weaknesses
Since there are almost no changes from my last review of the paper, I'll keep the core of my argument. I’ll split my review into two parts — one on the empirical analysis and one on the proposed optimizer.

### Empirical analysis and design rationale

The paper’s primary findings are almost entirely empirical, and this lack of theory leaves several key decisions unclear—especially because the empirical signals themselves are not particularly strong. Axis sharing is constrained to whole fan-in or fan-out dimensions purely for implementation convenience, with no exploration of alternative groupings or proof that these axes are optimal (e.g., would results change if one considers randomly partitioning a layer’s parameters into two equal-size groups?). Similarly, the paper adopts (the interesting metric) SNR with a heuristic threshold as the only compression criterion, although simple variance (which answers “How much l2 loss do we pay if we collapse this vector to a scalar?”) would align more directly with the intuition the authors cite (“If entries along a dimension exhibit low variance relative to their mean, they can be effectively represented by a single value”).

Learning rate is the only hyperparameter the paper systematically analyzes. It is presented as the dominant knob that shifts SNR and thus determines which layers can be compressed, yet the text provides no a-priori reason why learning rate—rather than, say, Adam’s momentum coefficients or the batch size—should hold that position.


### Optimizer details
The optimizer requires selecting a compression axis for each layer or layer type. Choosing a compression axis means either relying on proxies or heuristics, or collecting fresh SNR statistics. My concern is that the latter defeats the purpose in some cases, and the former is not reliable.

Alternatively, we can train a small proxy model or reuse generic rules. Yet the authors themselves show that preferred compression axes shift with dataset, width, and vocabulary size. Even within the same dataset and width, layers of the same type show different preferences. Depth-averaging does not fully solve the problem for users operating at the tightest memory margins or in domains whose depth-specific SNR patterns have not been studied yet. Even the stronger patterns they find—for example, compressing along the embedding dimension versus the token dimension—may not yield an SNR above the cutoff needed to justify compression.

Full-size SNR collection defeats the purpose. To decide the sharing axis, you must first run the uncompressed model under standard Adam long enough to gather per-layer SNR statistics. During this warm-up, you still store the full second-moment tensors, so the memory spike SlimAdam tries to avoid is paid up front. For practitioners who want to fit a slightly larger model into fixed hardware, this spike means the maximum model size is still bounded by Adam’s footprint during the warm-up, undermining the value of a lighter optimizer.

### Questions
Please address my concerns above, especially around the empirical nature of the evidence, axis selection, and practicality at tight memory budgets. In addition, a high-level clarification would help: how should we interpret “compressibility” in this work beyond plots of SNR? In other words, is SNR a sufficient observable for when per-parameter adaptivity is redundant, and how does its dependence on learning rate versus other hyperparameters shape the generality of your claims?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes to compress the second moment tensor of Adam by replacing per coordinate value with the average across specific dimensions. The method defines the signal to noise ratio of the second moment during training and compresses when this ratio is large.

### Strengths
As the memory of the optimizer accounts for a significant fraction of the memory requirement for neural network training, compressing it is an important problem. The paper studies a simple method for this task and gives thorough evaluation on different tasks and different modules of the network.

### Weaknesses
The method seems to have a large overlap with existing literature. The paper mentions Adam-mini, which already has a large overlap in terms of both the algorithm and the intuition behind the approach. Similar techniques also appear in several other papers such as

Lean and Mean Adaptive Optimization via Subset-Norm and Subspace-Momentum with Convergence Guarantees. ICML 2025

APOLLO: SGD-like Memory, AdamW-level Performance. MLSys 2025

These papers additionally save memory for the momentum, resulting in less memory than the method proposed here. In light of these works in addition to Adam-mini, the contribution of the new paper seems limited.

A minor point: please cite the published versions of the references. For example, the Adam-mini paper is in ICLR 2025.

### Questions
The pre-conditioner changes over time as Adam changes V in every step of training. The SNR analysis is fixed up front and the state is compressed in exactly the same way throughout but the condition number of V could change over time. Do you see any change in the condition number of V over time, and if not, is this a property of the training data and are there cases where the condition number of V changes? 

In some work, it is mentioned that gradient descent operates on the "edge of stability", do you see any changes if SNR analysis is done at different step sizes?

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
The authors present SlimAdam, a memory-efficient  version of Adam optimizer which achieves up to 99% memory savings by compressing the large second-moment statistics used in Adam's adaptive learning rate computations.  Rather than storing full per-parameter second moments, SlimAdam takes mean of these moments along the fan-in or fan-out dimensions 'when' appropriate. The 'when' in context is guided by a Signal-to-Noise Ratio (SNR) metric. 

SNR measures the concentration of second moment values (square of mean/variance). Higher SNR indicates tighter clustering, which justifies compression. And the compression is applied only in the layers where SNR is high, and state granularity is retained where SNR is low. The authors also state that since different layers show compression viability across different dimensions (fan-in/fan-out), derivation of compression rules are required for each model. To determine compression rules for each layer, the authors propose training a small proxy model at a reduced learning rate. SNR statistics from the proxy reliably generalize across larger models of the same architecture and task, informing safe compression dimensions for the full target model. 

The authors conduct a compreshensive empirical analysis across a wide range of large models and training tasks, revealing nuanced differences in compressibility across various layer types. Their findings highlight that attention components (such as keys and queries), value and projection layers, MLP layers, and token embedding/vocabulary layers each exhibit distinct compression characteristics. This detailed analysis reveals important, insightful architectural patterns that govern how adaptive moment compressibility varies.

Overall, the paper makes a relevant contribution to efficient optimization for large-scale deep learning, addressing a critical bottleneck in resource consumption. It balances rigorous analysis with practical effectiveness, although clearer exposition, especially regarding the proxy model methodology, would improve accessibility. SlimAdam, hence, is a valuable addition for researchers without sacrificing Adam's effectiveness.

### Strengths
1) SlimAdam achieves 99% memory savings compared to the original Adam optimizer, while fully preserving Adam’s effectiveness. It can be seamlessly swapped in place of Adam without requiring any code modifications or additional overhead.

2) The paper presents clear and well-motivated research questions supported by extensive experiments across diverse model architectures and tasks, demonstrating robust generality.

3) The authors provide a detailed algorithmic description alongside publicly available code, ensuring reproducibility.

4) Ablation studies are thoughtfully designed and thoroughly explained, offering valuable insights into the contributions of individual components and hyperparameters.

### Weaknesses
1) The main method (SlimAdam algorithm) is explained only in the appendix, and critical implementation insights (proxy model construction, SNR statistics collection) are not clearly presented in the main text. This prevents immediate accessibility and understanding.

2) The concept and practicalities of the proxy model for collecting SNR statistics are not deeply explained. Details about how proxy model size affects SNR relevance and how well proxy-derived rules scale to actual large models could be clearer. Also, how much compute overhead is added for such proxy runs should also be mentioned.

3) The details of SNR statistics adoption over the training steps in the actual model could be explained as well.

Minor Weaknesses:
Appendix C.1 is not completely written.

### Questions
1) How does proxy model size affect the SNR statistics for different tasks and architectures?

2) The paper states that for proxy model ignores early SNR statistics, and averages SNR values for next few steps rather than all steps; is the same applied for full model as well- meaning, is compressibility not applied for the first few runs, and how is it adapted over training steps?

I am amenable to changing the score if the questions and weaknesses are addressed.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a metric called SNR (gradient version) to address the high memory usage of Adam’s second momentum. The paper compresses the second-momentum tensor along these dimensions to a single mean value (or small number of values). The authors claim this approach (SlimAdam) can save up to 99% of the second-moment memory while maintaining Adam’s property and stability.

### Strengths
•	The idea of quantifying the compressibility of Adam's second moments on a per-layer, per-dimension basis is interesting.

•	The authors conducted extensive experiments across various architectures (GPT, ViT, ResNet) and tasks.

### Weaknesses
1.	Lack of Theoretical Foundation: As an optimizer paper, it lacks a convergence proof and relies almost entirely on experimental observations (e.g., using a 10x lower LR).

2.	Missing Essential Data: The paper does not include 'loss vs. step' curves, a critical metric for evaluating optimizers.

3.	Methodological Ambiguity: There is no logical basis for the proxy model design or the SNR threshold of $\alpha=1$.

4.	Questionable SNR Justification: The underlying assumption of mapping the mean of $V_t$ to 'signal' and its variance to 'noise' is not justified. (Since the justification of SNR relates to original vs. noise) 

5.	Exaggerated Contribution & Poor Comparison: The 99% claim is misleading (it's 50% of the total), and comparisons to SOTA optimizers that compress both moments (e.g., SMMF: Square-Matricized Momentum Factorization for Memory-Efficient Optimization which reduces the first and second momentums so that the total compression ratio is by up to 96%) are missing.

### Questions
•	What is the theoretical justification for treating the mean as 'signal' and the variance as 'noise' in your SNR definition $SNR_K = \mathbb{E}[(\mathbb{E}_K[V_t])^2 / Var_K[V_t]]$ from an optimization perspective? Is there a theoretical basis to claim that a low-variance (high SNR) tensor is inherently 'compressible'?

•	What is the specific theoretical justification for choosing the SNR threshold $\alpha=1$? Can you guarantee this value is universally optimal across different tasks and model architectures?

•	Can you provide loss-vs-step curves for your key results (e.g., Figure 8) to demonstrate that SlimAdam achieves the same 'final' performance with the same convergence speed as Adam?

•	Compared to optimizers like SMMF (2025), which compress both first and second moments for a 96% total memory saving, what is the practical advantage of SlimAdam, which only compresses the second moment for a ~50% total saving?

### Soundness
2

### Presentation
2

### Contribution
1
