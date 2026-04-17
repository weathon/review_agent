# Fine-Tuning Quantized Neural Networks With Zeroth-Order Optimization

Sifeng Shang1 Jiayi Zhou1 Chenyu Lin1 Minxian Li2 **Kaiyang Zhou**1, 1Hong Kong Baptist University 2Nanjing University of Science and Technology https://github.com/maifoundations/QZO

## Abstract

As the size of large language models grows exponentially, GPU memory has become a bottleneck for adapting these models to downstream tasks. In this paper, we aim to push the limits of memory-efficient training by minimizing memory usage on model weights, gradients, and optimizer states, within a unified framework. Our idea is to eliminate both gradients and optimizer states using zeroth-order optimization, which approximates gradients by perturbing weights during forward passes to identify gradient directions. To minimize memory usage on weights, we employ model quantization, e.g., converting from bfloat16 to int4. However, directly applying zeroth-order optimization to quantized weights is infeasible due to the precision gap between discrete weights and continuous gradients, which would otherwise require de-quantization and re-quantization. To overcome this challenge, we propose Quantized Zeroth-order Optimization (QZO), a simple yet effective approach that perturbs the continuous quantization scale for gradient estimation and uses a directional derivative clipping method to stabilize training. QZO is orthogonal to both scalar-based and codebook-based post-training quantization methods. Compared to full-parameter fine-tuning in 16 bits, QZO can reduce the total memory cost by more than 18× for 4-bit LLMs, and enables fine-tuning Llama-2-13B within a single 24GB GPU.

## 1 Introduction

Pre-trained large language models (LLMs) (Zhang et al., 2022; Touvron et al., 2023a;b; Grattafiori et al., 2024) have demonstrated great potential in numerous downstream applications, ranging from sentiment classification and text summarization, to more challenging open-ended question answering and creative writing. However, with the model size growing at an exponential rate, adapting LLMs to downstream tasks presents significant challenges to computational resources. For instance, finetuning a Llama-7B model stored in bfloat16 typically requires 56GB GPU memory: 14GB for model weights, 14GB for gradients, and another 28GB for optimizer states when adaptive gradient-based optimization methods are used (e.g., the first and second moments in AdamW (Loshchilov & Hutter, 2017), which cost twice the size of gradients). Such an enormous memory cost makes it infeasible for researchers and practitioners with limited computational resources to fine-tune LLMs. In general, there are four key components that determine memory usage: (1) model weights, (2) gradients (typically the same size as weights), (3) optimizer states (often twice the size as gradients), and (4) activations cached for gradient computation. Since activations are mostly affected by the size of mini-batch, existing memory-efficient training methods mainly target the first three components (Zhao et al., 2024; Malladi et al., 2023). In this work, we aim to push the limits of memory-efficient training by minimizing memory usage on model weights, gradients, and optimizer states, within a *unified* framework. Our main idea is to eliminate gradients and optimizer states using zeroth-order optimization (Spall, 1992), which gets rid of backpropagation by approximating gradients solely through forward passes
(i.e., perturbing model weights to identify gradient directions). When it comes to model weights, the Corresponding author 1

![1_image_0.png](1_image_0.png)

optimal approach is to quantize the weights, e.g., converting from bfloat16 to int4 can significantly cut the memory cost by 4×. However, directly applying zeroth-order optimization to quantized weights is non-trivial because (1) quantized weights cannot be perturbed in the continuous space, and (2) the gradients estimated by a zeroth-order optimizer are continuous and therefore cannot be used to update discrete quantized weights (which would otherwise require de-quantization and re-quantization).1 To overcome the aforementioned challenges, we propose a novel approach called Quantized Zerothorder Optimization (QZO), which enables quantized neural networks to be fine-tuned with zerothorder optimization, hence achieving maximum reduction in memory consumption—compared to full-parameter fine-tuning in 16 bits, QZO significantly reduces the total memory cost by 18× for 4-bit LLMs (see Figure 1). Specifically, QZO approximates the gradients of quantized weights by perturbing the continuous quantization scale parameter(s) rather than the discrete weights, which are kept fixed throughout training. To further stabilize training, we propose a gradient clipping method and provide a theoretical proof to justify that the clipping method essentially reduces the variance of the gradient estimate. We evaluate QZO on different families of LLMs including OPT (Zhang et al., 2022) and Llama (Touvron et al., 2023b; Grattafiori et al., 2024), as well as using a diverse set of quantization methods. The experiments are conducted on five popular NLP benchmarks including both classification and generation tasks. Using 4-bit LLMs, QZO significantly outperforms both quantized and un-quantized zero-shot models while performing on par with MeZO (Malladi et al., 2023), which applies zerothorder optimization to un-quantized models. In the extreme quantization case where the model is quantized to 2-bit, QZO still beats the zero-shot baseline by a large margin, demonstrating the effectiveness of QZO in fine-tuning quantized models. We also provide both theoretical evidence and ablation experiments to demonstrate the effectiveness of directional derivative clipping in stabilizing the training, which functions through reducing the variance of the gradient estimates.

## 2 Related Work

Memory-Efficient Training Fine-tuning LLMs often requires a significant amount of GPU memory, making it challenging for model adaptation on resource-constrained hardware. In general, current memory-efficient training methods mainly focus on reducing GPU memory usage for the following components: (1) learnable model weights, (2) gradients, (3) optimizer states storing additional gradient information, and (4) activations cached for gradient computation. To save memory cost for optimizer states, GaLore (Zhao et al., 2024) projects the first and second moments of gradients in AdamW (Loshchilov & Hutter, 2017) onto a low-rank subspace. MeZO (Malladi et al., 2023) eliminates gradients and optimizer states by using a zeroth-order optimizer (Spall, 1992), which estimates gradients using only forward passes and therefore keeps the memory cost the same as inference. CoLM (Nguyen et al., 2025) uses small mini-batches whose gradients match those of large mini-batches, leading to huge memory reduction in activations. Our approach further pushes the limits of memory-efficient training by fine-tuning quantized LLMs with zeroth-order optimization, which significantly cuts memory usage across all components requiring GPU memory.

LLM Quantization Post-training quantization (PTQ) is a popular paradigm for compressing LLMs. Most PTQ methods (Dettmers et al., 2022; Frantar et al., 2023; Lin et al., 2024; Xiao et al.,
2023; Ashkboos et al., 2024) reduce the bit width for each model parameter by representing the numerical range with low-precision integers while using full precision for quantization parameters. These methods can achieve up to 4-bit quantization, resulting in up to 4× reduction in memory usage compared to the widely-used BF16 representation. Different from the popular scalar-based quantization paradigm, recent research (Tseng et al., 2024; Egiazarian et al., 2024; Liu et al., 2024) has explored using codebooks for storing full-precision numbers, which are indexed with integers to represent the original model weights. These codebook-based methods can achieve extreme quantization in 2 or 3 bits without observing significant performance drops. Typically, quantized LLMs are not suitable for fine-tuning because continuous gradients cannot be directly applied to updating discrete quantized weights (which would require de-quantization and re-quantization). Our approach seamlessly combines memory-efficient training with quantization to enable fine-tuning on quantized LLMs, achieving maximal reduction on GPU memory usage. More importantly, our approach is orthogonal to most PTQ methods, including both 4-bit and 2-bit quantization methods. Zeroth-order Fine-tuning for Quantized Models Inspired by a foundational approach, ZO- signSGD (Liu et al., 2019), several prior works (Feng et al., 2024; Zhou et al., 2025; Bar & Giryes, 2025) expand on this study to enable the fine-tuning of quantized models, using a shared paradigm that involves quantizing perturbation noises and directly applying sign-based SGD on discrete, quantized weights. Although sharing a similar spirit in minimizing the memory footprint, namely combining zeroth-order optimization with quantization, the proposed QZO approach is inherently more efficient and flexible, as it does not require quantization of perturbation noises or re-quantization of model weights at each optimization iteration. Furthermore, it can be applied to existing scalar-based or codebook-based PTQ methods, such as GPTQ (Frantar et al., 2023) and AQLM (Egiazarian et al., 2024), in a plug-and-play manner.

## 3 Methodology 3.1 Background: Zeroth-Order Optimization

Zeroth-order optimization (ZO) methods are often used in cases where gradients and higher-order derivatives of the objective cannot be directly computed or are unreliable (Conn et al., 2009). The pioneering work, Simultaneous Perturbation Stochastic Approximation (SPSA) (Spall, 1992), is defined as follows, Definition 3.1 (Simultaneous Perturbation Stochastic Approximation, SPSA (Spall, 1992)). Given a model parameterized by θ ∈ R
d and a loss function L, SPSA estimates the gradients of θ on a mini-batch B *using the following formula:*

$$\tilde{\nabla}_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta};\mathcal{B})=\frac{\mathcal{L}(\mathbf{\theta}+\epsilon\mathbf{z};\mathcal{B})-\mathcal{L}(\mathbf{\theta}-\epsilon\mathbf{z};\mathcal{B})}{2\epsilon}\mathbf{z}\approx\mathbf{z}\mathbf{z}^{\top}\nabla_{\mathbf{\theta}}\mathcal{L}(\mathbf{\theta};\mathcal{B}),\tag{1}$$

where z ∈ R
dis a random vector sampled from N (0, Id), and ϵ *the perturbation scale.*
Built on top of SPSA, a recent work (Malladi et al., 2023) proposed memory-efficient zeroth-order optimization (MeZO) for LLMs. In particular, MeZO uses random seeds as a trick to eliminate the storage cost of z, and as a result, the memory footprint is kept the same level as inference. MeZO also replaces the regular SGD (Robbins & Monro, 1951) with zeroth-order stochastic gradient descent (ZO-SGD), which is defined below: Definition 3.2 (Zeroth-Order Stochastic Gradient Descent, ZO-SGD (Malladi et al., 2023)). *Given a* learning rate η, ZO-SGD updates the parameters θt at t-th step using gradients estimated by SPSA
as follows:
θt+1 = θt − η∇ˆθtL(θt; Bt) (2)

$$\theta_{t+1}=\theta_{t}-\eta\hat{\nabla}\theta_{t}\mathcal{L}(\theta_{t};\mathcal{B}_{t})$$

where Bt *denotes the input mini-batch at step* t.

$\mathbf{M}$
3.2 QZO: QUANTIZED ZEROTH-ORDER OPTIMIZATION
QZO minimizes the memory usage not only on gradients and optimizer states but also on model weights—this can save huge memory cost when using large models of more than 10B parameters, e.g., when using bfloat16, a 10B model's weights consume 20GB of memory, while using int4, the weights only take 5GB of memory. QZO consists of two core modules: Quantized Simultaneous Perturbation Stochastic Approximation (Q-SPSA), and directional derivative clipping. The former extends SPSA to quantized weights while the latter stabilizes training by reducing the variance of gradient estimation.

## 3.2.1 From Spsa To Q-Spsa

SPSA (Eq. 1) cannot be directly applied to quantized weights because (1) quantized weights are discrete and therefore cannot be perturbed in the continuous space, and (2) the continuous gradients cannot be used to update discrete weights, which would otherwise require de-quantization and re-quantization. To overcome these challenges, we propose Quantized Simultaneous Perturbation Stochastic Approximation (Q-SPSA), which only applies perturbation to the continuous quantization scale. We begin by introducing quantization and de-quantization, which are two essential steps in model quantization. Concretely, for each single element w in a weight set W, these two steps can be formulated as

$$(2)$$

$$\begin{array}{l}{{\overline{{{w}}}=\lfloor\frac{{w}}{{\Delta}}\rfloor,}}\\ {{w=\Delta\cdot\overline{{{w}}},}}\end{array}$$
$$({\mathfrak{I}})$$
$$(4)$$

where ∆ denotes an element-wise quantization scale, and w the quantized counterpart stored using lower bits. The weight set W is determined by the choice of quantization group, while the implementation of ∆ varies among different quantization methods. For example, when ∆ = absmax(W)
2 k−1−1
, Eqs. 3 and 4 refer to the standard scalar-based quantization in k-bit. Since the de-quantization process in Eq. 4 aligns with the normal forward propagation, we decompose the model parameters θ in Eq. 1 into ∆ ⊙ θ¯, and perturb the scaling component ∆ while keeping the discrete weights θ¯ fixed. Therefore, Q-SPSA can be formulated as Definition 3.3 (Quantized Simultaneous Perturbation Stochastic Approximation, Q-SPSA). *Given a* quantized model with integer parameters θ¯ ∈ R
d and quantization scales ∆, and a loss function L,
Q-SPSA estimates the gradients of ∆ over a mini-batch B *using the following formula:*

$$\begin{split}\tilde{\nabla}_{\boldsymbol{\Delta}}\mathcal{L}(\boldsymbol{\Delta}\odot\bar{\boldsymbol{\theta}};\mathcal{B})&=\frac{\mathcal{L}((\boldsymbol{\Delta}+\epsilon\boldsymbol{z})\odot\bar{\boldsymbol{\theta}};\mathcal{B})-\mathcal{L}((\boldsymbol{\Delta}-\epsilon\boldsymbol{z})\odot\bar{\boldsymbol{\theta}};\mathcal{B})}{2\epsilon}\boldsymbol{z}\\ &\approx\boldsymbol{z}\boldsymbol{z}^{\top}\nabla_{\boldsymbol{\Delta}}\mathcal{L}(\boldsymbol{\Delta}\odot\bar{\boldsymbol{\theta}};\mathcal{B}),\end{split}\tag{5}$$

where z ∈ R
dis a random vector sampled from N (0, Id), ϵ the perturbation scale, and ⊙ the Hadamard product. Similar to MeZO, all quantization scales within a linear layer are perturbed to save computation. In practice, one may choose to fine-tune the continuous quantization scale only, or combine Q-SPSA
with SPSA to jointly update the unquantized counterparts. It is worth noting that Q-SPSA can be applied to both scalar-based and codebook-based quantization methods: in the experiments we show that our approach can successfully fine-tune both 4-bit LLMs quantized by the scalar-based GPTQ (Frantar et al., 2023) and 2-bit LLMs quantized by the codebook-based AQLM (Egiazarian et al., 2024) (in this case both the channel-wise scales and un-quantized weights are updated).

## 3.2.2 Ddc: Directional Derivative Clipping

Gradient estimation via ZO is notorious for causing unstable training due to large gradient variance (Malladi et al., 2023). This was also observed when combining Q-SPSA with the vanilla ZO-SGD method in our preliminary experiments where training often collapsed. To mitigate this problem, we propose Directional Derivative Clipping (DDC) and apply this method before updating the model with ZO-SGD at each optimization step. Specifically, the gradient estimate in Eq. 5 can be viewed as a product of the random vector z and the estimated directional derivative of loss function along z w.r.t. ∆ (which is essentially a scalar). Let d denote the estimated directional derivative, Eq. 5 can be re-written as ∇ˆ ∆L(∆ ⊙ θ¯; B) = d · z.

Then, DDC applies clipping to d by:

$$d^{\prime}=\begin{cases}C,&\text{if}d>C\\ d,&d\in[-C,C]\\ -C,&\text{if}d<-C\end{cases}$$
$${\bar{\vartheta}};{\mathcal{B}}){\mid}{\mid}^{2}\!{\mid}$$
$$\left(7\right)$$
$\mathbf{v}$
$$(\mathbf{6})$$

where C is a non-negative constant. The gradient estimate then becomes ∇ˆ ∆L
′(∆ ⊙ θ¯; B) = d
′· z, which is plugged into ZO-SGD. We provide theoretical evidence to highlight that DDC can reduce the variance of the gradient estimate and thereby stabilize the training. We first propose the following theorem as a preliminary to our analysis. The proof of Theorem 1 is available in Appendix A.

Theorem 1. *Clipped gradient estimate* ∇ˆ ∆L
′(∆ ⊙ θ¯; B) *is an unbiased estimate of the full gradient* of loss w.r.t quantization sclaes ∇∆L(∆ ⊙ θ¯).

Since d
′2 ≤ d 2 by definition of DDC in Eq. 6, the following inequality holds:

E[||∇ˆ ∆L
$${}^{\prime}(\Delta\odot{\bar{\theta}};{\mathcal{B}})||^{2}]=\mathbb{E}[d^{\prime2}||z$$
′2||z||2] ≤ E[d
2||z||2] = E[||∇ˆ ∆L(∆ ⊙ θ¯; B)||2] (7)
Therefore, the element-wise variance of the clipped gradient estimate has the following derivation:
V ar[∇ˆ ∆k L ′(∆ ⊙ θ¯; B)] = E[||∇ˆ ∆k L ′(∆ ⊙ θ¯; B)||2] − E[∇ˆ ∆k L ′(∆ ⊙ θ¯; B)]2 ≤ E[||∇ˆ ∆k L(∆ ⊙ θ¯; B)||2] − E[∇ˆ ∆k L ′(∆ ⊙ θ¯; B)]2 = V ar[∇ˆ ∆k L(∆ ⊙ θ¯; B)] + E[∇ˆ ∆k L(∆ ⊙ θ¯; B)]2 − E[∇ˆ ∆k L ′(∆ ⊙ θ¯; B)]2 = V ar[∇ˆ ∆k L(∆ ⊙ θ¯; B)] + ∇∆k L(∆ ⊙ θ¯)2− E[∇ˆ ∆k L ′(∆ ⊙ θ¯; B)]2 (8)
$$)]\leq V a r[\hat{\nabla}_{\Delta_{k}}]$$
By Theorem 1, *V ar*[∇ˆ ∆k L
′(∆ ⊙ θ¯; B)] ≤ *V ar*[∇ˆ ∆k L(∆ ⊙ θ¯; B)] holds almost surely.

Our experimental results in Section 4.3 also reveal that DDC effectively stabilizes the training through rectifying abnormal loss values, and the ablation study also demonstrates that QZO is relatively robust to the magnitude of C.

## 3.2.3 Algorithm

We summarize QZO in Algorithm 1. Note that although the quantization scales are perturbed per parameter in the pseudo code, in practice one may perturb the entire quantization scales of a linear layer to save training time (Malladi et al., 2023). Remarks QZO seamlessly combines ZO with quantization and therefore leads to maximum reduction in memory usage: gradients and optimizer states are eliminated while model weights are compressed. To further cut memory usage on activations, one can divide the batch size while increasing the total number of optimization steps, or release activations during forward passes since ZO does not need to cache activations for gradient computation.

## 4 Experiments

4.1 EXPERIMENTAL SETUP Models and Datasets We evaluate our approach using three 7B-level LLMs, namely OPT-6.7B (Zhang et al., 2022), Llama-2-7B (Touvron et al., 2023b), and Llama-3.1- Algorithm 1 Quantized Zeroth-order Optimization Require: quantization scales ∆ ∈ R
d, quantized weights θ¯ ∈ R
d, loss function L : R
d → R
learning rate ηt, optimization steps T, perturbation scales ϵ, clipping threshold C. for t = 1*...T* do Sample batch of inputs B and random seed s ∆ ← PERTURB_SCALES(∆, ϵ, s)
ℓ+ ← L(∆ ⊙ θ¯; B) ▷ 1 st forward pass
∆ ← PERTURB_SCALES(∆, −2ϵ, s)
ℓ− ← L(∆ ⊙ θ¯; B) ▷ 2 nd forward pass
∆ ← PERTURB_SCALES(∆, ϵ, s) d ← (ℓ+ − ℓ−)/(2ϵ)
d
′ ← CLIP(d, −*C, C*) ▷ Directional derivative clipping, Eq. 6 Reset random number generator with seed s for ∆i ∈ ∆ do z ∼ N (0, 1)
∆i ← max(∆i − ηt ∗ d
′ ∗ z, 0) ▷ Ensure non-negative scales end for end for procedure PE R T U R B_SC A L E S(∆, ϵ, s)
Reset random number generator with seed s for ∆i ∈ ∆ do z ∼ N (0, 1)
∆i ← ∆i + ϵz end for end procedure 8B (Grattafiori et al., 2024), and one large-sized model with 13B parameters, i.e., Llama-2-13B (Touvron et al., 2023b). For QZO, the 7B models are quantized to 4-bit while the 13B model to 2-bit to test QZO's effectiveness under extreme quantization. Following prior work (Malladi et al., 2023), we evaluate our approach on five popular NLP datasets covering both classification and generation tasks. Specifically, for classification, we use SST2 (Socher et al., 2013) and three subsets from SuperGLUE collection (Wang et al., 2019), i.e., RTE (Dagan et al., 2005; Haim et al., 2006; Giampiccolo et al., 2007; Bentivogli et al., 2009), CB (De Marneffe et al., 2019) and BoolQ (Clark et al., 2019). For generation, we use SQuAD (Rajpurkar et al., 2016), which is a question answering dataset. Following the common practice, we randomly sample 1,000 examples for training, 500 examples for validation, and 1,000 examples for testing. We report accuracy for classification tasks, whereas the metric for generation tasks is F1 score. Baseline Methods A wide range of baseline methods is chosen for comparison to justify QZO's effectiveness. Specifically, QZO is compared with: (1) Zero-Shot, and Zero-Shot-Q, the original and quantized zero-shot models, respectively, which are viewed as the lower-bound; (2) Fine-tuning on 16-bit models, which is considered as the upper-bound;2(3) MeZO (Malladi et al., 2023), which applies ZO to un-quantized models. Implementation Details For 4-bit quantization, we apply GPTQ (Frantar et al., 2023) to the 7B-level LLMs (i.e., OPT-6.7B, Llama-2-7B, and Llama-3.1-8B).3 The quantization group in GPTQ
is set to 128. For extreme quantization in 2-bit, we apply AQLM (Egiazarian et al., 2024) with 1 codebook of 16 bits to Llama-2-13B.4 We use QZO to fine-tune the channel-wise scales in AQLM.

Following prior work (Egiazarian et al., 2024; Tseng et al., 2024), the un-quantized parts are jointly fine-tuned using the regular SPSA and ZO-SGD. To accelerate QZO fine-tuning in 2-bit, we also

| Model       | Memory      | Classficiation   | Generation   |      |       |       |      |      |
|-------------|-------------|------------------|--------------|------|-------|-------|------|------|
| Precision   | Profiling   | SST-2            | RTE          | CB   | BoolQ | SQuAD |      |      |
| Fine-tuning | 16 bits     | 26.8GB           | 95.4         | 79.8 | 73.2  | 69.6  | 77.6 |      |
| Zero-Shot   | 16 bits     | -                | 61.2         | 55.2 | 51.8  | 59.5  | 36.5 |      |
| Zero-Shot-Q | 4 bits      | -                | 60.1         | 53.8 | 51.8  | 59.1  | 35.9 |      |
| MeZO        | 16 bits     | 14.8GB           | 93.0         | 64.6 | 67.9  | 66.8  | 79.6 |      |
| QZO         | 4 bits      | 4.8GB            | 87.6         | 61.7 | 67.9  | 66.4  | 78.5 |      |
| OPT-6.7B    | Fine-tuning | 16 bits          | 26.0GB       | 92.8 | 63.2  | 60.7  | 75.0 | 83.7 |
| Zero-Shot   | 16 bits     | -                | 58.1         | 61.7 | 32.1  | 66.0  | 55.6 |      |
| Zero-Shot-Q | 4 bits      | -                | 58.5         | 53.4 | 35.7  | 64.6  | 53.6 |      |
| MeZO        | 16 bits     | 14.8GB           | 83.5         | 58.1 | 67.9  | 69.6  | 80.7 |      |
| QZO         | 4 bits      | 5.0GB            | 90.0         | 59.2 | 69.6  | 68.2  | 85.5 |      |
| Llama-2-7B  | Fine-tuning | 16 bits          | 31.9GB       | 93.7 | 71.5  | 62.5  | 83.4 | 84.9 |
| Zero-Shot   | 16 bits     | -                | 59.6         | 45.8 | 46.4  | 66.1  | 64.8 |      |
| Zero-Shot-Q | 4 bits      | -                | 58.7         | 50.2 | 37.5  | 65.0  | 59.2 |      |
| MeZO        | 16 bits     | 20.5GB           | 92.5         | 70.0 | 91.1  | 83.4  | 86.9 |      |
| QZO         | 4 bits      | 6.3GB            | 93.0         | 66.8 | 69.6  | 78.2  | 88.3 |      |
| Llama-3-8B  |             |                  |              |      |       |       |      |      |

Table 2: Training statistics collected on SST-2. Overall, QZO is both memory-efficient and computation-efficient.

Trainable

Paramters

Total FLOPs

(SST-2)

| Trainable Paramters   |            |             |             |
|-----------------------|------------|-------------|-------------|
| Fine-tuning           | 6.65 × 109 | 2.17 × 1016 |             |
| OPT-6.7B              | MeZO       | 6.65 × 109  | 9.91 × 1017 |
| QZO                   | 5.03 × 107 | 8.19 × 1013 |             |
| Fine-tuning           | 6.74 × 109 | 2.47 × 1016 |             |
| Llama-2-7B            | MeZO       | 6.74 × 109  | 1.13 × 1018 |
| QZO                   | 5.06 × 107 | 2.26 × 1016 |             |
| Fine-tuning           | 8.03 × 109 | 2.48 × 1016 |             |
| Llama-3.1-8B          | MeZO       | 8.03 × 109  | 1.13 × 1018 |
| QZO                   | 5.45 × 107 | 7.9 × 1016  |             |

modify AQLM's Triton inference kernel to disentangle matrix reconstruction and matrix-vector multiplication.5 For QZO, we set the learning rate to 10−7, the batch size to 16, training steps to 20k, the perturbation scale ϵ to 10−3, and the clipping threshold C to 100. For fine-tuning experiments with SGD, the learning rate is initialized as 8 × 10−4 with a linearly scheduled decay, and the batch size is set to 8. A single Nvidia RTX 4090 GPU (24GB) is used for all experiments (except Fine-tuning, which requires an A100 80GB GPU). For MeZO, we adopt the official code.6

## 4.2 Main Results

QZO on 4-bit Quantization Table 1 compares QZO with different baselines across three model architectures on the five NLP datasets. The detailed training statistics are shown in Table 2. Following MeZO, memory profiling measures the peak memory usage during the first 100 optimization steps. The dataset used for memory profiling is SST2 and the (per-device) batch size is set to 1 to test the minimum VRAM requirement. We summarize our main findings below.

| Model       | Memory      | Classification   | Generation   |      |       |       |      |      |
|-------------|-------------|------------------|--------------|------|-------|-------|------|------|
| Precision   | Profiling   | SST-2            | RTE          | CB   | BoolQ | SQuAD |      |      |
| Llama-2-13B | Zero-Shot-Q | 2 bits           | -            | 57.6 | 53.1  | 46.4  | 69.2 | 55.4 |
| QZO         | 2 bits      | 5.78GB           | 80.5         | 54.5 | 55.4  | 70.2  | 59.4 |      |

Table 3: Experiments based on Llama-2-13B. QZO demonstrates strong potential under extreme quantization.

![7_image_0.png](7_image_0.png)

QZO demonstrates effectiveness consistently across all model architectures and NLP tasks. Specifically, QZO achieves significant improvements over Zero-Shot-Q, meaning that QZO successfully fine-tunes these quantized LLMs. On most datasets, QZO performs on par with MeZO, despite using 3× *less memory*; sometimes QZO even beats MeZO with noticeable margins, e.g., 85.5 vs. 80.7 on SQuAD when using Llama-2-7B. It is worth highlighting that MeZO is based on 16-bit models while QZO is based on 4-bit models *with much lower precision*. Compared with the upper-bound, i.e., fine-tuning, the gap is still huge on some of the tasks. This makes sense because ZO methods rely merely on forward passes for gradient estimation, which would be much less accurate than that of backpropagation. QZO demonstrates both memory-efficiency and computation-efficiency. QZO pushes memoryefficiency to the extreme by eliminating gradients and optimizer states while reducing weights precision. Therefore, the memory usage is minimal compared to the baselines like MeZO and Fine-tuning. Table 2 compares QZO with MeZO and Fine-tuning on learnable parameter count and FLOPs. It is worth noting that QZO uses only about 1% of the trainable parameters and 1% of the FLOPs of MeZO. This is because QZO only fine-tunes the continuous quantization scale while leaving most weights (which are quantized) fixed. We expect the difference to be further increased when more powerful quantization methods are used. QZO on 2-bit Quantization Table 3 shows that QZO beats the zero-shot model with significant margins. The results strongly justify QZO's effectiveness under extreme quantization. QZO has the potential to be applied to on-device learning scenarios for edge devices.

## 4.3 Ablation Studies

In this section, we mainly evaluate the DDC component. Recall that DDC (Directional Derivative Clipping, Eq. 6) clips abnormal directional derivatives estimated via QZO (i.e., d in Eq. 6). We use QZO to train two Llama-2-7B models, with and without using DDC, and record the directional derivatives and loss values for the first 1,000 steps. Figure 2 shows that without DDC the directional

![8_image_0.png](8_image_0.png)

derivative often gets abnormal values that go beyond the range of [−*C, C*] (C is the clipping threshold in Eq. 6), leading to NaN value for the loss (which means the training collapses).

We also study how sensitive QZO is to the clipping threshold C. Intuitively, a small C should effectively avoid abnormal directional derivatives, but may suffer from underfitting due to a small optimization step size. A large C fixes this issue, but also increases the risk of producing abnormal values. For quantitative analysis, we train Llama-2-7B models on SST-2 with different values of C and record the final accuracies. The results are presented in Figure 3. The trend of the line plot suggests underfitting at C ≤ 50, and stable performances can be observed when C ≥ 75. When C is set to a value bigger than 150, the training becomes unstable and sometimes collapse, which algins with the observation in Figure 2 (QZO w/ DDC can be seen as setting C to an infinitely large value).

## 5 Conclusion, Limitations, And Future Work

QZO enables fine-tuning quantized neural networks via ZO, which greatly reduces memory usage related to model weights, gradients, and optimizer states. We show that QZO works for a wide range of LLMs and is compatible with both scalar-based and codebook-based quantization methods. When using 4-bit LLMs, QZO achieves performance on par with MeZO, while using 3× less GPU memory. In the extreme quantization scenario, QZO successfully fine-tunes 2-bit LLama-2-13B across different NLP datasets. The results indicate that QZO has the potential to be applied to on-device learning for edge devices. In addition to LLMs, we have also applied QZO to fine-tuning text-to-image generation models, namely Stable Diffusion 3.5 Large (Esser et al., 2024). The results and discussions are presented in Appendix F. QZO fine-tunes Stable Diffusion 3.5 Large using only 12.4GB of memory in a single Nvidia RTX 4090 GPU. The visualization results are also encouraging: the data distribution generated by QZO is visually closer to the ground truth than the zero-shot model. However, QZO has some limitations. First, QZO's performance depends on how good the quantization method is. Specifically, if the quantization method has a large quantization error, this makes the forward passes in ZO noisy and therefore could make the gradient estimation less accurate. On the other hand, QZO could benefit from a better quantization method with higher accuracy. Therefore, practitioners are suggested to choose high-precision quantization methods for QZO to maximize the gains. Second, the performance on diffusion models lags behind LLMs because there is a noticeable gap between QZO's images and the ground truth. This may be caused by the mismatch in the noise scheduling between ZO and diffusion. One potential solution is to redesign the noise scheduling in ZO such that it aligns with diffusion. We leave this as future work.

## 6 Ethics Statement

We clarify that our research is free from the issues in the code of ethics. Our research focuses on the efficiency of LLM training and does not include any human subjects. The datasets used do not include sensitive content that violates data privacy.

## 7 Reproducibility Statement

Our code has been publicly released to ensure reproducibility of experiments. All the datasets involved are also publicly accessible. The proof of Theorem 1 is provided in the Appendix.

## 8 Acknowledgement

This research is supported by Hong Kong Research Grants Council Early Career Scheme (No.

22200824).

## References

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. Quarot: Outlier-free 4-bit inference in rotated LLMs. In *The Thirty-eighth Annual Conference on Neural Information Processing Systems*, 2024. URL https://openreview.net/forum?id=dfqsW38v1X.

Noga Bar and Raja Giryes. Zoqo: Zero-order quantized optimization. In *ICASSP 2025-2025 IEEE*
International Conference on Acoustics, Speech and Signal Processing (ICASSP), pp. 1–5. IEEE,
2025.

Luisa Bentivogli, Peter Clark, Ido Dagan, and Danilo Giampiccolo. The fifth pascal recognizing textual entailment challenge. TAC, 7(8):1, 2009.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of NAACL-HLT, pp. 2924–2936, 2019.

Andrew R Conn, Katya Scheinberg, and Luis N Vicente. *Introduction to derivative-free optimization*.

SIAM, 2009.

Ido Dagan, Oren Glickman, and Bernardo Magnini. The pascal recognising textual entailment challenge. In *Machine learning challenges workshop*, pp. 177–190. Springer, 2005.

Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser. The commitmentbank: Investigating projection in naturally occurring discourse. In *proceedings of Sinn und Bedeutung*,
volume 23, pp. 107–124, 2019.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. GPT3.int8(): 8-bit matrix multiplication for transformers at scale. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), *Advances in Neural Information Processing Systems*, 2022. URL https://openreview.net/forum?id=dXiGWqBoxaD.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of quantized llms. *Advances in neural information processing systems*, 36:10088–10115, 2023.

Vage Egiazarian, Andrei Panferov, Denis Kuznedelev, Elias Frantar, Artem Babenko, and Dan Alistarh. Extreme compression of large language models via additive quantization. In International Conference on Machine Learning, pp. 12284–12303. PMLR, 2024.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In *Forty-first international conference on machine learning*, 2024.

Chen Feng, Shaojie Zhuo, Xiaopeng Zhang, Ramchalam K Ramakrishnan, Zhaocong Yuan, and Andrew Z Li. Stepping forward on the last mile. *Advances in Neural Information Processing* Systems, 37:94851–94870, 2024.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. OPTQ: Accurate quantization for generative pre-trained transformers. In *The Eleventh International Conference on Learning* Representations, 2023. URL https://openreview.net/forum?id=tcbBPnfwxS.

Alireza Ganjdanesh, Reza Shirkavand, Shangqian Gao, and Heng Huang. Not all prompts are made equal: Prompt-based pruning of text-to-image diffusion models. *arXiv preprint arXiv:2406.12042*, 2024.

Danilo Giampiccolo, Bernardo Magnini, Ido Dagan, and William B Dolan. The third pascal recognizing textual entailment challenge. In Proceedings of the ACL-PASCAL workshop on textual entailment and paraphrasing, pp. 1–9, 2007.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv preprint arXiv:2407.21783*, 2024.

R Bar Haim, Ido Dagan, Bill Dolan, Lisa Ferro, Danilo Giampiccolo, Bernardo Magnini, and Idan Szpektor. The second pascal recognising textual entailment challenge. In *Proceedings of the* Second PASCAL Challenges Workshop on Recognising Textual Entailment, volume 7, pp. 785–794, 2006.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for llm compression and acceleration. In *MLSys*, 2024.

Sijia Liu, Pin-Yu Chen, Xiangyi Chen, and Mingyi Hong. signsgd via zeroth-order oracle. In International conference on learning representations, 2019.

Yifei Liu, Jicheng Wen, Yang Wang, Shengyu Ye, Li Lyna Zhang, Ting Cao, Cheng Li, and Mao Yang. Vptq: Extreme low-bit vector post-training quantization for large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 8181–8196, 2024.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D Lee, Danqi Chen, and Sanjeev Arora. Fine-tuning language models with just forward passes. Advances in Neural Information Processing Systems, 36:53038–53075, 2023.

Dang Nguyen, Wenhan Yang, Rathul Anand, Yu Yang, and Baharan Mirzasoleiman. Mini-batch coresets for memory-efficient language model training on data mixtures. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview. net/forum?id=bAFVlpFQvT.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. Squad: 100,000+ questions for machine comprehension of text. In *Proceedings of the 2016 Conference on Empirical Methods in* Natural Language Processing, pp. 2383–2392, 2016.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pp. 400–407, 1951.

Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In *Proceedings of the 2013 conference on empirical methods in natural language processing*, pp.

1631–1642, 2013.

James C Spall. Multivariate stochastic approximation using a simultaneous perturbation gradient approximation. *IEEE transactions on automatic control*, 37(3):332–341, 1992.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. *arXiv preprint arXiv:2302.13971*, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023b.

Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, and Christopher De Sa. Quip \#:
Even better llm quantization with hadamard incoherence and lattice codebooks. In International Conference on Machine Learning, pp. 48630–48656. PMLR, 2024.

Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Superglue: A stickier benchmark for general-purpose language understanding systems. *Advances in neural information processing systems*, 32, 2019.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. SmoothQuant:
Accurate and efficient post-training quantization for large language models. In *Proceedings of the* 40th International Conference on Machine Learning, 2023.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. *arXiv preprint arXiv:2205.01068*, 2022.

Jiawei Zhao, Zhenyu Zhang, Beidi Chen, Zhangyang Wang, Anima Anandkumar, and Yuandong Tian. Galore: Memory-efficient llm training by gradient low-rank projection. In International Conference on Machine Learning, pp. 61121–61143. PMLR, 2024.

Jiajun Zhou, Yifan Yang, Kai Zhen, Ziyue Liu, Yequan Zhao, Ershad Banijamali, Athanasios Mouchtaris, Ngai Wong, and Zheng Zhang. Quzo: Quantized zeroth-order fine-tuning for large language models. *arXiv preprint arXiv:2502.12346*, 2025.