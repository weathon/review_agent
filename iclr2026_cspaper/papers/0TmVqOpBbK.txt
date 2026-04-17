# Scaling Laws Meet Model Architecture: To- Ward Inference-Efficient Llms

Song Bian∗
UW-Madison Tao Yu†
Amazon Web Services Shivaram Venkataraman UW-Madison Youngsuk Park Amazon Web Services

## Abstract

Scaling the number of parameters and the size of training data has proven to be an effective strategy for improving large language model (LLM) performance. Yet, as these models grow increasingly powerful and widely deployed, the cost of inference has become a pressing concern. Despite its importance, the tradeoff between model accuracy and inference efficiency remains underexplored. In this work, we examine how key architectural factors, hidden size, the allocation of parameters between MLP and attention (mlp-to-attention ratio), and groupedquery attention (GQA), influence both inference cost and accuracy. We introduce a conditional scaling law that augments the Chinchilla framework with architectural information, along with a search framework for identifying architectures that are simultaneously inference-efficient and accurate. To validate our approach, we train more than 200 models spanning 80M to 3B parameters and 8B to 100B training tokens, and fit the proposed conditional scaling law. Our results show that the conditional scaling law reliably predicts optimal architectural choices and that the resulting models outperform existing open-source baselines. Under the same training budget, optimized architectures achieve up to 2.1% higher accuracy and 42% greater inference throughput compared to LLaMA-3.2.

![0_image_0.png](0_image_0.png)

Figure 1: **Scaling sweep results.** (left) Inference throughput (tokens/s) and (right) Scaling-lawpredicted training loss contours over hidden size dmodel and mlp-to-attention ratio. Our conditional scaling law enables concurrent gains in throughput and reductions in predicted training loss under a fixed parameter budget. Dotted points indicate the architectures used to fit the scaling law.

1

## 1 Introduction

Scaling law studies Kaplan et al. (2020); Hoffmann et al. (2022); Muennighoff et al. (2023); Krajewski et al. (2024); Abnar et al. (2025) have shown that increasing model parameters, training tokens, dataset quality, and compute budget consistently reduces pre-training loss, improves downstream task performance Hendrycks et al. (2021); Austin et al. (2021), and enables the emergence of novel capabilities Wei et al. (2022). These insights have driven the development of many state-of-the-art large language models Touvron et al. (2023); Yang et al. (2025); Guo et al. (2025). However, as the field advances, it has become increasingly clear that focusing exclusively on training overlooks the practical challenges of deploying these models at scale Chien et al. (2023); Wu et al. (2024); Muhamed et al. (2023). A major limitation of existing scaling laws is their omission of inference costs, which constitute the dominant expense in deploying large models in real-world applications Sardana et al. (2023); Park et al. (2024). Moreover, the growing use of LLMs in reasoning systems highlights the need for scaling laws that account for inference costs Snell et al. (2024); Brown et al. (2024); Luo et al. (2024); Qi et al. (2024); Guan et al. (2025). Therefore, we ask the following question:
Can we explicitly capture the trade-off between inference efficiency and accuracy of large language models?

To address this question, a recent study Sardana et al. (2023) proposed scaling laws that incorporate the total FLOPs from both training and inference. However, their formulation requires estimating the total number of tokens generated over a model's entire lifespan. Because inference is performed repeatedly during deployment, this assumption renders the proposed scaling law impractical for real-world use. Another study Bian et al. (2025) extends Chinchilla scaling laws by incorporating model architecture. However, this work has notable limitations. First, the study considers only the aspect ratio, defined as hidden size over number of layers, as the architectural factor. Yet, as shown in Figure 2, aspect ratio alone fails to capture the full range of factors that influence inference efficiency in large language models. Second, the depth of the model strongly influences accuracy: cutting layers tends to impair the model's generalization after fine-tuning Petty et al. (2023). Finally, the study lacks a general framework for incorporating broader architectural factors, including hidden size and GQA, into scaling laws.

![1_image_0.png](1_image_0.png)

In this work, we fix the number of layers and study the effect of other architectural factors, including GQA, hidden size, and the mlp-to-attention ratio. This design choice is motivated by recent open-weight models such as LLaMA Touvron et al. (2023), Qwen Yang et al. (2025), Gemma Team et al. (2024a), and Phi Abdin et al. (2024), which, despite having a comparable number of parameters, adopt markedly different architectural designs. Our primary goal is to investigate how model architecture influences both inference efficiency and model accuracy. We begin by comparing the inference efficiency of models with identical parameter counts but varying architectures. Next, we train over 200 models, ranging from 80M to 297M parameters on up to 30B tokens, to systematically characterize the relationship between architectural design and accuracy. Guided by these empirical findings, we introduce a conditional extension of the Chinchilla scaling laws that incorporates architectural parameters, establishing a general framework for identifying model architectures that balance inference efficiency and performance. Finally, we validate this framework by fitting the proposed scaling law on models between 80M and 297M parameters, and evaluating its predictions when scaling up to pretrain 3B-parameter models. Our results demonstrate that, under identical training setups, the derived optimal 3B-parameter ar-
Figure 2: Although larger models generally achieve lower inference throughput than smaller ones, Qwen2.5-1.5B outperforms Qwen3-0.6B. Despite having the same number of layers, Qwen2.5-1.5B benefits from a higher hidden size, GQA, and mlp-to-attention ratio.

chitecture achieves up to 42% higher inference throughput than the LLaMA-3.2-3B architecture, while maintaining better accuracy.

## 2 Background

Accurately predicting the performance of large language models during scaling is essential. This enables us to answer key questions: (i) what is the optimal allocation of available resources between model size and training tokens, and (ii) what performance gains can be expected from additional resources? Fortunately, the model loss has been observed to follow a power-law relationship with respect to the number of parameters N and training tokens D Hoffmann et al. (2022); Muennighoff et al. (2023) with:

$$L(N,D)=E+{\frac{A}{N^{\alpha}}}+{\frac{B}{D^{\beta}}}$$
$$(1)$$

where L is the model loss, N is the number of total parameters and D is the number of tokens used for training and A, B, E, α, β are parameters to be learned. To fit the learnable parameters in Eq. (1), Chinchilla Hoffmann et al. (2022) employs two strategies: (i) training models with a fixed number of parameters while varying the number of training tokens, and (ii) training models under a fixed compute budget1, varying both parameters and tokens. The resulting data are combined to fit the learned parameters in Eq. (1). With the fitted scaling laws, Chinchilla addresses the following question to determine optimal allocation:
L(*N, D*) s.t. FLOPs(N, D) = C (2)

$$\arg\operatorname*{min}_{N,D}L(N,D){\mathrm{~s.t.~FLOPs}}(N,D)=C$$
$$(2)$$

where C denotes the resource constraint, N the total number of parameters, and D the number of training tokens. In this paper, we do not address how to optimally allocate compute between model size and training data under a fixed compute budget. Instead, our focus is on identifying model architectures that optimize inference efficiency and accuracy under fixed parameter and token budgets. For example, given a model with 7B parameters trained on 14T tokens, we study how to design an architecture that satisfies both efficiency and accuracy requirements.

## 3 Model Architecture-Aware Scaling Laws 3.1 Model Architecture Variations

The architecture of a decoder-only transformer is composed of a sequence of stacked decoder blocks, each sharing the same structure to facilitate model-parallel deployment across devices. Under this design, the overall architecture of dense LLMs is primarily determined by the hidden size and the MLP intermediate size, which together specify the attention and MLP layers structure. This work studies the optimal model architecture given a fixed total number of non-embedding parameters Nnon-embed (at different levels). Although the number of layers nlayer also plays a critical role (closely related to aspect ratio (Petty et al., 2023)), varying nlayer under a fixed Nnon-embed substantially impacts both inference cost and accuracy (Tay et al., 2021; Alabdulmohsin et al., 2023). Therefore, we fix nlayer and focus on the effects of hidden size dmodel and the mlp-to-attention ratio rmlp/attn on inference efficiency (§3.2) and accuracy (§3.3), noting that nlayer still varies across different Nnon-embed levels. In §3.3, we introduce a conditional scaling law to predict the performance of architectural variants, and in §3.4, we present a lightweight framework for identifying architectures that optimally balance inference efficiency and accuracy. Note that the number of attention parameters is primarily determined by the hidden size dmodel and the attention projection dimension, since most open-weight models adopt non-square *q, k, v* projection matrices, as seen in Gemma (Team et al., 2024a) and Qwen3 (Yang et al., 2025). For consistency, we fix the per-head dimension dhead to 64 for models with Nnon-embed ≤1B and to 128 1The compute cost is approximated as FLOPs(N, D) ≈ 6ND in Hoffmann et al. (2022); Muennighoff et al. (2023), where N denotes the number of parameters and D the number of training tokens. In this work, we adopt the same settings as prior studies.

![3_image_0.png](3_image_0.png)

for models with Nnon-embed ≥3B. Consequently, to maintain a constant rmlp/attn, we adjust the number of attention heads nhead rather than altering the projection dimension directly. This design choice also provides flexibility to incorporate architectural variants such as grouped-query attention.

## 3.2 Inference Efficiency

Inspired by the success and widespread adoption of open-weight dense models such as Qwen3 (Yang et al., 2025), LLaMA-3.2 (Dubey et al., 2024), and the Gemma-2 (Team et al., 2024b) family, we construct architectural variants by modifying the configurations of the LLaMA-3.2 and Qwen3 dense models (Figure 12-14 in Appendix F). In addition to hidden size and the mlp-to-attention ratio, we find that group-query attention has a critical impact on inference efficiency, even though it only modestly reduces the number of attention parameters (by shrinking the key and value matrices). To disentangle these effects, we conduct controlled ablations of hidden size, MLP-to-attention ratio, and GQA under the following setups:
- hidden size d*model*: fix Nnon-embed, rmlp/attn and GQA= 4, vary dmodel and number of attention heads nhead (Figure 3 left).

- mlp-to-attention ratio rmlp/*attn*: fix Nnon-embed, dmodel and GQA= 4, vary nhead and intermediate size (Figure 3 right).

- *GQA:* fix Nnon-embed, dmodel and rmlp/attn, vary nhead and number of key-value heads (Appendix F).

Figure 3 shows the ablation of varying hidden sizes dmodel and mlp-to-attention rmlp/attn on the LLaMA-3.1-8B model variants. We observe that larger hidden size (or fewer attention heads) and higher mlp-to-attention ratios improve inference throughput. Similar trends are observed in the LLaMA-3.2-1B and 3B model variants (Appendix F). These gains arise in part because larger dmodel and higher rmlp/attn reduce the total FLOPs, as detailed in the inference FLOPs analysis (Appendix K). In addition, these architectural choices shrink the KV cache, lowering I/O cost during inference and further improving throughput Adnan et al. (2024). Figure 11 in Appendix F presents the GQA ablation, confirming prior observations Ainslie et al. (2023) that increasing GQA consistently improves inference throughput. A comparable set of ablation experiments on Qwen3 models, also reported in Appendix F, further corroborates these findings.

## 3.3 A Conditional Scaling Law

Improving inference efficiency should not come at the expense of significantly reducing model accuracy, making it crucial to understand how architectural choices affect accuracy and training loss. Because training large-scale language models is prohibitively expensive, a common strategy is to study smaller models and use scaling laws to extrapolate insights to larger scales, for example, the Chinchilla scaling laws (Hoffmann et al., 2022). However, incorporating multiple architectural factors into such laws remains challenging. To address this, we examine the effect of architectural

![4_image_0.png](4_image_0.png)

![4_image_1.png](4_image_1.png)

Lo s s

![4_image_2.png](4_image_2.png)

choices on training loss L in a conditional manner, varying one factor at a time while keeping the others fixed.

hidden size d**model**. We note that dmodel generally scales linearly with 
√Nnon-embed. Assuming squared attention weight matrices, the number of attention parameters Nattn can be expressed as

$$4d_{m o d e l}^{2}\propto N_{\mathrm{atm}}=N_{\mathrm{non\mbox{-}embed}}\times\frac{r}{r+1},$$
,
where r = rmlp/attn is fixed, and the constant factor 4 arises from the query, key, value, and output projection layers in each attention block. To capture this scaling behavior, we normalize dmodel by
√Nnon-embed and examine its relation to loss L in Figure 4. The resulting U-shaped curves L(d/√N | r, N, D) exhibit nearly identical optima across different model sizes. Moreover, Figure 4 confirms that excessively large hidden sizes, which reduce the number of attention heads nhead, can degrade accuracy—a phenomenon consistently observed in prior analyses of transformer capacity and head allocation (Kaplan et al., 2020; Hoffmann et al., 2022).

mlp-to-attention ratio rmlp/**attn**. Figure 5 illustrates how the loss varies with rmlp/attn, conditioned on dmodel fixed at different levels, where we consistently observe a U-shaped curve L(r | d/√*N, N, D*). While the attention mechanism is central to the success of transformers (Vaswani, 2017), recent open-weight models have allocated a progressively smaller fraction of parameters to attention as overall model size increases (e.g., LLaMA and Qwen families). Our analysis indicates that this trend is not universally optimal: there exists an interior optimum in the allocation of attention parameters, and deviating from it in either direction degrades model performance. This suggests that careful tuning of the mlp-to-attention ratio is critical for scaling transformers effectively. As shown in Figures 4 and 5, both hidden size and the MLP-to-attention ratio exhibit U-shaped relationships with training loss. To capture these trends, we fit the function c0 + c1 log x + c2/x separately for x = rmlp/attn and dmodel/
√Nnon-embed. This formulation effectively models the U-
shaped behavior while ensuring sublinear growth as x increases. However, incorporating rmlp/attn, dmodel, N, and D into a unified, architecture-aware scaling law remains challenging. Since fitting a single all-purpose scaling law L(d/√*N, r, N, D*) is unrealistic across all possible configurations, we instead propose a two-step conditional approach:
1. For given N and D, obtain the optimal loss Lopt(*N, D*) = min L(*N, D*) = min E +
A
Nα +
B
Dβfrom the Chinchilla scaling law (Eq. 1) as a reference point.

2. Calibrate the loss of architectural variants L(d/√N, r | *N, D*) relative to this reference.

We focus on two simple and transparent calibration schemes:
- (multiplicative)

$$L(d/\sqrt{N},r\mid N,D)=(a_{0}+a_{1}\log(\frac{d}{\sqrt{N}})+a_{2}\frac{\sqrt{N}}{d})\cdot(b_{0}+b_{1}\log r+\frac{b_{2}}{r})\cdot L_{\rm opt}\tag{3}$$
 - (additive) $L(d/\sqrt{N},r\mid N,D)=(a_0+a_1\log(\frac{d}{\sqrt{N}})+a_2)$  -
$$\log({\frac{d}{\sqrt{N}}})+a_{2}{\frac{\sqrt{N}}{d}})+(b_{1}\log r+{\frac{b_{2}}{r}})+L_{\mathrm{opt}}$$

Here, ai and bi are learnable parameters that are shared across all *N, D*. Note that both functional forms assume the effects of rmlp/attn and dmodel on loss are separable.

3.4 SEARCHING FOR INFERENCE-EFFICIENT ACCURATE MODELS With the conditional scaling law, we can identify architectures that are both inference-efficient and accurate by solving the following optimization problem: given N, D, and a set of architectural choices P,
argmaxP
IN (P), s.t. L(P | *N, D*) ≤ Lt, (4)
where IN (P) denotes the inference efficiency of an architecture P with total Nnon-embed parameters, and Lt,(≥ Lopt) is the maximum allowable training loss.

As shown in Figure 11 (Appendix F), GQA has a substantial impact on inference efficiency; However, unlike hidden size and the mlp-to-attention ratio, GQA does not exhibit a consistent continuous relationship with loss (Figure 24, Appendix I) and is highly variable, making it challenging to identify settings that achieve both accuracy and efficiency. Fortunately, the search space for GQA is relatively small once Nnon-embed, dmodel, and rmlp/attn are fixed, since GQA must be a prime factor of the number of attention heads nhead. In practice, we perform a local GQA search by enumerating feasible values and applying early stopping once performance falls below that of the GQA= 4 baseline. Algorithm 1 summarizes our overall framework for identifying inference-efficient and accurate architectures. Algorithm 1: Searching for Inference-Efficient Accurate Model Input: Model parameters N, training tokens D, target loss Lt; inference efficiency IN (·);
optional: the optimal loss Lopt(*N, D*)
Train smaller models to fit the Chinchilla scaling laws (Eq. 1) if Lopt(*N, D*) is unavailable Solve the constrained optimization (Eq. 4) for dmodel, rmlp/attn and corresponding architecture P
Perform a local search over GQA values with early stopping to maximize inference efficiency return Final model architecture {P, GQA}

## 4 Experiment Setup

We first detail the experimental setup of training, inference, and downstream task evaluation, and then describe how we derive the conditional scaling law and scale up to larger sizes. Training Setup. We sample the training data from Dolma-v1.7 Soldaini et al. (2024), which contains data from 15 different sources. Tokens are sampled with probability proportional to each source's contribution, ensuring the sampled dataset preserves a similar distribution to Dolmav1.7. We train decoder-only LLaMA-3.2 (Dubey et al., 2024) style transformers with Nnon-embed in {80M, 145M, 297M, 1B, 3B}, for each Nnon-embed, we obtain model architecture candidates by varying hidden size dmodel/
√Nnon-embed and mlp-to-attention ratio rmlp/attn. (changing intermediate size

![6_image_0.png](6_image_0.png)

and number of attention heads nhead) while holding other architectural factors fixed e.g. GQA= 4.

A full list of over 200 model architectures used can be found in Appendix D. All models are trained on 100Nnon-emb tokens (5× Chinchilla optimal) to ensure convergence. We tuned training hyperparameters (mainly following prior work Chen et al. (2025)), with a full list in Appendix E. Inference Setup. We evaluate the inference efficiency using the vLLM framework Kwon et al. (2023). By default, inputs consist of 4096 tokens and outputs of 1024 tokens. We report the averaged inference throughput (tokens/second) from 5 repeated runs. Unless otherwise specified, all experiments are conducted on NVIDIA Ampere A100 GPUs (40GB) with vLLM. LLM Evaluation Setup. Following prior works Biderman et al. (2023); Zhang et al. (2024),
we evaluate pretrained models in the zero-shot setting using lm-evaluation-harness2 on nine benchmarks: ARC-Easy Clark et al. (2018), ARC-Challenge Clark et al. (2018), LAM- BADA Paperno et al. (2016), HellaSwag Zellers et al. (2019), OpenBookQA Mihaylov et al. (2018), PIQA Bisk et al. (2020), SciQ Welbl et al. (2017), WinoGrande Sakaguchi et al. (2021), and CoQA Reddy et al. (2019). Fitting Scaling Laws. Following Gadre et al. (2024); Bian et al. (2025), we use the Levenberg- Marquardt algorithm to fit the conditional scaling laws (Eq. 3). The Levenberg–Marquardt algorithm does least-squares curve fitting by estimating βˆ as the solution to arg minβPm i=1 [yi − f(xi, β)]2, where (xi, yi) are the observed data pairs. Note that instead of fitting the Chinchilla scaling law, we empirically searched over architecture variants to find the optimal loss Lopt(*N, D*) for Nnon-embed <1B scale.

We scale up the scale law fitting in the following progressive manner:
(Task 1) fit on the 80M results and evaluate on 145M results; (Task 2) fit on 80, 145M results and evaluate on 297M results; (Task 3) fit on 80, 145, 297M results and evaluate on 1B results; This ensures a robust and consistent way of scaling up the model sizes and evaluating our conditional scaling law. Following prior work Kumar et al. (2024), we evaluate the fitted scaling law with mean squared error (MSE) metric, defined as 1n Pn i=1(li − ˆli)
2 where li denotes the actual loss and ˆlithe predicted loss. We additionally report the Spearman's rank correlation coefficient Spearman (1961) to compare predicted and actual rankings. Both metrics are calculated on the val data points.

## 5 Experiment Results

We begin by evaluating the predictive performances of the conditional scaling laws with multiplicative calibration. We then conduct ablation studies to assess the impact of data selection and to Table 1: **Large-Scale Model Results.** We evaluate the scaling laws at 1B and 3B scales by training Panda-1B, Surefire-1B, and Panda-3B, and compare them with LLaMA-3.2-1B and LLaMA-3.23B, respectively. The Avg. column reports the mean accuracy across the nine downstream tasks. Panda-1B and 3B are trained using the optimal architectural configurations predicted by our scaling laws, whereas Surefire-1B and 3B satisfy the loss constraint in Eq. (4) and achieve Pareto optimality.

Models dmodel f*size* nlayers GQA dmodel/

√N r Loss (↓) Avg. (↑)

LLaMA-3.2-1B 2048 8192 16 4 0.066 4.80 2.803 54.9

Panda-1B 2560 4096 16 4 0.082 1.07 2.782 57.0

Surefire-1B 2560 6144 16 9 0.082 3.6 2.804 55.4

LLaMA-3.2-3B 3072 8192 28 3 0.058 4.80 2.625 61.9

Panda-3B 4096 4096 28 3 0.077 1 2.619 62.5

Surefire-3B 4096 4096 28 7 0.077 1 2.620 62.6

![7_image_0.png](7_image_0.png)

Lo
evaluate the performance of the scaling laws under additive calibration. Finally, we apply the fitted scaling laws to guide the training of large-scale models following the search framework (§5.1). Predictive Accuracy. As Task 1-3 described in §4, we fit the conditional scaling laws on 80M, (80M, 145M), and (80M, 145M, 297M) loss-architecture data points, and subsequently evaluate on 145M, 297M, and 1B data, respectively. In Figure 6, the low MSE and high Spearman correlation in tasks across different model scales validate the effectiveness and strong predictive performance of the proposed conditional scaling laws.

Ablation of Outliers. The mlp-to-attention ratio rmlp/attn of open-weights models typically fall between 0.5 and 5, for example, the mlp-to-attention ratio for LLaMA-3.2-1B, LLaMA-3.2-3B, and Qwen3-8B are 4.81, 1.5, and 4.67, respectively. In Figure 6, we fit the conditional scaling law using only model architectures with rmlp/attn ∈ [0.5, 5]. We ablate this choice by training model architectures with outlier rmlp/attn below 0.5 and above 5 (such as 0.1, 12.6) in Appendix D. In Figure 25 (left) and Figure 25 (center) in Appendix J, we show on Task 3 a comparison of fitting the conditional scaling law without and with these outliers (with a clear Spearman correlation score degradation), which suggests to exclude extreme outliers for better predicted performances.

Ablation of Calibration. In Figure 25 (right), we ablate an alternative formulation of the scaling laws with additive calibration, as discussed in §3.3. The results on Task 3 show that multiplicative and additive calibrations achieve similar MSE and Spearman correlations. Note that, unlike the conventional unified formulation, both calibrations assume that the effects of rmlp/attn and dmodel on loss are separable. We further ablate more complex joint, non-separable formulations in Appendix J and find that they do not provide superior predictive performance. The two-step reference-andcalibration framework appears robust enough that simple calibrations perform well.

| √ N          | r      | Loss (↓)   | Avg. (↑)   |     |         |      |       |      |
|--------------|--------|------------|------------|-----|---------|------|-------|------|
| Models       | dmodel | fsize      | nlayers    | GQA | dmodel/ |      |       |      |
| LLaMA-3.2-3B | 3072   | 8192       | 28         | 3   | 0.058   | 4.80 | 2.625 | 61.9 |
| Panda-3B     | 4096   | 4096       | 28         | 3   | 0.077   | 1    | 2.619 | 62.5 |
| Panda-3B◦    | 4096   | 4608       | 28         | 3   | 0.076   | 1.23 | 2.606 | 62.5 |

Table 2: **3B Model Ablations.** We assess the robustness of fitting-data strategy at 3B scale by training Panda-3B (using 80M, 145M, and 297M data) and Panda-3B◦(using only on 1B data), and compare both with LLaMA-3.2-3B. Avg. denotes mean accuracy across nine downstream tasks.

## 5.1 Optimal Model Architecture

Validating the conditional scaling law. We validate the conditional scaling law at the 1B scale by applying multiplicative calibration on Task 3 using data from the (80M, 145M, and 297M) model variants. The learned parameters are a0 = 2.697, a1 = 0.0974, a2 = 0.0078, b0 = 0.3870, b1 = 0.0063, and b2 = 0.*0065*.

From this, we obtain the optimal architectural configuration of dmodel/
√N = 0.08, r = 1.032 for 1B model by solving ∂L
∂dmodel
= 0 and ∂L
∂r 
= 0. Using this configuration, we train a LLaMA-3.2-style 1B dense model on 100B tokens, denoted as Panda-1B. Panda-1B outperforms the open-weight LLaMA-3.2-1B baseline configs by 2.1% on average across downstream tasks (Table 1). Figure 7 (left) further confirms the effectiveness of the conditional scaling law by showing that Panda-1B achieves the lowest training loss among the exhaustively trained 1B variants under the same setup. We also scale up our methodology to 3B models. Using the same approach but with data from the 80M, 145M, 297M, and 1B variants, we fit the scaling law and obtain dmodel/
√N = 0.08 and r = 1.055 for the Panda 3B model. Trained on 100B tokens, Panda-3B outperforms the open weight LLaMA-3.2-3B configuration by 0.6% on average across downstream tasks (Table 1).

With all components in place, we apply the search framework for inference-efficient and accurate models (Alg. 1). For the Nnon-embed = 1B and 3B setting trained on 100B tokens, we set the target loss Lt to match the training loss achieved by the LLaMA-3.2-1B and LLaMA-3.2-3B architectures, respectively.

Ablation of inference efficiency. Although inference efficiency IN (P) could, in principle, be expressed analytically, it depends heavily on hardware and inference configurations. Therefore, rather than solving for IN (P) directly, we search over feasible configurations Pithat satisfy the loss constraint on A100 with vLLM and select Pareto-optimal points, which we denote as Surefire1B and Surefire-3B. Surefire-1B and Surefire-3B outperform LLaMA-3.2-1B and LLaMA-3.2-3B on downstream tasks (Table 1 with details in Appendix L) and deliver up to 42% higher inference throughput (Figure 7, center and right). We also ablate inference efficiency using both vLLM and SGLang Zheng et al. (2023) on A100 and NVIDIA H200 GPUs (Appendix F, G). The results remain consistent with our vLLM–A100 evaluation: Surefire-1B and 3B outperform LLaMA-3.2-1B and 3B across all settings, achieving up to 47% higher throughput with SGLang on H200. This demonstrates that the efficiency gains transfer across serving stacks and hardware platforms. Detailed throughput statistics are provided in Table 6. Ablation of fitting data strategy. While we adopt a progressive strategy for selecting fitting data across tasks (§4), results from small models (e.g., 80M) may not reliably predict behaviors at larger scales such as 3B. To assess this, we fit the conditional scaling law for the 3B model using only the 1B variants. As shown in Figure 8, fitting with 1B data yields lower MSE and higher Spearman correlation when predicting 3B behavior, suggesting that the law's coefficients shift with model size. We therefore refit the law with multiplicative calibration using only the 1B variants, yielding the coefficients a0 = 2.319, a1 = 0.238, a2 = 0.0176, b0 = 0.5104, b1 = 0.0051, and b2 = 0.0062. This produces an alternative optimal configuration for the 3B model, with dmodel/
√N = 0.074 and r = 1.229. We train a 3B model (Panda-3B◦) under this configuration on 100B tokens and compare it with both LLaMA-3.2-3B and Panda-3B (fitted from 80M, 145M, 297M, and 1B data). As shown in Table 2, Panda-3B◦achieves a lower training loss and comparable downstream accuracy to Panda9

![9_image_0.png](9_image_0.png)

3B, with detailed results given in Appendix L. These findings suggest that when scaling up, it is often sufficient, and sometimes preferable, to fit the law using models within a closer size range to the target, such as about one third of its scale.

## 6 Related Work

Scaling laws are powerful tools to predict the performance of large language models. Existing scaling laws Hoffmann et al. (2022); Muennighoff et al. (2023); Sardana et al. (2023); Kumar et al. (2024); Gadre et al. (2024); Ruan et al. (2024) characterize how model performance varies with model size, dataset size, data quality, and compute budget. With the rise of Mixture-of-Experts (MoE) Shazeer et al. (2017); Guo et al. (2025), a powerful architecture for large language models, recent studies Krajewski et al. (2024); Abnar et al. (2025) extend scaling laws to account for the number of experts, expert granularity, active parameters, and sparsity. Due to space constraints, we defer additional related work to Appendix B.

## 7 Limitations And Future Work

While our team has made notable progress, several open challenges remain that offer promising directions for future research. First, due to limitations in resources and time, our evaluation does not extend to 7B models. Second, our analysis is restricted to dense models, and it remains unclear whether the results extend to Mixture of Experts (MoE) architectures Shazeer et al. (2017). While we report inference efficiency measurements for MoE models under varying architectural choices in Appendix M, we have not yet established scaling laws for MoE architectures. Finally, our analysis is limited to pre-training, and it remains unclear how the results would change under post-training.

## 8 Conclusion

This work explores the trade-off between model accuracy and inference cost under a fixed training budget. We begin by demonstrating how architectural choices influence both inference throughput and model accuracy. Building on this, we extend Chinchilla scaling laws to incorporate architectural factors and propose a two-step conditional framework for optimal architecture search: (i) train small models to fit the conditional scaling law (Eq. 3), and (ii) solve Eq. 4 for the predicted optimal architecture, followed by a local search over GQA to maximize inference efficiency. Using the fitted scaling laws and our framework, we trained models up to 3B parameters, achieving up to 42% higher inference throughput and 2.1% accuracy gains across nine downstream tasks. In Table 7 and Table 8 of Appendix H, we compare design choices across existing open-source models at the 1B and 3B scales, further underscoring the need for our inference-efficient accurate model designs.

## Reproducibility Statement

All experiments in this work were conducted using publicly available frameworks. Section 4 provides details of our training, inference, and evaluation setups. In particular, we used Megatron-LM (Shoeybi et al., 2019) for model training, vLLM (Kwon et al., 2023) and SGLang (Zheng et al., 2023) for efficient inference, and lm-eval-harness (Gao et al., 2024a)
for standardized evaluations.

## Acknowledgements

Song Bian and Shivaram Venkataraman acknowledge the support of the NSF Diamond project OAC- 2311767 (Democratizing Large Neural Network Model Training for Science).

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sebastien Bubeck, Ronen Eldan, Suriya Gunasekar, ´
Michael Harrison, Russell J Hewett, Mojan Javaheripi, Piero Kauffmann, et al. Phi-4 technical report. *arXiv preprint arXiv:2412.08905*, 2024.

Samira Abnar, Harshay Shah, Dan Busbridge, Alaaeldin Mohamed Elnouby Ali, Josh Susskind, and Vimal Thilak. Parameters vs flops: Scaling laws for optimal sparsity for mixture-of-experts language models. *arXiv preprint arXiv:2501.12370*, 2025.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. *arXiv preprint arXiv:2303.08774*, 2023.

Muhammad Adnan, Akhil Arunkumar, Gaurav Jain, Prashant J Nair, Ilya Soloveychik, and Purushotham Kamath. Keyformer: Kv cache reduction through key tokens selection for efficient generative inference. *Proceedings of Machine Learning and Systems*, 6:114–127, 2024.

Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury Zemlyanskiy, Federico Lebron, and Sumit ´
Sanghai. Gqa: Training generalized multi-query transformer models from multi-head checkpoints. *arXiv preprint arXiv:2305.13245*, 2023.

Ibrahim M Alabdulmohsin, Xiaohua Zhai, Alexander Kolesnikov, and Lucas Beyer. Getting vit in shape: Scaling laws for compute-optimal model design. Advances in Neural Information Processing Systems, 36:16406–16425, 2023.

Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. *arXiv preprint arXiv:2108.07732*, 2021.

Song Bian. *Architecture Design for Efficient LLM Training and Inference*. PhD thesis, The University of Wisconsin-Madison, 2026.

Song Bian, Minghao Yan, and Shivaram Venkataraman. Scaling inference-efficient language models. *arXiv preprint arXiv:2501.18107*, 2025.

Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O'Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al.

Pythia: A suite for analyzing large language models across training and scaling. In International Conference on Machine Learning, pp. 2397–2430. PMLR, 2023.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 7432–7439, 2020.

Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V Le, Christopher Re, and ´
Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated sampling. arXiv preprint arXiv:2407.21787, 2024.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Keshigeyan Chandrasegaran, Michael Poli, Daniel Y Fu, Dongjun Kim, Lea M Hadzic, Manling Li, Agrim Gupta, Stefano Massaroli, Azalia Mirhoseini, Juan Carlos Niebles, et al. Exploring diffusion transformer designs via grafting. *arXiv preprint arXiv:2506.05340*, 2025.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde De Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. *arXiv preprint arXiv:2107.03374*, 2021.

Mengzhao Chen, Chaoyi Zhang, Jing Liu, Yutao Zeng, Zeyue Xue, Zhiheng Liu, Yunshui Li, Jin Ma, Jie Huang, Xun Zhou, et al. Scaling law for quantization-aware training. arXiv preprint arXiv:2505.14302, 2025.

Andrew A Chien, Liuzixuan Lin, Hai Nguyen, Varsha Rao, Tristan Sharma, and Rajini Wijayawardana. Reducing the carbon impact of generative ai inference (today and in 2035). In *Proceedings* of the 2nd workshop on sustainable computer systems, pp. 1–7, 2023.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.

arXiv preprint arXiv:1803.05457, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Tri Dao and Albert Gu. Transformers are ssms: Generalized models and efficient algorithms through structured state space duality. *arXiv preprint arXiv:2405.21060*, 2024.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Samir Yitzhak Gadre, Georgios Smyrnis, Vaishaal Shankar, Suchin Gururangan, Mitchell Wortsman, Rulin Shao, Jean Mercat, Alex Fang, Jeffrey Li, Sedrick Keh, et al. Language models scale reliably with over-training and on downstream tasks. *arXiv preprint arXiv:2403.08540*, 2024.

Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac'h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model evaluation harness, 07 2024a. URL https://zenodo.org/records/12608602.

Yizhao Gao, Zhichen Zeng, Dayou Du, Shijie Cao, Peiyuan Zhou, Jiaxing Qi, Junjie Lai, Hayden Kwok-Hay So, Ting Cao, Fan Yang, et al. Seerattention: Learning intrinsic sparse attention in your llms. *arXiv preprint arXiv:2410.13276*, 2024b.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv* preprint arXiv:2312.00752, 2023.

Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang.

rstar-math: Small llms can master math reasoning with self-evolved deep thinking. arXiv preprint arXiv:2501.04519, 2025.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. *arXiv* preprint arXiv:2103.03874, 2021.