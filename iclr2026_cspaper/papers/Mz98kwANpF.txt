000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

## Abstract

Parameter-Efficient Fine-Tuning (PEFT) is essential for adapting Large Language Models (LLMs). In practice, LLMs are often required to handle a diverse set of tasks from multiple domains, a scenario naturally addressed by multi-task learning (MTL). Within this MTL context, a prevailing trend involves LoRA variants with multiple adapters or heads, which advocate for structural diversity to capture taskspecific knowledge. Our findings present a direct challenge to this paradigm. We first show that a simplified multi-head architecture with high inter-head similarity substantially outperforms complex multi-adapter and multi-head systems. This leads us to question the multi-component paradigm itself, and we further demonstrate that a standard single-adapter LoRA, with a sufficiently increased rank, also achieves highly competitive performance. These results lead us to a new hypothesis: learning task-shared representations provides a highly effective and promising path towards multi-task learning, offering a powerful alternative to the architectural isolation of task-specific features. To validate this, we propose Align-LoRA, which incorporates an explicit loss to align task representations within the shared adapter space. Theoretical analysis and experiments confirm that Align-LoRA significantly surpasses baselines, establishing a simpler yet more effective paradigm for adapting LLMs to multiple tasks. The code is available anonymously.

## 1 Introduction

In recent years, large language models (LLMs) have demonstrated unprecedented performance across a wide range of natural language processing (NLP) tasks (Brown, 2020; Zhao et al., 2023; Chang et al., 2024b). Despite their strong generalization abilities, LLMs often require further adaptation to align with domain-specific requirements or to incorporate updated knowledge (Agiza et al., 2024; Xin et al., 2024). Supervised fine-tuning (SFT) plays a critical role in this process, but full parameter fine-tuning (FFT), which updates all model parameters, poses significant challenges in terms of computational and memory costs (Mao et al., 2025). To address these demands, parameter-efficient fine-tuning (PEFT) methods have been proposed to adapt LLMs by updating only a small subset of parameters (Han et al., 2024; Chang et al., 2024a). Among these, Low-Rank Adaptation (LoRA) (Hu et al., 2021) has become a widely adopted approach. It approximates the full-rank weight update matrix by decomposing it into two low-rank matrices: a down-projection matrix A and an up-projection matrix B. In practice, adapting LLMs often involves data from multiple domains or tasks, naturally aligning with the multi-task learning (MTL) paradigm. Consequently, this has motivated the development of LoRA variants specifically designed for MTL. An early approach is the Multi-Adapter architecture, which employs multiple, distinct pairs of downprojection (A) and up-projection (B) matrices for different tasks (Wang et al., 2023). To improve parameter efficiency, the Multi-Head architecture was introduced, typically sharing a single A matrix while maintaining multiple task-specific head matrices (B) (Tian et al., 2024). Furthermore, many of these multi-component architectures employ a routing mechanism, inspired by the Mixtureof-Experts (MoE) framework, to dynamically select or weigh the outputs of different adapters for a given input. Recent prevalent methods like R-LoRA (Liu et al., 2025) further refine this by explicitly encouraging diversity among heads to mitigate redundancy. **Despite architectural differences,**
1

# Align, Don'T Divide: Revisiting The Lora Architecture In Multi-Task Learning

Anonymous authors Paper under double-blind review 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 these methods are all built on a common premise: that effective multi-task adaptation requires architectural isolation of task-specific knowledge. However, this pursuit of architectural isolation introduces a significant practical drawback. The specialized components cannot be merged into the backbone model, resulting in non-negligible inference latency from their processing in every forward pass. Motivated by this trade-off, our work begins with an empirical re-evaluation of this premise, seeking a multi-task adaptation method that eliminates such a latency penalty. We first reveal a paradoxical finding: by simplifying a complex multi-head architecture into a model we term M-LoRA (which removes the dynamic router), we observe that its performance surpasses its more complex counterparts. This occurs despite the simplified model exhibiting higher inter-head similarity, a result that directly challenges the prevailing assumption that component diversity is beneficial. This outcome led us to a more fundamental question: **Is the multi-component structure truly necessary for effective multi-task adaptation?** In pursuit of an answer, we discovered that merely increasing the rank of a standard, single-adapter LoRA is sufficient to match or even outperform these intricate multi-component variants. Collectively, the findings that a simplified multi-head model excels and that a high-rank single-head model is equally or more effective point to a new and unexplored hypothesis: learning task-shared representations provides a highly effective and promising path towards multi-task learning, offering a powerful alternative to the architectural isolation of task-specific features. To directly validate this hypothesis and operationalize this principle, we propose Align-LoRA. This method enhances a standard LoRA by augmenting its training objective with a component based on the Kullback-Leibler (KL) Divergence (Kullback & Leibler, 1951), which encourages the alignment of task representations in the shared low-rank space without adding parameters or inference overhead. Our key contributions are fourfold:
- We demonstrate that a simplified multi-head LoRA (**M-LoRA**) with high head similarity outperforms complex variants, challenging the prevailing assumption that architectural isolation of task-specific knowledge is necessary.

- We show that simply **increasing the rank** of a standard LoRA can match the performance of multi-component architectures, questioning their fundamental necessity for multi-task learning.

- We propose a new hypothesis: learning task-shared representations provides a highly effective and promising path towards multi-task learning, offering a powerful alternative to the architectural isolation of task-specific features.

- We introduce **Align-LoRA**, a novel method that validates our hypothesis by explicitly aligning representations, achieving superior performance and setting a new direction for multi-task PEFT.

## 2 Related Works 2.1 Low-Rank Adaptation (Lora)

Current LLMs typically adopt a decoder-only architecture, consisting of stacked transformer blocks (Zhao et al., 2023). Each block contains two core components with residual connections: a multi-head self-attention (MHA) layer and a feed-forward network (FFN) (Vaswani, 2017). Both layers rely on dense learnable weight matrices W for feature transformation.

To efficiently adapt LLMs under resource constraints, LoRA (Hu et al., 2021) offers an effective solution. It is inspired by the hypothesis that the intrinsic dimensionality of weight updates in LLMs is low. LoRA approximates the weight update ∆W using two low-rank matrices A ∈ R
r×n and B ∈ R
m×r, whereW ∈ R
m×n is the original weight matrix. The rank r is chosen to be significantly smaller than min(m, n), reducing the number of trainable parameters from O(mn) to O(r(m+n)). The forward pass is modified as follows:
h = (W + ∆W)x = Wx + BAx, (1)
where ∆W = BA denotes the low-rank update. A key advantage of LoRA is that after training, the low-rank update ∆W can be merged back into the original weights W, introducing zero inference overhead.

![2_image_0.png](2_image_0.png)

Several works have built upon the original LoRA framework. AdaLoRA (Zhang et al., 2023) dynamically allocates the rank budget, while DoRA (Liu et al., 2024b) decomposes weight updates into magnitude and direction. Other methods like PiSSA (Meng et al., 2025) and NLoRA (Guo et al., 2025) have focused on improving performance through better initialization and decomposition strategies, highlighting the ongoing effort to enhance LoRA's effectiveness.

## 2.2 Multi-Component Lora

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 To adapt LoRA for multi-task learning (MTL), a natural extension is to employ multiple trainable components. Early works proposed the **Multi-Adapter** architecture, which utilizes multiple independent LoRA adapters (i.e., distinct BiAi pairs) for different tasks. Notable examples of this approach include Multi-LoRA (Wang et al., 2023), MixLoRA (Li et al., 2024), LoRAMoE (Dou et al., 2023), MoELoRA (Liu et al., 2024a), and LoRAHub (Huang et al., 2023). The **Multi-Head** architecture was developed to improve parameter efficiency, driven by the key insight that LoRA's matrices have distinct roles. It was observed that down-projection matrices (A) capture redundant, **task-general knowledge**, while up-projection matrices (B) learn diverse, task-specific features. Consequently, the Multi-Head design, exemplified by methods like HydraLoRA (Tian et al., 2024), MALoRA (Wang et al., 2024), MTLLoRA (Agiza et al., 2024), and R-LoRA (Liu et al., 2025), employs a single shared A matrix with multiple distinct Bi heads. To further enhance task specialization within this paradigm, R-LoRA introduced a randomization technique to also reduce similarity among the head matrices. Figure 1 illustrates the architectural differences between three key paradigms: the original LoRA, the multi-adapter architecture, and the multi-head architecture. The **Multi-Head** architecture, exemplified by methods like HydraLoRA (Tian et al., 2024) and R-
LoRA (Liu et al., 2025), uses a shared down-projection matrix A and multiple head matrices Bi.

The aggregated weight update in this structure is a dynamically weighted sum of each head's output:

$$\Delta\mathbf{W}=\sum_{i=1}^{N}\omega_{i}(\mathbf{x})\cdot\mathbf{B}_{i}\mathbf{A}.$$

$$(2)$$
-) $\overline{=}$
ωi(x) · BiA. (2)
Drawing inspiration from the Mixture-of-Experts (MoE) framework, this dynamic routing mechanism employs a learnable routing matrix Wr and a gating function, such as softmax or Top-K, to assign weights to each "expert" adapter based on the input x. The widely used softmax-based router is formulated as:
ω(x) = Softmax(Wrx). (3)
However, this complexity introduces a critical trade-off. A significant drawback of input-dependent routing is that the aggregated update ∆W can no longer be pre-computed. Consequently, the adapter weights **cannot be merged** into the frozen backbone model post-training. This results in nonnegligible inference latency, as the router and multiple heads must be processed for each forward pass, sacrificing one of LoRA's most significant practical advantages.

![3_image_0.png](3_image_0.png)

## 3 Observations

In this section, we critically examine the prevailing assumption that component diversity are essential for effective multi-task adaptation with LoRA. By questioning the fundamental necessity of the prevalent multi-head paradigm, our investigation leads to a new hypothesis centered on the pivotal role of shared knowledge.

## 3.1 M-Lora: A Simplified Variant

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Prevalent methods like R-LoRA (Liu et al., 2025) are built on the premise that encouraging diversity among adapter heads is crucial for capturing distinct, task-specific knowledge. To directly test the hypothesis on head diversity, we propose **M-LoRA** (Base Multi-Head LoRA), a minimal ablation variant of R-LoRA. While preserving R-LoRA's core designs, such as multi-head randomization for initialization and input differentiation via Dropout, M-LoRA's sole architectural change is the removal of the dynamic routing module. Instead, it aggregates the outputs of its head matrices by simple summation, allowing us to directly study the effect of eliminating explicit, input-dependent diversification. The framework of M-LoRA is provided in the Appendix B.

## 3.2 The Paradox Of Diversity: Less Is More

We fine-tune the Qwen2.5-3B (Qwen Team, 2024) model using HydraLoRA, R-LoRA, and M- LoRA on a benchmark comprising five distinct tasks: QNLI (Wang, 2018), PiQA (Bisk et al., 2020), Winogrande (Sakaguchi et al., 2021), ARC (easy & challenge) (Clark et al., 2018), and GSM8K (Cobbe et al., 2021). To quantify inter-head similarity, we compute a matrix of pairwise cosine similarities between all flattened head vectors (Bi). The final metric is the mean of this matrix's off-diagonal values. All Experimental details in this work, including implementation specifics, hyperparameter settings, dataset descriptions, baseline configurations, and other relevant information, are documented in the Appendix G. Our findings reveal a paradox regarding the role of head diversity in multi-task adaptation. Figure 2, which plots the inter-head cosine similarity, shows that R-LoRA successfully achieves its design goal of maximizing diversity, exhibiting the lowest similarity. In stark contrast, M-LoRA, which lacks any diversity-enforcing mechanism, displays the opposite effect, yielding a high degree 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

| Schemes    | QNLI   | PiQA   | Winogrande   | ARC   | GSM8K   | Avg   | %Para   |
|------------|--------|--------|--------------|-------|---------|-------|---------|
| HydraLoRA  | 81.91  | 84.21  | 70.92        | 87.21 | 45.95   | 74.04 | 0.45    |
| w/o Router | 81.33  | 83.51  | 70.14        | 86.46 | 45.35   | 73.58 | 0.41    |
| R-LoRA     | 82.03  | 85.55  | 71.84        | 87.69 | 46.25   | 74.67 | 0.45    |
| M-LoRA     | 82.52  | 86.76  | 72.95        | 88.15 | 46.85   | 75.45 | 0.41    |

Table 1: Comparative study of several multi-head LoRA variants across five tasks.

of head redundancy with similarity medians consistently exceeding 0.85. Paradoxically, as demonstrated in Table 1, this high-redundancy model achieves superior multi-task performance. Despite its architectural simplicity, M-LoRA consistently and significantly outperforms the more complex HydraLoRA and R-LoRA. This outcome presents a fundamental contradiction to the philosophy of prior work: the architectural configuration that seemingly violates the principle of head diversity actually enhances multi-task generalization.

## 3.3 Task-Shared Vs. Task-Specific Learning

This section seeks to explain the surprising effectiveness of M-LoRA. Its strong multi-task generalization, achieved despite high inter-head similarity, challenges conventional assumptions and offers a new perspective on the core principles of multi-task adaptation in LoRA. Improving multitask learning (MTL) has largely followed two distinct paths: isolating **task-specific** knowledge to mitigate interference, or enhancing **task-shared** knowledge to improve generalization. To date, the predominant focus has been on the former. Recent multi-task LoRA methods, such as LoRA MoE (Dou et al., 2023) and R-LoRA (Liu et al., 2025), have predominantly focused on isolating task-specific knowledge. **In contrast, the alternative path of actively enhancing task-shared** knowledge within the LoRA framework has remained unexplored. M-LoRA challenges this specialization-focused paradigm. We hypothesize that the high similarity is not a sign of failed specialization, but a feature derived from the architecture's implicit regularization. The key mechanism is the interplay between removing the router and retaining the multi-head dropout. In models like R-LoRA, the heads are treated as competing "specialists," and the dynamic router attempts to select the single best expert, which can often lead to redundancy or load imbalance. In contrast, by replacing the dynamic router with simple averaging, M-LoRA compels the multiple B heads to form a collaborative ensemble. As illustrated in Figure 4, M-LoRA achieves this by retaining the multi-head dropout mechanism from R-LoRA (Liu et al., 2025). The dropout forces each head to learn from a slightly different input perspective. By forcing all heads to contribute (via summation), they are compelled to converge on a robust, task-general representation that works well from all perspectives. To validate this mechanism, we performed an ablation on a non-dropout multi-head variant, HydraLoRA. As shown in Table 1, removing the router from HydraLoRA ('w/o Router') causes its average performance to drop, while M-LoRA's collaborative ensemble achieves the highest performance among tested variants. This strongly confirms that the multi-head dropout is the critical factor that, when combined with router removal, transforms the heads from isolated "specialists" into effective "collaborators" and significantly enhances task-general learning. The success of M-LoRA suggests a revised viewpoint: learning task-shared representations provides a highly effective and promising path towards multi-task learning, offering a powerful alternative to the architectural isolation of task-specific features.

## 4 Increasing Rank: A Unified Adapter

Given M-LoRA's strong performance with highly redundant heads, which largely learn shared knowledge, this section explores a critical question: Is the multi-head architecture itself truly necessary for multi-task generalization, or does it merely serve as a means to increase total trainable parameters rather than offering genuine benefits? To test this, we design a straightforward yet powerful experiment. We abandon the multi-component structure entirely and instead use a standard, single-adapter LoRA. We reallocate the entire parame270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

| indicates results from Tian et al. (2024). Metrics Base LoRA LoRAHub* LoRA MoE* HydraLoRA R-LoRA LoRA† M-LoRA 7B 31.61 37.05 39.70 40.30 41.46 42.24 42.21 42.83 13B 38.42 40.73 41.90 43.70 44.31 44.96 45.02 46.16 % Param - 0.06 1.24 2.98 0.34 0.34 0.34 0.32   |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| 8 , etc.) indicates the rank value used for each variant. Metrics Base LoRA4 LoRA8 LoRA9 LoRA10   | HydraLoRA   | R-LoRA   | M-LoRA   |       |       |       |       |       |
|---------------------------------------------------------------------------------------------------|-------------|----------|----------|-------|-------|-------|-------|-------|
| 7B                                                                                                | 39.82       | 43.21    | 46.66    | 48.18 | 49.51 | 49.12 | 49.51 | 49.74 |
| 14B                                                                                               | 45.33       | 48.18    | 51.82    | 52.74 | 54.23 | 53.76 | 54.08 | 54.18 |
| Rank                                                                                              | -           | 4        | 8        | 9     | 10    | 4     | 4     | 4     |
| % Param                                                                                           | -           | 0.10     | 0.20     | 0.22  | 0.25  | 0.25  | 0.25  | 0.22  |

ter budget of the complex variants into this single adapter by simply increasing its rank, r. Following the experimental setup of HydraLoRA (Tian et al., 2024), we conduct fine-tuning on a curated subset of the Flanv2 dataset (Liu et al., 2022). This training data is sampled from dozens of individual datasets and organized into ten distinct task categories, providing comprehensive training across both Natural Language Understanding and Natural Language Generation capabilities. We then evaluate the models' multi-task generalization on the challenging Big-Bench Hard(BBH) benchmark (Suzgun et al., 2022), which is designed to test generalization. The results, presented in Table 2 and Table 3, reveal a clear trend. Across different base models, including LLaMA2 (Touvron et al., 2023) and Qwen2.5 (Qwen Team, 2024), the performance of a standard LoRA adapter consistently improves with its rank. Crucially, when its rank is scaled to a comparable parameter count, a simple, single-adapter LoRA achieves performance that is competitive with, and at times superior to, sophisticated multi-component architectures such as LoRA- Hub (Huang et al., 2023), LoRA MoE (Liu et al., 2024a), HydraLoRA (Tian et al., 2024), and R-LoRA (Liu et al., 2025). This finding provides compelling evidence that **the architectural complexity of multi-adapter**
designs may not be a prerequisite for achieving strong multi-task generalization. Our results indicate that a simple, unified adapter with sufficient capacity delivers comparable performance.

This challenges not only the trend toward elaborate structures but also the underlying strategy of isolating task-specific features, suggesting it is a less effective path to generalization than previously assumed and that the research focus on specialized components may warrant reconsideration.

## 5 Beyond Rank: Representation Alignment

Our investigation in the preceding sections has led to two critical conclusions. First, based on our analysis in Section 3, we formed a guiding hypothesis: learning task-general, shared knowledge may be more critical than enforcing task-specific separation. Second, our findings demonstrate that the architectural complexity introduced by multi-component designs is unnecessary for achieving strong multi-task generalization. This calls into question the prevailing assumption that specialized structures are a prerequisite for effective multi-task learning. These conclusions motivate a shift in our approach. Moving away from structural complexity, we adopt the standard, high-rank LoRA architecture as a rational and efficient baseline. This simplification, however, raises a more fundamental inquiry: How can we move beyond merely increasing the rank and take a more principled step towards better multi-task learning? This leads us to the two central questions addressed in this work:
1. How can we validate our hypothesis about the primacy of shared knowledge?

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 2. How can we design a mechanism to explicitly enhance the learning of these shared representations within a single, unified LoRA adapter?

To address these questions, we introduce **Align-LoRA**, a novel framework as follows.

## 5.1 Align-Lora

To enhance multi-task generalization, we introduce **Align-LoRA**, a method that encourages the model to learn task-shared features. Align-LoRA introduces an alignment loss, Lalign, to explicitly minimize the statistical distance between the low-dimensional representations generated by the shared LoRA down-projection matrix, A. **To the best of our knowledge, this is the first work to** systematically apply statistical distance metrics for this purpose within the multi-task LoRA framework, drawing inspiration from their foundational use in domain adaptation (Pan et al., 2010).

Our primary approach instantiates Lalign using the **Kullback-Leibler (KL) divergence** (Kullback &
Leibler, 1951), a classic statistical method for measuring the distance between distributions. However, we hypothesize that the principle of aligning representations is broadly applicable and not contingent on a single metric. To validate this, we introduce a variant that employs the multi-kernel Maximum Mean Discrepancy (MK-MMD) (Gretton et al., 2012). The strong performance of both instantiations, as we will demonstrate, validates our core thesis: that explicitly aligning the low-dimensional representations of different tasks is a robust and viable strategy. The formulation for MK-MMD is detailed in Appendix E.

Let T = {T1, T2*, . . . , T*M} be a set of M tasks. For an input x from task Ti with contextualized embeddings XTi, the representation we align is the output of the down-projection matrix:
ϕTi
(x) = A · XTi
. (4)
Our choice to operate on this rank-r latent space is motivated by two key factors. First, it directly targets the component responsible for shared knowledge. Recent studies have consistently found that the down-projection matrix A tends to learn task-general features while the up-projection matrix B captures task-specific knowledge (Agiza et al., 2024; Wang et al., 2024; Tian et al., 2024). By applying our alignment loss to the output of A, we directly enhance its natural function, promoting the development of robust, shared representations. Second, this approach is highly efficient. As demonstrated in prior work (Liu et al., 2025), performing operations in the low-dimensional space significantly reduces computational load and GPU memory demand, ensuring our method's practicality. To measure and minimize the distance between the representation distributions from different tasks, we model the batch-wise distribution for each task Ti as a multivariate Gaussian with a diagonal covariance matrix, N (µi, diag(σ 2 i)). The mean µi and variance σ 2 iare empirically estimated from the output vectors {ϕTi(x)} in a given batch. Since standard KL divergence is asymmetric, we employ a symmetric formulation. The total alignment loss, LKL, is the sum of these symmetric pairwise divergences across all unique task pairs (Ti, Tj ) where *i < j*:

$${\mathcal{L}}_{\mathrm{KL}}=\sum_{i=1}^{M}\sum_{j=i+1}^{M}{\frac{1}{2}}\left(D_{\mathrm{KL}}(p_{T_{i}}\|p_{T_{j}})+D_{\mathrm{KL}}(p_{T_{j}}\|p_{T_{i}})\right),$$
$$(5)$$

where pTiis the modeled Gaussian distribution over the low-dimensional representations of task Ti.

This loss drives the empirical mean and variance of each task's distribution toward a common value. The alignment loss is incorporated as an auxiliary objective to the primary language modeling task. The total loss function is therefore defined as:
Ltotal = Llm + λ · Lalign, (6)
where Llm is the primary language modeling loss, Lalign is the auxiliary alignment loss (LKL), and λ is a scalar hyperparameter controlling the influence of the auxiliary task.

A key advantage of Align-LoRA is its compatibility with various LoRA-based strategies and initialization schemes. Importantly, unlike multi-component LoRA variants, Align-LoRA introduces no additional modules that increase overhead. Consequently, its trained weights can be merged directly into the base model, incurring **zero inference latency**. This property ensures both efficiency and practicality, making Align-LoRA a lightweight yet effective solution for multi-task adaptation. For a more detailed analysis of this inference efficiency, please refer to Appendix C.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

| variants, highlighting its strong capabilities for multi-task generalization. Metrics LoRA LoRAMoE HydraLoRA R-LoRA M-LoRA A-LoRA-M A-LoRA-K Qwen2.5-7B 48.36 47.18 47.38 48.32 48.44 47.53 50.28 LLaMA3-8B 44.89 44.18 44.03 45.01 45.35 45.42 48.84 Qwen2.5-14B 52.93 50.74 51.92 52.21 53.78 52.24 55.11 Rank 10 4 4 4 4 8 8 % Param 0.25 0.38 0.25 0.25 0.22 0.20 0.20   |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

| perior performance, demonstrating its strong multi-task generalization capabilities. Schemes Task1 2 3 4 5 6 7 8 Avg   | %Par   |       |       |       |       |       |       |       |       |      |
|------------------------------------------------------------------------------------------------------------------------|--------|-------|-------|-------|-------|-------|-------|-------|-------|------|
| Qwen2.5-3B LoRA                                                                                                        | 86.31  | 56.42 | 84.65 | 72.76 | 91.37 | 87.91 | 87.60 | 44.80 | 76.48 | 0.45 |
| LoRAMoE                                                                                                                | 87.41  | 58.21 | 85.64 | 73.37 | 92.18 | 87.40 | 87.35 | 44.80 | 77.05 | 0.68 |
| HydraLoRA                                                                                                              | 86.58  | 56.42 | 85.00 | 73.36 | 92.18 | 87.33 | 88.38 | 45.15 | 76.80 | 0.45 |
| R-LoRA                                                                                                                 | 87.12  | 57.95 | 88.13 | 73.89 | 94.71 | 88.25 | 88.26 | 45.60 | 77.99 | 0.45 |
| M-LoRA                                                                                                                 | 88.02  | 57.95 | 88.87 | 74.21 | 94.71 | 88.91 | 89.07 | 46.35 | 78.51 | 0.42 |
| A-LoRA-M                                                                                                               | 87.94  | 58.03 | 88.87 | 74.12 | 94.51 | 88.85 | 88.61 | 45.88 | 78.35 | 0.42 |
| A-LoRA-K                                                                                                               | 89.25  | 59.88 | 90.35 | 75.41 | 95.33 | 89.55 | 91.95 | 48.75 | 80.06 | 0.42 |
| Qwen2.5-7B LoRA                                                                                                        | 88.41  | 60.78 | 88.42 | 81.58 | 93.52 | 91.20 | 91.79 | 48.15 | 80.48 | 0.25 |
| LoRAMoE                                                                                                                | 89.52  | 61.44 | 88.86 | 82.94 | 92.87 | 91.54 | 91.89 | 48.72 | 80.97 | 0.38 |
| HydraLoRA                                                                                                              | 88.66  | 61.23 | 89.55 | 81.72 | 93.57 | 91.67 | 91.74 | 48.70 | 80.86 | 0.25 |
| R-LoRA                                                                                                                 | 89.80  | 62.51 | 89.36 | 83.78 | 95.12 | 91.02 | 92.17 | 50.15 | 81.74 | 0.25 |
| M-LoRA                                                                                                                 | 91.35  | 62.51 | 91.98 | 84.70 | 95.93 | 91.02 | 91.97 | 50.20 | 82.46 | 0.22 |
| A-LoRA-M                                                                                                               | 90.86  | 62.45 | 91.68 | 84.59 | 95.93 | 90.74 | 91.75 | 50.45 | 82.31 | 0.20 |
| A-LoRA-K                                                                                                               | 92.23  | 64.85 | 92.89 | 85.73 | 95.93 | 93.35 | 92.93 | 53.66 | 83.95 | 0.20 |

## 5.2 Experiment

In this section, we evaluate the performance of our proposed **Align-LoRA** (abbreviated as A-LoRA) against standard LoRA and its multi-component variants. We denote our two alignment approaches with suffixes: **A-LoRA-K** for the variant using KL divergence and **A-LoRA-M** for the one using MMD. We conduct two distinct experiments to provide a comprehensive assessment of both multitask generalization and adaptation capabilities. Detailed information about both the experimental setup and the datasets used for each task is provided in the Appendix G.3. Detailed descriptions of the baseline methods are provided in Appendix J. First, to measure multi-task generalization, we fine-tuned models on the five-task dataset from Section 3 and evaluated them on the challenging, unseen tasks of the BBH benchmark. The results are presented in Table 4. Across different model families (Qwen2.5 (Qwen Team, 2024) and LLaMA3 (Grattafiori et al., 2024)) and scales, both A-LoRA-K and A-LoRA-M significantly outperform the baselines. Notably, they achieve this superior performance while using a smaller budget of trainable parameters than the sophisticated multi-component variants. This demonstrates Align-
LoRA's highly efficient use of parameters to generalize knowledge from training tasks to a different, more complex reasoning domain. Second, to validate the model's multi-task adaptation performance on in-domain tasks, we conducted experiments on a broader eight-task benchmark, evaluating each model on the corresponding test sets. As shown in the detailed results in Table 5, A-LoRA-K once again achieves the highest average 8

![8_image_0.png](8_image_0.png)

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 score across models from 3B to 7B. Impressively, it secures this top performance while utilizing fewer trainable parameters than the more complex multi-component variants. This result highlights its strong and robust adaptability across a wide range of tasks. Finally, we present several supplementary experiments to provide a comprehensive analysis of Align-LoRA-K. A sensitivity analysis on the hyperparameter λ, shown in Figure 3, reveals that our method is robust, consistently outperforming baselines across various λ values while maintaining relative stability. Furthermore, we provide several supplementary analyses in the appendix to validate and expand upon our findings. These include feature visualizations (Appendix I.1), which confirm the explicit alignment of task representations. Our analysis of training efficiency (Appendix D) demonstrates that Align-LoRA achieves the lowest FLOPs and the fastest overall training time, stemming from its use of a smaller parameter budget. Crucially, we also present comprehensive robustness checks: validating the method's efficacy when applied exclusively to Attention modules versus MLP layers (Appendix H.1); demonstrating that the alignment mechanism is universally applicable and enhances multi-head architectures (e.g., M-LoRA+Align) by better combining task-general and task-specific knowledge (Appendix I); and its sustained superiority on highly heterogeneous and complex task benchmarks (Appendix H.2). The consistent improvements from both A-LoRA-K and A-LoRA-M, demonstrated across a wide range of models, scales, and task benchmarks, provide compelling evidence for our central thesis. The fact that both the KL and MMD-based alignment strategies elevate performance above the standard LoRA baseline confirms that **explicit representation alignment is an effective strategy** for improving multi-task generalization. This success can be attributed to the alignment loss mechanism: by encouraging the representations from different tasks to map onto a shared subspace within the latent space of the down-projection matrix A, we explicitly strengthen the ability of A to learn task-general features. **This provides further, direct proof that learning task-shared** representations provides a highly effective and promising path towards multi-task learning, offering a powerful alternative to the architectural isolation of task-specific features.

## 5.3 Theoretical Analysis

To theoretically analyze the generalization performance of Align-LoRA in multi-task scenarios, we derive a novel generalization bound for MTL. Our key insight is that by explicitly aligning the representation distributions across multiple tasks, Align-LoRA can effectively reduce the distribution discrepancy among tasks. This alignment leads to a tighter generalization error bound compared to traditional multi-component LoRA variants.

Formally, let M be the number of tasks, Di (i = 1, 2*, . . . , M*) be the data distribution of task i, and Dˆi be the corresponding training dataset. Let Rtrain(f; Dˆi) denote the empirical training risk of model f on Dˆi, and ntotal =PM
i=1 |Dˆi| be the total number of training samples across all tasks. The generalization bound for Align-LoRA is given by:
486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

## References

Ahmed Agiza, Marina Neseem, and Sherief Reda. Mtlora: Low-rank adaptation approach for efficient multi-task learning. In *Proceedings of the IEEE/CVF Conference on Computer Vision and* Pattern Recognition, pp. 16196–16205, 2024.

Tom B Brown. Language models are few-shot learners. *arXiv preprint arXiv:2005.14165*, 2020.

$$R_{\rm MTL}(f)\leq\frac{1}{M}\sum_{i=1}^{M}R_{\rm train}(f;\tilde{\mathcal{D}}_{i})+\frac{\lambda}{M}\sum_{i<j}\Delta(\mathcal{D}_{i},\mathcal{D}_{j})+O\left(\sqrt{\frac{\log(1/M)}{n_{\rm total}}}\right)$$

![9_image_0.png](9_image_0.png)

 ,
where RMTL(f) = 1M
PM
i=1 E(x,y)∼Di ℓ(f(x), y) is the average expected risk over all tasks,
∆(Di, Dj ) measures the distribution discrepancy between task i and task j (e.g., using KL divergence or MK-MMD), λ > 0 is a weight parameter balancing the training risk and distribution alignment term, δ ∈ (0, 1) is the confidence parameter. The crucial advantage of Align-LoRA is its distribution alignment mechanism, which actively minimizes ∆(Di, Dj ) during training. This significant reduction in cross-task distribution discrepancy directly leads to a tighter generalization bound, as the second term in the bound is effectively controlled. For the detailed derivation of this theoretical result, including all technical assumptions and proof steps, please refer to the Appendix F.

## 6 Conclusion

In this work, we revisited multi-task generalization in LoRA, critically examining the prevailing approach of using multi-component designs to separate task-specific knowledge. Our investigation yielded two key insights that challenge this paradigm. First, we demonstrated that a simplified multi-head LoRA (**M-LoRA**) with highly redundant head matrices can outperform more complex, diversity-focused variants. Second, we showed that simply **increasing the rank** of a standard LoRA is sufficient to match the performance of these multi-component architectures. This calls their fundamental utility into question, as they fail to deliver significant performance gains over a simpler baseline while introducing additional inference latency and complexities from non-mergeable routers. Based on these findings, we proposed a new hypothesis: **learning task-shared representations** provides a highly effective and promising path towards multi-task learning, offering a powerful alternative to the architectural isolation of task-specific features.. Our hypothesis deliberately steers research toward what has been a largely unexplored direction in the LoRA framework: the active enhancement of task-shared knowledge. To formally explore this promising path and validate our hypothesis, we introduced **Align-LoRA**, a novel method that explicitly aligns representations to foster the learning of shared knowledge. Our findings, substantiated by comprehensive empirical evidence and theoretical analysis, confirm that Align-LoRA achieves superior performance, validating our hypothesis and charting a new, more efficient direction for multitask PEFT. We believe this shift in focus, which moves from separating task-specific knowledge via multi-component architectures to learning task-shared knowledge via representation alignment, is a more promising direction for future research.

## 7 Reproducibility Statement

The code for Align-LoRA is available at both the anonymous link Align-LoRA and Supplement Material. Implementation details can be found in the Appendix K. Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira. Analysis of representations for domain adaptation. *Advances in neural information processing systems*, 19, 2006.

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 7432–7439, 2020.

Yupeng Chang, Yi Chang, and Yuan Wu. Ba-lora: Bias-alleviating low-rank adaptation to mitigate catastrophic inheritance in large language models. *arXiv preprint arXiv:2408.04556*, 2024a.

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, et al. A survey on evaluation of large language models. ACM Transactions on Intelligent Systems and Technology, 15(3):1–45, 2024b.

Xilun Chen and Claire Cardie. Multinomial adversarial networks for multi-domain text classification. *arXiv preprint arXiv:1802.05694*, 2018.

Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In *NAACL*, 2019.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1, 2018.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Shihan Dou, Enyu Zhou, Yan Liu, Songyang Gao, Jun Zhao, Wei Shen, Yuhao Zhou, Zhiheng Xi, Xiao Wang, Xiaoran Fan, et al. Loramoe: Revolutionizing mixture of experts for maintaining world knowledge in language model alignment. *arXiv preprint arXiv:2312.09979*, 4(7), 2023.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. *arXiv e-prints*, pp. arXiv–2407, 2024.

Arthur Gretton, Dino Sejdinovic, Heiko Strathmann, Sivaraman Balakrishnan, Massimiliano Pontil, Kenji Fukumizu, and Bharath K Sriperumbudur. Optimal kernel choice for large-scale two-sample tests. *Advances in neural information processing systems*, 25, 2012.

Chenlu Guo, Yuan Wu, and Yi Chang. Nlora: Nystrom-initiated low-rank adaptation for large ¨
language models, 2025. URL https://arxiv.org/abs/2502.14482.

Zeyu Han, Chao Gao, Jinyang Liu, Jeff Zhang, and Sai Qian Zhang. Parameter-efficient fine-tuning for large models: A comprehensive survey. *arXiv preprint arXiv:2403.14608*, 2024.

Dou Hu, Lingwei Wei, Wei Zhou, and Songlin Hu. Impartial multi-task representation learning via variance-invariant probabilistic decoding. In Wanxiang Che, Joyce Nabende, Ekaterina Shutova, and Mohammad Taher Pilehvar (eds.), *Proceedings of the 63rd Annual Meeting of the Association* for Computational Linguistics (Volume 1: Long Papers), pp. 19883–19897, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251-0. doi: 10.18653/v1/ 2025.acl-long.975. URL https://aclanthology.org/2025.acl-long.975/.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, and Min Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition. *arXiv preprint arXiv:2307.13269*, 2023.

Solomon Kullback and Richard A Leibler. On information and sufficiency. The annals of mathematical statistics, 22(1):79–86, 1951.

Dengchun Li, Yingzi Ma, Naizheng Wang, Zhengmao Ye, Zhiyuan Cheng, Yinghao Tang, Yan Zhang, Lei Duan, Jie Zuo, Cal Yang, et al. Mixlora: Enhancing large language models finetuning with lora-based mixture of experts. *arXiv preprint arXiv:2404.15159*, 2024.

Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. *Advances in Neural Information Processing Systems*, 35:1950–1965, 2022.

Jinda Liu, Yi Chang, and Yuan Wu. R-lora: Random initialization of multi-head lora for multi-task learning, 2025. URL https://arxiv.org/abs/2502.15455.

Qidong Liu, Xian Wu, Xiangyu Zhao, Yuanshao Zhu, Derong Xu, Feng Tian, and Yefeng Zheng.

When moe meets llms: Parameter efficient fine-tuning for multi-task medical applications, 2024a. URL https://arxiv.org/abs/2310.18339.

Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-
Ting Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. arXiv preprint arXiv:2402.09353, 2024b.

Mingsheng Long, Yue Cao, Jianmin Wang, and Michael Jordan. Learning transferable features with deep adaptation networks. In *International conference on machine learning*, pp. 97–105. PMLR, 2015.

Laurens van der Maaten and Geoffrey Hinton. Visualizing data using t-sne. *Journal of machine* learning research, 9(Nov):2579–2605, 2008.

Yuren Mao, Yuhang Ge, Yijiang Fan, Wenyi Xu, Yu Mi, Zhonghao Hu, and Yunjun Gao. A survey on lora of large language models. *Frontiers of Computer Science*, 19(7):197605, 2025.

Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors adaptation of large language models. *Advances in Neural Information Processing Systems*, 37:121038–121072, 2025.

Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct electricity? a new dataset for open book question answering. In *EMNLP*, 2018.

Sinno Jialin Pan, Ivor W Tsang, James T Kwok, and Qiang Yang. Domain adaptation via transfer component analysis. *IEEE transactions on neural networks*, 22(2):199–210, 2010.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.

github.io/blog/qwen2.5/.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An adversarial winograd schema challenge at scale. *Communications of the ACM*, 64(9):99–106, 2021.

Maarten Sap, Hannah Rashkin, Derek Chen, Ronan LeBras, and Yejin Choi. Socialiqa: Commonsense reasoning about social interactions. *arXiv preprint arXiv:1904.09728*, 2019.

Dino Sejdinovic, Bharath Sriperumbudur, Arthur Gretton, and Kenji Fukumizu. Equivalence of distance-based and rkhs-based statistics in hypothesis testing. *The annals of statistics*, pp. 2263– 2291, 2013.

Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, ¨
Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny Zhou, et al. Challenging big-bench tasks and whether chain-of-thought can solve them. *arXiv preprint arXiv:2210.09261*, 2022.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. *arXiv preprint arXiv:2307.09288*, 2023.

A Vaswani. Attention is all you need. *Advances in Neural Information Processing Systems*, 2017. Alex Wang. Glue: A multi-task benchmark and analysis platform for natural language understanding. *arXiv preprint arXiv:1804.07461*, 2018.

Chunlin Tian, Zhan Shi, Zhijiang Guo, Li Li, and Chengzhong Xu. Hydralora: An asymmetric lora architecture for efficient fine-tuning, 2024. URL https://arxiv.org/abs/2404.19245.