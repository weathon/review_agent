000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Thoughtbubbles: An Unsupervised Method For Parallel Thinking In Latent Space

Anonymous authors Paper under double-blind review

## Abstract

Current approaches for scaling inference-time compute in transformers rely on training them to emit explicit chain-of-thought tokens before producing an answer. While these methods are powerful, they are limited because they cannot be applied during pretraining and are limited to only serially-generated, natural-language verbalization to scale inference-time compute. In this work, we propose Thoughtbubbles, a transformer variant that natively performs parallel adaptive computation in latent space by learning to fork or delete residual streams. Thus, tokens that require a large amount of computation can form a "bubble" of cloned residuals in the middle of the network for additional computation. Crucially, this behavior is learned during pretraining with only language modeling loss. **Thoughtbubbles** outperforms both standard decoder LMs as well as non-adaptive parallel computation approaches on OpenWebText and peS2o perplexity and in zero-shot evaluations such as HellaSwag and LAMBADA after pretraining across 150M to 772M parameter scales. The implicit nature of our method enables adaptive computation to be learned starting at pretraining time, paving the way to unify train-time and test-time scaling behaviors.

## 1 Introduction

Despite their unprecedented success, Transformers (Vaswani et al., 2017) only have a fixed computation budget and working memory, which present both a theoretical (Merrill & Sabharwal, 2023) and practical limit (Sanford et al., 2024) for solving complex, multi-step problems. Due to the growing interest in extending the capabilities of transformers for difficult multi-step problems, many efforts are underway to surpass this bounded-computation limitation of transformers. The earliest and simplest is Chain of Thought (CoT) (Wei et al., 2023), where a transformer language model is explicitly prompted to provide a set of reasoning steps. This technique allows the model to break a problem down to subproblems, solve them individually, and cache intermediate results for the full solution—enabling a simple form of problem adaptivity (Merrill & Sabharwal, 2024). Extending this result, Pfau et al. (2024) show both theoretically and practically that CoT improves the expressiveness of transformers—even when the CoT traces are replaced with a unique thinking token (dots) at test time: indicating that even residual streams alone, not serial recurrence, can improve computational performance. Such an insertion of additional residual streams, the so-called "pause tokens", has since become a growing trend of recent architecture research. Though methods vary in terms of where to actually insert the thinking tokens(Herel & Mikolov, 2024; Sun et al., 2025; Goyal et al., 2024), all pause token approaches insert additional computation streams prior to computation—limiting the model's ability to allocate intermediate computation that is useful only in some, but not all layers (e.g., computation which is useful only after a few layers of attention). As Sun et al. (2025) notes, determining the location of these pause tokens often requires manual design following the structure of the problem, which may be intractable for general language models. In response, we introduce **Thoughtbubbles**, a novel Transformer-based architecture which enables the unsupervised and dynamic allocation of additional parallel residual streams for extra computation and memory. We achieve this by introducing a novel forking mechanism between some layers, 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2 Methods

which computes and maintains a cumulative score for every residual stream and uses it to decide whether to *create* extra residuals or to *delete* existing ones. This formulation makes dynamic computation a budget-bounded allocation problem of these scores. In order to train these scores to be useful, we use these scores to mask both the model's ability to attend to residual streams with low scores as well as limit the model's ability to update them at each layer. This attenuation forces the model to provide higher scores to residual streams it deems more important, which will also result in increased forking of those streams. At the end of encoding, our model will decode each residual streams separately, including forked ones. Weighted by their scores, streams for the same token are then averaged together to produce the final distribution. Thus, our approach will essentially create "bubbles" of latent computation consisting of forked residuals for difficult tokens (i.e., those with high cumulative scores) for additional thinking, before merging them to produce the final output token. We conduct a variety of pretraining experiments across 150M to 772M scales and make the following contributions:

![1_image_0.png](1_image_0.png) 

Figure 1: Overview of our method: input tokens fork to form a bubble of latent computation (orange), which is then contracted to produce the final token. Some extraneous tokens may fork (dark blue), but then be pruned.

1. We introduce the first-known architecture to enable the unsupervised dynamic allocation of latent parallel computation, trainable as a regular decoder LM without any additional signal beyond language modeling loss.

2. We demonstrate that our approach performs better in validation perplexity as well in zeroshot evals of LAMBADA and HellaSwag against two competitive baselines—a regular parameter matched transformer, as well as a non-adaptive *computation matched* approach where the input residual is copied multiple times as filler tokens for additional computation before decoding. We additionally perform competitively against BLiMP and PIQA.

3. We further show that our method correctly allocates computation at *interpretable* regions of extra computation. In particular, our method allocates more computation at regions of higher uncertainty (i.e., posterior entropy).

We release pretrained adaptive compute LMs and a PyTorch implementation for the community.1

## 2.1 Overall Architecture

Our architecture is a GPT-2-style decoder-only transformer (Radford et al., 2019), trained using the cross-entropy language modeling objective. To achieve parallel computation, we want to allocate more residual streams corresponding to tokens that require more computation. To enable this, we propose a special type of transformer operation named "forking", described in section 2.3, which can duplicate or remove some input residual streams for future computation. The amount of forking is controlled by assigning a "cumulative score" between 0 and 1 to each residual stream. Each forking operation computes a "keep score" for each residual stream, which is multiplied to the cumulative score to update it, as well as a new "fork score" for the new residual. We describe the computation of these scores in section 2.3.

1URL will be available upon acceptance

![2_image_0.png](2_image_0.png) 

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 2.2 Notation 2.3 Forking

Residual stream insertion and deletion are performed in special forking layers inserted between our score-attenuated transformer blocks, described in section 2.4. Each "forking" layer k parametrized by θ carries a new "forking decision" function f
(k)
θ: R
dmodel −→ R
2. We apply this new function on each member of the residual stream in order to produce the fork and keep scores, which we then bottleneck using a top-k judgment in order to produce the forked output.

Scoring. For each residual, x
(k−1)
i(note that the notation here is irrespective of forks or the original token, a distinction which we make later), we first apply the forking decision function along with a sigmoid activation σ to obtain a fork and keep scores: This setup reduces the dynamic computation task to determining which residual streams to keep or delete based on the value of cumulative scores: we take the top-k of the scores and perform their corresponding (i.e., keep / fork) actions. As long as the "useful" tokens receive the highest scores, the extra computation should help the performance of the model. To train the model to use the scores correctly, attention and residual updates are attenuated by the cumulative scores (section 2.4). That is, the tokens that the model needs to attend to and update the most become implicitly the highest-scoring tokens to be duplicated.

Additionally, we take special care about the RoPE position embeddings: we apply a "partial rotation" to the forked tokens proportional to the number of forks: the more forks a token has, the "closer together" each of their forks are. This design is described in detail in appendix D. We will use x
(k)
j ∈ R
dmodel to denote the j th residual stream at the k th layer. To emphasize that a particular token is the j th fork of token i, we will write x
(k)
i,j . We fork tokens to the left of the original input token. Thus, the original token is always x
(0)
i,0. A sequence of q forks and original token can be written as hx
(k)
i,q *. . . x*
(k)
i,0 i.

Lastly, we use L to denote the input sequence length (i.e., "input block size", the embedded input to the first block is x
(0) 1,0
. . . x
(0)
l,0
), N to denote the block size at the input to each layer (i.e., before the first layer, N = L). We omit the layer index for N to avoid clutter. Additionally, we take a parameter κ for the maximum block size. This means that the maximum number of forks each layer is κ − N.

$$\sigma\Big(f_{\theta}^{(k)}\Big(x_{i}^{(k-1)}\Big)\Big)=\Big[p_{\mathrm{fork,i}}^{(k)},p_{\mathrm{keep,i}}^{(k)}\Big]\,.$$
keep,ii. (1)
$$(1)$$

3 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 To learn useful scores, in all blocks, both residual writes and attention computation are modulated by the cumulative scores. Intuitively, this prevents the model from relying on tokens that are about to be deleted due to their low scores. Specifically, we stack the cumulative scores to a vector P
(k) ∈ R
κ:

$P^{(k)}=\left[p_{\text{cum},1}^{(k)},\ldots,p_{\text{cum},\kappa}^{(k)}\right]$  **relation computation and residual updates.** We define $\mathbf{t}$
cum,κi(7)
and use it to modulate both the attention computation and residual updates. We define the attenuated attention operation as:

to calculate the residual values of $Y^{(k)}$ as follows:  $$X^{(k)^{\prime}}=\text{Ant}\Big{(}f_{G}\Big{(}\text{LN}\Big{(}X^{(k)}\Big{)}\Big{)},f_{K}\Big{(}\text{LN}\Big{(}X^{(k)}\Big{)}\Big{)},f_{V}\Big{(}\text{LN}\Big{(}X^{(k)}\Big{)}\Big{)}\Big{)}\odot P^{(k)}\mathbb{1}^{\top}+X^{(k)}$$ $$X^{(k+1)}=\text{MLP}\Big{(}\text{LN}\Big{(}X^{(k)^{\prime}}\Big{)}\Big{)}\odot P^{(k)}\mathbb{1}^{\top}+X^{(k)^{\prime}}$$

⊤ + X(k)(9)

Given a set of scores for a layer k, we create a list P =
hpˆ
(k)
fork,0, pˆ
′(k)
keep,0*. . .* pˆ
(k)
fork,n, pˆ
′(k)
keep,ni, we compute a top-k to downsample this list to obtain Pκ where |Pκ| = κ. Using this list, we assemble the new residual stream set X(k) by the following two rules:

$x_{j}^{(k)}\in X^{(k)}$ if $\hat{p}_{\rm{leep},j}^{\prime}\in P_{\kappa}$  $x_{j_{\rm{tot}}}^{(k)}\in X^{(k)}$ if $\hat{p}_{\rm{tot},j}\in P_{\kappa}$
$$({\mathfrak{H}})$$
$$(6)$$
In order to differentiate the forks from their sources, a per-layer learned fork embedding v
′(k)
θ ∈
R

dmodel is added to their parent at initialization: x
(k)
jfork = x
(k)
j + v
′(k)
θ. We arrange the output tokens such that if a new forked residual is created, it is placed to the *left* of its parent. We define the new cumulative scores p
(k)
cum as pˆfork,j for newly forked residuals, and pˆkeep,j for kept residuals (note that this is score for which the rightmost token does not have forced-maximum score of 1, allowing the model to ignore the rightmost token if desired.)

## 2.4 Residual Update Attenuation

Forking Judgments. To make sure we have a source token from which to predict each next token, we must ensure that at least one instance is kept throughout the whole model. To do so, we first define a modified keep score that is forced to be 1 (the maximum) for the original, rightmost tokens:

$\hat{P}_{\text{keep},(k,j)}^{\prime}=\begin{cases}1\text{if}j=0\\ \hat{P}_{\text{keep},(k,j)}\text{otherwise}\end{cases}$
$$(4)$$
$$(T)$$
$$(8)$$
(9)  $\binom{10}{10}$  . 
We then update the fork and keep scores inductively based on a "cumulative score" (pcum) propagated from previous layers:

$$\hat{p}^{(k)}_{\rm fork,\,i}=p^{(k-1)}_{\rm cum,\,i}\cdot p^{(k)}_{\rm fork,\,i}$$ $$\hat{p}^{(k)}_{\rm keep,\,i}=p^{(k-1)}_{\rm cum,\,i}\cdot p^{(k)}_{\rm keep,\,i}$$
$$(2)$$
$$({\mathfrak{I}})$$

We initialize the cumulative scores for each input token at the first layer as p
(0)
cum,(i,0) = 1. A subset of these pˆ
′keep, pˆfork scores is used as p
(k+1)
cum , after deciding which ones to keep, as described later.

Note that, in practice, all scores (keep, fork, cumulative) are implemented in log-space for stability instead of being in probability space as shown here.

$$\mathrm{{Atm}}\!\left(Q^{(k)},K^{(k)},V^{(k)}\right)=\mathrm{{softmax}}\!\left(\frac{Q^{(k)}K^{(k)\top}+\mathbb{1}\log\left(P^{(k)}\right)^{\top}}{\sqrt{d_{\mathrm{{model}}}}}\right)\!\left(V^{(k)}\odot P^{(k)}\right)$$
(k)(8)
for X(k) being the cocatenated list of residual streams in the input of the layer, LN being layernorm, and f*Q,K,V* being the attention projections. If forking occurred prior to this layer, X(k)is as defined in eqs. 5 and eq. (6) , *after* forking takes place. where ⊙ is the element-wise multiplication. We modify the transformer block (Vaswani et al., 2017) to attentuate the residual whites by P
(k)as follows:
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 2.5 Output Averaging

After all transformer layers, we obtain a residual stream set where an input token might be represented by multiple residual streams. To compute a single output distribution for these distributions, we decode each of the residual streams and mix the resulting probability distributions using the cumulative scores. For Decθ : R
d model *−→ |*V | being the vocabulary output projection, and f being the last layer of the network, we have:

$$x_{i}^{(k)}=\frac{1}{\sum_{j}p_{\mathrm{cum},(i,j)}^{(f)}}\sum_{j}p_{\mathrm{cum},(i,j)}^{(f)}\mathrm{softmax}\left(\mathrm{Dec}_{\theta}\;\left(x_{i,j}^{(k)}\right)\right).$$
$$(11)$$
. (11)
We compute this weighted average using the log-sum-exp trick (Blanchard et al., 2021) for stability.

## 2.6 Scoring And Sampling

Because of the possibility of varying κ at inference time, there are two main ways inference can be performed in our model. Naively, we can set the inference budget κinference to be the same as in training time κtrain, two or four times the block size atr training. We call this **fixed forking**. Alternatively, we can set the inference budget to be the same *ratio* as the training budget. κinference is set to a value that maintains its same ratio to block size as during training; that, if κtrain = 2ltrain, then κsample = 2lsample. We call this **dynamic forking**, and discuss this method further in appendix E.1.

Scoring To obtain a probability judgment from our model of a sequence, we provide the entire sequence as input to our model and obtain the posterior probabilities our model assigns to each token of our sequence. For all of our results in table 1, we use dynamic forking. Sampling We perform autoregression with both fixed and dynamic forking, and discuss the tradeoffs of both, in section 5.1. Note that dynamic forking is especially pertinent here because initial sequneces for autoregression is small.

## 3 Experimental Setup 3.1 Parameter Selection And Training

Because our architecture takes token embeddings as input and produces token probabilities, it trains exactly like a regular decoder-only language model. As mentioned above, this means that the loss function can be standard language-modeling cross-entropy loss. Optimization is performed by the AdamW optimizer (Loshchilov & Hutter, 2017). Further optimization and architecture details can be found in appendix A.

We insert the first forking layer after a few regular transformer blocks to ensure that the forking score judgments see a broader context window. This is important in order to judge a token's relative importance compared to the others. For all models in section 4, we train models at various scales with token forking placed prior to layers 3, 7, and 11. This means that for models with more layers, the majority of the latter half of transformer will contain no forking. We discuss this choice in appendix B.

## 3.2 Pretraining Datasets

We pretrain our approach on two datasets: OpenWebText (Gokaslan et al., 2019), a standard web-text pretraining corpus, as well as peS2o (Soldaini & Lo, 2023), a collection of academic papers sourced from the Semantic Scholar Open Research corpus (Lo et al., 2020). Pretraining is conducted for 2.5 billion tokens (75,000 steps). We sample batches at random positions throughout each dataset without additional masking.

## 3.3 Baselines

Regular Transformer. We first compare against a GPT-2-like (Radford et al., 2019) transformer with RoPE (Su et al., 2024). Our model is based on nanoGPT2. We make no changes other than removing the learned position embeddings and including rotational ones in the attention pass.

2https://github.com/karpathy/nanoGPT

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

Duplicated Filler Tokens. Though a regular transformer is a parameter-matched baseline, our approach will necessarily utilize more computation due to the expanded latent block size (i.e., after forking, the block-size is longer). A naive model of parallel computation that would allow us to slightly exceed the computation of our approach is by simply copying the input residual multiple times before running the transformer, and then taking the rightmost residual for decoding.

## 3.4 Pretraining Evaluations

After pretraining, we conduct a variety of zero-shot evaluations on our models and baselines to examine their quality. They include the model's measured perplexity on a holdout validation set, LAMBADA (Paperno et al., 2016) for context extraction, HellaSwag (Zellers et al., 2019) for common sense reasoning, BLiMP (Warstadt et al., 2020) for syntax understanding, and PIQA (Bisk et al., 2020) for embodied physical inference. For each zero-shot downstream task, we use the dynamic budget as described in appendix E.1. We describe in detail the implementation of the zero-shot evaluations, including what to measure, in appendix E. Across all evaluations, we use the trained models as-is without additional fine-tuning.

## 4 Results

| Dataset       | Size     | Approach   | Perplexity (↓)   | LAMBADA (↑)   | HellaSwag (↑)   | BLiMP (↑)   | PIQA (↑)   |
|---------------|----------|------------|------------------|---------------|-----------------|-------------|------------|
| Baseline      | 21.22    | 23.9       | 30.6             | 79.6          | 62.3            |             |            |
| Copy-3        | 21.20    | 22.8       | 29.0             | 81.2          | 60.4            |             |            |
| 772M          | Copy-5   | 20.90      | 19.9             | 29.1          | 80.9            | 60.2        |            |
| Ours (κ = 2L) | 20.19    | 27.9       | 31.1             | 80.4          | 62.0            |             |            |
| Ours (κ = 4L) | 19.74    | 29.4       | 32.25            | 81.6          | 61.9            |             |            |
| Baseline      | 21.56    | 22.1       | 28.7             | 79.0          | 60.5            |             |            |
| Copy-3        | 21.51    | 21.9       | 28.6             | 80.5          | 60.1            |             |            |
| 319M          | Copy-5   | 21.28      | 21.1             | 28.4          | 79.6            | 60.5        |            |
| Ours (κ = 2L) | 20.55    | 22.9       | 29.3             | 78.3          | 60.9            |             |            |
| Ours (κ = 4L) | 20.23    | 23.2       | 29.0             | 78.8          | 60.1            |             |            |
| OpenWebText   | Baseline | 24.51      | 18.2             | 26.9          | 76.7            | 57.9        |            |
| Copy-3        | 24.44    | 17.6       | 27.1             | 79.3          | 58.9            |             |            |
| 150M          | Copy-5   | 24.40      | 18.9             | 26.9          | 78.8            | 59.4        |            |
| Ours (κ = 2L) | 23.78    | 21.1       | 27.3             | 77.5          | 59.0            |             |            |
| Ours (κ = 2L) | 23.19    | 25.5       | 27.7             | 78.1          | 60.6            |             |            |
| Baseline      | 14.64    | 9.9        | 27.3             | 69.8          | 55.4            |             |            |
| Copy-3        | 14.37    | 9.5        | 27.2             | 73.3          | 55.3            |             |            |
| 772M          | Copy-5   | 14.50      | 10.3             | 27.3          | 71.6            | 54.5        |            |
| Ours (κ = 2L) | 13.98    | 10.5       | 27.4             | 68.4          | 56.3            |             |            |
| Ours (κ = 4L) | 13.77    | 12.9       | 27.6             | 67.4          | 54.6            |             |            |
| Baseline      | 16.61    | 9.3        | 26.4             | 68.4          | 55.3            |             |            |
| Copy-3        | 16.41    | 9.4        | 27.2             | 71.8          | 54.7            |             |            |
| 319M          | Copy-5   | 16.16      | 8.5              | 26.6          | 70.1            | 55.1        |            |
| Ours (κ = 2L) | 15.84    | 10.5       | 26.5             | 67.0          | 53.8            |             |            |
| Ours (κ = 4L) | 15.61    | 12.3       | 27.2             | 68.6          | 53.6            |             |            |
| peS2o         | Baseline | 17.10      | 8.1              | 26.4          | 68.6            | 54.5        |            |
| Copy-3        | 16.95    | 7.1        | 26.3             | 69.6          | 54.1            |             |            |
| 150M          | Copy-5   | 16.90      | 7.2              | 26.0          | 69.3            | 54.0        |            |
| Ours (κ = 2L) | 16.90    | 5.0        | 26.2             | 66.6          | 55.1            |             |            |
| Ours (κ = 4L) | 16.42    | 10.3       | 26.9             | 67.9          | 55.1            |             |            |

Table 1: Zero-shot evaluation results across all model scales after pretraining on 2.5 billion tokens. Each setting is parameter-matched; baseline is a standard GPT-2-like model; copy-3 and copy-5 are models where the input residuals are copied multiple times and can attend to each other; ours is the thoughtbubbles transformer, with forking budget set to 2 (κ = 2L) and 4 (κ = 4L) times the input block size. The latter of which is roughly FLOPs-matched against copy-5 baseline. Our approach performs the best against all baselines in validation perplexity, even exceeding models of bigger scale. Across both parameter and computation matched settings, we find that our model scores the lowest perplexity across all evaluations. Figure 3 highlights the scalability of our approach: surprisingly, our approach at a 319M parameter scale has lower perplexity on OpenWebText than the baseline approach at the 772M scale.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_0.png](6_image_0.png)

## 5 Analysis

Forks meaningfully influence the value of the parent token. In fig. 4, we see the rightmost ("og") token attends to its children with attention scores more than an order of magnitude higher than other tokens—second only to attention of those tokens to themselves. This result indicate that the forking tokens play a large role in the computation of the residual update for the rightmost token than most other tokens, indicating their utility in computing the final output.

![6_image_1.png](6_image_1.png) 
Our model dynamically allocates more computation at regions of higher uncertainty without explicit supervision . . . Despite no explicit interventions or regularization, our method learned to allocate more computation at areas of greater uncertainty. We see in fig. 5 that our method allocates more computation tokens with high output distribution entropy; this is true both for the entropy measured from the forking model as well as an independently trained, parameter matched decoder LM that does not fork. This is in-line with recent literature (Wang et al., 2025) that highlights the informativeness of high entropy tokens.

Figure 4: Analysis of attention allocation between the main (rightmost, "og") token and its child forks on our approach trained on openwebtext. Note that since we place child token embeddings to the to the left of the main token, forked children cannot attend to its parent.

. . . but will reduce computation at areas of greatest uncertainty. Despite the previous point, however, we note that our model allocates relatively less budget at tokens of the highest uncertainty, forming a concave parabolic relationship between entropy and computation allocation. We hypothesize that this is due to the relatively higher utility of further computation at areas of moderate (but not Across most zero-shot evaluations, our approach outperforms baselines. For all LAMBADA and HellaSwag evaluations, we find that our approach outperforms both the parameter-matched baselines as well as the computation-matched baselines. However, we note that for BLiMP (syntax understanding) our model only outperforms the parameter-matched, but not computation-matched baselines—indicating that pruned dynamic parallel computation may not be as helpful for syntax matches. Finally, our model performs similarly to the baseline for embodied reasoning. We attribute this degraded performance to the fact that a short (2.5BT) training may not capture enough information for the embodied NLI to be effectively measured.

![7_image_0.png](7_image_0.png)

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Figure 5: Normalized number of forks in the final layer across a window of 4 tokens as a function of the mean entropy of those 4 tokens on OpenWebText . Left: entropy as measured by the forking transformer; right: entropy as measured by a baseline decoder LM. low) uncertainty: for instance, while choosing between a few options; conversely, areas of highest uncertainty are often caused by the edges of clauses or coreferences, where additional computation will not help resolve the uncertainty.

## 5.1 Autoregression

![7_image_1.png](7_image_1.png)

| Approach              | Perplexity   |
|-----------------------|--------------|
| Ours (Blockwise)      | 20.97        |
| Ours (Fixed Budget)   | 23.10        |
| Ours (Dynamic Budget) | 21.18        |
| Baseline              | 22.15        |

1 Figure 6: Perplexity distribution and mean perplexity of our 772M (κ = 2L) model over smaller subset of OpenWebText dev set between blockwise forward versus autoregression. Left: naive autoregression; right: autoregression with forking budget proportional to input size. Lower is better.

As seen in fig. 6, implementing autoregression naively with a fixed block size irrespective of the input sequence length results in a distribution shift between blockwise forward pass and autoregression—since the maximum allowed number of forks is much higher if input sequence length is smaller while the total budget remains the same. However, if we apply the forking budget scaling mitigation described in appendix E.1, we find that our model performs roughly equivalently to the blockwise forward pass, and retains our approach's performance gains above baseline. This result indicates that, while our result can adapt to different inference-time input sizes, care must be taken to scale the adaptive computation budget accordingly.

## 6 Related Work

Chain-of-Thought Approaches Chain-of-thought (Wei et al., 2023) is a simple form of adaptive computation which uses natural-language-based autoregression with additional tokens to achieve thinking. Variants of this approach include simply supervising the output chain (Zhang & Ding, 2024), to replacing them with continuous traces (Hao et al., 2024) or controlled non-adaptive filler tokens (Pfau et al., 2024). Unlike chain-of-thought, our method performs adaptive computation not with recurrence but parallel computation, improving efficiency as well as being able to be trained without additional supervision. Adaptive Computation Methods vary to force a dynamic amount of computation from a neural model based on the problem. The oldest approaches involves explicitly forcing recurrence (Graves, 2016), while modern LMs yield performance improvements through forcing very simple interventions to existing chain of thoughts (Muennighoff et al., 2025), skipping or adding recurrent compute across layers without adding additional streams of computation (Dehghani et al., 2019; Csordas´
432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 et al., 2024; Chen et al., 2024; Murty et al., 2023; Raposo et al., 2024; Kallini et al., 2024), or—most similar to our approach—by adding additional residual streams when computation is needed (Herel & Mikolov, 2024; Goyal et al., 2024; Sun et al., 2025). Unlike prior art, our method removes the need to insert latent tokens explicitly during training or inference, but still gives the ability to gain additional streams of computation through latent residual streams. Analysis of Latent Computation There's a large and robust literature on the complexity-theoretic power of transformers. Results have shown the limited expressive power of standard transformer computation (Merrill & Sabharwal, 2023), and the additional power that chain-of-thought or even padding tokens add to the computation (Merrill & Sabharwal, 2025; London & Kanade, 2025). Work has also shown the limitations given by single special-token thinking approaches that are not input adaptive (Vennam et al., 2024). Prior work have also shown through techniques in intepretability that even simple chain of thought computation carries implicit intermediate computation similar to depth-bounded recurrence (Brinkmann et al., 2024). We also demonstrate here the power of adaptive latent computation in our work by demonstrating its superior performance even against computation matched baselines; furthermore, we demonstrate that we are indeed performing additional computation in "decisive" high entropy tokens, in line with prior analyses (Wang et al., 2025).

## 7 Conclusion

In this work we introduce **thoughtbubbles**, the first adaptive parallel computation architecture that's 1) trainable without additional supervision beyond LM loss 2) allocates computation and memory at interpretable regions of uncertainty and 3) performs better than baseline models in both perplexity and across a suite of zero-shot evals on both parameter-matched and computation-matched settings. This method unlocks the previously missing input-adaptivity of transformer computation, which allows our model to solve more difficult tasks that require scaling inference-time computation. We demonstrate the efficacy of our method via a suite of zero-shot evaluations on models pretrained on both OpenWebText and peS2o in both computation and parameter matched settings. Excitingly, our method at a smaller 319M scale outperformed baselines at 772M scale. Most importantly, our model enables learning latent adaptive computation in a language model already during the pre-training phase. Unlike CoT approaches, it does not rely on being exposed to step-wise instructions during pre-training. We hope that this will unlock a new generation of transformer architectures with more general latent computation, which in turn enables more helpful and capable adaptive compute models.

## 8 Limitations

Time-matched evaluations Our current approach is implemented in raw PyTorch, requiring no other hardware-level adaptations apart from scatter-max kernels.3 As such, though our implementation exceeds the performance of a computation-matched approach, its raw wall-clock efficiency is relatively low. Further efforts in implementing hardware-adaptive kernels for operations like forking gather will enable faster computational performance.

Top-K Gradient Bottleneck While forking in earlier layers improves the performance of our models, we note that, interestingly, too much forking results in no further performance improvement at a fixed block size budget (appendix B). We believe this is due to certain tokens with high cumulative scores early on in the model being dropped by hard top-k decisions later in the model, thus resulting in no gradients to update the early large cumulative scores. By implementing training time randomization and noise, this can be mitigated to improve deep forking performance. Downstream Reasoning Tasks Given our hardware limitations, we cannot experiment with approaches at sufficient scale to enable non-noisy measurements on hard reasoning datasets such as GSM8k (Cobbe et al., 2021), which—without customized training regimes—usually emerges with good performance only around the multi-billion-parameter scale (Liu et al., 2023). In future work, we hope to perform these evaluations with additional resources.

3https://github.com/rusty1s/pytorch_scatter

## Use Of Large Language Models

We use LLMs for copyediting only. All ideas and content contained herein are our own.

## References

Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical commonsense in natural language. In *Proceedings of the AAAI conference on artificial intelligence*, volume 34, pp. 7432–7439, 2020.

Pierre Blanchard, Desmond J Higham, and Nicholas J Higham. Accurately computing the log-sumexp and softmax functions. *IMA Journal of Numerical Analysis*, 41(4):2311–2330, 2021.

Jannik Brinkmann, Abhay Sheshadri, Victor Levoso, Paul Swoboda, and Christian Bartelt. A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task, June 2024.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, and Łukasz Kaiser. Universal Transformers, March 2019.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. Think Before You Speak: Training Language Models with Pause Tokens. 2024.

David Herel and Tomas Mikolov. Thinking Tokens for Language Modeling, May 2024. Bingbin Liu, Sebastien Bubeck, Ronen Eldan, Janardhan Kulkarni, Yuanzhi Li, Anh Nguyen, Rachel Ward, and Yi Zhang. Tinygsm: achieving¿ 80% on gsm8k with small language models. *arXiv* preprint arXiv:2312.09241, 2023.

Qian Chen, Wen Wang, Qinglin Zhang, Siqi Zheng, Shiliang Zhang, Chong Deng, Hai Yu, Jiaqing Liu, Yukun Ma, and Chong Zhang. Skip-layer attention: Bridging abstract and detailed dependencies in transformers. *arXiv preprint arXiv:2406.11274*, 2024.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems. *arXiv preprint arXiv:2110.14168*, 2021.

Robert Csord ´ as, Kazuki Irie, J ´ urgen Schmidhuber, Christopher Potts, and Christopher D. Man- ¨
ning. Moeut: Mixture-of-experts universal transformers. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 28589–28614. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/ file/321387ba926b8e58d3591c0aeb52ffc2-Paper-Conference.pdf.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory- ´
efficient exact attention with io-awareness. *Advances in neural information processing systems*, 35:16344–16359, 2022.

Aaron Gokaslan, Vanya Cohen, Ellie Pavlick, and Stefanie Tellex. Openwebtext corpus. http:
//Skylion007.github.io/OpenWebTextCorpus, 2019.

Alex Graves. Adaptive computation time for recurrent neural networks. *arXiv preprint* arXiv:1603.08983, 2016.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian.

Training Large Language Models to Reason in a Continuous Latent Space, December 2024.

Julie Kallini, Shikhar Murty, Christopher D Manning, Christopher Potts, and Robert Csord ´ as. ´
Mrt5: Dynamic token merging for efficient byte-level language models. arXiv preprint arXiv:2410.20771, 2024.

Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel Weld. S2ORC: The semantic scholar open research corpus. In Dan Jurafsky, Joyce Chai, Natalie Schluter, and Joel Tetreault (eds.), *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics*, pp. 4969–4983, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/ 2020.acl-main.447. URL https://aclanthology.org/2020.acl-main.447/.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Charles London and Varun Kanade. Pause Tokens Strictly Increase the Expressivity of Constant-
Depth Transformers, May 2025.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. *Transactions of the Association for Computational Linguistics*, 11:531–545, 2023. doi:
10.1162/tacl a 00562. URL https://aclanthology.org/2023.tacl-1.31/.

William Merrill and Ashish Sabharwal. The expressive power of transformers with chain of thought.

In *The Twelfth International Conference on Learning Representations*, 2024.

William Merrill and Ashish Sabharwal. Exact Expressive Power of Transformers with Padding, May 2025.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candes, and Tatsunori Hashimoto. S1: Simple test-time ` scaling, March 2025.

Shikhar Murty, Pratyusha Sharma, Jacob Andreas, and Christopher Manning. Pushdown layers:
Encoding recursive structure in transformer language models. In Houda Bouamor, Juan Pino, and Kalika Bali (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3233–3247, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.195. URL https://aclanthology.org/ 2023.emnlp-main.195/.

Denis Paperno, German Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, ´
Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. The LAMBADA dataset: ´ Word prediction requiring a broad discourse context. In Katrin Erk and Noah A. Smith (eds.), Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1525–1534, Berlin, Germany, August 2016. Association for Computational Linguistics. doi: 10.18653/v1/P16-1144. URL https://aclanthology.org/ P16-1144/.

Jacob Pfau, William Merrill, and Samuel R. Bowman. Let's Think Dot by Dot: Hidden Computation in Transformer Language Models, April 2024.

Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. 2019.

David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-depths: Dynamically allocating compute in transformer-based language models. *arXiv preprint arXiv:2404.02258*, 2024.

Clayton Sanford, Bahare Fatemi, Ethan Hall, Anton Tsitsulin, Mehran Kazemi, Jonathan Halcrow, Bryan Perozzi, and Vahab Mirrokni. Understanding transformer reasoning capabilities via graph algorithms. *Advances in Neural Information Processing Systems*, 37:78320–78370, 2024.

Koustuv Sinha, Shagun Sodhani, Jin Dong, Joelle Pineau, and William L. Hamilton. CLUTRR: A
diagnostic benchmark for inductive reasoning from text. In Kentaro Inui, Jing Jiang, Vincent Ng, and Xiaojun Wan (eds.), *Proceedings of the 2019 Conference on Empirical Methods in Natural* Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 4506–4515, Hong Kong, China, November 2019. Association for Computational Linguistics. doi: 10.18653/v1/D19-1458. URL https://aclanthology. org/D19-1458/.

Luca Soldaini and Kyle Lo. peS2o (Pretraining Efficiently on S2ORC) Dataset. Technical report, Allen Institute for AI, 2023. ODC-By, https://github.com/allenai/pes2o.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.,
2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

Sreeram Vennam, David Valente, David Herel, and Ponnurangam Kumaraguru. Rethinking thinking tokens: Understanding why they underperform in practice. *arXiv preprint arXiv:2411.11371*,
2024.

Shenzhi Wang, Le Yu, Chang Gao, Chujie Zheng, Shixuan Liu, Rui Lu, Kai Dang, Xionghui Chen, Jianxin Yang, Zhenru Zhang, et al. Beyond the 80/20 rule: High-entropy minority tokens drive effective reinforcement learning for llm reasoning. *arXiv preprint arXiv:2506.01939*, 2025.

Alex Warstadt, Alicia Parrish, Haokun Liu, Anhad Mohananey, Wei Peng, Sheng-Fu Wang, and Samuel R Bowman. Blimp: The benchmark of linguistic minimal pairs for english. Transactions of the Association for Computational Linguistics, 8:377–392, 2020.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models, January 2023.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. HellaSwag: Can a machine really finish your sentence? In Anna Korhonen, David Traum, and Llu´ıs Marquez ` (eds.), *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics*, pp. 4791–4800, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.

18653/v1/P19-1472. URL https://aclanthology.org/P19-1472/.

Xiang Zhang and Dujian Ding. Supervised Chain of Thought, October 2024. Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. *Neurocomputing*, 568:127063, 2024.

Yuchang Sun, Yanxi Chen, Yaliang Li, and Bolin Ding. Enhancing latent computation in transformers with latent tokens. *arXiv preprint arXiv:2505.12629*, 2025.