000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Chain Of Thought In Order: Discovering Learning-Friendly Orders For Arithmetic

Anonymous authors Paper under double-blind review

## Abstract

The chain of thought, i.e., step-by-step reasoning, is one of the fundamental mechanisms of Transformers. While the design of intermediate reasoning steps has been extensively studied and shown to critically influence performance, the ordering of these steps has received little attention, despite its significant effect on the difficulty of reasoning. This study addresses a novel task of unraveling the chain of thought—reordering decoder input tokens into a learning-friendly sequence for Transformers, for learning arithmetic tasks. The proposed pipeline first trains a Transformer on a mixture of target sequences arranged in different orders and then identifies benign orders as those with fast loss drops in the early stage. As the search space grows factorially in sequence length, we propose a two-stage hierarchical approach for inter- and intra-block reordering. Experiments on four order-sensitive arithmetic tasks show that our method identifies a learning-friendly order out of a few billion candidates. Notably, on the multiplication task, it recovered the reverse-digit order reported in prior studies.

## 1 Introduction

Autoregressive generation is central to the success of the Transformer (Vaswani et al., 2017) in reasoning tasks, which leads to many successes of the end-to-end learning of arithmetic and hard symbolic computations, such as (Lample & Charton, 2020; Charton, 2022; Kera et al., 2024; 2025; Alfarano et al., 2024; Wenger et al., 2022; Li et al., 2023a;b). The autoregressive nature makes each reasoning step conditioned on the preceding context, and careful design of intermediate reasoning steps, such as *chain of thought* (Wei et al., 2022), guides the model's reasoning toward the final answer of the target problem. For example, it has been known that learning the parity function—the prediction of the parity of the input bit string—is challenging (Shalev-Shwartz et al., 2017; Hahn & Rofin, 2024). However, Kim & Suzuki (2025) recently has shown that the step-by-step prediction of the parity of the first k bits with k = 1, 2*, . . .*, makes the learning successful. One important yet underexplored aspect is the order of the chain of thought—not only which steps to include, but also the order in which they are arranged can greatly impact learning. For example, Shen et al. (2023) has shown that Transformers learn multiplication of two integers with better generalization to larger integers (i.e., those with more digits) when the product is predicted from least to most significant digits (cf. Figure 1). While this particular case can be explained by the carries, which flow from least to most significant digits, a systematic way of determining a learningfriendly order of the chain of thought remains unknown. In this study, we address a new task of reordering decoder input tokens into a learning-friendly order for better learning of arithmetic tasks. Exploiting the observation that neural networks tend to learn from easy to hard instances during training (Arpit et al., 2017; Forouzesh & Thiran, 2024; Swayamdipta et al., 2020), we train a Transformer on a mixture of target sequences in different orders and identify those that lead to a faster loss drop in the early stages of training. To better handle longer sequences, we propose a two-stage hierarchical approach, where the global stage finds block-level orders, while the local stage reorders tokens within each block. The experiments demonstrate that the proposed method successfully reorders the target sequences. We designed three arithmetic tasks that are relatively easy to compute with the (input and) target 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 sequence in the forward order, but not with other orders. Starting from random orders, the proposed method succeeds up to thirteen tokens (i.e., 13! > 6×109 permutations), increasing the success rate of arithmetic computation from approximately 10 % to 100 %. We also applied our method to the multiplication task in (Shen et al., 2023) and successfully rediscovered the reverse orders. Our contributions are summarized as follows:
- We address a novel task, unraveling the chain of thought. This aims at discovering a learning-friendly order of decoder input tokens, thereby making the learning more successful for in-distribution samples and generalizable to out-of-distribution samples.

- We propose a method that efficiently determines learning-friendly orders from the loss profile at the early stage of training. Empirically, this filters a few thousand candidates in a single epoch, and combined with a hierarchical strategy, the best order can be found out of a few billion candidates.

- We introduce order-sensitive arithmetic tasks using non-injective maps, with which one can evaluate reordering methods. Our extensive experiments present that the proposed method successfully discovers learning-friendly orders and rediscover the previously reported the learning-friendly order in the multiplication task.

## 2 Related Work

Transformers for mathematical tasks. Transformers have recently been applied to mathematical problem-solving with encouraging results. (Lample & Charton, 2020) has demonstrated that a Transformer can carry out integral calculus with a high success rate, opening the possibility that sequence-to-sequence models can handle algebraic tasks. Since that study, applications have expanded to arithmetic (Charton, 2022), linear algebra (Charton, 2022), computational algebra
(Kera et al., 2024; 2025), and coding theory, as well as cryptography (Wenger et al., 2022; Li et al., 2023a;b). One reason behind these successes is the autoregressive generation scheme. Although theory has suggested that learning high-sensitive functions such as parity is difficult (Hahn & Rofin, 2024), recent work achieved a high success rate on parity tasks by applying a chain of thought prompting (Wei et al., 2022; Kojima et al., 2022; Chen et al., 2023; Yao et al., 2023; Zhang et al., 2024) to arithmetic and by exploiting the generated output tokens effectively (Kim & Suzuki, 2025). Positional encoding is also crucial for arithmetic problems; prior work (Jelassi et al., 2023) has shown that relative-position and abacus-style embeddings improve generalization to out-of-distribution data. These studies collectively show that task-specific representations and positional encodings strongly influence performance. In particular, prior work (Shen et al., 2023) analyzed in detail how digit order affects multiplication success rate and demonstrated that generating digits from the least significant position upward raises the success rate; however, the ordering was chosen heuristically rather than by an automated procedure. Systematic optimization of the output order itself in arithmetic tasks remains unaddressed. This study is the first to exploratively optimize the output-sequence permutation for each task, automatically discovering a learning-friendly target order. Easy-to-hard learning dynamics in deep neural networks. The observation that deep neural networks can be trained even on randomly assigned labels—while still achieving excellent generalization on real data—led to a line of research into how models adapt to data during training (Zhang et al., 2017). (Arpit et al., 2017) has experimentally shown that networks first pick up simple regularities between inputs and labels and only later transition to memorizing harder, noise-like examples. More broadly, deep neural networks are known to learn easy instances in a dataset before gradually fitting the more difficult ones; in image domains, this behavior is often referred to as spectral bias (Rahaman et al., 2019). This property is now widely exploited in curriculum learning (Jiang et al., 2018; Han et al., 2018; Baldock et al., 2021) and data-quality control (Swayamdipta et al.,
2020). For example, integrating each sample's learning curve can reveal mislabeled data (Forouzesh & Thiran, 2024). Most prior work, however, analyzes such dynamics by injecting noise directly into the target labels themselves. In contrast, this study focuses on the ordering of the target sequences. The dataset is rearranged with multiple permutation matrices, and the model is trained on these reordered versions to investigate how sequence order affects learning.

![2_image_0.png](2_image_0.png)

Figure 1: Success rates for the multiplication of two integers. Matrix rows and columns indicate the number of digits in each operand. Evaluation is conducted with 100 samples for each digit position. (a) The model is trained to output from the most significant digit. (b) The model is trained to output from the least significant digit.

## 3 Unraveling The Chain Of Thought

![2_image_1.png](2_image_1.png)

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

$$\theta_{\mathrm{ERM}}^{\pi}=\operatorname*{arg\,min}_{\theta}\;\frac{1}{m}\sum_{i=1}^{m}\ell({\cal T}_{\theta},X_{i},\pi(Y_{i})),$$
$$(3.1)$$
$$(3.2)$$

with ℓ denotes a loss function. Our goal is to discover a permutation π that minimizes the expected risk:

$$\pi^{*}=\arg\operatorname*{min}_{\pi\in S_{L}}\mathbb{E}_{(X,Y)\sim{\mathcal{D}}}\left[\ell\big({\mathcal{T}}_{\theta_{\mathrm{ERM}}^{\pi}},X,\pi(Y)\big)\right].$$
, X, π(Y ). (3.2)
A permutation π(Y ) of a target sequence Y = [y1*, . . . , y*L] ∈ Σ
L can be represented as a matrix product Y P, where P ∈ {0, 1}
L×L is a permutation matrix.

Challenges. The optimization over permutations is challenging because one has to test all possible permutations, which is L! for those over {1*, ..., L*}. One may introduce a soft permutation matrix Let SL be the symmetric group of order L, i.e., the set of all permutations over {1*, . . . , L*}. We address the problem of discovering a permutation π ∈ SL over the token sequence (of length L) to the Transformer decoder that improves the overall learning effectiveness of the Transformer. The Transformer decoder generates output sequences in an auto-regressive manner. It is widely known—especially in the context of chain-of-thought prompting—that the order of generation can have a crucial impact on the reasoning ability of Transformers. For example, Figure 1 shows that, in the task of multiplying two integers, the digits of the target integer (each treated as a token) should be presented in reverse order—from lower to higher digits—because this allows the Transformer to compute carries step by step.

More generally, for example, let X = [x1*, . . . , x*L] be a sequence of numbers, which is the input sequence to the Transformer. If the target sequence is defined by a map f(*x, y*) that is non-injective with respect to y (e.g., f(*x, y*) = max{x + y, 0}) as Y = [y1*, . . . , y*L] with y1 = x1 and yi+1 =
f(xi + yi) for i > 1, learning from reverse order Y
r = [yL*, . . . , y*1] is significantly harder than that from the forward order because of non-injective f(*x, y*). We now introduce our formal problem setup and its challenges. Formulation. Let Σ be the set of all tokens. We denote the set of all finite token sequences by Σ∗ and its restriction to length-L sequences by Σ
L. Let Tθ : Σ∗ × Σ
L → Σ
L be a Transformer with parameter θ. Hereinafter, we assume that the target sequence length is fixed. Now, let (X, Y ) ∼ D be an input–target sequence pair (*X, Y* ) with |Y | = L from a joint distribution D. The empirical risk minimizer θERM with finite sample set D = {(Xi, Yi)}
m i=1 and permutation π ∈ SL is 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

## 4 Proposed Method

However, as shown in Figure 2, such an approach leads to an immediate loss drop at the early stage of training, because the soft permutation P˜ causes information leakage from future tokens; each token in YiP is a soft mixture of all the tokens in Y , which undermines the next-token prediction.

Introducing an additional loss that strongly penalizes non-dominant entries in P˜ and encourages it to approximate a hard permutation matrix P can mitigate such leakage. However, training over nearly hard permutation matrices induces a highly non-convex loss surface, and the optimization process is prone to getting trapped in local minima (Mena et al., 2018; Jang et al., 2017). Our experiments empirically observed that a few thousand permutations can be handled at once through this approach. However, the number of permutations grows factorially, which leads us to introduce the following two-stage hierarchical optimization, where aforementioned loss profiling (i.e., P1 and P2) is performed to determine learning-friendly orders at each level. Figure 4 illustrates our hierarchical method. We start with the initial set of permutation candidates P0 = {P1*, . . . , P*T }. The global stage splits each token sequence into several blocks and finds a good permutation at the block level. The local stage refines this coarse ordering by permuting the tokens within each block discovered at the global stage. Formally, the two stages operate as follows.

Global stage. Let the search depth be K and T = (K + 1)!. Let P1 := P0. For k = 1*, . . . , K*,
we conceptually split each target sequence into k blocks, 1and apply the loss profiling to the new 1When k = 1, the sequence is not split into blocks P˜ ∈ [0, 1]L×L and perform empirical risk minimization jointly over θ and P˜; namely,

$$\operatorname*{min}_{\theta,\bar{P}}\;\frac{1}{m}\sum_{i=1}^{m}\ell({\cal T}_{\theta},X_{i},Y_{i}\bar{P}).$$
$$(3.3)$$

ℓ(Tθ, Xi, YiP˜). (3.3)
$$(4.1)$$

Figure 3: Evaluation loss

![3_image_0.png](3_image_0.png) curves when trained with two different orders.

P1. Let E ∈ N. Train a Transformer for E epochs on a mixed dataset D¯ := ST
t=1 DPt. Let Tθ′
be the Transformer after training.

P2. Compute the loss on the validation set D′for each permutation; namely, for t = 1*, . . . , T*,
compute

$${\mathcal{L}}(D^{\prime},P_{t})={\frac{1}{m^{\prime}}}\sum_{i=1}^{m^{\prime}}\ell({\mathcal{T}}_{\theta^{\prime}},X_{i}^{\prime},Y_{i}^{\prime}P_{t}).$$

Then, the most learning-friendly order P∗:= Pτ is determined with τ =
arg mint L(D′, Pt).

We discovered this is also the case with the training with different decoder token orders, see Figure 3. Exploiting this observation, we proposed to train a Transformer only for a few epochs on a dataset with various orders in mixture and identify learning-friendly orders as "easy samples," for which the loss drops faster.

More formally, let D = {(Xi, Yi)}
m i=1 and D′ = {(X′i, Y ′
i)}
m′
i=1 be training and validation sets, respectively. Let P = {P1*, . . . , P*T } be the set of T candidate permutation matrices. Let DPt be the set D
with reordered target sequences by Pt, i.e., DPt = {(Xi, YiPt)}
m i=1.

We determine learning-friendly orders through the following *loss profiling*.

We introduce our strategy for discovering a suitable permutation of target token sequences. The key idea is to leverage a characteristic of the training dynamics of deep neural networks: they tend to learn easy samples in the early stages of training, and gradually adapt to harder samples later. This phenomenon has been reported in several contexts in the literature, such as Arpit et al. (2017) for learning with noisy labels and Baldock et al. (2021) for identifying difficult examples.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

![4_image_0.png](4_image_0.png)

$$(4.2)$$

Figure 4: Search flow of our hierarchical approach. **Global stage**: The proposed method generates T candidate permutations by swapping the sequence at the macro-level, exchanging P token blocks to quickly spot coarse, learning-friendly orders. **Local stage**: inside each chosen block, the proposed method further permutes the tokens, refining the sequence to discover a final permutation that maximizes learning ease.

$$\bigcup_{P\in{\mathcal{P}}_{k}}\{P Q_{1},\ldots,P Q_{k!}\},$$
{P Q1*, . . . , P Q*k!}, (4.2)
where Qi ∈ [0, 1]L×L are the block-level permutations. The best ⌊T /(k + 1)!⌋ permutations define the new candidate set Pk+1. We then apply the loss profiling to the final candidate set Pg := PK+1 and determine the best permutation Pg. This permutation is then refined with the local stage.

Local stage. Let P1 ∈ Pg be the initial permutation. We again conceptually split each target sequence into blocks of size l. Let Ri1*, . . . , R*il! ∈ [0, 1]L×L be all the permutations inside the i-th block. These permutations do not change the orders outside the i-th block. For each block length l = {2, 3*, . . . ,* ⌊L/2⌋}2, we apply the loss profiling to the new candidate set:

$[L/l]$  $\bigcup\{PR_{1}^{i},\ldots,PR_{l!}^{i}\}$,  $i$=1
$$(4.3)$$
$$(4.4)$$

and denote the lowest-loss result by Pl. Keeping each block's internal order fixed, we perform loss profiling over the ⌊L/l⌋ block-reordering candidates:

$$\{P_{l}Q_{1}^{\prime},\,P_{l}Q_{2}^{\prime},\,\ldots,\,P_{l}Q_{\lfloor L/l\rfloor}^{\prime}\}.$$
⌊L/l⌋}. (4.4)
The best candidate becomes the initial permutation for the next block size l + 1. Computational overheads. While the proposed framework repeats training to narrow down the permutations to learning-friendly ones, several aspects keep it practically efficient. First, each of the training runs only for 800–1,600 steps (equivalently, 1–2 epochs with 105samples of batch size 128) as the difference of loss drop speeds between candidate permutations becomes readily evident in the early stage. Second, a single training can handle a few thousand permutations (up to 7! = 5, 040 in our experiments). Third, our global–local framework provides efficient exploration. Specifically, with a global-stage depth of K, it needs K runs of training, and the local stage needs 2(⌊L/2⌋ − 1) runs. In our experiments, the longest exploration took 1–7 hours on a single GPU of the NVIDIA A6000ada to find the learning-friendly permutation. It is also worth noting that using a small Transformer model in the exploration is sufficient, as the learning-friendly orders must be universal. permutation set:

## 5 Experiments 5.1 Order-Sensitive Tasks

where ReLU(z) = max(z, 0). The forward order is trivial to learn because each step depends only
on the current input token xi and the immediately preceding output yi−1; in the reverse order, that
dependency becomes latent.
SQUARE**-19.** The recurrence performs the squared sum modulo 19 of the i-th input token xi and the previous output token yi values:
y1 = x1 and yi = x
2
i + y
2
$\Phi\in\{-9,...,9\}$, $i=2,...,L$. (5.3)
The squaring operation is non-injective. The values range in {−9*, . . . ,* 9}, and for any z ∈ {−9, . . . , 9*} \ {*0}, the preimage of z
2cannot be uniquely determined.
I**NDEX**. The recurrence performs input-element pointing based on the latest target tokens.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 In the forward order, memorizing just 192 = 361 cases suffices to output the target sequence.

In reverse order Y
r, however, even with y5 = −9 known, y4 is still ambiguous between 1 and
−1, so learning becomes much harder. Generation examples for the remaining tasks are provided in Appendix D. Our experiments will focus on the aforementioned three tasks, but the following PROD task will also be used to show that our method can reproduce the observation in Shen et al. (2023). PROD. Given two zero-padded input numbers a and b, the target sequence is their product Y = [ab]. When the digits are emitted from least significant to most significant, we denote the sequence by Y (forward order); when the digits are emitted in the opposite direction, we denote it by Y
r(reverse order). Unlike the three tasks proposed above, this multiplication task has been examined in earlier studies. Although it does not satisfy the recurrence in (5.1), it still exhibits moderate order sensitivity.

## 5.2 Experimental Setup

Datasets. We generated datasets for the tasks given in Section 5.1. The target length L ranges between {5, 6*, . . . ,* 100}. The INDEX task introduces a window size d ∈ {2, 4, 8}. The training set

$$y_{i-1}\,),$$

To evaluate the proposed method, we introduce three tasks. They can be learned relatively easily with the forward order, which however becomes challenging with the reverse or random orders.

Let X = [x1, x2*, ...*] be an input sequence and Y = [y1*, ..., y*L] an target sequence of fixed length L.

At a high level, the target sequence of the three tasks is defined by the following recurrence.

yi = f(X, y1*, . . . , y*i−1), (5.1)
where f(·) is a *non-injective* function with respect to y1*, . . . , y*i−1. Namely, other than the forward order, one cannot uniquely determine preceding target tokens y1*, . . . , y*i−1 from X and yi*, . . . , y*L.

Any disruption of the natural left-to-right order—such as reversing or randomly permuting the targets—breaks the causal chain and substantially increases the learning difficulty.

RELU. The recurrence performs the rectified sum:
y1 = x1 and yi = ReLUxi + yi−1
, i = 2*, . . . , L,* (5.2)

$$y_{1}=x_{2}$$
$$y_{1}=x_{1}\quad\text{and}\quad y_{i}=x_{p},\quad p=\sum_{j=1}^{d}y_{i-j}\mod L,\quad i=2,\ldots,L,\tag{5.4}$$
$\eqref{eq:walpha}$
$$x_{i}+y_{i-1}),\quad i=2,\ldots,L,$$

$$(S.2)$$
where d ≤ L is a fixed window size. Forward order enables the model to compute p incrementally,
whereas a reversed or random order destroys the causal chain.
Example 5.1 (SQUARE-19). Given the input sequence X = [7, −2, 4, 1, 3] and the initial value
y1 = x1 = 7, applying the recurrence in (5.3) produces
$y_{1}=7,\quad y_{2}=(-2)^{2}+7^{2}\bmod19=-4,\quad y_{3}=4^{2}+(-4)^{2}\bmod19=-6,$  $y_{4}=1^{2}+(-6)^{2}\bmod19=-1,\quad y_{5}=3^{2}+(-1)^{2}\bmod19=-9.$
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 contains 100,000 samples, and the evaluation set contains 1,000 samples. Different random seeds (42 for training and 123 for evaluation) make the two sets disjoint. Training setup. We use the GPT-2 architecture (Radford et al., 2019) of two sizes, small and large. The small model consists of one layer and one attention head, and is used for exploration with the proposed method, while the large one has six layers and is used for the final training with the discovered learning-friendly order to pursue the accuracy. The other parameters are as follows: the embedding and feed-forward dimensions are (demb, dffn) = (512, 2048), and dropout is set to 0.1.

Positional embeddings are randomly initialized and optimized throughout training. The model is trained for 1 and 10 epochs for small and large models, respectively, with AdamW (Loshchilov &
Hutter, 2019) (β1 = 0.9, β2 = 0.999), a linearly decaying learning rate starting from 5.0 × 10−5, and a batch size of 128. Exploration setup. The proposed method uses loss profiling in the global–local pipeline. In the loss profiling, a Transformer is trained on the train set, and then the permutations (i.e., orders) are ranked by the evaluation loss on the evaluation set. The success rate is the proportion of completely correct target sequences generated by the Transformer to the total number of samples in the evaluation set.

Initialization. We tested several initializations of the initial candidate set P0 in the loss profiling
(Section 5.4) and global stage (Section 5.5).

- Pg consists of the identity permutation plus random permutations. For example, if the set size is 100, it includes one identity permutation and 99 random ones.

- Pf consists of permutations obtained by splitting the forward and reverse orders into column-wise blocks and swapping those blocks.

- Pr consists of permutations chosen uniformly at random. - Pb consists of permutations formed by splitting the forward and reverse sequence into length b and permutes those blocks, and fix b = 5 in experiments.

The original target sequences in our dataset are all in the forward order, corresponding to the identity permutation. Examples of these permutation sets are provided in Appendix E.

## 5.3 Learning With Forward And Reverse Orders

We first show that the learning is easy with forward order, while it becomes significantly challenging with the reverse order. Table 1 reports the success rate when trained with the forward and the reverse orders. As explained in Section 5.1, every task is configured to be learningfriendly in the forward order but learning-unfriendly in the reverse order. Consistent with this design, Table 1 shows that the model almost fully learns each task in the forward order, whereas in the reverse order, the success rate never exceeds roughly 10 %. A closer look at taskspecific trends reveals that the success rate for the RELU and SQUARE-19 tasks remains almost unchanged as the target length grows. By contrast, for INDEX, the forward order success rate declines with the window size d, indicating that the model struggles when each prediction depends on a larger number of previous outputs.

Table 1: Success rates for the forward and the reverse. The forward order is significantly more learning friendly.

## 5.4 Loss Profiling For Discovering The Forward Order

| Task          | Target length   | Success rate (%) Forward Reverse   |
|---------------|-----------------|------------------------------------|
| L = 20        | 99.6            | 0.6                                |
| L = 50        | 99.9            | 5.6                                |
| L = 100       | 99.4            | 0.0                                |
| L = 20        | 100             | 0.1                                |
| L = 50        | 100             | 0.0                                |
| L = 100       | 100             | 0.0                                |
| L = 13, d = 2 | 100             | 9.8                                |
| L = 13, d = 4 | 62.3            | 1.3                                |
| L = 13, d = 8 | 81.8            | 2.2                                |
| L = 31, d = 2 | 100             | 0.8                                |

We next justify the loss profiling before the proposed global–local pipeline. We trained a Transformer on a set of permutations, Pg, containing one learning-friendly forward order (ID=0) and 127 randomly generated learning-unfriendly orders. We set L = 50 for the RELU and SQUARE-19 tasks. For the INDEX task, we set L = 31 and d = 4. Figure 5(a) shows the evaluation loss for each of the 128 permutations at loss profiling. The forward order (ID=0) achieves the lowest loss among other orders across all tasks, suggesting one can select 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png)

Figure 5: (a) Evaluation loss for each permutation obtained via loss profiling. ID=0 corresponds to the forward order, while the others are randomly generated permutations. (b) Success rate of the model when retrained on permutations ranked by the loss values from (a). Permutations are ordered on the x-axis from best (left) to worst (right).

Table 2: The orders discovered by the proposed method in its global and local stages. Depth denotes the hierarchy level K reached in the global stage. Each order is listed relative to the forward sequence; when the list starts at 0, the forward order has been recovered. Forward orders identified at a given stage are highlighted in bold.

Task Target Length Depth Order after global stage Discovered final order

| RELU            |
|-----------------|
| SQUARE-19 INDEX |

INDEX

L = 13, d = 2 K = 6 [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] L = 13, d = 4 K = 6 [0, 1, 7, 6, 4, 2, 5, 8, 3, 9, 10, 11, 12] [0, 1, 7, 6, 4, 2, 5, 8, 3, 9, 10, 11, 12] L = 13, d = 8 K = 6 [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12, 0, 11] [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12, 0, 11]

PROD L = 10 K = 6 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

a learning-friendly permutation via loss profiling if the set contains one. This effect is particularly pronounced in the INDEX task, which is the hardest task among the three. Now, we ranked the 128 orders according to the evaluation losses shown in Figure 5 and then trained a Transformer for each of the top 32 orders. Figure 5(b) shows that the success rate generally aligns with the rank; training with a highly ranked order leads to a high success rate in the RELU and SQUARE-19 tasks. For the INDEX task, which is the hardest task among the three, the success rate was all close to zero (omitted from the plot). Still, the top-ranked order (i.e., forward order) is the most learningfriendly order by the construction of the task. This result indicates that the loss profiling is more advantageous in finding *implicit* learning-friendly orders than exhaustively repeating full training and success rate evaluation, even ignoring the computational burden of the latter. The result also justifies using small Transformers in the exploration stage, even for hard tasks. One only needs to use large and powerful models in the final training with the discovered order.

## 5.5 Global–Local Method With Loss Profiling

| L = 8         | K = 4   | [0, 2, 1, 3, 4, 5, 6, 7]                   | [0, 1, 2, 3, 4, 5, 6, 7]                   |
|---------------|---------|--------------------------------------------|--------------------------------------------|
| L = 9         | K = 5   | [0, 1, 2, 3, 4, 5, 6, 7, 8]                | [0, 1, 2, 3, 4, 5, 6, 7, 8]                |
| L = 10        | K = 6   | [6, 7, 8, 9, 5, 4, 2, 3, 1, 0]             | [4, 5, 6, 7, 8, 9, 0, 1, 2, 3]             |
| L = 11        | K = 6   | [8, 9, 10, 7, 6, 5, 4, 3, 2, 1, 0]         | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]         |
| L = 12        | K = 6   | [6, 7, 8, 9, 10, 11, 5, 4, 2, 3, 1, 0]     | [1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11]     |
| L = 13        | K = 6   | [11, 12, 10, 9, 8, 7, 6, 5, 4, 2, 3, 1, 0] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| L = 8         | K = 4   | [1, 2, 4, 5, 0, 6, 7, 3]                   | [1, 2, 4, 5, 0, 6, 7, 3]                   |
| L = 9         | K = 5   | [0, 1, 2, 3, 4, 5, 6, 7, 8]                | [0, 1, 2, 3, 4, 5, 6, 7, 8]                |
| L = 10        | K = 6   | [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]             | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]             |
| L = 11        | K = 6   | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]         | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]         |
| L = 12        | K = 6   | [1, 2, 3, 4, 5, 6, 7, 11, 10, 9, 0, 8]     | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]     |
| L = 13        | K = 6   | [0, 1, 2, 3, 12, 11, 10, 4, 5, 6, 7, 8, 9] | [8, 9, 0, 1, 2, 3, 4, 10, 11, 12, 5, 6, 7] |
| L = 13, d = 2 | K = 6   | [1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] |
| L = 13, d = 4 | K = 6   | [0, 1, 7, 6, 4, 2, 5, 8, 3, 9, 10, 11, 12] | [0, 1, 7, 6, 4, 2, 5, 8, 3, 9, 10, 11, 12] |
| L = 13, d = 8 | K = 6   | [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12, 0, 11] | [1, 2, 3, 4, 5, 6, 7, 8, 10, 9, 12, 0, 11] |

We now demonstrate that the proposed method can discover the learning-friendly orders up to L =
13 (i.e., roughly 6 billion possible orders) with random initialization Pr and L = 40 with a structured 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

![8_image_0.png](8_image_0.png)

(b) Structured initialization
initialization Pb. One should use the former if no prior knowledge of the ordering is available, while the latter (or something similar) can be designed for some tasks. For example, on polynomial tasks, one may permute the tokens at a monomial level, which naturally leads to initialization with blocklevel permutations.

Initialization with Pr. Table 2 shows the permutation discovered at the global stage and the final one. After the global stage, tokens that are neighbors in the input usually remain adjacent, showing that the method first captures coarse structure. The subsequent local stage then fine-tunes this order and moves the order closer to the optimal forward arrangement. For the RELU and SQUARE-19 tasks, global orders were often already learning-friendly, and retraining a model on them always produces a higher success rate than training on the reverse order (see Figure 6(a)). The INDEX task proves harder: as the reference width d grows, learning is difficult even in the forward order (see Section 5.1), which flattens the loss landscape and makes good permutations harder to rank. In the PROD, the proposed method succeeds in rediscovering the least-significant-digit first order reported by Shen et al. (2023), and it finds the optimal order for target lengths up to 13, identifying a single solution among roughly 13! ≈ 6 × 109 possibilities.

Structured initialization with Pb. When the search is initialized with Pb, the proposed method scales to much longer target sequences. Figure 6(b) shows the resulting success rate curves for RELU and SQUARE-19: the optimal order is found for both tasks up to L = 30, and for RELU
even at L = 40. At L = 40, the theoretical permutation space still contains about 1047 elements, indicating that once implausible candidates are pruned, the proposed method can explore the remaining space far more effectively. These results demonstrate that our hierarchical search can recover optimal orders in both the most challenging fully random scenario and the more realistic, blockrestricted setting, and that its advantage grows as the candidate space is made more coherent.

## 6 Conclusion

This study addressed a new task of reordering decoder input tokens for Transformers in learning arithmetic tasks. In essence, the proposed method performs short-term training on a mixture of target sequences in different orders and discovers easy samples for which the loss drops faster, as learningfriendly orders. To search the factorially large space efficiently, we propose a two-stage hierarchical approach combining global block-level exploration with local refinement. The experiments on three order-sensitive arithmetic tasks (RELU, SQUARE-19, and INDEX) demonstrated that the proposed method discovers a learning-friendly order, improving the success rate from about 10 % to near 100 % and works for target lengths up to 13 tokens (13! > 6 × 109 permutations). Moreover, it rediscovered the reverse-digit order reported in earlier work on the PROD task. This study presents an automatically unraveling chain of thought that markedly enhances a Transformer's reasoning ability. The extension to longer sequences and target sequences at a variable length will be future work.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Reproducibility Statement and the Use of LLMs. Common details of our experimental setup are described in Section 5.2, while any parameters or settings specific to each experiment are provided at the beginning of each experiment in Section 5. The source code used for experiments is provided as supplemental material and will be publicly available after clean-up. The LLMs were used for assistance purposes only. We used them to improve our writing, speed up coding, and to assist in part of our literature search by listing papers with OpenAI's ChatGPT (OpenAI, 2024). No essential contributions were made by the LLMs.

## References

Alberto Alfarano, Francois Charton, and Amaury Hayat. Global lyapunov functions: a long-standing open problem in mathematics, with symbolic transformers. In *Advances in Neural Information* Processing Systems, 2024.

Devansh Arpit, Stanisław Jastrzundefinedbski, Nicolas Ballas, David Krueger, Emmanuel Bengio, Maxinder S. Kanwal, Tegan Maharaj, Asja Fischer, Aaron Courville, Yoshua Bengio, and Simon Lacoste-Julien. A closer look at memorization in deep networks. In International Conference on Machine Learning, 2017.

Robert J. N. Baldock, Hartmut Maennel, and Behnam Neyshabur. Deep learning through the lens of example difficulty. In *Advances in Neural Information Processing Systems*, 2021.

Franc¸ois Charton. Linear algebra with transformers. *Transactions on Machine Learning Research*,
2022.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. Transactions on Machine Learning Research, 2023.

Mahsa Forouzesh and Patrick Thiran. Differences between hard and noisy-labeled samples: An empirical study. In Proceedings of the 2024 SIAM International Conference on Data Mining (SDM), pp. 91–99, 2024.

Michael Hahn and Mark Rofin. Why are sensitive functions hard for transformers? In Proceedings of the Annual Meeting of the Association for Computational Linguistics, 2024.

Bo Han, Quanming Yao, Xingrui Yu, Gang Niu, Miao Xu, Weihua Hu, Ivor Tsang, and Masashi Sugiyama. Co-teaching: Robust training of deep neural networks with extremely noisy labels. In Advances in Neural Information Processing Systems, 2018.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In International Conference on Learning Representations, 2017.

Samy Jelassi, Stephane d'Ascoli, Carles Domingo-Enrich, Yuhuai Wu, Yuanzhi Li, and Franc¸ois ´
Charton. Length generalization in arithmetic transformers, 2023.

Lu Jiang, Zhengyuan Zhou, Thomas Leung, Li-Jia Li, and Li Fei-Fei. Mentornet: Learning datadriven curriculum for very deep neural networks on corrupted labels. In *International Conference* on Machine Learning, 2018.

Hiroshi Kera, Yuki Ishihara, Yuta Kambe, Tristan Vaccon, and Kazuhiro Yokoyama. Learning to compute grobner bases. In ¨ *Advances in Neural Information Processing Systems*, 2024.

Hiroshi Kera, Nico Pelleriti, Yuki Ishihara, Max Zimmer, and Sebastian Pokutta. Computational algebra with attention: Transformer oracles for border basis algorithms. *arXiv:2505.23696*, 2025.

Juno Kim and Taiji Suzuki. Transformers provably solve parity efficiently with chain of thought. In International Conference on Learning Representations, 2025.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. In *Advances in Neural Information Processing Systems*, 2022.

Guillaume Lample and Franc¸ois Charton. Deep learning for symbolic mathematics. In International Conference on Learning Representations, 2020.

Cathy Yuanchen Li, Jana Sotakov ´ a, Emily Wenger, Mohamed Malhou, Evrard Garcelon, Franc¸ois ´
Charton, and Kristin Lauter. Salsapicante: A machine learning attack on lwe with binary secrets. In *Proceedings of the ACM SIGSAC Conference on Computer and Communications Security*, 2023a.

Cathy Yuanchen Li, Emily Wenger, Zeyuan Allen-Zhu, Francois Charton, and Kristin E. Lauter.

Salsa verde: a machine learning attack on learning with errors with sparse small secrets. In Advances in Neural Information Processing Systems, 2023b.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.

Gonzalo Mena, David Belanger, Scott Linderman, and Jasper Snoek. Learning latent permutations with gumbel-sinkhorn networks. In *International Conference on Learning Representations*, 2018.

OpenAI. Chatgpt. https://openai.com/chatgpt, 2024. Model: GPT-4. Accessed: 202509-25.

Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. Language models are unsupervised multitask learners. *OpenAI*, 2019.

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International Conference on Machine Learning, 2019.

Shai Shalev-Shwartz, Ohad Shamir, and Shaked Shammah. Failures of gradient-based deep learning.

In *International Conference on Machine Learning*, 2017.

Ruoqi Shen, Sebastien Bubeck, Ronen Eldan, Yin Tat Lee, Yuanzhi Li, and Yi Zhang. Positional ´
description matters for transformers arithmetic. *arXiv:2311.14737*, 2023.

Swabha Swayamdipta, Roy Schwartz, Nicholas Lourie, Yizhong Wang, Hannaneh Hajishirzi, Noah A. Smith, and Yejin Choi. Dataset cartography: Mapping and diagnosing datasets with training dynamics. In *Proceedings of EMNLP*, 2020.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, 2022.

Emily Wenger, Mingjie Chen, Franc¸ois Charton, and Kristin E. Lauter. SALSA: attacking lattice cryptography with transformers. In *Advances in Neural Information Processing Systems*, 2022.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L. Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems, 2023.

Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep learning requires rethinking generalization. In International Conference on Learning Representations, 2017.

Xuan Zhang, Chao Du, Tianyu Pang, Qian Liu, Wei Gao, and Min Lin. Chain of preference optimization: Improving chain-of-thought reasoning in LLMs. In *Advances in Neural Information* Processing Systems, 2024.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 We present the attention maps obtained when training a Transformer on our proposed RELU task

![11_image_0.png](11_image_0.png) with datasets reordered in different ways. For this analysis, we use a GPT-2 model with a single layer and a single attention head. Figure 7 shows the attention maps for target length L = 20 under four target orders. The forward and reverse orders are defined in Section 5.1. The one-permuted order swaps exactly one pair of adjacent target tokens, whereas the random order is a random permutation of the forward sequence. Figure 8 illustrates how the attention maps change as the target length increases.

(a) Forward order (b) one permuted order

![11_image_1.png](11_image_1.png) (c) Random order (d) Reverse order

| Task          | Target length   | Sparsity   |       |
|---------------|-----------------|------------|-------|
| Forward       | Reverse         |            |       |
| L = 20        | 1.160           | 1.640      |       |
| RELU          | L = 50          | 1.462      | 4.319 |
| L = 100       | 1.687           | 3.195      |       |
| L = 20        | 1.117           | 1.531      |       |
| SQUARE-19     | L = 50          | 1.773      | 1.914 |
| L = 100       | 1.407           | 1.990      |       |
| L = 13, d = 2 | 0.848           | 2.574      |       |
| INDEX         | L = 13, d = 4   | 0.887      | 1.486 |
| L = 13, d = 8 | 1.116           | 1.596      |       |

## A Visualizing Attention Map