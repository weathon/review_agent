000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 The Mamba model has gained significant attention for its computational advantages over Transformer-based models, while achieving comparable performance across a wide range of language tasks. Like Transformers, Mamba exhibits in-context learning (ICL) capabilities, i.e., making predictions for new tasks based on a prompt containing input-label pairs and a query, without requiring fine-tuning. Despite its empirical success, the theoretical understanding of Mamba remains limited, largely due to the nonlinearity introduced by its gating mechanism. To the best of our knowledge, this paper presents the first theoretical analysis of the training dynamics of a one-layer Mamba model, which consists of a linear attention component followed by a nonlinear gating layer, and its ICL generalization on unseen binary classification tasks, even when the prompt includes additive outliers. Our analysis shows that Mamba leverages the linear attention layer to select informative context examples and uses the nonlinear gating layer to suppress the influence of outliers.

By establishing and comparing to the analysis of linear Transformers under the same setting, we show that although Mamba may require more training iterations to converge, it maintains accurate predictions even when the proportion of outliers exceeds the threshold that a linear Transformer can tolerate. These theoretical findings are supported by empirical experiments.

## 1 Introduction

Transformer-based large language models (LLMs) (Brown et al., 2020; Achiam et al., 2023; Guo et al., 2025) have demonstrated remarkable capabilities across a wide range of language, vision, and reasoning tasks. However, they face efficiency challenges when processing long sequences due to the quadratic time and memory complexity of the self-attention mechanism with respect to sequence length (Gu & Dao, 2023; Dao & Gu, 2024). To address this, many efficient alternative architectures have been proposed, including state space models (SSMs) such as S4 (Gu et al., 2021; 2022) and H3 (Fu et al., 2023a). Among them, Mamba (Gu & Dao, 2023) has attracted significant attention for its strong empirical performance, linear computational complexity, and hardware-friendly properties that enable efficient parallelization. These advantages have sparked growing interest in understanding the mechanism of Mamba and whether it can match or surpass the capabilities of Transformer models. One particularly intriguing property of LLMs is *in-context learning (ICL)* (Brown et al., 2020; Garg et al., 2022), which allows a pre-trained model to generalize to new tasks without any parameter updates. By simply augmenting the input with a prompt containing a few labeled examples from the new task, the model can produce accurate predictions for unseen tasks. While LLMs have demonstrated impressive ICL generalization, their performance is sensitive to the quality of the context examples (Liu et al., 2022; Wu et al., 2023b). In particular, ICL performance can degrade significantly in the presence of outliers or adversarial attacks on prompts, such as data poisoning, resulting in incorrect predictions (Wan et al., 2023; Kandpal et al., 2023; Qiang et al., 2023; He et al., 2024; Zhao et al., 2024; Anwar et al., 2024).

Recent empirical work (Park et al., 2024; Halloran et al., 2024; Grazzi et al., 2024; Jelassi et al.,
2024; Arora et al., 2024; Waleffe et al., 2024) has demonstrated that Mamba can also perform ICL on function learning and natural language processing tasks. (Park et al., 2024; Grazzi et al., 2024) show that Mamba is competitive with Transformers of similar size in some ICL tasks and outperforms them in settings with many outliers, such as regression with corrupted examples. On the other hand, studies such as (Park et al., 2024; Waleffe et al., 2024; Arora et al., 2024; Jelassi et al., 2024) identify

# Can Mamba Learn In Context With Outliers? A Theoretical Generalization Analysis

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 limitations of Mamba in retrieval-based and long-context reasoning tasks. Despite these empirical insights, several fundamental questions remain open:
Why and how can a Mamba model be trained to perform in-context generalization to new tasks?

How robust is it to outliers? Under what conditions can Mamba outperform Transformers for ICL? (Li et al., 2024b) and (Li et al., 2025b) analyze Mamba-like models, e.g., simplified H3 and gated linear attention, and show that the global minima of the loss landscapes correspond to models whose outputs, when given a prompt, implicitly perform a weighted preconditioned gradient descent using the context examples. This serves as the counterpart to the preconditioned gradient descent interpretation of ICL in Transformers (Ahn et al., 2023). Joseph et al. (2024) shows that continuous SSMs can learn dynamic systems in context. Bondaschi et al. (2025) proves that Mamba is expressive enough to represent optimal Laplacian smoothing. However, these studies do not address whether practical training methods can reliably yield Mamba models with ICL capabilities, nor do they provide theoretical guarantees for generalization or robustness in the presence of outliers.

## 1.1 Major Contributions

This paper presents the first theoretical analysis of the training dynamics of Mamba models and their resulting ICL performance, including scenarios where context examples in the prompt contain outliers. We focus on training Mamba on binary classification tasks where input data consist of both relevant patterns, which determine the label, and irrelevant patterns, which do not. Additionally, context inputs may include additive outliers that perturb the labels. While our analysis is based on one-layer Mamba architectures, this setting aligns with the scope of state-of-the-art theoretical studies on the training dynamics and generalization of Transformers and other neural networks, which also typically focus on one-hidden-layer models (Zhang et al., 2023; Li et al., 2024a;b; 2025b). Our main contributions are as follows:
1. **Quantitative analysis of ICL emergence and robustness to outliers in Mamba**. We characterize the number of context examples and training iterations required for a Mamba model to acquire ICL capabilities for new tasks that were not present during training. We prove that when trained with prompts that may contain a finite number of outlier patterns, Mamba can generalize in-context on new tasks when the context examples contain unseen outliers that are linear combinations of the training-time outliers. Furthermore, Mamba can maintain accurate ICL generalization even when the fraction of outlier-containing context examples approaches 1, demonstrating strong robustness. 2. **Theoretical comparison between Mamba and linear Transformers**. We provide a theoretical characterization of the convergence and generalization properties of one-layer single-head linear Transformers trained on the same tasks. While linear Transformers may converge faster with smaller batch sizes, they can only in-context generalize effectively when the fraction of outlier-containing context examples is less than 1/2, much less than that for Mamba. Moreover, linear Transformers require significantly more context examples than Mamba to achieve comparable generalization performance. This highlights Mamba's superior robustness to a high density of outliers in ICL. 3.**Theoretical characterization of the mechanism by which Mamba implements ICL**. We show that the equivalent linear attention mechanism in Mamba selects context examples that share the same relevant pattern as the query, while the nonlinear gating mechanism suppresses corrupted examples and applies an exponential decay in importance based on index distance, emphasizing examples closer to the query. Together, these mechanisms enable Mamba to suppress irrelevant or corrupted context examples and focus on informative and nearby ones, achieving effective and robust ICL.

## 1.2 Related Works

Theoretical Analysis of ICL. Existing theoretical works of ICL primarily focus on Transformerbased models. Garg et al. (2022); Akyürek et al. (2023); Bai et al. (2023); Von Oswald et al. (2023); Ahn et al. (2023) illustrate that Transformers can implement many machine learning algorithms, such as gradient-based methods, via ICL. Zhang et al. (2023); Huang et al. (2023); Wu et al. (2023a); Li et al. (2024a); Chen et al. (2024a) provably investigate the training dynamics and generalization of ICL on single/multi-head Transformers. Yang et al. (2024d); Kim & Suzuki (2024); Oko et al. (2024) extend the analysis to learning complicated nonlinear functions by ICL. Connections Between Mamba and Transformers. Ali et al. (2024) finds that Mamba exhibits explainability metrics comparable to those of Transformers. Dao & Gu (2024) shows that SSMs

## 2 Problem Formulation

and variants of attention mechanisms share a large intersection and can be viewed as duals of each other. Han et al. (2024) notes a similarity between the forget gate in Mamba and the positional encodings in Transformers. The complementary strengths, Mamba's computational efficiency and Transformers' ability to capture global dependencies, have motivated the development of hybrid architectures (Hatamizadeh & Kautz, 2024; Lenz et al., 2025; Xu et al., 2024). Optimization and Generalization of the Attention Architecture. Some other works focus on the optimization and generalization of attention-based models without nonlinear gating beyond the ICL setting. Jelassi et al. (2022); Li et al. (2023a;b); Jiang et al. (2024); Yang et al. (2024a); Li et al. (2025a) study the generalization of one-layer Transformers in classification tasks by formulating spatial association, key features, or the semantic structure of the input. Huang et al. (2024); Nichani et al. (2025); Ren et al. (2024) investigate the problem in next-token prediction based on the partial order, bigram, or semantic association assumption. Chen et al. (2024a); He et al. (2025) extend the analysis to multi-head attention networks.

The learning model, Mamba, is proposed in (Gu & Dao, 2023)1 Given the input U = (u1, *· · ·* ,um) ∈
R

d0×m, the model outputs oi recursively through the hidden states hi, i ∈ [m]. Starting from h0 = U, a one-layer Mamba can be formulated as

$$(1)$$
 A Hamilton can be formulated as  $\mathbf{h}_i=\mathbf{h}_{i-1}\odot\tilde{\mathbf{A}}_i+(\mathbf{u}_i\mathbf{1}_m^\top)\odot\tilde{\mathbf{B}}_i\quad\in\mathbb{R}^{d_0\times m},\quad\forall i\in[m]$  $\mathbf{o}_i=\mathbf{h}_i\mathbf{C}_i\quad\in\mathbb{R}^{d_0}$,
where B˜i = (B˜ ⊤
1,i, *· · ·* , B˜ ⊤
d0,i)
⊤ ∈ R
d0×m with B˜j,i = (∆j,iBi)(exp(∆j,iA) − Im)(∆j,iA)
−1 and Bi = u
⊤
i W⊤
B ∈ R
1×m, WB ∈ R
m×d0, A˜i = (A˜⊤
1,i, *· · ·* , A˜⊤
d0,i)
⊤ ∈ R
d0×m with A˜j,i =
diag(exp(∆j,iA))⊤, Ci = WC ui ∈ R
m with WC ∈ R
m×d0. 1m is an all-ones vector in R
m. ⊙
108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 and exp(·) are element-wise product and exponential operations, respectively. diag(·) : R
d0×d0 →
R

d0 outputs the diagonal of the input as a vector. σ(·) : z ∈ R 7→ (1 + exp(−z))−1 ∈ R is the sigmoid function. ∆j,i = softplus(w⊤
j ui) = log(1 + exp(w⊤
j ui)) ∈ R, which is parameterized by W = (w1, *· · ·* , wd0) ∈ R
d0×d0. Denote w = wd0. Following the assumption in Theorem 1 of (Gu
& Dao, 2023), we select A = −Im ∈ R
m×m for simplicity of analysis.

Following the theoretical setup used in recent in-context learning (ICL) analyses (Garg et al., 2022; Huang et al., 2023; Li et al., 2024a;b; 2025b), we consider training a model on prompts from a subset of tasks to endow it with ICL capabilities on unseen tasks. This framework is motivated by the observation (Chen et al., 2024c) that although LLMs are typically trained without supervised labels, natural text often contains implicit input-output pairs, i.e., phrases following similar templates, that resemble the prompt-query format used in our setup. Specifically, we consider a set of binary classification tasks T , where for a certain task f ∈ T , the label z ∈ {+1, −1} of a given input query x*query* ∈ R
dis determined by z = f(xquery) ∈ {+1, −1}. Then, the prompt P for x*query* is constructed as

 Let us  $\pmb{P}=\begin{pmatrix}\pmb{x_1}&\pmb{x_2}&\cdots&\pmb{x_l}&\pmb{x_{query}}\\ y_1&y_2&\cdots&y_l&0\end{pmatrix}:=(\pmb{p_1},\pmb{p_2},\cdots,\pmb{p_{query}})\in\mathbb{R}^{(d+1)\times(l+1)}$,  1. 
where yi = f(xi), i ∈ [l]. With the prompt P in (200) as the input to the Mamba model in (1) with m = l + 1 and d0 = d + 1, the output of one-layer Mamba can be rewritten as

$$F(\Psi;\mathbf{P})=\overset{\top}{\mathbf{e}_{d+1}^{\top}}\mathbf{o}_{l+1}=\sum_{i=1}^{l+1}G_{i,l+1}(\mathbf{w})y_{i}\mathbf{p}_{i}^{\top}\mathbf{W}_{B}^{\top}\mathbf{W}_{C}\mathbf{p}_{query},$$  where $G_{i,l+1}(\mathbf{w})=\begin{cases}\sigma(\mathbf{w}^{\top}\mathbf{p}_{i})\prod_{j=i+1}^{l+1}(1-\sigma(\mathbf{w}^{\top}\mathbf{p}_{j})),&i<l+1,\\ \sigma(\mathbf{w}^{\top}\mathbf{p}_{query}),&i=l+1,\end{cases}$
$$\mathbf{(2)}$$
$$({\mathfrak{I}})$$

where ed+1 = (0, *· · ·* , 0, 1)⊤ ∈ R
d+1 and Ψ = {WB,WC , w} is the set of trainable parameters.

The derivation of (3) can be found in Appendix E.1. From (3), one can observe that a one-layer Mamba is equivalent to a **linear attention** layer parameterized by WB and WC followed by a nonlinear gating layer Gi,l+1(w) for i ∈ [l + 1]. Specifically, WB and WC can be respectively 1The theoretical extension of our framework to other SSM/linear RNN models is discussed in Appendix E.6.

3 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 interpreted as the key and query parameters in a Transformer model. Therefore, a Transformer with linear attention, commonly studied in the context of ICL (Zhang et al., 2023), can be viewed as a special case of the formulation in (3) by removing the nonlinear gating, i.e., setting Gi,l+1(w) = 1 for all i ∈ [l + 1]. We adopt this simplified formulation when comparing Mamba and Transformers in Section 3.4. Given N training examples consisting of prompt-label pairs (P
n, zn)
N
n=1, the model is trained by solving the empirical risk minimization problem using the hinge loss:

$$\min_{\Psi}\frac{1}{N}\sum_{n=1}^{N}\ell(\Psi;\mathbf{P}^{n},z^{n}),\;\text{where}\ell(\Psi;\mathbf{P}^{n},z^{n})=\max\{0,1-z^{n}\cdot F(\Psi;\mathbf{P}^{n})\}.$$
$$(4)$$
n=1
n)}. (4)
Each prompt P
n is generated from a distribution D, where the query x n query and all context inputs x n i are sampled independently, and the associated task f n is drawn from a set of training tasks Ttr ⊂ T .

Training Algorithm: The model is trained using stochastic gradient descent (SGD) with step size η with batch size B, summarized in Algorithm 1. W(0)
B and W(0)
Care initialized such that the first d diagonal entries of W(0)
B and W(0)
Care set as δ ∈ (0, 0.2]. w(0) follows Gaussian N (0, Id+1/(d + 1)). ICL Generalization in the Presence of Outliers: The testing prompt P
′follows an unknown distribution D′, which is different from the training prompt P and may contain outliers. Then, the ICL generalization of the model Ψ is computed as the classification error across all tasks in T ,
including those never appear during the training stage, i.e.,
L
0−1 f∈T ,P ′∼D′ (Ψ; P
′, z) = Ef∈T ,P ′∼D′-1[z · F(Ψ; P
′) < 0]. (5)

## 3 Main Theoretical Results

We first summarize insights of our theoretical results in Section 3.1. Then, we introduce our formulation for analysis in Section 3.2. Section 3.3 presents the theoretical results of learning for ICL generalization with Mamba. Section 3.4 analyzes linear Transformers for a comparison with Mamba models. We finally characterize the ICL mechanism by the trained Mamba in Section 3.5.

## 3.1 Main Theoretical Insights

We formulate a class of binary classification tasks where the labels in each task are determined by two selected relevant patterns. Such data formulation stems from the sparse representation assumption (Wright et al., 2010) for real-world data and is widely adopted in theoretical analysis (Li et al., 2024a; Huang et al., 2023; Jiang et al., 2024). The model is trained on a subset of these tasks using prompts that may include context examples corrupted by additive outliers. We then evaluate the model's performance on unseen tasks, where the prompts can contain outliers not observed during training. P1. Theoretical Characterization of Learning Dynamics, ICL Generalization, and Robustness to Outliers in Mamba Models. We provide quantitative guarantees that training with prompts can lead to favorable ICL generalization on unseen tasks, and these results hold even in the presence of outliers (Theorems 1 and 2). Specifically, if a fraction pa ∈ [0, 1) of the context examples in the training prompts contain additive outliers, we prove that the learned model still generalizes accurately at test time, as long as the fraction of outliers in the testing prompt, denoted by α, is less than min{1, pa · ltr/lts} where ltr and lts are the number of examples in the training and testing prompts, respectively. Notably, the outliers in the test prompt may be previously unseen, but should contain a positive linear combinations of outlier patterns seen during training. P2. A Comparison Between One-Layer **Mamba and Linear Transformer Models.** We theoretically analyze the convergence and ICL generalization of a one-layer linear Transformer (Theorems 3 and 4) for comparison. Our results show that linear Transformers require smaller batch sizes, fewer iterations, and milder constraints on the magnitude of outliers and the prompt length for successful training convergence compared to Mamba. However, linear Transformers can only generalize well when the test prompt has an outlier fraction α < 1/2, whereas Mamba could maintain accurate generalization even if α goes to 1. Moreover, even when both models can achieve ICL, e.g., when α is close to 1/2, linear Transformers require significantly more context examples to achieve comparable performance. Thus, despite requiring more effort during training, Mamba models demonstrate superior robustness to outliers during ICL.

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 P3. Mechanism of Mamba Models in Implementing ICL. Our analysis shows that the linear attention layer in Mamba selectively emphasizes context examples that share the same relevant pattern as the query, while the nonlinear gating layer promotes examples that are both close to the query and free of additive outliers. This dual mechanism enables the trained Mamba to suppress irrelevant or corrupted context examples and focus on informative examples close to the query, thus achieving successful and robust ICL.

## 3.2 Data And Tasks Modeling

Assume there are M1 relevant patterns {µj}
M1 j=1 and M2 irrelevant patterns {νk}
M2 k=1 with M1+M2 <
d. All the patterns from {µj}
M1 j=1 ∪ {νk}
M2 k=1 are orthogonal to each other, with ∥µj∥ = ∥νk∥ = β for j ∈ [M1], k ∈ [M2], and the constant β ≥ 1. Each input x contains one relevant pattern that determines the label, and one irrelevant pattern that does not affect the label. We consider a set of binary classification tasks in T where the binary labels are determined by the relevant patterns. For instance, for a task f that is determined by (µa, µb), *a, b* ∈ [M1], the label of x*query* is z = 1 (or z = −1) if the input x*query* contains µa (or µb), respectively.

Training Stage: For a given task f, we consider learning with a pa ∈ [0, 1) fraction of examples containing additive outliers
{v
∗
r }
V
r=1 that are orthogonal to each other and can affect the label of corresponding examples in each prompt, where v
∗s ⊥ µj , v
∗s ⊥
νk for any j ∈ [M1], k ∈ [M2], and s ∈ [V ]. The input of each context example satisfies Figure 1: An example of outliers in context inputs.

$\mathbf{c}=\begin{cases}\mu_{j}+\kappa\nu_{k}+\kappa_{a}\mathbf{v}_{s}^{*},&\text{with a probability of}p_{a}\\ \mu_{j}+\kappa\nu_{k},&\text{with a probability of}1-p_{a},\end{cases}$
(6)
for some s ∈ [V ], where j ∈ [M1] and k ∈ [M2] are arbitrarily selected. κ follows a uniform distribution U(−*K, K*) with K ≤ 1/2. v
∗
sis uniformly sampled from {v
∗
r }
V
r=1. No additive outliers exist in x*query*. We then present the definition of training prompts. Definition 1. (Training prompts) Given a task f ∈ T with µa and µb as the two different decisive patterns, a training prompt P ∼ D with ltr *context examples is constructed as follows.* - xquery follows the second line of (6) with j equally selected from {a, b} *and contains no* v
∗s
.

- Each xi contains µa or µb with equal probability i ∈ [ltr]*, following (6).*
- yi = +1 (or yi = −1) if the relevant pattern of xiis µa (or µb), and xi *does not contain any* v
∗
s.

yiis selected from {+1, −1} with equal probability if xi *contains a certain* v
∗
sfor s ∈ [V ].

When pa = 0, the setup reduces to the case where context examples contain no outliers, aligning with the theoretical setup in (Huang et al., 2023; Zhang et al., 2023; Li et al., 2024a). We include outliers in the training prompt to encourage the model to learn to ignore examples containing outliers. This improves robustness during inference when prompts may also include such outliers. Our motivation stems from noise-aware training to mitigate data poisoning or hijacking attacks in ICL (Wan et al., 2023; He et al., 2024; Qiang et al., 2023), where prompts are corrupted with noisy or random labels. Inference Stage: During inference, we consider that the outliers in the testing prompt can differ from those in the training prompt in several ways, including their direction, magnitude, and the fraction of examples affected. Specifically, the data input during the testing follow

$$\mathbf{x}={\begin{cases}\mathbf{\mu}_{j}+\kappa^{\prime}\mathbf{\nu}_{k}+\kappa_{a}^{\prime}\mathbf{v}_{s}^{*\prime},&{\text{with a probability of}\alpha}\\ \mathbf{\mu}_{j}+\kappa^{\prime}\mathbf{\nu}_{k},&{\text{with a probability of}1-\alpha,}\end{cases}}$$

for some v
∗ s
′ ∈ V′, κ
′a > 0, and κ
′ ∼ U(−K′, K′) with K′ > 1. α ∈ [0, 1) is the probability of examples containing the testing additive outliers in V
′.

Definition 2. (Testing prompts) Given a task f ∈ T with µa and µb as the relevant patterns, a testing P
′ ∼ D′ with lts context examples is constructed as follows. each testing query xquery only follows the second line of (7) without outliers. Each context input xi, i ∈ [lts], follows (7). If xi *does* not contain any v
∗s ∈ V′*, then* yi = +1 (or yi = −1) if the relevant pattern of xiis µa (or µb*). If* xi contains a certain v
∗
s ∈ V′, then yi *can be an arbitrary function that maps* xito {+1, −1}.

$$\left(7\right)$$

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

$$\mathcal{V}^{\prime}=\Big{\{}v\Big{|}v=\sum_{i=1}^{V}\lambda_{i}v_{i}^{*}+u,\sum_{i=1}^{V}\lambda_{i}\geq L>0,u\perp\{v_{r}^{*}\}_{r=1}^{V}\cup\{\mu_{i}\}_{j=1}^{M_{1}}\cup\{\nu_{k}\}_{k=1}^{M_{2}}\Big{\}},$$
k=1o, (11)
$(\downarrow\downarrow)$? 
6

To enable the model learned from data in training tasks Ttr to generalize well across all tasks in T , we require Condition 3.2 from (Li et al., 2024a) for Ttr. We restate this condition as Condition 1,
along with a construction of a training task set that satisfies it in the Appendix. The high-level idea is
that the training tasks Ttr should uniformly cover all of the relevant patterns and labels appearing in T such that no bias from the training tasks is introduced to the learning process. Following (Shi et al., 2021; Li et al., 2023a), we assume the training labels are balanced, i.e.,
|{n : z
n = +1*}| − |{*n : z
n = −1}| = O(
√N). Let BT := max{ϵ
−2, M1(1 − pa)
−1} · log ϵ
−1.
We have the following result.
Theorem 1. (Convergence and Sample Complexity of Mamba) For any ϵ > 0*, of (i)* B ≳ BM :=
max{BT , β−4V
2κ
−2
a(1 − pa)
−2log ϵ
−1}, (ii) V β−4 ≲ κa ≲ V β(1 − pa)p
−1
aϵ
−1*, and (iii)*
p
−1
a *poly*(Mκa
1) ≳ ltr ≳ (1 − pa)
−1log M1, (8)
then (iv) afterT ≥ TM = Θ(η
$$T\geq T_{M}=\Theta(\eta^{-1}(1-p_{a})^{-1}\beta^{-2}M_{1})$$
−2M1) (9)
iterations with η ≤ 1 and using N = BT *samples, we have that*
$\left(\mathfrak{g}\right)$. 
$$\mathbb{E}_{f\in{\mathcal{T}},\mathbf{P}\sim{\mathcal{D}}}[\ell(\Psi^{(T)};\mathbf{P},z)]\leq\epsilon.$$
Ef∈T ,P ∼D[ℓ(Ψ(T); P , z)] ≤ ϵ. (10)
Remark 1. Theorem 1 provides the convergence and sample complexity analysis of training a one-layer Mamba model to enhance its ICL ability. We characterize the sufficient conditions on the batch size, the magnitude of additive outliers, the prompt length, and the required number of iterations. The convergent model has desirable generalization on all tasks in T , including those not appearing in the training data, when the prompt is constructed in the same way as the training data. Condition (ii) requires that the magnitude of outliers be moderate and scale with V . This ensures that outliers are neither too small to be easily detectable by the model nor excessively large (i.e., less than Θ(ϵ
−1)), which would diminish the influence of relevant patterns. Conditions (iii) and (iv) show that the required number of context examples in the prompt and the number of iterations scale as
(1 − pa)
−1. This implies a higher fraction of outlier-containing context examples slows convergence and requires more context examples. The proof sketch of Theorem 1 can be found in Appendix A.

Remark 2. **(Comparison with existing works)** When pa = 0, Theorem 1 corresponds to the case where Mamba is trained with prompts that contain no outliers and serves as the Mamba counterpart to Theorem 3.3 in (Li et al., 2024a), which addresses Transformers. Although (Huang et al., 2023; Li et al., 2024a) analyze ICL training without outliers for Transformers, their analyses do not directly extend to Mamba due to the significant structural differences between the two architectures. To the best of our knowledge, we are the first to analyze the training dynamics of Mamba in the ICL setting, under a more general scenario where prompts may contain outliers. We then study the generalization performance on testing prompts with distribution-shifted additive outliers using the trained Mamba. Theorem 2. (ICL Generalization on Distribution-shifted Prompts with Outliers) During the inference, if (a) the outlier pattern v
∗ s
′*belongs to* 3.3 LEARNING, GENERALIZATION, AND SAMPLE COMPLEXITY ANALYSIS OF MAMBA The testing prompt P
′ differs from the training prompt P in two key aspects. First, the outlier patterns, the magnitude of the outliers, and the magnitude of the irrelevant patterns can differ from those in P . While the training prompts include V distinct outlier patterns, the testing prompts may contain an unbounded number of outlier variations. Second, the labels associated with examples containing outliers can be generated by any deterministic or probabilistic function. This flexibility allows our framework to model a wide range of noisy testing prompts in practice. For instance, Example 1. Consider a data poisoning attack on a text sentiment classification task in (Wan et al., 2023; He et al., 2024). In one such attack as shown in Figure 1, whenever the phrase "James Bond" is inserted into the example, the label is always set to positive, regardless of the original sentiment of the input. This illustrates a case where all examples containing the outlier are deterministically mapped to a targeted label +1.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

(b) the outlier magnitude κ
′a ∈ [κa, Θ(*V βp*a
−1κ
−1
a L
−1(1 − pa)ϵ
−1)], (c) α < min(1, paltr/lts),
and (d) the number of context examples
$$\alpha^{-1}p o y(M_{1}^{\kappa a})\gtrsim l_{t s}\geq(1-\alpha)^{-1}\log M_{1},$$
−1log M1, (12)
then for testing prompt P
′ defined by Definition 2, the trained model Ψ(T)*satisfies*

$$(12)$$
_d by Definition 2, the trained model $L^{0-1}_{f\in\mathcal{T},\mathbf{P}^{\prime}\sim\mathcal{D}^{\prime}}(\Psi^{(T)};\mathbf{P}^{\prime},z)\leq\epsilon$._
$$(13)$$
′, z) ≤ ϵ. (13)
Remark 3. Theorem 2 shows that the model trained under Theorem 1 generalizes well and remains robust when tested on prompts containing a signification fraction of unseen distribution-shifted outliers. Each additive outlier in the test prompt should contain a linear combination of the V training outlier patterns, with coefficients summing to a positive value (Condition (a)). This formulation captures a wide range of possible outlier patterns at test time. Notably, the fraction of examples with outliers α in the test prompt is less than min(1, paltr/lts), which can be close to 1 if the prompt length is selected in a way such that paltr/lts ≥ 1 (Condition (c)). Thus, Mamba can be trained to maintain ICL generalization in the presence of a large fraction of outlier examples. Conditions (b) and (d) impose mild requirements on the outlier magnitude and the context length, respectively. Condition (b) requires that the magnitude of test-time outliers be at least as large as that of the training outliers. Condition (d) ensures that the context prompt is sufficiently long to include enough clean examples for correct prediction, while also imposing an upper bound on the total number of outliers.

3.4 A THEORETICAL COMPARISON BETWEEN ONE-LAYER SINGLE-HEAD LINEAR
TRANSFORMERS AND MAMBA MODELS
In this section, we compare Mamba with linear Transformer with one layer and a single head, where
the Transformer model is formulated by setting the nonlinear gating function Gi,l+1(w) = 1 in (3) for i ∈ [l + 1], as discussed in Section 2. The comparison is made between sufficient conditions for the desired generalization. This is a common practice used in existing works (Fu et al., 2023b; Jiang et al., 2024) for neural network analysis. The provided upper bounds are aligned with our
experimental results in Section 4.1 for comparing robustness.
Theorem 3. (Convergence and Sample Complexity for Transformer Models) As long as (i) B ≳ BT ,
(ii) κa ≲ V β(1 − pa)p
−1
aϵ
−1*, (iii)* ltr ≳ (1 − pa)
−1log M1*, then (iv) after*
$$\begin{array}{l}{{T\geq T_{T}=\Theta(\eta^{-1}(1-p_{a})^{-1}\beta^{-2}l_{t r}^{-1}M_{1})}}\end{array}$$
tr M1) (14)

$\mathfrak{U}(T)\colon\boldsymbol{P}\geq)$]. 
iterations with η ≤ 1 and N = BT samples, we have that Ef∈T ,P ∼D[ℓ(Ψ(T); P , z)] ≤ ϵ.
Remark 4. Theorem 3 characterizes the sufficient conditions for the convergence and generalization of training a one-layer single-head Transformer with linear attention using prompts containing outliers as formulated by Definition 1. Comparing conditions (i)-(iv) with those in Theorem 1 on Mamba models, one can see that, to achieve a ϵ generalization error, linear Transformers need a smaller batch size, a smaller number of training iterations, and a less restrictive requirement for the prompt length and the magnitude of additive outliers. To see this, Theorem 1 indicates that the required batch size for Mamba models is at least BM, which is defined as the larger of value BT and another constant, while the required batch size for linear Transformers is BT . The required number of training iterations for
Mamba is TM, which equals Θ(ltr) · TT , and that is larger than that for linear Transformers, TT , by
a scaling of Θ(ltr) > 1. The required conditions for κa for linear Transformers does not include a lower bound, and the upper bound is larger than that of Mamba models when ϵ is small enough.
Moreover, Mamba requires an ltr that shares the same lower bound as that of the linear Transformers,
but it does not require an upper bound. Theorem 4. *(Generalization using Transformers) During the inference, if (a) in Theorem 2, (b)* κ
′a ≤ Θ(*V βp*a
−1(1 − pa)κ
−1
a L
−1ltrϵ
−1), (c) α ∈ [0, 1/2)*, and (d) the number of context examples*
$$l_{t s}\gtrsim\operatorname*{max}\{\Theta((1-\alpha)^{-1}),\Theta((1/2-\alpha)^{-2}\alpha)\}\log M_{1},$$
−2α)} log M1, (15)
then the trained model Ψ(T)*satisfies* L
0−1 f∈T ,P ′∼D′ (Ψ(T); P
′, z) ≤ ϵ.

Remark 5. Theorem 4 establishes the conditions under which a one-layer single-head Transformer model, trained according to Theorem 3, can generalize effectively on testing prompts with possible outliers, as defined in Definition 2. In contrast to Theorem 2 for Mamba, the Transformer guarantees generalization only when the outlier fraction satisfies α < 1/2, whereas Mamba can remain

$$(14)$$

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 robust when α goes to 1 (Condition (c)). This highlights that Mamba achieves better in-context generalization performance in the presence of distribution-shifted additive outliers, particularly when outlier-containing context examples are in the majority. This conclusion is consistent with the empirical findings of (Park et al., 2024), which observed that Mamba outperforms Transformers in many-outlier regression tasks. Remark 6. We would like to clarify that our theoretical comparison between Mamba and the linear Transformer is conducted under the one-layer, single-head setting, and both models are trained on prompts that contain outliers. Such an analysis is conducted to rigorously probe how the nonlinear gating affects model training, in-context generalization, and robustness, as the gating is the only difference between the two architectures. Large Transformer models, with appropriate training methods and ICL prompt design, can indeed achieve favorable robustness (Wan et al., 2023; He et al., 2024) against outliers. We include additional experiments and discussion about multi-head attention and softmax attention in Appendix B.1.

## 3.5 The Mechanism Of Mamba In Implementing Icl

We next examine the mechanism by which the trained Mamba model from Theorem 1 performs ICL on prompts containing additive outliers. This analysis provides deeper insights into the differences between Mamba and Transformer models. We begin by showing, in Corollary 1, that the linear attention of the learned Mamba model assigns greater weight to context examples that share the same relevant pattern as the query.

Corollary 1. Let N1 ⊆ [lts] denote the index sets of context examples that share the same relevant pattern as the query xquery. Then, for the model trained by Theorem 1 after T ≥ TM iterations in
(9), we have with a high probability, for P
′ 
X
defined by Definition 2, i∈N1 p˜
⊤
i W(T)
B
⊤W(T)
C p˜*query* ≥ Θ(1); X
i∈[lts]\N1 p˜
⊤
i W(T)
B
⊤W(T)
C p˜*query* ≤ Θ((1 − pa)
−1ϵ). (16)
Remark 7. Corollary 1 illustrates that for the testing prompt P
′, the learned Mamba model will let the attention scores be concentrated on examples with the same relevant pattern as the query, i.e., the sum of these attention scores will increase to be larger than Θ(1), while the sum of attention score on examples with other different relevant pattern from the query is upper bounded by a small order of (1 − pa)
−1ϵ. This enforces the model to focus on examples with the same relevant pattern as the query when making the prediction. Corollary 1 reveals an insight similar to the "induction head" mechanism (Olsson et al., 2022; Chan et al., 2022; Reddy, 2024) observed in softmax attention layers for ICL. However, our result is established in the context of linear attention, suggesting that different attention variants may share fundamentally similar internal mechanisms. We then show that the nonlinear gating mechanism in Mamba models enables ICL by effectively ignoring context examples containing outliers and focusing on those that are closer to the query. Corollary 2. **(i) Gating suppresses outlier examples.** *For the trained model by Theorem 1 after* T ≥ TM iterations in (9), we have that with a high probability, for p˜i*that contain a* v
∗s
′ ∈ V′,
Gi,lts+1(w(T)) ≤ O(*poly*(M1)
−1). (17)
(ii) Gating induces local bias. Denote h(j) ∈ [lts] (j ≤ lts) as the index of context example that is the j*-th closest to the query and does not contain any* v
∗s
′ ∈ V′*. Then, with a high probability,*
Gh(j),lts+1(w(T)) ≥ Θ(1/2 j−1). (18)
Remark 8. Corollary 2 indicates that the nonlinear gating Gi,lts+1(w(T)) serves two main purposes:
(i) filtering out examples containing additive outliers and (ii) inducing a local bias, as observed in (Han et al., 2024), that focuses on examples near the query. Specifically, (17) unveils that on examples with outliers, Gi,lts+1(w(T)) is close to 0, effectively suppressing their influence. (18) shows that for clean examples, the nonlinear gating values decay exponentially with the distance (in index) from the query. Hence, combing Corollaries 1 and 2, one can see that the model primarily relies on examples that are close to the query, do not contain outliers, and share the same relevant pattern as the query for prediction, resulting in desirable ICL performance even in the presence of outliers. Corollary 2 characterizes the role of the nonlinear gating layer, Mamba's key structural difference from the Transformer. This distinction explains their performance gap: while nonlinear gating makes Mamba more challenging to optimize, it also enables Mamba to suppress outlier-containing examples more effectively, resulting in superior robustness when handling prompts with many outliers.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 4 Experiment

We generate synthetic data following Section 3.22. Let d = 30, M1 = 6, M2 = 10, V = 3. For generalization with unseen outliers, let v
∗ 1
′ = 0.7v
∗
1 + 0.6v
∗
2 − 0.4v
∗
3, v
∗ 2
′ = 0.4v
∗
1 + 0.7v
∗
2 − 0.6v
∗
3, v
∗ 3
′ = −0.7v
∗
1 + 0.5v
∗
2 + 0.5v
∗
3, with L = 0.3. lts = ltr = 20. Let δ = 0.2, β = 3, κa = 2.

4.1 COMPARISON BETWEEN ONE-LAYER MAMBA AND LINEAR TRANSFORMER MODELS ON
ICL WITH OUTLIERS
The learning model is a one-layer Mamba defined in (3) and a one-layer single-head Transformer by making Gi,l+1(w) = 1 for i ∈ [l + 1]. We set pa = 0.6. We consider three types of outlier-relevant labeling functions during inference. If the context examples in a given prompt P′contains any additive outlier, the corresponding context label will be (A) flipped, (B) mapping to one targeted label out of {+1, −1}, or (C) randomly chosen from {+1, −1} with equal probability. Figure 2 shows that under three different forms of outliers, the classification error of Mamba is smaller than 0.01 even when α is close to 0.8. In contrast, the classification error of linear Transformers is large as long as α > 1/2. This is consistent with Remark 5: the one-layer single-head linear attention can tolerate at most a 1/2 fraction of outliers in the prompt, whereas Mamba can tolerate a fraction of outliers close to that seen during training, which can be close to 1.

![8_image_0.png](8_image_0.png)

![8_image_2.png](8_image_2.png)

![8_image_1.png](8_image_1.png)

| Mamba   | LT     |        |
|---------|--------|--------|
| FQ      | 99.73% | 93.68% |
| R       | 99.67% | 94.12% |
| CQ      | 82.73% | 93.96% |

Table 1: ICL accuracy of 3-layer Mamba and linear Transformers (LT) with different example placements. Mamba performs better than linear Transformers if outliers are FQ or R, but exhibits a significant performance drop in the CQ setting.

The learning model is a three-layer Mamba and a three-layer single-head linear Transformer. pa = 0.4.

Figure 3 shows the first-layer attention scores in the testing prompt. The sum of attention scores on the examples that share the same pattern as the query is significantly larger than that on examples with other patterns, and this gap increases during training. This verifies Corollary 1. Figure 4 shows that the first-layer gating values with α = 0.3 of outlier-containing examples are very small (red bars), while those of clean examples are relatively large and exhibit an approximately exponential decay with increasing distance from the query (green bars). This is consistent with (17) and (18) in Corollary 2. The results of attention scores and gating values in the other two layers exhibit the same trend as the first layer and are shown in Section B in Appendix due to the space limit. Next, we study the impact of the positions of context examples with α = 0.5. Table 1 presents the ICL
performance under three different placements of outlier examples: all positioned farthest from the query (FQ), closest to the query (CQ), or at random positions (R). We find that Mamba's performance in the scenario of FQ and R placements is clearly better than that of the linear Transformer. However, Mamba is highly sensitive to the position of outliers, whereas the linear Transformer (LT) is much 2Additional synthetic and real-world data experiments can be found in Appendices B.1 and B.2.

## 5 Conclusion, Limitations, And Future Works

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 less affected. This is because, when outliers are placed close to the query, the clean examples that share the same pattern as the query are pushed farther away, and the gating values on these examples decay exponentially according to (18), thereby degrading ICL performance, which is aligned with the empirical findings in (Wang et al., 2025). This paper theoretically studies the learning dynamics, ICL generalization, and the robustness to outliers of Mamba models, together with a characterization of how different components of Mamba contribute to the ICL mechanism. Our analysis also provides a theoretical comparison between Mamba and linear Transformer models. Although based on a one-layer Mamba structure on binary classification tasks, this work provides a deeper theoretical understanding and provable advantages of Mamba. Future directions include designing general Mamba-based language/multi-modal models.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Kwangjun Ahn, Xiang Cheng, Hadi Daneshmand, and Suvrit Sra. Transformers learn to implement preconditioned gradient descent for in-context learning. *arXiv preprint arXiv:2306.00297*, 2023.

Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In *The Eleventh International* Conference on Learning Representations, 2023.

Ameen Ali, Itamar Zimerman, and Lior Wolf. The hidden attention of mamba models. *arXiv preprint* arXiv:2403.01590, 2024.

Usman Anwar, Johannes Von Oswald, Louis Kirsch, David Krueger, and Spencer Frei. Adversarial robustness of in-context learning in transformers for linear regression. arXiv preprint arXiv:2411.05189, 2024.

Simran Arora, Sabri Eyuboglu, Michael Zhang, Aman Timalsina, Silas Alberti, James Zou, Atri Rudra, and Christopher Re. Simple linear attention language models balance the recall-throughput tradeoff. In *International Conference on Machine Learning*, pp. 1763–1840. PMLR, 2024.

Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, and Song Mei. Transformers as statisticians:
Provable in-context learning with in-context algorithm selection. *arXiv preprint arXiv:2306.04637*, 2023.

Marco Bondaschi, Nived Rajaraman, Xiuying Wei, Kannan Ramchandran, Razvan Pascanu, Caglar Gulcehre, Michael Gastpar, and Ashok Vardhan Makkuva. From markov to laplace: How mamba in-context learns markov chains. *arXiv preprint arXiv:2502.10178*, 2025.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in Neural Information Processing Systems*, 33:1877–1901, 2020.

Stephanie Chan, Adam Santoro, Andrew Lampinen, Jane Wang, Aaditya Singh, Pierre Richemond, James McClelland, and Felix Hill. Data distributional properties drive emergent in-context learning in transformers. *Advances in neural information processing systems*, 35:18878–18891, 2022.

Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Training dynamics of multi-head softmax attention for in-context learning: Emergence, convergence, and optimality. arXiv preprint arXiv:2402.19442, 2024a.

Siyu Chen, Heejune Sheen, Tianhao Wang, and Zhuoran Yang. Unveiling induction heads: Provable training dynamics and feature learning in transformers. Advances in Neural Information Processing Systems, 37:66479–66567, 2024b.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Yanda Chen, Chen Zhao, Zhou Yu, Kathleen McKeown, and He He. Parallel structures in pre-training data yield in-context learning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 8582–8592, 2024c.

Tri Dao and Albert Gu. Transformers are ssms: generalized models and efficient algorithms through structured state space duality. In Proceedings of the 41st International Conference on Machine Learning, pp. 10041–10071, 2024.

Daniel Y Fu, Tri Dao, Khaled Kamal Saab, Armin W Thomas, Atri Rudra, and Christopher Re.

Hungry hungry hippos: Towards language modeling with state space models. In *The Eleventh* International Conference on Learning Representations, 2023a.

Hengyu Fu, Tianyu Guo, Yu Bai, and Song Mei. What can a single attention layer learn? a study through the random features lens. *Advances in Neural Information Processing Systems*, 36:
11912–11951, 2023b.

Shivam Garg, Dimitris Tsipras, Percy S Liang, and Gregory Valiant. What can transformers learn in-context? a case study of simple function classes. Advances in Neural Information Processing Systems, 35:30583–30598, 2022.

Riccardo Grazzi, Julien Niklas Siems, Simon Schrodi, Thomas Brox, and Frank Hutter. Is mamba capable of in-context learning? In *International Conference on Automated Machine Learning*, pp.

1–1. PMLR, 2024.

Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

Albert Gu, Isys Johnson, Karan Goel, Khaled Saab, Tri Dao, Atri Rudra, and Christopher Ré.

Combining recurrent, convolutional, and continuous-time models with linear state space layers.

Advances in neural information processing systems, 34:572–585, 2021.

Albert Gu, Karan Goel, and Christopher Re. Efficiently modeling long sequences with structured state spaces. In *International Conference on Learning Representations*, 2022.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. *arXiv preprint arXiv:2501.12948*, 2025.

John T Halloran, Manbir Gulati, and Paul F Roysdon. Mamba state-space models can be strong downstream learners. *arXiv e-prints*, pp. arXiv–2406, 2024.

Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, and Gao Huang. Demystify mamba in vision: A linear attention perspective. *arXiv* preprint arXiv:2405.16605, 2024.

Ali Hatamizadeh and Jan Kautz. Mambavision: A hybrid mamba-transformer vision backbone. arXiv preprint arXiv:2407.08083, 2024.

Jianliang He, Xintian Pan, Siyu Chen, and Zhuoran Yang. In-context linear regression demystified:
Training dynamics and mechanistic interpretability of multi-head softmax attention. arXiv preprint arXiv:2503.12734, 2025.

Pengfei He, Han Xu, Yue Xing, Hui Liu, Makoto Yamada, and Jiliang Tang. Data poisoning for in-context learning. *arXiv preprint arXiv:2402.02160*, 2024.

Ruiquan Huang, Yingbin Liang, and Jing Yang. Non-asymptotic convergence of training transformers for next-token prediction. *Advances in Neural Information Processing Systems*, 37:80634–80673, 2024.

Yu Huang, Yuan Cheng, and Yingbin Liang. In-context convergence of transformers. In *NeurIPS*
2023 Workshop on Mathematics of Modern Machine Learning, 2023.

Samy Jelassi, Michael Sander, and Yuanzhi Li. Vision transformers provably learn spatial structure.

Advances in Neural Information Processing Systems, 35:37822–37836, 2022.

Samy Jelassi, David Brandfonbrener, Sham M Kakade, et al. Repeat after me: Transformers are better than state space models at copying. In Forty-first International Conference on Machine Learning, 2024.

Jiarui Jiang, Wei Huang, Miao Zhang, Taiji Suzuki, and Liqiang Nie. Unveil benign overfitting for transformer in vision: Training dynamics, convergence, and generalization. Advances in Neural Information Processing Systems, 37:135464–135625, 2024.

Federico Arangath Joseph, Kilian Konstantin Haefeli, Noah Liniger, and Caglar Gulcehre. Hippoprophecy: State-space models can provably learn dynamical systems in context. arXiv preprint arXiv:2407.09375, 2024.

Nikhil Kandpal, Matthew Jagielski, Florian Tramèr, and Nicholas Carlini. Backdoor attacks for incontext learning with language models. In The Second Workshop on New Frontiers in Adversarial Machine Learning, 2023.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Juno Kim and Taiji Suzuki. Transformers learn nonlinear features in context: Nonconvex mean-field dynamics on the attention landscape. In *International Conference on Machine Learning*, pp. 24527–24561. PMLR, 2024.

Barak Lenz, Opher Lieber, Alan Arazi, Amir Bergman, Avshalom Manevich, Barak Peleg, Ben Aviram, Chen Almagor, Clara Fridman, Dan Padnos, et al. Jamba: Hybrid transformer-mamba language models. In *The Thirteenth International Conference on Learning Representations*, 2025.

Hongkang Li, Meng Wang, Sijia Liu, and Pin-Yu Chen. A theoretical understanding of shallow vision transformers: Learning, generalization, and sample complexity. In The Eleventh International Conference on Learning Representations, 2023a. URL https://openreview.net/forum? id=jClGv3Qjhb.

Hongkang Li, Meng Wang, Tengfei Ma, Sijia Liu, ZAIXI ZHANG, and Pin-Yu Chen. What improves the generalization of graph transformer? a theoretical dive into self-attention and positional encoding. In *NeurIPS 2023 Workshop: New Frontiers in Graph Learning*, 2023b. URL https://openreview.net/forum?id=BaxFC3z9R6.

Hongkang Li, Meng Wang, Songtao Lu, Xiaodong Cui, and Pin-Yu Chen. How do nonlinear transformers learn and generalize in in-context learning? In Forty-first International Conference on Machine Learning, 2024a. URL https://openreview.net/forum?id=I4HTPws9P6.

Hongkang Li, Yihua Zhang, Shuai Zhang, Meng Wang, Sijia Liu, and Pin-Yu Chen. When is task vector provably effective for model editing? a generalization analysis of nonlinear transformers. arXiv preprint arXiv:2504.10957, 2025a.

Yingcong Li, Ankit S Rawat, and Samet Oymak. Fine-grained analysis of in-context linear estimation:
Data, architecture, and beyond. *Advances in Neural Information Processing Systems*, 37:138324–
138364, 2024b.

Yingcong Li, Davoud Ataee Tarzanagh, Ankit Singh Rawat, Maryam Fazel, and Samet Oymak.

Gating is weighting: Understanding gated linear attention through in-context learning. *arXiv* preprint arXiv:2504.04308, 2025b.

Mehryar Mohri, Afshin Rostamizadeh, and Ameet Talwalkar. *Foundations of machine learning*. MIT
press, 2018.

Eshaan Nichani, Jason D. Lee, and Alberto Bietti. Understanding factual recall in transformers via associative memories. In *The Thirteenth International Conference on Learning Representations*, 2025. URL https://openreview.net/forum?id=hwSmPOAmhk.

Jiachang Liu, Dinghan Shen, Yizhe Zhang, William B Dolan, Lawrence Carin, and Weizhu Chen.

What makes good in-context examples for gpt-3? In Proceedings of Deep Learning Inside Out (DeeLIO 2022): The 3rd Workshop on Knowledge Extraction and Integration for Deep Learning Architectures, pp. 100–114, 2022.