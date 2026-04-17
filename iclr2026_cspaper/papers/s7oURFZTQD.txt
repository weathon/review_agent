000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Multi-grade deep learning (MGDL) has recently emerged as an alternative to standard end-to-end training, referred to here as single-grade deep learning (SGDL),
showing strong empirical promise. This work provides both theoretical and experimental evidence of MGDL's computational advantages. We establish convergence guarantees for gradient descent (GD) applied to MGDL, demonstrating greater robustness to learning-rate choices compared to SGDL. In the case of ReLU activations with single-layer grades, we further show that MGDL reduces to a sequence of convex optimization subproblems. For more general settings, we analyze the eigenvalue distributions of Jacobian matrices from GD iterations, revealing structural properties underlying MGDL's enhanced stability. Practically, we benchmark MGDL against SGDL on image regression, denoising, and deblurring tasks, as well as on CIFAR-10 and CIFAR-100, covering fully connected networks, CNNs, and transformers. These results establish MGDL as a scalable framework that unites rigorous theoretical guarantees with broad empirical improvements.

## 1 Introduction

Deep learning has transformed fields from computer vision He et al. (2016); Krizhevsky et al. (2012) to medicine Chen et al. (2018); Jumper et al. (2021) and scientific computing Raissi et al. (2019). Despite these successes, training deep neural networks (DNNs) remains challenging due to nonconvex optimization, vanishing/exploding gradients, and spectral bias that favors low-frequency features Rahaman et al. (2019); Xu et al. (2019). Gradient descent can also exhibit short-term oscillations near the Edge of Stability Arora et al. (2022); Cohen et al. (2021), making conventional training inefficient, hard to interpret, and limited in generalization. These challenges motivate multigrade deep learning (MGDL) Xu (2025), which incrementally builds networks to improve stability, accuracy, and interpretability. MGDL decomposes end-to-end optimization into a sequence of smaller problems, each training a shallow network on the residuals of previous grades. Previously learned networks remain fixed and act as adaptive "basis" functions or features. This iterative refinement reduces optimization complexity and progressively enhances learning. MGDL has demonstrated superior performance over standard end-to-end training, which we refer to here as single-grade deep learning (SGDL), in regression Fang & Xu (2024); Xu (2023), oscillatory Fredholm integral equations Jiang & Xu (2024), and PDEs Xu & Zeng (2023), effectively mitigating spectral bias. We provide a mathematical explanation for why MGDL outperforms SGDL. Focusing on gradient descent, we establish convergence theorems showing MGDL's greater robustness to learning-rate choices. When each grade uses a single ReLU layer, MGDL reduces a highly nonconvex problem to a sequence of convex subproblems, enhancing trainability. Further analysis of a linear surrogate iterative scheme based on the Jacobian of the original map shows that MGDL's eigenvalues lie within
(−1, 1), ensuring stable convergence, whereas SGDL's can exceed this range, causing oscillatory loss. Additional experiments benchmark MGDL against SGDL on image regression, denoising, and deblurring tasks, as well as CIFAR-10 and CIFAR-100 classification, using fully connected networks, CNNs, and transformers. These results demonstrate that MGDL unifies rigorous theoretical guarantees with broad empirical improvements as a scalable framework.

Anonymous authors Paper under double-blind review

## Abstract

# Why Multi-Grade Deep Learning Outperforms Single-Grade: Theory And Practice

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Key contributions of this paper: 1. We provide a rigorous convergence analysis of gradient descent for SGDL and MGDL, offering deeper insight into MGDL's computational advantages. 2. We prove that if each grade of MGDL employs a single hidden ReLU layer, the originally nonconvex optimization problem decomposes into a sequence of convex subproblems. 3. Extensive experiments on image regression, denoising, deblurring, CIFAR-10, and CIFAR- 100 classification, including fully connected networks, CNNs, and transformers, demonstrate that MGDL consistently outperforms SGDL with greater stability. 4. We analyze the impact of learning rate, showing that MGDL is more robust than SGDL. 5. We study a linear approximation of GD dynamics and the eigenvalue distribution of the associated Jacobian to explain MGDL's convergence and stability advantages.

## 2 Standard Deep Learning Model

In this section, we review the standard deep learning model and analyze the convergence of the gradient descent (GD) applied to its optimization problem.

A deep neural network (DNN) is a composition of affine maps and nonlinear activations with input layer, D − 1 hidden layers, and an output layer. Let d0 = d (input dimension), dD = t (output dimension), and dj the width of layer j. For j = 1*, . . . , D*, the weights and biases are Wj ∈
R

dj−1×dj and bj ∈ R
dj, with ReLU activation σ(x) = max{0, x} applied componentwise.

Given x ∈ R
d, the hidden layers are defined recursively:
H1(x) := σW⊤
1 x + b1, Hj+1(x) := σW⊤
j+1Hj (x) + bj+1, j = 1*, . . . , D* − 2.

The output is ND{Wj , bj}
D
j=1; x= ND(x) := W⊤DHD−1 (x) + bD. For data D =
{(xn, yn)}
N
n=1, the loss is

$\pi$
$${\mathcal{L}}(\{\mathbf{W}_{j},\mathbf{b}_{j}\}_{j=1}^{D};\mathbb{D})={\frac{1}{2N}}\sum\nolimits_{n=1}^{N}\|\mathbf{y}_{n}-{\mathcal{N}}_{D}(\mathbf{x}_{n})\|^{2}.$$
$$(1)$$
2. (1)
The SGDL model minimizes this loss over parameters Θ = {Wj , bj}
D
j=1, yielding optimal Θ∗and trained network ND(Θ∗; ·).

Among the most common optimization methods for deep learning are stochastic gradient descent
(SGD) Kiefer & Wolfowitz (1952); Robbins & Monro (1951) and Adam Kingma & Ba (2015), both rooted in gradient descent (GD). We therefore study GD for minimizing the loss in equation 1.

To facilitate convergence analysis, we stack all parameters {Wj , bj}
D
j=1 into a single vector. For any matrix or vector A, let A denote its vectorization: stacking columns if A is a matrix, taking A = A if it is a column vector, and A = A⊤ if a row vector. The parameter vector is W := W⊤
1, b⊤
1, . . . , W⊤
D , b⊤D
⊤, with total dimension M =PD
j=1(dj−1 + 1)dj .

We consider GD for a general objective F : RM → R, assumed nonnegative, twice continuously differentiable, and generally nonconvex. The iteration is

$$W^{k+1}=W^{k}-\eta{\frac{\partial{\mathcal{F}}}{\partial W}}(W^{k})$$

∂W (Wk), (2)
where k is the iteration index and η > 0 the learning rate. In our setting, F is the loss L in equation 1. We analyze the convergence of GD for minimizing equation 1. Assume there exists a compact convex set W ⊂ RM such that for some η0 > 0, all GD iterates Wk∞
k=0 from equation 2 with F = L remain in W whenever η ∈ (0, η0). Convergence depends on the Hessian of L over W,
where we set α := supW∈W ∥HL(W)∥, with *∥ · ∥* the spectral norm. Since HL(W) ∈ RM×M, α captures the effect of network depth and size. The following theorem, proved in Appendix A, establishes convergence of GD with F := L, extending Theorem 6 in Xu (2025), which assumes zero biases.

![2_image_0.png](2_image_0.png)

(i) limk→∞ L(Wk) = L
∗*for some* L
∗ ≥ 0;
(ii) limk→∞
∂L
∂W (Wk) = 0;
(iii) Every cluster point Wˆ of {Wk} satisfies ∂L
∂W (Wˆ ) = 0.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 3 Multi-Grade Deep Learning

Deep neural networks are defined by weight matrices and bias vectors, with parameter counts scaling rapidly with depth—for example, LeNet-5 has 60K parameters LeCun et al. (1998), ResNet-152 60.2M He et al. (2016), and GPT-3 175B Brown et al. (2020). End-to-end training at such scales is hampered by optimization and stability issues: (i) deeper networks induce highly nonconvex loss landscapes, often trapping solutions in poor local minima Bengio et al. (2006); and (ii) training suffers from vanishing or exploding gradients, which impede convergence Glorot & Bengio (2010); Goodfellow et al. (2016); Pascanu et al. (2013). To address these challenges, multi-grade deep learning (MGDL) Xu (2025) trains networks in stages, where each shallow grade builds on the residuals of the previous one and propagates its output forward, incrementally approximating the target function.

Given data D = {(xn, yn)}
N
n=1, MGDL decomposes learning a depth-D DNN into *L < D* sequential grades. Each grade trains a shallow network NDlon residuals from the previous grade,
with depths 1 < Dl < D and PL
l=1 Dl = D + L − 1. Let Θl = {Wlj , blj}
Dl
j=1 denote grade-l
parameters. The model is defined recursively by
g1(Θ1; x) := ND1(Θ1; x), gl+1(Θl+1; x) := NDl+1 (Θl+1; ·)◦HDl−1(Θ∗
l; ·)*◦· · ·◦H*D1−1(Θ∗1; ·)(x). (3)
The grade-l loss is
$${\mathcal{L}}_{l}(\Theta_{l};\mathbb{D})={\frac{1}{2N}}\sum\nolimits_{n=1}^{N}\|\mathbf{e}_{l n}-g_{l}(\Theta_{l};\mathbf{x}_{n})\|^{2},$$
2, (4)
with residuals e1n = yn and e(l+1)n = eln − gl(Θ∗
l; xn). Each Θ∗
l minimizes Ll given earlier
grades. After L grades, the MGDL output is g¯L({Θ∗
l
}
L
l=1; x) = PL
l=1 gl(Θ∗
l
; x). Figure 1 illustrates
the multi-grade architecture at grade three.
For optimization, set x1n := xn and recursively define xln := HDl−1−1(Θ∗
l−1
; ·) ◦ · · · ◦
HD1−1(Θ∗1; ·)(xn), and dataset Dl = {(xln, eln)}
N
n=1. The grade-l loss is
$\mathbf{a}\cdot\mathbf{a}=\mathbf{a}\cdot\mathbf{a}$. 
$${\mathcal{L}}_{l}(\Theta_{l};\mathbb{D}_{l})={\frac{1}{2N}}\sum\nolimits_{n=1}^{N}\|\mathbf{e}_{l n}-{\mathcal{N}}_{D_{l}}(\Theta_{l};\mathbf{x}_{l n})\|^{2}.$$
Theorem 1. Let {Wk}∞
k=0 be generated by equation 2 with F = L and initial guess W0*. Suppose* σ is twice continuously differentiable and the iterates remain in a convex compact set W ⊂ RM*. If* the learning rate η ∈ (0, 2/α), then:

$$W_{l}^{k+1}=W_{l}^{k}-\eta\frac{\partial{\mathcal{L}}_{l}}{\partial W_{l}}(W_{l}^{k}).$$
$\eqref{eq:walpha}$. 
3 This section reviews MGDL, and analyzes GD convergence at each grade. MGDL's training time scales linearly with the number of grades (assuming comparable layer and neuron counts), while its memory cost is much lower than that of a single deep network, since each grade trains only a shallow model. Let Wl:= (W⊤
l1, b⊤
l1*, . . . , W*⊤
lDl
, b⊤
lDl
)
⊤ ∈ RMl, with Ml:= PDl j=1(dl(j−1) + 1)dlj . The GD iteration is 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Assuming Wk l ⊂ Wl for some convex compact Wl ⊂ RMl, define αl:= supWl∈Wl
∥HLl(Wl)∥.

Theorem 2. Let {Wk l} *be generated by the above GD iteration from* W0 l. Assume {(xln, eln)} ⊂
R

dl0 × R
dis bounded, σ *is twice continuously differentiable, and* {Wk l
} ⊂ Wl*. If* ηl ∈ (0, 2/αl),
then

_(i) $\lim_{k\to\infty}\mathcal{L}_{l}(W^{k}_{l})=L^{*}_{l}$ for some $L^{*}_{l}\geq0$;_
(iii) Every cluster point Wˆl of {Wk l
} satisfies ∂Ll
∂Wl
(Wˆl) = 0.

## 4 Convex Optimization In Mgdl With Single-Layer Relu Grades

In this section, we show that when each grade in MGDL is realized as a single hidden-layer ReLU network, the overall nonconvex optimization problem decomposes into a sequence of convex subproblems. For clarity, we consider bias-free networks with scalar output; the extension to biased networks is analogous. which we adopt as the building block of MGDL.

$$\mathcal{N}(\mathbf{x}):=\sum_{j=1}^{m}\big{(}\sigma(\mathbf{w}_{j}^{\top}\mathbf{x})-\sigma(\mathbf{v}_{j}^{\top}\mathbf{x})\big{)},\tag{5}$$

A two-layer ReLU network with m neurons is N˜ (x) := Pm j=1 αjσ(w˜
⊤
j x), with hidden parameters w˜ j and outputs αj . Since σ(ax) = aσ(x) for a ≥ 0, each term can be written as αjσ(w˜
⊤
j x) =
σ(w⊤
j x) − σ(v
⊤
j x) for suitable wj , vj , making N˜ equivalent to

Suppose grade l of MGDL is a single hidden-layer ReLU network with 2ml neurons. By equation 5,
its output is
its output is  $$(\mathcal{N}_{l}\circ h_{l-1}^{*})(\mathbf{x}):=\sum\nolimits_{j=1}^{m_{l}}\Big{(}\sigma(\mathbf{w}_{ij}^{\top}h_{l-1}^{*}(\mathbf{x}))-\sigma(\mathbf{v}_{ij}^{\top}h_{l-1}^{*}(\mathbf{x}))\Big{)}.\tag{6}$$  The input features $h_{l-1}^{*}$ are defined recursively by $h_{0}^{*}(\mathbf{x})=\mathbf{x}$, and $h_{l-1}^{*}(\mathbf{x}):=(\mathcal{H}_{l-1}^{*}\circ\cdots\circ\mathcal{H}_{1}^{*})(\mathbf{x})$, $\mathbf{x}$ is defined as 
with feature map
$$p_{l}^{*}:=\min_{\{\mathbf{w}_{ij},\mathbf{v}_{ij}\}_{j=1}^{m_{l}}}\frac{1}{2}\Big{\|}\sum\nolimits_{j=1}^{m_{l}}\left(\sigma(\mathbf{X}_{l}\mathbf{w}_{ij})-\sigma(\mathbf{X}_{l}\mathbf{v}_{ij})\right)-\mathbf{e}_{l}\Big{\|}^{2}.\tag{7}$$
  **Outlec the convex program**  $$q_{l}^{\star}:=\min_{\{\mathbf{w}_{li},\mathbf{v}_{li}\in C_{li}\}_{i=1}^{P_{1}}}\frac{1}{2}\Big{\|}\sum\nolimits_{i=1}^{P_{1}}\mathbf{D}_{li}\mathbf{X}_{l}(\mathbf{w}_{li}-\mathbf{v}_{li})-\mathbf{e}_{l}\Big{\|}^{2}.\tag{8}$$
(ii) limk→∞
$$\circ\frac{\partial{\mathcal{L}}_{l}}{\partial W_{l}}(W_{l}^{k})=0;$$
Theorem 2, proved in Appendix A, parallels Theorem 1, with the key distinction that MGDL optimizes shallow subproblems at each grade. This mitigates vanishing/exploding gradients and allows a broader admissible learning-rate range (ηl ∈ (0, 2/αl) with αl ≪ α), thereby improving stability and robustness compared to SGDL.

$\mathcal{H}_{k}^{*}(\mathbf{z})=\left(\sigma((\mathbf{w}_{k1}^{*})^{\top}\mathbf{z}),\ldots,\sigma((\mathbf{w}_{km_{k}}^{*})^{\top}\mathbf{z}),\sigma((\mathbf{v}_{k1}^{*})^{\top}\mathbf{z}),\ldots,\sigma((\mathbf{v}_{km_{k}}^{*})^{\top}\mathbf{z})\right)^{\top},\;\;k\in\mathbb{N}_{l-1}.$
Let the data matrix at grade l be Xl:= [xl1*, . . . ,* xlN ]
⊤ ∈ R
N×dl with xln := h
∗ l−1
(xn). At grade l, we solve the nonconvex problem Following Pilanci & Ergen (2020), we show that equation 7 is equivalent to a convex program.

For any wl ∈ R
ml−1, define diag(1[Xlwl ≥ 0]), where 1[Xlwl ≥ 0] ∈ {0, 1}
N with entries 1[x
⊤
lnwl ≥ 0]. Since Xlis fixed, only finitely many such matrices exist Cover (2006); Stanley et al.

(2007); denote them Dl1*, . . . ,* DlPl
. This induces a partition {Cli}
Pl i=1 of R
ml−1, where Cli := {wl:
(2Dli−IN )Xlwl ≥ 0}. Each Cli is convex, closed under addition, and satisfies R
ml−1 =SPl i=1 Cli.

Within Cli, ReLU is linear, that is, σ(Xlwl) = DliXlwl, for wl ∈ Cli.

Using this, we introduce the convex program Theorem 3. Let σ be ReLU. If ml ≥ Pl*, then problems equation 7 and equation 8 attain the same* optimal value. Moreover, any optimal solution of equation 8 is also optimal for equation 7 when ml = Pl.

Proof. Linearity within each region implies that feasible points of equation 8 are feasible for equation 7, hence p
∗
l ≤ q
∗ l
. Conversely, given an optimal solution {w∗
lj , v
∗
lj} of equation 7, regrouping parameters by the partition {Cli} and using closure under addition yields aggregated vectors w˜
∗
li, v˜
∗ li that form a feasible point of equation 8 with the same objective value, so q
∗
l ≤ p
∗
l. Thus p
∗
l = q
∗
l.

When ml = Pl, the correspondence is exact, and optimal solutions coincide.

Unlike Pilanci & Ergen (2020), which convexifies single hidden-layer ReLU networks via explicit regularization, our multi-grade decomposition reformulates deep ReLU networks as a sequence of convex programs, extending convexification from shallow to deep architectures.

## 5 Performance Comparison Of Mgdl And Sgdl

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 In this section, we compare MGDL and SGDL on image reconstruction tasks—regression, denoising, and deblurring—as well as on the CIFAR-100 classification dataset Krizhevsky (2009). The results demonstrate that MGDL consistently outperforms SGDL, which suffers from training instability and lower accuracy. For image reconstruction, we employ full connected networks for both SGDL and MGDL, and evaluate performance using PSNR equation 30. For classification, we use convolutional neural networks (CNNs). In both cases, ReLU activations are applied, and training is performed using the Adam optimizer Kingma & Ba (2015). Overall, MGDL achieves superior stability and accuracy across both reconstruction and classification tasks.

Image regression. We model grayscale images as functions f : R
2 → R, mapping pixel coordinates to intensity values. The training set consists of a regularly spaced grid covering one quarter of the pixels, while the test set includes all pixels. We evaluate SGDL and MGDL on six images of varying sizes (Figure 9). For images (b)–(f), we use the fully connected architecture in 26 with (nin, nout, nhidden, nh) = (2, 1, 128, 8) for SGDL and the architecture in 27 with (nin, nout, nhidden, nh, L) = (2, 1, 128, 2, 4) for MGDL. For image (g), we employ a deeper network, setting G = 12 for SGDL and g = 3 for MGDL.

Numerical results are summarized in Table 1 and Figure 11. Table 1 reports PSNR values, showing that MGDL consistently outperforms SGDL with gains of 0.42–3.94 dB across all testing images. Figure 11 plots the training losses: SGDL exhibits persistent oscillations for all images, while MGDL shows image-dependent behavior. For Barbara, Butterfly, and Walnut, MGDL oscillates initially but stabilizes in later stages, whereas for Pirate and Chest, oscillations appear earlier before converging. Overall, MGDL tends to stabilize or decrease steadily over time, in contrast to the sustained oscillations of SGDL. The *Cameraman* image further illustrates these differences. Figures 10(a)–(b) show the training losses: SGDL suffers from strong oscillations, leading to unstable predictions, as seen in Figures 10(c)–(f) at iterations 9800, 9850, 9900, and 9950, with corresponding PSNR fluctuations. In contrast, MGDL exhibits a steadily decreasing loss (b), and its predictions (g)–(j) improve consistently across iterations. These results highlight the robustness and reliability of MGDL compared with SGDL in image regression tasks.

Image denoising. We address the problem of recovering a clean image f ∈ R
n×n from a noisy observation ˆf := f + ϵ, where the noise entries are i.i.d. Gaussian with zero mean and standard deviation s, i.e., [ϵ]i,j ∼ N (0, s2). The optimization problem is formulated in Appendix B, with the transform operator A set to the identity. SGDL adopts structure 26 (2, 1, 128, 12), while MGDL uses 27 (2, 1, 128, 3, 4). We test six noise levels, s *= 10*, 20, 30, 40, 50, 60, as illustrated in Figure 12. Results are summarized in Table 2 and Figures 13-15. MGDL consistently outperforms SGDL with PSNR gains of 0.16–4.23 dB. During training, SGDL shows persistent oscillations, while MGDL improves steadily, especially from grades 2–4.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

Image Method TrPSNR TePSNR Cameraman SGDL 27.05 24.79

MGDL 31.80 25.21

Barbara SGDL 23.*14 22*.75

MGDL 24.*36 23*.84

Butterfly SGDL 26.22 24.87

MGDL 28.23 27.06

Pirate SGDL 24.20 24.34

MGDL 27.40 26.45

Chest SGDL 34.77 34.56

MGDL 39.44 38.50

Walnut SGDL 19.*94 20*.05

MGDL 21.*83 21*.31

Table 3: PSNR comparison for image deblurring.

image method 3 5 7 Butterfly SGDL 25*.43 24.20 22.*70

MGDL **27.06 25.19 23.65**

Pirate SGDL 24.72 23.79 23.13

MGDL **26.47 24.95 23.98**

Chest SGDL 35.40 34.61 33.69

MGDL **38.24 36.51 35.14** Figure 2: Impact of learning rate.

Image deblurring. We address the problem of recovering f from a blurred observation ˆf := Kf +ϵ, where K is a Gaussian blurring operator and [ϵ]i,j ∼ N (0, s2) with s = 3. The optimization problem and operator A = K are detailed in Appendix B. The SGDL and MGDL structures are the same as those used in *Image Denoising*, respectively. We test three blurring levels (sˆ = 3, sˆ = 5, and sˆ = 7; Figure 16). Results are summarized in Table 3 and Figures 17-19. MGDL achieves PSNR improvements of 0.85–2.84 dB over SGDL. While SGDL exhibits strong PSNR oscillations during training, MGDL shows stable and consistent gains, particularly from grades 2 to 4. Classification on CIFAR-100. We address the problem of image classification on the CIFAR-100 dataset, evaluating SGDL and MGDL in terms of both accuracy and training dynamics. We use mean squared error (MSE) as the loss function, with architectures specified in equation 28 and 29.

We test two learning rates, 5×10−4and 1×10−4. Results are shown in Figure 3. For both settings, SGDL converges to a loss around 10−2, whereas MGDL reaches approximately 10−4, nearly two orders of magnitude lower. In terms of stability, SGDL begins oscillating once the loss falls below 10−1, while MGDL remains stable until reaching 10−3. These results demonstrate that MGDL
delivers superior accuracy and significantly greater training stability compared to SGDL. Results on image reconstruction and CIFAR-100 classification show that MGDL consistently outperforms SGDL. Whereas SGDL exhibits pronounced oscillations in loss or PSNR during training, MGDL achieves a steady decrease in loss or a consistent increase in PSNR. The underlying reasons are analyzed in Section 7.

![5_image_1.png](5_image_1.png)

Noise Method Butterfly Pirate Chest

![5_image_0.png](5_image_0.png)

10 SGDL 27.53 25.13 36.20

MGDL 31.67 29.36 38.58

20 SGDL 26.73 25.*02 35*.34

MGDL 28.39 27.*74 36*.89

30 SGDL 26.05 24.63 34.30

MGDL 27.09 27.20 35.48

40 SGDL 25.54 24.*47 33*.55

MGDL 26.37 26.*25 34*.61

50 SGDL 24.65 24.01 33.51

MGDL 25.84 25.77 33.94

60 SGDL 24.30 23.*82 32*.90

MGDL 25.21 25.*32 33*.06

Table 1: PSNR comparison for image regression.

## 6 Impact Of Learning Rate On Sgdl And Mgdl

We examine the effect of learning rate on SGDL and MGDL, both trained using gradient descent.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Synthetic data regression. We approximate g : [0, 1] → R defined by g(x) := PM
j=1 sin (2πκjx + φj ), x ∈ [0, 1], where φj ∼ U(0, 2π). Two settings are considered: (1)
M = 3, κ *= [1*, 5.5, 10]; (2) M = 5, κ = [1, 8.25, 15.5, 22.75, 30]. The training set contains 1,024 equally spaced points, and the validation set 1,000 uniformly sampled points. SGDL adopts structure 26 (1, 1, 32, 4), while MGDL uses structure 27 (1, 1, 32, 1, 4). Learning rates are selected from [0.001, 0.5], with 106training epochs. Figure 2 illustrates the impact of learning rate (left: Setting 1, right: Setting 2; 'NaN' indicates divergence). In Setting 1 (low-frequency function), both methods perform well, while MGDL is robust across a wider range: SGDL achieves loss < 0.001 only for η ∈ [0.03, 0.08], whereas MGDL sustains this performance for η ∈ [0.01, 0.3]. In Setting 2 (high-frequency function), SGDL converges only at η ≈ 0.005 and diverges for larger rates, while MGDL remains stable with loss < 0.01 for η ∈ [0.08, 0.3]. Image regression. We consider image regression as in Section 5. SGDL use 26 (2, 1, 128, 8),
while MGDL uses 27 (2, 1, 128, 2, 4). Learning rates are selected from [0.001, 1], with 105training epochs. Figure 20 illustrates results on 'Resolution Chart', 'Cameraman', 'Barbara', and 'Pirate'. MGDL consistently achieves higher accuracy, while SGDL fails on 'Cameraman' and 'Pirate' for η near 1. MGDL remains stable across this wide range of learning rates. Summary. Across both synthetic and image regression, MGDL demonstrates markedly greater robustness to the choice of learning rate, maintaining effective training and high accuracy over a wider interval, whereas SGDL is sensitive and often fails with large learning rates.

## 7 Eigenvalue Analysis For Sgdl And Mgdl

We analyze gradient descent (GD) equation 2 for SGDL and MGDL, expressing it as a Picard iteration Wk+1 = (I − η
∂F
∂W )Wkand linearizing the gradient via Taylor expansion: ∂F
∂W (Wk) =
HF (Wk−1)Wk + u k−1 + r k−1, with remainder r k−1 of order (Wk − Wk−1)
2. Neglecting r k−1 gives the linearized update W˜ k+1 = Ak−1W˜ k − ηuk−1, Ak−1 = I − ηHF (Wk−1).

Theorem 4. Let F : RM → R *be nonnegative and twice continuously differentiable, with*
{Wk} ⊂ Ω*, a convex compact set. If* τ := supW∈Ω ∥I − ηHF (W)∥ < 1, then {W˜ k} *converges.*
Moreover, if F is thrice continuously differentiable, the sequences {Wk} and {W˜ k} (with matching initializations) converge to the same limit if τ < 1.

Hence, convergence is governed by the spectrum of I − ηHF (W). Eigenvalues in (−1, 1) ensure stable loss decay. Explicit Hessians for SGDL (F = L) and MGDL (F = Ll) under ReLU are given in the Supplementary Material.

We next monitor the eigenvalues of I−ηHF (Wk) during training. In deep networks such as SGDL,
these eigenvalues often exit (−1, 1), producing oscillatory loss. In contrast, the shallower structure of MGDL keeps them inside (−1, 1), leading to smooth loss decay.

Synthetic data regression. Setup follows *Synthetic data regression* in Section 6. Both models are trained via gradient descent with learning rate η ∈ [0.001, 0.5], selected by lowest validation loss.

Results are shown in Figures 4 (Setting 1) and 21 (Setting 2). For SGDL under Setting 1, Figure 4 (first subfigure) shows the ten smallest (solid) and ten largest
(dashed) eigenvalues during training (106epochs). The smallest eigenvalue drops well below −1, while indices 1–5 stay near −1. The largest eigenvalues slightly exceed 1. The loss decreases overall but oscillates, correlating with the number of eigenvalues below or near −1. For MGDL, the ten smallest eigenvalues remain within (−1, 1) across grades 1–4, while the largest stay slightly above 1, producing smooth loss decay (Figure 4, second and fourth subfigures). In Setting 2 (higher-frequency target), SGDL's eigenvalues initially stay in (−1, 1) but later drop to −1, causing strong loss oscillations up to 106epochs. MGDL maintains eigenvalues in (−1, 1),
Figure 4: Training process of SGDL (η = 0.08) and MGDL (η = 0.06) for Setting 1.

![7_image_2.png](7_image_2.png)

Image regression. Following Section 5, shallow networks are used to enable Hessian computation:
SGDL with architecture 26 (2, 1, 48, 4) and MGDL with architecture 27 (2, 1, 48, 1, 4). For SGDL, the smallest eigenvalue approaching −1 causes oscillatory loss, while MGDL's eigenvalues remain in (−1, 1), yielding stable reduction (Figures 5-25). Image denoising. SGDL's smallest eigenvalue approaches −1, causing oscillatory loss; MGDL
keeps all eigenvalues in (−1, 1), ensuring steady reduction (Figures 26-29). CIFAR-10 classification. Using 10,000 sampled images, fully connected ReLU networks (26 (3072, 10, 128, 8) for SGDL and 27 (3072, 10, 128, 2, 4) for MGDL) are trained with squared loss and full-batch gradient descent (Figure 6). With learning rate 0.004 0.004, SGDL reaches loss 7.16×10−3in 26,878 s; MGDL achieves 2.56×10−3in 22,177 s. SGDL shows strong oscillations with eigenvalues often below −1, whereas MGDL exhibits mild oscillations in grade 1 and smooth loss reduction in subsequent grades, with eigenvalues strictly within (−1, 1). Across tasks— synthetic regression, image regression/denoising, and CIFAR-10—SGDL's eigenvalues often fall below −1, causing loss oscillations, while MGDL's stay within (−1, 1), explaining its superior stability. ensuring stable training and better accuracy (Figure 22, third and fourth subfigures). Across both

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png) settings, the smallest eigenvalue predominantly determines loss behavior.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

## 8 Multi-Grade Transformers (Mgt)

The Transformer Vaswani et al. (2017) is a widely used architecture based on self-attention, enabling

![7_image_3.png](7_image_3.png) global information exchange. We introduce a MGT and apply it to time series regression.

A single-grade Transformer (SGT) embeds inputs into dmodel-dimensional vectors with positional encoding, processes them through nh Transformer blocks (self-attention + feedforward with residu432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 als), and outputs predictions:
Input → Embedding → (Attention(dmodel, nhead) + MLP) × nh → Output. (9)
MGT trains multiple grades, each a Transformer of form equation 9 with a single block. Grade 1 uses positional encoding, while later grades inherit positional information and refine residuals. Unlike SGT, which trains a deep stack at once, MGT decomposes training into smaller stages, yielding greater stability, fewer oscillations, and improved convergence and generalization.

Time series regression on synthetic data. We consider predicting the next s = 1 value from the past d = 64 observations, with problem settings, data generation, and network architectures detailed in Appendix C. The first 80% of the sequence is used for training and the last 20% for testing.

Table 4 reports the training and testing mean squared errors (TrMSE, TeMSE), while Figure 7 shows predictions on data. Although both methods fit the training data effectively, MGT achieves significantly better generalization, attaining a test error of 1.6 × 10−1compared to 2.6 for SGT, while requiring only 28% of the training time. As shown in Figure 7, SGT's predictions deteriorate sharply when test sequences deviate from the training distribution, while MGT maintains accurate predictions.

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

Figure 7: Synthetic time series: train/test (1–2) and zoomed test (3–4).
Time series regression on financial data. We analyze the SPX (S&P 500 Index) using daily data from Yahoo Finance or Bloomberg , spanning January 1, 2000, to August 22, 2025. The task is to predict the next s = 1 value from the past d = 20 observations. Details on data preparation and architectures are given in Appendix C. The last 5% of the data is reserved for testing, with 5% of the remainder for validation and the rest for training. Table 5 summarizes mean squared errors (TrMSE, VaMSE, TeMSE), and Figure 8 shows predictions. Although oth models fit the training data affectively, MGT achieves substantially better generalization, attaining a test error of 1.8 × 10−2compared to 8.9 × 10−2for SGT, and requires only 33% of the training time. Crucially, as shown in Figure 8, SGT collapses under distribution shift, with predictions diverging sharply from reality, whereas MGT remains accurate and stable throughout.

![8_image_2.png](8_image_2.png)

We analyzed MGDL from both theoretical and numerical perspectives. Spectral analysis revealed that MGDL keeps eigenvalues of the iteration matrix within (−1, 1), ensuring stable convergence, while SGDL often produces eigenvalues outside this range, leading to oscillatory training. A convergence theorem further confirmed that eigenvalue behavior governs loss dynamics. Experiments on synthetic regression, image reconstruction, and classification consistently showed MGDL's advantages: greater stability, robustness to learning rates, and better accuracy in challenging settings. These results establish MGDL as a principled and effective alternative to SGDL, combining convex reformulations with practical performance gains. Use of Large Language Models. Large Language Models were used to refine the text and ensure grammatical accuracy.

## 9 Conclusion References

Reproducibility Statement. Anonymous code and instructions for all experiments are provided in the supplementary material: Why MGDL outperforms SGDL. Sanjeev Arora, Zhiyuan Li, and Abhishek Panigrahi. Understanding gradient descent on the edge of stability in deep learning. In *International Conference on Machine Learning*, pp. 948–1024. PMLR, 2022.

Yoshua Bengio, Pascal Lamblin, Dan Popovici, and Hugo Larochelle. Greedy layer-wise training of deep networks. *Advances in neural information processing systems*, 19, 2006.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. *Advances in neural information processing systems*, 33:1877–1901, 2020.

Antoni Buades, Bartomeu Coll, and J-M Morel. A non-local algorithm for image denoising. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05), volume 2, pp. 60–65. Ieee, 2005.

Hongming Chen, Ola Engkvist, Yinhai Wang, Marcus Olivecrona, and Thomas Blaschke. The rise of deep learning in drug discovery. *Drug discovery today*, 23(6):1241–1250, 2018.

Jeremy M Cohen, Simran Kaur, Yuanzhi Li, J Zico Kolter, and Ameet Talwalkar. Gradient descent on neural networks typically occurs at the edge of stability. *arXiv preprint arXiv:2103.00065*, 2021.

Thomas M Cover. Geometrical and statistical properties of systems of linear inequalities with applications in pattern recognition. *IEEE transactions on electronic computers*, (3):326–334, 2006.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Jianfeng Guo, C Ross Schmidtlein, Andrzej Krol, Si Li, Yizun Lin, Sangtae Ahn, Charles Stearns, and Yuesheng Xu. A fast convergent ordered-subsets algorithm with subiteration-dependent preconditioners for pet image reconstruction. *IEEE transactions on medical imaging*, 41(11):3289– 3300, 2022.

Kostadin Dabov, Alessandro Foi, Vladimir Katkovnik, and Karen Egiazarian. Image denoising by sparse 3-d transform-domain collaborative filtering. *IEEE Transactions on image processing*, 16 (8):2080–2095, 2007.

Ronglong Fang and Yuesheng Xu. Addressing spectral bias of deep neural networks by multi-grade deep learning. In A. Globerson, L. Mackey, D. Belgrave, A. Fan, U. Paquet, J. Tomczak, and C. Zhang (eds.), Advances in Neural Information Processing Systems, volume 37, pp. 114122–114146. Curran Associates, Inc., 2024. URL https://proceedings.neurips.cc/paper_files/paper/2024/ file/cf1129594f603fde9e1913d10b7dbf77-Paper-Conference.pdf.

Ronglong Fang, Yuesheng Xu, and Mingsong Yan. Inexact fixed-point proximity algorithm for the ℓ0 sparse regularization problem. *Journal of Scientific Computing*, 100(2):58, 2024.

Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on artificial intelligence and statistics, pp. 249–256. JMLR Workshop and Conference Proceedings, 2010.

Ian Goodfellow, Yoshua Bengio, Aaron Courville, and Yoshua Bengio. *Deep learning*, volume 1.

MIT press Cambridge, 2016.

Rob Fergus, Barun Singh, Aaron Hertzmann, Sam T Roweis, and William T Freeman. Removing camera shake from a single photograph. In *Acm Siggraph 2006 Papers*, pp. 787–794. 2006.

Amir Beck and Marc Teboulle. A fast iterative shrinkage-thresholding algorithm for linear inverse problems. *SIAM journal on imaging sciences*, 2(1):183–202, 2009.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition*, pp. 770–778, 2016.

Jie Jiang and Yuesheng Xu. Deep neural network solutions for oscillatory fredholm integral equations. *Journal of Integral Equations and Applications*, 36(1):23–55, 2024.

John Jumper, Richard Evans, Alexander Pritzel, Tim Green, Michael Figurnov, Olaf Ronneberger, Kathryn Tunyasuvunakool, Russ Bates, Augustin Zˇ´ıdek, Anna Potapenko, et al. Highly accurate protein structure prediction with alphafold. *nature*, 596(7873):583–589, 2021.

Jack Kiefer and Jacob Wolfowitz. Stochastic estimation of the maximum of a regression function.

The Annals of Mathematical Statistics, pp. 462–466, 1952.

D.P. Kingma and J. Ba. Adam: A method for stochastic optimization. In *Proceedings of the 3rd* International Conference on Learning Representations (ICLR), 2015. URL https://arxiv. org/abs/1412.6980.

Dilip Krishnan and Rob Fergus. Fast image deconvolution using hyper-laplacian priors. *Advances* in neural information processing systems, 22, 2009.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012.

Yann LeCun, Leon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to ´
document recognition. *Proceedings of the IEEE*, 86(11):2278–2324, 1998.

Qia Li, Lixin Shen, Yuesheng Xu, and Na Zhang. Multi-step fixed-point proximity algorithms for solving a class of optimization problems arising from image processing. Advances in Computational Mathematics, 41(2):387–422, 2015.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Charles A Micchelli, Lixin Shen, and Yuesheng Xu. Proximity algorithms for image models: denoising. *Inverse Problems*, 27(4):045009, 2011.

Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio. On the difficulty of training recurrent neural networks. In *International conference on machine learning*, pp. 1310–1318. Pmlr, 2013.

Mert Pilanci and Tolga Ergen. Neural networks are convex regularizers: Exact polynomial-time convex optimization formulations for two-layer networks. In International Conference on Machine Learning, pp. 7695–7705. PMLR, 2020.

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In International conference on machine learning, pp. 5301–5310. PMLR, 2019.

Maziar Raissi, Paris Perdikaris, and George E Karniadakis. Physics-informed neural networks: A
deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational physics*, 378:686–707, 2019.

Herbert Robbins and Sutton Monro. A stochastic approximation method. The annals of mathematical statistics, pp. 400–407, 1951.

Leonid I Rudin, Stanley Osher, and Emad Fatemi. Nonlinear total variation based noise removal algorithms. *Physica D: nonlinear phenomena*, 60(1-4):259–268, 1992.

Lixin Shen, Yuesheng Xu, and Xueying Zeng. Wavelet inpainting with the ℓ0 sparse regularization.

Applied and Computational Harmonic Analysis, 41(1):26–53, 2016.

Richard P Stanley et al. An introduction to hyperplane arrangements. *Geometric combinatorics*, 13:
389–496, 2007.

## A Convergence Proof Proofs Of Theorem 1 And Theorem 2

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Proof. Since F is twice continuously differentiable, we can expand F(Wk+1) at Wk yields with an error term

$${\mathcal{F}}(W^{k+1})={\mathcal{F}}(W^{k})+\left({\frac{\partial{\mathcal{F}}}{\partial W}}\right)^{-1}(W^{k})\Delta W^{k}+r_{k}$$
∂W 
Therefore, Zhi-Qin John Xu, Yaoyu Zhang, and Yanyang Xiao. Training behavior of deep neural network in frequency domain. In Neural Information Processing: 26th International Conference, ICONIP 2019, Sydney, NSW, Australia, December 12–15, 2019, Proceedings, Part I 26, pp. 264–274. Springer, 2019.

We begin by establishing the convergence of the general gradient descent iteration 2, which serves as the foundation for the proofs of Theorems 1 and 2. For a compact convex set Ω ⊂ RM, we let

$$\alpha:=\operatorname*{sup}_{W\in\Omega}\|\mathbf{H}_{{\mathcal{F}}}(W)\|$$
$$(10)$$

$$\mathbf{\tau}_{k}-W^{k}\mathbf{\tau}_{l}=0;$$
∥HF (W)∥ (10)
where ∥·∥ is the spectral norm of a matrix.

Theorem 5. Suppose F : RM → R *is a nonnegative, twice continuously differentiable function* and Ω ⊂ RM *is a convex, compact set. Let* Wk	∞
k=1 *be a sequence generate from equation 2 for a* given initial guess W0 *and assume that* Wk	∞
k=1 ⊂ Ω. If the learning rate η ∈ (0, 2/α), then the following statements hold:
(i) limk→∞ F(Wk) = F
∗*for some* F
∗ ≥ 0;

(ii) limk→∞
∂F
∂W (Wk) = 0 and limk→∞ ∥Wk+1 − Wk∥ = 0;
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

Tingting Wu and Yuesheng Xu. Inverting incomplete fourier transforms by a sparse regularization model and applications in seismic wavefield modeling. *Journal of Scientific Computing*, 92(2): 48, 2022.

Yuesheng Xu. Successive affine learning for deep neural networks. *Analysis and Applications, to* appear. arXiv preprint arXiv:2305.07996, 2023.

Yuesheng Xu. Multi-grade deep learning. Communications on Applied Mathematics and Computation, pp. 1–52, 2025.

Yuesheng Xu and Taishan Zeng. Multi-grade deep learning for partial differential equations with applications to the burgers equation. *arXiv preprint arXiv:2309.07401*, 2023.

$${\mathcal{F}}(W^{k+1})={\mathcal{F}}(W^{k})-{\frac{1}{\eta}}\|\Delta W^{k}\|^{2}+r_{k}$$
2 + rk (12)
$$(11)$$

$$(12)^{\frac{1}{2}}$$

$$r_{k}=\frac{1}{2}(\Delta W^{k})^{\top}\mathbf{H}_{\mathcal{F}}(\bar{W})\Delta W^{k}$$

where ∆Wk = Wk+1 − Wkand W¯ is a point between Wkand Wk+1. By using equation equation 2, we have that∂F
$$\frac{\partial{\mathcal{F}}}{\partial W}\left(W^{k}\right)=-\frac{1}{\eta}\Delta W^{k}.$$
∆Wk. (11)
(iii) *Every cluster point* Wˆ of Wk	∞
k=0 *satisfies* ∂F
∂W (Wˆ ) = 0 .