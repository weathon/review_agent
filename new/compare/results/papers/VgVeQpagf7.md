000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# High Performance Differentially Private Fine-Tuning Using Dataset Distillation

Anonymous authors Paper under double-blind review

## Abstract

Differentially Private Stochastic Gradient Descent (DP-SGD), which iteratively perturbs clipped per-sample gradients and tracks the cumulative privacy risk using composition accounting, has become a cornerstone in private deep learning. Despite its versatility, DP-SGD in practice faces several limitations. It is constrained by the number of gradient iterations permissible under a limited privacy budget, and is restricted by incompatibilities with common deep learning techniques like ensembling and BatchNorm, and typically produces only a single trained model. In this work, we propose an algorithm for generating a differentially-private (DP) synthetic version of a sensitive dataset. This allows the synthetic dataset to be distributed and postprocessed freely without additional privacy loss, giving more flexibility than DP-SGD. Building on dataset distillation—by producing compact synthetic datasets that preserve downstream performance— we introduce SPS (Summarize–Privatize–Synthesize) and its enhanced variant SPS+. In contrast to prior works, SPS is, to our knowledge, the first alternative to DP-SGD that attains higher accuracy on image-classification tasks. Concretely, on CIFAR 10 / CIFAR 100 with privacy budget ϵ = 1, SPS+ achieves 96.2/76.6% top-1 accuracy, outperforming state-of-the-art (SOTA) DP-SGD results (94.8/70.3%).

## 1 Introduction

Deep learning is effective in predictive performance and efficient in implementation when trained on non-sensitive data, but it falls short when datasets contain private information. Numerous practical attacks can efficiently reconstruct training data from trained models in the absence of proper protections Zhu et al. (2019a); Buzaglo et al. (2023). To provably mitigate adversarial inference, a widely adopted framework is *Differential Privacy* (DP) Dwork (2006), which quantifies and allows one to control the advantage of any adversary in determining the membership of an individual data record. Deep Learning with DP constraints has long proved challenging: not only do practitioners pay a price in terms of accuracy, the most common and performant method for differentially private learning: Differentially-Private SGD (DP-SGD) Abadi et al. (2016), comes at a computational and practical cost. DP-SGD faces practical limitations: restricted iterations due to composition budgets, incompatibility with BatchNorm (Ioffe, 2015) and ensembling Ganaie et al. (2022), and expensive per-sample gradient computation. Furthermore, if the data is to be used again in the future, one must retrain old models so account for the additional privacy cost of training the new ones. Alternatively, one can *privatize* a dataset by generating a synthetic version that satisfies DP guarantees. This approach offers far greater flexibility than DP-SGD: the synthetic corpus can be publicly released, reused to train multiple models, and probed with data-attribution–based explainability methods—capabilities that are typically infeasible under standard DP-SGD. An especially compelling use case arises when sensitive data are siloed across multiple databases, each with its own privacy restrictions. If each curator independently releases a DP-synthetic dataset, these can be aggregated into a global public corpus, allowing mutual benefit without leaking private information.

Despite this promise, generation-based approaches have historically lagged gradient-based training in accuracy, limiting their practicality. Meanwhile, recent progress in dataset distillation—the study of crafting small synthetic datasets that train to high accuracy—offers guidance for building high-performance synthetic data. For example, the D3S algorithm (Loo et al., 2024), which matches intermediate-layer feature statistics, is among 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 the first methods to scale to larger models. Although recent work has combined dataset-distillation techniques with DP (Vinaroz & Park, 2024), its performance remains below that of DP-SGD. In this work, we show for the first time that distillation-based approaches to privacy can match or exceed the accuracy of DP-SGD on private image classification while providing greater flexibility. In particular, our contribution can be summarized as follows:
1. **Present SPS**, a differentially private dataset distillation algorithm that adapts D3S to work with public pre-trained models while privatizing intermediate activation statistics; 2. **Develop multistage clipping and grouped pseudo-classes** techniques that significantly improve performance in high-privacy regimes, yielding the enhanced SPS+ algorithm; 3. **Demonstrate competitive performance** with DP-SGD on CIFAR-10/100 classification, becoming the **first** generation-based method to match gradient-based approaches; 4. **Show practical advantages** of data-based privacy including support for model ensembling, federated learning, and continual learning without additional privacy cost.

## 2 Background 2.1 Differential Privacy

DP defines the privacy risk by measuring the worst-case (maximal) divergence between the output distributions of some mechanism M on two *adjacent* datasets differing in a single data point. In the following, we formally introduce a well-known variant of DP, Renyi Differential Privacy (RDP). ´
Definition 2.1 ( (*α, ϵ*(α))−Renyi Differential Privacy Mironov (2017b)) ´ . Given a universe X , we say that two datasets X, X′ ⊆ X ∗ are adjacent, denoted as X ∼ X′, if X = X′ ∪ {x} or X′ = X ∪ {x} for some additional datapoint x ∈ X . A randomized algorithm M *satisfies* (α, ϵ(α))-Renyi Differential Privacy (RDP), ´ α > 1*, if for any pair of adjacent datasets* X ∼ X′,
Dα(PM(X)∥PM(X′)) ≤ ϵ(α). Here, PM(X) and PM(X′) represent the distributions of M(X) and M(X′)*, respectively, and*

$$\mathrm{D}_{\alpha}(\mathrm{P}\|\mathrm{Q})={\frac{1}{\alpha-1}}\log\int\mathbf{q}(o)({\frac{\mathbf{p}(o)}{\mathbf{q}(o)}})^{\alpha}\,\,d o,$$
$\left(\mathrm{l}\right)$. 
αdo, (1)
represents α-Renyi Divergence between two distributions ´ P and Q *whose density functions are* p and *q, respectively.* For a given α, a smaller ϵ(α) implies a more significant challenge for an adversary to distinguish the participation of an arbitrarily-selected data point, and the mechanism M preserves privacy better. RDP can also be converted back into the classic approximate (*ϵ, δ*) DP (Mironov, 2017a; Canonne et al., 2020). We provide a full description in section C. In many applications, multiple accesses to a sensitive dataset are required, and each additional release increases the potential privacy leakage. RDP can be used to elegantly handle the composition of privacy leakage, as given by Lemma 2.2 Lemma 2.2 (Composition of RDP (Proposition 1 of Mironov (2017b))). Let f : *D 7→ R*1 be (α, ϵ1)-RDP and g : R1 × D 7→ R2 be (α, ϵ2)*-RDP, then the mechanism defined as* (X, Y )*, where* X ∼ f(D) and Y ∼ g(X, D), satisfies (α, ϵ1 + ϵ2)*-RDP.*
Lemma 2.2 suggests that the security parameter ϵ(α) in RDP increases *linearly* under composition. Most existing DP composition methods, including RDP (Lemma 2.2) and more refined techniques such as f-DP Dong et al. (2022) and characteristic function-based approaches Zhu et al. (2022), need to consider the *worst-case* scenario for each composed mechanism. Consequently, they are unable to handle *unbounded* composition. In the context of machine learning, this limitation implies that a fixed privacy budget *cannot* support training an unlimited number of privatized model using conventional composition techniques. This motivates the use of *private synthetic data generation* as a means to enable unrestricted private data utilization. Another concrete application of composition is DP-SGD Abadi et al. (2016). It interprets a standard T-iteration SGD process as a sequence of T adaptively composed single-iteration gradient computations and updates, and clips and adds noise to gradients to ensure DP-guarantees under T compositions. Similarly, the iteration number T must be bounded.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

![2_image_0.png](2_image_0.png)

## 2.2 Dp Synthetic Data Generation

For private synthetic data generation, DP-SGD has been used to train generative models—including diffusion models (Ghalebikesabi et al., 2023; Dockhorn et al., 2022) and GANs (Xie et al., 2018)—but scaling these models remains challenging. Larger models typically require more iterations to converge, yet DP-SGD is limited by a finite composition budget; moreover, the injected noise grows with model dimensionality (Xiao et al., 2023), so bigger models endure heavier noise, further slowing convergence. Similar composition constraints also affect alternatives such as PATE (Private Aggregation of Teacher Ensembles) (Papernot et al., 2016), which requires access to additional unlabeled public data. Other DP data-generation strategies exist: in images, one can leverage public diffusion models (Lin et al., 2024), or—closest to our approach—statistically match privatized mean embeddings (Harder et al., 2020). However, in the image domain, downstream classification performance still trails direct DP-SGD training: the best reported CIFAR-10 accuracy from synthetic data is 89.1% (Lin et al., 2024), well below DP-SGD's > 95% (De et al., 2022). By contrast, in the text domain, fine-tuning on private synthetic data can outperform direct DP-SGD training (Xie et al., 2024; Amin et al., 2024). To our knowledge, our work is the *first* to achieve such parity in the image domain.

## 3 Our Method 2.3 Dataset Distillation

Dataset distillation (DD) aims to produce compact synthetic datasets that preserve high downstream accuracy (Wang et al., 2018). Among major DD families—bilevel optimization (Wang et al., 2018; Loo et al., 2023), kernel-based methods (Nguyen et al., 2021a;b), trajectory matching (Cazenavette et al., 2022), and *activation–statistic matching* (e.g., D3S (Loo et al., 2024; Yin et al., 2023a))—the last is particularly well-suited to differential privacy. These methods align intermediate activation statistics (e.g., means and covariances at BatchNorm layers) between the full and distilled datasets.

Crucially, statistic-matching algorithms like D3S need to privatize only the statistic-collection phase, enabling DP via a single noise-addition step. By contrast, other DD approaches must privatize each optimization iteration, incurring costly iterative composition (Dwork, 2006). Prior private DD attempts, such as DP-KIP (Vinaroz & Park, 2024), attain only 58.7% on CIFAR-10 at
(*ϵ, δ*) = (10, 10−5), far below DP-SGD's > 93% on the same benchmark (De et al., 2022).

In this section, we present the Summarize–Privatize–Synthesize (SPS) algorithm. Our method builds on the D3S dataset–distillation framework (Loo et al., 2024), but introduces substantial modifications to address the challenges of the differentially private (DP) setting. We first describe SPS in section 3, then introduce *multistage clipping* and *grouped pseudo-classes* in section 4, yielding the enhanced SPS+ algorithm, which performs significantly better on multiclass tasks.

## 3.1 Review: D3S Dataset Distillation

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Dataset Distillation with Domain Shift (D3S) matches intermediate activation statistics between the full dataset XT and the distilled set XS. Statistics are computed using a model θT trained on XT .

Concretely, let the post–BatchNorm activation at layer l for example i be zi,l ∈ R
Dl×Hl×Wl. After averaging over spatial dimensions (Hl, Wl), D3S forms per-layer means and covariances for the full and distilled sets, µ T l
, µS
l ∈ R
Dl, Σ
T l
, Σ
S
l ∈ R
Dl×Dl, and seeks to match these statistics. Treating the distributions of zi,l as Gaussian, the objective minimizes the KL divergence between the corresponding normals together with a supervised loss on distilled labels YS (pre-assigned), evaluated by θT :

$$\mathcal{L}_{\text{DSS}}=\sum_{l=1}^{L}D_{KL}\Big{(}\mathcal{N}(\mu_{l}^{T},\Sigma_{l}^{T})||\mathcal{N}(\mu_{l}^{S},\Sigma_{l}^{S})\Big{)}+\text{x-cnt}\Big{(}f_{\theta_{T}}(\mathcal{X}_{S}),\mathcal{Y}_{S}\Big{)}\,,$$

D3S further averages the loss across multiple trained models and employs an exponential–moving–average scheme to estimate µ S
land Σ
S
l when the distilled batch size is large. Optimizing this loss yields synthetic images XS. To obtain soft targets, a knowledge–distillation–style procedure assigns soft labels to XS using θT .

However, the foregoing procedure is not private. First, the collection of µ T
land Σ
T
lis non-private.

More critically, the algorithm depends on a model θT trained on the full private dataset—both to guide image synthesis and to assign labels—so a private distillation method must eliminate (or carefully privatize) this reliance on θT .

## 3.2 Adapting To Private Generation

Here we describe how we develop SPS, an algorithm similar to D3S which overcomes the aforementioned issues and works in the differentially private setting.

## 3.2.1 Removing The Trained Model

To address the issues above, we first remove reliance on the privately trained model θT . The most straightforward remedy is to use a publicly pretrained model θP trained on a large non-sensitive corpus—a common practice in the DP literature (Mehta et al., 2022; De et al., 2022; Ganesh et al.,
2023). However, substituting θT with θP introduces two challenges: **(i) missing class assignments**
and **(ii) missing soft-label information**. In D3S, a cross-entropy term enforces class alignment for synthesized images, and soft targets are obtained via knowledge distillation from θT—both unavailable without θT .

To circumvent this, for a dataset with C classes we collect C sets of class-conditional statistics at a subset of layers LC ⊆ [L] (typically the last three). These statistics must be rich enough to capture distributional structure lost with hard labels, so we model *full* multivariate Gaussian summaries rather than matching means alone. During synthesis, images intended for class c match the corresponding class-conditional statistics via a KL-divergence objective, while all synthetic data additionally match *global* (class-marginal) statistics averaged across classes. Let the global statistics be (µ T ,G
l, Σ
T ,G
l) and, for class c, the class-conditional statistics be (µ T ,c l, Σ
T ,c l); the resulting loss is:

$$\mathcal{L}_{\text{SPS}}=\sum_{l=1}^{L}D_{KL}\Big{(}N(\mu_{l}^{T,G},\Sigma_{l}^{T,G})||\mathcal{N}(\mu_{l}^{S,G},\Sigma_{l}^{S,G})\Big{)}+\lambda_{C}\sum_{c=1}^{C}\sum_{l\in L_{C}}D_{KL}\Big{(}N(\mu_{l}^{T,e},\Sigma_{l}^{T,c})||\mathcal{N}(\mu_{l}^{S,e},\Sigma_{l}^{S,c})\Big{)}\tag{2}$$

For some scalar hyperparameter λC , the assigned classes are then used for **hard labels** as opposed to
soft ones. We show the importance of this change in section B.1. Additionally, we use use different
dimensions DG and DC for global and class-specific statistics, typically choosing DG > DC . This
is necessary because the class-specific statistics are subject to noise which is larger by a factor of C when privatized, so we must keep their dimensionality smaller. To do this, for intermediate activation
zi,l ∈ R
Cl×H×W for data point i, we project it and apply a nonlinearity to get the embedding used
for the global and class-specific embedding:
$$z_{i,l}^{G}=2\sigma(M_{l}^{G}z_{i,l})-1\in\mathbb{R}^{D_{G}\times H\times W},$$
DG×H×W , zC
$$z_{i,l}^{C}=2\sigma(M_{l}^{C}z_{i,l})-1\in\mathbb{R}^{D_{C}\times H\times W}$$
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

$$\begin{array}{l}{\mbox{$\mathfrak{i}$}=\underbrace{[m_{i,1}^{G},u_{i,1}^{G},\ldots m_{i,L}^{G},u_{i,L}^{G},\underbrace{m_{i,1}^{1},u_{i,1}^{1},\ldots m_{i,L}^{1},u_{i,L}^{1}}_{v_{i}^{G}=1},\ldots m_{i,1}^{C},u_{i,1}^{C},\ldots m_{i,L}^{C},u_{i,L}^{C}]$}}\\ \end{array}$$  dimensionality of the released statistics $v_{i}$ is $d_{\mbox{\tiny{tot}}}\ =\ L\,D_{G}^{\mbox{\tiny{layer}}}\ +\ C\,|\,L_{C}|\,D_{G}^{\mbox{\tiny{layer}}}$.  
$${}^{(4)}$$

G + C |LC | D
C, where D
layer G = DG +
DG(DG+1)
2, corresponding to the dimensionality of a single layer's mean and covariance1. The same applies analogously for the per-class statistics captured by D
layer C. Thus, the overall dimensionality can be tuned by adjusting DG, DC , and |LC |. Tuning the dimensionality of the privatized statistic is a key advantage of SPS of DP-SGD. Whereas DP-SGD is limited by the high dimensionality of gradients (∼ 107), by tuning DC and DG to be small, the dimensionality of SPS can be significantly smaller (∼ 105), thereby improving the SNR of the privatized statistic.

We are interested in releasing the aggregate sum of these statistic vectors: v˜ =PN
i=1 vi, computed over N datapoints. This sum can be privatized using the Gaussian Mechanism (Dwork, 2006).

Specifically, we clip each vito satisfy ∥vi∥2 ≤ ∥v∥max := KclipqLDlayer G + |LC |D
layer C, where Kclip is a positive constant (typically on the order of 10−1). We then add Gaussian noise with standard deviation σ = b0 ∥v∥max, where b0 is chosen according to the privacy budget. The final released vector is

$${\bar{v}}\ =\ \sum_{i=1}^{N}\mathrm{clip}(v_{i})\ +\ {\mathcal{N}}\Big{(}0,\,b_{0}^{2}\|v\|_{\mathrm{max}}^{2}I\Big{)},\qquad\mathrm{clip}(v_{i})\ =\ v_{i}\cdot\min\biggl{\{}1,\,{\frac{\|v\|_{\mathrm{max}}}{\|v_{i}\|_{2}}}\biggr{\}}\,.$$
. (4)
Here, clip(·) denotes the standard ℓ2 clipping function.

## 3.2.3 Post-Processing The Statistic Vector

After obtaining the full statistic vector v˜, we need to convert it back into means and covariances. Splitting v˜ back into summed first and second moments, we have m˜
G
l, u˜
G
lfor each layer and m˜
c l, u˜
c l for each class and layer. Focusing on the global statistics, we process these into means and covariances as follows:

$$\mu_{l}^{G}=\frac{1}{N}\bar{m}_{l}^{G},\qquad\qquad\hat{\Sigma}_{l}^{G}=\frac{1}{N}\mathrm{triu}^{-1}(\tilde{u}_{l}^{G})-\mu_{l}^{G}\mu_{l}^{T G}$$

Where triu−1converts u˜ back into the symmetric matrix s˜. We additionally clip negative eigenvalues of Σˆ G
lto produce Σ
G
l, which we use during optimization. Details are provided in section A.3 We apply a similar procedure for the class-specific statistics, instead normalizing by NC
. This yields our final set of statistics for matching. Once we have these statistics, we synthesize datapoints by iteratively optimizing eq. (2), initializing with random noise images in XS. We generate images until we have reached our desired dataset size |XS|. Full algorithm pseudocode is available in section A.1.

It is noted that the second moment is a symmetric matrix and thus we only need to release the upper triangular part. Let u = triu(s) be the upper triangular part of s. Flattening and concatenating these into a single statistic vector vi, we have, for each datapoint:

mG
i,l =1 HW X H h=1 X W w=1 z G i,l ∈ R DG , sG i,l =1 HW X H h=1 X W w=1 z G i,lz ⊺G i,l ∈ R DG×DG , 1 ≤ l ≤ L i,l = 1(yi=c) HW XH h=1 XW w=1 z c i,l ∈ R DC , sc i,l = 1(yi=c) HW XH h=1 XW w=1 z c i,lz ⊺c i,l ∈ R DC ×DC , l ∈ LC , 1 ≤ c ≤ C
mc
$\left(\uparrow\right)$
(3)
The second change we need to make is to ensure collecting the global and class summarization are implemented in a privacy-preserving manner. For each datapoint xi, we release the first and second moments given by:
where MG
land MC
lare random projection matrices of dimension R
DG×Dl and R
DC ×Dl, respectively, and σ is the sigmoid function.

## 3.2.2 Privatizing The Statistics

5

## 3.2.4 Better Clipping By Redistributing Noise

The standard clip-and-add-noise procedure performs poorly because per-class statistics are normalized by CN while global statistics use 1N
, making per-class statistics C times more susceptible to noise. This, in turn, degrades class matching in synthesized images. We remedy this by upscaling the per-class statistics by 
√S, where S =LDlayer G
|LC |D
layer C
, and release vi = [v G
i,
√Sv1 i*, . . . ,* 
√SvC
i]. After adding noise, we divide by 
√S when reconstructing the per-class statistics. This redistributes noise to impact the global parameters more, while keeping the same privacy cost b0. Correspondingly, we clip according to |v|max = KclipqLDlayer G + S|LC |D
layer G = Kclipq2LDlayer G .

## 3.2.5 Better Optimization

Finally, we detail a few techniques used in optimization to improve performance.

Smooth Activations. We use SiLU activations instead of ReLU in pretrained models θP to facilitate optimization through the network. Smooth activations improve input optimization tasks like reconstruction attacks and adversarial robustness (Shahin Shamsabadi et al., 2023; Xie et al., 2020), making them well-suited for our image synthesis process. Sharpness Aware Minimization (SAM) at validation time. We fine-tune on distilled datasets using the GSAM optimizer (Foret et al., 2021; Zhuang et al., 2022). This choice is motivated by the evidence that SAM-style methods improve generalization under label noise (Baek et al., 2024), which is pertinent here because privatization injects noise. Although SAM typically complicates DP training by requiring two gradient evaluations per step, our setting—training on privatized (DP) synthetic data—falls under the post-processing property (Dwork, 2006). Consequently, any downstream optimizer, including GSAM, can be used without incurring additional privacy cost.

## 4 Sps+: Sps With Better Clipping And Grouping

While SPS provides differential privacy, it performs poorly in the high-privacy regime. In particular, the per-class statistics v c iaccumulate noise at a rate of O(C/N), which can become prohibitive in few-sample settings. To address this, we introduce two key enhancements: *multistage clipping* (MC) and *grouped pseudo-classes* (GPC), together yielding the improved SPS+ algorithm.

## 4.1 Multistage Clipping

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4.2 Grouped Pseudoclasses

To address O(C/N) noise rate, rather than directly matching C estimates of v c, each with noise rate O(C/N), we proposed generating P > C *pseudo-classes*, which are composed of random groups Nc/p > 1 real classes, and matching pseudoclasses statistics with each other. As a result, each class belongs to P Nc/p Cpseudo-classes. Each pseudoclass' statistics estimate has a more favourable O(C
NNC/p
) noise rate, allowing better optimization of eq. (2). Importantly, this technique **only**
works due to dynamics of optimizing the loss function, specifically the Σ inversion in the KL- In the basic SPS framework, a single measurement of v˜ must be privatized. Prior work on DP mean estimation (Bie et al., 2023) shows that multistage methods—which iteratively adjust the clipping radius and recenter around previous (biased) estimates—can empirically outperform single-shot estimation. Multistage clipping (MC) adapts this idea to SPS. Specifically, we begin with clipping center 0 and factor K1 clip, producing an initial synthetic dataset X
1 S. In the next stage, we recenter the clipping operation at the empirical means computed from X
1 S, initialize the optimization with X
1 S, adopt a new clipping factor K2 clip, and jointly optimize LSPS over both stages to obtain X
2S. This process is repeated for M stages, each time re-centering, re-clipping, and re-initializing from the previous synthetic dataset. A full description of the modified measurement procedure is given in section A.6. The privacy guarantee follows directly from composition, resulting in M-fold DP.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

![6_image_0.png](6_image_0.png) 

| models.                                        | Method          | CIFAR-10   | CIFAR-100   |                                             |            |            |            |            |
|------------------------------------------------|-----------------|------------|-------------|---------------------------------------------|------------|------------|------------|------------|
| ϵ = 1                                          | ϵ = 2           | ϵ = 4      | ϵ = 8       | ϵ = 1                                       | ϵ = 2      | ϵ = 4      | ϵ = 8      |            |
| DP-SGD (De et al., 2022) (WRN28-10) 94.8 ± 0.1 | 95.4 ± 0.2      | 96.1 ± 0.1 | 96.6 ± 0.1  | 70.3 ± 0.1 74.7 ± 0.2 79.2 ± 0.2 81.8 ± 0.1 |            |            |            |            |
| SPS (WRN28-10)                                 | 93.2 ± 0.2      | 94.6 ± 0.2 | 95.0 ± 0.2  | 95.9 ± 0.1                                  | 48.9 ± 1.1 | 54.0 ± 1.0 | 66.3 ± 0.4 | 70.7 ± 0.4 |
| SPS (WRN34-10)                                 | 93.9 ± 0.1      | 94.9 ± 0.2 | 95.4 ± 0.2  | 96.1 ± 0.1                                  | 50.6 ± 1.2 | 53.7 ± 0.9 | 67.2 ± 0.1 | 72.2 ± 0.3 |
| SPS (WRN28-10 Ensemble)                        | 94.9            | 95.8       | 96.0        | 96.5                                        | 57.0       | 59.7       | 71.6       | 74.9       |
| SPS (WRN34-10 Ensemble)                        | 95.3            | 95.9       | 96.3        | 96.8                                        | 59.2       | 59.6       | 71.8       | 75.9       |
| SPS+ (WRN28-10)                                | 95.1 ± 0.3      | 95.9 ± 0.1 | 96.3 ± 0.1  | 96.3 ± 0.2                                  | 71.0 ± 0.3 | 74.3 ± 0.3 | 76.2 ± 0.3 | 77.5 ± 0.1 |
| SPS+ (WRN34-10)                                | 95.5 ± 0.1      | 96.0 ± 0.1 | 96.4 ± 0.1  | 96.6 ± 0.1                                  | 71.9 ± 0.5 | 75.2 ± 0.4 | 77.2 ± 0.2 | 78.4 ± 0.2 |
| SPS+ (WRN28-10 Ensemble)                       | 96.0            | 96.5       | 96.9        | 97.1                                        | 75.9       | 78.4       | 80.1       | 80.9       |
| SPS+ (WRN34-10 Ensemble)                       | 96.2            | 96.8       | 97.1        | 97.2                                        | 76.6       | 79.2       | 80.7       | 81.6       |
| Private Evolution (Lin et al., 2024)           | 89.13% (ϵ = 10) | -          |             |                                             |            |            |            |            |

divergence, and the eigenvalue clipping of Σ. This method **does not offer benefits for direct mean** estimation. More details are available in section A.5.

## 4.3 Privacy Guarantee For Sps

The privacy-sensitive step of SPS is the release of aggregation of individuals vis, which is given in eq. (4). This is directly private sum estimation using the Gaussian Mechanism. For SPS+, this is composed M times for each measurement step. We have the following privacy guarantee: Theorem 4.1 (Privacy of SPS). The release of v˜ in eq. (4) for M models satisfies (α, ϵ)*-RDP, where* ϵ =
Mα 2b 20 for α > 1*. Proof. See section C.1* □
This is a direct result of the M-fold composition of Gaussian Mechnisms under RDP. This can be converted to (*ϵ, δ*)-DP using Proposition 12 in Canonne et al. (2020), which is implemented in the RDP-account in Ahmed et al. (2025), which we use in the comparison with prior works.

## 5 Results 5.1 Fine-Tuning With Public Datasets

To validate our method, we first test it on fine-tuning publicly available pretrained datasets. Specifically, we take the task of generating synthetic private version of CIFAR-10 and CIFAR-100 (Krizhevsky, 2009) using a pretrained Wide ResNet 22-8 with SiLU activations trained on 32 × 32 resolution ImageNet (He et al., 2015; Zagoruyko & Komodakis, 2017; Deng et al., 2009), in line with prior work (De et al., 2022). Our privatized datasets are the same size as the original (50,000 images). During distillation, we vary the privacy budget ϵ ∈ {1, 2, 4, 8}, with a fixed δ = 10−5. For SPS+, we keep P = 20, 200 pseudoclasses for CIFAR-10 and CIFAR-100, respectively, and vary M, the number of stages. Details on the choice of hyperparameters are given in section D.2.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Since our method produces data rather than a model, we include a second validation step in which a model is trained on the privatized dataset. By the DP post-processing property (Dwork, 2006), any choice of optimizer, model, or training configuration in this stage does not affect the privacy budget. This flexibility allows us to explore a range of options.

![7_image_1.png](7_image_1.png)

Figure 3: FID vs. Accuracy for CIFAR- 10 and CIFAR=100 of SPS+ (M = 2, 4 for CF10/100, respectively)
Specifically, we fine-tune Wide ResNets of sizes 28-10 and 34-10 (both with ReLU activations), pretrained on 32×32 Downsampled ImageNet, using the GSAM optimizer. We also evaluate ensembles of E = 5 fine-tuned models. In contrast, standard DP-SGD pipelines impose much stricter limitations: ensembles would require additional composition, and larger models such as WRN- 34-10 would incur extra privacy cost due to their higher parameter count. Table 1 summarizes our results, alongside two key baselines: the state-of-the-art DP training algorithm from (De et al., 2022) and the best reported accuracy from DP generation, Private Evolution (Lin et al., 2024). Comparisons to additional gradient- and generation-based methods are provided in section F. For brevity, we present only the strongest gradient-based and generationbased baselines in the main text. On CIFAR-10, SPS—especially with ensembling—consistently outperforms DP-SGD across all privacy budgets, but falls short on CIFAR-100. SPS+ matches or exceeds DP-SGD in every setting, particularly under strict privacy budgets with many classes (e.g., CIFAR-100 at ϵ = 1). We further observe that images distilled by SPS using a WRN-22-8 transfer effectively to larger models never seen during distillation, demonstrating the flexibility of our approach. Unless otherwise noted, we focus on SPS+ in all subsequent experiments, due to its superior overall performance.

## 5.2 Out Of Domain Image Classification

To validate SPS under public data domain mismatch, a common challenge in private learning, we tested on CAMELYON17 (Bandi), a histopathology dataset of lymph node sections annotated for metastatic cancer, using 64 × 64 ImageNet as pretraining data. We use SPS in this setting as in the binary classification case, the pseudo-class method does not apply. We generated 100k synthetic images (50k) for each class at resolution 64×64 (downsampled from 96×96), and evaluated our classification performance in table 2. We compare against existing baseline methods Private Evolution (Lin et al., 2024), DP-Diffusion (Ghalebikesabi et al., 2023) and DP-SGD. SPS successfully handles the setting where there is significant mismatch between the private and public datasets.

Table 2: CAMELYON17 Classification Performance (δ = 3 · 10−6)
Accuracy Ours (ϵ = 8) **92.6%**
DP-Diffusion (ϵ = 10) (Ghalebikesabi et al., 2023) 91.1%
Private Evolution (ϵ = 7.56) (Lin et al., 2024) 79.6% DP-SGD (ϵ = 10) (Ghalebikesabi et al., 2023) 90.5%

![7_image_0.png](7_image_0.png)

Figure 4: Visualization of SPS+ distilled images on CIFAR-100 with M = 1 stages. As privacy budget increases, distilled images have better visual fidelity.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 A key advantage of data-based privacy is interpretability. Unlike DP training, which precludes data attribution methods (Koh & Liang, 2020), our method produces inspectable synthetic data. Fig 4 shows distilled CIFAR-100 images across privacy budgets: as ϵ increases, images evolve from abstract textures to recognizable class-specific representations, and consequently FID decreases. fig. 3 additionally visualizes the negative correlation between downstream performance and FID for SPS+.

## 5.4 Simultaneous Distillation And Privatization

Table 3: Oversized synthesis performance of SPS+ on CIFAR-100. Oversized synthetic datasets can further improve performance

| Distilled Dataset size   |      |      |      |      |        |
|--------------------------|------|------|------|------|--------|
| ϵ                        | 1×   | 2×   | 3×   | 4×   | DP-SGD |
| 1                        | 76.6 | 76.4 | 76.0 | 75.9 | 70.3   |
| 2                        | 79.2 | 79.4 | 79.4 | 79.3 | 74.7   |
| 4                        | 80.7 | 81.2 | 81.3 | 81.1 | 79.2   |
| 8                        | 81.6 | 81.8 | 82.1 | 81.9 | 81.8   |

In section 5.1, we evaluated performance when the distilled dataset had a 1 : 1 compression ratio, i.e., equal in size to the original dataset. A notable property of our approach is that once the statistics v˜ are privatized, the number of synthetic images to generate can be chosen freely—without incurring any additional privacy cost. This enables simultaneous distillation and privatization by setting |XS| < |XT |. In this section, we investigate this setting using M = 2 for CIFAR-10 and M = 4 for CIFAR-100. We follow the evaluation protocol of section 5.1, training a single WRN-34-10 model at various privacy budgets and reporting results in Fig 5. Larger synthetic datasets lead to higher accuracy, but even when using a privatized dataset with only 10% of the original size, performance on CIFAR-10 drops by merely ∼ 1%, highlighting the efficiency of our approach. Oversized dataset distillation. SPS+ also allows synthetizing datasets *larger than the original* dataset. table 3 shows the effect of creating distilled datasets up to four times the size of the original, evaluated on WRN34-10 ensembles. For CIFAR-100, further performance gains is unlocked with oversized distilled datasets.

## 5.5 Federated Learning

Federated learning enables multiple parties to collaboratively train models without sharing raw data. Traditional approaches exchange gradients, but these are vulnerable to reconstruction attacks (Zhu et al., 2019b) and require synchronized communication rounds. By contrast, our data-based approach enables *asynchronous* federated learning: each party independently generates privatized datasets using SPS+ and shares them without synchronization constraints. We evaluate this by splitting CIFAR-10 into five partitions of 10,000 images each, with each party running SPS+ independently. Specifically each of N = 5 parties have a disjoint subset of the original dataset: X
1 T
, ...X
N
T
,, with |X 1T
| = 10,000. Each party run SPS+ locally and independently, generating synthetic privatized sets X
1 S*, ...*X
N
S. These are sent to a server, which combines the datasets into a single one: X
combined S =SN
i X
iS
, and trains on X
combined S. SPS also supports a

## 5.3 Visualizing The Distilled Images

![8_image_0.png](8_image_0.png)

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 variant that addresses this by performing *centralized generation* while still keeping only privatized statistics on the server, which we discuss in section B.6. As shown in fig. 5, compared to FedLAP-DP (Wang et al., 2024) and FedDM (Xiong et al., 2022), Federated SPS+ significantly outperforms, particularly under strict privacy budgets (fig. 5). Federated SPS+ also successfully aggregates data from multiple sources, improving performance with more sources. For example, at ϵ = 1, accuracy improves from 86% with a single data source to 89.5% with five sources.

## 5.6 Continual Learning

Continual learning involves training models sequentially on a stream of tasks or datasets. In the private setting, this presents a fundamental challenge: revisiting past data consumes additional privacy budget, while discarding it leads to catastrophic forgetting (French, 1999). Data-based privacy methods address this by enabling unlimited reuse of previously privatized datasets without incurring extra privacy cost.

We evaluate class-incremental learning on CIFAR-100 by splitting the dataset into 10 subsets and each has 10 classes. Each subset is privatized with SPS+ under budget (*ϵ, δ*), and at stage 1 ≤ τ ≤
10 we train on the cumulative privatized sets {X 1S
, . . . , X
τS
}. Results show that performance remains close to the regular, non-continual training: for example, at ϵ = 4, our method achieves 68.1±0.7% accuracy, compared to 76.9 ± 0.4% for standard training (fig. 5).

## 6 Limitations, Discussion, And Conclusion

In this paper we present SPS and SPS+, algorithms which convert a dataset into a private version leveraging a model pretrained on public data. They are the dataset-distillation based which yields competitive results with fine-tuning based on DP-SGD. SPS also yields more flexibility than DP- SGD, at it supports tasks such as private federated learning and continual learning without modification. Despite SPS's strong performance, there are several areas for improvement. The cost of generating these images is relatively heavy (see section F.1 for discussion). Future work could look at whether SPS generation can be amortized, or whether public generators could be used with SPS-style losses, similar to GLaD (Cazenavette et al., 2023) for dataset distillation. In this work we also focused on the simpler class-balanced setting, but future work could study SPS for classes with extreme class imbalance. Other work could look at extending SPS to discrete modalities such as text. Overall, SPS presents a promising new alternative for private deep learning which offers flexibility beyond DP-SGD.

## 7 Reproducibility Statement

Full algorithm pseudocode and descriptions are available in section A.1. Code is provided in the supplementary material.

## References

Martin Abadi, Andy Chu, Ian Goodfellow, H Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. Deep learning with differential privacy. In *Proceedings of the 2016 ACM SIGSAC* conference on computer and communications security, pp. 308–318, 2016.

Karim Ahmed, Lavinia Maria Nedelea, celiayz, Christoph Dibak, Filippo Balicchia, benjamin de charmoy, Brian Michalski, Esteban Gehring, Osuke, Saket Kumar, Veronica Tang, acheam0, brandonedmunds2, rossy312, and yclian0528. google/differential-privacy, 5 2025. URL https: //github.com/google/differential-privacy.

Kareem Amin, Alex Bie, Weiwei Kong, Alexey Kurakin, Natalia Ponomareva, Umar Syed, Andreas Terzis, and Sergei Vassilvitskii. Private prediction for large-scale synthetic text generation, 2024. URL https://arxiv.org/abs/2407.12108.

Christina Baek, Zico Kolter, and Aditi Raghunathan. Why is sam robust to label noise?, 2024. URL
https://arxiv.org/abs/2405.03676.

Peter Bandi. Camelyon17 dataset. URL https://camelyon17.grand-challenge.org/
data.

Alex Bie, Gautam Kamath, and Vikrant Singhal. Private estimation with public data, 2023. URL
https://arxiv.org/abs/2208.07984.

Sourav Biswas, Yihe Dong, Gautam Kamath, and Jonathan Ullman. Coinpress: Practical private mean and covariance estimation, 2022. URL https://arxiv.org/abs/2006.06618.

Zhiqi Bu, Jialin Mao, and Shiyun Xu. Scalable and efficient training of large convolutional neural networks with differential privacy, 2022. URL https://arxiv.org/abs/2205.10683.

Gon Buzaglo, Niv Haim, Gilad Yehudai, Gal Vardi, Yakir Oz, Yaniv Nikankin, and Michal Irani.

Deconstructing data reconstruction: Multiclass, weight decay and general losses. Advances in Neural Information Processing Systems, 36:51515–51535, 2023.

Clement L. Canonne, Gautam Kamath, and Thomas Steinke. The discrete gaussian for differential ´
privacy. *CoRR*, abs/2004.00010, 2020. URL https://arxiv.org/abs/2004.00010.

George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu. Dataset distillation by matching training trajectories, 2022. URL https://arxiv.org/abs/2203. 11932.

George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A. Efros, and Jun-Yan Zhu. Generalizing dataset distillation via deep generative prior, 2023. URL https://arxiv.org/abs/ 2305.01649.

Dingfan Chen, Raouf Kerkouche, and Mario Fritz. Private set generation with discriminative information, 2022. URL https://arxiv.org/abs/2211.04446.

Soham De, Leonard Berrada, Jamie Hayes, Samuel L Smith, and Borja Balle. Unlocking high-accuracy differentially private image classification through scale. *arXiv preprint* arXiv:2204.13650, 2022.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In *2009 IEEE Conference on Computer Vision and Pattern Recognition*, pp. 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.

Tim Dockhorn, Tianshi Cao, Arash Vahdat, and Karsten Kreis. Differentially private diffusion models. *arXiv preprint arXiv:2210.09929*, 2022.

Jinshuo Dong, Aaron Roth, and Weijie J Su. Gaussian differential privacy. Journal of the Royal Statistical Society: Series B (Statistical 6Methodology), 84(1):3–37, 2022.

Cynthia Dwork. Differential privacy. In International colloquium on automata, languages, and programming, pp. 1–12. Springer, 2006.

Cynthia Dwork, Krishnaram Kenthapadi, Frank McSherry, Ilya Mironov, and Moni Naor. Our data, ourselves: Privacy via distributed noise generation. In Advances in Cryptology-EUROCRYPT 2006: 24th Annual International Conference on the Theory and Applications of Cryptographic Techniques, St. Petersburg, Russia, May 28-June 1, 2006. Proceedings 25, pp. 486–503. Springer, 2006.

Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for efficiently improving generalization, 2021. URL https://arxiv.org/abs/2010. 01412.

Robert M. French. Catastrophic forgetting in connectionist networks. Trends in Cognitive Sciences, 3(4):128–135, 1999. ISSN 1364-6613. doi: https://doi.org/10. 1016/S1364-6613(99)01294-2. URL https://www.sciencedirect.com/science/ article/pii/S1364661399012942.

Mudasir A Ganaie, Minghui Hu, Ashwani Kumar Malik, Muhammad Tanveer, and Ponnuthurai N
Suganthan. Ensemble deep learning: A review. Engineering Applications of Artificial Intelligence, 115:105151, 2022.

Arun Ganesh, Mahdi Haghifam, Milad Nasr, Sewoong Oh, Thomas Steinke, Om Thakkar, Abhradeep Thakurta, and Lun Wang. Why is public pretraining necessary for private model training?, 2023. URL https://arxiv.org/abs/2302.09483.

Sahra Ghalebikesabi, Leonard Berrada, Sven Gowal, Ira Ktena, Robert Stanforth, Jamie Hayes, Soham De, Samuel L Smith, Olivia Wiles, and Borja Balle. Differentially private diffusion models generate useful synthetic images. *arXiv preprint arXiv:2302.13861*, 2023.

Frederik Harder, Kamil Adamczewski, and Mijung Park. Differentially private mean embeddings with random features (DP-MERF) for simple & practical synthetic data generation. *CoRR*, abs/2002.11603, 2020. URL https://arxiv.org/abs/2002.11603.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. URL https://arxiv.org/abs/1512.03385.

Peter Kairouz, Sewoong Oh, and Pramod Viswanath. The composition theorem for differential privacy. In *International conference on machine learning*, pp. 1376–1385. PMLR, 2015.

Pang Wei Koh and Percy Liang. Understanding black-box predictions via influence functions, 2020.

URL https://arxiv.org/abs/1703.04730.

Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, CIFAR,
2009.

Wenmin Li, Shunsuke Sakai, and Tatsuhito Hasegawa. Contrastive learning-enhanced trajectory matching for small-scale dataset distillation, 2025. URL https://arxiv.org/abs/2505.

15267.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Zinan Lin, Sivakanth Gopi, Janardhan Kulkarni, Harsha Nori, and Sergey Yekhanin. Differentially private synthetic data via foundation model apis 1: Images, 2024. URL https://arxiv. org/abs/2305.15560.

Dai Liu, Jindong Gu, Hu Cao, Carsten Trinitis, and Martin Schulz. Dataset distillation by automatic training trajectories, 2024. URL https://arxiv.org/abs/2407.14245.

Noel Loo, Ramin Hasani, Mathias Lechner, and Daniela Rus. Dataset distillation with convexified implicit gradients, 2023. URL https://arxiv.org/abs/2302.06755.

Noel Loo, Alaa Maalouf, Ramin Hasani, Mathias Lechner, Alexander Amini, and Daniela Rus.

Large scale dataset distillation with domain shift. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.net/forum?id=0FWPKHMCSc.

Harsh Mehta, Abhradeep Thakurta, Alexey Kurakin, and Ashok Cutkosky. Large scale transfer learning for differentially private image classification, 2022. URL https://arxiv.org/ abs/2205.02973.

Ilya Mironov. Renyi differential privacy. In ´ 2017 IEEE 30th Computer Security Foundations Symposium (CSF), pp. 263–275. IEEE, August 2017a. doi: 10.1109/csf.2017.11. URL http: //dx.doi.org/10.1109/CSF.2017.11.

Ilya Mironov. Renyi differential privacy. In ´ 2017 IEEE 30th computer security foundations symposium (CSF), pp. 263–275. IEEE, 2017b.

Timothy Nguyen, Zhourong Chen, and Jaehoon Lee. Dataset meta-learning from kernel ridgeregression. In *International Conference on Learning Representations*, 2021a. URL https: //openreview.net/forum?id=l-PrrQrK0QR.

Sergey Ioffe. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*, 2015.