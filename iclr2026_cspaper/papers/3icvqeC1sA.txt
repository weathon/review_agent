000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Chaosnexus: A Foundation Model For Univer- Sal Chaotic System Forecasting With Multi- Scale Representations

Accurately forecasting chaotic systems, prevalent in domains including weather prediction and fluid dynamics, remains a significant scientific challenge. The inherent sensitivity of these systems to initial conditions, coupled with a scarcity of observational data, severely constrains traditional modeling approaches. Since these models are typically trained for specific systems, they lack zero-shot or fewshot capabilities on novel or data-limited scenarios. While emerging foundation **REVISE** models address this via pretraining on multiple systems, existing architectures typically operate at a single resolution, often failing to capture the intrinsic multiscale temporal structures where distinct dynamical patterns unfold. To overcome this limitation, we introduce ChaosNexus, a universal forecasting model driven by our ScaleFormer architecture. It explicitly captures the multi-scale structure of chaotic dynamics with a U-Net-inspired design, enabling the simultaneous modeling of fine-grained fluctuations and coarse-grained trends. Augmented with Mixture-of-Experts layers and a wavelet-based frequency fingerprint, the model can generalizes across heterogeneous dynamical regimes. On a large-scale testbed comprising over 9,000 synthetic chaotic systems, it demonstrates notable improvements in the fidelity of long-term attractor statistics while achieving competitive point-wise forecasting accuracy compared to the leading baseline. This robust performance extends to real-world applications with exceptional data efficiency. For instance, in 5-day global weather forecasting, ChaosNexus achieves a competitive zero-shot mean error below 1°C, a result that further improves with few-shot finetuning. Moreover, experiments on the scaling behavior of ChaosNexus provide a guiding principle for scientific foundation models: cross-system generalization stems from the diversity of training systems, rather than sheer data volume.

Chaotic systems, characterized by their deterministic nature yet high sensitivity to initial conditions, are ubiquitous in the natural world and across diverse scientific and engineering disciplines, including weather forecasting (Shukla, 1998; Rind, 1999), fluid dynamics (Yorke & Yorke, 2005; Najm, 2009), and neural processes (Jia et al., 2023; Vignesh et al., 2025). The intrinsic complexity of such systems renders accurate forecasting both an essential and formidable task, particularly in realworld contexts where data acquisition is resource-intensive and observational records are sparse. While this sensitivity makes precise long-term point-wise prediction impossible, the system's behavior is not entirely random; it is confined to a complex geometric structure known as a strange attractor (Rossler, 1976; Grassberger & Procaccia, 1983), which possesses unique and invariant sta- ¨ tistical properties. An effective forecasting model should not only predict the short-term evolution but also reproduce the long-term geometry and statistics of the system's attractor.

The intrinsic difficulty of forecasting chaotic systems is further compounded by the challenge of data sparsity. Traditional system-specific models (Srinivasan et al., 2022; Brenner et al., 2022; Hess et al., 2023) typically require extensive and high-quality observational data from a novel system to accurately infer its underlying dynamics and attractor geometry, creating a significant bottleneck in practical applications. This has motivated a recent paradigm shift toward pretraining a single, univer- **REVISE** sal model (Jiao et al., 2025; Hemmer & Durstewitz, 2025; Lai et al., 2025), based on the proposition

## Abstract

1

## 1 Introduction

Anonymous authors Paper under double-blind review 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107

## 2 Related Works

Chaotic System Forecasting. Forecasting chaotic systems is a central challenge in science and engineering. Reservoir computing (RC)-based methods (Srinivasan et al., 2022; Gauthier et al., 2021; **REVISE** that a model exposed to a vast and heterogeneous collection of observational data spanning diverse dynamical systems and operating regimes can learn a rich repertoire of underlying patterns and principles common to chaotic behavior. By leveraging large-scale data during pretraining, such a model can then be applied to a target system with little or no in-distribution data. This strategy is designed to exploit cross-system similarities to compensate for downstream data sparsity, thereby reducing the burden of data acquisition and enhancing out-of-distribution forecasting performance. Existing works, notably Panda (Lai et al., 2025) and DynaMix (Hemmer & Durstewitz, 2025), in- **REVISE** stantiate this paradigm through distinct architectural designs. Panda demonstrates its feasibility by pretraining Transformer blocks on a large-scale corpus of synthetic chaotic ODE systems, achieving strong zero-shot forecasts on unseen dynamical systems. DynaMix explores this direction by using a mixture of almost-linear RNN experts with delay- and sinusoidal-based embeddings to reconstruct long-term statistics of novel low-dimensional dynamics. However, individual chaotic systems exhibit multi-scale temporal structure: essential dynamical patterns unfold across a continuum of time scales, and different systems may concentrate energy in widely separated frequency bands. An architecture that operates at a single temporal resolution must either truncate long-range dependencies, oversmooth fast oscillations, or conflate behaviors that live on distinct scales, thereby obscuring system-specific attractor geometries and degrading long-horizon stability. Consequently, although Panda and DynaMix achieve strong zero-shot performance on many benchmarks, their lack of an explicit representation of this intrinsic multi-scale structure may limit out-of-distribution generalization performance when applied to more heterogeneous chaotic dynamics. To overcome these obstacles, we introduce ChaosNexus, a foundation model for universal chaotic dynamics forecasting. At its core is our proposed ScaleFormer, a U-Net-inspired Transformer architecture designed to master the multi-scale nature of chaotic systems. Its encoder progressively models fine-grained to coarse temporal contexts through hierarchical patch merging, while the symmetric decoder, aided by skip connections, reconstructs fine-grained details via patch expansion. To facilitate robust cross-system generalization, each Transformer block is equipped with a Mixture-of- Experts (MoE) layer that allocates specialized parameters to different dynamical regimes on top of a shared backbone. Furthermore, we condition the model on a frequency fingerprint derived from a wavelet scattering transform, providing a stable spectral signature that captures the system's intrinsic oscillatory and modulatory behavior. ChaosNexus is pretrained on the chaotic-system corpus introduced by Panda (Lai et al., 2025), **REVISE** consisting of approximately 20,000 synthetically generated ODE systems. Training is guided by a composite objective that jointly enforces short-term predictive accuracy and the preservation of longterm statistical properties. Through extensive experiments, we show that ChaosNexus sets a new state-of-the-art in zero-shot forecasting on chaotic benchmarks. Its remarkable sample efficiency is further highlighted on real-world weather forecasting: ChaosNexus achieves zero-shot temperature MAE below 1
◦C, outperforming competitive baselines even when they are fine-tuned on more than 470K samples from the target system. Finally, our scaling analysis reveals a key design principle for future chaotic foundation models: generalization benefits more from increasing the diversity of systems in the pretraining corpus than from increasing the number of trajectories per system. Our **REVISE** primary contributions are summarized as follows: - We propose ChaosNexus, a foundation model for chaotic system forecasting strengthened by explicitly considering the multi-scale structure of chaotic dynamics, enhancing its out-of-distribution generalization performances on diverse systems.

- We design a multi-scale ScaleFormer architecture that couples hierarchical temporal representations with Mixture-of-Experts layers and a wavelet-based frequency fingerprint to capture the multi-scale temporal and spectral structure of chaotic dynamics while allocating specialized parameters to individual systems and dynamical regimes.

- We show that ChaosNexus attains state-of-the-art zero-shot performance on thousands of synthetic chaotic systems and strong zero-shot accuracy on 5-day global weather forecasting.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Li et al., 2024) represent a key advance: they employ fixed read-in weights to lift inputs into the high-dimensional state space of a randomly initialized reservoir, while training only a linear readout. Concurrently, deep learning models like recurrent neural networks (RNNs) have proven effective, though they often require techniques such as teacher forcing to counteract training instabilities like exploding gradients on chaotic trajectories (Brenner et al., 2022; Hess et al., 2023). More recent works aim to preserve the geometric and statistical properties of system attractors within neural operators. This is achieved through methods like evolution regularization with optimal transport and Maximum Mean Discrepancy (MMD), or by imposing mathematical constraints such as unitarity that leverage system ergodicity (Cheng et al., 2025; He et al., 2025). Despite their success, these frameworks are specialized models, designed and trained for a single, specific system. This inherent lack of generalization renders them impractical for real-world chaotic systems where data is often sparse and systems are unseen, precluding their application in zero-shot or few-shot forecasting. Out-of-distribution Generalization in Dynamical Systems. Out-of-distribution generalization in dynamical systems is a rapidly growing area of research. Norton et al. (2025) demonstrated that reservoir computers can generalize to unobserved basins of attraction in multistable systems when trained on sufficiently rich transient dynamics, thereby learning a global representation from a single basin. Another prominent strategy involves decomposing system dynamics into shared and specific components, where a base model captures common physical laws and low-dimensional vectors encode system-specific characteristics, leveraging data from multiple regimes to learn fundamental representations of the underlying dynamics (Brenner et al., 2024; Wang et al., 2025; Huang et al., 2023). A complementary paradigm focuses on pretraining foundation models on large syn- **REVISE** thetic datasets encompassing diverse governing equations, parameter regimes, and initial conditions (Nzoyem et al., 2025; Subramanian et al., 2023; Herde et al., 2024; McCabe et al., 2024; Seifner et al., 2024), and most of these works target PDEs with rich spatiotemporal structure. Within the domain of ODE-based chaotic systems, Panda (Lai et al., 2025) trains Transformer blocks on a large-scale corpus of synthetic chaotic systems and demonstrates strong zero-shot forecasting performance on many unseen systems. DynaMix (Hemmer & Durstewitz, 2025) instead employs a mixture of almost-linear RNN experts with delay- and sinusoidal-based embeddings to reconstruct long-term statistics of chaotic dynamics. Although these works clearly demonstrate the benefits of pretraining for generalization, their architectural designs largely overlook the inherent multi-scale temporal structure of chaotic dynamics. In contrast, we propose a U-Net–inspired multi-scale Transformer backbone, ScaleFormer, equipped with per-scale MoE layers and a wavelet-based frequency fingerprint, which explicitly encodes multi-scale temporal and spectral structure and improves outof-distribution generalization across thousands of heterogeneous chaotic systems.

## 3 Methodology

Problem Statement and Model Overview. We address the problem of chaotic system forecasting:
given historical observations X1:T = (x1, x2, *· · ·* , xT ) ∈ R
T ×Vspanning T times of a chaotic system with V variables, we forecast its successive H steps, *i.e.,* XˆT +1:T +H = fθ(X1:T ) ∈ R
H×V,
where fθ denotes the forecasting model. Here, we aim to design a foundation model fθ that can directly produce faithful forecasting results based on historical observations, with little or no further in-distribution data required for training. We demonstrate the overall architecture of ChaosNexus in Figure 1, which comprises three key components: (i) input dynamics embedding, (ii) the Scale- Former backbone, and (iii) frequency-enhanced joint scale readout. The details of our framework are shown as follows.

## 3.1 Input Dynamics Embedding

In chaotic systems, instantaneous observations are often noisy and insufficient to reveal the governing dynamics. We therefore segment the input trajectory X ∈ R
T ×Vinto S = ⌊
T
D
⌋ + 1 non-overlapped temporal patches of length D. Each patch P ∈ R
D×Vencapsulates a short-time trajectory segment, thereby providing essential local dynamical context. Motivated by Koopman theory (Koopman, 1931; Mauroy et al., 2020; Brunton et al., 2021), which posits that nonlinear dynamics can be linearized by lifting them to a suitable high-dimensional space of observables, we first enrich each patch with random polynomial and Fourier features (Appendix C.1), an approach 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

![3_image_0.png](3_image_0.png)

adopted from recent work (Lai et al., 2025). The augmented patch is then mapped to an embedding u with embedding dimension de via a linear layer.

## 3.2 Scaleformer Architecture

The patch embeddings are then fed into the ScaleFormer, an encoder-decoder architecture composed of stacked Transformer blocks. Instead of applying standard attention to patches flattened across all dimensions with O(S
2V
2) complexity, each Transformer block employs dual axial attention. This mechanism factorizes the computation by performing attention sequentially along the variable and temporal axes, reducing the overall complexity to O(S
2 + V
2). Crucially, the variable attention module can capture the strong coupling between variables—a fundamental property of chaotic dynamics often absent in standard time series. To better accommodate different sequence lengths and enhance generalization, we employ rotary positional embeddings (RoPE) (Su et al., 2024) instead of conventional absolute positional encodings. We also employ pre-normalization to enhance training stability and FlashAttention (Dao et al., 2022) to improve efficiency. Given an input patch embedding up, the computational flow of our modified Transformer block is:

$$\mathbf{h}_{p}=\text{VA}(\text{RN}(\mathbf{u}_{p}))+\mathbf{u}_{p},\qquad\bar{\mathbf{h}}_{p}=\text{TA}(\text{RN}(\mathbf{h}_{p}))+\mathbf{h}_{p},\qquad\bar{\mathbf{h}}_{p}=\text{MoE}(\text{RN}(\bar{\mathbf{h}}_{p}))+\bar{\mathbf{h}}_{p},\tag{1}$$

where VA and TA are axial variable and temporal attention operations, respectively. RN denotes the root mean square (RMS) layer normalization (Zhang & Sennrich, 2019). We replace the standard feed-forward network (FFN) with a Mixture-of-Experts (MoE) layer (Dai et al., 2024), which allows a single model to distinguish the dynamics of multiple chaotic systems by enabling different experts to specialize in their unique characteristics. The MoE layer consists of M specialist experts and one shared expert, which are all implemented with standard feed-forward layers. A gating network activates a sparse combination of these experts for each input. Its output is a weighted sum of the 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Encoding and Patch Merging. The encoder blocks progressively builds a hierarchy of representations at increasingly coarse resolutions. Following each Transformer block at level i, a patch merging layer reduces the temporal resolution by a factor of two while doubling the feature dimension. This down-sampling is achieved by concatenating the features of adjacent temporal patches and applying a learnable linear projection. Given the output of the i-th encoder block, H
(i)
enc ∈ R
S
2i−1 ×V ×2 i−1de, the patch merging is formulated as:

$\mathbf{H}_{\rm enc}^{\prime}={\rm Concat}(\mathbf{H}_{\rm enc}^{(i)}[0::2,\ldots],\mathbf{H}_{\rm enc}^{(i)}[1::2,\ldots])\mathbf{W}_{\rm enc}^{(i)}+\mathbf{b}_{\rm enc}^{(i)},$
where the output H
′(i)
enc ∈ R
S
2i ×V ×2 ideserves as the input to the next encoder level. This allows successive layers to capture features ranging from fine-grained details to coarse, global structures. The hierarchical encoding process culminates in a bottleneck layer positioned at the deepest level of the architecture, which consists of a linear layer that processes the feature representation at the coarsest temporal scale, bridging the transition from the encoding path to the decoding path. Decoding and Patch Expansion. The decoder blocks reconstructs the high-resolution representation from the low-dimensional features produced by the encoder and a final bottleneck layer.

Each decoder block is followed by a patch expansion layer that reverses the merging process.

It up-samples the features by doubling the temporal resolution and halving the channel dimension via a linear transformation and a reshape operation. For the i-th decoder level, the input H
(i)
dec ∈ R
S
2i ×V ×2 ideis expanded, producing an output H
′(i)
dec ∈ R
S
2i−1 ×V ×2 i−1deas follows:

$${\mathbf{H}}_{\mathrm{dec}}^{\prime(i)}=\mathrm{Reshape}({\mathbf{W}}_{\mathrm{dec}}^{(i)}{\mathbf{H}}_{\mathrm{dec}}^{(i)}+{\mathbf{b}}_{\mathrm{dec}}^{(i)}),$$
$$(6)$$
dec), (6)
Skip Connections. To mitigate the loss of fine-grained information during down-sampling, we introduce skip connections linking encoder and decoder blocks at corresponding resolutions. The output H
(i)
enc from the i-th encoder layer is passed through a dedicated skip connection block implemented with 1D convolutions and then fused with the up-sampled features H
′(i)
dec from the corresponding decoder layer. This fusion provides the decoder with direct access to high-resolution encoder features, which is crucial for accurate reconstruction of the system's dynamics. Further details are provided in Appendix C.2.

## 3.3 Frequency-Enhanced Joint Scale Readout

The decoder of ScaleFormer produces a set of representations {H
(i)
dec}
L
i=1 capturing system dynamics at L different temporal scales. To synthesize these into a single, comprehensive representation for forecasting, we first apply temporal mean pooling to each decoder output to obtain system-level features H¯ (i)for each scale. These features are then concatenated and projected through a linear fusion layer to produce a unified dynamics representation Huni ∈ R
de×Vcontains integrated multiscale information:

$\mathbf{H}_{\rm uni}={\rm Concat}(\mathbf{H}^{(1)},\mathbf{H}^{(2)},\cdots,\mathbf{H}^{(L)})\mathbf{W}_{f}+\mathbf{b}_{f}$.  
A robust foundation model must not only model temporal evolution but also identify the underlying dynamical system or its current regime. To this end, we condition our model on frequency-domain information, which serves as a fingerprint for the system's dynamics. We employ the wavelet scattering transform on the historical observations X to extract a stable, multi-scale summary of its spectral content (Appendix C.3). The resulting scattering coefficients, Fw ∈ R
C×T
′×V, are temporally pooled to yield a single frequency fingerprint, F¯w ∈ R
C×V. It distills the system's intrinsic

$$\phi_{i,p}=\begin{cases}s_{i,p},&s_{i,p}\in\mathrm{TopK}(\{s_{j,p}\}_{j=1}^{M},K),\\ 0,&\mathrm{otherwise},\end{cases}$$  $t+1,p=\mathrm{Sigmoid}(\mathbf{W}_{M+1}\bar{\mathbf{h}}_{p}),\qquad s_{:,p}=\mathrm{Softmax}(\mathbf{W}\bar{\mathbf{h}}_{p}),$
where si,p is the score of the i-th specialist expert. Ws are trainable parameters.

shared expert and the top K specialist experts:

$$\mathrm{MoE}(\bar{\mathbf{h}}_{p})=\phi_{M+1,p}\mathrm{FFN}_{M+1}(\bar{\mathbf{h}}_{p})+\sum_{i=1}^{M}(\phi_{i,p}\mathrm{FFN}_{i}(\bar{\mathbf{h}}_{p})),$$
$$(2)$$
$$({\mathfrak{I}})$$
$$(4)$$
$$({\mathfrak{H}})$$

## 3.4 Training Objective

The total objective function for ChaosNexus is composed of three distinct components: a primary forecasting loss, an auxiliary load balancing loss for the MoE layers, and a distributional regularization term to preserve the system's statistical properties. The primary training objective is the Mean Squared Error (MSE), which measures the point-wise accuracy, formulated as:

$${\mathcal{L}}_{\rm mse}=\frac{1}{B}\sum_{n=1}^{B}||{\hat{\mathbf{X}}}_{T+1:T+H}^{n}-{\mathbf{X}}_{T+1:T+H}^{n}||_{2}^{2},\tag{1}$$
$$({\boldsymbol{T}})$$

where Wo and bo are learnable parameters. This allows the model to leverage both the learned multiscale temporal patterns and the intrinsic spectral properties of the system for accurate prediction.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

## 4.1 Zero-Shot Forecasting

Setups. We utilize the benchmark dataset consisting of synthetic chaotic systems from Panda (Lai **REVISE** et al., 2025). Its training set contains 20K novel chaotic ODEs, generated synthetically by an evolutionary algorithm that evolved from 129 known systems (Gilpin, 2021; 2023). The data was further As is standard for Mixture-of-Experts (MoE) models, relying solely on the prediction loss can lead to expert load imbalance, where the gating network disproportionately favors a small subset of experts (Shazeer et al., 2017). This leaves other experts under-trained and limits the model's overall capacity. To mitigate this, we incorporate an auxiliary load balancing loss from Dai et al. (2024):

$\mathcal{L}_{\text{balance}}=M\sum_{i=1}^{M}f_{i}r_{i}$, (13.1)
$$({\mathfrak{s}})$$
$$(\mathbf{9})$$
where fiis the fraction of patches routed to expert i, and riis the average routing probability assigned to it. This encourages more uniform expert utilization. Due to the sensitive dependence on initial conditions in chaotic systems, point-wise accuracy is often insufficient for long-horizon forecasting. A robust forecast must also reproduce the geometric and statistical properties of the system's attractor. To enforce this, we introduce a regularization term based on the Maximum Mean Discrepancy (MMD), which minimizes the divergence between the state distribution of predicted trajectories and that of the ground-truth trajectories (Appendix C.4):

$${\cal L}_{\rm reg}=\frac{1}{B^{2}}\sum_{i,j}\kappa(\hat{\mathbf{X}}^{i},\hat{\mathbf{X}}^{j})+\frac{1}{B^{2}}\sum_{i,j}\kappa(\mathbf{X}^{i},\mathbf{X}^{j})-\frac{2}{B^{2}}\sum_{i,j}\kappa(\hat{\mathbf{X}}^{i},\mathbf{X}^{j}),\tag{10}$$

where {Xˆ n}
B
n=1 and {Xn}
B
n=1 represent batches of the full predicted and ground-truth trajectories.

Following prior work, we use a mixture of rational quadratic kernels for the kernel function κ (Schiff et al., 2024; Seeger, 2004; Reiss et al., 2019). The final objective function is a weighted sum of these three components: L = Lmse + λ1Lbalance + λ2Lreg, where λ1, λ2 are hyperparameters that control the relative weights of the auxiliary loss terms.

## 4 Experiments

In this section, we present comprehensive experiments to evaluate the forecasting capabilities of our proposed model. Due to space constraints, we present the main findings here and provide further in-depth analyses, including supplementary benchmark results, extensive ablation studies, model sensitivity and internal mechanics, as well as visualizations of forecasting cases in Appendix A.

where Xˆ n and Xn are the predicted and ground-truth of the n-th trajectory in a batch with size B.

$${\hat{X}}_{T+1:T+H}=\mathrm{Concat}(H_{\mathrm{uni}},{\bar{F}}_{w})W_{o}+b_{o},$$
XˆT +1:T +H = Concat(Huni,F¯w)Wo + bo, (7)
oscillatory and modulatory behaviors into a fixed-size representation, enhancing the model's ability to distinguish between different dynamical systems. The final multi-step forecast is produced by a linear prediction head that combines the unified dynamics Huni and the frequency fingerprint F¯w:
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 diversified with dynamics-preserving augmentations like time-delay embedding (Takens, 2006). The held-out test set, used for evaluation, comprises 9.3K systems derived from a disjoint seed population (Appendix D.1). We use symmetric mean absolute percentage error (sMAPE) (Lai et al., 2025) **REVISE** of 128 and 512 timesteps to evaluate the point-wise forecasting accuracy. We also consider the correlation dimension error (Dfrac), the Kullback–Leibler (KL) divergence between system attractors (Dstsp), the largest Lyapunov exponent error (DLyap), and the weighted mean energy error (MELRw)
to evaluate the fidelity in key statistical properties of system attractors (Zhang & Gilpin, 2024). These complementary metrics jointly assess both point-wise accuracy and long-term preservation of attractor geometry, which are essential to whether the model has captured the underlying chaotic dynamics. We compare our proposed method against several state-of-the-art time series foundation models with different parameter sizes, including Panda (Lai et al., 2025), Time-MoE (Shi et al., **REVISE** 2024), TimesFM (Das et al., 2024), Chronos (Ansari et al., 2024), Moirai-MoE (Liu et al., 2024a), Timer-XL (Liu et al., 2024b), DynaMix (Hemmer & Durstewitz, 2025), Parrot (Zhang & Gilpin, 2025), where '-S', '-B, '-L' refer to small, base, large in parameter size, respectively. To assess the adaptability of general-purpose models to this specific domain, we also include Chronos-S-SFT, a variant of the Chronos-S model that has been fine-tuned on our chaotic systems training corpus. For all other baseline models, we load their officially released pre-trained weights for evaluation. We **REVISE** choose these baselines because they are all foundation models intended for generalization, aligning with our zero-shot evaluation on previously unseen chaotic systems. Details of experimental setups are demonstrated in Appendix D. Results. We conduct a zero-shot evaluation on the held-out test set of chaotic systems. For a fair comparison, all models use a context length of 512 to autoregressively forecast 512 steps into the future. While ChaosNexus and the Panda baseline are pretrained on the chaotic systems corpus, other baselines are general-purpose time-series foundation models, for which we employ the official pretrained weights. As shown in Figure 2, ChaosNexus demonstrates point-wise accuracy competi- **REVISE** tive with the baseline, achieving an average sMAPE of 68.901 at 128 steps. Regarding the long-term dynamics, ChaosNexus exhibits superior fidelity. It reduces the average correlation dimension error (Dfrac) to 0.203. Notably, it attains an average KL divergence of attractors (Dstsp) of 1.206. Table 2 in Appendix A.4 further demonstrates the superior performance of ChaosNexus on DLyap and MELRw. Given that the sensitive dependence on initial conditions renders any long-term pointwise forecast of a chaotic system ultimately unreliable (Li et al., 2021; Jiang et al., 2023; Schiff et al., 2024), the strong performance of ChaosNexus in long-term statistical metrics is therefore **REVISE** compelling evidence that it can infer intrinsic dynamics of new systems from the contexts rather than superficial pattern memorizing. Notably, leading general-purpose time-series foundation models, despite being pretrained on larger time-series datasets than ours (Appendix D.3), struggle on chaotic system forecasting. We also observe that their generalization capabilities can be improved (from Chronos-SFT-S) after further fine-tuned on chaotic systems corpus. This contrast provides compelling evidence for our claim that chaotic dynamics possess unique differences from general time series. It also validates the necessity of building domain-specific foundation models on chaotic

![6_image_0.png](6_image_0.png)

Figure 2: Zero-shot forecasting performances of models on synthetic chaotic systems. Each box shows the median (center line), the middle 50% of results (box), and the overall range (whiskers). The inset plot shows the mean performance with the 95% CI of ChaosNexus and Panda. Asterisks **REVISE** indicate statistically significant differences determined by the Wilcoxon signed-rank test (*: p < 0.05, **: p < 0.01).

![7_image_0.png](7_image_0.png)

data and underscores the importance of the specialized architectural designs for multi-scale feature extraction and system disentanglement in ChaosNexus.

## 4.2 Few-Shot Forecasting

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Setups. Weather is an inherently chaotic system (Lorenz, 1969; 1982; 2017). For a rigorous evaluation on a real-world chaotic system, we utilize the WEATHER-5K dataset (Han et al., 2024). This dataset comprises hourly meteorological data from 5,672 global weather stations over a 10-year period from 2014 to 2023. It is then chronologically split, with data from 2014 to 2021 used for training, 2022 for validation, and 2023 for testing. Each sample includes five variables: temperature, dew point, wind speed, wind direction, and sea-level pressure. Given the profound real-world importance of forecasting absolute values, we primarily employ the Mean Absolute Error (MAE)
to directly measure the discrepancy between predicted and ground-truth observations. MAE is the **REVISE**
gold-standard metric in this application, as researchers value the absolute accuracy of these weatherrelated variables. The forecasting task is to predict the subsequent 120 hours of all variables given 512 hours of historical context. To assess few-shot performance under data-scarce conditions, we fine-tune models on two small subsets of the training data: 0.1% (85K samples) and 0.5% (473K samples). In all few-shot experiments, ChaosNexus is first pretrained on the synthetic chaotic sys- **REVISE** tems corpus and then fine-tuned on exactly the same WEATHER-5K subsets as the baselines, which are trained from scratch without pretraining.Besides foundation models included in Section 4.1, we select several strong deep learning baselines in this benchmark, including FEDformer, CrossFormer, PatchTST, and Koopa. They are widely adopted architectures for time-series forecasting, making them appropriate references for this single-system, real-world benchmark. We also report the performance of our model in a zero-shot setting, without any fine-tuning on the weather dataset. Further details of setups are provided in Appendix F. Results. Figure 3 presents the forecasting results for the temperature variable. Remarkably, Chaos- Nexus in a zero-shot setting—without any fine-tuning—surpasses all baselines in their few-shot configurations. It achieves a mean error strictly below 1°C for 5-day (120-hour) global temperature forecasts. In stark contrast, the baseline models exhibit an MAE of at least 3°C, even when fine-tuned on the same data. The performance of ChaosNexus further improves with few-shot finetuning, especially for longer prediction horizons. This suggests that while pre-training endows the model with a robust, universal understanding of chaotic behavior, fine-tuning allows it to adapt these principles to the specific physical constraints and periodicities (e.g., diurnal and seasonal cycles) inherent in meteorological systems. This process grounds the model's abstract dynamical representations in real-world physics, enhancing its ability to generate accurate and stable long-term forecasts. Detailed results of all weather variables and performances of foundation models are shown in the ADD Appendix A.6. We find that foundation models designed for chaotic system forecasting and trained on our corpus of synthetic chaotic dynamics, including ChaosNexus, Panda, and Chronos-S-SFT,
perform significantly better than those trained on general time series, even though they use a much larger corpus (see Table 9). It demonstrates that pretraining specifically on chaotic systems provides a more relevant inductive bias for weather forecasting. Moreover, ChaosNexus also outperforms Panda on many variable forecasting tasks, highlighting the contribution of our multi-scale architectural designs.

![8_image_0.png](8_image_0.png) 

## 4.3 Scaling Behavior

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

## 4.4 Multi-Scale Feature Analysis

An investigation into scaling behavior is crucial for the development of foundation models, since understanding how model performance scales with key factors such as parameter count and data volume is essential for guiding future research and resource allocation. Parameter Scaling. We first explored the impact of model size on performance. We generated a suite of models with varying parameter counts, ranging from 2.83M to 52.63M, by systematically adjusting the number of encoder and decoder layers, as well as the dimension de of the embedding space. The results demonstrated in Figure 4(a) reveal a consistent trend: increasing the model's parameter count yields steady improvements in performance. For instance, scaling the model from 2.83M to 52.63M parameters improved the sMAPE@128 by 49.83%, which demonstrates that larger models possess a greater capacity to capture the complex dynamics inherent in the data. Data Scaling. We further investigated the model's performance as a function of the training data size under two distinct settings. First, we fix the diversity, *i.e.,* the total number, of training systems, while varying the number of trajectories sampled from each system, leading to only different training time points. Second, we increase the diversity of systems while holding the number of training time points constant. From Figure 4(b), we find that merely increasing the number of time points for a fixed set of systems did not lead to a significant enhancement in zero-shot performance. In contrast, Figure 4(c) demonstrates that increasing the number of distinct systems in the training set substantially improved the model's ability to generalize. These findings also support established **REVISE** research (Norton et al., 2025; Lai et al., 2025)on data scaling. While prior work, such as (Lai et al., 2025), establishes the scaling law for system diversity, which our Figure 4(c) corroborates, our complementary analysis in Figure 4(b) provides a refinement. The negligible gain from scaling per-system data volume suggests that effective generalization is driven by corpus-level diversity, i.e., the number of systems rather than by per-system trajectories. To investigate the inner workings of our multi-scale architecture, we visualize the input signal's patch partitioning alongside the temporal attention maps from shallow and deep layers of both the encoder and decoder. As illustrated in Figure 5 and 8, we select three systems from the test set with progressively weaker regularity (left to right in Figure 5), thus increasing the forecasting difficulty.

Patch Partition Patterns. We find that the shallow layers, which operate on smaller patches, are adept at capturing local, high-frequency fluctuations. In contrast, the deeper layers, processing merged patches that represent longer time intervals, focus on capturing long-term trends and global structures. This is particularly evident in 5(b), where a shallow-layer patch may encompass only a peak or a trough, whereas a deep-layer patch spans an entire peak-valley cycle.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539

![9_image_0.png](9_image_0.png) 

Temporal Attention Patterns of Encoder Layers. The encoder's attention patterns distinctly reflect this multi-scale processing. The deep encoder layers (upper right of each subfigure) consistently exhibit globalized attention distributions, indicating a focus on synthesizing long-range dependencies. The shallow encoder layers (upper left), however, display system-specific patterns. For the highly regular system in 5(a), the map forms a Toeplitz-like structure (Bajwa et al., 2007), analogous to a convolutional operation, suggesting the model applies fixed-pattern filters to scan the time series. For the more complex system in 5(c), the attention forms distinct blocks, indicating that the model concentrates on specific temporal segments whose interplay is deemed critical for understanding the system's state. The system in 5(b) presents a hybrid pattern, blending the features of 5(a) and 5(c) to capture its intermediate complexity. Temporal Attention Patterns of Decoder Layers. The decoder's attention mechanisms operate differently, functioning primarily as a selector. This aligns with our architectural design, where the decoder's outputs are mean-pooled over the temporal dimension for the final forecast. The model must therefore learn to select and combine specific patterns from the historical context to support its predictions. The deep decoder layers show a pronounced focus on the final patch, capturing the most recent temporal dependencies crucial for autoregressive prediction. The shallow decoder layers, conversely, appear to anticipate future dynamics; for instance, in 5(b), after observing a descending phase, the model intensifies its attention on historical ascending patterns, selectively weighting the context that is most relevant for the anticipated future trajectory.

## 5 Conclusions

We introduce ChaosNexus, a foundation model that features a universal, pre-trained approach to chaotic system forecasting, effectively overcoming data sparsity. Its novel multi-scale ScaleFormer architecture, augmented with Mixture-of-Experts layers and a wavelet-based frequency fingerprint, achieves state-of-the-art zero-shot performance by accurately predicting both point-wise evolution and long-term attractor properties. Crucially, our scaling analysis reveals that generalization is driven by the diversity of systems in the pre-training corpus, not the sheer volume of trajectories per system. This key insight provides a clear roadmap for developing powerful, data-efficient models for complex scientific applications.

## 540

541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 The authors have read and adhered to the ICLR Code of Ethics. The research presented in this paper is foundational and focuses on the modeling of chaotic systems, with primary applications in scientific domains such as meteorology. All data used for training and evaluation is either synthetically generated from mathematical principles or derived from publicly available, non-personal scientific datasets, ensuring no privacy concerns. This work does not involve human subjects, and we do not foresee any direct negative societal impacts or risks of perpetuating social biases. Our aim is to advance the scientific understanding and predictive capabilities for complex physical systems for the benefit of the scientific community.

## Reproducibility Statement

We are committed to ensuring the reproducibility of our research. The complete source code for the ChaosNexus model, along with scripts for data processing, training, and evaluation, is publicly available in an anonymous repository at https://anonymous.4open.science/r/ ChaosNexus-C809. We acknowledge the authors of previous open-source projects (Lai et al., **REVISE**
2025) whose codebases served as a foundation for our implementation. A detailed description of our proposed ScaleFormer architecture, including the patch merging/expansion mechanisms and the Mixture-of-Experts layers, is provided in Section 3. A comprehensive breakdown of implementation details for key components, such as input feature augmentation, skip connections, the wavelet scattering transform, and the MMD regularization term, can be found in Appendix C. Detailed descriptions of the datasets are provided in the appendices: the generation process and augmentations for the synthetic chaotic systems are in Appendix D.1, and the specifics of the WEATHER-5K benchmark are in Appendix F.1. All hyperparameters used for our model variants are explicitly listed in Table 8 in Appendix B. The full experimental protocol, including training procedures and the precise definitions of our evaluation metrics, is detailed in Appendix D.2 and E. All baseline models used in our comparisons are described in Appendix D.3 and F.2.

## References

Joakim Anden and St ´ ephane Mallat. Deep scattering spectrum. ´ IEEE Transactions on Signal Processing, 62(16):4114–4128, 2014.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, et al. Chronos: Learning the language of time series. *arXiv preprint arXiv:2403.07815*, 2024.

Waheed U Bajwa, Jarvis D Haupt, Gil M Raz, Stephen J Wright, and Robert D Nowak. Toeplitzstructured compressed sensing matrices. In 2007 IEEE/SP 14th Workshop on Statistical Signal Processing, pp. 294–298. IEEE, 2007.

Manuel Brenner, Florian Hess, Jonas M Mikhaeil, Leonard F Bereska, Zahra Monfared, Po-Chen Kuo, and Daniel Durstewitz. Tractable dendritic rnns for reconstructing nonlinear dynamical systems. In *International conference on machine learning*, pp. 2292–2320. Pmlr, 2022.

Manuel Brenner, Elias Weber, Georgia Koppe, and Daniel Durstewitz. Learning interpretable hierarchical dynamical systems models from time series data. *arXiv preprint arXiv:2410.04814*, 2024.

Joan Bruna and Stephane Mallat. Invariant scattering convolution networks. ´ *IEEE transactions on* pattern analysis and machine intelligence, 35(8):1872–1886, 2013.

Steven L Brunton, Marko Budisiˇ c, Eurika Kaiser, and J Nathan Kutz. Modern koopman theory for ´
dynamical systems. *arXiv preprint arXiv:2102.12086*, 2021.

Xiaoyuan Cheng, Yi He, Yiming Yang, Xiao Xue, Sibo Cheng, Daniel Giles, Xiaohang Tang, and Yukun Hu. Learning chaos in a linear way. *arXiv preprint arXiv:2503.14702*, 2025.

## Ethics Statement

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Damai Dai, Chengqi Deng, Chenggang Zhao, RX Xu, Huazuo Gao, Deli Chen, Jiashi Li, Wangding Zeng, Xingkai Yu, Yu Wu, et al. Deepseekmoe: Towards ultimate expert specialization in mixtureof-experts language models. *arXiv preprint arXiv:2401.06066*, 2024.

Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory- ´
efficient exact attention with io-awareness. *Advances in neural information processing systems*,
35:16344–16359, 2022.

Abhimanyu Das, Weihao Kong, Rajat Sen, and Yichen Zhou. A decoder-only foundation model for time-series forecasting. In *Forty-first International Conference on Machine Learning*, 2024.

Daniel J Gauthier, Erik Bollt, Aaron Griffith, and Wendson AS Barbosa. Next generation reservoir computing. *Nature communications*, 12(1):5564, 2021.

William Gilpin. Chaos as an interpretable benchmark for forecasting and modelling. arXiv preprint arXiv:2110.05266, 2021.

William Gilpin. Model scale versus domain knowledge in statistical forecasting of chaotic systems.

Physical Review Research, 5(4):043252, 2023.

Niclas Goring, Florian Hess, Manuel Brenner, Zahra Monfared, and Daniel Durstewitz. Out-of- ¨
domain generalization in dynamical systems reconstruction. *arXiv preprint arXiv:2402.18377*, 2024.

Peter Grassberger and Itamar Procaccia. Characterization of strange attractors. *Physical review* letters, 50(5):346, 1983.

Tao Han, Song Guo, Zhenghao Chen, Wanghan Xu, and Lei Bai. Weather-5k: A large-scale global station weather dataset towards comprehensive time-series forecasting benchmark. *arXiv e-prints*, pp. arXiv–2406, 2024.

Yi He, Yiming Yang, Xiaoyuan Cheng, Hai Wang, Xiao Xue, Boli Chen, and Yukun Hu.

Chaos meets attention: Transformers for large-scale dynamical prediction. arXiv preprint arXiv:2504.20858, 2025.

Christoph Jurgen Hemmer and Daniel Durstewitz. True zero-shot inference of dynamical systems ¨
preserving long-term statistics. *arXiv preprint arXiv:2505.13192*, 2025.

Maximilian Herde, Bogdan Raonic, Tobias Rohner, Roger Kappeli, Roberto Molinaro, Emmanuel ¨
de Bezenac, and Siddhartha Mishra. Poseidon: Efficient foundation models for pdes. ´ Advances in Neural Information Processing Systems, 37:72525–72624, 2024.

John R Hershey and Peder A Olsen. Approximating the kullback leibler divergence between gaussian mixture models. In *2007 IEEE International Conference on Acoustics, Speech and Signal* Processing-ICASSP'07, volume 4, pp. IV–317. IEEE, 2007.

Florian Hess, Zahra Monfared, Manuel Brenner, and Daniel Durstewitz. Generalized teacher forcing for learning chaotic dynamics. *arXiv preprint arXiv:2306.04406*, 2023.

Zijie Huang, Yizhou Sun, and Wei Wang. Generalizing graph ode for learning complex system dynamics across environments. In Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, pp. 798–809, 2023.

Junen Jia, Feifei Yang, and Jun Ma. A bimembrane neuron for computational neuroscience. Chaos, Solitons & Fractals, 173:113689, 2023.

Ruoxi Jiang, Peter Y Lu, Elena Orlova, and Rebecca Willett. Training neural operators to preserve invariant measures of chaotic attractors. *Advances in Neural Information Processing Systems*, 36:
27645–27669, 2023.

Anran Jiao, Haiyang He, Rishikesh Ranade, Jay Pathak, and Lu Lu. One-shot learning for solution operators of partial differential equations. *Nature Communications*, 16(1):8386, 2025.

Bernard O Koopman. Hamiltonian systems and transformation in hilbert space. Proceedings of the National Academy of Sciences, 17(5):315–318, 1931.