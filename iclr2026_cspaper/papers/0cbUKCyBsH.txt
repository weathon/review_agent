000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 The field of time series forecasting faces a critical performance plateau, where even billion-parameter foundation models struggle to outperform simple linear baselines. We argue this stagnation stems not from model architecture but from a universally adopted yet flawed 'self-stimulation' assumption, where models ignore the external influences that drive real-world systems by predicting the future using only the historical values of time series. Through a control-theoretic lens, we formally prove that this assumption imposes a hard, mathematical barrier on forecasting accuracy. To break this barrier, we introduce Influence-Aware Time Series Forecasting (IATSF), a new paradigm that reframes the task from correlationbased inference to dynamic system modeling. To operationalize this paradigm, we provide two foundational contributions. First, we introduce a leak-free, temporallysynced benchmark—a critical resource for the community—that incorporates textual influences to capture the qualitative or uncertain dynamics missed by traditional variables. Second, we develop FIATS, a lightweight, principled model engineered to interpret these influences. Its novel channel-aware mechanisms allow it to adjust its sensitivity to both textual signals and historical data in a channel-specific manner. Our results demonstrate that explicitly modeling external influences is not just an incremental improvement but the primary path forward for meaningful progress in time series forecasting.

## 1 Introduction

The field of time series forecasting has reached a critical performance plateau. Despite the development of sophisticated deep learning architectures (Nie et al., 2023; Liu et al., 2023; Jin et al., 2023) and even billion-parameter foundation models (Ansari et al., 2024; Shi et al., 2025; Woo et al., 2024), these advanced models deliver only marginal performance gains over simple linear baselines (Zeng et al., 2023; Xu et al., 2023; Toner & Darlow, 2024). We contend this lack of progress is not an issue of model complexity, but stems from a universally adopted yet flawed assumption: "self-stimulation," where models predict the future using only the historical time series obserevation, thereby ignoring the external influences that drive real-world systems. Through a control-theoretic lens, we show that this assumption imposes a mathematical barrier on forecasting accuracy. By implicitly treating unobserved influences as random noise, traditional models are mathematically constrained to predict a blurry, "averaged-out" future, ignoring the sharp patterns caused by specific real-world events. While incorporating pre-defined exogenous variables is a step forward (Arango et al., 2025; Wang et al., 2024b), this approach often lacks the flexibility to capture the nuanced, non-quantifiable events that drive system dynamics. More recent work has turned to textual data (Williams et al., 2025; Aksu et al., 2024; Wang et al., 2024a), but these approaches, particularly those leveraging large language models (LLMs), have often lacked a rigorous theoretical grounding for how influences should be modeled. Our analytical framework provides this missing foundation, explicitly demonstrating that incorporating influence-related context is essential to lower the forecasting error bound.

To break this barrier, we introduce **Influence-Aware Time Series Forecasting (IATSF)**, a new paradigm that reframes the task from merely continuing observed patterns to modeling the dynamic system that generates them. We focus on textual data due to its ubiquity and its ability to encode nuanced, non-quantifiable signals often missed by traditional variables. This approach aligns forecasting with real-world system dynamics and unlocks new potential for interpretability and adaptability.

# Influence-Aware Forecasting: Breaking The Self-Stimulation Barrier In Time Series

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Despite these theoretical advances, practical adoption remains challenging due to the lack of datasets and models that are compatible with influence-aware forecasting. Existing multimodal time series forecasting (TSF) approaches often rely on large language models (LLMs) and datasets (Liu et al., 2024a; Williams et al., 2025) optimized for prompting rather than structured influence modeling. Consequently, these datasets often have: (1) short horizons limiting meaningful influence evaluation; (2) overly simplistic or ambiguous textual descriptions causing information leakage or irrelevance; and (3) poor temporal synchronization between textual and numerical data. To address these limitations and operationalize our theoretical insights, we introduce the Temporal-Synced IATSF benchmark, explicitly designed with leak-free textual influences synchronized to extended, realistic forecasting horizons. To demonstrate the effectiveness of influence-aware forecasting, we propose FIATS (Forecaster for Influence-Aware Time Series), a lightweight, *LLM-free* baseline model designed as an architectural embodiment of our control-theoretic principles. Inspired by our analysis, FIATS reframes the cross-attention mechanism to explicitly model the control process where external influences guide system dynamics. This is achieved through novel mechanisms, including a Channel-Aware Adaptive Sensitivity Modeling (CASM) mechanism and an Influence-Modulated Decoder with Channel- Aware Parameter Sharing (CAPS). This principled design enables FIATS to learn each channel's specific sensitivity to an influence and apply this insight to the forecast, directly operationalizing our theoretical findings.

Our extensive experiments across synthetic, physics-based, and market datasets demonstrate that modeling external influences is not just an incremental improvement but the primary path forward for meaningful progress in time series forecasting. Our key contributions are: - A control-theoretic analysis that reveals intrinsic forecasting barriers caused by the "selfstimulation" assumption and proves that influence-aware modeling reduces error bounds.

- The introduction of IATSF, a paradigm that models time series with external influences, bridging the gap between traditional TSF and real-world dynamic systems.

- The operationalization of IATSF with the Temporal-Synced IATSF benchmark and the LLM-free FIATS model, whose performance gains are shown to stem from principled influence modeling, not architectural complexity.

## 2 Motivation: Tsf From System Analysis Perspective

A fundamental disconnect exists in time series forecasting: while real-world data is generated by dynamic systems shaped by external events, standard models typically operate in a closed loop, using only historical data. This oversight is common even in popular benchmarks like the ETT dataset (Zhou et al., 2021), where crucial external factors like human activity and environmental conditions are ignored even though this system is profoundly affected. While traditional methods like ARIMAX (Majka) can incorporate numerical exogenous variables, they cannot process the rich, qualitative information found in textual sources like news reports or policy updates. Recently, multimodal models have begun to leverage this textual data (Williams et al., 2025; Aksu et al., 2024; Wang et al., 2024a), but their approaches often lack a clear theoretical justification for how influences should be modeled.

To systematically address this qualitative gap, we formally identify and analyze the intrinsic limitations of ignoring qualitative external influences from a dynamical systems perspective1.

## 2.1 Time Series Are Observation Of Real-World Dynamic Systems

Consider a general dynamical system characterized by hidden states Z ∈ R
m, evolving based on historical states and independent external influences (Khalil, 2002; Ogata, 2010; Franklin et al., 2010):

$$Z_{f}=F(Z_{h},U_{t}),\quad X=O(Z)$$
Zf = F(Zh, Ut), X = O(Z) (1)
where F represents the true system dynamics, Ut denotes time-varying independent external influences, O represents observation, X for the the observed signal. For analytical clarity, we assume full observability, i.e. X = Z. We also discuss a simple linear system case Xf = AXh + BUt, Traditional forecasting adopts a *self-stimulation* paradigm where models fθ attempt to approximate system dynamics using only historical observations:

$$f_{\theta}^{*}=\arg\operatorname*{min}_{\theta}\mathbb{E}\left[\|\epsilon\|^{2}\right]=\arg\operatorname*{min}_{\theta}\mathbb{E}\left[\|F(X_{h},U_{t})-f_{\theta}(X_{h})\|^{2}\right]$$

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

## 3 Iatsf: Influence-Aware Time Series Forecasting 3.1 Task Formulation

We propose Influence-Aware Time Series Forecasting (IATSF) to overcome the self-stimulation limitation. The key innovation lies in explicit influence modeling:

$$f_{\theta}^{*}=\arg\operatorname*{min}_{\theta}\mathbb{E}\left[\|F(X_{h},U_{t})-f_{\theta}(X_{h},U_{t})\|^{2}\right]$$

where Ut represents measurable influences. This paradigm enables breaking the error bound in proposition 2.1 through influence-aware learning, as detailed in Fig. 1. As shown above, instead of assuming the external influence stays the same in the TSF, IATSF aims to predict a *conditioned future* with the observed or predicted influence even though it is not fully observed or precise. The error reduction mechanism is formalized through our second proposition: Proposition 3.1 P
(Partial influence Efficacy). For a system with p independent influences Ut =
p i=1 U
i t*, incorporating any known influence* U
j treduces the error covariance by:
∆Cov(ϵ) = ∇UjFΣj (∇UjF)
⊤ (6)
For linear systems, this reduces the lower bound by BjΣjB⊤
j.

Token-wise Time Series Decoder

![2_image_0.png](2_image_0.png)

The critical limitation stems from implicitly treating unobserved influences as hidden random variables
Ut ∼ PU . This induces an irreducible forecasting error, as formalized by our first proposition: Proposition 2.1 (Self-Stimulation Error Bound). For any self-stimulated model fθ, it converges to
predicting conditional expectation F
∗(Xh, µ) ≜ EU [F(Xh, U)], the prediction error covariance
satisfies:
anonymous:  $where\ \mu=\mathbb{E}(U_{t}),\quad\Sigma=Cov(U_{t}).$ $\ For\ linear\ systems,\ this\ fails\ back\ to:$  $\square$
⊤ (4)
$$C o v(\epsilon)\stackrel{.}{\succeq}B\Sigma B^{\top}$$
Proposition 2.1 reveals two fundamental limitations: 1) Self-stimulated models converge to predicting conditional expectations, rather than true dynamics, explaining prevalent averaging effects in practice as shown in Fig. 1, and 2) An irreducible error floor exists due to influence stochasticity. This establishes a theoretical performance ceiling for conventional TSF approaches.

(, )
 = { 
Train  (, )

 =  =  (, 
()  **= {** 
 , **? ? ~(, )**
∗(, ) 

$$(2)$$
$$({\mathfrak{I}})$$
$$(4)$$

Collapsed Pattern due to Self-Stimulate Assumption FIATS (Influence-Aware) **FIATS w/o Influence**

$$(5)$$

Same Encoder Arch.

Similar Pattern Good Performance TSF: DLinear **TSF: PatchTST**
Figure 1: The real system runs under various influences. The influence-Aware method can effectively approximate the real system according to the dataset while traditional self-stimulated method can only approximate a average scenario with persistent error, lead to bad or even collapse result. The right panel shows visualization result of a frequency modulated system which is very sensitive to the influence, i.e. large ∇U F.

where A governs self-stimulated state transitions and B encodes influence sensitivity. Standard forecasting datasets D = {(X
(i)
h, X(i)
f)}
N
i=1 are generated through sliding window on the observed signals, where Xh, Xf stand for look-back window and forecasting horizon segment accordingly.

2.2 THE IMPLICIT SELF-STIMULATION ASSUMPTION IN TSF Proposition 3.1 demonstrates that *any measurable influence information* reduces forecasting uncertainty, even with incomplete influence knowledge. This motivates our key insight: textual descriptions of influences provide viable information for uncertainty reduction, despite non-numeric formats.

## 3.2 Language As An Influence Modality

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Incorporating exogenous variables is a common approach (Arango et al., 2025; Wang et al., 2024b), but it typically requires numerical time series or one-hot encoded inputs sampled at the same rate as the target series—even when the actual influences are sparse. This limits flexibility, especially when new events occur. In real-world settings, many impactful factors—such as weather anomalies, geopolitical shifts, or human decisions—are hard to quantify but still essential for accurate forecasting. To address this, we propose modeling influences using linguistic descriptors, which naturally capture compositional and relational semantics through lexical encoding. This allows for expressive representations of complex events (e.g., "simultaneous port strikes and agricultural subsidies") without incurring combinatorial overhead. This design offers several key advantages: Expert Knowledge Integration: Textual interfaces facilitate the direct inclusion of domain-specific expertise via natural language specifications (e.g., "anticipated regulatory changes will suppress industrial output"). This makes it easier to incorporate human input or LLM-driven forecasting through linguistic conditioning of influences.

Generalizability: Textual representations provide flexibility across various contexts, allowing models to generalize more effectively to new or unseen influence scenarios. The use of natural language reduces reliance on rigid, pre-encoded numerical data, enabling better adaptability to diverse situations. Cross-Modal Influence-Modulating: By embedding both linguistic influence descriptors and their temporal effects in a shared space, neural architectures can learn latent mappings that help modulate the forecasting according to the influence.

## 4 Iatsf Benchmark 4.1 Leak-Free Dataset Design

The IATSF benchmark is explicitly constructed to be leak-free, adhering to the principle that models must not access future system states. To enforce this, we only include **independently** evolving influences—external factors that influence the system but are not themselves outcomes of it. Including variables that directly describe or summarize the time series trajectory (as in (Liu et al., 2024a; Jin et al., 2023)) would violate this principle by introducing future state information; see Appendix N for further discussion.

Since system responses to influences often occur much faster than the sampling interval (e.g., photovoltaic panels react to sunlight in milliseconds), we assume influences take effect instantaneously and denote the up-to-date influence as Uf . In deployment, ground-truth future influences are unavailable, so our benchmark restricts inputs to: (1) **Known information** (e.g., holidays); (2) Predictions of Uf from expert sources (e.g., weather reports); and (3) **Hypothetical events** for
"what-if" scenario analysis. Evaluation strategies accounting for prediction errors in influences are detailed in Appendix B.3.

## 4.2 Iatsf Datasets

Each instance in IATSF is defined as D =
n((X
(i)
h
, U(i)
f, D), X(i)
f
)
oN
i=1
, comprising historical time series Xh, future-aligned influences Uf , channel descriptors D, and the ground truth future Xf . The primary challenge in creating such a benchmark is sourcing influences that are both time-synced and truly independent of the system's state, a requirement that makes standard datasets like ETT (Zhou et al., 2021) unsuitable. Our benchmark addresses this gap by providing datasets across three distinct categories designed for IATSF validation, with full details in Appendix O. Toy Systems for Theoretical Validation This category provides a controlled environment to isolate the impact of influences and empirically verify our theoretical propositions without the noise of complex real-world dynamics. It includes: (1) **Frequency Modulated Toy**, a fully synthetic system where influences precisely control signal frequency, offering a theoretical error bound of zero for 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 a perfect model; and (2) **Electricity Utility**, which uses real-world appliance data augmented with simple, discrete textual influences like holidays to test the model on basic, real-world patterns. Complex Real-World Systems To test our paradigm on more challenging real-world problems, we use two complex systems where forecasts can be aided by actual influential information. To ensure the external factor is independent and easily obtainable, we use publicly available weather forecasts as the influence. We then evaluate forecasting performance on two distinct systems whose dynamics are affected by weather: (1) **Atmospheric Physics**, which is an ideal system for this task as its variables (e.g., solar radiation, air pressure, dew point) are intrinsically linked to the weather condition. For example, a forecast of "clear skies" allows an IATSF model to infer high solar radiation, a connection a self-stimulated TSF model cannot make. (2) **NYC Traffic Speed**, where the link is less direct but still significant. Urban traffic is potentially influenced by weather; for instance, a "heavy rain" forecast may correlate with slower traffic due to reduced visibility and slick roads. This tests the model's ability to extract more subtle, correlational signals from the influence text. Human-Driven Business Systems This category evaluates IATSF's ability to model volatile market dynamics where historical patterns are often unreliable due to specific events (e.g., a product update). Textual influences are often the primary signal for future performance, especially for new products with limited history (the 'cold-start' problem). The **Game Active User Dataset (GAUD)** tracks daily active users for 90 games, with developer logs as influences, testing the model's practical utility for business decision-making under uncertainty.

## 5 Fiats: A Simple System-Aware Baseline Model For Iatsf

Having established through control theory that breaking the 'self-stimulation' barrier is essential for forecasting progress, we now introduce the model designed to achieve this. We propose FIATS, the architectural embodiment of our **Influence-Aware Time Series Forecasting (IATSF)** paradigm. While recent studies (Aksu et al., 2024; Williams et al., 2025; Liu et al., 2024a; Wang et al., 2024a; Niu et al., 2025) explore text-informed forecasting with large language models (LLMs), their architectural complexity and significant overhead obscure whether performance gains stem from genuine influence modeling or simply from increased model capacity. To provide a rigorous and interpretable validation of the IATSF framework, we designed FIATS as the first LLM-free, numerical-based forecaster built from first principles for this task. As illustrated in Fig. 2, FIATS integrates a standard patch-based time series encoder with a novel influence semantic encoder and decoder that operate on text embeddings directly, avoiding the variance and token overhead associated with generative LLMs. The novelty is as follows: Temporal-Synced Influence Real-world systems often respond rapidly to influences, necessitating temporal alignment between text and time series data. FIATS addresses this by synchronizing each time series patch with the last influence observed, e.g. for patch start from 10:15, sync with the last timestep with influence update of 10:00. This ensures the model uses only leak-free, contemporaneous influences when forecasting subsequent patches, preventing future information leakage while maintaining temporal relevance.

Channel-aware Adaptive Sensitivity Modeling (CASM) Proposition 3.1 shows that the error reduction depends on the system's sensitivity to the influence. CASM is designed to explicitly model this sensitivity for each channel, learning how a given influence (e.g., 'clear skies') should affect different time series (e.g., solar radiation vs. atmospheric pressure). Starting from linear systems where time series are observed by Xf = CZf = CAZh + CBUf , channel-specific sensitivity to influences is governed by dxif dUf
= c iB. This indicates that each channel responds differently to external influences. The error analysis is discussed in Appendix B.4. Cross-attention provides an ideal framework for this, as its core mechanism naturally computes a weighted alignment between two sets of inputs—in our case, mapping textual influences (keys) to specific time series channels (queries). Specifically, to capture this without introducing excessive parameters, we introduce Channel-aware Adaptive Sensitivity Modeling Block, as shown in the right panel of the Fig. 2:
- Query as Channel-wise Sensitivity C˜ = *Desc* · WQ: Channel descriptions *Desc* ∈ R
CN×D are served as query (CN as channel number). The query projection explicitly learns how textual channel features (e.g., "atmospheric pressure") influence influence sensitivity for each channel. This allows the model to adjust how influences are perceived based on channel-specific characteristics.

- *Key as influence Filter* B˜Uf = (*News* · WK)
⊤: The key projection maps temporal-synced news embeddings *News* ∈ RM×D to a system sensitivity matrix (M as news number), allowing the model to filter out irrelevant influences (e.g., excluding "tech stock news" when forecasting atmospheric physics). This ensures that only pertinent influences are considered for each system.

![5_image_0.png](5_image_0.png)

෩ = 
 
⋅ 

෩ =  ⋅  ෩ 
= ⋅  
⨀
Token-wise Time Series Decoder
 ∈ ℝ×× 
 ∈ ℝ××× … × 
Self-Attention × 
 ∈ ℝ
CASM Block

 ∈ ℝ×
 ∈ ℝ××× 
 ∈ ℝ××× 
Influence Encoder
 ∈ ℝ××  ∈ ℝ×××  **∈ ℝ**××× 
- Value as influence Translator U˜f = *News* · WV : Value projection learns to maps news text embed-
Scalar Vector
 = Attn **, ,  =** 
softmax ෩ 

ୃ
෩ Sensitivity Weight ෩ Figure 2: **Architecture of FIATS.** FIATS integrates three inputs: time series data from a look-back window, temporal-synced news embeddings, and channel description embeddings. The influence encoder employs CASM blocks in a residual connection along with multiple self-attention layers to enhance feature extraction. The CAPS influence-modulated decoder projects the historical time series embeddings into the future, guided by channel-aware, time-synced influences. A token-wise decoder is used to prevent overfitting in the final linear layer, as discussed in (Lee et al., 2023).

The above analysis show that the attention mechanism can effectively generate the channel-aware influence U
c f. This design allows identical influences to differentially impact channels based on their descriptions. Unlike static sensitivity coefficients found in classical systems, this formulation maintains the nonlinear characteristics provided by the transformer block, allowing for greater learning flexibility to approximate complex nonlinear system. Additionally, it aligns well with the theoretical framework, making the model more *interpretable*. The attention map produced by the CASM layer directly reveals the sensitivity of each channel to various influences, providing clear insights into how influences impact different channels based on their specific descriptions. Channel-Aware Parameter Sharing (CAPS) While CASM addresses heterogeneous influence responses, channels also exhibit inherent differences in their temporal patterns - a critical factor neglected by conventional parameter sharing. Previous shared models approximate all channels with a same set of parameters introducing persistent errors ϵi = oi(Z) −
1 k Pk j=1 oj (Z) where oi for real system channel-specific dynamics. To mitigate this issue, FIATS introduces a lightweight channel-aware decoding mechanism. All channels are first encoded into a shared latent space Z˜ by a unified time-series encoder. Then, a channel-conditioned decoder is used to adaptively project this latent representation into a channelaware space, conditioned by the channel-specific time-synced influence embeddings U
cf
. decoder approximates channel-specific adjustments by modulating the shared latent space through crossattention *Attention*(Q = U
c t
, K, V = Z˜) to simulate such nonlinear projection. To avoid future information leakage, we apply causal attention mask here. We will omit the analysis. This design introduces minimal overhead while enabling the model to account for channel heterogeneity in a flexible, data-driven manner. Additionally, the attention maps produced by the channel-aware decoder are interpretable: they reveal how each channel selectively attends to historical time series data under different influences. We provide visualizations and further analysis of these attention patterns in the following session.

## 6 Experiments

Baseline Models FIATS is benchmarked against several state-of-the-art (SOTA) methods. These include linear-based models (Zeng et al., 2023; Xu et al., 2023) , transformer-based models (Nie 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 et al., 2023; Liu et al., 2023), and fine-tuned LLM-based multimodal method (Jin et al., 2023). Additionally, we compare pretrained time series "foundation models" (Ansari et al., 2024; Woo et al., 2024; Shi et al., 2025). This selection covers a range of approaches, including self-stimulated linear and nonlinear models, data-specific and pretrained models, LLM-based cross-modal models.

## 6.1 Rq1: Can Iatsf Overcome The Limitations Of Self-Stimulation?

To directly test our theoretical claims from Section 2, we evaluate FIATS on our two toy systems, which provide controlled environments to isolate the impact of influences. The **FM Toy** dataset offers a theoretical error bound of zero, while the **Electricity Utility** dataset tests the paradigm on real data with simple, discrete influences (e.g., holidays). Statistical Results The results provide strong empirical evidence for our theory. On the FM Toy dataset, Table 1 shows that FIATS achieves a near-zero error, directly approaching the theoretical lower bound. In stark contrast, all self-stimulated TSF methods, including massive pre-trained models, fail spectacularly. This confirms that the performance bottleneck is indeed the flawed "self-stimulation" assumption, not model scale. The visualization in Fig. 1 shows these models producing collapsed, averaged-out forecasts, perfectly aligning with Proposition 2.1. On the Electricity dataset, FIATS
again demonstrates SOTA performance by effectively leveraging minimal textual cues. Interestingly, even a powerful model like TimeLLM shows some capability here, suggesting that when influences are simple and causal links are obvious, large models can partially succeed, but our lightweight, principled approach is more effective and efficient.

Table 1: Forecasting result in MSE, comparing the *influence-aware FIATS* against various TSF
methods. The best result is highlighted in bold and the second best is highlighted in underscore.

Dataset Pred.

Len. FIATS FITS DLinear PatchTST iTrans. Chronos-L MOIRAI-L Time-MoE-U TimeLLM
14 **0.003** 0.282 0.151 0.006 0.136 0.012 0.013 0.012 0.231 28 **0.008** 0.692 0.297 0.029 0.295 0.047 0.062 0.035 0.382 60 **0.020** 0.909 0.442 0.075 0.494 0.129 0.133 0.107 0.551 FM Toy 120 **0.027** 0.883 0.632 0.168 0.747 0.374 0.385 0.295 0.788 96 **0.124** 0.134 0.140 0.130 0.148 0.154 0.152 0.149 0.131 192 **0.144** 0.149 0.153 0.149 0.162 0.177 0.171 0.168 0.152 336 **0.158** 0.165 0.169 0.166 0.178 0.197 0.192 0.183 0.160 Electricity Utility 720 **0.190** 0.203 0.204 0.210 0.225 0.242 0.236 0.229 0.192 96 **0.443** 0.973 0.957 0.858 0.858 0.913 0.997 0.980 0.974 192 **0.609** 1.161 1.123 1.031 1.026 1.217 1.272 1.250 1.232 336 **0.685** 1.306 1.262 1.176 1.195 1.512 1.594 1.421 1.575 NYC
Traffic Speed 720 **0.710** 1.457 1.378 1.275 1.295 1.799 1.825 1.592 1.729 96 **0.182** 0.248 0.294 0.252 0.267 0.293 0.299 0.258 0.294 192 **0.205** 0.297 0.340 0.304 0.327 0.357 0.356 0.318 0.342 336 **0.235** 0.354 0.393 0.364 0.404 0.448 0.457 0.413 0.393 Atmospheric Physics 2014-19 720 **0.281** 0.430 0.456 0.439 0.495 0.512 0.532 0.508 0.461 96 **0.410** 0.436 0.487 0.464 0.456 0.447 0.453 0.437 -
192 **0.438** 0.524 0.568 0.567 0.578 0.552 0.557 0.542 - 336 **0.455** 0.601 0.644 0.644 0.698 0.685 0.673 0.647 -
Atmospheric Physics 2014-24 720 **0.497** 0.692 0.725 0.745 0.832 0.754 0.765 0.734 -
6.2 RQ2: DOES IATSF EXCEL IN COMPLEX REAL-WORLD SYSTEMS? Having validated our theory in controlled settings, we now investigate if the IATSF paradigm provides meaningful gains in noisy, complex systems. We use two distinct datasets, **Atmospheric Physics** and **NYC Traffic Speed**, which are both potentially correlated with the weather condition - an independent influence. The Atmospheric Physics system has strong, direct physical links to weather, while the NYC Traffic system has a more subtle, indirect relationship. This setup allows us to test whether FIATS can successfully extract these different types of correlations from the training data and leverage them to outperform self-stimulated models that are blind to this external context.

Statistical Results The results confirm FIATS's ability to capitalize on external information. As shown in Table 1, FIATS consistently outperforms all baselines, achieving an average MSE reduction of 36.0% on Atmospheric Physics and 44.3% on NYC Traffic Speed compared to the strongest self-stimulated baseline, PatchTST. This performance gap highlights that even for complex systems, providing external context is critical. Pretrained models like Chronos-L, despite their vast training data, underperform FIATS, underscoring that scaling data alone cannot compensate for missing, crucial influence information. Table 2 further breaks down performance by channel, showing that FIATS achieves substantial gains even on variables not directly mentioned in the weather reports (e.g., pressure p, air density ρ, and vapor pressure VPdef), demonstrating its ability to infer latent correlation.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 Table 2: A selection of channel-wise performance on Atmos.

Phy. 2014-19 dataset in MSE.

Channel FIATS FITS DLinear PatchTST iTrans. IMP. p (mbar) **0.136** 0.863 0.823 0.930 1.032 **83.43%** Tpot (K) **0.182** 0.316 0.352 0.322 0.353 42.18%
VPdef (mbar) **0.283** 0.638 0.696 0.674 0.803 **55.59%**
rho (g/m³) **0.192** 0.390 0.411 0.418 0.453 **50.73%** raining (s) **0.790** 0.873 0.937 0.859 0.994 8.04%
SWDR (W/m²) **0.182** 0.308 0.385 0.296 0.377 38.39%
Case Study: Visualization and Controllability To understand why FIATS excels, Fig. 3 visualizes three representative channels from the Atmospheric Physics dataset. The first channel, atmospheric pressure (p), is sensitive to regional climate shifts but lacks strong short-term historical correlation. Its slow, subtle changes are challenging for traditional TSF models. PatchTST fails to capture these dynamics, defaulting to a flat prediction, while FIATS successfully models the trend by conditioning on relevant influences. The second channel, rainfall duration, is sparse and lacks periodicity. PatchTST outputs near-zero values—its conditional expectation under uncertainty—while FIATS adjusts its predictions based on available influence signals. It correctly forecasts the first rainfall event but misses the second due to misaligned or absent external information, reflecting a candid dependence on accurate influence input. The third channel, solar radiation (SWDR), is not explicitly mentioned in the influence but is indirectly implied. FIATS captures its phase and amplitude accurately, thanks to the CASM design that enables cross-channel sensitivity modeling. PatchTST, by comparison, produces generic, misaligned waveforms.

![7_image_0.png](7_image_0.png) 
Figure 3: Visualization of three channels on the 15,000th test sample of the Atmos. Phy. 2014-19 dataset. Blue indicates ground truth, Red shows FIATS, Green represents PatchTST, and Orange denotes FIATS with swapped influences on the second and fourth forecast days. The CAPS influencemodulated decoder exhibits distinct attention patterns across channels.

## 6.3 Rq3: How Does Iatsf Handle Human-Driven Market Dynamics?

We next evaluate IATSF's ability to model systems driven by human decisions and external events using the **GAUD** dataset. This scenario presents unique challenges, including high variability and cold-start problems for new games, where historical data is sparse but influence information (e.g., developer logs, marketing) is available.

![7_image_1.png](7_image_1.png)

Imp rovem ent w.
Figure 4: Performance improvement with respect to the PatchTST on each time series in GAUD.

As shown in Fig. 4, FIATS consistently outperforms PatchTST, achieving an average improvement of 12.6% and ranking first on 59.6% of the games. The advantage is most pronounced for games released after 2021, where short time series cause traditional models to fail. FIATS's ability to generalize from textual influences allows it to deliver robust forecasts even in these data-scarce, cold-start scenarios. This demonstrates the paradigm's practical utility for business decision-making and market analysis.

## 6.4 Rq4: What Makes The Fiats Architecture Effective And Robust?

Finally, we conduct a series of ablations and analyses to verify that the observed performance gains stem from our principled architectural design rather than confounding factors.

Interpretability via Attention Maps The CASM block analysis in Fig. 5 shows how the model focuses on different temporal features across layers. In the first layer, attention centers on the first sentence, providing temporal context for daily and annual periodicity. The second layer shifts attention to channel-specific signals, particularly the sixth sentence describing atmospheric pressure, reflecting the model's sensitivity to channel-specific patterns and influences. By the third layer, attention 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

![8_image_0.png](8_image_0.png)

Figure 5: Attention map of the CASM (i.e. sensitivity weight) on the 15000th test sample of Atmos. Phy. 2014-19 dataset. We use three cross attention block in residual connection. The horizontal axis stands for channels and vertical stands for the 7 sentences of the weather report summary. diversifies, focusing on relevant influence aspects for each channel. The CAPS influence-modulated decoder, shown in Fig. 3, demonstrates distinct attention patterns across channels, highlighting the model's ability to align time series data with textual influences. Channels associated with periodic variables like SWDR exhibit clear periodicity in attention maps, indicating effective capture of cyclical patterns. The rainfall channel highlights historical rainfall, showcasing the model's sensitivity to key moments. This adaptability is driven by CASM, enabling the model to tailor its attention based on each channel's unique characteristics.

Table 3: Ablation result on Atmos. Phy. 201419 in MSE.

Pred.

Len.

Openai 512 MiniLM mpnet Zero Desc.

Zero News 96 **0.182** 0.186 0.196 0.209 0.249 192 **0.205** 0.214 0.216 0.260 0.302 336 0.235 **0.232** 0.251 0.302 0.359 720 0.281 **0.272** 0.291 0.356 0.432 Ablation Studies We first prove the necessity of our core components. As shown in Table 3, when we remove influence inputs entirely ("Zero News"), performance drops to that of a self-stimulated model, proving that the gains come from the influences themselves. Crucially, removing channel descriptions ("Zero Desc.") also significantly degrades performance, confirming the critical role of the CASM
mechanism in modeling channel-specific sensitivities. Next, we test how the quality of influences affects the model. Fig. 6 shows that while FIATS is robust to minor semantic noise, performance degrades as influence inputs become less accurate, a finding that directly supports Proposition 3.1. Finally, we confirm the architecture's generalizability by swapping text embedding models; Table 3 shows that performance remains stable across different embedding spaces. Summary of Findings Our experiments decisively validate the IATSF paradigm. Controlled experiments (RQ1) confirm our controltheoretic analysis, showing FIATS approaches the theoretical error bound while even the largest foundation models fail without influence data. This success extends to complex real-world systems (RQ2, RQ3), where FIATS consistently outperforms SOTA baselines. Crucially, architectural analyses (RQ4) attribute these gains to our principled design choices—CASM and CAPS—not model scale. The results demonstrate that IATSF, operationalized through the interpretable and robust FIATS model, represents a validated, theoretically-grounded, and efficient path forward for the field.

## 7 Conclusion, Limitation & Future Work

This paper presents Influence-Aware Time Series Forecasting (IATSF), leveraging a control-theoretic framework to address errors from the self-stimulation assumption and improve forecasting accuracy through influence modeling.

We demonstrate the effectiveness of IATSF using the Temporal-Synced IATSF benchmark and the FIATS model, which outperforms state-of-the-art methods, including those based on large language models. Our findings emphasize that influence-aware modeling, rather than simply increasing model complexity, is crucial for enhancing forecasting performance. While FIATS shows some capability in noise tolerance and generalization, challenges persist in modeling complex chaotic systems, where influences may not have immediate effects and varying credibility of news sources or temporal misalignment could lead to inaccurate influence observations. Overcoming these challenges will require more advanced models, potentially benefiting from pretraining techniques. These areas will be explored in future research. Additionally, the analysis framework can inspire further exploration, such as modeling multichannel correlation.

![8_image_1.png](8_image_1.png) 

## Broader Impact, Ethic Statement And Code Availability

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 We use Large Language Models (LLMs), including ChatGPT and Gemini, solely for polishing the writing of this paper.

## References

Openai embedding api. https://platform.openai.com/docs/guides/embeddings. URL https://
platform.openai.com/docs/guides/embeddings.

Taha Aksu, Chenghao Liu, Amrita Saha, Sarah Tan, Caiming Xiong, and Doyen Sahoo. Xforecast:
Evaluating natural language explanations for time series forecasting, 2024. URL https:// arxiv.org/abs/2410.14180.

Abdul Fatir Ansari, Lorenzo Stella, Caner Turkmen, Xiyuan Zhang, Pedro Mercado, Huibin Shen, Oleksandr Shchur, Syama Sundar Rangapuram, Sebastian Pineda Arango, Shubham Kapoor, Jasper Zschiegner, Danielle C. Maddix, Hao Wang, Michael W. Mahoney, Kari Torkkola, Andrew Gordon Wilson, Michael Bohlke-Schneider, and Yuyang Wang. Chronos: Learning the language of time series, 2024. URL https://arxiv.org/abs/2403.07815.

Sebastian Pineda Arango, Pedro Mercado, Shubham Kapoor, Abdul Fatir Ansari, Lorenzo Stella, Huibin Shen, Hugo Senetaire, Caner Turkmen, Oleksandr Shchur, Danielle C. Maddix, Michael Bohlke-Schneider, Yuyang Wang, and Syama Sundar Rangapuram. Chronosx: Adapting pretrained time series models with exogenous variables, 2025. URL https://arxiv.org/abs/2503. 12107.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding, 2019. URL https://arxiv.org/
abs/1810.04805.

Gene F. Franklin, J. David Powell, and Abbas Emami-Naeini. *Feedback Control of Dynamic Systems*.

Pearson, Upper Saddle River, NJ, 6th edition, 2010. ISBN 9780136019695.

Furong Jia, Kevin Wang, Yixiang Zheng, Defu Cao, and Yan Liu. Gpt4mts: Prompt-based large language model for multimodal time-series forecasting. In *Proceedings of the AAAI Conference on* Artificial Intelligence, volume 38, pp. 23343–23351, 2024.

Ming Jin, Shiyu Wang, Lintao Ma, Zhixuan Chu, James Y Zhang, Xiaoming Shi, Pin-Yu Chen, Yuxuan Liang, Yuan-Fang Li, Shirui Pan, et al. Time-llm: Time series forecasting by reprogramming large language models. *arXiv preprint arXiv:2310.01728*, 2023.

Hassan K. Khalil. *Nonlinear Systems*. Prentice Hall, Upper Saddle River, NJ, 3rd edition, 2002.

ISBN 0130673897.

Seunghan Lee, Taeyoung Park, and Kibok Lee. Learning to embed time series patches independently.

arXiv preprint arXiv:2312.16427, 2023.

Haoxin Liu, Shangqing Xu, Zhiyuan Zhao, Lingkai Kong, Harshavardhan Kamarthi, Aditya B.

Sasanur, Megha Sharma, Jiaming Cui, Qingsong Wen, Chao Zhang, and B. Aditya Prakash. Time-mmd: Multi-domain multimodal dataset for time series analysis, 2024a. URL https:
//arxiv.org/abs/2406.08627.

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here. We comply with intellectual property agreements for all data sources. Data are properly anonymized and content generated by OpenAI API is free for general use, with no concerns regarding sensitive or illegal activity in our dataset.

The code for TGForecaster and dataset samples are available at: https://anonymous.4open. science/r/IATSF_review-F624.

## Llm Usage Statement

Mengpu Liu, Mengying Zhu, Xiuyuan Wang, Guofang Ma, Jianwei Yin, and Xiaolin Zheng. Echo-gl:
Earnings calls-driven heterogeneous graph learning for stock movement prediction. In *Proceedings* of the AAAI Conference on Artificial Intelligence, volume 38, pp. 13972–13980, 2024b.

Yong Liu, Tengge Hu, Haoran Zhang, Haixu Wu, Shiyu Wang, Lintao Ma, and Mingsheng Long.

itransformer: Inverted transformers are effective for time series forecasting. *arXiv preprint* arXiv:2310.06625, 2023.

Marcin Majka. Arimax: Time series forecasting with external variables. Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, and Jayant Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In *International Conference on Learning* Representations, 2023.

Wenzhe Niu, Zongxia Xie, Yanru Sun, Wei He, Man Xu, and Chao Hao. Langtime: A languageguided unified model for time series forecasting with proximal policy optimization, 2025. URL https://arxiv.org/abs/2503.08271.

Katsuhiko Ogata. *Modern Control Engineering*. Prentice Hall, Upper Saddle River, NJ, 5th edition, 2010. ISBN 9780136156734.

Ramit Sawhney, Arnav Wadhwa, Shivam Agarwal, and Rajiv Shah. Fast: Financial news and tweet based time aware network for stock trading. In *Proceedings of the 16th conference of the european* chapter of the association for computational linguistics: main volume, pp. 2164–2175, 2021.

Xiaoming Shi, Shiyu Wang, Yuqi Nie, Dianqi Li, Zhou Ye, Qingsong Wen, and Ming Jin. Timemoe: Billion-scale time series foundation models with mixture of experts, 2025. URL https:
//arxiv.org/abs/2409.16040.

Kaitao Song, Xu Tan, Tao Qin, Jianfeng Lu, and Tie-Yan Liu. Mpnet: Masked and permuted pre-training for language understanding. *Advances in neural information processing systems*, 33:
16857–16867, 2020.

William Toner and Luke Darlow. An analysis of linear time series forecasting models. arXiv preprint arXiv:2403.14587, 2024.

Wenhui Wang, Furu Wei, Li Dong, Hangbo Bao, Nan Yang, and Ming Zhou. Minilm: Deep selfattention distillation for task-agnostic compression of pre-trained transformers. *Advances in Neural* Information Processing Systems, 33:5776–5788, 2020.

Xinlei Wang, Maike Feng, Jing Qiu, Jinjin Gu, and Junhua Zhao. From news to forecast: Integrating event analysis in llm-based time series forecasting with reflection, 2024a. URL https:// arxiv.org/abs/2409.17515.

Yuxuan Wang, Haixu Wu, Jiaxiang Dong, Yong Liu, Yunzhong Qiu, Haoran Zhang, Jianmin Wang, and Mingsheng Long. Timexer: Empowering transformers for time series forecasting with exogenous variables. *arXiv preprint arXiv:2402.19072*, 2024b.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Andrew Robert Williams, Arjun Ashok, Étienne Marcotte, Valentina Zantedeschi, Jithendaraa Subramanian, Roland Riachi, James Requeima, Alexandre Lacoste, Irina Rish, Nicolas Chapados, and Alexandre Drouin. Context is key: A benchmark for forecasting with essential textual information, 2025. URL https://arxiv.org/abs/2410.18959.

Gerald Woo, Chenghao Liu, Akshat Kumar, Caiming Xiong, Silvio Savarese, and Doyen Sahoo.

Unified training of universal time series forecasting transformers, 2024. URL https://arxiv. org/abs/2402.02592.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. *Advances in Neural Information Processing* Systems, 34:22419–22430, 2021.

Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, and Mingsheng Long. Timesnet:
Temporal 2d-variation modeling for general time series analysis. In *International Conference on* Learning Representations, 2023.

Zhijian Xu, Ailing Zeng, and Qiang Xu. Fits: Modeling time series with 10k parameters. arXiv preprint arXiv:2307.03756, 2023.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? 2023.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In *International Conference* on Machine Learning, 2022a.

Tian Zhou, Ziqing Ma, xue wang, Qingsong Wen, Liang Sun, Tao Yao, Wotao Yin, and Rong Jin.

FiLM: Frequency improved legendre memory model for long-term time series forecasting. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022b. URL https://openreview.net/forum?id= zTQdHSQUQWc.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.

Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11106–11115, 2021.