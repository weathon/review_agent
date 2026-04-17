000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 This paper introduces FedMPDD (Federated Learning via Multi-Projected Directional Derivatives), a novel algorithm that simultaneously optimizes bandwidth utilization and enhances privacy in Federated Learning. The core idea of FedMPDD is to encode each client's high-dimensional gradient by computing its directional derivatives along multiple random vectors. This compresses the gradient into a much smaller message, significantly reducing uplink communication costs from O(d) to O(m), where m ≪ d. The server then decodes the aggregated information by projecting it back onto the same random vectors. Our key insight is that averaging multiple projections overcomes the dimension-dependent convergence limitations of a single projection. We provide a rigorous theoretical analysis, establishing that FedMPDD converges at a rate of O(1/K), matching the performance of FedSGD. Furthermore, we demonstrate that our method provides inherent privacy against gradient inversion attacks due to the geometric properties of low-rank projections, offering a tunable privacy-utility trade-off controlled by the number of projections. Extensive experiments on benchmark datasets validate our theory, showing that FedMPDD drastically reduces network congestion and provides strong privacy protection, all while maintaining high model performance, outperforming existing methods in resource-constrained scenarios.

## 1 Introduction

Federated Learning (FL) is a foundational paradigm for collaboratively training models across N edge devices by leveraging their local computational resources (McMahan et al., 2017; Kairouz et al., 2021; Chen & Ran, 2019) to solve the distributed optimization problem:

$${\mathrm{minimize}}_{\mathbf{x}\in\mathbb{R}^{d}}\ f(\mathbf{x})={\frac{1}{N}}\sum_{i=1}^{N}f_{i}(\mathbf{x}).$$

$$(1)$$
fi(x). (1)
Here, fi: R
d → R is the local objective (loss) function of client i, and f : R
d → R is the global objective. Lacking central access to local objectives, FL iteratively communicates between a server and a client subset Ak (| Ak| = βN, where β ∈ (0, 1] denotes the client participation rate). In each round k, the server sends the global model xk. Selected clients compute local updates (e.g., mini-batch gradients gi(xk)) and transmit them to the server. The server aggregates these gradients by averaging: g(xk) = 1 βN
Pi∈Ak gi(xk), and updates the global model: xk+1 = xk − η g(xk),
as in FedSGD (McMahan et al., 2017), where η is a suitable learning rate. Privacy Preservation Measures. Although FL avoids direct data sharing by performing local training, recent studies have shown that a client's raw data can be reconstructed from its transmitted gradients (Zhu et al., 2019; Zhao et al., 2020a; Huang et al., 2021; Yin et al., 2021a; Melis et al.,
2019; Li et al., 2022) via a mechanism known as Gradient Inversion Attacks (GIAs). In GIAs within FL, a honest-but-curious server (or any adversary with the same level of server's global information)
employs a deep neural network as an inversion model. This inversion network is trained to find input data that, when used to compute gradients, closely matches the gradients shared by legitimate clients, thereby revealing private information (Yin et al., 2021b; Geiping et al., 2020; Jere et al., 2020; Nasr et al., 2019). In FL, a common privacy preservation measure is via a local differential privacy (LDP) framework (Wei et al., 2020; Truex et al., 2020; Zhao et al., 2020b; Seif et al., 2020), where each client adds noise to its gradient before uploading it to the server. By injecting noise locally, Anonymous authors Paper under double-blind review

## Abstract

1

# Communication-Efficient And Private Feder- Ated Learning Via Projected Directional Derivative

LDP enhances privacy protection without relying on a trusted aggregator. However, this approach introduces a fundamental trade-off between privacy and convergence (Jere et al., 2020). Communication Cost Management. A key bottleneck in FL algorithms like FedSGD is the substantial uplink communication overhead from transmitting d-dimensional gradients gi(xk) from clients to the server, requiring 32d bits per client per round, assuming single-precision floatingpoint representation (32 bits per value). For instance, a ResNet-18 model (∼ 11 × 106 parameters)
necessitates approximately 42MB transmission per client per round. This high cost severely impacts efficiency in bandwidth-constrained real-world deployments (Chen et al., 2021; Niknam et al., 2020; Shahid et al., 2021; Li et al., 2020). Strategies to reduce FL communication volume or frequency fall into three main classes: model compression, *local computation with client selection*, and *gradient compression*. Model compression reduces global model size, for example, by using smaller models with local representations (Liang et al., 2020). Local computation and client selection decrease communication frequency or the number of clients per round through techniques like multiple local updates (Stich, 2018; McMahan et al., 2017; Karimireddy et al., 2020) and client subset selection (Sattler et al., 2019; Liang et al., 2020). Gradient compression reduces the size of transmitted gradients using methods like quantization (Alistarh et al., 2017; Horvóth et al., 2022; Karimireddy et al., 2019; Shlezinger et al., 2020; Bernstein et al., 2018; Reisizadeh et al., 2020; Suresh et al., 2017), sparsification (Ivkin et al., 2019; Lin et al., 2017), and structured/sketched updates (Konecnˇ y et al., 2016; Wang et al., 2020; Azam et al., 2021; Han et al., 2024; Ivkin et al., `
2019; Cho et al., 2024; Kuo et al., 2024; Qi et al., 2024; Yi et al., 2023; Lin et al., 2022). Statement of Contribution. This paper proposes a novel framework that jointly tackles critical communication efficiency and privacy leakage concerns to enable more practical and widespread adoption of FL. Unlike existing structured projections or sketched updates that primarily focus on compression, our approach introduces a fundamentally new multiplicative encoding paradigm through the *projected directional derivative* within the FL framework. Our algorithm follows the core structure of FedSGD but employs the *projected directional derivative* gˆi(xk) = u
⊤
k,igi(xk)uk,i
(defined in (1)) instead of the stochastic gradient gi(xk) to achieve communication efficiency and privacy preservation for each client i through an inherent geometric mechanism that exploits the nullspace properties of low-rank projections. We decompose the projected directional gradient into a scalar directional derivative u
⊤
k,igi(xk) computed locally by client i. Each client independently samples uk,i, then transmits only two scalars: the directional derivative u
⊤
k,igi(xk) and the random seed rk,i used to generate uk,i. On the server side, the received seed rk,i is used to reconstruct the identical d-dimensional vector uk,i, enabling the server to form the gradient estimator gˆi(xk) without ever transmitting the full vector.

The privacy protection emerges from a novel theoretical insight: the rank deficiency of uk,iu
⊤
k,i in gˆi(xk) = u
⊤
k,igi(xk)uk,i = uk,iu
⊤
k,igi(xk) creates an underdetermined system that prevents unique gradient recovery, fundamentally different from differential privacy approaches. The noisy gradient estimator from a single projected directional derivative can degrade performance. To address this, we propose our main algorithm FedMPDD (Federated Learning via Multi-Projected Directional Derivatives), which introduces a principled multi-projection aggregation mechanism that averages m projected directional derivatives to estimate the gradient and form our FL algorithm, leading to the following key contributions:
- **Jointly Improved Communication Efficiency and Privacy:** FedMPDD significantly reduces uplink communication to m + 1 scalars per client per round (O(m) bits, where m ≪ d). The privacy against GIAs is inherent in the multi-projection encoding due to the rank-deficiency of matrix 1m Uk,iU
⊤
k,i, where Uk,i is the aggregate matrix of {u
(j)
k,i}
m j=1, with m serving as a tunable parameter for the privacy-communication-accuracy trade-off. Unlike additive noise methods, FedMPDD offers a uniform privacy protection regardless of the magnitude of the clients' gradients, eliminating the fluctuating nature of LDP (Remark 3).

- **Enhanced Convergence Rate**: Unlike single-projection approach, FedMPDD achieves a convergence rate of O(1/
√K) (Theorem 2), comparable to standard baselines, through our novel multi-projection averaging mechanism that mitigates the dimensional dependence while preserving the inherent privacy guarantees.

Extensive comparative theoretical analysis and empirical evaluations demonstrate FedMPDD's superior balance of communication efficiency, privacy preservation, and model performance.

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Related Work. The *projected directional derivative* has been explored in optimization, including balancing computational cost and memory in deep learning gradient calculations (Fournier et al., 2023; Ren et al., 2022; Silver et al., 2021; Baydin et al., 2022) and in zeroth-order optimization (Nesterov & Spokoiny, 2017). To our knowledge, this is the first work to introduce the projected directional derivative in FL, specifically through a novel multi-projection decomposition that addresses the dimension-dependent convergence limitations of single-projection methods while achieving joint communication efficiency and privacy preservation. Regarding joint privacy preservation and communication management, some communication management techniques offer privacy as a side benefit (Lang et al., 2023; Agarwal et al., 2018). Amiri et al. (Amiri et al., 2021) combined differential privacy (DP) with gradient compression by adding Gaussian noise before quantization, satisfying DP. Lyu et al. (Lyu, 2021) proposed a 1-bit compressor integrating DP into quantization but relies on a fixed quantizer, limiting its adaptability to varying communication budgets. These approaches typically assume a trusted server, unlike LDP, which provides client-side privacy guarantees against potentially curious servers (Chaudhuri et al., 2022). Our method is fundamentally distinct from structured and sketched updates, which reduce communication by projecting data onto a *fixed*, low-dimensional subspace. Structured updates, such as low-rank adaptation, use a pre-defined subspace for parameters (Hu et al., 2022; Yi et al., 2023; Qi et al., 2024; Zhang et al., 2018; Ullrich et al., 2017; Bertsimas et al., 2023; Cho et al., 2024).

Similarly, sketched updates compress gradients using a shared random matrix, fixed at initialization, via techniques like projection matrices (Park & Choi, 2023; Azam et al., 2021; Guo et al., 2024) or Count-Sketch (Ivkin et al., 2019; Rothchild et al., 2020; Haddadpour et al., 2020; Jiang et al., 2018). These approaches rely on a static projection for all clients and rounds, with some variants incurring significant computational overhead to minimize reconstruction error in the fixed subspace (Lin et al., 2022). In contrast, our approach uses a *dynamic* projection strategy based on multi-projected directional derivatives. We compute gˆi(xk) = (u
⊤
k,igi(xk))uk,i, where the projection directions uk,i are randomly and independently sampled for each client i at every round k. By averaging multiple projections (m ≪ d), we overcome the rank-deficiency of a single direction. This mechanism simultaneously achieves significant communication reduction (to O(m) bits per client) and provides inherent privacy through the nullspace effect, representing a fundamental departure from methods reliant on fixed subspaces or post-hoc privacy solutions.

## 2 Fl Via Multi–Projected Directional Derivatives

The *projected directional derivative* is formally defined as follows.

Definition 1 (*projected directional derivative*). Let f : R
d → R be a differentiable function. The projected directional derivative is defined as ∇cf(x) := u
⊤∇f(x)u, where ∇f(x) is the gradient of f and u ∈ R
dis a random perturbation vector with entries uithat are independently and identically distributed (i.i.d.) with zero mean and unit variance. □
The projected directional derivative satisfies ∇cf(x)
⊤∇f(x) ≥ 0 and E[∇cf(x)] = ∇f(x) (unbiased estimator for gradient) and thus makes xk+1 = xk − η ∇cf(x) to behave as a iterative successive descent (i.e., E[f(xk+1)|xk] ≈ f(xk) − η∥∇f(x)∥
2 < f(xk) for small η > 0). This stands in contrast to structured/sketched updates, whose gradient estimator ∇ff(x) is often biased (E[∇ff(x)] ̸=
∇f(x)) and can violate the descent condition, thus lacking a general guarantee of progress. In the context of FedSGD we consider an implementation that instead of the stochastic gradient g(xk), we employ the *projected directional derivative* gˆ(xk) defined as:

$$\hat{\mathbf{g}}(\mathbf{x}_{k})=\frac{1}{\beta N}\big{(}\sum\nolimits_{i\in A_{k}\underbrace{\mathbf{u}_{k,i}^{\top}\mathbf{g}_{i}(\mathbf{x}_{k})}_{s_{i}^{k}\text{-chart upload}}}\underbrace{\mathbf{u}_{k,i}}_{\text{Serre-side projection}}\big{)}=\frac{1}{\beta N}\sum\nolimits_{i\in A_{k}}s_{i}^{k}\,\mathbf{u}_{k,i}.\tag{2}$$

Considering the decomposition shown in (2), propose Federated Learning via Projected Directional Derivative (FedPDD), see Algorithm 1 in Appendix G, that proceeds as follows: each iteration begins with the server broadcasting the current model xk ∈ R
dto a sampled client set Ak (line 4). Upon receipt, each client generates a local random vector uk,i using its private seed rk,i (line 6). The client encodes its local stochastic gradient into a scalar sk,i using the directional derivative along uk,i, and uploads this scalar together with the seed rk,i, which is essential for enabling convergence 162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 17: **Decode:** ∆sum ← ∆sum +
s k i
[j]
mu
(j)
k,i 18: **end for** 19: **end for**
20: **Aggregate:** gˆ(xk) = 1 βN
∆sum 21: **Model update:** xk+1 = xk − η gˆ(xk)
22: **end for**
23: **Output:** xK
1: **Input:** x0 ∈R
d, learning rate η, rounds K, \# random directions m, client fraction β ∈ (0, 1]
2: for k = 0, 1*, . . . , K* − 1 do 3: Server samples client set Ak with |Ak| = βN 4: Server broadcasts xk to all i ∈ Ak 5: **for each** client i ∈ Ak **in parallel do** 6: Compute local stochastic gradient gi(xk)
7: for j = 1*, . . . , m* do ▷ loop over projected directions 8: Client generates i.i.d. Rademacher vector u
(j)
k,i *∈ {−*1, +1}
dusing seed rk,i and index j 9: **Encode:** s k i[j] ←u
(j)
k,i⊤gi(xk)
10: **end for** 11: Upload s k i ∈ Rm and rk,i ∈ R to the server 12: **end for**
13: ∆sum ← 0d ▷ reset the estimator 14: for each client i ∈ Ak do ▷ on the server side 15: for j = 1*, . . . , m* do ▷ re-generate the same projected directions 16: Server generates i.i.d. Rademacher vector u
(j)
k,i *∈ {−*1, +1}
dusing seed rk,i and index j

(line 9). The server aggregates and decodes these scalars to update the global model xk+1 (lines
12-17). The design of FedPDD incorporates two key strategic choices. First, it employs a scalar seed
rk,i to generate the identical d-dimensional vector uk,i on the server side, thereby eliminating the need to transmit uk,i directly while still ensuring convergence. Second, although any zero-mean, unitvariance distribution for uk,i guarantees an unbiased projected directional derivative, our strategic choice of the Rademacher distribution for generating uk,i, as explained in Appendix Lemma 3, yields
lower variance compared to the standard normal distribution.
The preliminary FedPDD algorithm offers two appealing properties. First, it provides significant
communication reduction, as clients only transmit two scalars instead of a full gradient vector. Second,
it ensures intrinsic privacy preservation due to the rank-deficient nature of the single-vector projection
$\hat{\mathbf{g}}_{i}(\mathbf{x}_{k})=(\mathbf{u}_{k,i}^{\top}\mathbf{g}_{i}(\mathbf{x}_{k}))\mathbf{u}_{k,i}=\underbrace{(\mathbf{u}_{k,i}\mathbf{u}_{k,i}^{\top})}_{\text{Known,Rank-1Matrix}}\mathbf{g}_{i}(\mathbf{x}_{k}),$
gi(xk), (3)
which prevents unique gradient reconstruction. However, as our detailed analysis in Appendix B demonstrates, these benefits are offset by poor convergence performance. The high variance introduced by the single, rank-1 projection leads to a convergence rate of O(d/√K), which scales poorly with the model dimension d. Despite the projected directional derivative being an unbiased estimator, the rank-one map uk,iu
⊤
k,i leads to a magnitude scaling of 
√d compared to the gradient:

$\eqref{eq:walpha}$
$$\mathbb{E}_{\mathbf{u}}\left[\left|\left|\hat{\mathbf{g}}_{i}(\mathbf{x}_{k})\right|\right|\right]\leq\sqrt{\mathbb{E}_{\mathbf{u}}\left[\left|\left|\hat{\mathbf{g}}_{i}(\mathbf{x}_{k})\right|\right|^{2}\right]}=\sqrt{d\|\mathbf{g}_{i}(\mathbf{x}_{k})\|^{2}}=\sqrt{d}\left|\left|\mathbf{g}_{i}(\mathbf{x}_{k})\right|\right|.$$
This uncontrollable 
√d scaling factor leads to higher variance and potential overshooting in FedPDD
algorithm. The larger variance of the gradient estimator necessitates a smaller step size (e.g.,
η = O(1/(d
√K))), significantly slowing convergence (see Remark 3). This critical limitation negates the per-round communication savings over the full training process. Therefore, to address this shortcoming, we introduce FedMPDD (Federated learning with Multi-Projected Directional Derivatives), a generalized version of FedPDD algorithm, as presented in Algorithm 2, which is based on a multi-projection approach. Multi-projected directional derivatives. To address the limitation described above, we extend the estimator by sampling multiple directions. Specifically, at iteration k, each selected client i draws m i.i.d. Rademacher vectors {u
(j)
k,i}
m j=1 *⊂ {−*1, +1}
d. To better understand the mechanism, we now stack the sampled vectors as columns to form the matrix Uk,i ∈ R
d×m. Using this construction, the generalized estimator is defined as:
gˆi(xk):= 
1 m Xm j=1

$\mathbf{u}_{k,i}^{(m)}\Big]\in\mathbb{R}^{d\times m}$
  **or is defined as:**  $${\bf u}_{k,i}^{(j)\top}{\bf g}_{i}({\bf x}_{k}){\bf u}_{k,i}^{(j)}=\frac{1}{m}\,U_{k,i}\big{(}U_{k,i}^{\top}{\bf g}_{i}({\bf x}_{k})\big{)},\quad U_{k,i}=\left[{\bf u}_{k,i}^{(1)}\quad{\bf u}_{k,i}^{(2)}\quad\cdots\right]$$
Algorithm 2 FedMPDD: Federated Learning via Multi-Projected Directional Derivatives As E[Uk,iU
⊤
k,i] = mId, the estimator remains unbiased: E-gˆi(xk)=
1 m E[Uk,iU
⊤
k,i] gi(xk) =
gi(xk).

By constructing, the mapping 1m Uk,iU
⊤
k,i satisfies the high-probability operator-norm Johnson–Lindenstrauss (JL) Lemma (Matoušek, 2008) (Lemma 6 in the Appendix). According to the JL Lemma, if the number of sampled random directions satisfies m = O
ln(d/δ)
ε2
, then with probability at least 1 − δ, the following bound holds:

$$\|\frac{1}{m}\,U_{k,i}\big{(}U_{k,i}^{\top}{\bf g}_{i}({\bf x}_{k})\big{)}\|\ \leq\ (1+\varepsilon)\,\|{\bf g}_{i}({\bf x}_{k})\|\,.\tag{1}$$

This result implies that the mapping operator 1m Uk,iU
⊤
k,i approximately preserves the norm of the client's gradient with high probability, provided a sufficient number of sampled directions. Moreover, as m → ∞, the mapping approaches the identity operator 1m Uk,iU
⊤
k,i → Id in expectation, due to the unbiasedness of the projection. Motivated by this probabilistic guarantee, which grows only logarithmically with the ambient dimension d, we design FedMPDD algorithm, presented in Algorithm 2, as a generalization of FedPDD algorithm. The following result provides a convergence guarantee for FedMPDD.

Theorem 2 (Convergence Bound of FedMPDD Algorithm). *Let the step size* η =1 L
√K
, and suppose that Assumption 1 holds. Let number of random vectors be m = O ln(d/δ)
ε 2. Then, *FedMPDD*
algorithm converges to a stationary point of problem (1) *at a rate of* O(1/
√K)*, satisfying the* following upper bound with probability at least 1 − δ,

$$\frac{1}{K}\sum_{k=0}^{K-1}\mathbb{E}\left[\|\nabla f(\mathbf{x}_{k})\|^{2}\right]\leq\underbrace{O\bigg{(}\frac{L(f(\mathbf{x}_{0})-f^{*})}{K^{0.5}}\bigg{)}}_{\text{due to calibration}}+\underbrace{O\bigg{(}\frac{\sigma^{2}(1/\beta-1)}{K^{1.5}}\bigg{)}}_{\text{due to short coupling}}+\underbrace{O\bigg{(}\frac{\epsilon G^{2}}{K^{0.5}}\bigg{)}}_{\text{due to Rabi-spotting determined duration}}$$
$$(4)$$
* [16] M. C.  
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 where 0 < ϵ < 1 is the distortion parameter, β ∈ (0, 1] *denotes the client participation fraction, and* f
⋆ *denotes the global minimum of* f.

Remark 1 (Computational Cost of FedMPDD). The client-side encoding in FedMPDD has a computational cost of O(dm) (see lines 7–10 of the FedMPDD algorithm; as reported in Table A.10 for one representative experiment, this computational time is negligible and does not constitute a bottleneck in our experiments). While this may initially seem costly, it is often offset in practice, since in many federated learning settings, client models are deep neural networks and computing the full stochastic gradient (line 6) is already expensive. Recent work (Baydin et al., 2022; Ren et al., 2022; Silver et al., 2021) has shown that computing the inner product u
⊤giis significantly more efficient than computing the full gradient gi, because the operation can be implemented as a Jacobian-vector product (JVP), which leverages efficient vector-matrix multiplication in deep networks. Specifically, projected-forward methods reduce the time complexity of gradient computation from O(h 2pT2) (for full forward-mode autodiff) to O(h 2T + hpT), where h, p, and T denote the hidden dimension, the number of parameters per layer, and the total number of layers, respectively. Motivated by these insights, FedMPDD can avoid computing gi explicitly (line 6) by fixing a single mini-batch B
k iand reusing it across all random directions. The encoding step (line 9) is then performed via the projectedforward approach using JVPs. We can show that when m < hpT
h+p
, this strategy reduces overall client-side computation, making FedMPDD particularly suitable for resource-constrained devices.

We empirically evaluate this strategy in our follow-up study (see Section F). For further details on the computational and memory complexity of the projected-forward approach, see Table. F.1. □.

Communication Reduction and Efficiency in **FedMPDD**: FedMPDD presented in Algorithm 2 significantly reduces per-round uplink communication by enabling clients to transmit only an mdimensional vector s k i ∈ R
m together with a scalar seed number rk,i ∈ R, instead of the full d-dimensional gradient (m ≪ d). This is achieved by encoding the client's d-dimensional gradient through its projection onto a set of m random scalars (line 9). Moreover, the total communication cost over the full training horizon is reduced to O(1/
√K × βN × m), where β is the client participation ratio and N is the number of clients. Since m grows only logarithmically with the problem dimension d, the communication savings become even more substantial for large-scale models. Intrinsic Privacy Preservation: The privacy guarantees of FedMPDD are demonstrated under a standard honest-but-curious threat model, which provides a formal basis for our analysis.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 Definition 2 (*Threat Model*). An **honest-but-curious** adversary (e.g., the server) correctly follows the protocol but attempts to infer private client data by analyzing all accessible information. This includes communication messages, model architecture, and global hyperparameters. Against this adversary, FedMPDD's privacy stems from its rank-deficient projection (m ≪ d), which creates a quantifiable uncertainty for any party observing the transmitted data. We formalize this protection below.

Lemma 1 (Gradient Reconstruction Error). For the reconstructed gradient estimator gˆi(xk) =
1 m Pm j=1 u
(j)
k,i (u
(j)
k,i )
⊤gi(xk), the expected relative squared error between the reconstructed and true stochastic gradients is given by:

$$\mathbb{E}_{U}\left[\|\hat{\mathbf{g}}_{i}(\mathbf{x}_{k})-\mathbf{g}_{i}(\mathbf{x}_{k})\|^{2}\right]/\|\mathbf{g}_{i}(\mathbf{x}_{k})\|^{2}={\frac{d-1}{m}}.$$
m. (6)
This inherent gradient ambiguity provides a formal defense against GIAs by establishing a lower bound on an adversary's ability to reconstruct the original private input data. Lemma 2 (Lower Bound on Private Data Reconstruction Error). Suppose an adversary attempts to reconstruct a private input vector v by minimizing the loss between the observed projected gradient and a dummy gradient, L(ˆv) := ∥
1 m Uk,iU
⊤
k,igi(v, c; xk)−gi(ˆ*v, c*; xk)∥. The expected reconstruction error for the attack-optimal output vˆ
∗is lower bounded by:

$$\mathbb{E}\big[\|v-{\hat{v}}^{*}\|^{2}\big]\geq{\frac{d-1}{m\cdot L_{v}(\mathbf{x})^{2}}}\|\mathbf{g}_{i}(v,c;\mathbf{x}_{k})\|^{2},$$
2, (7)
where Lv(x) is the Lipschitz constant of the gradient with respect to v.

Together, these lemmas establish a direct link between our projection mechanism and a concrete privacy guarantee. The gradient reconstruction error of d−1 m translates into a formal lower bound on data recovery, creating a privacy barrier that scales with model dimension d. Our approach offers fundamental advantages over additive-noise methods like Local Differential Privacy (LDP). In LDP, the privacy level is inconsistent, as its relative reconstruction error is proportional to 1/∥gi(xk)∥
2as shown in Remark 5 in Appendix C. This creates a dilemma: large gradients are poorly protected, while small gradients can be overwhelmed by noise, harming model convergence. Achieving consistent privacy with LDP would require large, performance-degrading noise values. In contrast, FedMPDD provides a consistent relative reconstruction error of d−1 m , which is independent of the gradient's magnitude. This design simultaneously ensures: (i) consistent privacy without harming utility (by preserving the descent direction), and (ii) high communication efficiency. For a detailed derivation and discussion, please see Appendix C and E. Our theoretical guarantees are supported by empirical results. For example, Fig. 2 illustrates the FedMPDD Algorithm's data obfuscation on a client's private data for different values of m, as detailed in our numerical examples section. Similar to the observation in Remark 5, FedMPDD inherently provides a consistent relative reconstruction error of (d − 1)/m at each communication round. This consistent privacy benefit, independent of gradient size, can be observed in Fig. 1 in our numerical study (detailed in Appendix A).

Remark 2 (Multi-Round Privacy Composition). A key consideration is privacy erosion from an adversary observing a client over multiple rounds. Our formal analysis in Appendix D shows that even in a worst-case scenario (e.g., a static gradient), unique gradient recovery is impossible as long as the total number of observed projections is less than the model's dimension. Specifically, privacy is guaranteed if T × *m < d*, where T is the number of rounds. While the natural evolution of gradients during training provides stronger practical protection, this bound establishes a fundamental privacy guarantee for our method. □

$$(6)$$
$$(7)$$

Figure 1: The SSIM scores from the GIA (Yu et al., 2025) on the LeNet model, using the projected directional derivative estimator with m = 600 in FedMPDD, remain consistently low (below 0.04) over 100 training epochs. As showed by Lemma 1 and 2, this demonstrates that the privacy level remains stable and is independent of the training stage.

324

![6_image_0.png](6_image_0.png) 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Privacy-Communication Trade-off: The parameter m serves as a tunable knob for the privacycommunication-accuracy trade-off. A larger value of m directly translates to higher communication overhead per round, as the uplink message size scales with m. Conversely, as shown in Lemmas 1 and 2, a larger m improves the accuracy of reconstructed gradients, reflected in the reduced expected reconstruction error of d−1 m . Regarding privacy, a larger m implies less privacy, as more information about the original gradient is revealed during the projection process. This multi-faceted relationship highlights a fundamental trade-off: improving one aspect (e.g., accuracy) often comes at the expense of another (e.g., privacy or communication efficiency). This mirrors similar trade-offs observed in differential privacy, where stronger privacy guarantees typically lead to a reduction in model utility or require higher communication/computation. The worst-case bound *T < d/m* directly quantifies this trade-off: smaller m allows for more training rounds while maintaining privacy, but may require more communication rounds to achieve convergence. Our experimental results across two attack families, including the recent gradient inversion attack (Yu et al., 2025) and the well-known deep leakage from gradients method (Zhu et al., 2019), support this theoretical finding and demonstrate that the predicted privacy protection holds in practice.

## 3 Numerical Experiments

Experimental Setup. We conducted experiments on three standard machine learning and FL benchmark datasets using four model architectures with varying parameter sizes (For details see Appendix H.1). To comprehensively evaluate performance, we tested client participation rates of 10%, 50%, and 100% across different tasks, considering both IID and non-IID data distributions (where each client accesses only a subset of classes in multi-class classification). Hyperparameter tuning details for each model are in Appendix H.2. For gradient inversion attacks, we employed two algorithms: (i) the recent method proposed by Yu et al. (2025), and (ii) the well-known Deep Leakage from Gradients (DLG) algorithm (Zhu et al., 2019), which reconstructs original input data (e.g., images) from shared gradients in distributed learning. Performance vs. Joint Communication Efficiency and Privacy Leakage Mitigation. To highlight the communication cost reduction of FedMPDD, we compare it against a recent sketching-based method (Lin et al., 2022), a structured-based method (Yang et al., 2024), a top-k sparsification method (Alistarh et al., 2018), and the quantization-based method QSGD (Alistarh et al., 2017). For performance evaluation, FedSGD serves as the accuracy baseline. Our communication cost analysis includes total and per-round uplink overhead, as well as i) performance under a constrained communication budget and ii) the total communication cost to achieve target accuracy. To empirically validate FedMPDD's privacy enhancement against GIAs, we compare it to LDP with varying noise levels in image classification tasks.

To evaluate the quality of reconstructed images after the attack, we employ the Structural Similarity Index Measure (SSIM) (Lang et al., 2023), a widely used metric for assessing image similarity, where an SSIM value closer to 1 indicates a higher resemblance between the reconstructed image and the ground truth. Due to space limitations, we only show the subset of the results and the full set of table and training and accuracy curves will be presented in Appendix A. Note that in our experiments, we did not fine-tune the value of m to explicitly optimize for the minimal communication cost and

![7_image_0.png](7_image_0.png)

) 
) 

Table 1: Comparison of test accuracy (under a fixed communication budget), communication cost
(under a target accuracy), privacy leakage, and reconstruction quality using the attack of (Yu et al., 2025) on MNIST (IID) with LeNet.

| Method                     | Bytes Budget (GB)   | Test Acc (%)   | Target Acc (%)   | Used Bytes (GB)   | Defendability   | SSIM   |
|----------------------------|---------------------|----------------|------------------|-------------------|-----------------|--------|
| FedSGD                     | 0.09                | 11.45          | 60               | 1.439             | ✗               | 1.00   |
| FedSGD + Laplace (var=0.5) | 0.09                | 11.13          | 60               | 1.611             | ✓               | ≪ 0.03 |
| FedSGD + Laplace (var=1)   | 0.09                | 11.41          | 60               | 1.869             | ✓               | ≪ 0.03 |
| FedMPDD (m=400, 2% d)      | 0.09                | 77.37          | 60               | 0.052             | ✓               | ≪ 0.03 |
| FedMPDD (m=600, 3% d)      | 0.09                | 67.75          | 60               | 0.079             | ✓               | ≪ 0.03 |
| FedMPDD (m=800, 4% d)      | 0.09                | 58.49          | 60               | 0.093             | ✓               | ≪ 0.03 |
| QSGD (8-bit)               | 0.09                | 21.66          | 60               | 0.376             | ✗               | 0.98   |
| Top-k (k=400)              | 0.09                | 65.75          | 60               | 0.077             | ✗               | 0.89   |
| lp-proj                    | 0.09                | 73.01          | 60               | 0.069             | ✗               | 0.75   |

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 maximal privacy guarantees achievable by FedMPDD. Instead, we selected m = O
ln(d/δ)
ε 2for sufficiently small values of δ and ε. In Table A.9 of Appendix A, we present experiments over a wide range of m values to further illustrate the findings of Theorem 2. As m becomes too small, both the rate of convergence and the accuracy deteriorate. Moreover, as we show in our reported results, the chosen values of m grow slightly with the parameter dimension d (ranging from a simple logistic model to a deep CNN model with over 300,000 parameters) while maintaining convergence performance comparable to FedSGD, making the proposed algorithm well-suited for large-scale problems. This empirical observation further supports the theoretical guarantees in Theorem 2, where m is required to grow logarithmically with the dimension d to retain the O(1/
√K) convergence rate of FedSGD. Tables 1 and 2 demonstrate FedMPDD's effectiveness in jointly reducing communication cost and enhancing privacy. Our byte budget represents the *total uplink communication* permitted between active clients and the server across all training iterations, unlike per-round limits. We analyze the results from two complementary perspectives, beginning with those reported in Table 2. Fixed budget (0.9 GB): In Table 2 FedSGD and its Laplace-noised variants rapidly exceed the communication budget in the very first iteration, making them impractical under realistic constraints.

In contrast, FedMPDD stays well within the budget, achieving competitive accuracy thanks to its efficient projected directional derivative encoding. For example, with m = 600 (0.2% of d), FedMPDD
reaches 40.8% test accuracy, significantly higher than QSGD (12.9%) and other baselines such as lpproj (34.7%) (Lin et al., 2022), Top-k (38.1%) (Alistarh et al., 2018), and SA-FedLora (35.8%) (Yang et al., 2024). Importantly, although these baselines remain within the budget, they fail to provide consistent privacy guarantees, as their SSIM values (0.74 to 0.91) reveal substantial leakage under gradient inversion attacks. In contrast, FedMPDD achieves both stronger accuracy and substantially lower SSIM (0.14 to 0.22), highlighting its ability to simultaneously reduce communication and preserve privacy, rigorously established in Lemmas 1 and 2, arising from the (d − m)-dimensional nullspace of the multi-projected directional derivative." Fixed accuracy (60% target): To achieve the same target accuracy, in Table 2 FedSGD and its noisy variants consume over 470 GB, exceeding the budget by several orders of magnitude and leaking private information (SSIM > 0.8). Similarly, QSGD requires more than 117 GB while still failing on privacy. lp-proj, Top-k, and SA-FedLora are more communication-efficient, which is their primary goal, requiring only 1.8 to 2.3 GB, but they still exhibit weak privacy protection due to their high SSIM values. By contrast, FedMPDD with m = 600 requires only 1.3 GB, representing a **more than**
432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

| Method                     | Bytes Budget (GB)   | Test Acc (%)   | Target Acc (%)   | Used Bytes (GB)   | Defendability   | SSIM   |
|----------------------------|---------------------|----------------|------------------|-------------------|-----------------|--------|
| FedSGD                     | 0.90                | ⋆              | 60               | 471.96            | ✗               | 0.96   |
| FedSGD + Laplace (var=0.1) | 0.90                | ⋆              | 60               | 471.96            | ✗               | 0.84   |
| FedSGD + Laplace (var=10)  | 0.90                | ⋆              | 60               | not reached       | ✓               | 0.23   |
| FedMPDD (m=600, 0.2 % d)   | 0.90                | 40.84          | 60               | 1.32              | ✓               | 0.14   |
| FedMPDD (m=2000, 0.6 % d)  | 0.90                | 36.26          | 60               | 3.26              | ✓               | 0.22   |
| QSGD (8-bit)               | 0.90                | 12.97          | 60               | 117.98            | ✗               | 0.93   |
| lp-proj                    | 0.90                | 34.72          | 60               | 1.84              | ✗               | 0.74   |
| Top-k (k=600)              | 0.90                | 38.11          | 60               | 2.30              | ✗               | 0.91   |
| SA-FedLora                 | 0.90                | 35.84          | 60               | 2.10              | ✗               | 0.83   |

356× **reduction** compared to FedSGD, and with m = 2000, it requires just 3.3 GB, still a 144× reduction. Crucially, FedMPDD attains these communication savings while keeping SSIM < 0.22, ensuring strong and constant privacy level. Taken together, these results demonstrate that FedMPDD outperforms all baselines across both evaluation criteria: it matches or exceeds their communication efficiency while uniquely combining this with robust privacy protection. Competing methods (lp-proj, Top-k, SA-FedLora) achieve communication reduction but fail on privacy, whereas FedMPDD achieves both simultaneously.

Figure 2 illustrates FedMPDD's privacy-preserving strength under the GIA (Yu et al., 2025). The left plot shows SSIM scores over iterations, and the right panel visualizes reconstructed CIFAR-10 samples. Laplace noise with variance 0.1 (a typical LDP setting) fails to protect data, yielding high SSIM and clear reconstructions, while variance 10 provides privacy but severely degrades model accuracy. In contrast, FedMPDD with m = 2000 achieves a comparable privacy level to Laplace(10) without adding noise, since its protection arises from the (d − m)-dimensional nullspace of the projection, rigorously analyzed in Lemmas 1 and 2. At the same time, it reduces per-round communication by more than 150×, highlighting FedMPDD's dual benefit of strong privacy and efficiency.

While increasing m accelerates convergence, it also incurs higher communication cost and potentially greater privacy leakage (as expected from Lemma 1 and 2, increasing m decreases the inherent privacy protection at a rate of O(1/m), resulting in higher SSIM scores and more successful image reconstructions). However, as illustrated, for instance, in Fig. A.9 in the appendix, smaller values of m can actually achieve comparable or even faster convergence to the target accuracy, while simultaneously offering stronger privacy guarantees as a beneficial side effect. This makes FedMPDD particularly suitable for large-scale problems where both privacy and communication efficiency are critical. This behavior can be intuitively explained by the nullspace effect of the *projected* directional derivative mechanism, which effectively suppresses certain components of noise in the stochastic gradient, thereby stabilizing the optimization. For additional visualizations of another attack model (Zhu et al., 2019) across different architectures, as well as full training and accuracy curves under various methods, please refer to Appendix A.

## 4 Conclusion

We introduced FedMPDD, a novel FL framework addressing communication efficiency and privacy leakage through a gradient encoding and decoding mechanism based on multi-projected directional derivatives. Building upon the single-projection FedPDD, which offered initial communication and privacy benefits but suffered from dimension-dependent convergence, FedMPDD averaged multiple projections to achieve comparable convergence rates to baselines. Our theoretical analysis and empirical evaluations demonstrated FedMPDD's superior balance of communication cost, performance, and privacy, facilitated by its efficient gradient encoding and decoding. We achieved significant uplink communication reductions compared to baseline methods, including structured, sketched, quantized, and sparsified approaches, while simultaneously ensuring robust and uniform privacy against GIAs, unlike the fluctuating and often weak privacy guarantees of LDP. The tunable parameter m allowed flexible trade-offs. Notably, smaller m values sometimes yielded faster convergence with stronger privacy. For future work and further discussions see Appendix F. Table 2: Comparison of test accuracy (under a fixed communication budget), communication cost (under a target accuracy), privacy leakage, and reconstruction quality using the attack of (Yu et al., 2025) on CIFAR10 (IID) with CNN model from (McMahan et al., 2017). ⋆ indicates budget exceeded in the first iteration.

## References

Naman Agarwal, Ananda Theertha Suresh, Felix Xinnan X Yu, Sanjiv Kumar, and Brendan McMahan.

cpsgd: Communication-efficient and differentially-private distributed sgd. Advances in Neural Information Processing Systems, 31, 2018.

Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communicationefficient sgd via gradient quantization and encoding. Advances in neural information processing systems, 30, 2017.

Dan Alistarh, Torsten Hoefler, Mikael Johansson, Nikola Konstantinov, Sarit Khirirat, and Cédric Renggli. The convergence of sparsified gradient methods. *Advances in Neural Information* Processing Systems, 31, 2018.

Saba Amiri, Adam Belloum, Sander Klous, and Leon Gommans. Compressive differentially private federated learning through universal vector quantization. In *AAAI Workshop on Privacy-Preserving* Artificial Intelligence, pp. 2–9, 2021.

Sheikh Shams Azam, Seyyedali Hosseinalipour, Qiang Qiu, and Christopher Brinton. Recycling model updates in federated learning: Are gradient subspaces low-rank? In International Conference on Learning Representations, 2021.

Atılım Güne¸s Baydin, Barak A Pearlmutter, Don Syme, Frank Wood, and Philip Torr. Gradients without backpropagation. *arXiv preprint arXiv:2202.08587*, 2022.

Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signsgd:
Compressed optimisation for non-convex problems. In International Conference on Machine Learning, pp. 560–569. PMLR, 2018.

Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A new perspective on low-rank optimization. *Mathematical Programming*, 202(1):47–92, 2023.

Kamalika Chaudhuri, Chuan Guo, and Mike Rabbat. Privacy-aware compression for federated data analysis. In *Uncertainty in Artificial Intelligence*, pp. 296–306. PMLR, 2022.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Jiasi Chen and Xukan Ran. Deep learning with edge computing: A review. *Proceedings of the IEEE*,
107(8):1655–1674, 2019.

Mingzhe Chen, Nir Shlezinger, H Vincent Poor, Yonina C Eldar, and Shuguang Cui. Communicationefficient federated learning. *Proceedings of the National Academy of Sciences*, 118(17): e2024789118, 2021.

Yae Jee Cho, Luyang Liu, Zheng Xu, Aldi Fahrezi, and Gauri Joshi. Heterogeneous lora for federated fine-tuning of on-device foundation models. *arXiv preprint arXiv:2401.06432*, 2024.

Louis Fournier, Stéphane Rivaud, Eugene Belilovsky, Michael Eickenberg, and Edouard Oyallon.

Can forward gradient match backpropagation? In *International Conference on Machine Learning*, pp. 10249–10264. PMLR, 2023.

Jonas Geiping, Hartmut Bauermeister, Hannah Dröge, and Michael Moeller. Inverting gradients-how easy is it to break privacy in federated learning? Advances in neural information processing systems, 33:16937–16947, 2020.

Mingzhao Guo, Dongzhu Liu, Osvaldo Simeone, and Dingzhu Wen. Low-rank gradient compression with error feedback for mimo wireless federated learning. *arXiv preprint arXiv:2401.07496*, 2024.

Farzin Haddadpour, Belhal Karimi, Ping Li, and Xiaoyun Li. Fedsketch: Communication-efficient and private federated learning via sketching. *arXiv preprint arXiv:2008.04975*, 2020.

Yuze Han, Xiang Li, Shiyun Lin, and Zhihua Zhang. A random projection approach to personalized federated learning: Enhancing communication efficiency, robustness, and fairness. Journal of Machine Learning Research, 25(380):1–88, 2024.

Roger A Horn and Charles R Johnson. *Matrix analysis*. Cambridge university press, 2012. Samuel Horvóth, Chen-Yu Ho, Ludovit Horvath, Atal Narayan Sahu, Marco Canini, and Peter Richtárik. Natural compression for distributed deep learning. In Mathematical and Scientific Machine Learning, pp. 129–141. PMLR, 2022.

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. *ICLR*, 1(2):3, 2022.

Yangsibo Huang, Samyak Gupta, Zhao Song, Kai Li, and Sanjeev Arora. Evaluating gradient inversion attacks and defenses in federated learning. *Advances in neural information processing* systems, 34:7232–7241, 2021.

Nikita Ivkin, Daniel Rothchild, Enayat Ullah, Ion Stoica, Raman Arora, et al. Communicationefficient distributed sgd with sketching. *Advances in Neural Information Processing Systems*, 32, 2019.

Malhar S Jere, Tyler Farnan, and Farinaz Koushanfar. A taxonomy of attacks on federated learning.

IEEE Security & Privacy, 19(2):20–28, 2020.

Jiawei Jiang, Fangcheng Fu, Tong Yang, and Bin Cui. Sketchml: Accelerating distributed machine learning with data sketches. In Proceedings of the 2018 International Conference on Management of Data, pp. 1269–1284, 2018.

Peter Kairouz, H Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Kallista Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. Foundations and trends® *in machine learning*,
14(1–2):1–210, 2021.

Sai Praneeth Karimireddy, Quentin Rebjock, Sebastian Stich, and Martin Jaggi. Error feedback fixes signsgd and other gradient compression schemes. In International Conference on Machine Learning, pp. 3252–3261. PMLR, 2019.

Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In International conference on machine learning, pp. 5132–5143. PMLR, 2020.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Jakub Konecnˇ y, H Brendan McMahan, Felix X Yu, Peter Richtárik, Ananda Theertha Suresh, and `
Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492, 2016.

Kevin Kuo, Arian Raje, Kousik Rajesh, and Virginia Smith. Federated lora with sparse communication.

arXiv preprint arXiv:2406.05233, 2024.

Natalie Lang, Elad Sofer, Tomer Shaked, and Nir Shlezinger. Joint privacy enhancement and quantization in federated learning. *IEEE Transactions on Signal Processing*, 71:295–310, 2023.

Tian Li, Anit Kumar Sahu, Ameet Talwalkar, and Virginia Smith. Federated learning: Challenges, methods, and future directions. *IEEE signal processing magazine*, 37(3):50–60, 2020.

Zhaohua Li, Le Wang, Guangyao Chen, Zhiqiang Zhang, Muhammad Shafiq, and Zhaoquan Gu.

E2egi: End-to-end gradient inversion in federated learning. IEEE Journal of Biomedical and Health Informatics, 27(2):756–767, 2022.

Paul Pu Liang, Terrance Liu, Liu Ziyin, Nicholas B Allen, Randy P Auerbach, David Brent, Ruslan Salakhutdinov, and Louis-Philippe Morency. Think locally, act globally: Federated learning with local and global representations. *arXiv preprint arXiv:2001.01523*, 2020.

Shiyun Lin, Yuze Han, Xiang Li, and Zhihua Zhang. Personalized federated learning towards communication efficiency, robustness and fairness. *Advances in Neural Information Processing* Systems, 35:30471–30485, 2022.

Yujun Lin, Song Han, Huizi Mao, Yu Wang, and William J Dally. Deep gradient compression:
Reducing the communication bandwidth for distributed training. *arXiv preprint arXiv:1712.01887*, 2017.

Ken Liu, Shengyuan Hu, Steven Z Wu, and Virginia Smith. On privacy and personalization in cross-silo federated learning. *Advances in neural information processing systems*, 35:5925–5940, 2022.

Lingjuan Lyu. Dp-signsgd: When efficiency meets privacy and robustness. *arXiv preprint* arXiv:2105.04808, 2021.

Jiˇrí Matoušek. On variants of the johnson–lindenstrauss lemma. *Random Structures & Algorithms*,
33(2):142–156, 2008.

Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas.

Communication-efficient learning of deep networks from decentralized data. In Artificial intelligence and statistics, pp. 1273–1282. PMLR, 2017.

Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. Exploiting unintended feature leakage in collaborative learning. In *2019 IEEE symposium on security and privacy (SP)*, pp. 691–706. IEEE, 2019.

Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning:
Passive and active white-box inference attacks against centralized and federated learning. In 2019 IEEE symposium on security and privacy (SP), pp. 739–753. IEEE, 2019.

Yurii Nesterov and Vladimir Spokoiny. Random gradient-free minimization of convex functions.

Foundations of Computational Mathematics, 17(2):527–566, 2017.

Solmaz Niknam, Harpreet S Dhillon, and Jeffrey H Reed. Federated learning for wireless communications: Motivation, opportunities, and challenges. *IEEE Communications Magazine*, 58(6):46–51, 2020.

Sangjun Park and Wan Choi. Regulated subspace projection based local model update compression for communication-efficient federated learning. *IEEE Journal on Selected Areas in Communications*, 41(4):964–976, 2023.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Jiaxing Qi, Zhongzhi Luan, Shaohan Huang, Carol Fung, Hailong Yang, and Depei Qian. Fdlora:
personalized federated learning of large language model via dual lora tuning. arXiv preprint arXiv:2406.07925, 2024.

Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani.

Fedpaq: A communication-efficient federated learning method with periodic averaging and quantization. In *International conference on artificial intelligence and statistics*, pp. 2021–2031. PMLR,
2020.

Mengye Ren, Simon Kornblith, Renjie Liao, and Geoffrey Hinton. Scaling forward gradient with local losses. *arXiv preprint arXiv:2210.03310*, 2022.

Daniel Rothchild, Ashwinee Panda, Enayat Ullah, Nikita Ivkin, Ion Stoica, Vladimir Braverman, Joseph Gonzalez, and Raman Arora. Fetchsgd: Communication-efficient federated learning with sketching. In *International Conference on Machine Learning*, pp. 8253–8265. PMLR, 2020.

Mark Rudelson and Roman Vershynin. The littlewood–offord problem and invertibility of random matrices. *Advances in Mathematics*, 218(2):600–633, 2008.

Felix Sattler, Simon Wiedemann, Klaus-Robert Müller, and Wojciech Samek. Robust and communication-efficient federated learning from non-iid data. IEEE transactions on neural networks and learning systems, 31(9):3400–3413, 2019.

Mohamed Seif, Ravi Tandon, and Ming Li. Wireless federated learning with local differential privacy.

In *2020 IEEE International Symposium on Information Theory (ISIT)*, pp. 2604–2609. IEEE, 2020.

Osama Shahid, Seyedamin Pouriyeh, Reza M Parizi, Quan Z Sheng, Gautam Srivastava, and Liang Zhao. Communication efficiency in federated learning: Achievements and challenges. *arXiv* preprint arXiv:2107.10996, 2021.