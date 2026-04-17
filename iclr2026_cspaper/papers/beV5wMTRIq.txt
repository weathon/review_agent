000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Climate and weather prediction has traditionally relied on computationally demanding numerical simulations grounded in atmospheric physics, yet deep-learning approaches are emerging as transformative alternatives. Existing methods, however, are often purely data-driven and physics-agnostic, overlooking essential physical principles and struggling to generalize. To address these challenges, we present the Physics-Aware Tensor Field Neural PDE (PA-TFNP), a forecasting framework that embeds rotation-equivariant tensor-field neural operators directly on the sphere, couples them with a numerically rigorous gradient operator based on spherical transforms and physically consistent boundary treatment, and augments the learned dynamics with diffusion terms derived from the atmospheric primitive equations. These innovations enable our model to achieve superior performance through strict physical fidelity and efficient learning. The proposed PA-TFNP achieves state-of-the-art performance in global and regional weather prediction, outperforming ClimODE by 78.92% on global hourly data with a comparable number of parameters.

## 1 Introduction

Accurate climate and weather prediction is crucial for understanding environmental phenomena, preparing for extreme events, and enabling informed decisions. Traditional numerical simulations grounded in atmospheric physics (Rabier et al., 2000; Rawlins et al., 2007; Thompson, 1961) have achieved remarkable accuracy over medium timescales, leveraging systems of partial differential equations (PDEs) to model atmospheric dynamics and capture processes like advection, diffusion, and thermodynamics (Lions et al., 1992; Haltiner, 1971; Coiffier, 2011). However, solving these PDEs is computationally expensive, and extensive or proprietary datasets (Yu, 2010; Warner, 2010) pose significant scalability challenges, often making real-time or high-resolution global predictions infeasible. Moreover, traditional models struggle with rapidly changing climate patterns not well-represented in historical data (Neelin, 2010), highlighting the need for methods that are computationally efficient and can learn from observed data while maintaining physical consistency (Bader et al., 2008). In recent years, machine learning approaches have emerged as transformative alternatives to traditional simulations, challenging the mechanistic modeling paradigm with data-driven methods (Bi et al., 2023; Lam et al., 2023; Bodnar et al., 2024; Kochkov et al., 2024). These models learn complex spatiotemporal patterns directly from observations, bypassing the need to solve costly PDEs. They have shown promise in tasks ranging from high-resolution weather forecasting to global climate simulations (Bihlo, 2021; Verma et al., 2024; Pathak et al., 2022), capturing intricate dependencies for near-term predictions and localized events. Despite these successes, many remain physics-agnostic, relying solely on learned correlations rather than leveraging physical principles. Consequently, they struggle to enforce fundamental conservation laws, such as mass or energy conservation, and lack mechanisms to maintain incompressibility in fluid dynamics. This limits their generalization across diverse geophysical scenarios and leads to error accumulation over extended timeframes, undermining long-term forecasting reliability. To address these limitations, we propose the Physics-Aware Tensor Field Neural PDE (PA-TFNP), a novel framework designed to enhance climate and weather prediction by combining the strengths of deep learning with physical principles. In contrast to recent neural surrogates—such as ClimODE and ClimaX—that operate on flattened latitude–longitude grids or impose physics only through auxiliary losses, PA-TFNP learns directly on spherical tensor fields, preserving rotational symmetry throughout the network. It fuses a rotation-equivariant tensor-

# Physics-Aware Tensor Field Neural Pde For Climate And Weather Prediction

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 field operator with a mathematically consistent spherical-transform gradient and physically sound boundary conditions, giving the model intrinsic knowledge of physics laws rather than relying on post-hoc corrections. Furthermore, PA-TFNP embeds diffusion dynamics explicitly derived from the atmospheric primitive equations, enabling realistic long-term dynamics. This integration of geometry, numerics, and physics delivers substantial gains over existing benchmarks while demanding significantly fewer computational resources, proving that physical fidelity and efficiency can coexist in modern weather-forecasting systems. Our key contributions are as follows:
- We propose a Tensor Field Neural PDE framework (TFNP) powered by tensor-field neural networks that not only captures rotationally equivariant spatiotemporal patterns but also consistently outperforms the latest benchmark models across diverse climate and weatherprediction tasks.

- We devise a numerically rigorous spherical-transform-based gradient operator with physically consistent boundary conditions that stabilizes training and sharpens predictive precision, particularly near domain boundaries.

- We embed diffusion dynamics informed by the atmospheric Primitive Equations into our network, capturing key atmospheric processes and thereby improving both the accuracy and stability of weather forecasts.

Through these contributions, our method achieves significant improvements in both accuracy and robustness, effectively bridging the gap between physics-driven simulations and data-driven machine learning approaches.

## 2 Related Works

Numerical weather prediction. Conventional climate and weather forecasting primarily depends on physics-based numerical simulations (Shuman, 1989; Warner, 2010). In particular, short-term forecasts rely on established Numerical Weather Prediction (NWP) systems—such as the Unified Model (UM) (Bush et al., 2020) or other frameworks used in the U.S. (Powers et al., 2017) and Europe—that solve the so-called primitive equations (Wedi et al., 2015), a topic of extensive mathematical and computational research (Lions et al., 1992). Meanwhile, longer-term forecasts employ dedicated climate models, with Earth System Models (ESMs) (Mukhopadhyay et al., 2019) representing the cutting edge by coupling atmospheric, cryospheric, terrestrial, and oceanic processes. Although these modeling approaches have seen considerable success, they still face notable challenges, including sensitivity to initial conditions, structural inconsistencies across models (Bauer et al., 2015), significant computational burdens, and marked regional variability. Deep learning for forecasting. Recent advances in deep learning have yielded promising results for weather forecasting by bypassing some of the complexities of physics-based simulations. For instance, Rasp et al. (2020) applied pre-training with ResNet for medium-range weather prediction, and utilized large ensembles of deep models to capture sub-seasonal variations (Han et al., 2024). Other notable works include radar-based deep generative models for nowcasting (Ravuri et al., 2021) and graph neural network-based forecasting in GraphCast (Lam et al., 2023). In addition, FourCastNet (Kurth et al., 2023) and Pangu-Weather (Bi et al., 2023) represent state-of-the-art neural forecasting approaches that harness data-driven backbones, such as Vision Transformer, UNet, and autoencoders. Despite their empirical strengths, these methods tend to overlook key physical principles and seldom provide uncertainty estimates, limiting their interpretability and robustness.

Physics-Informed Machine Learning. Neural ODEs frame time derivatives as learnable neural networks (Fermanian et al., 2021), and have been extended to incorporate physics-based constraints
(Verma et al., 2024). Physics-Informed Neural Networks (PINNs) (Cai et al., 2021) embed mechanistic knowledge into DEs, and a broader line of research focuses on discovering interpretable differential equations (Brunton and Kutz, 2024). Extending such ideas to Neural PDEs often requires specialized spatial discretizations (Kochkov et al., 2024) or functional representations (Seol et al., 2024). Several studies have also used machine learning to improve fluid dynamics models (Choi et al., 2024). Notably, most of these works deal with smaller-scale fluid systems rather than the global scope demanded by climate or weather applications.

## 3 Methodology 108

109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Our model is fundamentally constructed using the Method of Lines (MOL) framework, as described in (Verma et al., 2024). This approach initially formulates the problem in terms of partial differential equations (PDEs) governing the evolution of multiple variables. To approximate the spatial derivatives in these PDEs, we employ a finite difference scheme, converting the PDEs into a system of ordinary differential equations (ODEs). Subsequently, we effectively approximate the temporal dynamics of the atmospheric variables by solving this system through a neural ODE framework (Chen et al., 2018). The detailed formulation is outlined below.

## 3.1 Preliminary

$${\frac{d\mathbf{Q}(t)}{d t}}=$$
  ∂q(x1,t) ∂t ... ∂q(xN ,t) ∂t   ≈   Fb(q(x1, t), {q(xn, t)}n∈N(1)), u(x1, t), {u(xn, t)}n∈N(1)) ... Fb(q(xN , t), {q(xn, t)}n∈N(N)), u(xN , t), {u(xn, t)}n∈N(N)) 
Consider a set of d atmospheric variables denoted by q(x, t) = {qi(x, t)}
d i=1 (e.g. temperature, geopotential height) that depend on the spatial location x ∈ [−90, 90]×[0, 360] (representing latitude and longitude on a sphere domain, such as Earth) and time t > 0. Observations of these variables are collected at a set of uniform grid points {xn}
N
n=1, where the spatial domain consists of H latitude points and W longitude points, resulting in a total of N = HW observations. In addition, we can consider the velocity field U(t) = {{ui(xn, t)}
d i=1}
N
n=1 that governs the advection of atmospheric variables. Given the velocity field, we model the temporal evolution of these variables using the following governing equations as in (Verma et al., 2024).

$$\begin{array}{l}{{\frac{\partial}{\partial t}q_{i}(\mathbf{x},t)=-\mathbf{u}_{i}(\mathbf{x},t)\cdot\nabla q_{i}(\mathbf{x},t)-q_{i}(\mathbf{x},t)\nabla\cdot\mathbf{u}_{i}(\mathbf{x},t),}}\\ {{\frac{\partial}{\partial t}\mathbf{u}_{i}(\mathbf{x},t)=f_{\eta}\left(\mathbf{Q}(t),\nabla\mathbf{Q}(t),\mathbf{U}(t),g(\{\mathbf{x}_{n}\}_{n=1}^{N},t)\right),}}\end{array}$$

where ∇ denotes the spatial gradient, Q(t) represents the set {q(xn, t)}
N
n=1, g is a spatio-temporal embedding function and fη is a trainable neural network with parameter η. The second equation implies that the velocity of each variable could be influenced by the other variables. To transform Equation (1) into a system of ODEs, we approximate the spatial derivatives using a finite difference, denoted as Fb (see Section 3.3 for details). The system for all variables Q(t) at the points of the grid is given below. Here, N (i) denotes the index set corresponding to the neighborhood of the grid point xi required for the finite-difference approximation. The system that governs U(t) can be formulated analogously. Consequently, the complete system consists of 3N d components when each atmospheric variable is considered a separate component. By integrating Equation (2) using the Runge-Kutta method to solve this system, we can estimate the values of the variables {qi}
d i=1 at all grid points {xn}
N
n=1.

$$\begin{bmatrix}\mathbf{Q}(t)\\ \mathbf{U}(t)\end{bmatrix}=\begin{bmatrix}\mathbf{Q}(t_{0})\\ \mathbf{U}(t_{0})\end{bmatrix}+\int_{t_{0}}^{t}\left(\frac{d\mathbf{Q}(s)}{ds}\right)ds\tag{1}$$
$$(1)$$
$\bigstar\in\mathbb{R}^{N d}$... 
$$\left(2\right)$$

Using the estimated Q(t) and real data, we train fη by minimizing the negative log-likelihood loss function, as defined in Sections 3.7 and 3.8 of (Verma et al., 2024).

## 3.2 Tensor Field Neural Pde (Tfnp)

In this paper, we parametrize the nonlinear operator fη in Equation (1) (illustrated in Figure 1) with a Tensor Field Network (TFN) f*T F N* (Thomas et al., 2018; Weiler et al., 2018; Kondor et al., 2018), combined with an attention mechanism, fatt (Vaswani et al., 2017), rather than employing a convolutional neural network (CNN). Although CNNs are often adopted for fη because they can approximate finite difference schemes on a uniform Euclidean grid (Brandstetter et al., 2022; Long et al., 2018), global climate data are typically sampled uniformly in latitude and longitude coordinates.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215

$$f_{TFN}(I[i,c_{\rm out}])=I\otimes I=\sum_{c_{1}=1}^{C_{\rm out}}\sum_{c_{2}=1}^{C_{\rm out}}W[c_{\rm out},c_{1},c_{2}](I[i,c_{1}]\cdot I[i,c_{2}]),\quad\forall i\in[N].$$

Here, Cin, Cout denote the input and output channel dimensions of f*T F N* . Additionally, we incorporate an attention-based network, fatt, following the architecture proposed in (Verma et al., 2024). Consequently, the final fη is constructed as the sum of the attention network fatt and the Tensor Field Network fT F N ,

![3_image_0.png](3_image_0.png)

This leads to geometric distortions near the polar regions, negatively impacting prediction accuracy.

Moreover, CNNs inherently fail to capture rotation-equivariant properties essential for processing spherical data. As in Figure 1, while rotations around the polar axis correspond to straightforward transformations in a periodic domain, rotations around the equatorial axis involve transformations coupled with reflections. Consequently, a CNN with fixed filters cannot approximate rotations of the latter type, as local features along the boundaries separating regions A, B, C, and D become distorted. We adopted a neural network based on tensor products instead of CNNs to mitigate this problem. This approach is inherently rotation equivariant, ensuring that transformations affect points near the poles and the equator consistently, without introducing distortion. The detailed formulation is as follows. The function fη takes as input Q(t) ∈ R
N×d, ∇Q(t) ∈ R
N×2d, U(t) ∈ R
N×2d, and g({xn}
N
n=1, t) ∈ R
N×e, where e denotes the embedding dimension introduced by g. If inputs in T
time steps are considered simultaneously, the dimension of input I is given by T ×N ×(5d+e). After reshaping I into a tensor of size N ×Cin, we can define the neural network fη : R
N×Cin → R
N×Cout as a tensor product-based function. This function is parameterized by a trainable weight tensor W[cout, c1.c2] for indices cout, c1, c2 ∈ [Cout], [Cin], [Cin], and is formulated as:
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

![4_image_0.png](4_image_0.png)

## 3.3 Physics-Aware Tensor Field Neural Pde (Pa-Tfnp)

In this section, we introduce PA-TFNP, an extension of TFNP that incorporates physical constraints into the model. We make three key modifications. First, we apply boundary conditions that reflect the domain's physical properties. Second, spatial derivatives are computed using spherical operators to capture Earth's geometry. Third, we augment the inputs to fη(·) in Equation equation 1 with physically relevant features: ground wind magnitude, lapse rate, and wind vorticity. We also modify the PDE solver to blend neural outputs with physics-based tendencies for improved interpretability and fidelity.

ClimODE exhibits unexpected errors near the boundary of the domain (see Figure 2), primarily due to the discretization of the sphere onto a longitude–latitude rectangular domain. This issue arises from the absence of proper boundary conditions in the original ClimODE formulation (Verma et al., 2024). The boundary conditions are implemented through an appropriate padding strategy and incorporated into the advection–diffusion equation during gradient computation. We propose two padding strategies, Neumann padding and average padding, both reflecting the physical characteristics of the domain. In both strategies, circular padding is applied along the longitudinal boundaries, effectively transforming the rectangular domain into a cylindrical one. For Neumann padding, replicate padding is used along the latitudinal boundaries, corresponding to homogeneous Neumann boundary conditions at the north and south poles (see Figure 2a). In the case of average padding, we extend the domain by padding with the average values of the boundary: µ1 =
1 64 P64 i=1 u1,i and µ2 =
1 64 P64 i=1 u2,i. This transforms the rectangular domain into a sphere-like domain (see Figure 2b). Figure 2c illustrates that TFN, equipped with this padding scheme, effectively captures the solution behavior near the poles. With a rotation-equivariant property, TFNP maintains consistent prediction accuracy across

## Boundary Conditions

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 all regions, including areas near the poles, resulting in robust predictions both at the boundaries and within the domain interior.

## Spatial Derivative Approximation

This section outlines the computation of the spatial derivatives in Equation (1). The method proposed in (Verma et al., 2024) estimates the derivatives by directly computing finite difference approximations along latitude and longitude, respectively. However, in a spherical domain, a given longitudinal difference corresponds to varying Euclidean distances depending on latitude. To account for this, we adopt a central finite difference scheme with a distance correction term:

$$\nabla q_{i}((\phi,\lambda),t)$$
$$\begin{array}{l}{{\forall q_{i}((\phi,\lambda),t)}}\\ {{\approx\left(\frac{q_{i}((\phi+h,\lambda),t)-q_{i}((\phi-h,\lambda),t)}{R h\pi/180},\frac{q_{i}((\phi,\lambda+w),t)-q_{i}((\phi,\lambda-w),t)}{R h\pi\cos\phi/180}\right),}}\end{array}$$
$$(3)$$
,(3)
where R represents the Earth's radius, and h and w denote the uniform grid spacing in latitude and longitude, respectively. Given the inherent periodicity in the longitudinal direction (λ), all grid points along this axis can be treated as interior points. Furthermore, we impose boundary conditions such as Neumann or periodic conditions on the latitude (ϕ), ensuring that all points within the domain are treated as interior points. Under these conditions, the central finite difference scheme can be consistently applied throughout the entire domain.

## Additional Physics-Derived Features

To augment the original TFNP framework, we introduce three physics-informed features: (i) the nearsurface wind magnitude |V10| =pu 210 + v 2 10, **(ii)** the low-tropospheric lapse rate ∆t = t − t2m, and
(iii) the relative vorticity ζ = ∂yv10 − ∂xu10, computed using spherical gradients. These quantities capture dynamic and thermodynamic processes essential to atmospheric motion.

## Modified Primitive Equation

To improve physical realism and long-term stability, we extend the neural advection formulation in Equation 1 by incorporating physics-inspired diffusion and momentum correction terms. First, scalar quantities such as temperature, humidity, and geopotential exhibit diffusive behavior in the real atmosphere, caused by unresolved subgrid turbulence and eddy transport Haltiner (1971); Lions et al. (1992); Warner (2010). To reflect this, we introduce a spatially varying diffusion term
with a learnable non-negative coefficient α(x) ∈ R
d×H×W . The scalar transport equation is modified
as follows:
$${\frac{\partial q_{i}(\mathbf{x},t)}{\partial t}}=-\mathbf{u}_{i}(\mathbf{x},t)\cdot\nabla q_{i}(\mathbf{x},t)-q_{i}(\mathbf{x},t)\nabla\cdot\mathbf{u}_{i}(\mathbf{x},t)+\alpha(\mathbf{x})\Delta q_{i}(\mathbf{x},t),$$
where the last term mimics anisotropic and spatially varying diffusion. Next, we augment the
neural tendency with physically meaningful momentum dynamics for the learned velocity field ui.
Specifically, we apply a time-dependent blending of neural predictions and physically grounded operators:
$${\frac{\partial\mathbf{u}_{i}(\mathbf{x},t)}{\partial t}}=(1-\beta_{t})\,f_{\eta}\big(\mathbf{Q}(t),\nabla\mathbf{Q}(t),\mathbf{U}(t),g\big(\{\mathbf{x}_{n}\}_{n=1}^{N},t\big)\big)+\beta_{t}\,f_{\mathrm{phys}}(\mathbf{x},t,\mathbf{u}_{i}),$$
$$f_{\mathrm{phys}}({\bf x},t,{\bf u}_{i})=-\nabla\Phi+\nu\Delta{\bf u}_{i}-\gamma{\bf u}_{i},$$
where Φ denotes the geopotential field (i.e., Φ = z), and ν, γ are learnable viscosity and linear drag coefficients, respectively. This hybrid formulation preserves the expressiveness of neural models while enforcing core physical constraints, improving both predictive performance and stability in long-range forecasts.

where the blend factor βt = 1 − exp(−t/τ0) gradually shifts preference from neural inference to physical consistency over time. The physical operator fphys imposes structure on the velocity evolution by incorporating key dynamical effects:

## 324

325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 We evaluate the performance of PA-TFNP by comparing it with the neural ODE, ClimaX (Nguyen et al., 2023) and ClimODE (Verma et al., 2024), a state-of-the-art data-driven global climate forecasting model. To ensure a fair comparison, we follow the experimental setup of (Verma et al., 2024), except for specific modifications detailed below. We utilize the ERA5 dataset from Weather-
Bench (Rasp et al., 2020), selecting d = 5 key atmospheric variables: ground temperature (t2m), atmospheric temperature (t), geopotential height (z), and ground wind components (u10, v10). All variables are normalized to the range [0, 1] using min-max scaling. Further details on dataset preprocessing and training settings remain consistent with those in (Verma et al., 2024) and Appendix B. All experiments were conducted using a single RTX 4090 GPU.

## 4.1 Global Weather Forecasting Across Varying Temporal And Spatial Resolutions

To evaluate the scalability and generalization of PA–TFNP across both spatial and temporal dimen-

![6_image_0.png](6_image_0.png) sions, we conduct experiments on global weather forecasting at two different settings: (a) long-term prediction over 5 days at a coarse resolution (5.625◦), and (b) short-term prediction over 6 to 42 hours at a finer resolution (11.25◦). Figure 3 summarizes the RMSE results for the five key atmospheric variables (z, t, t2m, u10, v10), comparing PA–TFNP with the state-of-the-art ClimODE baseline. Across both resolutions, PA–TFNP consistently outperforms ClimODE. In the long-term setting (first row in Figure 3), our model demonstrates particularly large improvements in forecasting geopotential height and atmospheric temperature. Similarly, in the short-term setting (second row in Figure 3), PA–TFNP shows improved accuracy across all lead times, with gains becoming more pronounced beyond 24 hours. This indicates that the model maintains robustness even as the forecasting horizon increases. These results confirm the effectiveness of PA–TFNP in learning global-scale spatiotemporal dynamics, while preserving accuracy across varying resolutions and forecast ranges.

## 4.2 Short-Term Regional Weather Forecasting

We evaluate short-term (up to 24 hours) regional weather forecasting over the Australia and the South American region. Table 1 presents the RMSE (mean ± standard deviation) of various models across five key atmospheric variables. Our proposed model, PA–TFNP, demonstrates strong predictive accuracy overall, particularly for the geopotential height (z) and temperature (t) variables, where it consistently outperforms all baselines across all lead times. Compared to the current state-ofthe-art model, ClimODE, PA–TFNP achieves lower RMSE, especially at longer horizons (18–24h), demonstrating improved temporal robustness.

## 4 Experiments

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

| Australia              | South America   |                                 |                  |                                 |                    |                  |                  |             |             |
|------------------------|-----------------|---------------------------------|------------------|---------------------------------|--------------------|------------------|------------------|-------------|-------------|
| Value Hour NODE ClimaX | ClimODE         | PA–TFNP NODE ClimaX             | ClimODE          | PA–TFNP                         |                    |                  |                  |             |             |
| 6                      | 251.4           | 190.2 103.8 ± 14.6              | 79.5 ± 19.9      | 225.6                           | 205.4 107.7 ± 20.2 | 87.5 ± 22.0      |                  |             |             |
| 12                     | 344.8           | 184.7 170.7 ± 21.0 118.8 ± 30.1 | 365.6            | 220.2 169.4 ± 29.6 128.2 ± 31.3 |                    |                  |                  |             |             |
| 18                     | 539.9           | 222.2 211.1 ± 31.6 161.6 ± 43.8 | 551.9            | 269.1 237.8 ± 32.2 174.1 ± 43.4 |                    |                  |                  |             |             |
| 24                     | 632.7           | 324.9 308.2 ± 30.6 205.8 ± 59.5 | 660.3            | 301.8 292.0 ± 38.9 221.3 ± 57.8 |                    |                  |                  |             |             |
| z                      | 6               | 1.37                            | 1.19             | 1.05 ± 0.12                     | 0.87 ± 0.14        | 1.58             | 1.38 0.97 ± 0.13 | 1.01 ± 0.16 |             |
| 12                     | 2.18            | 1.30                            | 1.20 ± 0.16      | 1.07 ± 0.18                     | 2.18               | 1.62             | 1.25 ± 0.18      | 1.18 ± 0.18 |             |
| 18                     | 2.68            | 1.39                            | 1.33 ± 0.21      | 1.19 ± 0.20                     | 2.74               | 1.79             | 1.43 ± 0.20      | 1.29 ± 0.18 |             |
| 24                     | 3.32            | 1.92                            | 1.63 ± 0.24      | 1.31 ± 0.23                     | 3.41               | 1.97             | 1.65 ± 0.26      | 1.44 ± 0.21 |             |
| t                      | 6               | 1.88                            | 1.57 0.80 ± 0.13 | 2.42 ± 0.70                     | 2.12               | 1.85 1.33 ± 0.26 | 1.73 ± 0.67      |             |             |
| 12                     | 2.02            | 1.57 1.10 ± 0.22                | 2.98 ± 1.50      | 2.42                            | 2.08 1.04 ± 0.17   | 2.37 ± 1.20      |                  |             |             |
| 18                     | 3.51            | 1.72 1.23 ± 0.24                | 2.37 ± 0.55      | 2.60                            | 2.15 0.98 ± 0.17   | 1.87 ± 0.84      |                  |             |             |
| 24                     | 2.46            | 2.15                            | 1.25 ± 0.25      | 1.16 ± 0.24                     | 2.56               | 2.23             | 1.17 ± 0.26      | 1.15 ± 0.27 |             |
| t2m                    | 6               | 1.91                            | 1.40 1.35 ± 0.17 | 1.43 ± 0.19                     | 1.94               | 1.27 1.25 ± 0.18 | 1.42 ± 0.27      |             |             |
| 12                     | 2.86            | 1.77                            | 1.78 ± 0.21      | 1.74 ± 0.22                     | 2.74               | 1.57 1.49 ± 0.23 | 1.56 ± 0.30      |             |             |
| u10                    | 18              | 3.44                            | 2.03             | 1.96 ± 0.25                     | 1.88 ± 0.26        | 3.24             | 1.83             | 1.81 ± 0.29 | 1.69 ± 0.29 |
| 24                     | 3.91            | 2.64                            | 2.33 ± 0.33      | 2.06 ± 0.28                     | 3.77               | 2.04             | 2.08 ± 0.35      | 1.86 ± 0.32 |             |
| 6                      | 2.38            | 1.47 1.44 ± 0.20                | 1.56 ± 0.19      | 2.29                            | 1.31 1.30 ± 0.21   | 1.68 ± 0.39      |                  |             |             |
| 12                     | 3.60            | 1.79                            | 1.87 ± 0.26      | 1.78 ± 0.25                     | 3.42               | 1.64             | 1.71 ± 0.28      | 1.93 ± 0.40 |             |
| v10                    | 18              | 4.31                            | 2.33             | 2.23 ± 0.23                     | 2.04 ± 0.26        | 4.16             | 1.90             | 2.07 ± 0.31 | 1.88 ± 0.37 |
| 24                     | 4.88            | 2.58                            | 2.53 ± 0.32      | 2.23 ± 0.30                     | 4.76               | 2.14             | 2.43 ± 0.34      | 2.06 ± 0.37 |             |

For wind components, PA–TFNP slightly outperforms ClimODE in most settings, particularly at longer lead times. Notably, for t2m, PA–TFNP underperforms at earlier lead times but catches up or surpasses baselines at 24h. This may indicate a trade-off between local variance sensitivity and longer-horizon stability.

## 4.3 Monthly Averaged Weather Forecasting

Next, we evaluate the predictive accuracy of ClimODE, CilmaX, TFNP, and PA-TFNP over a two-month lead time. All models predict the global two-month averaged future states based on an initial monthly average state. Table 2 provides a detailed comparison of RMSE values for various atmospheric variables, showing that PA-TFNP consistently outperforms other benchmarks, particularly in predicting geopotential height (z), atmospheric temperature (t) and ground temperature
(t2m). The lower RMSE values in the results indicate that PA-TFNP more accurately captures complex climate patterns, offering enhanced reliability for extended-range climate forecasting.

## 4.4 Ablation Studies

Assessing rotational equivariance: ClimODE vs TFNP. To further evaluate the spatial prediction capabilities of TFNP, we compare its performance with ClimODE in terms of absolute prediction error across five key atmospheric variables (see Figure 6 in Appendix A). The results demonstrate that TFNP consistently achieves lower error magnitudes than ClimODE, particularly in geophysically challenging regions such as the poles and the equator. These regions are often prone to distortions due to their rotational properties, where ClimODE exhibits noticeable artifacts. In contrast, TFNP maintains strong spatial consistency, owing to its rotation-equivariant architecture. These findings underscore the importance of incorporating geometric inductive biases, such as rotational equivariance, in improving model robustness and accuracy in global-scale geophysical forecasting.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

| Value   | Months   | ClimaX         | ClimODE         | TFNP (ours)    | PA-TFNP (ours)   |
|---------|----------|----------------|-----------------|----------------|------------------|
| z       | 1        | 580.73         | 692.10 ± 119.80 | 529.44 ± 95.77 | 502.01 ± 79.50   |
| 2       | 773.40   | 870.57 ± 72.58 | 527.07 ± 84.54  | 562.39 ± 70.13 |                  |
| t       | 1        | 2.89           | 2.81 ± 0.48     | 2.58 ± 0.56    | 2.48 ± 0.45      |
| 2       | 4.39     | 3.20 ± 1.02    | 2.42 ± 0.42     | 2.44 ± 0.21    |                  |
| t2m     | 1        | 2.97           | 4.33 ± 0.38     | 2.63 ± 0.52    | 2.53 ± 0.34      |
| 2       | 5.07     | 4.99 ± 0.48    | 2.95 ± 0.45     | 2.95 ± 0.30    |                  |
| u10     | 1        | 1.80           | 1.98 ± 0.19     | 1.86 ± 0.23    | 1.83 ± 0.23      |
| 2       | 1.92     | 2.09 ± 0.11    | 2.40 ± 0.22     | 2.32 ± 0.21    |                  |
| v10     | 1        | 1.50           | 1.66 ± 0.18     | 1.40 ± 0.10    | 1.39 ± 0.12      |
| 2       | 1.71     | 1.98 ± 0.11    | 1.95 ± 0.18     | 1.91 ± 0.21    |                  |

Benefits of Physics-Aware Modeling for Long-Term Stability: TFNP vs PA-TFNP. To evaluate

![8_image_0.png](8_image_0.png) the effectiveness of Physics-Aware modeling, we compared the performance of the PA-TFNP model, which incorporate physical operators and features against the TFNP model. Experimental results shows that PA-TFNP consistently outperforms the TFNP model at extended forecast horizons beyond 24 hours, across all scalar quantities. These results underscore the importance of embedding physical properties within predictive models to achieve stable and reliable long-term forecasting, as clearly illustrated in Figure 4.

## 5 Conclusion And Limitations

In this work, we have presented the Physics-Aware TFNP, a novel framework that combines deep learning with fundamental physical principles to tackle climate and weather prediction tasks more accurately and robustly. By integrating gradient computation and boundary treatment methods rooted in numerical techniques and by incorporating physically consistent diffusion terms and divergence-free conditions, our approach addresses the shortcomings of both purely data-driven and physics-agnostic models. TFNP not only demonstrates state-of-the-art forecasting performance but also maintains physical fidelity, offering enhanced interpretability and reliability. We anticipate that the mathematical principles introduced here will generalize across a broad range of scientific computing domains, thereby accelerating progress in both global and regional weather prediction. As expected, the rotation-equivariant feature of the proposed PA-TFNP plays an important role in the global forecasting model. However, this characteristic appears to offer limited benefits for regional forecasting. This limitation warrants further investigation in future work. We have added diffusion terms to the model equations for all predictive variables. However, the modification of the model equation should be tailored to each variable, as their physical interpretations differ significantly. For instance, the temperature variable and ground wind variables represent fundamentally different physical phenomena and therefore should be modeled using distinct equations.

## References

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Florence Rabier, Heikki Järvinen, E Klinker, J-F Mahfouf, and A Simmons. The ecmwf operational implementation of four-dimensional variational assimilation. i: Experimental results with simplified physics. *Quarterly Journal of the Royal Meteorological Society*, 126(564):1143–1170, 2000.

F Rawlins, SP Ballard, KJ Bovis, AM Clayton, D Li, GW Inverarity, AC Lorenc, and TJ Payne. The met office global four-dimensional variational data assimilation scheme. Quarterly Journal of the Royal Meteorological Society: A journal of the atmospheric sciences, applied meteorology and physical oceanography, 133(623):347–362, 2007.

Philip Duncan Thompson. Numerical weather analysis and prediction. *(No Title)*, 1961. Jacques-Louis Lions, Roger Temam, and Shouhong Wang. New formulations of the primitive equations of atmosphere and applications. *Nonlinearity*, 5(2):237, 1992.

G.J. Haltiner. *Numerical Weather Prediction*. Wiley, 1971. ISBN 9780471345800. URL https:
//books.google.co.kr/books?id=RTZRAAAAMAAJ.

Jean Coiffier. *Fundamentals of numerical weather prediction*. Cambridge University Press, 2011. Tsann-Wang Yu. Advances and challenges in numerical weather and climate prediction. In AIP
Conference Proceedings, volume 1280, pages 142–158. American Institute of Physics, 2010.

Thomas Tomkins Warner. *Numerical weather and climate prediction*. cambridge university press, 2010.

J David Neelin. *Climate change and climate modeling*. Cambridge University Press, 2010. David Bader, Curt Covey, William Gutowski, Isaac Held, Kenneth Kunkel, Ronald Miller, Robin Tokmakian, and Minghua Zhang. Climate models: an assessment of strengths and limitations. 2008.

Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate mediumrange global weather forecasting with 3d neural networks. *Nature*, 619(7970):533–538, 2023.

Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful medium-range global weather forecasting. *Science*, 382(6677):1416–1421, 2023.

Cristian Bodnar, Wessel P Bruinsma, Ana Lucic, Megan Stanley, Johannes Brandstetter, Patrick Garvan, Maik Riechert, Jonathan Weyn, Haiyu Dong, Anna Vaughan, et al. Aurora: A foundation model of the atmosphere. *arXiv preprint arXiv:2405.13063*, 2024.

Dmitrii Kochkov, Janni Yuval, Ian Langmore, Peter Norgaard, Jamie Smith, Griffin Mooers, Milan Klöwer, James Lottes, Stephan Rasp, Peter Düben, et al. Neural general circulation models for weather and climate. *Nature*, 632(8027):1060–1066, 2024.

Alex Bihlo. A generative adversarial network approach to (ensemble) weather prediction. *Neural* Networks, 139:1–16, 2021.

Yogesh Verma, Markus Heinonen, and Vikas Garg. ClimODE: Climate forecasting with physicsinformed neural ODEs. In *The Twelfth International Conference on Learning Representations*, 2024. URL https://openreview.net/forum?id=xuY33XhEGR.

Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay, Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, et al. Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators. arXiv preprint arXiv:2202.11214, 2022.

Frederick G. Shuman. History of numerical weather prediction at the national meteorological center. *Weather and Forecasting*, 4(3):286 - 296, 1989. doi: 10.1175/1520-0434(1989)004<0286: HONWPA>2.0.CO;2. URL https://journals.ametsoc.org/view/journals/ wefo/4/3/1520-0434_1989_004_0286_honwpa_2_0_co_2.xml.

Mike Bush, Tom Allen, Caroline Bain, Ian Boutle, John Edwards, Anke Finnenkoetter, Charmaine Franklin, Kirsty Hanley, Humphrey Lean, Adrian Lock, et al. The first met office unified model– jules regional atmosphere and land configuration, ral1. *Geoscientific Model Development*, 13(4):
1999–2029, 2020.

Jordan G Powers, Joseph B Klemp, William C Skamarock, Christopher A Davis, Jimy Dudhia, David O Gill, Janice L Coen, David J Gochis, Ravan Ahmadov, Steven E Peckham, et al. The weather research and forecasting model: Overview, system efforts, and future directions. Bulletin of the American Meteorological Society, 98(8):1717–1737, 2017.

NP Wedi, P Bauer, W Denoninck, M Diamantakis, M Hamrud, C Kuhnlein, S Malardel, K Mogensen, G Mozdzynski, and PK Smolarkiewicz. The modelling infrastructure of the integrated forecasting system: Recent advances and future challenges. 2015.

P Mukhopadhyay, VS Prasad, R Phani Murali Krishna, Medha Deshpande, Malay Ganai, Snehlata Tirkey, Sahadat Sarkar, Tanmoy Goswami, CJ Johny, Kumar Roy, et al. Performance of a very high-resolution global forecast system model (gfs t1534) at 12.5 km over the indian region during the 2016–2017 monsoon seasons. *Journal of Earth System Science*, 128:1–18, 2019.

Peter Bauer, Alan Thorpe, and Gilbert Brunet. The quiet revolution of numerical weather prediction.

Nature, 525(7567):47–55, 2015.

Stephan Rasp, Peter D Dueben, Sebastian Scher, Jonathan A Weyn, Soukayna Mouatadid, and Nils Thuerey. Weatherbench: a benchmark data set for data-driven weather forecasting. *Journal of* Advances in Modeling Earth Systems, 12(11):e2020MS002203, 2020.

Tao Han, Song Guo, Fenghua Ling, Kang Chen, Junchao Gong, Jingjia Luo, Junxia Gu, Kan Dai, Wanli Ouyang, and Lei Bai. Fengwu-ghr: Learning the kilometer-scale medium-range global weather forecasting. *arXiv preprint arXiv:2402.00059*, 2024.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Suman Ravuri, Karel Lenc, Matthew Willson, Dmitry Kangin, Remi Lam, Piotr Mirowski, Megan Fitzsimons, Maria Athanassiadou, Sheleem Kashem, Sam Madge, et al. Skilful precipitation nowcasting using deep generative models of radar. *Nature*, 597(7878):672–677, 2021.

Thorsten Kurth, Shashank Subramanian, Peter Harrington, Jaideep Pathak, Morteza Mardani, David Hall, Andrea Miele, Karthik Kashinath, and Anima Anandkumar. Fourcastnet: Accelerating global high-resolution weather forecasting using adaptive fourier neural operators. In *Proceedings of the* platform for advanced scientific computing conference, pages 1–11, 2023.

Adeline Fermanian, Pierre Marion, Jean-Philippe Vert, and Gérard Biau. Framing rnn as a kernel method: A neural ode approach. *Advances in Neural Information Processing Systems*, 34:3121– 3134, 2021.

Shengze Cai, Zhiping Mao, Zhicheng Wang, Minglang Yin, and George Em Karniadakis. Physicsinformed neural networks (pinns) for fluid mechanics: A review. *Acta Mechanica Sinica*, 37(12):
1727–1738, 2021.

Steven L Brunton and J Nathan Kutz. Promising directions of machine learning for partial differential equations. *Nature Computational Science*, 4(7):483–494, 2024.

Yunchang Seol, Suho Kim, Minwoo Jung, and Youngjoon Hong. A novel physics-aware graph network using high-order numerical methods in weather forecasting model. Knowledge-Based Systems, page 112158, 2024.

Junho Choi, Taehyun Yun, Namjung Kim, and Youngjoon Hong. Spectral operator learning for parametric pdes without data reliance. *Computer Methods in Applied Mechanics and Engineering*,
420:116678, 2024.

Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary differential equations. *Advances in neural information processing systems*, 31, 2018.

Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley.

Tensor field networks: Rotation- and translation-equivariant neural networks for 3d point clouds, 2018. URL https://arxiv.org/abs/1802.08219.

Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco Cohen. 3d steerable cnns:
Learning rotationally equivariant features in volumetric data, 2018. URL https://arxiv. org/abs/1807.02547.

Risi Kondor, Zhen Lin, and Shubhendu Trivedi. Clebsch-gordan nets: a fully fourier space spherical convolutional neural network, 2018. URL https://arxiv.org/abs/1806.09231.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc.,
2017. URL https://proceedings.neurips.cc/paper_files/paper/2017/ file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.

Johannes Brandstetter, Daniel E. Worrall, and Max Welling. Message passing neural PDE solvers. In International Conference on Learning Representations, 2022. URL https://openreview. net/forum?id=vSix3HPYKSU.

Zichao Long, Yiping Lu, Xianzhong Ma, and Bin Dong. Pde-net: Learning pdes from data. In International conference on machine learning, pages 3208–3216. PMLR, 2018.

Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, and Aditya Grover. Climax:
A foundation model for weather and climate. *arXiv preprint arXiv:2301.10343*, 2023.