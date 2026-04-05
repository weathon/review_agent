# PHYSICS-AWARE TENSOR FIELD NEURAL PDE FOR CLIMATE AND WEATHER PREDICTION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Climate and weather prediction has traditionally relied on computationally demanding numerical simulations grounded in atmospheric physics, yet deep-learning
approaches are emerging as transformative alternatives. Existing methods, however, are often purely data-driven and physics-agnostic, overlooking essential
physical principles and struggling to generalize. To address these challenges, we
present the Physics-Aware Tensor Field Neural PDE (PA-TFNP), a forecasting
framework that embeds rotation-equivariant tensor-field neural operators directly
on the sphere, couples them with a numerically rigorous gradient operator based on
spherical transforms and physically consistent boundary treatment, and augments
the learned dynamics with diffusion terms derived from the atmospheric primitive
equations. These innovations enable our model to achieve superior performance
through strict physical fidelity and efficient learning. The proposed PA-TFNP
achieves state-of-the-art performance in global and regional weather prediction,
outperforming ClimODE by 78.92% on global hourly data with a comparable
number of parameters.


1 INTRODUCTION


Accurate climate and weather prediction is crucial for understanding environmental phenomena,
preparing for extreme events, and enabling informed decisions. Traditional numerical simulations
grounded in atmospheric physics (Rabier et al., 2000; Rawlins et al., 2007; Thompson, 1961) have
achieved remarkable accuracy over medium timescales, leveraging systems of partial differential
equations (PDEs) to model atmospheric dynamics and capture processes like advection, diffusion, and
thermodynamics (Lions et al., 1992; Haltiner, 1971; Coiffier, 2011). However, solving these PDEs is
computationally expensive, and extensive or proprietary datasets (Yu, 2010; Warner, 2010) pose significant scalability challenges, often making real-time or high-resolution global predictions infeasible.
Moreover, traditional models struggle with rapidly changing climate patterns not well-represented in
historical data (Neelin, 2010), highlighting the need for methods that are computationally efficient
and can learn from observed data while maintaining physical consistency (Bader et al., 2008).


In recent years, machine learning approaches have emerged as transformative alternatives to traditional
simulations, challenging the mechanistic modeling paradigm with data-driven methods (Bi et al.,
2023; Lam et al., 2023; Bodnar et al., 2024; Kochkov et al., 2024). These models learn complex
spatiotemporal patterns directly from observations, bypassing the need to solve costly PDEs. They
have shown promise in tasks ranging from high-resolution weather forecasting to global climate
simulations (Bihlo, 2021; Verma et al., 2024; Pathak et al., 2022), capturing intricate dependencies for
near-term predictions and localized events. Despite these successes, many remain physics-agnostic,
relying solely on learned correlations rather than leveraging physical principles. Consequently, they
struggle to enforce fundamental conservation laws, such as mass or energy conservation, and lack
mechanisms to maintain incompressibility in fluid dynamics. This limits their generalization across
diverse geophysical scenarios and leads to error accumulation over extended timeframes, undermining
long-term forecasting reliability. To address these limitations, we propose the Physics-Aware Tensor
Field Neural PDE (PA-TFNP), a novel framework designed to enhance climate and weather prediction
by combining the strengths of deep learning with physical principles. In contrast to recent neural
surrogates—such as ClimODE and ClimaX—that operate on flattened latitude–longitude grids or
impose physics only through auxiliary losses, PA-TFNP learns directly on spherical tensor fields,
preserving rotational symmetry throughout the network. It fuses a rotation-equivariant tensor

1


field operator with a mathematically consistent spherical-transform gradient and physically sound
boundary conditions, giving the model intrinsic knowledge of physics laws rather than relying
on post-hoc corrections. Furthermore, PA-TFNP embeds diffusion dynamics explicitly derived
from the atmospheric primitive equations, enabling realistic long-term dynamics. This integration of
geometry, numerics, and physics delivers substantial gains over existing benchmarks while demanding
significantly fewer computational resources, proving that physical fidelity and efficiency can coexist
in modern weather-forecasting systems. Our key contributions are as follows:


    - We propose a Tensor Field Neural PDE framework (TFNP) powered by tensor-field neural
networks that not only captures rotationally equivariant spatiotemporal patterns but also
consistently outperforms the latest benchmark models across diverse climate and weatherprediction tasks.


    - We devise a numerically rigorous spherical-transform-based gradient operator with physically consistent boundary conditions that stabilizes training and sharpens predictive precision,
particularly near domain boundaries.


    - We embed diffusion dynamics informed by the atmospheric Primitive Equations into our
network, capturing key atmospheric processes and thereby improving both the accuracy and
stability of weather forecasts.


Through these contributions, our method achieves significant improvements in both accuracy and
robustness, effectively bridging the gap between physics-driven simulations and data-driven machine
learning approaches.


2 RELATED WORKS


**Numerical weather prediction** . Conventional climate and weather forecasting primarily depends on
physics-based numerical simulations (Shuman, 1989; Warner, 2010). In particular, short-term forecasts rely on established Numerical Weather Prediction (NWP) systems—such as the Unified Model
(UM) (Bush et al., 2020) or other frameworks used in the U.S. (Powers et al., 2017) and Europe—that
solve the so-called primitive equations (Wedi et al., 2015), a topic of extensive mathematical and
computational research (Lions et al., 1992). Meanwhile, longer-term forecasts employ dedicated
climate models, with Earth System Models (ESMs) (Mukhopadhyay et al., 2019) representing the
cutting edge by coupling atmospheric, cryospheric, terrestrial, and oceanic processes. Although these
modeling approaches have seen considerable success, they still face notable challenges, including sensitivity to initial conditions, structural inconsistencies across models (Bauer et al., 2015), significant
computational burdens, and marked regional variability.


**Deep learning for forecasting** . Recent advances in deep learning have yielded promising results
for weather forecasting by bypassing some of the complexities of physics-based simulations. For
instance, Rasp et al. (2020) applied pre-training with ResNet for medium-range weather prediction,
and utilized large ensembles of deep models to capture sub-seasonal variations (Han et al., 2024).
Other notable works include radar-based deep generative models for nowcasting (Ravuri et al., 2021)
and graph neural network-based forecasting in GraphCast (Lam et al., 2023). In addition, FourCastNet
(Kurth et al., 2023) and Pangu-Weather (Bi et al., 2023) represent state-of-the-art neural forecasting
approaches that harness data-driven backbones, such as Vision Transformer, UNet, and autoencoders.
Despite their empirical strengths, these methods tend to overlook key physical principles and seldom
provide uncertainty estimates, limiting their interpretability and robustness.


**Physics-Informed Machine Learning** . Neural ODEs frame time derivatives as learnable neural
networks (Fermanian et al., 2021), and have been extended to incorporate physics-based constraints
(Verma et al., 2024). Physics-Informed Neural Networks (PINNs) (Cai et al., 2021) embed mechanistic knowledge into DEs, and a broader line of research focuses on discovering interpretable
differential equations (Brunton and Kutz, 2024). Extending such ideas to Neural PDEs often requires
specialized spatial discretizations (Kochkov et al., 2024) or functional representations (Seol et al.,
2024). Several studies have also used machine learning to improve fluid dynamics models (Choi
et al., 2024). Notably, most of these works deal with smaller-scale fluid systems rather than the global
scope demanded by climate or weather applications.


2


3 METHODOLOGY


Our model is fundamentally constructed using the Method of Lines (MOL) framework, as described
in (Verma et al., 2024). This approach initially formulates the problem in terms of partial differential
equations (PDEs) governing the evolution of multiple variables. To approximate the spatial derivatives
in these PDEs, we employ a finite difference scheme, converting the PDEs into a system of ordinary
differential equations (ODEs). Subsequently, we effectively approximate the temporal dynamics
of the atmospheric variables by solving this system through a neural ODE framework (Chen et al.,
2018). The detailed formulation is outlined below.


3.1 PRELIMINARY


Consider a set of _d_ atmospheric variables denoted by **q** ( **x** _, t_ ) = _{qi_ ( **x** _, t_ ) _}_ _[d]_ _i_ =1 [(e.g.] [temperature,]
geopotential height) that depend on the spatial location **x** _∈_ [ _−_ 90 _,_ 90] _×_ [0 _,_ 360] (representing latitude
and longitude on a sphere domain, such as Earth) and time _t >_ 0. Observations of these variables are
collected at a set of uniform grid points _{_ **x** _n}_ _[N]_ _n_ =1 [, where the spatial domain consists of] _[ H]_ [latitude]
points and _W_ longitude points, resulting in a total of _N_ = _HW_ observations. In addition, we can
consider the velocity field **U** ( _t_ ) = _{{_ **u** _i_ ( **x** _n, t_ ) _}_ _[d]_ _i_ =1 _[}]_ _n_ _[N]_ =1 [that governs the advection of atmospheric]
variables. Given the velocity field, we model the temporal evolution of these variables using the
following governing equations as in (Verma et al., 2024).


_∂_
(1)
_∂t_ _[q][i]_ [(] **[x]** _[, t]_ [) =] _[ −]_ **[u]** _[i]_ [(] **[x]** _[, t]_ [)] _[ · ∇][q][i]_ [(] **[x]** _[, t]_ [)] _[ −]_ _[q][i]_ [(] **[x]** _[, t]_ [)] _[∇·]_ **[ u]** _[i]_ [(] **[x]** _[, t]_ [)] _[,]_

_∂_         - **Q** ( _t_ ) _, ∇_ **Q** ( _t_ ) _,_ **U** ( _t_ ) _, g_ ( _{_ **x** _n}_ _[N]_ _n_ =1 _[, t]_ [)]         - _,_
_∂t_ **[u]** _[i]_ [(] **[x]** _[, t]_ [) =] _[ f][η]_


where _∇_ denotes the spatial gradient, **Q** ( _t_ ) represents the set _{_ **q** ( **x** _n, t_ ) _}_ _[N]_ _n_ =1 [,] _[ g]_ [ is a spatio-temporal]
embedding function and _fη_ is a trainable neural network with parameter _η_ . The second equation
implies that the velocity of each variable could be influenced by the other variables.


To transform Equation (1) into a system of ODEs, we approximate the spatial derivatives using a
finite difference, denoted as _F_ (see Section 3.3 for details). The system for all variables _Q_ ( _t_ ) at the

[�]
points of the grid is given below.


Using the estimated **Q** ( _t_ ) and real data, we train _fη_ by minimizing the negative log-likelihood loss
function, as defined in Sections 3.7 and 3.8 of (Verma et al., 2024).


3.2 TENSOR FIELD NEURAL PDE (TFNP)


In this paper, we parametrize the nonlinear operator _fη_ in Equation (1) (illustrated in Figure 1)
with a Tensor Field Network (TFN) _fT F N_ (Thomas et al., 2018; Weiler et al., 2018; Kondor et al.,
2018), combined with an attention mechanism, _fatt_ (Vaswani et al., 2017), rather than employing
a convolutional neural network (CNN). Although CNNs are often adopted for _fη_ because they can
approximate finite difference schemes on a uniform Euclidean grid (Brandstetter et al., 2022; Long
et al., 2018), global climate data are typically sampled uniformly in latitude and longitude coordinates.


3


_∂_ **q** ( **x** 1 _,t_ )

_∂t_
...
_∂_ **q** ( **x** _N_ _,t_ )

_∂t_





 _≈_










_F_ �( **q** ( **x** 1 _, t_ ) _, {_ **q** ( **x** _n, t_ ) _}n∈N_ (1)) _,_ **u** ( **x** 1 _, t_ ) _, {_ **u** ( **x** _n, t_ ) _}n∈N_ (1))
...
_F_ �( **q** ( **x** _N_ _, t_ ) _, {_ **q** ( **x** _n, t_ ) _}n∈N_ ( _N_ )) _,_ **u** ( **x** _N_ _, t_ ) _, {_ **u** ( **x** _n, t_ ) _}n∈N_ ( _N_ ))





_d_ **Q** ( _t_ )

=
_dt_









 _∈_ R _[Nd]_ _._



Here, _N_ ( _i_ ) denotes the index set corresponding to the neighborhood of the grid point **x** _i_ required for
the finite-difference approximation. The system that governs **U** ( _t_ ) can be formulated analogously.
Consequently, the complete system consists of 3 _Nd_ components when each atmospheric variable
is considered a separate component. By integrating Equation (2) using the Runge-Kutta method to
solve this system, we can estimate the values of the variables _{qi}_ _[d]_ _i_ =1 [at all grid points] _[ {]_ **[x]** _[n][}]_ _n_ _[N]_ =1 [.]


_ds_ (2)


- **Q** ( _t_ )
**U** ( _t_ )


- = - **Q** ( _t_ 0)
**U** ( _t_ 0)


- - _t_
+


_t_ 0


- _d_ **Q** ( _s_ )

_ds_
_d_ **U** ( _s_ )

_ds_


|B|A|
|---|---|
|D|C|


|B|D|
|---|---|
|A|C|


Here, _Cin, Cout_ denote the input and output channel dimensions of _fT F N_ . Additionally, we incorporate an attention-based network, _fatt_, following the architecture proposed in (Verma et al., 2024).
Consequently, the final _fη_ is constructed as the sum of the attention network _fatt_ and the Tensor
Field Network _fT F N_,


4


180°


Rotated Map (w.r.t. polar axis)


Temperature


-60°C -20°C 20°C


Rotated Map (w.r.t. equatorial axis)


Figure 1: Graphical overview of PA-TFNP. The tensor field network (TFN) and attention layer are
employed to model _fη_ and the advection-diffusion equation is introduced. TFN accounts for the
spherical geometry of Earth. For instance, earth can be divided into four regions (A, B, C, D) based
on latitude [0 _[◦]_ _,_ 90 _[◦]_ ], [ _−_ 90 _[◦]_ _,_ 0 _[◦]_ ] and longitude [0 _[◦]_ _,_ 180 _[◦]_ ], [180 _[◦]_ _,_ 360 _[◦]_ ]. Projecting temperature data
onto the latitude-longitude plane forms the leftmost rectangular map. Rotation around the polar axis
leads to translation on this map, while rotation around the equatorial axis additionally reflects region
B and C. PA-TFNP processes these partitioned region and outputs corresponding (A*, B*, C*, D*),
ensuring rotational equivariance. Combining PA-TFNP with attention yields the final model _fη_ .


This leads to geometric distortions near the polar regions, negatively impacting prediction accuracy.
Moreover, CNNs inherently fail to capture rotation-equivariant properties essential for processing
spherical data. As in Figure 1, while rotations around the polar axis correspond to straightforward
transformations in a periodic domain, rotations around the equatorial axis involve transformations
coupled with reflections. Consequently, a CNN with fixed filters cannot approximate rotations of
the latter type, as local features along the boundaries separating regions _A_, _B_, _C_, and _D_ become
distorted. We adopted a neural network based on tensor products instead of CNNs to mitigate this
problem. This approach is inherently rotation equivariant, ensuring that transformations affect points
near the poles and the equator consistently, without introducing distortion. The detailed formulation
is as follows. The function _fη_ takes as input **Q** ( _t_ ) _∈_ R _[N]_ _[×][d]_ _, ∇_ **Q** ( _t_ ) _∈_ R _[N]_ _[×]_ [2] _[d]_ _,_ **U** ( _t_ ) _∈_ R _[N]_ _[×]_ [2] _[d]_, and
_g_ ( _{xn}_ _[N]_ _n_ =1 _[, t]_ [)] _[ ∈]_ [R] _[N]_ _[×][e]_ [, where] _[ e]_ [ denotes the embedding dimension introduced by] _[ g]_ [.] [If inputs in] _[ T]_
time steps are considered simultaneously, the dimension of input _I_ is given by _T ×N ×_ (5 _d_ + _e_ ). After
reshaping _I_ into a tensor of size _N ×Cin_, we can define the neural network _fη_ : R _[N]_ _[×][C][in]_ _→_ R _[N]_ _[×][C][out]_
as a tensor product-based function. This function is parameterized by a trainable weight tensor
_W_ [ _cout, c_ 1 _.c_ 2] for indices _cout, c_ 1 _, c_ 2 _∈_ [ _Cout_ ] _,_ [ _Cin_ ] _,_ [ _Cin_ ], and is formulated as:


_C_ out

- _W_ [ _c_ out _, c_ 1 _, c_ 2]( _I_ [ _i, c_ 1] _· I_ [ _i, c_ 2]) _,_ _∀i ∈_ [ _N_ ] _._


_c_ 2=1


_fT F N_ ( _I_ [ _i, c_ out]) = _I_ _⊗_ _I_ =


_C_ out


_c_ 1=1


(a) Neumann padding (b) Average padding


(c) Ground truth values and absolute errors


Figure 2: (a) Description of Neumann padding. (b) Description of the average padding. (c) Ground
truth values for z, t2, t2m, u10, and v10 and absolute errors of ClimODE and the proposed TFNP.


3.3 PHYSICS-AWARE TENSOR FIELD NEURAL PDE (PA-TFNP)


In this section, we introduce PA-TFNP, an extension of TFNP that incorporates physical constraints
into the model. We make three key modifications. First, we apply boundary conditions that reflect
the domain’s physical properties. Second, spatial derivatives are computed using spherical operators
to capture Earth’s geometry. Third, we augment the inputs to _fη_ ( _·_ ) in Equation equation 1 with
physically relevant features: ground wind magnitude, lapse rate, and wind vorticity. We also modify
the PDE solver to blend neural outputs with physics-based tendencies for improved interpretability
and fidelity.


BOUNDARY CONDITIONS


ClimODE exhibits unexpected errors near the boundary of the domain (see Figure 2), primarily
due to the discretization of the sphere onto a longitude–latitude rectangular domain. This issue
arises from the absence of proper boundary conditions in the original ClimODE formulation (Verma
et al., 2024). The boundary conditions are implemented through an appropriate padding strategy and
incorporated into the advection–diffusion equation during gradient computation. We propose two
padding strategies, Neumann padding and average padding, both reflecting the physical characteristics
of the domain.


In both strategies, circular padding is applied along the longitudinal boundaries, effectively transforming the rectangular domain into a cylindrical one. For Neumann padding, replicate padding is
used along the latitudinal boundaries, corresponding to homogeneous Neumann boundary conditions
at the north and south poles (see Figure 2a). In the case of average padding, we extend the domain by
padding with the average values of the boundary: _µ_ 1 = 641 �64 _i_ =1 _[u]_ [1] _[,i]_ [ and] _[ µ]_ [2] [=] 641 �64 _i_ =1 _[u]_ [2] _[,i]_ [.] [This]
transforms the rectangular domain into a sphere-like domain (see Figure 2b). Figure 2c illustrates
that TFN, equipped with this padding scheme, effectively captures the solution behavior near the
poles. With a rotation-equivariant property, TFNP maintains consistent prediction accuracy across


5


all regions, including areas near the poles, resulting in robust predictions both at the boundaries and
within the domain interior.


SPATIAL DERIVATIVE APPROXIMATION


This section outlines the computation of the spatial derivatives in Equation (1). The method proposed
in (Verma et al., 2024) estimates the derivatives by directly computing finite difference approximations
along latitude and longitude, respectively. However, in a spherical domain, a given longitudinal
difference corresponds to varying Euclidean distances depending on latitude. To account for this, we
adopt a central finite difference scheme with a distance correction term:


_∇qi_ (( _ϕ, λ_ ) _, t_ )


where _R_ represents the Earth’s radius, and _h_ and _w_ denote the uniform grid spacing in latitude and
longitude, respectively. Given the inherent periodicity in the longitudinal direction ( _λ_ ), all grid points
along this axis can be treated as interior points. Furthermore, we impose boundary conditions such
as Neumann or periodic conditions on the latitude ( _ϕ_ ), ensuring that all points within the domain
are treated as interior points. Under these conditions, the central finite difference scheme can be
consistently applied throughout the entire domain.


ADDITIONAL PHYSICS-DERIVED FEATURES


To augment the original TFNP framework, we introduce three physics-informed features: **(i)** the nearsurface wind magnitude _|_ _**V**_ 10 _|_ = ~~�~~ _u_ [2] 10 [+] _[ v]_ 10 [2] [,] **[ (ii)]** [ the low-tropospheric lapse rate][ ∆] _[t]_ [ =] _[ t][ −]_ _[t]_ [2] _[m]_ [, and]
**(iii)** the relative vorticity _ζ_ = _∂yv_ 10 _−_ _∂xu_ 10, computed using spherical gradients. These quantities
capture dynamic and thermodynamic processes essential to atmospheric motion.


MODIFIED PRIMITIVE EQUATION


To improve physical realism and long-term stability, we extend the neural advection formulation in
Equation 1 by incorporating physics-inspired diffusion and momentum correction terms.


First, scalar quantities such as temperature, humidity, and geopotential exhibit diffusive behavior in
the real atmosphere, caused by unresolved subgrid turbulence and eddy transport Haltiner (1971);
Lions et al. (1992); Warner (2010). To reflect this, we introduce a spatially varying diffusion term
with a learnable non-negative coefficient _α_ ( **x** ) _∈_ R _[d][×][H][×][W]_ . The scalar transport equation is modified
as follows:


_∂qi_ ( **x** _, t_ )

= _−_ **u** _i_ ( **x** _, t_ ) _· ∇qi_ ( **x** _, t_ ) _−_ _qi_ ( **x** _, t_ ) _∇·_ **u** _i_ ( **x** _, t_ ) + _α_ ( **x** )∆ _qi_ ( **x** _, t_ ) _,_
_∂t_


where the last term mimics anisotropic and spatially varying diffusion. Next, we augment the
neural tendency with physically meaningful momentum dynamics for the learned velocity field **u** _i_ .
Specifically, we apply a time-dependent blending of neural predictions and physically grounded
operators:


_∂_ **u** _i_ ( **x** _, t_ ) = (1 _−_ _βt_ ) _fη_    - **Q** ( _t_ ) _, ∇_ **Q** ( _t_ ) _,_ **U** ( _t_ ) _, g_ ( _{_ **x** _n}_ _[N]_ _n_ =1 _[, t]_ [)]    - + _βt f_ phys( **x** _, t,_ **u** _i_ ) _,_

_∂t_


where the blend factor _βt_ = 1 _−_ exp( _−t/τ_ 0) gradually shifts preference from neural inference
to physical consistency over time. The physical operator _f_ phys imposes structure on the velocity
evolution by incorporating key dynamical effects:


_f_ phys( **x** _, t,_ **u** _i_ ) = _−∇_ Φ + _ν_ ∆ **u** _i −_ _γ_ **u** _i,_


where Φ denotes the geopotential field (i.e., Φ = _z_ ), and _ν_, _γ_ are learnable viscosity and linear drag
coefficients, respectively. This hybrid formulation preserves the expressiveness of neural models
while enforcing core physical constraints, improving both predictive performance and stability in
long-range forecasts.


6


_≈_ - _qi_ (( _ϕ_ + _h, λ_ ) _, t_ ) _−_ _qi_ (( _ϕ −_ _h, λ_ ) _, t_ ) _,_ _[q][i]_ [((] _[ϕ, λ]_ [ +] _[ w]_ [)] _[, t]_ [)] _[ −]_ _[q][i]_ [((] _[ϕ, λ][ −]_ _[w]_ [)] _[, t]_ [)]
_Rhπ/_ 180 _Rhπ_ cos _ϕ/_ 180


- (3)
_,_


4 EXPERIMENTS


We evaluate the performance of PA-TFNP by comparing it with the neural ODE, ClimaX (Nguyen
et al., 2023) and ClimODE (Verma et al., 2024), a state-of-the-art data-driven global climate forecasting model. To ensure a fair comparison, we follow the experimental setup of (Verma et al.,
2024), except for specific modifications detailed below. We utilize the ERA5 dataset from WeatherBench (Rasp et al., 2020), selecting _d_ = 5 key atmospheric variables: ground temperature ( _t_ 2 _m_ ),
atmospheric temperature ( _t_ ), geopotential height ( _z_ ), and ground wind components ( _u_ 10 _, v_ 10).
All variables are normalized to the range [0, 1] using min-max scaling. Further details on dataset
preprocessing and training settings remain consistent with those in (Verma et al., 2024) and Appendix
B. All experiments were conducted using a single RTX 4090 GPU.


4.1 GLOBAL WEATHER FORECASTING ACROSS VARYING TEMPORAL AND SPATIAL
RESOLUTIONS


To evaluate the scalability and generalization of PA–TFNP across both spatial and temporal dimensions, we conduct experiments on global weather forecasting at two different settings: (a) long-term
prediction over 5 days at a coarse resolution (5 _._ 625 _[◦]_ ), and (b) short-term prediction over 6 to 42 hours
at a finer resolution (11 _._ 25 _[◦]_ ). Figure 3 summarizes the RMSE results for the five key atmospheric
variables (z, t, t2m, u10, v10), comparing PA–TFNP with the state-of-the-art ClimODE baseline.


Across both resolutions, PA–TFNP consistently outperforms ClimODE. In the long-term setting (first
row in Figure 3), our model demonstrates particularly large improvements in forecasting geopotential
height and atmospheric temperature. Similarly, in the short-term setting (second row in Figure 3),
PA–TFNP shows improved accuracy across all lead times, with gains becoming more pronounced
beyond 24 hours. This indicates that the model maintains robustness even as the forecasting horizon
increases. These results confirm the effectiveness of PA–TFNP in learning global-scale spatiotemporal
dynamics, while preserving accuracy across varying resolutions and forecast ranges.


Figure 3: Comparison of RMSE values for ClimODE and the proposed PA-TFNP (Ours) across two
spatiotemporal resolutions. The results highlight the performance differences for key atmospheric
variables. Results are reported as mean _±_ standard deviation. First row: long-term prediction at
a resolution of 5 _._ 625 _[◦]_ . Second row: Short-term prediction at a resolution of 11 _._ 25 _[◦]_ . PA-TFNP
outperforms ClimODE by 38.12% on daily data and by 78.92% on hourly data.


4.2 SHORT-TERM REGIONAL WEATHER FORECASTING


We evaluate short-term (up to 24 hours) regional weather forecasting over the Australia and the South
American region. Table 1 presents the RMSE (mean _±_ standard deviation) of various models across
five key atmospheric variables. Our proposed model, PA–TFNP, demonstrates strong predictive
accuracy overall, particularly for the geopotential height (z) and temperature (t) variables, where
it consistently outperforms all baselines across all lead times. Compared to the current state-ofthe-art model, ClimODE, PA–TFNP achieves lower RMSE, especially at longer horizons (18–24h),
demonstrating improved temporal robustness.


7


Table 1: Comparison of RMSE values for baseline models and the proposed PA-TFNP (Ours) across
different regions. The results highlight the performance differences for key atmospheric variables.
Results are reported as mean _±_ standard deviation.


Australia South America


Value Hour NODE ClimaX ClimODE **PA–TFNP** NODE ClimaX ClimODE **PA–TFNP**


For wind components, PA–TFNP slightly outperforms ClimODE in most settings, particularly at
longer lead times. Notably, for t2m, PA–TFNP underperforms at earlier lead times but catches up
or surpasses baselines at 24h. This may indicate a trade-off between local variance sensitivity and
longer-horizon stability.


4.3 MONTHLY AVERAGED WEATHER FORECASTING


Next, we evaluate the predictive accuracy of ClimODE, CilmaX, TFNP, and PA-TFNP over a
two-month lead time. All models predict the global two-month averaged future states based on
an initial monthly average state. Table 2 provides a detailed comparison of RMSE values for
various atmospheric variables, showing that PA-TFNP consistently outperforms other benchmarks,
particularly in predicting geopotential height (z), atmospheric temperature (t) and ground temperature
(t2m). The lower RMSE values in the results indicate that PA-TFNP more accurately captures
complex climate patterns, offering enhanced reliability for extended-range climate forecasting.


4.4 ABLATION STUDIES


**Assessing rotational equivariance:** **ClimODE vs TFNP.** To further evaluate the spatial prediction
capabilities of TFNP, we compare its performance with ClimODE in terms of absolute prediction
error across five key atmospheric variables (see Figure 6 in Appendix A). The results demonstrate
that TFNP consistently achieves lower error magnitudes than ClimODE, particularly in geophysically
challenging regions such as the poles and the equator. These regions are often prone to distortions
due to their rotational properties, where ClimODE exhibits noticeable artifacts. In contrast, TFNP
maintains strong spatial consistency, owing to its rotation-equivariant architecture. These findings
underscore the importance of incorporating geometric inductive biases, such as rotational equivariance,
in improving model robustness and accuracy in global-scale geophysical forecasting.


8


z


t


t2m


u10


v10


6 251.4 190.2 103 _._ 8 _±_ 14 _._ 6 **79** _._ **5** _±_ 19 _._ 9 225.6 205.4 107 _._ 7 _±_ 20 _._ 2 **87** _._ **5** _±_ 22 _._ 0
12 344.8 184.7 170 _._ 7 _±_ 21 _._ 0 _._ **8** _±_ 30 _._ 1 365.6 220.2 169 _._ 4 _±_ 29 _._ 6 _._ **2** _±_ 31 _._ 3
18 539.9 222.2 211 _._ 1 _±_ 31 _._ 6 _._ **6** _±_ 43 _._ 8 551.9 269.1 237 _._ 8 _±_ 32 _._ 2 _._ **1** _±_ 43 _._ 4
24 632.7 324.9 308 _._ 2 _±_ 30 _._ 6 _._ **8** _±_ 59 _._ 5 660.3 301.8 292 _._ 0 _±_ 38 _._ 9 _._ **3** _±_ 57 _._ 8


6 1.37 1.19 1 _._ 05 _±_ 0 _._ 12 **0** _._ **87** _±_ 0 _._ 14 1.58 1.38 **0** _._ **97** _±_ 0 _._ 13 1 _._ 01 _±_ 0 _._ 16
12 2.18 1.30 1 _._ 20 _±_ 0 _._ 16 **1** _._ **07** _±_ 0 _._ 18 2.18 1.62 1 _._ 25 _±_ 0 _._ 18 **1** _._ **18** _±_ 0 _._ 18
18 2.68 1.39 1 _._ 33 _±_ 0 _._ 21 **1** _._ **19** _±_ 0 _._ 20 2.74 1.79 1 _._ 43 _±_ 0 _._ 20 **1** _._ **29** _±_ 0 _._ 18
24 3.32 1.92 1 _._ 63 _±_ 0 _._ 24 **1** _._ **31** _±_ 0 _._ 23 3.41 1.97 1 _._ 65 _±_ 0 _._ 26 **1** _._ **44** _±_ 0 _._ 21


6 1.88 1.57 **0** _._ **80** _±_ 0 _._ 13 2 _._ 42 _±_ 0 _._ 70 2.12 1.85 **1** _._ **33** _±_ 0 _._ 26 1 _._ 73 _±_ 0 _._ 67
12 2.02 1.57 **1** _._ **10** _±_ 0 _._ 22 2 _._ 98 _±_ 1 _._ 50 2.42 2.08 **1** _._ **04** _±_ 0 _._ 17 2 _._ 37 _±_ 1 _._ 20
18 3.51 1.72 **1** _._ **23** _±_ 0 _._ 24 2 _._ 37 _±_ 0 _._ 55 2.60 2.15 **0** _._ **98** _±_ 0 _._ 17 1 _._ 87 _±_ 0 _._ 84
24 2.46 2.15 1 _._ 25 _±_ 0 _._ 25 **1** _._ **16** _±_ 0 _._ 24 2.56 2.23 1 _._ 17 _±_ 0 _._ 26 **1** _._ **15** _±_ 0 _._ 27


6 1.91 1.40 **1** _._ **35** _±_ 0 _._ 17 1 _._ 43 _±_ 0 _._ 19 1.94 1.27 **1** _._ **25** _±_ 0 _._ 18 1 _._ 42 _±_ 0 _._ 27
12 2.86 1.77 1 _._ 78 _±_ 0 _._ 21 **1** _._ **74** _±_ 0 _._ 22 2.74 1.57 **1** _._ **49** _±_ 0 _._ 23 1 _._ 56 _±_ 0 _._ 30
18 3.44 2.03 1 _._ 96 _±_ 0 _._ 25 **1** _._ **88** _±_ 0 _._ 26 3.24 1.83 1 _._ 81 _±_ 0 _._ 29 **1** _._ **69** _±_ 0 _._ 29
24 3.91 2.64 2 _._ 33 _±_ 0 _._ 33 **2** _._ **06** _±_ 0 _._ 28 3.77 2.04 2 _._ 08 _±_ 0 _._ 35 **1** _._ **86** _±_ 0 _._ 32


6 2.38 1.47 **1** _._ **44** _±_ 0 _._ 20 1 _._ 56 _±_ 0 _._ 19 2.29 1.31 **1** _._ **30** _±_ 0 _._ 21 1 _._ 68 _±_ 0 _._ 39
12 3.60 1.79 1 _._ 87 _±_ 0 _._ 26 **1** _._ **78** _±_ 0 _._ 25 3.42 **1** _._ **64** 1 _._ 71 _±_ 0 _._ 28 1 _._ 93 _±_ 0 _._ 40
18 4.31 2.33 2 _._ 23 _±_ 0 _._ 23 **2** _._ **04** _±_ 0 _._ 26 4.16 1.90 2 _._ 07 _±_ 0 _._ 31 **1** _._ **88** _±_ 0 _._ 37
24 4.88 2.58 2 _._ 53 _±_ 0 _._ 32 **2** _._ **23** _±_ 0 _._ 30 4.76 2.14 2 _._ 43 _±_ 0 _._ 34 **2** _._ **06** _±_ 0 _._ 37


Table 2: Comparison of RMSE values for different models across two months. The results highlight
the performance of TFNP and PA-TFNP compared to ClimODE and other baseline models for key
atmospheric variables.


Value Months ClimaX ClimODE TFNP (ours) PA-TFNP (ours)


1 580 _._ 73 692 _._ 10 _±_ 119 _._ 80 529 _._ 44 _±_ 95 _._ 77 _._ **01** _±_ 79 _._ 50
z
2 773 _._ 40 870 _._ 57 _±_ 72 _._ 58 527 _._ 07 _±_ 84 _._ 54 _._ **39** _±_ 70 _._ 13


1 2 _._ 89 2 _._ 81 _±_ 0 _._ 48 2 _._ 58 _±_ 0 _._ 56 **2** _._ **48** _±_ 0 _._ 45
t
2 4 _._ 39 3 _._ 20 _±_ 1 _._ 02 **2** _._ **42** _±_ 0 _._ 42 2 _._ 44 _±_ 0 _._ 21

1 2 _._ 97 4 _._ 33 _±_ 0 _._ 38 2 _._ 63 _±_ 0 _._ 52 **2** _._ **53** _±_ 0 _._ 34
t2m
2 5 _._ 07 4 _._ 99 _±_ 0 _._ 48 2 _._ 95 _±_ 0 _._ 45 **2** _._ **95** _±_ 0 _._ 30


1 **1** _._ **80** 1 _._ 98 _±_ 0 _._ 19 1 _._ 86 _±_ 0 _._ 23 1 _._ 83 _±_ 0 _._ 23
u10
2 **1** _._ **92** 2 _._ 09 _±_ 0 _._ 11 2 _._ 40 _±_ 0 _._ 22 2 _._ 32 _±_ 0 _._ 21


1 1 _._ 50 1 _._ 66 _±_ 0 _._ 18 1 _._ 40 _±_ 0 _._ 10 **1** _._ **39** _±_ 0 _._ 12
v10
2 **1** _._ **71** 1 _._ 98 _±_ 0 _._ 11 1 _._ 95 _±_ 0 _._ 18 1 _._ 91 _±_ 0 _._ 21


**Benefits of Physics-Aware Modeling for Long-Term Stability:** **TFNP vs PA-TFNP.** To evaluate
the effectiveness of Physics-Aware modeling, we compared the performance of the PA-TFNP model,
which incorporate physical operators and features against the TFNP model. Experimental results
shows that PA-TFNP consistently outperforms the TFNP model at extended forecast horizons beyond
24 hours, across all scalar quantities. These results underscore the importance of embedding physical
properties within predictive models to achieve stable and reliable long-term forecasting, as clearly
illustrated in Figure 4.


Figure 4: RMSE comparison of the TFNP baseline and Physics-Aware TFNP (PA-TFNP) models over
extended forecast horizons (up to 138 hours) across multiple atmospheric variables (z, t, t2m, u10,
v10). The PA-TFNP model, incorporating physical constraints, consistently demonstrates improved
accuracy, highlighting the importance of physics-informed modeling for stable long-term predictions.


5 CONCLUSION AND LIMITATIONS


In this work, we have presented the Physics-Aware TFNP, a novel framework that combines deep
learning with fundamental physical principles to tackle climate and weather prediction tasks more
accurately and robustly. By integrating gradient computation and boundary treatment methods
rooted in numerical techniques and by incorporating physically consistent diffusion terms and
divergence-free conditions, our approach addresses the shortcomings of both purely data-driven
and physics-agnostic models. TFNP not only demonstrates state-of-the-art forecasting performance
but also maintains physical fidelity, offering enhanced interpretability and reliability. We anticipate
that the mathematical principles introduced here will generalize across a broad range of scientific
computing domains, thereby accelerating progress in both global and regional weather prediction.


As expected, the rotation-equivariant feature of the proposed PA-TFNP plays an important role in the
global forecasting model. However, this characteristic appears to offer limited benefits for regional
forecasting. This limitation warrants further investigation in future work. We have added diffusion
terms to the model equations for all predictive variables. However, the modification of the model
equation should be tailored to each variable, as their physical interpretations differ significantly.
For instance, the temperature variable and ground wind variables represent fundamentally different
physical phenomena and therefore should be modeled using distinct equations.


9


REFERENCES


Florence Rabier, Heikki Järvinen, E Klinker, J-F Mahfouf, and A Simmons. The ecmwf operational
implementation of four-dimensional variational assimilation. i: Experimental results with simplified
physics. _Quarterly Journal of the Royal Meteorological Society_, 126(564):1143–1170, 2000.


F Rawlins, SP Ballard, KJ Bovis, AM Clayton, D Li, GW Inverarity, AC Lorenc, and TJ Payne. The
met office global four-dimensional variational data assimilation scheme. _Quarterly Journal of the_
_Royal Meteorological Society:_ _A journal of the atmospheric sciences, applied meteorology and_
_physical oceanography_, 133(623):347–362, 2007.


Philip Duncan Thompson. Numerical weather analysis and prediction. _(No Title)_, 1961.


Jacques-Louis Lions, Roger Temam, and Shouhong Wang. New formulations of the primitive
equations of atmosphere and applications. _Nonlinearity_, 5(2):237, 1992.


G.J. Haltiner. _Numerical Weather Prediction_ . Wiley, 1971. ISBN 9780471345800. [URL https:](https://books.google.co.kr/books?id=RTZRAAAAMAAJ)
[//books.google.co.kr/books?id=RTZRAAAAMAAJ.](https://books.google.co.kr/books?id=RTZRAAAAMAAJ)


Jean Coiffier. _Fundamentals of numerical weather prediction_ . Cambridge University Press, 2011.


Tsann-Wang Yu. Advances and challenges in numerical weather and climate prediction. In _AIP_
_Conference Proceedings_, volume 1280, pages 142–158. American Institute of Physics, 2010.


Thomas Tomkins Warner. _Numerical weather and climate prediction_ . cambridge university press,
2010.


J David Neelin. _Climate change and climate modeling_ . Cambridge University Press, 2010.


David Bader, Curt Covey, William Gutowski, Isaac Held, Kenneth Kunkel, Ronald Miller, Robin
Tokmakian, and Minghua Zhang. Climate models: an assessment of strengths and limitations.
2008.


Kaifeng Bi, Lingxi Xie, Hengheng Zhang, Xin Chen, Xiaotao Gu, and Qi Tian. Accurate mediumrange global weather forecasting with 3d neural networks. _Nature_, 619(7970):533–538, 2023.


Remi Lam, Alvaro Sanchez-Gonzalez, Matthew Willson, Peter Wirnsberger, Meire Fortunato, Ferran
Alet, Suman Ravuri, Timo Ewalds, Zach Eaton-Rosen, Weihua Hu, et al. Learning skillful
medium-range global weather forecasting. _Science_, 382(6677):1416–1421, 2023.


Cristian Bodnar, Wessel P Bruinsma, Ana Lucic, Megan Stanley, Johannes Brandstetter, Patrick
Garvan, Maik Riechert, Jonathan Weyn, Haiyu Dong, Anna Vaughan, et al. Aurora: A foundation
model of the atmosphere. _arXiv preprint arXiv:2405.13063_, 2024.


Dmitrii Kochkov, Janni Yuval, Ian Langmore, Peter Norgaard, Jamie Smith, Griffin Mooers, Milan
Klöwer, James Lottes, Stephan Rasp, Peter Düben, et al. Neural general circulation models for
weather and climate. _Nature_, 632(8027):1060–1066, 2024.


Alex Bihlo. A generative adversarial network approach to (ensemble) weather prediction. _Neural_
_Networks_, 139:1–16, 2021.


Yogesh Verma, Markus Heinonen, and Vikas Garg. ClimODE: Climate forecasting with physicsinformed neural ODEs. In _The Twelfth International Conference on Learning Representations_,
2024. [URL https://openreview.net/forum?id=xuY33XhEGR.](https://openreview.net/forum?id=xuY33XhEGR)


Jaideep Pathak, Shashank Subramanian, Peter Harrington, Sanjeev Raja, Ashesh Chattopadhyay,
Morteza Mardani, Thorsten Kurth, David Hall, Zongyi Li, Kamyar Azizzadenesheli, et al. Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators.
_arXiv preprint arXiv:2202.11214_, 2022.


Frederick G. Shuman. History of numerical weather prediction at the national meteorological center. _Weather and Forecasting_, 4(3):286 – 296, 1989. doi: 10.1175/1520-0434(1989)004<0286:
HONWPA>2.0.CO;2. URL [https://journals.ametsoc.org/view/journals/](https://journals.ametsoc.org/view/journals/wefo/4/3/1520-0434_1989_004_0286_honwpa_2_0_co_2.xml)
[wefo/4/3/1520-0434_1989_004_0286_honwpa_2_0_co_2.xml.](https://journals.ametsoc.org/view/journals/wefo/4/3/1520-0434_1989_004_0286_honwpa_2_0_co_2.xml)


10


Mike Bush, Tom Allen, Caroline Bain, Ian Boutle, John Edwards, Anke Finnenkoetter, Charmaine
Franklin, Kirsty Hanley, Humphrey Lean, Adrian Lock, et al. The first met office unified model–
jules regional atmosphere and land configuration, ral1. _Geoscientific Model Development_, 13(4):
1999–2029, 2020.


Jordan G Powers, Joseph B Klemp, William C Skamarock, Christopher A Davis, Jimy Dudhia,
David O Gill, Janice L Coen, David J Gochis, Ravan Ahmadov, Steven E Peckham, et al. The
weather research and forecasting model: Overview, system efforts, and future directions. _Bulletin_
_of the American Meteorological Society_, 98(8):1717–1737, 2017.


NP Wedi, P Bauer, W Denoninck, M Diamantakis, M Hamrud, C Kuhnlein, S Malardel, K Mogensen,
G Mozdzynski, and PK Smolarkiewicz. The modelling infrastructure of the integrated forecasting
system: Recent advances and future challenges. 2015.


P Mukhopadhyay, VS Prasad, R Phani Murali Krishna, Medha Deshpande, Malay Ganai, Snehlata
Tirkey, Sahadat Sarkar, Tanmoy Goswami, CJ Johny, Kumar Roy, et al. Performance of a very
high-resolution global forecast system model (gfs t1534) at 12.5 km over the indian region during
the 2016–2017 monsoon seasons. _Journal of Earth System Science_, 128:1–18, 2019.


Peter Bauer, Alan Thorpe, and Gilbert Brunet. The quiet revolution of numerical weather prediction.
_Nature_, 525(7567):47–55, 2015.


Stephan Rasp, Peter D Dueben, Sebastian Scher, Jonathan A Weyn, Soukayna Mouatadid, and Nils
Thuerey. Weatherbench: a benchmark data set for data-driven weather forecasting. _Journal of_
_Advances in Modeling Earth Systems_, 12(11):e2020MS002203, 2020.


Tao Han, Song Guo, Fenghua Ling, Kang Chen, Junchao Gong, Jingjia Luo, Junxia Gu, Kan Dai,
Wanli Ouyang, and Lei Bai. Fengwu-ghr: Learning the kilometer-scale medium-range global
weather forecasting. _arXiv preprint arXiv:2402.00059_, 2024.


Suman Ravuri, Karel Lenc, Matthew Willson, Dmitry Kangin, Remi Lam, Piotr Mirowski, Megan
Fitzsimons, Maria Athanassiadou, Sheleem Kashem, Sam Madge, et al. Skilful precipitation
nowcasting using deep generative models of radar. _Nature_, 597(7878):672–677, 2021.


Thorsten Kurth, Shashank Subramanian, Peter Harrington, Jaideep Pathak, Morteza Mardani, David
Hall, Andrea Miele, Karthik Kashinath, and Anima Anandkumar. Fourcastnet: Accelerating global
high-resolution weather forecasting using adaptive fourier neural operators. In _Proceedings of the_
_platform for advanced scientific computing conference_, pages 1–11, 2023.


Adeline Fermanian, Pierre Marion, Jean-Philippe Vert, and Gérard Biau. Framing rnn as a kernel
method: A neural ode approach. _Advances in Neural Information Processing Systems_, 34:3121–
3134, 2021.


Shengze Cai, Zhiping Mao, Zhicheng Wang, Minglang Yin, and George Em Karniadakis. Physicsinformed neural networks (pinns) for fluid mechanics: A review. _Acta Mechanica Sinica_, 37(12):
1727–1738, 2021.


Steven L Brunton and J Nathan Kutz. Promising directions of machine learning for partial differential
equations. _Nature Computational Science_, 4(7):483–494, 2024.


Yunchang Seol, Suho Kim, Minwoo Jung, and Youngjoon Hong. A novel physics-aware graph
network using high-order numerical methods in weather forecasting model. _Knowledge-Based_
_Systems_, page 112158, 2024.


Junho Choi, Taehyun Yun, Namjung Kim, and Youngjoon Hong. Spectral operator learning for
parametric pdes without data reliance. _Computer Methods in Applied Mechanics and Engineering_,
420:116678, 2024.


Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. Neural ordinary
differential equations. _Advances in neural information processing systems_, 31, 2018.


Nathaniel Thomas, Tess Smidt, Steven Kearnes, Lusann Yang, Li Li, Kai Kohlhoff, and Patrick Riley.
Tensor field networks: Rotation- and translation-equivariant neural networks for 3d point clouds,
2018. [URL https://arxiv.org/abs/1802.08219.](https://arxiv.org/abs/1802.08219)


11


Maurice Weiler, Mario Geiger, Max Welling, Wouter Boomsma, and Taco Cohen. 3d steerable cnns:
Learning rotationally equivariant features in volumetric data, 2018. URL [https://arxiv.](https://arxiv.org/abs/1807.02547)
[org/abs/1807.02547.](https://arxiv.org/abs/1807.02547)


Risi Kondor, Zhen Lin, and Shubhendu Trivedi. Clebsch-gordan nets: a fully fourier space spherical
convolutional neural network, 2018. [URL https://arxiv.org/abs/1806.09231.](https://arxiv.org/abs/1806.09231)


Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von
Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors,
_Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, volume 30. Curran Associates, Inc.,
2017. URL [https://proceedings.neurips.cc/paper_files/paper/2017/](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
[file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf.](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)


Johannes Brandstetter, Daniel E. Worrall, and Max Welling. Message passing neural PDE solvers. In
_International Conference on Learning Representations_, 2022. [URL https://openreview.](https://openreview.net/forum?id=vSix3HPYKSU)
[net/forum?id=vSix3HPYKSU.](https://openreview.net/forum?id=vSix3HPYKSU)


Zichao Long, Yiping Lu, Xianzhong Ma, and Bin Dong. Pde-net: Learning pdes from data. In
_International conference on machine learning_, pages 3208–3216. PMLR, 2018.


Tung Nguyen, Johannes Brandstetter, Ashish Kapoor, Jayesh K Gupta, and Aditya Grover. Climax:
A foundation model for weather and climate. _arXiv preprint arXiv:2301.10343_, 2023.


12


A FURTHER EXPERIMENTS


In this section, we present additional experimental results for the TFNP and the PA-TFNP model.
Table 2 and Figure 5 report the RMSE values for the two-month prediction task described in
Section 4.3. Figure 5 visualizes the RMSE values of ClimODE, TFNP, and PA-TFNP over a twomonth forecast horizon, based on global monthly averaged predictions. The comparison spans five
atmospheric variables ( _z_, _t_, _t_ 2 _m_, _u_ 10, and _v_ 10). The results clearly show that PA-TFNP achieves the
lowest RMSE in most variables, especially in _z_ and _t_, where its advantage over other models is more
pronounced. This supports our main claim that incorporating physics-aware inductive bias improves
long-range prediction performance.


Figure 5: RMSE comparison of ClimODE, TFNP, and PA-TFNP for two-month averaged predictions
across five key atmospheric variables: geopotential height ( _z_ ), temperature ( _t_ ), ground temperature
( _t_ 2 _m_ ), and wind components ( _u_ 10, _v_ 10). PA-TFNP consistently achieves the lowest RMSE for
most variables, particularly in _z_ and _t_, demonstrating enhanced accuracy and temporal stability for
long-range climate forecasting.


Figure 6 shows the absolute prediction errors of PA-TFNP and ClimODE across five key atmospheric
variables, as discussed in Section 4.3.


Figure 6: Qualitative comparison of absolute prediction errors from ClimODE and TFNP across
five atmospheric variables ( _z_, _v_ 10, _u_ 10, _t_ 2 _m_, _t_ ). The first row visualizes the spatial distribution of
prediction errors from ClimODE, while the second row shows those from TFNP. TFNP significantly
reduces errors and improves spatial consistency, particularly in polar regions where ClimODE suffers
from grid distortion effects. These results highlight TFNP’s ability to handle rotationally sensitive
areas through its rotation-equivariant architecture, enabling more robust global-scale predictions.


Next, in Table 3, we compare the PA-TFNP model with other baseline models—Neural ODE, ClimaX,
and ClimODE—for the North America region. See Table 1 for results in other regions, including
Australia and South America.


Figure 7 and Figure 8 present qualitative visualizations of PA-TFNP’s prediction performance for
monthly-averaged and hourly forecasting tasks, respectively. Across both long-term (monthly) and


13


Table 3: Comparison of RMSE values for baseline models and the proposed PA-TFNP (Ours) across
North America. Results are reported as mean _±_ standard deviation.


Value Hours NODE ClimaX ClimODE **PA–TFNP (Ours)**


short-term (hourly) settings, PA-TFNP demonstrates low absolute errors across the entire spatial
domain, including boundary regions. The model consistently provides accurate predictions for key
atmospheric variables, particularly temperature ( _t_ ) and geopotential height ( _z_ ), underscoring its
effectiveness in spatiotemporal climate modeling.


B DATASETS


The ERA5 dataset consists of weather records for five variables: round temperature (t2m), atmospheric
temperature (t), geopotential height (z), and ground wind components (u10, v10). It provides global
coverage on a uniform grid from 2006 to 2018. Data from 2006 to 2015 are used for training, 2016
for validation, and 2017-2018 for testing. We exclude the first two and last one months of each
year, considering only nine months per year. These months are further divided into three sequential
groups, where, for each group, we predicted atmospheric variables for two consecutive months based
on observations from the preceding month. The spatial grid is uniformly spaced at 5 _._ 625 _[◦]_ in both
latitude and longitude, with dimensions _H_ = 32 and _W_ = 64.


C TRAINING DETAILS


We employed the forward Euler method as our ODE solver to integrate the dynamical system in
Equation (1) and its variation, using a time resolution of 1/6 month (approximately 5 days). In our
neural ODE framework, this resolution is represented as 0.01 in normalized time to avoid excessive
computational costs of directly using a one-month unit. Model training and inference are performed
on a single NVIDIA RTX 4090 (24GB). All training hyperparameters for ClimODE and ClimaX
remain consistent with those in (Nguyen et al., 2023; Verma et al., 2024). During training, all
variables are normalized to [0,1] using min-max scaling; however, the original values are restored to
compute RMSD in Table 2.


14


z


t


t2m


u10


v10


6 232.8 273.4 134 _._ 5 _±_ 10 _._ 6 _._ **2** _±_ 41 _._ 3
12 469.2 329.5 225 _._ 0 _±_ 17 _._ 3 _._ **3** _±_ 59 _._ 1
18 667.2 543.0 307 _._ 7 _±_ 25 _._ 4 _._ **7** _±_ 81 _._ 8
24 893.7 494.8 390 _._ 1 _±_ 32 _._ 3 _._ **3** _±_ 107 _._ 6


6 1.96 1.62 **1** _._ **28** _±_ 0 _._ 06 1 _._ 45 _±_ 0 _._ 27
12 3.34 1.86 1 _._ 81 _±_ 0 _._ 13 **1** _._ **79** _±_ 0 _._ 37
18 4.21 2.75 2 _._ 03 _±_ 0 _._ 16 **1** _._ **97** _±_ 0 _._ 43
24 5.39 2.27 **2** _._ **23** _±_ 0 _._ 18 2 _._ 32 _±_ 0 _._ 48


6 2.65 1.75 **1** _._ **61** _±_ 0 _._ 12 3 _._ 48 _±_ 1 _._ 74
12 3.43 1.87 **1** _._ **87** _±_ 0 _._ 13 4 _._ 68 _±_ 1 _._ 03
18 3.53 2.27 **1** _._ **96** _±_ 0 _._ 33 3 _._ 41 _±_ 1 _._ 05
24 3.39 1.93 **2** _._ **15** _±_ 0 _._ 20 2 _._ 59 _±_ 0 _._ 64


6 1.96 1.74 **1** _._ **54** _±_ 0 _._ 19 1 _._ 69 _±_ 0 _._ 34
12 2.91 2.24 2 _._ 01 _±_ 0 _._ 20 **1** _._ **94** _±_ 0 _._ 41
18 3.40 3.42 2 _._ 17 _±_ 0 _._ 34 **2** _._ **08** _±_ 0 _._ 43
24 3.96 3.42 2 _._ 34 _±_ 0 _._ 32 **2** _._ **26** _±_ 0 _._ 43


6 2.36 1.83 **1** _._ **67** _±_ 0 _._ 23 1 _._ 79 _±_ 0 _._ 36
12 3.42 2.43 2 _._ 03 _±_ 0 _._ 31 **1** _._ **94** _±_ 0 _._ 41
18 4.35 3.92 2 _._ 31 _±_ 0 _._ 37 **2** _._ **20** _±_ 0 _._ 40
24 4.57 3.39 2 _._ 50 _±_ 0 _._ 41 **2** _._ **37** _±_ 0 _._ 42


True


Absolute Error (ClimODE)


Absolute Error (TFNP-PA)


Figure 7: Comparison of RMSE values for ClimODE and TFNP-PA in two-month averaged predictions for five atmospheric variables. TFNP consistently achieves lower RMSE, particularly for
temperature (t) and geopotential height (z), highlighting its improved forecasting performance.


C.1 LOSS FUNCTION


Consider the set of observations _{{qi_ _[obs]_ ( **x** _n, t_ ) _}_ _[d]_ _i_ =1 _[}]_ _n_ _[N]_ =1 [.] [To introduce stochasticity, we assume the]
following equation with the estimated mean and variance as in (Verma et al., 2024).:


_qi_ [obs][(] **[x]** _[, t]_ [)] _[ ∼N]_               - _qi_ ( **x** _, t_ ) + _µi_ ( **x** _, t_ ) _, σi_ [2][(] **[x]** _[, t]_ [)]               - _,_


where the function _ϵ_ - _qi_ ( **x** _, t_ ) _, g_ ( _{xn}_ _[N]_ _n_ =1 _[, t]_ [)] - estimates the additional mean _µi_ ( **x** _, t_ ) and variance
_σi_ [2][(] **[x]** _[, t]_ [)][.] [Given the observations] _[ {][q]_ _i_ _[obs]_ ( **x** _n, t_ ) _}_, we define the loss function _L_ ( _η_ ) using the negative
log-likelihood:


log _L_ ( _{{qi_ _[obs]_ ( **x** _n, t_ ) _}_ _[d]_ _i_ =1 _[}][N]_ _n_ =1 _[|{{][µ][i]_ [(] **[x]** _[n][, t]_ [)] _[}][d]_ _i_ =1 _[}][N]_ _n_ =1 _[,][ {{][σ][i]_ [(] **[x]** _[n][, t]_ [)] _[}][d]_ _i_ =1 _[}][N]_ _n_ =1 [)]


_N_

- _L_ ( _qi_ _[obs]_ ( **x** _n, t_ ) _|µi_ ( **x** _n, t_ ) _, σi_ ( **x** _n, t_ ))

_n_ =1


= log


_d_


_i_ =1


_N_

- log( _L_ ( _qi_ _[obs]_ ( **x** _n, t_ ) _|µi_ ( **x** _n, t_ ) _, σi_ ( **x** _n, t_ )))

_n_ =1


=


=


_d_


_i_ =1


_d_


_i_ =1


_N_


_n_ =1


- ( _qiobs_ ( **x** _n, t_ ) _−_ _µi_ ( **x** _n, t_ )) [2] + log - _√_ 2 _πσi_ ( **x** _n, t_ )� [�] _._

2( _σi_ ( **x** _n, t_ )) [2]


15


Table 4: Comparison of RMSE values for ClimODE and the proposed PA-TFNP (Ours) across two
spatiotemporal resolutions. The results highlight the performance differences for key atmospheric
variables. Results are reported as mean _±_ standard deviation.


where _α_ ( _h_ ) = cos( _h_ ) _/_ _H_ [1] - _Hh_ _[′]_ [ cos(] _[h][′]_ [)][ represents the latitude-dependent weighting factor.]


16


**(a)** Long-term prediction at a resolution of 5 _._ 625 _[◦]_ .


Value Day **ClimODE** **TFNP (Ours)**


**(b)** Short-term prediction at a resolution of 11 _._ 25 _[◦]_ .


Hour **ClimODE** **TFNP (Ours)**


6 3115 _._ 1 _±_ 216 _._ 6 _._ **2** _±_ 17 _._ 4
12 3156 _._ 4 _±_ 204 _._ 0 _._ **5** _±_ 59 _._ 6
18 3175 _._ 8 _±_ 176 _._ 3 _._ **3** _±_ 77 _._ 1
24 3202 _._ 5 _±_ 176 _._ 5 _._ **5** _±_ 87 _._ 8
30 3240 _._ 1 _±_ 178 _._ 9 _._ **8** _±_ 90 _._ 2
36 3216 _._ 3 _±_ 181 _._ 3 **1038** _._ **1** _±_ 90 _._ 7
42 3282 _._ 4 _±_ 224 _._ 7 **1067** _._ **6** _±_ 106 _._ 3


6 22 _._ 62 _±_ 1 _._ 41 **1** _._ **27** _±_ 0 _._ 28
12 22 _._ 73 _±_ 1 _._ 16 **3** _._ **22** _±_ 0 _._ 21
18 22 _._ 59 _±_ 1 _._ 07 **3** _._ **99** _±_ 0 _._ 28
24 22 _._ 49 _±_ 1 _._ 17 **4** _._ **29** _±_ 0 _._ 30
30 22 _._ 50 _±_ 1 _._ 09 **4** _._ **49** _±_ 0 _._ 31
36 22 _._ 51 _±_ 1 _._ 13 **4** _._ **61** _±_ 0 _._ 35
42 22 _._ 86 _±_ 1 _._ 15 **4** _._ **71** _±_ 0 _._ 37


6 38 _._ 58 _±_ 4 _._ 77 **0** _._ **92** _±_ 0 _._ 17
12 39 _._ 03 _±_ 4 _._ 70 **2** _._ **27** _±_ 0 _._ 18
18 38 _._ 48 _±_ 4 _._ 12 **2** _._ **80** _±_ 0 _._ 26
24 38 _._ 19 _±_ 4 _._ 50 **3** _._ **04** _±_ 0 _._ 28
30 39 _._ 19 _±_ 3 _._ 85 **3** _._ **20** _±_ 0 _._ 31
36 39 _._ 21 _±_ 4 _._ 21 **3** _._ **33** _±_ 0 _._ 34
42 37 _._ 87 _±_ 4 _._ 35 **3** _._ **42** _±_ 0 _._ 35


6 14 _._ 86 _±_ 0 _._ 92 **0** _._ **62** _±_ 0 _._ 04
12 15 _._ 44 _±_ 0 _._ 92 **3** _._ **98** _±_ 0 _._ 26
18 15 _._ 57 _±_ 0 _._ 92 **4** _._ **69** _±_ 0 _._ 27
24 15 _._ 67 _±_ 0 _._ 83 **4** _._ **97** _±_ 0 _._ 30
30 15 _._ 70 _±_ 0 _._ 88 **5** _._ **12** _±_ 0 _._ 35
36 15 _._ 86 _±_ 0 _._ 82 **5** _._ **23** _±_ 0 _._ 35
42 15 _._ 89 _±_ 0 _._ 87 **5** _._ **32** _±_ 0 _._ 35


6 13 _._ 86 _±_ 0 _._ 92 **0** _._ **59** _±_ 0 _._ 05
12 14 _._ 46 _±_ 0 _._ 92 **4** _._ **50** _±_ 0 _._ 36
18 14 _._ 74 _±_ 0 _._ 82 **5** _._ **17** _±_ 0 _._ 37
24 14 _._ 75 _±_ 0 _._ 74 **5** _._ **32** _±_ 0 _._ 37
30 14 _._ 78 _±_ 0 _._ 84 **5** _._ **36** _±_ 0 _._ 36
36 14 _._ 73 _±_ 0 _._ 74 **5** _._ **39** _±_ 0 _._ 33
42 14 _._ 69 _±_ 0 _._ 92 **5** _._ **43** _±_ 0 _._ 36


z


t


t2m


u10


v10


5 1104 _._ 0 _±_ 104 _._ 0 _._ **4** _±_ 21 _._ 6
10 1445 _._ 7 _±_ 103 _._ 1 _._ **3** _±_ 91 _._ 1
15 1430 _._ 2 _±_ 118 _._ 6 **1033** _._ **3** _±_ 87 _._ 5
20 1470 _._ 3 _±_ 129 _._ 0 **1058** _._ **2** _±_ 70 _._ 9
25 1445 _._ 6 _±_ 83 _._ 6 **1018** _._ **0** _±_ 81 _._ 1
30 1449 _._ 0 _±_ 108 _._ 5 **1014** _._ **4** _±_ 92 _._ 8
35 1457 _._ 8 _±_ 113 _._ 1 **1078** _._ **0** _±_ 68 _._ 6


5 6 _._ 12 _±_ 0 _._ 55 **1** _._ **15** _±_ 0 _._ 13
10 7 _._ 58 _±_ 0 _._ 45 **4** _._ **34** _±_ 0 _._ 47
15 7 _._ 73 _±_ 0 _._ 86 **4** _._ **49** _±_ 0 _._ 33
20 7 _._ 46 _±_ 0 _._ 56 **4** _._ **54** _±_ 0 _._ 30
25 7 _._ 67 _±_ 0 _._ 58 **4** _._ **64** _±_ 0 _._ 46
30 8 _._ 07 _±_ 0 _._ 83 **4** _._ **70** _±_ 0 _._ 42
35 8 _._ 16 _±_ 0 _._ 51 **4** _._ **98** _±_ 0 _._ 42


5 7 _._ 92 _±_ 0 _._ 71 **1** _._ **16** _±_ 0 _._ 13
10 8 _._ 78 _±_ 0 _._ 82 **3** _._ **09** _±_ 0 _._ 37
15 8 _._ 58 _±_ 0 _._ 80 **3** _._ **43** _±_ 0 _._ 38
20 8 _._ 76 _±_ 0 _._ 87 **3** _._ **59** _±_ 0 _._ 33
25 8 _._ 90 _±_ 0 _._ 95 **3** _._ **81** _±_ 0 _._ 45
30 9 _._ 09 _±_ 0 _._ 70 **4** _._ **04** _±_ 0 _._ 48
45 9 _._ 51 _±_ 1 _._ 23 **4** _._ **30** _±_ 0 _._ 62


5 3 _._ 99 _±_ 0 _._ 30 **0** _._ **96** _±_ 0 _._ 08
10 6 _._ 29 _±_ 0 _._ 32 **4** _._ **86** _±_ 0 _._ 25
15 6 _._ 27 _±_ 0 _._ 29 **5** _._ **17** _±_ 0 _._ 24
20 6 _._ 09 _±_ 0 _._ 20 **5** _._ **18** _±_ 0 _._ 22
25 6 _._ 08 _±_ 0 _._ 31 **5** _._ **12** _±_ 0 _._ 21
30 6 _._ 09 _±_ 0 _._ 30 **5** _._ **11** _±_ 0 _._ 26
35 6 _._ 22 _±_ 0 _._ 27 **5** _._ **21** _±_ 0 _._ 21


5 3 _._ 94 _±_ 0 _._ 44 **1** _._ **50** _±_ 0 _._ 08
10 6 _._ 41 _±_ 0 _._ 29 **4** _._ **83** _±_ 0 _._ 28
15 6 _._ 19 _±_ 0 _._ 37 **4** _._ **98** _±_ 0 _._ 38
20 6 _._ 18 _±_ 0 _._ 37 **5** _._ **00** _±_ 0 _._ 23
25 5 _._ 93 _±_ 0 _._ 29 **4** _._ **94** _±_ 0 _._ 30
30 6 _._ 29 _±_ 0 _._ 39 **4** _._ **88** _±_ 0 _._ 28
35 6 _._ 21 _±_ 0 _._ 31 **4** _._ **97** _±_ 0 _._ 28


To enhance numerical stability, we incorporate a small constant 10 _[−]_ [3] into the variance term and
introduce a regularization term weighted by _λ_ :


_d_


_i_ =1


_N_


_n_ =1


_N_
�( _σi_ [2][(] **[x]** _[n][, t]_ [))]

_n_ =1


- ( _qiobs_ ( **x** _n, t_ ) _−_ _µi_ ( **x** _n, t_ )) [2] + log( _σi_ - **x** _n, t_ ) + 10 _[−]_ [3][��] + _λ_

2( _σi_ ( **x** _n, t_ )) [2] + 10 _[−]_ [3]


_d_


_i_ =1


C.2 LATITUDE-WEIGHTED RMSE METRIC


To quantify prediction accuracy, we employ the latitude-weighted RMSE metric, defined as:


_H_


_h_


~~�~~
�� 1

_HW_


_W_

- _α_ ( _h_ )( _ythw −_ _uthw_ ) [2]


_w_


RMSE = [1]

_T_


_T_


_t_


Table 5: Training time for 1 epoch and the number of parameters.


Category Model Time [s] #Params


**All-data** ClimaX 115M


**Regional**
North ClimODE 305 _._ 00 2.75M
TFNP (Ours) 289 _._ 87 2.78M
South ClimODE 309 _._ 86 2.75M
TFNP (Ours) 295 _._ 46 2.78M
Australia ClimODE 310 _._ 08 2.75M
TFNP (Ours) 292 _._ 43 2.78M


**Global** ClimODE 23.69 / 55.60 2.75M
Long Term / High Resolution PA-TFNP (Ours) 11.27 / 31.39 0.196M


**Monthly** ClimODE 6 _._ 50 2.40M
TFNP (Ours) 2 _._ 87 0.098M
PA-TFNP (Ours) 3 _._ 30 0.194M


**Ablation** TFNP (Ours) 3 _._ 17 0.130M
PA-TFNP (Ours) 4 _._ 38 0.196M


C.3 COMPUTATIONAL COST


Table 5 reports the training time per epoch and the number of trainable parameters for various
models and experimental settings. Our proposed models (TFNP and PA-TFNP) demonstrate notable
efficiency in both training speed and model size compared to baseline models like ClimODE and
ClimaX. In regional settings, our models consistently show faster training times despite having
a comparable number of parameters. For global high-resolution forecasts, PA-TFNP achieves
significantly reduced training time, with the number of parameters less than 10% of ClimODE’s,
highlighting its scalability. Furthermore, under the monthly and ablation settings, our lightweight
PA-TFNP remains both computationally efficient and parameter-efficient, making it suitable for
practical deployment in climate modeling tasks.


D ADDITIONAL EXPLANATION ON PHYSICS-AWARE VARIANTS


This section details the physics-informed variant model presented in Section 3.3.


RATIONALE FOR ADDING PHYSICAL TERMS


Atmospheric phenomena, such as turbulence, can distribute energy across scales—an effect we
attempt to approximate with the Laplacian ∆ term. Even when the governing equations do not
explicitly include diffusion, some level of numerical diffusion is generally needed to prevent the
accumulation of artificial energy in simulations; see (Warner, 2010, Section 3.4.7). To address these
practical considerations, we include a term _α_ ∆( _·_ ) in the original transport equation equation 1, where
the non-negative coefficient _α_ is a learnable parameter. Setting _α_ = 0 preserves the original equation
equation 1, while a small _α_ _>_ 0 allows the model to represent physically motivated diffusion in a
controlled manner.


Similarly, the physics-informed forcing term _f_ phys( **x** _, t,_ **u** _i_ ) explicitly aims to incorporate essential
physical processes into the neural equation. We define


_f_ phys( **x** _, t,_ **u** _i_ ) = _−∇_ Φ + _ν_ ∆ **u** _i −_ _γ_ **u** _i,_


where Φ represents the geopotential field (with Φ = _z_ ), and _ν_, _γ_ are learnable coefficients associated
with viscosity and linear drag, respectively. Including these physical terms is intended to enhance
the consistency of the model with fundamental atmospheric physics. Our experiment shows that
our approach improves interpretability, prediction accuracy, and numerical stability, particularly in
longer-range atmospheric simulations.


17


Thus, our hybrid formulation of PA-TFNP aims to retain the expressive capability of neural network models while encouraging adherence to important physical constraints, potentially leading to
improved predictive performance for long-term forecasts.


MOTIVATION OF ADDITIONAL PHYSICS-DERIVED FEATURES


We introduce three physics-derived features and recall the governing relations in which each enters:


    - **Wind magnitude.** _|_ _**V**_ 10 _|_ = ~~�~~ _u_ [2] 10 [+] _[ v]_ 10 [2] [appears in the bulk aerodynamic surface stress]
formula
_**τ**_ = _ρ CD |_ _**V**_ 10 _|_ _**V**_ 10 _,_

where _**τ**_ is asurface stress vector, _ρ_ is air density and _CD_ a drag coefficient.

    - **Lapse rate.** ∆ _t_ = _t −_ _t_ 2 _m_ appears in various governing equations, particularly in parameterizations of turbulent mixing processes in the atmospheric boundary layer. Specifically, it
is utilized to estimate buoyancy-driven turbulence production or suppression, represented
mathematically by terms such as:


_B_ _∝−_ _[g]_ _KH_ ∆ _t,_

_θ_ 0

where _B_ denotes the buoyancy contribution to the turbulent kinetic energy budget, _g_ is
gravitational acceleration, _θ_ 0 represents a reference potential temperature, and _KH_ is the
eddy diffusivity for heat. Therefore, the lapse rate ∆ _t_ can be considered a valuable physical
feature when developing approximate neural network-based flow models, as it directly
encapsulates critical information about atmospheric stability and turbulence dynamics.

    - **Relative vorticity.** _ζ_ = _∂yv_ 10 _−_ _∂xu_ 10 quantifies the local horizontal spin of the wind and
features explicitly in the barotropic vorticity equation, quasi-geostrophic potential vorticity,
and Ertel potential vorticity. Its direct link to these conservation laws makes _ζ_ a clear,
physically interpretable variable for weather-prediction modeling.


SPATIAL DERIVATIVE


The spatial derivative approximation used in this study is based on the gradient operator in spherical
coordinates. Let _F_ ( _R, ϕ, λ_ ) be a scalar field defined on the surface of a sphere, where _ϕ_ denotes
latitude (in radians), _λ_ denotes longitude (in radians), and _R_ represents the Earth’s radius, which
is assumed to be constant throughout the domain. Although the notation S [2] typically refers to the
unit sphere, here we consider the spherical surface of radius _R_, denoted S [2] _R_ [=] _[ {]_ **[x]** _[ ∈]_ [R][3] [:] _[ ∥]_ **[x]** _[∥]_ [=] _[ R][}]_ [.]
Assuming no variation in the radial direction, the surface gradient takes the following form:


Here, ˆ _eϕ_ is the unit vector in the direction of increasing latitude (northward), and ˆ _eλ_ is the unit vector
in the direction of increasing longitude (eastward). Both lie in the tangent plane of the sphere at each
point.


Based on this formulation, the gradient is numerically approximated using a second-order central
finite difference scheme, yielding Equation equation 3, where the factor _π/_ 180 converts angular
increments from degrees to radians. The spherical derivative approximation is applied consistently
throughout the PA-TFNP model.


E BROADER IMPACT


The proposed PA-TFNP improves global and regional weather prediction by combining physical
interpretability with high forecasting accuracy and reduced computational cost. This enables faster,
more accessible predictions, especially valuable in regions with limited computing resources. The
model’s efficiency supports broader deployment of weather forecasting systems, contributing to
better preparedness for climate change. Careful integration with traditional methods and responsible
communication are essential for safe and effective use.


18


_∇_ S2 _R_ _[F]_ [(] _[ϕ, λ]_ [) =] [1]


[1] _∂F_

_R_ _∂ϕ_


_∂F_ 1 _∂F_

_∂ϕ_ _[e]_ [ˆ] _[ϕ]_ [ +] _R_ cos _ϕ_ _∂λ_


_∂λ_ _[e]_ [ˆ] _[λ][.]_


**1000**

**1001**

**1002**


**1003**

**1004**

**1005**

**1006**

**1007**

**1008**


**1009**

**1010**

**1011**

**1012**

**1013**


**1014**

**1015**

**1016**

**1017**

**1018**

**1019**


**1020**

**1021**

**1022**

**1023**

**1024**

**1025**


**True (6 hr)**


0 360 0 360 0 360 0 360 0 360


Figure 8: Comparison of RMSE values for TFNP-PA in long-term predictions over hourly predictions
for five atmospheric variables. TFNP consistently predicts over the entire domain, particularly for
temperature (t) and geopotential height (z), highlighting its improved forecasting performance.


19


0.8


0.9


0.3


0.9


0.3


0.2


0.0


0.7


0.7


0.3


0.7


0.3


0.2


0.0


0.3


0.7


0.2


0.8


Prediction (TFNP-PA)


0.7


0.3


0.7


0.3


0.2


0.0


0.3


0.2


0.0


Absolute Error


0.2


0.2


0.0


0 360 0 360 0 360 0 360 0 360


90


-90


**True (18 hr)**


90


-90


90


-90


90


-90


90


-90


Prediction (TFNP-PA)


90


90


-90


90


-90


90


-90


90


-90


90


-90


Absolute Error


-90


90


-90


0 360 0 360 0 360 0 360 0 360


90


-90


**True (30 hr)**


90


-90


90


-90


90


-90


90


-90


Prediction (TFNP-PA)


90


90


-90


90


-90


90


-90


90


-90


90


-90


Absolute Error


-90


90


-90


0 360 0 360 0 360 0 360 0 360

**True (42 hr)**


90


-90


90


-90


90


-90


90


-90


90


-90


Prediction (TFNP-PA)

90 90


-90


90


-90


90


-90


90


-90


90


-90


Absolute Error


-90


90


-90