000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Latent neural dynamics are a widely used model in neuroscience for describing the time evolution of collective neural activity. These models have been established as useful for neural decoding: for example, latent dynamical models of neural activity give state-of-the art predictions of ongoing kinematics in motor tasks. Despite their utility, the causal mechanisms behind the effectiveness of latent variable models remain poorly understood. To uncover how such latent variables causally encode behaviors, or how they change, would require methods for stimulating neural dynamics during an experiment. Algorithms to *drive* neural dynamics remain limited, however, due to the need to continually track and respond to changes in neural activity, to account for variation in neural responses under stimulation, and to select useful stimulations to apply from an extensive set of possibilities. Here, we develop a novel streaming method for stimulation-response modeling in affine latent spaces and an optimization framework for selecting high-dimensional stimulation patterns to drive low-dimensional dynamics. Our method integrates streaming latent space construction, an adaptive nonparametric model of the effects of stimulations, and projection maximization under feasibility constraints to determine stimuli that move dynamics along a desired vector. We demonstrate our approach on both simulated and real neural data (calcium fluorescence images, intracortial electrophysiological recordings). We evaluate our method across multiple latent space representations and multiple models of dynamics in parallel, and additionally provide a novel streaming estimator to determine which representation is most predictive of ongoing neural dynamics at any timepoint. This allows for direct comparison between different latent representations and the opportunity for adaptive selection of stimulations to best distinguish amongst neural subspace hypotheses. Finally, we demonstrate algorithm runtimes at faster than real-time speeds (<100 ms), making it compatible with future *in vivo* applications.

## 1 Introduction

Models of neural activity in low-dimensional spaces, often called 'neural manifolds', are increasingly state-of-the-art for describing the structure of the neurological activity that gives rise to ongoing behavior (Saxena & Cunningham, 2019; Vyas et al., 2020). Such neural population models have been very successful across areas in neuroscience, from determining latent task variables in decisionmaking (Peixoto et al., 2021) to decoding latent neural activity for predicting desired movements in brain-machine interfaces (Pandarinath et al., 2018). Additional developments in targeted stimulation technology have opened the door to causally testing underlying manifold hypotheses by manipulating the activity of individual and sets of neurons (Grosenick et al., 2015; Rajasethupathy et al., 2016; Jazayeri & Afraz, 2017; Tafazoli et al., 2020; Dal Maschio et al., 2017; Vinograd et al., 2024). For example, neuroscientists could test whether a pattern of neural sates forms a ring attractor via stimulating along or off the manifold in a targeted way. (Kim et al., 2017). Such higher-resolution stimulation technology is also being developed for clinical applications, where driving activity in a brain circuit has therapeutic benefits (Yang et al., 2021; Shah et al., 2024). As the number of possible stimulation targets or parameters grows, however, it becomes more challenging to determine ideal or even useful stimulation patterns. Selecting even just 30 neurons to stimulate from a population of 400 involves searching a space of over 1045 combinations, without considering stimulus magnitudes.

Designing stimulations to manipulate latent neural dynamics additionally requires considering the

# Adaptive Stimulation & Response Modeling Of Latent Neural Dynamics

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 time-dynamics of the system: a stimulation applied early in a trial and the same stimulation applied late in a trial could have vastly different effects due to an evolution in the underlying neural state. We therefore need a method for tracking activity in latent spaces, modeling the response to potentially high-dimensional stimulations at different locations on the manifold, and finally designing a stimulation customized to in-the-moment neural dynamics. Prior work has addressed specific elements of the problem of tracking neural dynamics and designing neural stimulations (Peixoto et al., 2021; Minai et al., 2024; SoldadoMagraner et al., 2025; Wagenmaker et al., 2024; O'Shea et al., 2022; Draelos & Pearson, 2020; Draelos et al., 2021). Designing stimulations from a high-dimensional set of possibilities is a significant challenge, and has been partially addressed using methods like input-output dynamical modeling (Yang et al., 2021) or Bayesian optimization (Minai et al., 2024). In many cases, spatial correlations, as in a 2D array for electrical microstimulation, can serve to reduce the complexity of the problem. In contrast, here we specifically target the situation where many neurons are at least somewhat individually addressable, as in the case of holographic optogenetic photostimulation (Adesnik & Abdeladim, 2021; Pegard ´ et al., 2017; Triplett et al., 2023). Actively learning from the results of stimulations can also be used to design better or more custom future stimulations, as demonstrated with techniques like active learning (Wagenmaker et al., 2024), or Bayesian variational inference (Draelos & Pearson, 2020). Our core contribution is a novel real-time method for designing neural stimulations that perturb latent dynamics in arbitrary directions. We propose a new model for learning a map between stimulations and their effects on latent neural dynamics. Using kernel regression, we nonparametrically regress changes in dynamics based on both the delivered stimulation and the neural latent state (location on the manifold) at the time of stimulation. We do not assume that responses to stimulations are robust, involve the neurons that the stimulation intended to target, or are static over time. We consider multiple possible models of these latent neural dynamics (Draelos et al., 2021; Churchland et al., 2012; Ablin et al., 2019), additionally develop a new method for streaming dimensionality reduction, and consider multiple possible manifold representations in parallel due to the streaming nature of our algorithm. Finally, we present a novel optimization problem to design high-dimensional stimulations that are aligned to specified desired movements in the low-dimensional space. The problem is constrained by the number of neurons or channels to target and by the non-negative magnitude of total stimulation applied, to simulate realistic experimental conditions. In this step, we leverage the differentiability of our stimulus-response mapping to design stimuli that can adapt to the idiosyncrasies of any individual experiment. We test using simulated and real neural data across two types of modalities: faster datarates with intracortical electrophysiological recordings and slower datarates with calcium fluorescence activity traces. We design and test multiple kinds of relevant stimulations in the latent subspace, with various constraints on the dimensionality of the resultant stimulation vector. The constraints accommodate realistic experimental conditions, where neurons can be individually addressed yet the number of simultaneous targets and/or the total amount of power is limited (Fernandez-Ruiz et al., 2022; Telliez et al., 2025). Our stimulation targets include the direction of highest neural variance (the first principal component), random feasible directions, and arbitrary (possibly partially infeasible) directions in the latent space. Our algorithms were able to quickly learn a stimulation-response mapping within roughly 10-20 total stimulations delivered, and kept end-to-end runtimes at less than 10 ms on average (and below 100 ms) to ensure real-time feasibility. We anticipate that our adaptive method will enable the next generation of experiments capable of designing and testing stimulations of latent neural dynamics in real time, for both basic neuroscience and brain-machine interface applications.

## 2 Methods

We give an overall procedure for the use of our framework in Algorithm 1. As neural data is acquired, it is dimension reduced to a latent space. A dynamical model is used to track ongoing latent neural dynamics within that low dimensional space. If a stimulation is delivered, we update a response mapping of its effect on the neural dynamics. If a new stimulation is desired, we solve an optimization problem to determine the closest feasible stimulation in the neural data space to result in a selected perturbation direction in the latent space. All components are updated at each time point in a steaming manner.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 1: **Given:** Neural data stream {yt}, latent space mapping Q, dynamical model f, stimulusresponse model Sˆ, response delay d.

2: **Returns:** Optimized stimulus u
∗at decision timepoints 3: **Initialize:** Set t ← 0, Sˆ ← ∅, stimulus history *H ← ∅*
4:
5: for t = 1 .... do 6: Observe new neural data yt ∈ R
N
7: Update latent projections: xt ← Q.update(yt) ▷ Observe and project to latent space 8: xˆt+1 ← f(xt) ▷ Predict next latent state 9: if stimulation delivered at time t − d (i.e., (t − d, ut−d) ∈ H) **then**
10: sobs ← xt − xˆt ▷ Compute observed response 11: Sˆ ← S. ˆ add(xt−d, ut−d, sobs, t) ▷ Update kernel regression 12: **else**
13: f ← f.update(xt, xt−1) ▷ Update dynamics model 14: **end if**
15: if new stimulation desired at time t **then**
16: **Given:** target direction v ∈ R
k 17: L(u) = −v
⊤s(u)
∥v∥∥s(u)∥ + λ1(∥u∥
max 0 − ∥u∥1) ▷ Optimization problem in (8)
18: where s(u) = Sˆ(xt*, u, t*) ▷ Predicted response via learned mapping 19: u
∗ ← argminu∈[0,1]N L(u) ▷ Solve with box constraints 20: Deliver stimulation u
∗to neural system 21: Add (*t, u*∗) to H ▷ Track pending stimulation 22: **end if** 23: **end for**

## 2.1 Streaming Construction Of Latent Spaces

Designing and adapting to stimulations in a dynamic latent space first requires that such lowdimensional representations be available in real time. There are multiple hypotheses for which kind of representation might best describe the underlying computation implemented by the brain; for example, highest-variance latent dimensions (Draelos et al., 2021), latents with rotational structure (Churchland et al., 2012), or maximally statistically independent latents (Ablin et al., 2019). Here, we propose a novel streaming latent space construction method, use it alongside two existing methods, and demonstrate that all algorithms are stable approximations of their offline counterparts. Novel streaming method. jPCA (Churchland et al., 2012) is a widely used subspace identification method that identifies planes (pairs of dimensions) with high rotational structure. It achieves this by solving for the best skew-symmetric linear dynamical system that describes the data, based on a comparison of the low-dimensional neural state X with its time derivative X˙:

$$M_{t}=\operatorname*{argmin}_{M}\left\|{\dot{X}}_{t}-X_{t-1}M\right\|_{2}^{2}\quad\quad{\mathrm{s.t.}}\ M=-M^{\top}$$
s.t. M = −M⊤ (1)
A dimensionality-reduction step is required to first transform the data into a latent space; (Churchland et al., 2012) used PCA and here we use proSVD (Draelos et al., 2021) as it provides real-time estimates. We then implemented a solution to equation (1) using the Sherman-Morrison formula.

jPCA makes a basis out of Mt's eigenvectors: UtΣtU
⊤
t = Mt. To stabilize the subspace, we added a new Orthogonal Procrustes step to stabilize each discovered plane of rotation independently:
Ut,i = (Ut)[2i : 2i+1] (2)

$$U_{t,i}=(U_{t})_{[2i:2i+1]}$$
$$\Omega_{t,i}=\operatorname*{argmin}_{\Omega^{\top}\Omega=I}\left\|(U_{t,i})\Omega-\tilde{U}_{t-1,i}\right\|$$
$$\forall i,\;\tilde{U}_{t,i}=(U_{t,i})\Omega_{t,i}$$

Algorithm 1 Real-time Adaptive Stimulation Framework Our novel streaming formulation, named sjPCA, allows us to iteratively estimate a jPCA space in real time that quickly identifies the same space as a later offline calculation.

$$(1)$$
$$(2)$$

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Comparison with existing

![3_image_0.png](3_image_0.png) methods. We compared the above method with two existing methods: proSVD and mmICA. proSVD is a fast, stable, online dimensionality reduction method. It uses an iterative factorization Y ≈ QRW⊤
of the high-dimensional data Y to learn a set of lowdimensional subspace vectors whose columns form Q, and an Orthogonal Procrustes minimization of the change in bases across time. proSVD seeks to track the highest variance k-subspace over time; when its inputs are centered, this corresponds to the space containing the top k principal components. The two dimensionality reduction methods discussed so far have focused on high variance as a proxy for importance, but other statistical features such as independence may construct better latent spaces. To compare against a non-variance-prioritizing method, we adapted iterative algorithm for independent component analysis (ICA) using a minimization-maximization framework, termed mmICA, that seeks to model input data as linear mixtures of independent components (Ablin et al., 2019). mmICA assumes that the neural data is a linear mixture of independent sources, and uses a maximum likelihood majorization-minimization algorithm to infer the mixture of components that recovers the initial independent sources. Here we apply a proSVD reduction to an initial latent space before using mmICA to learn independent latent dimensions. All methods converge to offline fits. Figure 1a demonstrates convergence to an offline fit. We use a simulated circular linear dynamical system embedded in a latent space for sjPCA and proSVD. mmICA is given a 6D system generated with Laplace random variables where the dimensions are jointly independent, to match the algorithm's assumptions of super-gaussian independent components. Error is measured appropriately for each unique space. For proSVD, we calculate the sum of absolute principle angles between Q:4 and the true plane of highest variance. For sjPCA, we similarly compute the sum of the absolute values of the principle angles between the true plane of highest rotation and the identified plane of highest rotation. For mmICA, error is calculated as the Frobenius norm of the difference between the found demixing matrix (normalized with respect to scaling and permutation, see (Ablin et al., 2019)) and the true demixing matrix. Each latent space could be used as a stand-alone space, or considered in parallel to adaptively determine the most useful (predictive) representation at a given timepoint or after certain stimulations are observed.

Figure 1: **Real-time manifold construction and dynamical** modeling. a. Each streaming dimensionality reduction method (colored lines) converges to a similar representation as one computed offline (black lines). Shaded regions are 1 standard deviation of the errors (N = 10 runs). b. Projecting real neural data (O'Doherty, 2024) into different latent spaces reveals distinct large-scale dynamical patterns. Quiver plots are the averaged dynamics, with the same 12.5s period of data shown in black. Arrows indicate average direction of flow. c. Running the algorithms in parallel allows us to adaptively switch between spaces based on current performance. Heatmaps are estimates of where that space is most likely to give the best predictive probability (modeled using Bubblewrap). Color denotes empirical frequency of being the best predictor across all data.

## 2.2 Dynamical Modeling Of Neural Latents

We utilize three existing methods for streaming prediction of latent neural dynamics: a simple Kalman filter (KF) (Kalman, 1960), a method based on variational joint filtering (VJF) (Zhao & Park, 2020), and a non-parametric method Bubblewrap (Draelos et al., 2021) that captures probability flow using a joint Gaussian mixture model-hidden Markov model. All models are well suited for 216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 modeling a linear dynamical system, with VJF and Bubblewrap preferred for higher noise regimes or less consistent and multimodal dynamics. With any of the above dynamical models and latent spaces determined in real time, we can iteratively estimate a flow field that represents the underlying neural manifold discovered by a construction method (Fig. 1b). This gives us the opportunity to compare across latent spaces in parallel and evaluate if there are local regions where the predicted flow field best aligns with newly observed neural data (either spontaneous or evoked via stimulation). All dynamical models in the previous section are evaluated for error in their predictions at every timepoint, allowing us to select from among latent spaces and dynamical models the best performing system at any time. To evaluate the predictive utility of the latent spaces we consider here, we determine the predictive error at each timepoint and aggregate this information within a local region of the latent space (Fig. 1c). Our algorithm thus finds times and locations where each of the spaces yields the best predictions. Such a method could be used, for example, to identify when an animal switches between subtasks with distinct manifold structures (Perkins et al., 2024).

## 2.3 Mapping Desired Responses To Stimuli

To use stimulation to interrogate neural latents, we need to first characterize how the stimulations affect the latent dynamics. But the mapping between stimuli and neural responses could be nontrivial. There is evidence that responses are driven by network structure and the state of the neural system, and to effectively design stimuli in a real-time setting we need to determine the specific system responses under a wide variety of possible conditions (O'Shea et al., 2022). We do not assume that the response to stimulation is robust nor faithful to the intended stimulation: a neuron may lack sufficient opsin to respond, or the point-spread function is non-optimal and causes out of focus excitation, or other inputs to the neural circuit are active; and thus the response can be different than expected (Ronzitti et al., 2017; Russell et al., 2024; Lees et al., 2024). Instantaneous response model. We first assume a latent dynamical system with the framework:
xt+1 = f(xt) + S(xt, ut) · 1{ut̸=0} + ϵt, (3)
where xt is the latent state at time t, f is the autonomous mapping of the state from one timepoint to the next, S is a function describing the effect of a stimulation on a location in the latent state, and ϵ is a noise term. Here, u denotes the stimulation vector itself, potentially quite high-dimensional, and a zero u results in no stimulation and therefore no response affecting the dynamics. Most of the time ut will be zero, as we are assuming stimulations happen somewhat sparsely on the timescale of the neural data acquisition. This means we can train our estimate of the dynamics, ˆf, on the datapoints where we know ut = 0, during periods of non-stimulation (details in Appendix A).

$$x_{t+1}=f(x_{t})+S(x_{t},u_{t})\cdot\mathbf{1}_{\{u_{t}\neq0\}}+\epsilon_{t},$$
$\eqref{eq:walpha}$. 
$${\hat{f}}_{t+1}={\begin{cases}\operatorname{update}({\hat{f}}_{t},x_{t+1},x_{t}),\\ {\hat{f}}_{t},\end{cases}}$$
$$\begin{array}{l}{{\mathrm{if~}u_{t}=0}}\\ {{\mathrm{if~}u_{t}\neq0}}\end{array}$$
$$(4)$$
(update(ˆft, xt+1, xt), if ut = 0
ˆft, if ut ̸= 0(4)
To estimate S, we can rearrange our dynamics equation: S(xt, ut) = xt+1−f(xt)−ϵt and substitute in ˆft: S(xt, ut) ≈ sobs = xt+1 − ˆft(xt). This gives the following update rule for Sˆ:

$${\hat{S}}_{t+1}={\begin{cases}{\hat{S}}_{t},\\ \mathrm{update}({\hat{S}}_{t},u_{t},s_{\mathrm{obs}},t),\end{cases}}$$
(Sˆt, if ut = 0
update(Sˆt, ut, sobs, t), if ut ̸= 0(5)
Together, we can use ˆf and Sˆ to create a joint predictive model:

$$\begin{array}{l}{{\mathrm{if~}u_{t}=0}}\\ {{\mathrm{if~}u_{t}\neq0}}\end{array}$$
$$(S)$$
$${\hat{x}}_{t+1}={\hat{f}}_{t}(x_{t})+{\hat{S}}_{t}(x_{t},u_{t},t)$$

xˆt+1 = ˆft(xt) + Sˆt(xt, ut, t) (6)
Delayed response model. In many cases, the response to stimulation is not instantaneous, or the peak response to stimulation is not instantaneous. We model these cases using a paradigm similar to the one above, but using a fixed delay d: xt+1 = f(xt) + S(xt−d, ut−d) · 1{ut−d̸=0} + ϵt. Training of ˆf is mostly the same when d > 0, except timesteps between a stimulus and its response are left out of training. (Even in steps where the parameters of the ˆf estimator is not updated, the estimated state is still tracked.) We assume that a new stimulus is not delivered before we see the effects of a previous stimulus, so that there is never more than one stimulus "pending" at a given time. We

$$({\mathfrak{h}})$$

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 also optionally model the additive effects of stimulation as being spread out over time; if xˆt+1 =
ˆft(xt)+Sˆt(xt, ut, t), we optionally regress a small number of coefficients β to model the continuing effects of stimulation even after the stimulus is over: xˆt+i+1 = ˆft+i(xt+i) + βi· Sˆt(xt, ut, t).

Stimulus-response mapping estimator (Sˆ). For our model of S, we employ a kernel regression to model the effects of latent state, stimulus, and sample age by interpolating between previously observed stimulus-response pairs. We choose radial basis functions for our kernels K, where each scaling constant is optionally tuned by stochastic coordinate descent at each new observation.

$$\hat{S}(x,u,t)=\frac{\sum_{i=1}^{N}K_{1}\left(x,X_{i}\right)K_{2}\left(u,U_{i}\right)K_{3}\left(t,T_{i}\right)s_{\mathrm{obs},i}}{\sum_{i=1}^{N}K_{1}\left(x,X_{i}\right)K_{2}\left(u,U_{i}\right)K_{3}\left(t,T_{i}\right)}.$$
$$\left(7\right)$$

Kernel regression works well on limited data (few experimental observations of the results of stimulations), handles possible non-linearities in the response space, and is thus sufficiently flexible for learning potentially non-trivial stimulation-response maps across arbitrary latent spaces (Chen &
Shah, 2025). The consideration of sample age (Ti) allows it to discount old samples; this means that the regression can tuneably respond to instabilities in the underlying mapping, whether they are due to changes in upstream processing steps or biological changes like plasticity. If the system is stable, it can also use a very large radial basis scaling constant to effectively ignore the time feature.

## 2.4 Optimization Of Selected Stimulations

Stimulations can be designed using a variety of methods: some are based on anatomical region (Shang et al., 2024), on functional tuning of individual neurons (Draelos et al., 2025), on estimated uncertainty (Draelos & Pearson, 2020), on optimal experimental design to choose between a set of predetermined stimuli (Wagenmaker et al., 2024), or simply via random selection of groups of neurons. Here, instead of choosing from a limited set of predetermined stimuli, our method considers all possible stimuli, presenting a considerably larger space to search for feasible stimulations that nonetheless result in a desired effect on the latent dynamics. The tradeoff for this increased flexibility is a more approximate optimization and solution. We define a goal vector v in the latent space along which we want to perturb the latent neural activity. We control the stimulus vector u, and we model the perturbation it produces as s (which depends on u). The goal is choose u to get s to align closely to v. Under ideal conditions, the values of u are the same as the responses they evoke s, and we can optimize u against v directly. We call this an identity stimulus-response mapping, or open-loop optimization. However, such mappings are often more complicated, which is why we also optionally model the evoked response as s = Sˆ(xt*, u, t*) using the learned stimulus-response mapping (Fig. 2a). This adaptation to nonlinear stimulus-response mappings is possible because of the differentiable form of the estimator we use for Sˆ.

We can only stimulate N neurons, and each neuron must have a stimulation value between 0 and the maximum possible, which we normalize to 1. Rather than employ the L0 constraint on the number of neurons, which would make the problem NP hard in general, we use an L1 constraint on u offset by N to encourage a solution with the number of non-zero elements close to n.

$$\operatorname*{min}_{u\in\mathbb{R}^{N}}-{\frac{v^{\top}s(u)}{\|v\|\|s(u)\|}}+\lambda_{1}(\|u\|_{0}^{\operatorname*{max}}-\|u\|_{1}),\quad{\mathrm{s.~t.}}\quad{\mathbf{0}}\preceq u\preceq{\mathbf{1}}$$

## 3 Experiments

All experiments were conducted on custom-built workstations running Ubuntu 22.04 and containing 128 GB of RAM, a 32-core i9 intel CPU, an NVIDIA 3060 Ti GPU (8 GB memory), with a 1TB SSD. Experiments could all be run at high speeds, meaning total computation time was kept to less than 100ms, and averaged less than 10ms end-to-end for each timepoint of observed data.

Toy model. Our toy model is a circular linear dynamical system defined using: xt = Axt−1 + ϵt, yt = Cxt + ηt, where A is a rotation matrix in the first two components with a period of 30 + 1π
(
1 π is added to discourage point clustering in adjacent rotations) and a decay to zero in the third component. C is an identity matrix, and ϵt and ηt are process and observation noises respectively, both distributed as N (0, I3 · 0.05). The initial state x0 is typically initialized to [20 0 0]⊤.

$$(8)^{\frac{1}{2}}$$

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

## 4 Results

![6_image_0.png](6_image_0.png)

if $ u_t=0$ or $ (x_1=0$ and $ x_2=0)$  if $ u_t=1$  ... 
$$(9)$$
-0 0 0⊤, if ut = 0 or (x1 = 0 and x2 = 0)
i⊤, if ut = 1(9)
Real data. For each of the real datasets, we simulated stimulations using an autoregressive function to model a fast rise in neural activity of the perturbed neurons and a slower decay back to baseline levels. We transformed the data using the following updates: yt = rt +at, at =
0.8 · at−1 + ut, where rt and yt are the original and simulated data, respectively, aiis the additive stimulation, and ut is the stimulation. Figure 3a illustrates two example stimulations applied to the calcium imaging dataset where only stimulated neurons are displayed.

Calcium imaging. For the calcium imaging data, we used calcium traces recorded from mouse visual cortex expressing GCaMP6s (Zong et al., 2022). During the recording, the mice were foraging for dropped cookie crumbs the experimenter periodically threw into the environment. Frames were recorded with a miniscope at 15 Hz, for a recording duration of 20 min. The recordings were analysed with Suite2p (Pachitariu et al., 2016), which extracted 592 neural traces. We de-meaned each channel of the florescence output F, defined F0 as the median of each channel, and performed subsequent analyses on ∆F
F0=
F −F0 F0.

Electrophysiological. We used an electrophysiological dataset from a nonhuman primate (O'Doherty, 2024), recorded from 130 units in the sensorimotor cortex (monkey I).

During the recording, the animal was performing a 2D random-touch task. Threshold crossings were extracted from a 24.4 kHz recording and binned at 30 Hz over a recording length of 649 s. Figure 2: a. Diagram of a trajectory (black) whose dynamics are predicted to advance via the dashed trajectory (gray). If a stimulation v occurs, the activity instead proceeds along the new trajectory. S shows the latent response to stimulation. b. A circular system with location-dependent perturbation effects, showing 10 cycles (black). Stimuli displaces along the third dimension (red arrows). c. Expected norm-error between our estimate Sˆ and the true generating S over time. d. Surface plots showing the ground-truth effect of stimulations (left: stable, right: rotating). Scattered points are previously observed stimulus-response examples, colored by error. e. Error in the 1-step-ahead prediction for our regression method (magenta) and a comparative method that is blind to stimulation effects (gray). The underlying stimulus-response function changes (vertical lines), but the model adapts its temporal kernel length constant to recover. Stimulations are a binary decision at each timpoint; variation in stimulation magnitude and direction is due to the spatial structure of the stimulation-response mapping, S. In the toy model, S is: We first applied our response mapping method to the toy model (Fig. 2). Our regression estimator Sˆ quickly learns the underlying mapping function S within a few seconds, or cycles, of the circular dynamics being observed. To model the kind of instabilities found in real experiments, we first introduced a jump discontinuity, such as when an electrophysiological probe's position is shifted.

## 4.1 Stimulation Response Modeling

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 To model such a discontinuity in the ground-truth stimulus-response mapping, we flipped the map 180◦at t=25s (Fig. 2d). While a non-adaptive model that assumes a stable mapping would suffer increased predictive error after such an event, our model recovers from the perturbation within 15s (Fig 2e, 'Flip'). A second kind of instability we considered is continuous drift, which could be caused by photobleaching, plasticity, or a change in neuromodulator levels. To model drift in the ground-truth stimulus-response mapping, we continuously rotated the stimulus-response mapping at a rate of 1 revolution every 30 s, starting at t = 45 s. Our model continuously adjusts to mitigate the error in estimating the unstable underlying system (Fig 2e, 'Rotate'). We quantify the error in 1step-ahead prediction across all timepoints for our method as well as for a method that is blind to the stimulation by withholding stimulation times from the dynamical model. Both methods employed the same underlying dynamical model (KF) and their errors were similar during periods of nonstimulation. During and post stimulation, our method out-performed the blind comparison method (bold lines show smoothed average errors over 50 experiments). We next considered real experimental data from (Zong et al., 2022) with simulated stimulations applied along the first latent dimension Q0 as constructed in real time by proSVD (Fig. 3). We confirmed that the applied stimulations had the intended effect on the neural data in both the original high-dimensional neural space and in the learned latent space (Fig. 3a, b, respectively). In real experiments, there is often a lag between stimulus delivery and response, so we introduced a response delay of 0.2 s (or 4 timepoints) at t = 304 s. For both regimes, the one-step-ahead prediction error from our model is less than the error from the blind model. For this dataset, we used the KF method as the dynamical model (see Appendix C for comparison across all models). In all cases, our method quickly learned a stimulation-response mapping to account for the effects of stimulations in the latent space, and out-performed the comparison method. Figure 3: a. We apply a stimulus to 14 out of 592 neurons at timepoints 302.6 s (with a delay

![7_image_0.png](7_image_0.png)

of 0 s) and 305.7 s (with a delay of 0.26 s). The new fluorescence traces (black) show a varied effect on activity post stimulation (green vertical line). b. The delivered stimuli have the desired effect of pushing the neural trajectory (black) along the first latent dimension Q0 in the latent space constructed with proSVD (rightwards; green arrows). c. We plot the 1-step-ahead prediction error as a function of time and dynamic stimulations. Our model (magenta) successfully learns the response to stimulations, whereas the blind model (gray) consistently shows greater error during periods of stimulation. Dashed lines show respective averages during stimulations.

## 4.2 Stimulation Optimization

Previous neuroscience experiments have delivered optogenetic stimuli, though none used strategies for stimulating along latent directions. We can asses the degree to which a stimulation had the desired effect by checking the angle between v, the effect of stimulation we desired, and sobs, the deviation from previously predicted dynamics. First we tried stimulating random individual neurons. We found that the effect of activating random neurons had generally low alignment with our desired result of Q0. We then tried maximally activating groups of random neurons; this also did not align well with Q0. We then found that using the stimulations found with our method produces responses highly aligned with Q0 in the latent space, while shuffled versions of our stimulations do not. Via four comparisons, we found that our optimization outperforms random methods in designing stimuli that produce our desired latent effects. We showed above that we can stimulate along the first dimension in the latent space. However, our system also needs to be able to design stimuli to move neural latents in arbitrary directions. Therefore, next, we quantified how well we can target perturbations in arbitrary directions in the latent space by comparing the s(u) from equation (8) to v. Figure 4: a. Random methods, such as randomly stimulating single neurons (Single), groups of neurons (Multiple), or randomized versions of the stimuli our method designs (Shuffled), all produce stimulations that are less aligned with our target effect than the optimized stimuli (Designed). b. The predicted angle between the responses we expect (s) and the desired response (v) for the designed stimuli. We compare the results of optimizations for population-wide inhibition (Negative), population-wide excitation (Dense), random directions in the latent space (Random), random directions constrained to be feasible (Feasible), and along the first latent (Q0). c. Observed stimulation error (angle between sobs and v) plotted against predicted stimulation error (angle between s and v). Predicted error functions as a loose lower bound on the observed error.

This quantifies

![8_image_0.png](8_image_0.png) how well the optimization predicts it was able to design the stimulus. First, we considered stimulating in an infeasible direction, equivalent to requesting blanket inhibition v ∝ −Q⊤1
(Fig. 4, 'Negative'). Due to our nonnegativity constraint, any effect of stimulation we design could not possibly be correlated with v, just like how it is complicated to optogenetically inhibit activity by targeting excitatory opsins expressed in an excitatory neural population (Li et al., 2019). As expected, we see the angles between the designed s from the optimization and the infeasible v were high. We next checked the performance against another infeasible direction, blanket excitation v ∝ Q⊤1 ('Dense'). This is similar to blanket excitation that can be delivered by traditional optogenetic manipulations. While inhibition is infeasible due to our non-negativity constraint, blanket excitation is infeasible due to our sparsity constraint. Third, we optimized to stimulate along random directions in the latent space ('Random'). The wide distribution of angles suggests that while some directions are easy for the optimization to target, others are not. We then optimized to stimulate along random feasible directions in the latent space, where we designed the requested vectors to be reachable using the excitation of fewer than 30 neurons ('Feasible'). This case had the best performance, with 517/600 optimizations giving an optimization misalignment of less than 1
◦. Finally, we checked optimizing stimulations to push the population activity along the first latent variable, Q0, which we found to be similarly easy, with 508/600 optimizations giving an optimization misalignment of less than 1
◦.

Another way our stimulations could be challenged is if we have a poor understanding of the mapping from a stimulus to the neural response. So far we have compared the angle between the predicted result of stimulation, s, and v and the estimated result of stimulation, sobs and v. We next quantified how these estimates of our error correspond to each other. If we predict based on our optimization that the effects of our stimulation will have a certain error, we should expect the result of the stimulation to have at least that error. If we compare the angle between s and v, the predicted error, with the angle between sobs and v, the observed error, we can see that for a variety of targets, the true angle between sobs and v is greater than the predicted error.

For non-'Negative' targeted stimulations, fewer than 6% of optimizations had a lower observed error than predicted. This relationship holds the least for optimizations for the Negative target, where about half of optimizations (296/600) have a lower error than the optimization predicted, possibly because its infeasibility led our model to predict the maximum possible error.

Figure 5: a. For each experiment, Sˆ captures more

![8_image_1.png](8_image_1.png)

of the true structure of S over time and has lower prediction error on new training samples, confirming the convergence of Sˆ as a standalone estimator. b. Proportion of the magnitude of the observed sobs aligned with v for the open loop cases under trivial and non-trivial mappings, and for the closed-loop case for non-trivial maps. 10 experiments are run with over 100 stimulations each; solid lines are average values across experiments.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 9 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 What if we had seen disagreement between sobs and s? This would indicate our Sˆ model is a poor match for the system's true S. The above experiments assumed that the result of a stimulation u was simply its projection into the latent space S(u) = Q⊤u. Because this requires no feedback, or information about the result of the stimulation, we call it open loop mode. In closed loop mode, we can assume a more general form for S, but it must be learned via our Sˆ estimator in real time. Using such an estimated Sˆ, we can see in Fig. 5a that Sˆ learns at approximately the same rate in a simple
(black) vs. non-simple (green) stimulus-response mapping (final error values for individual trials were overlapping: 2.21 ± .9 for the simple mapping and 1.95 ± 0.79 for the non-simple mapping
(mean ± std). This is because our Sˆ estimator is non-parametric and makes few assumptions about the underlying stimulus-response mapping. Thus the simple mapping is about as easy to learn as the non-simple mapping. If we then use this estimator to optimize in the non-trivial stimulusresponse mapping case, we find that on average, the stimuli designed through the model have a larger proportion of their magnitude aligned with v than the open-loop stimuli (see Appendix G for an analysis of angles in these experiments).

## 5 Discussion

In this work, we describe a new streaming algorithm for stimulation-response modeling of latent neural dynamics, along with a novel optimization procedure for determining high-dimensional stimulation patterns to drive them in a desired direction. This provides, for the first time, a method for adaptive stimulation of latent neural activity that accounts for realistic experimental constraints in the original neural space. Importantly, we considered non-negative constraints for excitationonly interventions, a limit on the number of total targets in a single stimulation, and constraints on the overall magnitude of the applied stimulation. Our optimization framework operated in both the high- and low-dimensional spaces appropriate for this problem of driving latent dynamics via high-dimensional neural stimulations under feasibility constraints. We demonstrated our method's capabilities on synthetic data and two real experimental datasets with simulated effects of arbitrary stimulations, applied both in and out of the learned spaces. One limitation of our demonstrated approach is that we did not explicitly test using non-linear methods to construct the latent spaces. However, we note that this component of our method could be exchanged without affecting the other components (e.g., using kernelized PCA (Scholkopf et al. ¨ , 1997) for dimension reduction). A second limitation is that our real data experiments were performed offline, though in a realistic streaming setting. All aspects of our approach run efficiently and are fast enough to make real-time adaptive stimulation experiments feasible (see benchmarking in the Supplementary Materials. We also did not include any explicit consideration of the effects of stimulations on behavior. We note that a straightforward extension of our response-modeling method would be to (separately or jointly) model changes in a lower-dimensional behavioral space. This is feasible for many motor-relevant experiments in neuroscience, as in a 2-dimensional maze or reaching task, or via projecting behavior to its own latent representation (Stringer et al., 2019; Sani et al., 2021; Schneider et al., 2023). Future work could also include additional feasibility constraints on the nature of the stimulation; for example, targeting neurons with more opsin for photostimulation or based on their functional response properties to external stimuli (Russell et al., 2024; Draelos et al., 2025; Daie et al., 2021).

## 6 Ethics Statement

The authors are not aware of any potential violations of the ICLR Code of Ethics. We do not use human data, we only use publicly available datasets. We do not expect any harm to come from this work's methodologies, insights, or feasible applications. We are not aware of any conflicts of interest. We are not aware of any research integrity issues.

## 7 Reproducibility Statement

We will make all code necessary to reproduce our work publicly available via an installable Python package and repository on Github. Additionally, the code behind our method and the code to generate all of the figures in this document is available in the Supplementary Material.

## References

Pierre Ablin, Alexandre Gramfort, Jean-Franc¸ois Cardoso, and Francis Bach. Stochastic algorithms with descent guarantees for ICA. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics, pp. 1564–1573. PMLR, April 2019.

Hillel Adesnik and Lamiae Abdeladim. Probing neural codes with two-photon holographic optogenetics. *Nature neuroscience*, 24(10):1356–1366, 2021.

Nicolo Cesa-Bianchi, Yishay Mansour, and Ohad Shamir. On the complexity of learning with kernels. In *Conference on Learning Theory*, pp. 297–325. PMLR, 2015.

George H. Chen and Devavrat Shah. Explaining the Success of Nearest Neighbor Methods in Prediction, February 2025.

Mark M. Churchland, John P. Cunningham, Matthew T. Kaufman, Justin D. Foster, Paul Nuyujukian, Stephen I. Ryu, and Krishna V. Shenoy. Neural population dynamics during reaching. Nature, 487(7405):51–56, July 2012. ISSN 1476-4687. doi: 10.1038/nature11129.

Kayvon Daie, Karel Svoboda, and Shaul Druckmann. Targeted photostimulation uncovers circuit motifs supporting short-term memory. *Nature neuroscience*, 24(2):259–265, 2021.

Marco Dal Maschio, Joseph C Donovan, Thomas O Helmbrecht, and Herwig Baier. Linking neurons to network function and behavior by two-photon holographic optogenetics and volumetric imaging. *Neuron*, 94(4):774–789, 2017.

Anne Draelos, Pranjal Gupta, Na Young Jun, Chaichontat Sriworarat, and John Pearson. Bubblewrap: Online tiling and real-time flow prediction on neural manifolds. Advances in neural information processing systems, 34:6062–6074, 2021.

Anne Draelos, Matthew D Loring, Maxim Nikitchenko, Chaichontat Sriworarat, Pranjal Gupta, Daniel Y Sprague, Eftychios Pnevmatikakis, Andrea Giovannucci, Tyler Benster, Karl Deisseroth, et al. A software platform for real-time and adaptive neuroscience experiments. Nature Communications, 16(1):9909, 2025.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Sung Soo Kim, Herve Rouault, Shaul Druckmann, and Vivek Jayaraman. Ring attractor dynamics ´
in the Drosophila central brain. *Science*, 356(6340):849–853, May 2017. doi: 10.1126/science. aal4835.

Nuo Li, Susu Chen, Zengcai V Guo, Han Chen, Yan Huo, Hidehiko K Inagaki, Guang Chen, Courtney Davis, David Hansel, Caiying Guo, and Karel Svoboda. Spatiotemporal constraints on optogenetic inactivation in cortical circuits. *eLife*, 8:e48622, November 2019. ISSN 2050-084X. doi: 10.7554/eLife.48622.

Anne Draelos and John Pearson. Online neural connectivity estimation with noisy group testing.

Advances in Neural Information Processing Systems, 33:7437–7448, 2020.

Antonio Fernandez-Ruiz, Azahara Oliva, and Hongyu Chang. High-resolution optogenetics in space and time. *Trends in Neurosciences*, 45(11):854–864, 2022.

Logan Grosenick, James H Marshel, and Karl Deisseroth. Closed-loop and activity-guided optogenetic control. *Neuron*, 86(1):106–139, 2015.

Mehrdad Jazayeri and Arash Afraz. Navigating the Neural Space in Search of the Neural Code.

Neuron, 93(5):1003–1014, March 2017. ISSN 0896-6273. doi: 10.1016/j.neuron.2017.02.019.

Rudolph Emil Kalman. A new approach to linear filtering and prediction problems. 1960. Robert M Lees, Bruno Pichler, and Adam M Packer. Contribution of optical resolution to the spatial precision of two-photon optogenetic photostimulation in vivo. *Neurophotonics*, 11(1):015006–
015006, 2024.

Junfan Li and Shizhong Liao. Nearly optimal algorithms with sublinear computational complexity for online kernel regression. In *International Conference on Machine Learning*, pp. 19743–19766. PMLR, 2023.

Yuki Minai, Joana Soldado-Magraner, Matthew A. Smith, and Byron M. Yu. MiSO: Optimizing brain stimulation to create neural activity states. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, November 2024.

Joseph O'Doherty. MC RTT: Macaque motor cortex spiking activity during self-paced reaching
(Version draft), 2024.

Daniel J. O'Shea, Lea Duncker, Werapong Goo, Xulu Sun, Saurabh Vyas, Eric M. Trautmann, Ilka Diester, Charu Ramakrishnan, Karl Deisseroth, Maneesh Sahani, and Krishna V. Shenoy. Direct neural perturbations reveal a dynamical mechanism for robust computation, December 2022.

Marius Pachitariu, Carsen Stringer, Sylvia Schroder, Mario Dipoppa, L Federico Rossi, Matteo ¨
Carandini, and Kenneth D Harris. Suite2p: beyond 10,000 neurons with standard two-photon microscopy. *BioRxiv*, pp. 061507, 2016.

Chethan Pandarinath, K Cora Ames, Abigail A Russo, Ali Farshchian, Lee E Miller, Eva L Dyer, and Jonathan C Kao. Latent factors and dynamics in motor cortex and their application to brain– machine interfaces. *Journal of Neuroscience*, 38(44):9390–9401, 2018.

Nicolas C Pegard, Alan R Mardinly, Ian Ant ´ on Oldenburg, Savitha Sridharan, Laura Waller, and ´
Hillel Adesnik. Three-dimensional scanless holographic optogenetics with temporal focusing (3d-shot). *Nature communications*, 8(1):1228, 2017.

Diogo Peixoto, Jessica R. Verhein, Roozbeh Kiani, Jonathan C. Kao, Paul Nuyujukian, Chandramouli Chandrasekaran, Julian Brown, Sania Fong, Stephen I. Ryu, Krishna V. Shenoy, and William T. Newsome. Decoding and perturbing decision states in real time. *Nature*, 591(7851): 604–609, March 2021. ISSN 1476-4687. doi: 10.1038/s41586-020-03181-9.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Sean M. Perkins, Elom A. Amematsro, John P. Cunningham, Qi Wang, and Mark M. Churchland.

An emerging view of neural geometry in motor cortex supports high-performance decoding, July 2024.

Priyamvada Rajasethupathy, Emily Ferenczi, and Karl Deisseroth. Targeting neural circuits. *Cell*,
165(3):524–534, 2016.

Emiliano Ronzitti, Cathie Ventalon, Marco Canepari, Benoˆıt C Forget, Eirini Papagiakoumou, and Valentina Emiliani. Recent advances in patterned photostimulation for optogenetics. Journal of Optics, 19(11):113001, 2017.

Lloyd E Russell, Mehmet Fis¸ek, Zidan Yang, Lynn Pei Tan, Adam M Packer, Henry WP Dalgleish, Selmaan N Chettih, Christopher D Harvey, and Michael Hausser. The influence of cortical activity ¨
on perception depends on behavioral state and sensory context. *Nature Communications*, 15(1):
2456, 2024.

Omid G Sani, Hamidreza Abbaspourazad, Yan T Wong, Bijan Pesaran, and Maryam M Shanechi.

Modeling behaviorally relevant neural dynamics enabled by preferential subspace identification. Nature neuroscience, 24(1):140–149, 2021.

Shreya Saxena and John P Cunningham. Towards the neural population doctrine. Current opinion in neurobiology, 55:103–111, 2019.

Steffen Schneider, Jin Hwa Lee, and Mackenzie Weygandt Mathis. Learnable latent embeddings for joint behavioural and neural analysis. *Nature*, 617(7960):360–368, 2023.

Bernhard Scholkopf, Alexander Smola, and Klaus-Robert M ¨ uller. Kernel principal component anal- ¨
ysis. In *International conference on artificial neural networks*, pp. 583–588. Springer, 1997.

Nishal Pradeepbhai Shah, AJ Phillips, Sasidhar Madugula, Amrith Lotlikar, Alex R Gogliettino, Madeline Rose Hays, Lauren Grosberg, Jeff Brown, Aditya Dusi, Pulkit Tandon, et al. Precise control of neural activity using dynamically optimized electrical stimulation. *Elife*, 13:e83424, 2024.