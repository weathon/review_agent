000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Comparison Of Neural Network Architectures In The Thermal Explosion Approximation Prob- Lem

Anonymous authors Paper under double-blind review

## Abstract

This study investigates the effect of neural network architecture on the accuracy of data-driven modeling of thermal explosions in a hydrogen–oxygen–air mixture. Using a reduced kinetic mechanism for 11 reagents, the thermal explosion process is simulated under specified initial pressure and temperature conditions, generating time-resolved data. We compare three architectures: a standard multilayer perceptron (MLP), a DeepONet–inspired model, and our U-Net–style residual network, evaluating their ability to capture transient dynamics and key reaction regimes. Our results demonstrate that network architecture has an important impact on predictive performance. The U-Net architecture consistently outperformed the other models, achieving a mean squared error (MSE) of 0.0013 with a standard deviation (STD) of 0.0218, demonstrating high fidelity in capturing both rapid transients and slower reaction dynamics. In contrast, the DeepONetinspired model and the MLP achieved MSEs of 0.0181 (STD 0.0581) and 0.0202 (STD 0.0682), respectively, indicating reduced accuracy and greater variability in predictions. The large spread in error is due to the fact that neural networks are not always able to accurately approximate the various modes of the combustion process. Despite testing various architectures and using a fairly large dataset, the problem remains unresolved. These findings indicate the importance of selecting appropriate network architectures to combine deep learning with chemically detailed kinetic modeling. Such careful selection open the way for more reliable and interpretable predictive models in combustion and reactive-flow applications.

## 1 Introduction

The numerical simulation of gas-dynamic processes in high-energy systems, such as engines, gas turbines, and other propulsion or power-generation devices, requires a detailed description of complex physicochemical phenomena. These include turbulent combustion, shock wave interactions, and multiphase flows, all of which are strongly influenced by chemical kinetics. Modern computational fluid dynamics (CFD) approaches employ high-fidelity reaction mechanisms that account for hundreds or even thousands of elementary chemical reactions among dozens of interacting species. Such detailed models are essential for accurately predicting critical processes like ignition delay, flame stabilization, deflagration-to-detonation transition (DDT), and detonation wave propagation. However, the high level of detail in these simulations comes at a considerable computational cost. Resolving all relevant spatial and temporal scales in 3D simulations demands immense computational resources, making such calculations impractical for routine engineering analyses or real-time applications. To address this challenge, researchers employ various acceleration techniques, including reduced-order modeling, adaptive mesh refinement, and hybrid combustion models that balance accuracy and efficiency. Despite these advancements, the trade-off between computational feasibility and predictive accuracy remains a key research focus in computational combustion and reactive gas dynamics (Pantano et al., 2004; Smirnov et al., 2015). The main computational bottleneck in coupling hydrodynamics with chemical kinetics lies in solving stiff systems of ordinary differential equations (ODEs) that describe the temporal evolution of temperature and species concentrations.

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 Despite advances in mechanism reduction techniques (Løvas, 2009; Koniavitis et al., 2016; Lu et al., ˚ 2021) and tabular approaches (Pope, 1997), the computational cost of detailed chemical kinetics still constitutes a substantial fraction of the total simulation time. In recent years, machine learning techniques, in particular neural network models, have attracted growing interest as a means to accelerate reactive flow simulations. Training a neural network to approximate the thermochemical evolution of a reactive medium can reduce computational demands while maintaining acceptable accuracy (Tonse et al., 1999; Ji & Deng, 2021). In this context, the selection of neural network architecture plays a decisive role in achieving an appropriate balance between predictive accuracy, generalization capability, and computational efficiency. A growing body of research indicates that architectural features—such as network depth, width, residual connections, and hierarchical representations—exert a profound influence on both the stability of predictions. For chemically reactive systems, architectures capable of capturing multiscale dependencies are particularly critical, as the temporal evolution of species involves both ultrafast radical-mediated pathways and comparatively slow thermodynamic relaxation processes. Consequently, network design should be regarded not only as an implementation detail, but as a fundamental determinant of whether a model can faithfully reproduce combustion phenomena across a broad spectrum of operating conditions.

Among the emerging operator-learning paradigms, DeepONet has attracted substantial attention.

By decomposing the mapping into a branch network (encoding input functions) and a trunk network (encoding target coordinates), DeepONet has demonstrated considerable success in learning nonlinear input–output operators, making it attractive for stiff ODEs and parametric PDEs (Lu et al., 2021; Wang et al., 2021; Li et al., 2020). Nevertheless, most existing studies have focused on relatively simplified scenarios—such as narrow ranges of boundary or initial conditions, reduceddimensional systems, or short integration horizons—which restricts their direct applicability to realistic combustion environments. In many studies with DeepONet, the training datasets and temporal discretization are chosen in an artificial way, not reflecting realistic combustion conditions. For example, in Goswami et al. (2024), the syngas dataset was generated at a fixed chemistry timestep (∆tchm = 10−8s) and the model was trained only to predict four pre-selected future instants (ti + 250, 500, 750, 1000), rather than learning dynamics at arbitrary temporal resolutions.

Such limitations arise because the branch–trunk decomposition tends to smooth operator mappings, whereas combustion dynamics often exhibit strongly localized nonlinearities and discontinuities. This suggests a fundamental open question: can operator-learning architectures such as DeepONet provide superior accuracy and robustness compared to conventional hierarchical models (e.g., U- Net–style residual networks) when applied to reactive flow simulations that more closely approximate real-world scenarios, characterized by extreme transients, multiscale coupling, and broad parameter variations. Addressing this question forms the motivation for the present comparative investigation, in which we systematically evaluate three representative neural architectures—a plain multilayer perceptron (MLP), our custom-designed U-Net–inspired residual network, and a DeepONetstyle operator-learning model—on the same high-dimensional dataset.

## 2 Problem Statement

In the present work, the thermal explosion of a hydrogen-oxygen mixture in air is modeled. Having set the initial pressure and temperature, the combustion of the mixture is calculated at regular intervals. The reduced kinetic mechanism proposed in (Tereza et al., 2019) is used to calculate the chemical transformations. During the mechanism 9 hydrogen-oxygen compounds are formed (H2, O2, H2O, OH, H, O, HO2, H2O2, OH*), as well as nitrogen and argon, which affect the dynamics of gases and reaction rates, but do not form compounds, their concentration is a constant value. Chemical kinetics describes the reactions of decay and association of chemical elements at every point in space, described by a system of differential equations:

$$\mathbf{\hat{\tau}}=f\left(p,T,\mathbf{X}\right)$$
∂X
∂t 
= f (*p, T,* X) (1)
where p is the pressure, T is the temperature, X is the vector of molar densities of elements. Computation of system (1) by numerical methods takes about 90 percent of time resources. The use of neural network models makes it possible to significantly speed up the process of data generation and calculations, which is important for practical application of combustion and detonation calculations. At the same time, high speed should not reduce the accuracy of predictions, since the physical validity and suitability of the model depend on it. Enhancing predictive accuracy can involve several strategies, including expanding and diversifying the training dataset. Nevertheless, the architecture of the neural network remains the primary determinant of performance, and this study focuses on elucidating its impact.

## 3 Data For Training And Testing Of Neural Network

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 The governing equation (1) possesses a unique solution that can be obtained numerically using the stiff ODE solver developed by (Novikov, 2007). This model describes the autoignition process of a preheated hydrogen-air mixture, characterized by a rapid transition to combustion under nearly adiabatic conditions in a confined volume. A key feature of the numerical algorithm is the semianalytical computation of the Jacobian matrix of the system's right-hand side, which significantly improves computational efficiency and ensures numerical stability when dealing with strongly stiff kinetics. Direct simulation of the complete gas-dynamic evolution inside the combustion chamber is computationally extremely demanding, even if simplified two-dimensional models are employed. To mitigate this difficulty while preserving accuracy, the computational domain is discretized into uniform volumetric cells. Chemical kinetics are then solved independently within each cell, provided with a set of initial conditions (pressure, temperature, and molar densities of chemical species). The stiff ODE solver computes the temporal evolution of the chemical system efficiently within each cell. For the present investigation, this solver was used to generate a large database of chemical states at discrete time intervals under a wide variety of randomized thermodynamic conditions. The sampling space was designed to cover practically relevant ranges of parameters: This strategy ensured that extreme combustion regimes were included, in terms of capturing the full diversity of chemical behavior: from slow reaction zones to sudden autoignition and explosive events. The resulting dataset provides a classical reference collection of kinetic trajectories. The generated data (see Fig. 1) were represented as 13-dimensional vectors, each comprising the time step ∆t, the system temperature, and the molar concentrations of 11 chemical species.

$$T\in[250,\,5000]\ \mathbf{K},$$
$$p\in[10^{4},\,2\times10^{7}]\ \mathrm{Pa},$$

∆t ∈ [10−10, 10−5] s.

Both the input and the output of the neural network therefore had dimension 13. The dataset is split into 50,000 training, 15,000 validation, 5,000 test samples. Dataset represents a broad and relatively unbiased sampling of the chemical system, containing numerous examples of trajectories with long induction times, abrupt ignition events, and smooth transitions to equilibrium. It serves as a general-purpose training dataset for building machine learning models of combustion dynamics.

## 4 Neural Network Architectures

Figure 2 illustrates the three neural network architectures explored in this work: (A) a plain multilayer perceptron (MLP), (B) a U-Net–like residual network, and (C) a DeepONet–style model. All three accept the same 13-dimensional input vector 162

![3_image_0.png](3_image_0.png) 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 where dt is the time increment, T the temperature, and C1 *. . . C*11 the species concentrations (the last two being N2 and Ar). Despite this common input and the shared training procedure, each architecture has distinctive structural features.

$$X={\big(}d t,\;T,\;C_{1},\ldots,C_{11}{\big)},$$

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

## 4.4 Training Details

Each layer in all models is a linear transformation hl = Wlhl−1 + bl, (2)
where Wl and bl are trainable weights and biases. The hidden activations are Leaky ReLU,
LeakyReLU(z) = max{0.01*z, z*}. (3)

![4_image_0.png](4_image_0.png) 

... 

... 

## 4.1 Plain Mlp

The simplest model is a deep fully connected network with five dense layers: input 13 × 100 → 100 × 120 → 120 × 120 → 120 × 100 → 100 × 13 output. A Leaky ReLU activation with negative slope 10−2follows every hidden layer. The first and last layers perform dimensional expansion and reduction, while the middle layers provide nonlinear mixing of features. To preserve physically invariant quantities, the output components for dt and the last two concentrations (N2, Ar) are directly copied from the input.

## 4.2 U-Net–Like Residual Network

This architecture adds hierarchical skip connections. The 13-dimensional input is first expanded through a 13 × 100 layer, then passed through three dense blocks: 100 × 120 → 120 × 120 → 120 × 100 with Leaky ReLU activations. The block output is added to the expansion output (a local skip), followed by a compression layer (100 → 13) and a global skip that adds the original input vector to the final output. As in the MLP, dt and N2/Ar are enforced to match the input, and the output is clamped to the range [−10, 10].

## 4.3 Deeponet–Style Model

This network separates processing of the scalar dt from the remaining 12 variables. One branch maps the 12 state variables through layers 12 × 120→120 × 120→120 × 120, reshaping the result to a 12 × 10 matrix. The second branch maps dt through 1 × 32 → 32 × 32 → 32 × 10. A matrix product of these branch outputs yields a 12-component fused vector, which is concatenated with dt to form the 13-dimensional output. The dt and N2/Ar components are again fixed to their input values. This design follows the operator-learning principle of DeepONet, where the trunk network (for dt) provides coefficients for the basis generated by the branch network.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 All models are trained with the Adam optimizer using a learning rate of 0.001, a batch size of 5,000, and were trained for 100 epochs. During training, each network minimizes the multi-step prediction error by recursively forecasting the state vector up to thirty steps ahead:

$$\mathrm{Loss}=\sum_{k=1}^{n_{\mathrm{samples}}}{\frac{1}{k}}\,\mathrm{MSE}\big(X_{t+k\Delta t},\;\hat{X}_{t+k\Delta t}\big),$$

$$(4)$$
MSEXt+k∆t, Xˆt+k∆t, (4)
where nsteps = 30. This strategy encourages the models to account for error accumulation while retaining their architectural differences.

## 5 Results

A systematic comparison of three neural network architectures shows how model design directly affects the accuracy of data-driven combustion-dynamics approximation. Performance was quantified using the mean squared error (MSE) on an identical test set to ensure fairness of evaluation. Statistical metrics for all three models are summarized in Table 1. The table includes not only the mean MSE and the standard deviation, but also the 95% confidence intervals (CI) for each model under identical testing conditions.

| Table 1: Comparison of MSE statistics for the three architectures on the identical test set Model Mean MSE Std. Dev. 95% CI MLP 2.029 × 10−2 6.829 × 10−2 [1.840 × 10−2 , 2.218 × 10−2 ] U-Net 1.374 × 10−3 2.183 × 10−2 [7.692 × 10−4 , 1.980 × 10−3 ] DeepONet 1.808 × 10−2 5.812 × 10−2 [1.647 × 10−2 , 1.969 × 10−2 ]   |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|

The comparatively large spread of errors for all three networks indicates that certain test trajectories remain challenging to approximate. Combustion dynamics often involve abrupt temperature rises, nonlinear chemical–kinetic feedback, and multi-scale temporal behavior. These features can produce regimes that are difficult for purely data-driven models to capture accurately, even when the training dataset is extensive and diverse. Despite identical training conditions and data preprocessing, the difference in MSE demonstrates that network architecture is an important factor. Direct comparison of the 95% CIs shows that the U-Net's interval [7.692 × 10−4, 1.980 × 10−3] does not overlap with the intervals of either MLP or DeepONet, confirming a statistically significant improvement. In contrast, the DeepONet and MLP
intervals [1.647 × 10−2, 1.969 × 10−2] and [1.840 × 10−2, 2.218 × 10−2] overlap substantially, suggesting comparable performance between these two architectures within the test conditions.

The standard deviation of U-Net (2.183 × 10−2) is also much smaller in absolute terms than that of MLP (6.829 × 10−2) and DeepONet (5.812 × 10−2), indicating more stable predictions across diverse trajectories. The U-Net's encoder–decoder design with skip connections appears to capture both global trends and localized transients without increasing computational cost relative to the simpler models. This multi-scale representation likely underlies its lower MSE, narrower CI, and reduced variability, making it the most reliable architecture among those tested. A detailed analysis of individual test cases further supports this conclusion. All trajectories in the figures (Figure 3 and Figure 4) are plotted in the same normalized space that was used to train the networks. Thus, the vertical axis in each subplot represents the dimensionless normalized value ensuring direct comparability of predicted and reference profiles.

Figure 3 illustrates a representative high-quality prediction: this trajectory belongs to the lowest 10% of test-sample MSE values (i.e., among the best cases for each model). The U-Net captures both transient peaks and long-term plateaus more accurately than the other models, with its output remaining phase-aligned with the true dynamics—peaks, plateaus, and sharp decays occur at the correct times.

![6_image_0.png](6_image_0.png)

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 By contrast, Figure 4 shows a trajectory from the upper quartile of the MSE distribution, representing

![6_image_1.png](6_image_1.png) a more difficult case. Even in this challenging example, the U-Net output remains aligned with the true temporal trends, preserving ignition peaks and decay phases, whereas the DeepONet and MLP predictions gradually drift away and exhibit phase lag. These findings emphasize that, for stiff chemical–kinetic systems, incorporating hierarchical feature extraction and residual connections—as in the U-Net—substantially enhances both accuracy and 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 E.A. Novikov. Investigation of (m,2)-methods of stiff systems calculation. *Vychislitel'nye* tekhnologii (Calculation technologies), 12(5):103–115, 2007.

Carlos Pantano, Sutanu Sarkar, and Forman A. Williams. Mixing of a conserved scalar in a turbulent reacting shear layer. *Journal of Fluid Mechanics*, 481:291–328, 2004. doi: 10.1017/ S0022112003003872.

This study demonstrates that neural network architecture plays a key role in the accuracy and robustness of data-driven modeling of reactive kinetics. Among the tested models, the U-Net–style residual network consistently outperformed both the MLP and the DeepONet-inspired architecture, achieving substantially lower mean squared error and reduced variability in predictions. Importantly, this improvement was not limited to absolute accuracy: the U-Net preserved the correct qualitative dynamics of combustion processes, maintaining synchrony with reference trajectories across sharp transients and plateau-like regimes, whereas the other architectures tended to drift away and lose physical consistency.

Our comparative analysis illustrates that the choice of architecture can be as critical as the size or the diversity of the dataset. While MLP and DeepONet-based models captured only partial aspects of the dynamics, the U-Net–style design provided stable and physically meaningful approximations without requiring additional data or computational cost. These findings illustrate the importance of architectural design in constructing reliable neural network surrogates for complex chemical systems. Overall, the results confirm the promise of U-Net–based architectures and emphasize the potential of combining deep learning with physically motivated design principles to create interpretable, accurate, and robust tools for chemical kinetics. ACKNOWLEDGMENTS Will be added after double-blind review.

## References

Somdatta Goswami, Ameya D. Jagtap, Hessam Babaee, Bryan T. Susi, and George Em Karniadakis.

Learning stiff chemical kinetics using extended deep neural operators. Computer Methods in Applied Mechanics and Engineering, 419:116674, 2024. doi: 10.1016/j.cma.2023.116674.

W. Ji and S. Deng. Kinet: A deep neural network representation of chemical kinetics. arXiv preprint arXiv:2108.00455, 2021. doi: 10.48550/arXiv.2108.00455. URL https://arxiv.org/ abs/2108.00455.

P. Koniavitis, S. Rigopoulos, and W. P. Jones. A methodology for derivation of rcce-reduced mechanisms via csp. *Combustion and Flame*, 183:126–143, 2016. doi: 10.1016/j.combustflame.2017. 05.010.

Zongyi Li, Nikola Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew Stuart, and Anima Anandkumar. Fourier neural operator for parametric partial differential equations. *arXiv preprint arXiv:2010.08895*, 2020. doi: 10.48550/arXiv.2010.08895. URL https://arxiv.org/abs/2010.08895.

Lu Lu, Pengzhan Jin, and George E. Karniadakis. Learning nonlinear operators via deeponet based on the universal approximation theorem of operators. *Nature Machine Intelligence*, 3:218–229, 2021. doi: 10.1038/s42256-021-00302-5.

Terese Løvas. Automatic generation of skeletal mechanisms for ignition combustion based on ˚
level of importance analysis. *Combustion and Flame*, 156(7):1348–1358, 2009. doi: 10.1016/j. combustflame.2009.03.009.

robustness. Such design choices are as important as dataset size or optimizer tuning and should guide future efforts to create reliable, physics-aware surrogates for combustion simulations.

## 6 Conclusions

S. B. Pope. Computationally efficient implementation of combustion chemistry using in situ adaptive tabulation. *Combustion Theory and Modelling*, 1(1):41–63, 1997. doi: 10.1080/713665229.

N. N. Smirnov, V. B. Betelin, V. F. Nikitin, L. I. Stamov, and D. I. Altoukhov. Accumulation of errors in numerical simulations of chemically reacting gas dynamics. *Acta Astronautica*, 117: 338–355, 2015. doi: 10.1016/j.actaastro.2015.08.013.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 A. M. Tereza, S. P. Medvedev, and V. N. Smirnov. Experimental study and numerical simulation of chemiluminescence emission during the self-ignition of hydrocarbon fuels. *Acta Astronautica*, 163:18–24, 2019. doi: 10.1016/j.actaastro.2019.03.001.

S. R. Tonse, N. W. Moriarty, N. J. Brown, and M. Frenklach. Piecewise reusable implementation of solution mapping. prism: an economical strategy for chemical kinetics. Israel Journal of Chemistry, 39(1):97–106, 1999. doi: 10.1002/ijch.199900010.

Sifan Wang, Wang Hanwen, and Paris Perdikaris. Learning the solution operator of parametric partial differential equations with physics-informed deeponets. *Scientific Reports*, 11:1–33, 2021.

doi: 10.48550/arXiv.2103.10974.