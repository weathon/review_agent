000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028

## 029

030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Feature selection (FS) is a fundamental challenge in machine learning, particularly for high-dimensional tabular data, where interpretability and computational efficiency are critical. Existing FS methods often cannot automatically detect the number of attributes required to solve a given task and involve user intervention or model retraining with different feature budgets. Additionally, they either neglect feature relationships (filter methods) or require time-consuming optimization (wrapper and embedded methods). To address these limitations, we propose AutoNFS, which combines the FS module based on Gumbel-Sigmoid sampling with a predictive model evaluating the relevance of the selected attributes. The model is trained end-to-end using a differentiable loss and automatically determines the minimal set of features essential to solve a given downstream task. Unlike existing approaches, AutoNFS achieves a nearly constant computational overhead regardless of input dimensionality, making it scalable to large data spaces. We evaluate AutoNFS on well-established classification and regression benchmarks as well as real-world metagenomic datasets. The results show that AutoNFS consistently outperforms both the classical and neural FS methods while selecting significantly fewer features. We share our implementation of AutoNFS at https://anonymous.4open.science/r/AutoNFS-8753

## 1 Introduction

# Autonfs: Automatic Neural Feature Selec- Tion

Anonymous authors Paper under double-blind review

## Abstract

Feature selection (FS) remains a long-standing challenge in machine learning and data analysis, particularly for high-dimensional tabular datasets, where interpretability and efficiency are crucial Theng & Bhoyar (2024); Dhal & Azad (2022). In practice, such datasets are often constructed by aggregating all available features or by manually engineering additional ones, which frequently leads to an excessive number of variables, many of which contribute little to downstream tasks. FS addresses this issue by identifying and removing redundant or irrelevant features, thereby improving the interpretability of the model, reducing complexity, and providing clearer insights. Furthermore, training a subsequent prediction model on reduced data helps mitigate model overfitting, reduce variance, and often improve predictive performance.

Existing FS approaches can be broadly categorized into filter Yu & Liu (2004); Smieja et al. (2014), ´
wrapper Kohavi & John (1997); Maldonado & Weber (2009), and embedded methods Tibshirani (1996b); Zou & Hastie (2005), each with inherent limitations. Filter methods rank features according to statistical relevance but remain independent of the learning model, potentially overlooking complex feature interactions. Wrapper methods iteratively select features using the predictive performance of a model as a criterion, but suffer from high computational costs. Embedded methods, such as L1 regularization or attention-based mechanisms, integrate FS within the learning process but may introduce instability or lack fine-grained control over feature importance. The computational cost of most FS algorithms grows rapidly with the number of input dimensions, making them inefficient for large datasets Tan et al. (2014). Additionally, the number of selected features is usually treated as a user-defined hyperparameter; an inappropriate choice can lead to suboptimal performance and require multiple retrainings. To address these limitations, we propose **AutoNFS**, a neural network for efficient and automatic FS. AutoNFS is a fully differentiable approach, consisting of two networks trained end-to-end (Figure 1). The masking network generates a mask that indicates selected features using temperature1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 controlled Gumbel-Sigmoid sampling Maddison et al. (2017); Jang et al. (2017a), while the target network is a predictive model to evaluate their relevance in a downstream task. Unlike existing methods, where the user must specify the desired number of features, AutoNFS automatically determines the minimal subset of features sufficient for the downstream task through a penalty loss component. Moreover, by designing AutoNFS as a modern neural network, it maintains almost constant computational overhead regardless of the dimensionality of the data, making it highly scalable in high dimensions. We evaluate AutoNFS on well-established classification and regression benchmarks with three scenarios of adding corrupted features Cherepanova et al. (2023). Our experiments demonstrate that AutoNFS consistently outperforms existing techniques while selecting significantly fewer features (Figures 2 and 3). These results are supplemented with the evaluation of AutoNFS in real-world metagenomic datasets (Table 2), analysis of its computational complexity (Figures 4a and 4b) and the visualization of its interpretability in the example of MNIST dataset (Figures 7 and 8). Our contributions can be summarized as follows.

- We propose AutoNFS, a novel neural network for end-to-end FS, leveraging Gumbel-
Sigmoid relaxation and a regularization term that penalizes the number of selected features.

- We show that AutoNFS automatically identifies a minimal yet sufficient subset of features, achieving a nearly constant computational overhead regardless of the input dimensionality, making it scalable for high-dimensional data.

- We validate our approach on well-established OpenML-based benchmarks for FS showing its advantage over related methods. In addition, it is examined on real-world metagenomic datasets, highlighting its effectiveness in high-dimensional biological data analysis.

## 2 Related Work

In Cheng (2024), the importance of FS is reviewed broadly, focusing on filter, wrapper, and embedded methods. Similar surveys have emphasized that the basic taxonomy remains relevant, but must now account for the issues of scalability, fairness, and interpretability in modern high-dimensional data analysis Guyon & Elisseeff (2003); Kohavi & John (1997); Chandrashekar & Sahin (2014); Brown et al. (2012). Due to the page limit, we refer the reader to Appendix A for a detailed description of the classical methods. The rise of deep learning has inspired neural approaches to FS Ho et al. (2021). Early attempts penalized input weights or used shallow gating networks Li et al. (2016). Later, continuous relaxations allowed discrete masks to be trained via SGD. Louizos et al. (2017) introduced Hard-Concrete gates for L0 regularization; Yamada et al. (2020b) proposed Stochastic Gates (STG); and Balın et al.

(2019) designed Concrete Autoencoders that explicitly reconstruct inputs from a subset of features. INVASE Yoon et al. (2018) went further, training an instance-specific selector and predictor in tandem. LassoNet Lemhadri et al. (2021) enforced a hierarchical coupling between a linear skip and deep features to guarantee consistency. Attention mechanisms in Transformers have also been used as feature selectors, but their explanatory validity is contested Serrano & Smith (2019); Jain & Wallace (2019); Gorishniy et al. (2023). Our work builds on this differentiable line. The technical foundation comes from the Gumbel–Softmax trick Jang et al. (2017a); Maddison et al. (2017), which provides low-variance gradients for sampling. This idea has been extended to subset selection through Gumbel-Top-k Kool et al. (2019), continuous relaxations for sampling without replacement Xie & Ermon (2019), and differentiable sorting operators Blondel et al. (2020). Strypsteen & Bertrand (2024) proposed Conditional Gumbel–Softmax to incorporate structural constraints into FS, such as sensor topologies. Unlike these, AutoNFS addresses unconstrained tabular data and eliminates the need to specify the number of features, letting it emerge from optimization through a cardinality penalty.

Another important line of work studies the acquisition of features *dynamic*, where features have costs and are revealed sequentially. Recent methods query features conditioned on previously observed values Covert et al. (2023); Yasuda et al. (2023), or use reinforcement learning to optimize acquisition policies (e.g., EDDI, budgeted classification) Ma et al. (2019); Janisch et al. (2019); Trapeznikov & Saligrama (2013). These methods are attractive when data acquisition is expensive (medical tests, sensor readings), but they solve a different problem than ours: we focus on learning a single global mask that amortizes selection across all samples, making inference fast and predictable. Finally, reliability and fairness in FS have also been addressed. Knockoff-based methods provide false discovery rate control Barber & Candès (2015); Romano et al. (2019), while stability selection explicitly balances sparsity and robustness Meinshausen & Buehlmann (2009). Greedy and OMP- style selectors have been extended to guarantee approximation bounds and fairness in large-scale problems Quinzan et al. (2023). These approaches focus on statistical guarantees, while our method emphasizes efficiency and scalability in neural training.

## 3 The Proposed Model

In this section, we introduce AutoNFS, a neural network approach for automatic selection of features, which are relevant for a given machine learning task. First, we give a brief overview of AutoNFS. Next, we describe its main building blocks. Finally, we summarize the training algorithm and the inference phase.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 where gi ∼ − log(− log(u)) with u ∼ Uniform(0, 1) is the Gumbel noise, σ is the sigmoid function, and τ > 0 is the temperature parameter.

For τ > 0, the mask m = (m1*, . . . , m*D) sampled from the Gumbel-Sigmoid distribution can take a continuous (non-binary) form. As τ decreases, the mask approaches the binary vector, which represents the final discrete mask. Slow decrease of the temperature τ allows the model to learn the optimal mask during network training.

## 3.1 Overview Of Autonfs

AutoNFS is a neural network that incorporates features selection into a process of learning a predictive model. It retrieves a variable-size subset of attributes that are the most informative for solving a given classification or regression task. The architecture of AutoNFS consists of two components: *masking and task networks*, see Figure 1. While the masking network generates a mask representing selected features, the task network solves the underlying task using the indicated attributes. The loss function of AutoNFS combines cross-entropy (for classification) or mean square error (for regression) with the penalty term, which encourages the model to minimize the number of selected features. In consequence, the task network plays the role of a discriminator, which verifies the usefulness of the features chosen for a given task. In contrast to traditional methods for FS, which iteratively add or reduce attributes, AutoNFS uses a differentiable mechanism to learn a mask based on the Gumbel-Sigmoid relaxation of the discrete distribution Jang et al. (2017a); Maddison et al. (2017). This design ensures that the computational time remains nearly constant regardless of the input dimensionality, making it particularly efficient for high-dimensional data.

## 3.2 Masking Network

The masking network f : R
De → R
D is responsible for generating a mask that indicates features selected for a given dataset {(xi, yi)}
N
i=1 ⊂ R
D. Given a randomly initialized input embedding e ∈ R
De, the network f outputs D-dimensional vector w = fϕ(e) ∈ R
D, which determines the mask. More precisely, the output vector w = (w1*, . . . , w*D) is transformed via a sequence of D Gumbel-Sigmoid functions to the (non-binary) mask vector m = (m1*, . . . , m*D), where mi = GS(wi; τ ) is given by the Gumbel-Sigmoid function with the temperature parameter τ > 0. Let us recall that the Gumbel-Sigmoid function is given by:

$$\mathbf{GS}(w_{i};\tau)=\sigma\left({\frac{w_{i}+g_{i}}{\tau}}\right)$$
,

![3_image_0.png](3_image_0.png)

where λ is hyperparameter. We experimentally verified that using a constant value λ = 1 gives satisfactory results across datasets. Thanks to the Gumbel-Sigmoid relaxation of the discrete mask distribution, we can learn the mask during end-to-end differentiable training.

## 3.4 Training Process

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 The complete loss function is then given by: To learn the optimal mask, we need to verify whether it is informative for the underlying task (e.g. classification). To this end, we first apply a mask m to the input example x, by element-wise multiplication xm = m ⊙ x. Next, we feed a task network g : R
D → Y with xm to obtain the final output g(xm). The relevance of features selected by m is quantified by the cross-entropy or mean-square loss denoted by L*task*(y; g(xm)). Furthermore, to encourage the model to eliminate redundant features, we penalize the model for every added attribute by:

$$\mathcal{L}_{select}=\frac{1}{D}\sum_{j=1}^D m_j.$$
D
$\blacksquare$
j=1
Ltotal = Ltask + λL*select*,

## 3.3 Task Network

Let us summarize the training algorithm described in Algorithm 1. Training starts with a fixed temperature τ = τ0 and a randomly initialized embedding e. Given an embedding e, the masking network f returns a mask vector m = (m1*, . . . , m*D) using the Gumbel-Sigmoid functions. Each continuous mask vector m sampled from Gumbel-Sigmoid is then applied to a mini-batch B to construct the reduced vectors xm = m ⊙ x, for x ∈ B. This vector goes to the task network g, which returns the response for a given task g(xm). The loss function L*total* is calculated and the gradient is propagated to: (1) embedding vector e, (2) weights of f and g. In particular, by learning the embedding vector e and the parameters of f, we optimize the mask vector. A critical aspect of our algorithm is the temperature annealing schedule. We begin with a high temperature (τ = 2.0), which produces soft masks that allow gradient flow to all features. As Algorithm 1 AutoNFS training procedure for classification

## 3.5 Feature Importance Quantification

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 1: **Input:** Dataset D = {(xi, yi)}
N
i=1, batch size B, initial temperature τ0 = 2.0, decay rate α = 0.997, total epochs E, FS balance parameter λ 2: **Initialize:** Embedding vector e ∈ R
de, masking network fϕ, task network gθ 3: τ ← τ0 4: for epoch = 1 to E do 5: for each mini-batch B = {(xi, yi)}
B
i=1 ⊂ D do 6: w ← fϕ(e) ▷ Compute logits for feature mask 7: g ← − log(− log(u)), where u ∼ Unif(0, 1) ▷ Sample Gumbel noise 8: m ← σ ((w + g)/τ ) ▷ Generate mask via Gumbel-Sigmoid 9: X ← {xi}
B i=1 10: Xmasked ← X ⊙ m ▷ Mask input features 11: Yˆ ← gθ(Xmasked) ▷ Forward pass through task network 12:
13: Ltask ← −PB
i=1 PC
c=1 yi,c log(ˆyi,c)
14: Lselect ← 1D
PD
j=1 mj 15: Ltotal ← Ltask + λ · Lselect 16:
17: e ← e − η1∇eLtotal ▷ Update embedding 18: ϕ ← ϕ − η1∇ϕLtotal ▷ Update masking network 19: θ ← θ − η2∇θLtotal ▷ Update task network 20: **end for**
21: τ ← τ · α ▷ Anneal temperature 22: **end for** training progresses, the temperature decays exponentially (typically with α = 0.997), causing the masks to become increasingly binary. This gradual transition serves multiple purposes:
- It allows the network to initially explore the full feature space. - It enables progressive commitment to more discrete FS decisions. - It leads to convergence on a nearly binary FS mask at the end of training.

The annealing process effectively functions as a curriculum, starting with easier optimization (continuous selection) and progressively transitioning to harder optimization (discrete selection). This process is related to exploration-exploitation trade-off, which parallels fundamental concepts in reinforcement learning (see Appendix B detailed discussion). After training, we quantify the importance of each feature by directly applying the learned selection mechanism with hard Gumbel-Sigmoid activation:
1. Calculating the feature logits of the trained embedding: w = fϕ(e).

2. Applying a hard threshold, that is, if σ(wi) > 0.5, then mi = 1, else mi = 0. 3. Interpreting the resulting binary vector m = (m1*, . . . , m*D) as the mask for feature selection.

This process produces a deterministic FS that clearly identifies relevant features for the task. Since our FS mechanism is parameterized by a single embedding vector that is independent of specific input examples, the selected features remain constant throughout the dataset. The resulting binary mask can be directly used to filter features, or features can be ranked by their logit values when a specific top-k selection is desired. Importantly, since the selection mechanism was jointly optimized with the task objective, the selected features capture both individual importance and interactive effects relevant to the specific task.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

![5_image_0.png](5_image_0.png) 

## 4 Experiments

To evaluate the effectiveness of AutoNFS, we conducted extensive experiments across multiple datasets (standard OpenML data and high-dimensional metagenomic datasets) and compared our approach with state-of-the-art FS methods. We verify the performance of the model and inspect the importance of selected attributes. Furthermore, we analyze the computational efficiency of our method compared to existing approaches and the influence of the parameter λ on the behavior of the algorithm Appendix F. We also provide further insight into the interpretability of the selected features in the example of MNIST, which can be found in Appendix G.

## 4.1 Benchmark Datasets

Table 1: Summary of datasets (left) and the number of attributes selected by AutoNFS under three considered scenarios (right). It is evident that AutoNFS not only eliminate auxiliary noisy features but also drastically reduces the number of the original attributes.

| Dataset Statistics   | Features Selected by AutoNFS   |            |          |        |           |              |
|----------------------|--------------------------------|------------|----------|--------|-----------|--------------|
| Dataset              | Samples                        | Classes    | Features | Random | Corrupted | Second-order |
| Features             | Features                       | Features   |          |        |           |              |
| AL (aloi)            | 108 000                        | 1000       | 128      | 65     | 65        | 69           |
| CH (california)      | 20 640                         | regression | 8        | 5      | 5         | 3            |
| EY (eye)             | 10 936                         | 3          | 26       | 8      | 11        | 12           |
| GE (gesture)         | 9 873                          | 5          | 32       | 11     | 16        | 22           |
| HE (helena)          | 65 196                         | 100        | 27       | 15     | 14        | 16           |
| HI (higgs_small)     | 98 050                         | 2          | 28       | 14     | 14        | 14           |
| HO (house)           | 22 784                         | regression | 16       | 10     | 10        | 9            |
| JA (jannis)          | 83 733                         | 4          | 54       | 17     | 16        | 18           |
| MI (microsoft)       | 1 200 192                      | regression | 136      | 47     | 61        | 42           |
| OT (otto)            | 61 878                         | 9          | 93       | 78     | 67        | 76           |
| YE (year)            | 515 345                        | regression | 90       | 69     | 28        | 29           |

Experimental setup We follow a recent benchmark introduced in Cherepanova et al. (2023). The reported results were achieved by extending their code base with AutoNFS.

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 The benchmark consists of three scenarios applied to 11 datasets (see LHS of Table 1). In each, a given dataset is corrupted by adding auxiliary features: (1) fully random features, (2) original features corrupted with Gaussian noise, and (3) a set of second-order features created by multiplying randomly selected features from the original dataset. We analyze a scenario in which 50% of the features in each dataset were artificially created. By applying FS algorithms, we aim to eliminate redundant features without compromising the predictive power of the representation. We compared AutoNFS with 10 established FS methods. All methods use MLP as a downstream classifier. We refer the reader to Appendix C for further details of the experimental setup. For each dataset and method, we compute performance metrics specific to the task (accuracy for classification, negative mean squared error for regression). We also report the mean rank across datasets to provide an overall performance assessment. Predictive performance The ranking summary of the results presented in Figure 2 shows an impressive performance of AutoNFS in each scenario. While the highest advantage of AutoNFS is observed for the case of features corrupted by Gaussian noise (average rank 2.1), in the remaining two scenarios (random and second-order features) AutoNFS still achieves the best ranks, beating the next competitors by 0.9 and 0.7 ranking points, respectively. It is important to note that all baseline methods select the same number of features as were in the initial representation (before corruption),
whereas our method automatically chooses a much smaller subset of the most relevant features, see the RHS of Table 1. As a result, AutoNFS consistently achieves competitive or superior performance while using significantly fewer features, highlighting its practical advantage. Detailed results presented in Tables 3 to 5 show that our algorithm obtains the highest or joint-highest scores on most datasets, demonstrating consistent and strong performance. Analysis of selected features In addition to predictive performance on downstream tasks, we analyze how the selected attributes match the original features (before adding auxiliary features). Figure 3a shows that AutoNFS achieves zero misselection errors for random and corrupted features and maintains low error rates of 0.17 for second-order features. It is important to note that the selection of features outside the original attributes in the latter case is acceptable since additional features were created by multiplying the original features. In consequence, these extra features may sometimes carry even more information than the individual original attributes. The application of the representation created by the baseline methods resulted in significantly higher misselection errors. Figure 3b presents the average predictive power of the individual features. More precisely, we measure how much predictive performance decreases when we remove one of the selected features. As can be seen, the average decrease for AutoNFS is equal to 0.313, which means that the returned set cannot be further reduced without affecting predictive performance. This demonstrates the superior precision of AutoNFS in identifying relevant features while automatically determining the optimal number to select. In general, these findings confirm that AutoNFS is broadly applicable to a wide range of machine learning tasks, including both classification and regression, while offering strong and reliable performance in various feature noise scenarios.

## 4.2 Metagenomic Dataset Analysis

To evaluate AutoNFS's effectiveness in real-world high-dimensional biological data, we applied it to 24 metagenomic datasets obtained from Curated Metagenomics Data Pasolli et al. (2017). These datasets represent a particularly challenging domain with high feature dimensionality (308-718 features) and complex biological interactions. In this experiment, we additionally verify how the constructed representation is useful for two types of downstream classifiers: MLP and Random Forest
(RF).

The results presented in Table 2 demonstrate that, on average, AutoNFS maintains predictive performance on downstream tasks while drastically reducing feature dimensionality (AutoNFS selected only 7.7% of the original features). In the case of MLP, AutoNFS achieved 0.7 improvements in pp accuracy, while for RF the improvement increased to 1.2 pp. This means that the high predictive performance of the representation generated by AutoNFS is independent of a downstream classifier.

378

![7_image_0.png](7_image_0.png) 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_1.png](7_image_1.png)

Table 2: Performance on metagenomic data reduced with AutoNFS. Although AutoNFS heavily reduces data dimensionality, it does not lead to the deterioration of the results on average. Each dataset's name is derived from the first author's surname and the year of publication.

MLP on MLP on RF on RF on Original Reduced

dataset full data AutoNFS full data AutoNFS dim. dim. NielsenHB_2014 0.613 0.643 **0.711** 0.634 370 33 WirbelJ_2018 0.558 **0.571** 0.776 **0.821** 639 32 KeohaneDM_2020 **0.469** 0.344 0.469 **0.531** 540 37 JieZ_2017 **0.693** 0.612 0.762 **0.770** 308 61 FengQ_2015 **0.662** 0.607 0.833 **0.889** 575 25 ThomasAM_2019c 0.582 **0.664** 0.627 **0.764** 438 32 LiJ_2017 0.341 0.511 **0.561** 0.432 651 43 ZellerG_2014 0.614 0.614 0.652 **0.871** 645 23 LifeLinesDeep_2016 0.513 0.546 **0.500 0.500** 526 79 ThomasAM_2018b **0.686** 0.614 **0.586 0.586** 621 31 HanniganGD_2017 0.467 0.633 **0.817** 0.533 477 22 YachidaS_2019 0.471 0.570 **0.636** 0.608 480 88 ZhuF_2020 **0.657** 0.559 **0.768** 0.739 718 33 ThomasAM_2018a **0.733** 0.567 0.817 **0.917** 292 24 LiJ_2014 0.454 **0.490** 0.500 **0.508** 503 46 LeChatelierE_2013 **0.551** 0.521 0.549 **0.620** 646 51 QinN_2014 0.746 **0.815** 0.833 **0.855** 652 38 QinJ_2012 0.551 **0.561** 0.616 **0.622** 436 59 NagySzakalD_2017 0.521 **0.583** 0.917 **0.958** 519 21 YuJ_2015 **0.653** 0.417 **0.674** 0.646 606 34

GuptaA_2019 0.812 **0.938** 0.875 **0.938** 683 19 VogtmannE_2016 0.667 0.681 **0.694 0.694** 381 38

AsnicarF_2021 0.503 0.528 **0.500 0.500** 537 90 RubelMA_2020 0.607 **0.717** 0.775 **0.796** 606 26 average 0.588 0.596 0.685 0.697 535 41

Figure 5 illustrates the process of FS. Observe that AutoNFS deeply explores the space of all features in the training phase and selects the final set of features at the end of the training.

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

![8_image_0.png](8_image_0.png) 

## 4.3 Computational Complexity Estimation

The estimated computational complexity reveal striking differences between FS methods, see Figure 4a. Denoting time complexity as an exponential function of the number of features t ≈ Dα, our empirical analysis shows that AutoNFS demonstrates near-constant time scaling (α ≈ 0.08). Conventional FS methods, such as the ANOVA F value and Mutual Informatio, exhibit linear scaling
(α ≈ 1.0), while Random Forest FS shows sublinear behavior (α ≈ 0.53). In contrast, Recursive Feature Elimination with Linear SVC demonstrates superlinear scaling (α ≈ 1.41), causing its performance to degrade more rapidly with increasing feature dimensions. The confidence intervals over 5 runs (Figure 4b) indicate that these estimates are statistically robust across the dimensionalities tested. This assessment provides compelling evidence for the exceptional efficiency advantage of AutoNFS in high-dimensional FS tasks, with its nearly constant-time behavior representing a significant algorithmic advancement over conventional methods.

## 5 Conclusion

We presented AutoNFS, a novel neural architecture for FS in a differentiable end-to-end manner using temperature-controlled Gumbel-Sigmoid sampling. The key innovation lies in its ability to automatically determine not only which features are relevant but also how many features should be retained, a common pain point in traditional FS methods. Whereas most existing techniques require the number of selected features to be manually specified or found through expensive hyperparameter tuning, AutoNFS learns this quantity during training. Experimental results in synthetic benchmarks and real-world datasets demonstrate that AutoNFS consistently selects fewer features than baselines, without compromising predictive performance. This reduction is beneficial in terms of computational efficiency and interpretability, but also validates the model's ability to avoid overfitting by ignoring redundant or noisy inputs. Looking ahead, this automatic feature count discovery opens doors for broader applications, such as real-time model compression, adaptive inference, or integration with AutoML frameworks. Moreover, the balance between sparsity and accuracy, controlled through a single λ parameter, makes AutoNFS a drop-in replacement for feature selectors in a wide range of tasks. Ethics statement. This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be highlighted here. Reproducibility statement. We have described all the details and hyperparameters of the proposed approach. We include our codebase as an anonymous repository and will publish it along with the paper. LLM statement. The authors used LLM tools to polish the writing.

## References

Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori Koyama. Optuna: A
next-generation hyperparameter optimization framework, 2019. URL https://arxiv.org/ abs/1907.10902.

Muhammed Fatih Balın, Abubakar Abid, and James Zou. Concrete autoencoders: Differentiable feature selection and reconstruction. In Proceedings of the 36th International Conference on Machine Learning, volume 97 of *Proceedings of Machine Learning Research*, pp. 444–453. PMLR, 2019. URL https://proceedings.mlr.press/v97/balin19a.html.

Muhammed Fatih Balın, Abubakar Abid, and James Zou. Concrete Autoencoders: Differentiable Feature Selection and Reconstruction. In Proceedings of the 36th International Conference on Machine Learning, pp. 444–453. PMLR, May 2019. URL https://proceedings.mlr. press/v97/balin19a.html. ISSN: 2640-3498.

Rina Foygel Barber and Emmanuel J. Candès. Controlling the false discovery rate via knockoffs.

The Annals of Statistics, 43(5), October 2015. ISSN 0090-5364. doi: 10.1214/15-aos1337. URL
http://dx.doi.org/10.1214/15-AOS1337.

Mathieu Blondel, Olivier Teboul, Quentin Berthet, and Josip Djolonga. Fast differentiable sorting and ranking. In *International Conference on Machine Learning*, pp. 950–959. PMLR, 2020.

L. Breiman. Random forests. *Machine Learning*, 45:5–32, 2001. URL https://api.

semanticscholar.org/CorpusID:89141.

Gavin Brown, Adam Pocock, Ming-Jie Zhao, and Mikel Luján. Conditional likelihood maximisation: A unifying framework for information theoretic feature selection. *Journal of Machine* Learning Research, 13:27–66, 2012.

Nicolò Cesa-Bianchi, Claudio Gentile, Gábor Lugosi, and Gergely Neu. Boltzmann exploration done right. In *Advances in Neural Information Processing Systems*, volume 30, 2017. URL https://arxiv.org/abs/1705.10257.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov, and Artem Babenko. Revisiting Deep Learning Models for Tabular Data, October 2023.

Isabelle Guyon and André Elisseeff. An introduction to variable and feature selection. Journal of Machine Learning Research, 3:1157–1182, 2003.

Girish Chandrashekar and Ferat Sahin. A survey on feature selection methods. Computers & Electrical Engineering, 40(1):16–28, 2014.

Tianqi Chen and Carlos Guestrin. XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '16, pp. 785–794. ACM, 2016.

Xueyi Cheng. A Comprehensive Study of Feature Selection Techniques in Machine Learning Models. *Artificial Intelligence and Digital Technology*, 1(1):65–78, November 2024.

Valeriia Cherepanova, Roman Levin, Gowthami Somepalli, Jonas Geiping, C Bayan Bruss, Andrew G Wilson, Tom Goldstein, and Micah Goldblum. A performance-driven benchmark for feature selection in tabular deep learning. *Advances in Neural Information Processing Systems*, 36:41956–41979, 2023.

Ian Connick Covert, Wei Qiu, Mingyu Lu, Na Yoon Kim, Nathan J White, and Su-In Lee. Learning to maximize mutual information for dynamic feature selection. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of *Proceedings* of Machine Learning Research, pp. 6424–6447. PMLR, 23–29 Jul 2023.

Pradip Dhal and Chandrashekhar Azad. A comprehensive survey on feature selection in the various fields of machine learning. *Applied intelligence*, 52(4):4543–4581, 2022.

Isabelle Guyon, Jason Weston, Stephen Barnhill, and Vladimir Vapnik. Gene Selection for Cancer Classification Using Support Vector Machines. *Machine Learning*, 46:389–422, January 2002. doi: 10.1023/A:1012487302797.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 1856–1865. PMLR, 2018. URL http://proceedings.mlr.press/v80/
haarnoja18b.html.

Lam Si Tung Ho, Nicholas Richardson, and Giang Tran. Adaptive Group Lasso Neural Network Models for Functions of Few Variables and Time-Dependent Data, December 2021.

Sarthak Jain and Byron C. Wallace. Attention is not Explanation. In Jill Burstein, Christy Doran, and Thamar Solorio (eds.), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp. 3543–3556, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics. doi: 10.18653/v1/N19-1357. URL https://aclanthology. org/N19-1357/.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparametrization with gumble-softmax. In International Conference on Learning Representations (ICLR 2017). OpenReview. net, 2017a.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax, 2017b.

Jaromír Janisch, Tomáš Pevný, and Viliam Lisý. Classification with Costly Features Using Deep Reinforcement Learning. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(01): 3959–3966, July 2019. ISSN 2374-3468. doi: 10.1609/aaai.v33i01.33013959. URL https: //ojs.aaai.org/index.php/AAAI/article/view/4287.

Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization, 2017. URL
https://arxiv.org/abs/1412.6980.

Scott Kirkpatrick, C. Daniel Gelatt, and Mario P. Vecchi. Optimization by simulated annealing.

Science, 220(4598):671–680, 1983. doi: 10.1126/science.220.4598.671.

Ron Kohavi and George H. John. Wrappers for feature subset selection. *Artificial Intelligence*, 97
(1):273–324, December 1997. ISSN 0004-3702.

Igor Kononenko. Estimating attributes: Analysis and extensions of relief. In *European Conference* on Machine Learning, 1994. URL https://api.semanticscholar.org/CorpusID: 8190856.

Wouter Kool, Herke Van Hoof, and Max Welling. Stochastic beams and where to find them: The gumbel-top-k trick for sampling sequences without replacement. In International conference on machine learning, pp. 3499–3508. PMLR, 2019.

Miron B. Kursa and Witold R. Rudnicki. Feature Selection with the Boruta Package. Journal of Statistical Software, 36(11):1–13, 2010. doi: 10.18637/jss.v036.i11. URL https://www. jstatsoft.org/index.php/jss/article/view/v036i11.

Tor Lattimore and Csaba Szepesvári. *Bandit Algorithms*. Cambridge University Press, 2020.

Ismael Lemhadri, Feng Ruan, Louis Abraham, and Robert Tibshirani. LassoNet: A Neural Network with Feature Sparsity, June 2021.

Lihong Li, Wei Chu, John Langford, and Robert E. Schapire. A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th International Conference on World Wide Web, pp. 661–670. ACM, 2010. doi: 10.1145/1772690.1772758.

Yifeng Li, Chih-Yu Chen, and Wyeth W. Wasserman. Deep Feature Selection: Theory and Application to Identify Enhancers and Promoters. Journal of Computational Biology: A Journal of Computational Molecular Cell Biology, 23(5):322–336, May 2016. ISSN 1557-8666. doi: 10.1089/cmb.2015.0189.

Christos Louizos, Max Welling, and Diederik P. Kingma. Learning sparse neural networks through l0 regularization. *ArXiv*, abs/1712.01312, 2017. URL https://api.semanticscholar. org/CorpusID:30535508.

Chao Ma, Sebastian Tschiatschek, Konstantina Palla, Jose Miguel Hernandez-Lobato, Sebastian Nowozin, and Cheng Zhang. EDDI: Efficient Dynamic Discovery of High-Value Information with Partial VAE. In Proceedings of the 36th International Conference on Machine Learning, pp. 4234–4243. PMLR, May 2019. URL https://proceedings.mlr.press/v97/ ma19c.html. ISSN: 2640-3498.

C Maddison, A Mnih, and Y Teh. The concrete distribution: A continuous relaxation of discrete random variables. In *Proceedings of the international conference on learning Representations*. International Conference on Learning Representations, 2017.

Sebastián Maldonado and Richard Weber. A wrapper method for feature selection using Support Vector Machines. *Information Sciences*, 179(13):2208–2217, June 2009.

Nicolai Meinshausen and Peter Buehlmann. Stability Selection, May 2009. URL http://
arxiv.org/abs/0809.2932. arXiv:0809.2932 [stat].

Edoardo Pasolli, Lucas Schiffer, Paolo Manghi, Audrey Renson, Valerie Obenchain, Duy Tin Truong, Francesco Beghini, Faizan Malik, Marcel Ramos, Jennifer B Dowd, Curtis Huttenhower, Martin Morgan, Nicola Segata, and Levi Waldron. Accessible, curated metagenomic data through ExperimentHub. *Nat. Methods*, 14(11):1023–1024, oct 2017.

Hanchuan Peng, Fuhui Long, and Chris Ding. Feature selection based on mutual information:
Criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on Pattern Analysis and Machine Intelligence, 27(8):1226–1238, 2005.

Liudmila Prokhorenkova, Gleb Gusev, Aleksandr Vorobev, Anna Veronika Dorogush, and Andrey Gulin. CatBoost: unbiased boosting with categorical features, January 2019. URL http:// arxiv.org/abs/1706.09516. arXiv:1706.09516 [cs].

Pavel Pudil, Jana Novovicová, and Josef Kittler. Floating search methods in feature selection. Pattern Recognit. Lett., 15:1119–1125, 1994. URL https://api.semanticscholar.org/ CorpusID:270333833.

Francesco Quinzan, Rajiv Khanna, Moshik Hershcovitch, Sarel Cohen, Daniel Waddington, Tobias Friedrich, and Michael W. Mahoney. Fast feature selection with fairness constraints. In Francisco Ruiz, Jennifer Dy, and Jan-Willem van de Meent (eds.), Proceedings of The 26th International Conference on Artificial Intelligence and Statistics, volume 206 of *Proceedings of* Machine Learning Research, pp. 7800–7823. PMLR, 25–27 Apr 2023.

David N. Reshef, Yakir A. Reshef, Hilary K. Finucane, Sharon R. Grossman, Gilean McVean, Peter J. Turnbaugh, Eric S. Lander, Michael Mitzenmacher, and Pardis C. Sabeti. Detecting novel associations in large data sets. *Science (New York, N.Y.)*, 334(6062):1518–1524, December 2011. ISSN 1095-9203. doi: 10.1126/science.1205438.

Marko Robnik-Sikonja and Igor Kononenko. Theoretical and Empirical Analysis of ReliefF and RReliefF. *Machine Learning*, 53:23–69, October 2003. doi: 10.1023/A:1025667309714.

Yaniv Romano, Matteo Sesia, and Emmanuel Candès. Deep knockoffs. Journal of the American Statistical Association, 115(532):1861–1872, October 2019. ISSN 1537-274X. doi: 10.1080/01621459.2019.1660174. URL http://dx.doi.org/10.1080/01621459. 2019.1660174.

Sofia Serrano and Noah A. Smith. Is attention interpretable? In Anna Korhonen, David Traum, and Lluís Màrquez (eds.), Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pp. 2931–2951, Florence, Italy, July 2019. Association for Computational Linguistics. doi: 10.18653/v1/P19-1282. URL https://aclanthology.org/P19-1282/.

Noah Simon, Jerome H. Friedman, Trevor J. Hastie, and Robert Tibshirani. A sparse-group lasso.

Journal of Computational and Graphical Statistics, 22:231 - 245, 2013. URL https://api. semanticscholar.org/CorpusID:2208574.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647