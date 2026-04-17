000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Active Learning For Flow Matching Model In Shape Design: A Perspective From Continuous Condition Dataset

Anonymous authors Paper under double-blind review

## Abstract

Although the flow matching model has demonstrated powerful capabilities in modern machine learning, its training notoriously relies on an incredibly large scale of high-quality labeled samples. Nevertheless, the acquisition of highquality labeled datasets is hindered by exorbitant labeling costs in certain fields, notably medical imaging and numerical simulation. Therefore, selecting the most informative samples for training at minimal cost poses a key challenge in these fields. This issue constitutes a central topic in active learning, a subfield of machine learning dedicated to maximizing model performance while minimizing annotation cost. The central challenge involves developing an optimal query strategy to acquire the most informative data samples with minimal labeling effort. This paper presents a pilot study that investigates the application of active learning, which traditionally explored within the context of discriminative models, to flow matching models. By analyzing flow matching models through a piecewise-linear neural network framework, this work elucidates how individual data points influence the diversity and accuracy of the model. Leveraging this analytical framework, we propose two distinct query strategies: one aimed at enhancing model diversity, and the other designed to improve model accuracy. We demonstrate that these two strategies are inherently conflicting, providing a partial explanation for the fundamental trade-off between diversity and accuracy in flow matching models from a dataset perspective. Furthermore, we introduce a mixed strategy that combines both strategies through a weighted mechanism, enabling adjustable control over the diversity-accuracy trade-off by tuning the corresponding weights.

Extensive experiments validate the effectiveness of our approach, showing that the proposed query strategies outperform those designed for discriminative models.

## 1 Introduction

Recently, flow matching models achieve state-of-the-art performance in image and various other generating tasks (Dhariwal & Nichol (2021); Ho et al. (2022); Saharia et al. (2022)) and are one of the fundamental building blocks of the more advanced image and video synthesis systems, e.g., DALL-E-3 (Ramesh et al. (2022)) and Veo3 (Esser et al. (2023)). The success of these models is attributed primarily to the availability of large-scale, high-quality labeled training datasets. However, the acquisition of high-quality labeled datasets is notoriously challenging in some domains due to exorbitant annotation costs. This is particularly true in fields like medical imaging Budd et al. (2021) and numerical simulation Wu et al. (2024), where the cost of obtaining labels far exceeds that of data acquisition. For instance, in medical imaging, the cost of annotating images by expert radiologists significantly exceeds the initial image acquisition cost. Similarly, in automotive engineering, while generating raw simulation models is relatively inexpensive, obtaining high-fidelity numerical simulation results, which require extensive validation and expert interpretation, entails substantially greater effort and expense. So a fundamental challenge in these fields is to select the most informative samples for labeling while minimizing cost. This problem defines the core mission of active learning, a machine learning subfield dedicated to maximizing model performance under constrained annotation resources by developing optimal query strategies.

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 The most common Active Learning strategies include uncertainty based sampling Ren et al. (2021); Li et al. (2024), query by committee Seung et al. (1992), and representation-based sampling Geifman & El-Yaniv (2017); Sener & Savarese (2017), etc. The core principle guiding these methods is to identify and query the most valuable samples to improve the model's decision boundary. Meanwhile, a parallel research direction explores the integration of generative models within the active learning framework. For example, GAAL Zhu & Bento (2017); Lan et al. (2024) proposed employing generative networks for data augmentation. However, its randomly generated samples do not necessarily yield higher informativeness than those in the original dataset. In contrast, BGADL Tran et al. (2019) simultaneously trains both a generative network and a classifier to produce samples within uncertain or disagreement regions. Subsequent methods, including VAAL Sinha et al. (2019) and TAVAAL Kim et al. (2021), further extended this concept by leveraging adversarial learning frameworks to enhance data augmentation and improve feature representation. However, these methods primarily focus on "generative models for active learning", rather than "active learning for generative models". In other words, their main objective is to boost the performance of discriminative models. Consequently, active learning specifically designed for generative models has received limited attention. For example, GALISPZhang et al. (2024) consider "subject of interest" which transforming the open querying problem in the label space into a semi-open one. Specifically, they design and test algorithms on a set of specific labels rather than in the entire label space.

In this paper, we discuss the generalization error of generative models in a manner analogous to that of discriminative models Sugiyama (2015). Specifically, we focus on the generation results across the entire condition space rather than under specific conditions. To conduct such analysis, we propose an analysis framework based on piecewise-linear neural networks Montufar et al. (2014); ´ Goujon et al. (2024), which helps us analyze the generation results of flow matching models. Specifically, we assume the flow matching model's neural network is piecewise-linear. Analyzing the generalization performance of closed-form flow matching models Scarvelis et al. (2023); Chen (2025) by this framework, we establish the generalization mechanisms of flow matching models and obtained the pattern of how data affects diversity and accuracy. Our analysis reveals that data with the same label in the dataset contributes to the diversity of the model, while data with different labels in the dataset contributes to the accuracy of the model. Our findings elucidate the fundamental diversity-accuracy trade-off inherent in dataset composition. Guided by this insight, we formulate two targeted sampling strategies designed to augment diversity and accuracy individually. Furthermore, we demonstrate that a weighted integration of these antagonistic strategies provides a practical means to navigate this trade-off and balance both performance metrics. Finally, we evaluated our query strategies on a synthetic dataset and three real-world shape design tasks. Shape design is an application of generative models. In this context, models are given continuous performance requirements (acting as labels) and are tasked with producing a corresponding design shape Heyrani Nobari et al. (2021). In addition, numerical solvers are used to accurately obtain labels for generated shapes, eliminating the need for manual annotation. The results demonstrate that our query strategy surpasses classical strategies designed for discriminative models in achieving either diversity or accuracy. Moreover, by strategically weighting these query strategies, we enable the formulation of tailored approachs that navigate the trade-off between diversity and accuracy.

The key contributions of this work are summarized as follows: 1) **Flow Matching Model Analysis Framework**: We introduce a novel analytical framework for flow matching models that leverages piecewise-linear neural networks and closed-form flow matching models, enabling rigorous theoretical characterization. This approach elucidates how individual data points influence the model's diversity and accuracy. 2) **Efficient Query Strategy for Active Learning**: Leveraging the proposed analytical framework, we present a pilot study on the application of active learning to flow matching models, introducing two novel query strategies: one aimed at enhancing model diversity and the other at improving model accuracy. These strategies represent competing objectives, underscoring the inherent tradeoff between diversity and accuracy from a data-centric perspective. 3) **Experimental Validation**: Experiments on multiple datasets demonstrate that the two proposed query strategies outperform the direct use of standard active learning method designed for discriminative models in terms of diversity and accuracy, respectively. Additionally, a weighted combination of the two strategies can be formed to create a hybrid query approach, allowing for a tunable tradeoff between diversity and accuracy by adjusting the corresponding weights.

## 2 Methodology 2.1 Problem Definition

In the pool-based active learning method, we define U
n = {X,Y } as an unlabeled dataset with n samples where where x ∈ X, y ∈ Y . L
m = {X , Y} is the current labeled training set with m samples, where x ∈ X , y ∈ Y. Our goal is to design a query strategy QD (U
n QD −−→ L
m)
to maximize the diversity score of the model, and a design query strategy QA (U
n QA −−→ L
m) to maximize the accuracy score of the model.

## 2.2 Piecewise-Linear Analysis Framework

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 In this paper, we leverage specific characteristics of neural networks to analyze the flow matching model, rather than analyzing the complex networks themselves. Particularly, this investigation centers on continuous and piecewise-linear neural networks (CPWL NNs)Montufar et al. (2014); ´ Goujon et al. (2024). The fundamental concept is that the neural networks can be formulated as piecewise-linear functions. Furthermore, researchers investigated the condensation phenomenon of neural networkLuo et al. (2021); Xu et al. (2025). They pointed out that under certain conditions, such as when using dropout or small initialization, the parameters of neural networks may undergo condensation. This means that after fully fitting the dataset, the network tends to reduce the number of effective parameters while also decreasing the number of inflection points. As a result, the network exhibits piecewise-linear interpolation behavior. In this paper, we hypothesize that neural networks employed in flow matching also exhibit the property of piecewise-linear interpolation.

Specifically, when condition c0 in the labels of the dataset, the flow field of the closed-form flow matching modelScarvelis et al. (2023); Chen (2025), when consider the optimal transmission noise schedule Lipman et al. (2022):

$$\mathbf{u}_{t}(\mathbf{x}^{\prime},\mathbf{c}_{0})={\frac{\sum_{i}^{m}p_{t,i}\mathbf{e}_{t,i}}{\sum_{i}^{m}p_{t,i}}}$$

$$(2)$$
$$(1)$$

where et,i is the noise that make xito x 0, xiis the data with label c0 in the dataset, m is the number of the data with label c0 in the dataset, pt,i is the probability density of et,i. Eq1 means the vector field is a linear combination of data in a dataset.

When condition c
∗is not in the labels of the dataset, the output of the neural network is defined as the interpolation of the the output of the neural network of the conditions near c
∗:

$$\mathbf{u}_{t}(\mathbf{x}^{\prime},a_{0}\mathbf{c}_{0}+a_{1}\mathbf{c}_{1}+...+a_{k}\mathbf{c}_{k})=a_{0}\mathbf{u}_{t}(\mathbf{x}^{\prime},\mathbf{c}_{0})+a_{1}\mathbf{u}_{t}(\mathbf{x}^{\prime},\mathbf{c}_{1})+...+a_{d}\mathbf{u}_{t}(\mathbf{x}^{\prime},\mathbf{c}_{d})$$
0, cd) (2)
$$\{\mathbf{x}^{*}|\mathbf{x}^{*}=a_{0}\mathbf{x}_{i}+a_{1}\mathbf{x}_{j}+...+a_{d}\mathbf{x}_{k}\}$$
∗ = a0xi + a1xj + ... + adxk} (3)
where xiis data with label c0 in the dataset, xj is data with label c1 in the dataset, xk is data with label cd in the dataset, etc.

$$({\mathfrak{I}})$$

where c
∗ = a0c0 +a1c1 +...+adcd. a0, a1, and ad are interpolation coefficients. c0, c1 and cd are the labels that exist in the dataset. c ∈ R
z, z = d, the label space is divided into several sub regions, each sub region being a convex hull with d+1 vertices. [a0, a1*, ..., a*d] can be easily calculated using the label [yi, yj , ..., yk] of [xi, xj *, ...,* xk]. Because d + 1 points form a d-dimensional plane.

Under this assumption, for any given condition c (c exists in the dataset), the flow matching model is constrained to output only the corresponding sample from the dataset Gu et al. (2023). Besides, by using *Lemma*1 proven in Appendix A, we know that the vector field in Eq2 will result in the generated sample x
∗, being an interpolation of xi, xj , xk, etc.

Eq3 provides the generation law of the closed-form piecewise-linear flow matching model. Specif-

![3_image_0.png](3_image_0.png) ically, interpolation in the label space results in corresponding interpolation in the data space as illustrated in Fig1a. It is worth noting that the label dimension d is generally smaller than the data dimension, meaning that the interpolation coefficients derived from the labels induce lowerdimensional interpolation in the data space. As a methodological note, the generated samples are provided without accounting for their respective generation probabilities. The probability of certain samples can be very small and even zero, as it is inherently contingent on the input condition, labels, and the characteristics of the data distribution. Therefore, Eq3 establishes an upper bound on the diversity of generated samples. Figure 1: Comparison of different point addition strategies and their generated samples, with the black line indicating the possible generated samples.

## 2.3 Diversity From Dataset

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Building upon the analysis in the previous subsection, we have derived the generation rules of the flow matching model. This understanding facilitates the analysis of how specific data points affect the resulting generated samples. Inspired by the method for estimating the number of samples a model can retrieve from a dataset Dombrowski et al. (2025), we quantify the number of individual sample points that the model can generate under a given condition c
∗. Specifically, we propose to increase the number of such individual samples as a query strategy.

For the sake of simplicity, consider the case of c ∈ R
1and d = 1. As shown in Fig1a, when c
∗ ∈ (c0, c1), and there are m samples labeled as c0 and n samples labeled as c1 in the dataset
(no data labeled between c0 and c1), the maximum generated sample type under condition c
∗is mn. As shown in Fig1b, when adding a new data with label ca ( c0 < ca < c1) to the dataset, the interval (c0, c1) is divided into two segments (c0, ca) and (ca, c1). When c
∗ ∈ (c0, ca), the model will generate up to m types of samples, and when c
∗ ∈ (ca, c1), the model will generate up to n types of samples. Compared to the original dataset, this point adding strategy reduces the number of types of points at each c
∗, thereby decreasing the diversity of the model. Therefore, to increase the diversity of the model, we can only consider adding data labeled c0 or c1. As shown in Fig1c, adding data points labeled c0 will result in the model generating up to (m + 1)n types of samples under condition c
∗ ∈ (c0, c1). While as shown in Fig1d, adding data points labeled c1 will result in the model generating up to m(n + 1) types of samples under condition c
∗ ∈ (c0, c1). Obviously, to increase the number of types of points, we need to balance the number of data labeled c0 and c1 in the dataset.

Through the aforementioned analysis, we can design a query strategy QD that increases model diversity:

$$Q_{D}=\operatorname*{arg\,max}_{\mathbf{x}\in\mathbf{X}}-\alpha d i s t a n c e(\mathbf{y},\mathbf{\mathcal{Y}})+\beta\Delta e n t r o p y+\gamma d i s t a n c e(\mathbf{x},\mathbf{\mathcal{X}})$$

where α, β, γ, are weighting coefficients. ∆*entropy* means the entropy increase of labels brought by new labels. *distance* means distance from data point to dataset, we chose the minimum Euclidean distance in the experiments. Specifically, the minimum distance between a data point and all points in the dataset.

$$(4)$$

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 The QD comprises 3 terms, the first term −*distance*(y, Y) encourages new data points to have labels similar to those in the existing dataset. The preceding analysis prescribes that the labels of new data must be strictly identical to those already in the dataset. However, obtaining such exact matches is typically infeasible in practice. Accordingly, we impose the weaker condition that the labels of new data exhibit sufficient similarity to the labels present in the dataset. For unlabeled data, we employ Radial Basis Function (RBF) Neural Networks for label prediction due to their favorable optimization properties. The second term ∆*entropy* encourages the new data points to promote a more uniform label distribution across the dataset. This entropy corresponds to classification entropy, rather than being computed directly as information entropy. Specifically, we first partition the dataset labels into clusters and then compute the entropy of the label distribution across these clusters. A cluster is defined as a set of data points whose inter-point distances fall below a given threshold. The last term *distance*(x, X ) is inspired by the coreset concept Sener & Savarese (2017).

It encourages the query strategy to select new data points that are farther from the existing dataset once the first two conditions are satisfied, thereby avoiding duplicating data and improving diversity.

## 2.4 Accuracy From Dataset

Through the analysis in subsection 2.2, it can be concluded that interpolation in the condition space leads to corresponding interpolation in the data space. Furthermore, *Lemma*2 provides the error bound of the model within a subregion, given by:

$$|f(\mathbf{x}^{*})-\mathbf{c}^{*}|\leq K m a x||\mathbf{c}_{i}-\mathbf{c}_{j}||^{2}$$
$$({\boldsymbol{S}})$$

∗| ≤ *Kmax*||ci − cj ||2(5)
where f(x) = y represents authentic labels, K is related to f and d. c
∗is the condition, x
∗is the generated sample generated by the model given c
∗. max||ci − cj ||2 means the maximum distance of any two points in the subregion of label space. In Eq5, the upper bound on the error within each subregion is determined by the maximum distance between any two points in the subregion. To reduce the error upper bound, a natural approach is to minimize this maximum distance. Accordingly, within the query strategy aimed at enhancing model accuracy, it is intuitive to select new data points whose labels are farthest from those already present in the dataset, as illustrated in Eq6. Essentially, QA performs the coreset algorithm Sener & Savarese (2017) in the label space.

$$Q_{A}=\operatorname*{arg\,max}_{\mathbf{x}\in\mathbf{X}}d i s t a n c e(\mathbf{y},\mathbf{\mathcal{Y}})$$
$$(6)$$
distance(y, Y) (6)
For unlabeled data, we employ Radial Basis Function (RBF) Neural Networks to infer their corresponding labels. Upon comparing Eq4 and Eq6, it becomes apparent that the two strategies exhibit a fundamental conflict: QD aims to seek new samples with *distance*(y, Y) being smaller, while QA aims to seek new samples with *distance*(y, Y) being larger. In other words, data sharing the same label enhance the model's diversity, whereas data with distinct labels improve its accuracy. This clarifies why diversity and accuracy represent a trade-off from the perspective of dataset composition. Furthermore, Eq4 and Eq6 do not incorporate the trained flow matching model, but instead operate directly on the dataset for data selection. This implies that the available annotation budget can be utilized efficiently by training only the RBF neural networks for label prediction, thereby avoiding the need for repeated training of the flow matching model. Considering that Eq4 solely enhances model diversity while Eq6 only improves model accuracy, a natural extension is to combine these two query strategies to balance the trade-off between diversity and accuracy. This leads to:

$$Q_{h y b r i d}=\omega Q_{D}+(1-\omega)Q_{A}$$
Q*hybrid* = ωQD + (1 − ω)QA (7)
where ω controls the ratio of QD to QA.

As shown in Fig2, the dataset is unevenly distributed in both the data space and the label space. Different query strategies lead to the selection of different new data points. In particular, the coreset method selects data points that ensure a more uniform coverage of the data space. The committee

$$(T)$$

270

![5_image_0.png](5_image_0.png) 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 method selects new data with the greatest output discrepancy among prediction models. Consequently, the selected samples tend to cluster at the edges of the label distribution, such as the regions corresponding to labels 0 and 1. This divergence arises from the distinct extrapolation strategies employed by the different prediction models. QD selects new data by ensuring a uniform distribution of labels across different clusters in the label space, while simultaneously maximizing the distance from the initial data in the data space. QA selects new data such that the labels of the data are uniformly distributed across the label space.

## 3 Experiment 3.1 Dataset And Metrics

For our experiments, we selected datasets with continuous rather than categorical labels. The first is an uneven synthetic dataset, chosen for intuitive visualization of the results. The second dataset is an airfoil dataset from the UIUC library, simulated using computational fluid dynamics solvers; the labels correspond to the lift-to-drag ratio coefficients, i.e., y ∈ R
1. The third dataset is a flying wing dataset simulated using computational fluid dynamics solvers; the labels represent the working condition and the lift coefficient, namely y ∈ R
3 Wang et al. (2025). The fourth dataset is a starship-like dataset Seedhouse (2022), the labels represent the lift coefficient, drag coefficient, pitch moment, and pressure center of the shapes, namely y ∈ R
4. The geometric models, such as airfoils, flying wings, and starships, are readily available; however, acquiring their corresponding labels necessitates extensive numerical simulations Wu et al. (2024). Our evaluation framework is designed to measure diversity and accuracy separately, rather than using a combined metric such as FID Yu et al. (2021). Diversity is quantified by a custom variant of the Vendi score Friedman & Dieng (2022), calculated as the average pairwise Euclidean distance of the generated data points. Accuracy is evaluated by the mean squared error of the real labels of generated samples against the given conditions. The labels in our study are derived from distinct sources depending on the dataset: from an analytically designed function in the case of the synthetic dataset, and from numerical simulations for the physical shape datasets, respectively.

$$d i v e r s i t y\,s c o r e=\int_{\mathbb{Y}}\mathbb{E}||\mathbf{x}_{g e n,i}-\mathbf{x}_{g e n,j}||_{2}\,d\mathbf{c}$$

E||xgen,i − x*gen,j* ||2 dc (8)
$$a c c u r a c y\,s c o r e=\int_{\mathbb{Y}}\mathbb{E}(\mathbf{c}-\mathbf{y}_{g e n,i})^{2}\,d\mathbf{c}$$

2dc (9)
$$({\mathfrak{g}})$$

$$(9)$$

6 324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 In each iteration of these tests, 6% of the data is selected. The initial (0-th) round of data selection is performed randomly for all methods, yielding identical start results. For the committee method, SVR, Random Forest, XGBoost, and RBF neural networks are employed to predict the labels of unlabeled data points; the variance of their predictions is then used as the criterion for selecting new samples. The anchor method operates by first selecting a set of fixed anchor conditions and subsequently choosing new data based on the predictive uncertainty estimated under these specific conditions Zhang et al. (2024).

![6_image_0.png](6_image_0.png)

Fig3 shows the samples generated by the model under different point selection strategies given condition 0.5. The optimal generation result under condition 0.5 is a circle located at the origin. Fig3a shows model trained on initial data points. It can be seen that due to insufficient data, even on the left half, the generated result is not a complete semicircle. Among all methods, QD has the highest diversity, while QA has the smallest diversity.

![6_image_1.png](6_image_1.png)

where xgen denotes a generated sample, ygen denotes its corresponding label. Conceptually, both the diversity (Eq8) and accuracy (Eq9) scores are defined directly on the label space Y, within which the Riemann integration is performed for evaluation. For our experiments, we employed a fully connected neural network with 8 layers and 512 hidden units per layer, using the LeakyReLU activation function. The model was trained with the AdamW optimizer for 4,000,000 steps with a batch size of 512. The learning rate was set to 1e-3 with a decay rate (gamma) of 0.9 applied every 100,000 steps. The model was evaluated over 100 sampling steps.

## 3.2 Results

Fig4 compares the diversity and accuracy across the four datasets. The results indicate that QD
achieves the highest diversity, even outperforming the model trained on the full dataset, although this 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png)

Figure 5: Generated airfoil samples under four different conditions. Each panel shows four distinct shapes corresponding to a single condition.

![7_image_1.png](7_image_1.png)

Figure 6: Generated flying wing samples under four different conditions. Each panel shows four

![7_image_2.png](7_image_2.png) distinct shapes corresponding to a single condition. Fig7 illustrates how the weight ω in Eq7 can be tuned to control the trade-off between diversity and accuracy: a larger ω prioritizes diversity, while a smaller ω favors accuracy. Fig5, Fig6, and Fig8 present a comparison of samples generated by the the model trained under QD and QA query comes at the cost of reduced accuracy. In contrast, QA yields the highest accuracy. The effectiveness of the anchor method is confined to the predefined anchor conditions, and it fails to generalize effectively to conditions outside this set.

![8_image_0.png](8_image_0.png)

strategies across different datasets. The results demonstrate that QD achieves higher diversity at the cost of lower accuracy, whereas QA prioritizes accuracy, resulting in lower diversity.

## 3.3 Ablation Study

The formulation of QD comprises three terms, the assessment of the relative impact of each term in

![8_image_1.png](8_image_1.png)

Fig9 shows that all three positively influence diversity. The *distance*(x, X ) term is identified as the most important factor, whereas the ∆*entropy* term has a comparatively minor effect.

## 4 Conclusion And Discussion

432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 This work tackles active learning for flow matching by first establishing a theoretical foundation via piecewise-linear network and closed-form flow matching models analysis. This framework precisely elucidates the distinct roles of data: label-consistent points drive diversity, while label-varied points bolster accuracy. Capitalizing on this insight, we devise specialized query strategies, one for diversity, the other for accuracy, and a hybrid strategy with adjustable weights to balance them. Comprehensive experiments confirm that our approach surpasses active learning strategies developed for discriminative models. A fundamental characteristic of our approach is its decoupling of the query process from the trained model, relying instead on dataset-level computations. While this allows for efficient allocation of the annotation budget by bypassing the batch-wise process, it also eliminates the need for cumbersome intermediate training cycles. The framework shifts the focus from model-internal diagnostics to data-centric selection, which consequently makes it challenging to directly address or refine the behavioral biases of the final trained model.

## References

Samuel Budd, Emma C Robinson, and Bernhard Kainz. A survey on active learning and human-inthe-loop deep learning for medical image analysis. *Medical image analysis*, 71:102062, 2021.

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 7346–7356, 2023.

Dan Friedman and Adji Bousso Dieng. The vendi score: A diversity evaluation metric for machine learning. *arXiv preprint arXiv:2210.02410*, 2022.

Yonatan Geifman and Ran El-Yaniv. Deep active learning over the long tail. arXiv preprint arXiv:1711.00941, 2017.

Alexis Goujon, Arian Etemadi, and Michael Unser. On the number of regions of piecewise linear neural networks. *Journal of Computational and Applied Mathematics*, 441:115667, 2024.

Xiangming Gu, Chao Du, Tianyu Pang, Chongxuan Li, Min Lin, and Ye Wang. On memorization in diffusion models. *arXiv preprint arXiv:2310.02664*, 2023.

Amin Heyrani Nobari, Wei Chen, and Faez Ahmed. Pcdgan: A continuous conditional diverse generative adversarial network for inverse design. In *Proceedings of the 27th ACM SIGKDD* conference on knowledge discovery & data mining, pp. 606–616, 2021.

Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David J
Fleet. Video diffusion models. *Advances in Neural Information Processing Systems*, 35:8633–
8646, 2022.

Kwanyoung Kim, Dongwon Park, Kwang In Kim, and Se Young Chun. Task-aware variational adversarial active learning. In *Proceedings of the IEEE/CVF conference on computer vision and* pattern recognition, pp. 8166–8175, 2021.

Guipeng Lan, Shuai Xiao, Jiachen Yang, Jiabao Wen, Wen Lu, and Xinbo Gao. Active learning inspired method in generative models. *Expert Systems with Applications*, 249:123582, 2024.

Dongyuan Li, Zhen Wang, Yankai Chen, Renhe Jiang, Weiping Ding, and Manabu Okumura. A
survey on deep active learning: Recent advances and new frontiers. *IEEE Transactions on Neural* Networks and Learning Systems, 36(4):5879–5899, 2024.

Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. *arXiv preprint arXiv:2210.02747*, 2022.

Tao Luo, Zhi-Qin John Xu, Zheng Ma, and Yaoyu Zhang. Phase diagram for two-layer relu neural networks at infinite-width limit. *Journal of Machine Learning Research*, 22(71):1–47, 2021.

Guido Montufar, Razvan Pascanu, Kyunghyun Cho, and Yoshua Bengio. On the number of linear ´
regions of deep neural networks. *Advances in neural information processing systems*, 27, 2014.

Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical textconditional image generation with clip latents. *arXiv preprint arXiv:2204.06125*, 1(2):3, 2022.

Pengzhen Ren, Yun Xiao, Xiaojun Chang, Po-Yao Huang, Zhihui Li, Brij B Gupta, Xiaojiang Chen, and Xin Wang. A survey of deep active learning. *ACM computing surveys (CSUR)*, 54(9):1–40, 2021.

Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. *Advances* in Neural Information Processing Systems, 34:8780–8794, 2021.

Mischa Dombrowski, Weitong Zhang, Sarah Cechnicka, Hadrien Reynaud, and Bernhard Kainz.

Image generation diversity issues and how to tame them. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 3029–3039, 2025.

Zhengdao Chen. On the interpolation effect of score smoothing. *arXiv preprint arXiv:2502.19499*,
2025.

Christopher Scarvelis, Haitz Saez de Oc ´ ariz Borde, and Justin Solomon. Closed-form diffusion ´
models. *arXiv preprint arXiv:2310.12395*, 2023.

Erik Seedhouse. Starship. In *SpaceX: Starship to Mars–The First 20 Years*, pp. 171–188. Springer, 2022.

Ozan Sener and Silvio Savarese. Active learning for convolutional neural networks: A core-set approach. *arXiv preprint arXiv:1708.00489*, 2017.

Samarth Sinha, Sayna Ebrahimi, and Trevor Darrell. Variational adversarial active learning. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 5972–5981, 2019.

Masashi Sugiyama. *Introduction to statistical machine learning*. Morgan Kaufmann, 2015.

Toan Tran, Thanh-Toan Do, Ian Reid, and Gustavo Carneiro. Bayesian generative active deep learning. In *International conference on machine learning*, pp. 6295–6304. PMLR, 2019.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 We use the mathematical notation of the flow matching model, x0 ∈ N(0, I), x1 is data in dataset. ut is the flow field at time t. Lemma 1. *In closed-form flow matching model, if* ut(x 0, a0c0+a1c1+...+adcd) = a0ut(x 0, c0)+
a1ut(x 0, c1) + ... + adut(x 0, cd)*, then the flow matching model will generate the data in* {x
∗|x
∗ =
a0xi + a1xj + ... + adxk} *when given* c
∗ = a0c0 + a1c1 + ... + adcd. xiis the data generated by the model when given c0, xj is the data generated by the model when given c1, and xk is the data generated by the model when given cd, etc. The dataset contains data labeled as c0, c1, cd*, etc.* Chitwan Saharia, Jonathan Ho, William Chan, Tim Salimans, David J Fleet, and Mohammad Norouzi. Image super-resolution via iterative refinement. IEEE transactions on pattern analysis and machine intelligence, 45(4):4713–4726, 2022.

Yueqing Wang, Peng Zhang, Yushuang Liu, Jianing Zhao, Jie Lin, and Yi Chen. Aerodynamic coefficients prediction via cross-attention fusion and physical-informed training. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 869–876, 2025.

Haixu Wu, Huakun Luo, Haowen Wang, Jianmin Wang, and Mingsheng Long. Transolver: A fast transformer solver for pdes on general geometries. *arXiv preprint arXiv:2402.02366*, 2024.

Zhi-Qin John Xu, Yaoyu Zhang, and Zhangchen Zhou. An overview of condensation phenomenon in deep learning. *arXiv preprint arXiv:2504.09484*, 2025.

Yu Yu, Weibin Zhang, and Yun Deng. Frechet inception distance (fid) for evaluating gans. *China* University of Mining Technology Beijing Graduate School, 3(11), 2021.

Xulu Zhang, Wengyu Zhang, Xiaoyong Wei, Jinlin Wu, Zhaoxiang Zhang, Zhen Lei, and Qing Li.

Generative active learning for image synthesis personalization. In Proceedings of the 32nd ACM International Conference on Multimedia, pp. 10669–10677, 2024.

Jia-Jie Zhu and Jose Bento. Generative adversarial active learning. ´ arXiv preprint arXiv:1702.07956, 2017.

## A Mathematical Proof

Proof. In closed-form flow matching model, the generated data is entirely from the dataset. Consider the optimal transmission path xt = (1 − t)x0 + tx1, and the loss function P||ut(xt, c) −
[(1 − t)x0 + tx1])||2. The flow field at condition c0 is:

$$\mathbf{u}_{t}(\mathbf{x}^{\prime},\mathbf{c}_{0})={\frac{\sum_{i}^{m}p_{t,i}\mathbf{e}_{t,i}}{\sum_{i}^{m}p_{t,i}}}$$
(10)
H Sebastian Seung, Manfred Opper, and Haim Sompolinsky. Query by committee. In Proceedings of the fifth annual workshop on Computational learning theory, pp. 287–294, 1992.

$$\mathbf{e}_{t,i}={\frac{\mathbf{x}^{\prime}-t\mathbf{x}_{1,i}}{1-t}}$$
$$\mathbf{e}_{t,i}={\frac{\mathbf{x}^{\prime}-t\mathbf{x}_{1,i}}{1-t}}$$ $$p_{t,i}(\mathbf{x}_{1},\mathbf{x}^{\prime})={\frac{1}{(2\pi)^{{\frac{d}{2}}}|\mathbf{\Sigma}_{t}|^{{\frac{1}{2}}}}}\exp[-{\frac{1}{2(1-t)}}||\mathbf{e}_{t,i}||^{2}]$$
$$(11)$$
$$(12)$$
ut(x 0, a0c0 + a1c1 + ... + adcd) (13) = a0ut(x 0, c0) + a1ut(x 0, c1) + ... + adut(x 0, cd) (14) = a0 Pm i P pt,iet,i m i pt,i + a1 Pn i P pt,jet,j n j pt,j + ... + ad Pok P pt,ket,k o k pt,k = Pm iPn j ...Pok pt,ipt,j ...pt,k(a0et,i + a1et,j + ... + adet,k) Pm i Pn j ...Pok pt,ipt,j ...pt,k(16) = Pm iPn j ...Pok pt,ipt,j ...pt,k (a0+a1+...+ad)x 0−t(a0xi+a1xj+...+adxk) P 1−t m i Pn j...Pokpt,ipt,j ...pt,k(17)
$$(15)$$
$$(13)$$ $$(14)$$
u ∗ t(x 0, c ∗) = Pnm...o l P pt,let,l nm...o lpt,l = Pnm...o lpt,l x 0−t(a0xi+a1xj+...+adxk) P 1−t nm...o lpt,l
$$(18)$$
$$(16)$$
$$(17)$$
$$(19)$$
$$|f(\mathbf{x}^{*})-\mathbf{c}^{*}|\leq K m a x||\mathbf{c}_{i}-\mathbf{c}_{j}||^{2}$$
$$\begin{array}{r l}{\left[a_{0}\quad\cdots\quad a_{d}\right]\quad{\left[\begin{matrix}c_{0,1}&\cdots&c_{0,d}\\ \vdots&\ddots&\vdots\\ c_{d,1}&\cdots&c_{d,d}\end{matrix}\right]}\quad=\ \left[c_{1}^{*}\quad\cdots\quad c_{d}^{*}\right]}\end{array}$$
$$(20)$$
$$\begin{array}{r c l c r c l}{{[a_{0}}}&{{\cdots}}&{{}}&{{a_{d}]}}&{{=}}&{{[c_{1}^{*}}}&{{\cdots}}&{{}}&{{c_{d}^{*}]}}\end{array}$$
$$\begin{array}{r l}{\left[c_{0,1}\quad\cdot\cdot\quad c_{0,d}\right]^{-1}}\\ {\vdots\quad\quad\cdot\cdot\quad\quad\vdots}\\ {c_{d,1}\quad\cdot\cdot\cdot\quad c_{d,d}\right]}\end{array}$$

et,i is the noise that make xito x 0at time t. pt,i(x1, x 0) is probability density of et,i. xiis data with label c0 in the dataset, xj is data with label c1 in the dataset, etc. m is the number of data with label c0, etc, and n is the number of data with label c1 in the dataset, etc.

Thus, While the vector field directly defined on {x
∗|x
∗ = a0xi + a1xj + ... + adxk} is:
u
∗
t(x 0, c
∗) means the model is trained on the interpolation data.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Comparing ut(x 0, a0c0 + a1c1 + ... + adcd) and u
∗ +t (x 0, c
∗), we can see that although the two vector fields are not exactly the same, their final generated results are consistent. The difference lies in the different noise schedules they choose.

Lemma 2. The sample error for the piecewise-linear neural network driven flow matching model is:
Proof. Consider a subregion of the label space, Its vertices are c0, c1,..., cd. There exists a unique set of weight coefficients for c
∗in d-dimensional space.

and we get: