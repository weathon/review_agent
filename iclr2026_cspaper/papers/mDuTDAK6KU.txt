000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053

# Koala: Kl–L0 Adversarial Detector Via Label Agreement

Anonymous authors Paper under double-blind review

## Abstract

Deep neural networks are highly susceptible to adversarial attacks, which pose significant risks to security- and safety-critical applications. We present KOALA (KL–Lo Adversarial detection via Label Agreement), a novel, semantics-free adversarial detector that requires no architectural changes or adversarial retraining. KOALA operates on a simple principle: it detects an adversarial attack when class predictions from two complementary similarity metrics disagree. These metrics—KL divergence and an L0-based similarity—are specifically chosen to detect different types of perturbations. The KL divergence metric is sensitive to dense, low-amplitude shifts, while the L0-based similarity is designed for sparse, high-impact changes. We provide a **formal proof of correctness** for our approach. The only training required is a simple fine-tuning step on a pre-trained image encoder using clean images to ensure the embeddings align well with both metrics. This makes KOALA a lightweight, plug-and-play solution for existing models and various data modalities. Our extensive experiments on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet confirm our theoretical claims. When the theorem's conditions are met, KOALA consistently and effectively detects adversarial examples. On the full test sets, KOALA achieves a precision of 0.94 and a recall of 0.81 on ResNet/CIFAR-10, and a precision of 0.66 and a recall of 0.85 on CLIP/Tiny-ImageNet.

## 1 Introduction

The increasing deployment of machine learning and deep learning models in safety-critical applications—such as autonomous driving, medical imaging, and security—underscores the need for robust and reliable systems. However, neural networks remain vulnerable to adversarial attacks, where small, often imperceptible perturbations to an input can cause the model to make a confident misclassification (Biggio et al., 2013; Xiao et al., 2018a;b; Szegedy et al., 2013). Protecting these models from such manipulation is a critical security and safety concern. Defenses against adversarial attacks generally fall into three categories (Aldahdooh et al., 2022). The first, *verification and certification*, aims to formally prove model robustness within a defined perturbation set (Khedr & Shoukry, 2024; Liu et al., 2021). While these methods provide strong guarantees, they do not actively improve the model's behavior in deployment. The second, *proactive* defenses, such as adversarial training and randomized smoothing, harden models by retraining or modifying their architecture (Madry et al., 2017b; Cohen et al., 2019; Shafahi et al., 2019). These methods can be computationally expensive, often require prior knowledge of attack types, and may lag behind novel attack strategies. The final category, *reactive detection*, augments a deployed model with a separate detector to flag adversarial inputs without altering the core network. We focus on this reactive detection paradigm. Prior work in this area has largely pursued two main avenues. The first involves *add-on detectors*, which rely on empirical observations of adversarial examples, such as their intrinsic statistics or the effects of feature space (Xu et al., 2018; Ma & Liu, 2019; Ma et al., 2018; Meng & Chen, 2017). Other methods train a separate detector head using adversarial examples (Metzen et al., 2017; Grosse et al., 2017). While these methods can be effective, they typically *lack formal guarantees of correctness*. The second involves *semantics-driven detectors* that leverage external information, such as label text, auxiliary classifiers, or handcrafted cues (Zhang et al., 2023; Zhou et al., 2024; Muller et al., 2024). While powerful, these approaches depend on 1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 domain-specific priors that may not always be available or vary across different deployments and data modalities. Critically, they also *lack proof of correctness* for their detection conditions. To address this issue, we present a novel perspective based on the geometry of norm-bounded adversarial perturbations. As shown in Figure 1, we observe that energy-bounded attacks manifest either as (i) *dense, low-amplitude* shifts across many coordinates or (ii) *sparse, high-impact* shifts on few coordinates. These characteristics are naturally captured by two complementary similarity measures: KL divergence, which is sensitive to broad, small-magnitude output shifts; and an L0-based score, which is sensitive to sparse, large-magnitude coordinate changes. In this work, we propose KOALA, a light-weight and semantics-free adversarial detector that flags input as attack when predictions derived from our two complementary metrics—KL divergence and the L0-based score—disagree. The only required training is a brief fine-tuning of an image encoder to align embeddings with both metrics simultaneously. This makes KOALA a simple, plug-and-play solution for existing models without the need for adversarial training or architectural changes.

Our approach is distinguished by a *formal mathematical guarantee*. We prove that under normbounded perturbations and mild assumptions on the separation between class prototypes and the input embedding, each metric induces a distinct prediction stability band. Once the margins between the classes are sufficiently large, no single perturbation can keep the input within both bands simultaneously. This mutual exclusivity forces a disagreement between the two metrics, leading to guaranteed detection. Our extensive experiments on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet corroborate this theory, demonstrating robust attack identification without relying on semantic priors, architectural modifications, or costly adversarial retraining. Our core contributions are summarized as follows: - We introduce KOALA, a novel, plug-in adversarial detector based on the disagreement between KL divergence and L0-based predictions.

- We provide a theoretical proof of correctness that defines the explicit conditions under which this disagreement—and thus detection—is guaranteed to occur.

- We propose a lightweight training recipe that only requires fine-tuning an encoder with clean images, avoiding the need for architectural changes or adversarial examples.

- Our comprehensive experimental results demonstrate strong detection performance, aligning with our theory and offering a valuable complement to existing robust training and certification methods.

Detectors trained with adversarial examples. An intuitive way of train an adversarial detectior is to train it on generated adversarial examples (Metzen et al., 2017; Grosse et al., 2017; Lee et al., 2024). While effective against the attacks seen during training, these detectors typically rely on prior knowledge of the threat model and can degrade under newly crafted or adaptive attacks. Our work is orthogonal to theirs in that our approach does not require adversarial training.

![1_image_0.png](1_image_0.png)

Figure 1: Motivation for combining KL and L0 **as an attack detector.** With an energy bound adversarial input, ∥δ∥2 ≤ ϵ, the resulting perturbation may be *dense* (distributed) or *sparse* (concentrated). Each metric defines a prediction-stability band: inside the band the label remains y
∗; outside it flips to yˆ. Dense attacks typically violate the L0 band (green), while sparse attacks violate the KL
band (orange). When two classification decisions disagree, we can detect adversarial attacks.

## 2 Related Work

Detectors utilizing intrinsic statistics of attacks. Compared to clean samples, adversarial inputs often exhibit systematic statistical deviations designed to fool neural networks. Leveraging this observation, prior work distinguishes clean from adversarial inputs by extract residual and structural information of clean data (Kong et al., 2025) or probing regularities in feature or activation space, e.g., invariant checking over internal activations (NIC) (Ma & Liu, 2019), prediction inconsistency under input transformations (feature squeezing) (Xu et al., 2018), local intrinsic dimensionality (LID) statistics (Ma et al., 2018), autoencoder-based reformers/detectors (MagNet) (Meng & Chen, 2017), Mahalanobis (Lee et al., 2018), CADet (Guille-Escuret et al., 2023), Bayesian-based uncertainty (Feinman et al., 2017), class-disentanglement (Yang et al., 2021) and adversarial direction comparision (Hu et al., 2019). These methods are generally empirical and lack formal proof-ofcorrectness guarantees against adaptive adversaries. While we provide explicit theoretical conditions under which our detector is provably correct, specifying when adversarial examples must be detected.

## 3 Methodology 3.1 The Koala Detector And Koala Head

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161 Semantics- and knowledge-driven detection. Attacks can also be detected by examining semantic inconsistencies at inference using domain knowledge and reasoning modules(Mumuni & Mumuni, 2024), e.g., MLN/GCN pipelines for certifiable robustness (Zhang et al., 2023), knowledge-enabled graph detection (Zhou et al., 2024; Song et al., 2025), and part-level reasoning for object tracking defenses (VOGUES) (Muller et al., 2024). These approaches can be powerful but have limitations across modalities and tasks, as their effectiveness depends on semantics and specific domain knowledge. In contrast, our method is semantics-free: it operates purely on representation geometry via a KL/L0 disagreement criterion and provides detector-specific correctness conditions, yielding a lightweight, plug-in detector for safety-critical models. We consider a neural network classifier comprised of two main components: i) a backbone encoder fθ : I → R
dthat maps input from the data space I (e.g., images) to feature embedding ∈ R
d; ii) a classifier head hθ : R
d → {1*, . . . , m*} uses the embedding to determine the final class.

In a traditional feedforward neural network, the backbone encoder corresponds to all layers up to the penultimate layer, while the classifier head is the final output layer (e.g., a fully connected layer followed by a softmax). Our method, KOALA, replaces this conventional classifier head with a novel component, which we term the KOALA Detector, operates on the embeddings produced by the backbone encoder to simultaneously classify the input and flag it as an attack when necessary. As shown in Figure 2, the KOALA Detector operates as a nearest prototype classifier (Snell et al.,
2017), which determines the predicted class yˆ ∈ {1*, . . . , m*} by finding the prototype vector—the precomputed centroid for each class—that is closest to the input's feature embedding in the normalized feature space, i.e., for feature vector p = fθ(I) of input I, the nearest prototype classifier head

## Yˆ = Arg Min K Distance(Ck, P)

for some **Distance** function and pre-selected prototype vectors (also known as class centroids) c1*, . . . ,* cm. This effectively classifies input based on its proximity to representatives of each class. Traditional nearest prototype classifiers use a single distance metric(e.g., Euclidean) to find the closest class prototype. In contrast, KOALA is designed to leverage multiple, complementary metrics for classification and adversarial detection. The motivation behind KOALA is the observation that adversarial perturbations can manifest in two distinct ways under an energy-limited budget:
- *Sparse, High-Impact Perturbations:* Few feature dimensions are modified with a large magnitude.

- *Dense, Low-Amplitude Perturbations:* Many feature dimensions are modified by small magnitude. These two types of attacks are difficult to detect with a single metric. KOALA addresses this by using a combination of L0 and KL divergence metrics:
- KL Divergence: This metric measures the shift in the output probability distribution. It is particularly sensitive to dense, low-amplitude perturbations that subtly influence the model's overall output, even if no single feature dimension is drastically altered. The KL Divergence is defined as:
162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215
- L0 distance: This metric measures the number of dimensions in the feature vector that have been perturbed above a certain threshold. It is therefore highly sensitive to sparse, high-impact changes, making it effective at detecting targeted, "surgical" attacks. The L0 distance metric is defined as:

$$L_{0}(\mathbf{c},\mathbf{p})\ =\mathbf{card}\Big(\{i:|c_{i}-p_{i}|-\tau\cdot\mu(\mathbf{c},\mathbf{p})>0\}\Big),$$
, (2)
where **card**({.}) denotes the cardinality of the set, µ(c, p) = 1d
Pd
i=1 |ci − pi| is the average
distance across all the entries of |c − p|, and τ ∈ [0, 1] is a threshold parameter. In other words,
the L0 metric counts the number of features whose value are above a certain threshold relative to
the average value of the feature vector.
The KOALA Detector operates by simultaneously leveraging the two complementary metrics above.
For a given input embedding p, the detector computes both the KL-divergence and the L0-based distance to all class prototype vectors ck. These computations yield two distinct class predictions:
$$\hat{y}_{\mathrm{KL}}=\arg\operatorname*{min}_{k}K L(\mathbf{c}_{k},\mathbf{p}),\qquad\hat{y}_{L_{0}}=\arg\operatorname*{min}_{k}L_{0}(\mathbf{c}_{k},\mathbf{p}).$$
L0(ck, p). (3)
The core of our detection mechanism lies in the disagreement between these two predictions. An input is declared attacked when the class predicted by the KL-divergence, yˆKL, does not match the class predicted by the L0-based metric, yˆL0. In this case, the detector abstains from making a final classification. If the two predictions agree, the input is considered benign, and the shared class prediction becomes the final output. This behavior is formally defined by the following decision rule:
(ˆa, yˆ) = (1, ⊥) if yˆL0̸= ˆyKL, else (0, yˆKL). (4)
where ba ∈ {0, 1} is the predicted attack label, with ba = 1 indicating an attack and yb the final predicted class, with ⊥ signifying an abstention (no class).

) = (1, $\perp$) if $\hat{y}_{L_0}\neq\hat{y}$  . 

## 3.2 Theoretical Guarantees

Our proposed method, KOALA, is not merely an empirical defense; it is grounded in a formal mathematical guarantee. We provide a proof of correctness under a set of mild and practical assumptions. The core idea is to show that a single adversarial perturbation cannot simultaneously fool both the KL- and L0-based classifiers.

The following assumptions underpin our main theorem:

![3_image_0.png](3_image_0.png)

Figure 2: **Training phase:** Class centroids C are computed as the centroid of image embeddings within each class. Each image embedding p is compared with C to compute the *Loss*KL and *Loss*L0.

The model is trained to make the L0 and KL distances small for the correct class while large for incorrect classes. **Inference phase:** An input image embedding p is compared with class centroids C
to calculate KL and L0-based predictions yˆKL and yˆL0. The predicted class is accepted only if both metrics agree; otherwise, the system flags the input as an adversarial attack detected (ba = 1).

$$K L(\mathbf{c}\|\mathbf{p})=\sum_{i=1}^{d}c_{i}\,\log{\frac{c_{i}}{p_{i}}}.$$
$$(1)$$
$$(2)$$
i=1
. (1)
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 A1 *Normalized Feature vector space:* All feature embeddings fθ(I) and class prototypes c1*, . . . ,* cm are normalized, i.e., their coordinates sum to 1 and are strictly positive. This is satisfied by using a softmax or similar normalization on the feature vectors.

A2 *Bounded Perturbation:* The adversarial perturbation δ in the feature space has a limited energy budget, i.e., ∥δ∥ ≤ ϵ. This is a standard assumption in adversarial robustness, following from the Lipschitz continuity of the backbone encoder.

A3 *Coordinate-wise Bound:* The magnitude of the perturbation on any single coordinate is bounded relative to the original value, |δi| ≤ 32 |p
∗
i|. This is a mild and practical condition, as extremely large, coordinate-wise perturbations are rarely effective or imperceptible.

A4 *Clean Example Alignment:* On clean, unperturbed inputs, both the KL and L0 metrics agree on the true class. This alignment is encouraged by our lightweight fine-tuning procedure, which shapes the embeddings to be meaningful under both metrics.

Building on these assumptions, our central result is Theorem 1, which establishes that a sufficiently large separation between class prototypes guarantees the detection of adversarial attacks. Theorem 1. If Assumptions A1-A4 are satisfied, and there exists a coordinate i *where the gap* between the true class prototype c
∗
iand the predicted adversarial class prototype cˆiis sufficiently large (i.e., |c
∗
i − cˆi| > Γi(ϵ), for some threshold Γi(ϵ)), then no perturbation δ with ∥δ∥ ≤ ϵ can simultaneously cause both the KL- and L0*-based predictions to favor the adversarial class.* In essence, the theorem proves that the KL and L0 stability bands are mutually exclusive for adversarial perturbations. An attack can push an embedding out of one stability band, causing a prediction flip, but it cannot simultaneously push it out of both. This forces a disagreement, leading to guaranteed detection. This result provides a rigorous foundation for KOALA's effectiveness, showing that if the feature space is properly structured (which our fine-tuning encourages), detection is not a probabilistic outcome but a mathematical certainty. Proof Sketch for Theorem 1: A complete proof of Theorem 1 is provided in the appendix B. Below, we provide a high-level sketch to convey the core intuition behind our guarantee. The proof's central idea is to show that, under a limited energy budget, an adversarial perturbation cannot simultaneously satisfy the conditions required to fool both the KL- and L0-based classifiers. We establish this through three key propositions:
(i) *Necessary Conditions for successful attack on KL-Divergence metric* (Prop. 2): To change the KL-based prediction from the true class prototype c
∗to an adversarial class prototype cˆ,
the adversarial perturbation δ must have a positive inner product with the vector cˆ − c
∗. This condition, means the perturbation must "align" with a particular direction in the feature space.

(ii) Necessary Conditions for successful attack on L0*-metric* Prop. 3): To change the L0 based prediction, the perturbation must alter a minimum number of feature dimensions (k) by a significant amount. This consumes a portion of the total perturbation energy (∥δ∥) allowed by the budget. The more dimensions that need to be flipped, the more energy is consumed, and the less is left for other purposes.

(iii) *The Incompatibility Condition* (Prop. 4): We show that these two conditions are fundamentally incompatible. For any given adversarial perturbation, we can always find a threshold τ for the L0 metric that forces a trade-off. The energy required to satisfy the L0 flip condition (moving a sufficient number of coordinates by a large enough magnitude) leaves insufficient residual energy to satisfy the KL-flip condition (aligning the perturbation with the vector cˆ − c
∗.

(iv) *Conclusion:* The final step proves that such a threshold τ always exists as long as there is a sufficiently large "coordinate gap" between the true class prototype and the adversarial class prototype. This means that if the feature space is well-structured–which our fine-tuning encourages–no single adversarial perturbation can successfully flip both predictions, forcing them to disagree and enabling our detection mechanism.

## 3.3 Fine-Tuning For Prototype Alignment

Our formal guarantees in Theorem 1 rely on the assumption that on clean inputs, the feature embeddings are well-aligned with their respective class prototypes under both KL-divergence and L0-based metrics (Assumption A4). To achieve this, we introduce a lightweight fine-tuning procedure for the backbone encoder fθ. This procedure is designed to simultaneously minimize the distance 270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323

between a clean image embedding and its corresponding class prototype across both metrics, thereby encouraging the "coordinate gap" crucial for our detection method. Our training objective is a composite loss that penalizes the dissimilarity between image embeddings
and their class prototypes. To ensure stable optimization, we first map the KL and L0 distances to a
comparable, differentiable, and range-bounded similarity score. - **KL-similarity loss**: We define the KL-based similarity between a class prototype c and an image embedding p as:
$$\operatorname{sim}_{K L}(\mathbf{c},\mathbf{p})=\exp\bigl(-\operatorname{KL}(\mathbf{c}\|\mathbf{p})\bigr)\in(0,1]$$
Using this similarity, we train the encoder with a standard binary cross-entropy loss over a set of positive and negative image-prototype pairs. This loss encourages the similarity of positive pairs (matching image and prototype) to be high and that of negative pairs (mismatched image and prototype) to be low. Formally, we finetune the model using the following loss function:
LKL = −E(i,j)∈P [y
∗
ij log sij + (1 − y
∗
ij ) log(1 − sij )], where sij = simKL(ci, pj ) (5)
Here, P denotes the set of image-prototype pairs, and y
∗
ij is a binary label (1 for a matching pair, 0 otherwise).

- L0**-similarity loss**: The L0 distance, which counts the number of perturbed dimensions, is nondifferentiable. To make it trainable, we use a smooth, differentiable surrogate. We approximate the L0 metric with a smoothed surrogate function Lc0(c, p) using the sigmoid function to obtain a continuous value.The L0-based similarity is then defined as a normalized, inverse measure of this surrogate:

$$\sin_{L_{0}}({\bf c},{\bf p})\ =\ 1-\widehat{\frac{L_{0}({\bf c},{\bf p})}{d}}\ \in[0,1],\quad\mbox{where}\widehat{L_{0}}({\bf c},{\bf p})\ =\ \sum_{i=1}^{d}\sigma\bigg{(}\frac{|c_{i}-p_{i}|-\tau\cdot\mu({\bf c},{\bf p})}{\phi}\bigg{)}$$

where ϕ > 0 is a smoothness parameter and σ(x) = 1 1+e−x is the sigmoid function. Similar to the KL loss, we use the binary cross entropy loss for L0-based similarity:

$${\mathcal{L}}_{L_{0}}=-\mathbb{E}_{(i,j)\in{\mathcal{P}}}[y_{i j}^{*}\log s_{i j}+(1-y_{i j}^{*})\log(1-s_{i j})],$$
ij ) log(1 − sij )], where sij = simL0(ci, pj ).
- **Total Objective:** The final training objective is a weighted sum of the two similarity losses:

$${\mathcal{L}}_{\mathrm{total}}\ =\ \omega_{L_{0}}\ {\mathcal{L}}_{L_{0}}\ +\ \omega_{K L}\ {\mathcal{L}}_{K L},$$
Ltotal = ωL0 LL0 + ωKL LKL, (6)
where ωL0and ωKL are non-negative mixing weights. This composite loss guides the encoder to produce embeddings that are simultaneously cohesive under both a dense-shift-sensitive metric (KL)
and a sparse-shift-sensitive metric L0, which is a key requirement for KOALA's guaranteed detection.

## 4 Experiments

Our experiments evaluate KOALA's performance on two distinct architectures and datasets, employing standard adversarial attacks to test its effectiveness.

- **Models and Datasets**. We use two models to demonstrate KOALA's versatility: a ResNet-18 model on CIFAR-10 and a CLIP model on Tiny-ImageNet. For both datasets, we randomly split the development sets into two equal halves to serve as the test and validation sets.

- **ResNet-18 on CIFAR-10:** We start with a baseline ResNet-18 backbone trained on CIFAR-
10 (Krizhevsky et al., 2009). The final fully connected layer (classifier head) is removed to produce image embeddings. Class prototypes (centroids) c1*, . . . ,* cm are computed as the mean embedding of all training examples for each class. The backbone is finetuned using the composite loss described in the Fine-Tuning section, with SGD optimizer, learning rate 1 × 10−3, weight decay 5 × 10−4, momentum 0.9, and batch size 128. The loss weights are set to ωL0 = 0.9 and ωKL = 0.1 (as L0 is harder to optimize) and the hyperparameters are τ = 0.75 and ϕ = 0.5.

4.1 EXPERIMENTAL SETUP
324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377

## 4.2 Experiment 1: Verifying Theoretical Guarantees

- **CLIP on Tiny-ImageNet:** We also fine-tune the pre-trained CLIP ViT-B/32 model on the Tiny-
ImageNet dataset. The class prototypes c k here are obtained by using the CLIP text encoder with prompt "a photo of [CLASS]". SGD is used for fine-tuning with learning rate 1 × 10−4, weight decay 0, momentum 0.9, and batch size 128. The loss weights again ωL0 = 0.9 and ωKL = 0.1.

- **Adversarial Attacks:** We generate a variety of adversarial examples using established attack methods. We report results on clean accuracy, adversarial accuracy, and adversarial detection rate. All attacks are constrained by the ℓ∞ norm with ϵ ∈ {2/255, 4/255} and a batch size of 128.

- **PGD (Projected Gradient Descent) (Madry et al., 2017a):** A classic iterative attack used to generate adversarial examples for both the ResNet and CLIP models.

- **CW (Carlini & Wagner, 2017) Attack:** A powerful, optimization-based attack on both models. - **AutoAttack (Croce & Hein, 2020):** A suite of four diverse attacks used to reliably test robustness, serving as a robust benchmark against both models.

- **Experiment Objective:** We validate our central theorem by evaluating KOALA's performance on examples that either satisfy or do not satisfy the conditions of Theorem 1. The primary goal is to show that when the conditions are met, attack detection is guaranteed. We partition the test sets of both CIFAR-10 and Tiny-ImageNet into two groups (i) **Theorem-Compliant Samples:** Inputs that satisfy the conditions of Theorem 1, specifically the sufficient inter-class prototype separation and (ii) Non-Compliant Samples: Inputs that do not satisfy these conditions. Table 1 provides a breakdown of the number of samples (sample size columns) in each group for both datasets, highlighting that the ResNet model on CIFAR-10 exhibits a larger inter-class separation than the CLIP model on Tiny-ImageNet. This is likely due to the massive scale of CLIP's pre-training data, which can lead to a more compact, less-separable embedding space for a smaller, specialized task like Tiny-ImageNet.

- **Evaluation Metrics:** To evaluate detection performance, we define a confusion matrix where an
"attacked" input (i.e., a = 1) is considered a positive result as follows:

TP := $[a=1]\wedge\left[(\widehat{a},\widehat{y})=(1,\bot)\ \vee\ (\widehat{a},\widehat{y})=(0,y^{\star})\right]$, TN := $[a=0]\wedge\left[(\widehat{a},\widehat{y})=(0,y^{\star})\right]$, FP := $[a=0]\wedge\left[(\widehat{a},\widehat{y})=(1,\bot)\ \vee\ (\widehat{a},\widehat{y})=(0,\neg y^{\star})\right]$, FN := $[a=1]\wedge\left[(\widehat{a},\widehat{y})=(0,\neg y^{\star})\right]$.  
Using these definitions, we report standard metrics: Accuracy, Precision, Recall, and F1-score:
Acc = T P +T N
N, Prec = T P
T P +F P , Rec = T P
T P +F N , F1 = 2 Prec Rec Prec+Rec , N = T P +T N +F P +*F N.*
- **Results and Analysis:** Table 1 summarizes the overall performance. Noteably, the recall scores are all 1.0 on the Theorem-compliant subset. This means every adversarial attacked input that satisfies the theorem's conditions is successfully detected, providing strong empirical support for our theoretical guarantee. The Accuracy and precision for theorem-compliant examples are 1.0 as well. This is because the theory assumes that clean, compliant examples are correctly classified by both the KL
and L0 heads, leading to prediction agreement and preventing false alarms.

As our theory predicts, the Theorem-compliant subset achieves a substantially higher Precision and Recall compared to the non-compliant subset, confirming that when the inter-class prototype separation is sufficiently large, adversarial perturbations are forced to cause a disagreement between the KL and L0 heads, leading to more reliable attack detection.

## 4.3 Experiment 2: Ablation Study On Metric Combinations

- **Experiment Objective:** We run an ablation study to validate our choice of using KL-divergence and L0-based metrics for attack detection. We compare the performance of our proposed KL+L0 combination against other plausible metric pairings: L0+Cosine, KL+Cosine, and L0+KL+Cosine. For each combination, we fine-tune the backbone encoder using a composite loss tailored to the specific metrics, then evaluate the detector's performance. It's important to note that all models were fine-tuned exclusively with clean, non-adversarial images. No adversarial training was performed.

- **Results and Analysis:** The results, summarized in Table 2, show that the KL+L0 combination consistently yields the best performance on the ResNet/CIFAR-10 setup, achieving the highest scores 378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

| Model                                    | Attack                   | Thm. 1 Compliant Samples   | Non Compliant Samples   |         |         |                     |                     |
|------------------------------------------|--------------------------|----------------------------|-------------------------|---------|---------|---------------------|---------------------|
| Perturbation Sample Size Acc Prec Rec F1 | Sample Size Acc Prec Rec | F1                         |                         |         |         |                     |                     |
| ResNet-CIFAR-10                          | ℓ 2/255 ∞                | 3345                       | 1.0                     | 1.0     | 1.0 1.0 | 1655                | 0.63 0.73 0.42 0.53 |
| 4/255 ℓ ∞                                | 2967                     | 1.0                        | 1.0                     | 1.0 1.0 | 2033    | 0.66 0.78 0.45 0.57 |                     |
| CLIP-TinyImageNet                        | ℓ 2/255 ∞                | 510                        | 1.0                     | 1.0     | 1.0 1.0 | 4490                | 0.67 0.63 0.84 0.72 |
| 4/255 ℓ ∞                                | 556                      | 1.0                        | 1.0                     | 1.0 1.0 | 4444    | 0.65 0.62 0.80 0.70 |                     |

Table 1: Results from Experiment 1: Detector metrics—accuracy, precision, recall, and F1—for ResNet-18 and CLIP (ViT-B/32) backbone finetuned with our Ltotal objective in equation 6 and evaluated under PGD on the two subsets: images that *satisfy* Theorem 1 vs. those that *do not*. across all four key metrics: Accuracy, Precision, Recall, and F1-score. This confirms our hypothesis that KL-divergence and L0-based metrics are highly complementary. The KL metric effectively captures dense, distribution-level shifts that often go undetected by other measures, while the L0 metric is uniquely sensitive to sparse, high-impact changes. Their combined use allows the detector to identify a wider range of adversarial attack types.

The results on the CLIP/Tiny-ImageNet setup, however, show that the L0+KL+Cosine combination slightly outperforms the others. This unexpected finding is an interesting artifact of the model's behavior. As shown in Table 6, the model fine-tuned with the L0+KL+Cosine loss exhibits a very low adversarial accuracy. This indicates that a single adversarial perturbation pushes the embedding into a region where all three metrics are essentially "randomly guessing" a class. The probability of all three classifiers independently guessing the same incorrect class is extremely low, leading to frequent disagreements and, consequently, a high attack detection rate. This outcome on the CLIP model underscores a critical distinction: a high detection rate does not always equate to a truly robust model. While the L0+KL+Cosine setup appears effective at flagging attacks on CLIP, it does so by breaking the underlying classification, rather than by preserving it. This contrasts with the ResNet results, where our KL+L0 combination shows a more balanced approach to robust classification and detection.

Metric Combinations Attack Perturbation ResNet-CIFAR-10 CLIP-TinyImageNet Accuracy Precision Recall F1 Accuracy Precision Recall F1

| to robust classification and detection. Metric Attack   | ResNet-CIFAR-10   | CLIP-TinyImageNet         |      |                           |      |      |      |      |      |
|---------------------------------------------------------|-------------------|---------------------------|------|---------------------------|------|------|------|------|------|
| Combinations                                            | Perturbation      | Accuracy Precision Recall | F1   | Accuracy Precision Recall | F1   |      |      |      |      |
| KL+L0                                                   | ℓ 2/255 ∞         | 0.88                      | 0.94 | 0.81                      | 0.87 | 0.71 | 0.66 | 0.85 | 0.74 |
| 4/255 ℓ ∞                                               | 0.87              | 0.94                      | 0.78 | 0.85                      | 0.69 | 0.65 | 0.82 | 0.73 |      |
| L0+Cosine                                               | ℓ 2/255 ∞         | 0.73                      | 0.91 | 0.52                      | 0.66 | 0.70 | 0.66 | 0.85 | 0.74 |
| 4/255 ℓ ∞                                               | 0.68              | 0.89                      | 0.41 | 0.56                      | 0.68 | 0.64 | 0.79 | 0.71 |      |
| KL+Cosine                                               | ℓ 2/255 ∞         | 0.78                      | 0.92 | 0.62                      | 0.74 | 0.70 | 0.66 | 0.82 | 0.73 |
| 4/255 ℓ ∞                                               | 0.76              | 0.91                      | 0.59 | 0.71                      | 0.71 | 0.67 | 0.84 | 0.74 |      |
| 2/255                                                   |                   |                           |      |                           |      |      |      |      |      |
| KL+L0+Cosine                                            | ℓ ∞               | 0.75                      | 0.91 | 0.55                      | 0.69 | 0.75 | 0.68 | 0.94 | 0.79 |
| ℓ 4/255 ∞                                               | 0.69              | 0.89                      | 0.44 | 0.59                      | 0.74 | 0.68 | 0.93 | 0.78 |      |

Table 2: Results from Experiment 2: Comparison of key detector performance metrics (accuracy, precision, recall, F1) for ResNet-18 and CLIP (ViT-B/32) models.

## 4.4 Experiment 3: Overall Adversarial Resilience Across Metric Combinations

- **Experiment Objective:** This experiment moves beyond attack detection metrics to evaluate the overall classification robustness of models fine-tuned with different metric combinations. We report both clean accuracy (performance on benign images) and adversarial accuracy (performance on successfully attacked images that were not detected) to assess how each fine-tuning objective impacts the underlying model's resilience. Again, our fine-tuning procedure is intentionally lightweight, relying solely on clean images. Unlike traditional adversarial defenses, our approach does not require costly adversarial examples or specialized training routines 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485

| Models                                                | Image        | Clean Image   | PGD attack(%) CW attack(%) Auto attack(%)   |       |       |       |       |       |
|-------------------------------------------------------|--------------|---------------|---------------------------------------------|-------|-------|-------|-------|-------|
| Encoder                                               | Accuracy (%) | 2/255         | 4/255                                       | 2/255 | 4/255 | 2/255 | 4/255 |       |
| ℓ ∞                                                   | ℓ ∞          | ℓ ∞           | ℓ ∞                                         | ℓ ∞   | ℓ ∞   |       |       |       |
| Baseline model                                        | ResNet18     | 95.16         | 45.5                                        | 33.11 | 45.99 | 35.98 | 45.49 | 31.95 |
| Cosine Similarity                                     | 94.98        | 45.8          | 37.8                                        | 37.80 | 33.00 | 35.40 | 22.02 |       |
| KL                                                    | 89.50        | 41.48         | 29.00                                       | 39.06 | 30.78 | 40.74 | 30.62 |       |
| L0                                                    | 94.96        | 49.08         | 32.66                                       | 47.02 | 35.30 | 42.56 | 35.88 |       |
| KL+L0                                                 | 94.78        | 57.32         | 54.60                                       | 57.52 | 54.08 | 52.28 | 51.12 |       |
| Cosine+L0                                             | 94.76        | 43.98         | 32.22                                       | 44.78 | 36.18 | 44.94 | 35.92 |       |
| KL+Cosine                                             | 94.36        | 55.60         | 51.32                                       | 45.02 | 34.08 | 45.48 | 34.18 |       |
| KL+L0+Cosine                                          | 94.48        | 44.66         | 32.86                                       | 45.42 | 34.52 | 45.84 | 35.52 |       |
| Note: All finetuning was done using clean images only |              |               |                                             |       |       |       |       |       |

Table 3: Clean and adversarial accuracy for the ResNet-18 backbone fine-tuned with seven different single/composite embedding objectives under a PGD attack. The KL+L0 objective demonstrates superior adversarial accuracy, highlighting the complementary nature of these two metrics.

Models Image

Encoder

Clean Image

Accuracy (%)

PGD attack Auto attack CW attack

ℓ

2/255

∞ ℓ

4/255

∞ ℓ

2/255

∞ ℓ

4/255

∞ ℓ

2/255

∞ ℓ

4/255 ∞

baseline model CLIP(ViT-B/32) 57.88 0.38 0.28 0.01 0.01 0.0 0.0

Cosine Similarity **62.44** 33.74 33.72 3.22 0.07 3.06 0.05 L0 54.34 53.31 43.42 25.43 18.35 **37.49 13.67** KL 57.65 **60.02 58.87** 19.35 11.76 25.69 11.16 KL+L0 55.88 26.50 25.47 16.18 9.57 11.91 5.84

Cosine+L0 56.46 16.28 16.09 1.03 0.02 1.15 0.01

Cosine+KL 57.62 55.01 53.87 5.25 0.44 5.02 0.39

KL+L0+Cosine 56.30 14.93 14.72 0.97 0.06 1.14 0.01

| Models                                                | Image        | Clean Image   |
|-------------------------------------------------------|--------------|---------------|
| Encoder                                               | Accuracy (%) |               |
| Note: All finetuning was done using clean images only |              |               |

Table 4: Clean and adversarial accuracy for the CLIP ViT-B/32 backbone fine-tuned with seven different single/composite embedding objectives under a PGD attack. The KL+L0 objective demonstrates superior adversarial accuracy, highlighting the complementary nature of these two metrics. - **Results and Analysis for CLIP Model on Tiny-ImageNet:** Table 4 presents the results for the finetuned CLIP model. Unlike the ResNet, the L0-only fine-tuning objective yields the highest adversarial robustness, which can be attributed to the models' different training histories and architectures. The CLIP model is pre-trained on a massive dataset using a cosine-contrastive objective, which naturally encourages inter-class variation to be concentrated in a few principal directions of the highdimensional text embedding space. Because of this pre-existing sparsity-aware structure, enforcing further alignment via the L0-based metric is especially effective. Conversely, the ResNet model is trained from scratch on a smaller dataset (CIFAR-10) using a cross-entropy loss, which encourages class separations that are dispersed over many coordinates. For such a model, a single metric is insufficient. The combined KL+L0 criterion becomes necessary to simultaneously account for both dense and sparse perturbations, thereby realizing the necessary gains in adversarial robustness. - **Results and Analysis for ResNet Model on CIFAR-10:** We fine-tuned a ResNet-18 backbone using seven different objectives: Cosine similarity, L0, KL, L0+KL, Cosine+KL, Cosine+L0, and Cosine+KL+L0. The results in Table 3 show that all models maintain comparable clean accuracy, indicating that the fine-tuning process does not degrade the model's core classification ability.

However, the models yield starkly different adversarial accuracies. Our proposed KL+L0 objective achieves the strongest adversarial performance because KL-divergence and L0-based metrics are fundamentally complementary: KL excels at capturing dense, distribution-level shifts, while L0 is sensitive to sparse, high-impact changes. Optimizing both simultaneously forces the embeddings to be robust against a wider variety of adversarial perturbations, leading to better overall resilience. In contrast, any objective that includes the Cosine similarity leads to significantly lower adversarial robustness. The Cosine similarity encourages an angular alignment that conflicts with the the perdimension alignment of KL and L0. The resulting optimization trade-off degrades the model's ability to resist attacks, highlighting why simply adding more metrics is not always beneficial.

## 5 Ethics Statement:

486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512

## 513

514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 Adversarial attacks pose significant risks to the safety and security of machine learning systems, particularly in sensitive applications such as autonomous vehicles and medical diagnostics. Our work on the KOALA's detection method aims to mitigate these risks by providing a robust, theoretically grounded defense. We believe that by enhancing the security of deep neural networks, our research contributes positively to the ethical deployment of AI technology. This work does not use any sensitive personal data or create new privacy risks. It focuses on improving model robustness against malicious manipulation, thereby helping to ensure that AI systems operate as intended and can be trusted in real-world, safety-critical scenarios. We are committed to transparency and will make our code and models publicly available to facilitate further research and independent verification.

## 6 Reproducibility Statement:

We provide all details needed to reproduce our results. Section 3 specifies our KOALA's architecture, theoretical guarantees, and training objectives; Section 4 describes training/evaluation datasets, architectures, attack settings, hyperparameters, and evaluation metrics. The appendix provides full proof of our theorem. We also provide an anonymous repository in the supplementary materials with training/evaluation scripts.

## 7 Usage Of Llm

We used the large language model (LLM) as a general-purpose writing assistant for copy-editing
(grammar, phrasing, and concision) and LaTeX formatting suggestions. The LLM did not generate ideas, claims, proofs, figures, or results. All technical content and experiments were authored and verified by the authors, who take full responsibility for the paper. LLMs are not authors.

## References

Ahmed Aldahdooh, Wassim Hamidouche, Sid Ahmed Fezza, and Olivier Déforges. Adversarial example detection for dnn models: a review and experimental comparison. Artificial Intelligence Review, 55(6):4403–4462, January 2022. ISSN 1573-7462. doi: 10.1007/s10462-021-10125-w.

URL http://dx.doi.org/10.1007/s10462-021-10125-w.

Battista Biggio, Igino Corona, Davide Maiorca, Blaine Nelson, Nedim Šrndic, Pavel Laskov, Giorgio ´
Giacinto, and Fabio Roli. *Evasion Attacks against Machine Learning at Test Time*, pp. 387–402. Springer Berlin Heidelberg, 2013. ISBN 9783642387098. doi: 10.1007/978-3-642-40994-3_25. URL http://dx.doi.org/10.1007/978-3-642-40994-3_25.

Nicholas Carlini and David Wagner. Towards evaluating the robustness of neural networks. In 2017 ieee symposium on security and privacy (sp), pp. 39–57. Ieee, 2017.

Jeremy Cohen, Elan Rosenfeld, and Zico Kolter. Certified adversarial robustness via randomized smoothing. In *international conference on machine learning*, pp. 1310–1320. PMLR, 2019.

Francesco Croce and Matthias Hein. Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks. In *International conference on machine learning*, pp. 2206–2216. PMLR, 2020.

Kathrin Grosse, Praveen Manoharan, Nicolas Papernot, Michael Backes, and Patrick McDaniel. On the (statistical) detection of adversarial examples. *arXiv preprint arXiv:1702.06280*, 2017.

Charles Guille-Escuret, Pau Rodriguez, David Vazquez, Ioannis Mitliagkas, and Joao Monteiro. Cadet: Fully self-supervised out-of-distribution detection with contrastive learning.

Reuben Feinman, Ryan R Curtin, Saurabh Shintre, and Andrew B Gardner. Detecting adversarial samples from artifacts. *arXiv preprint arXiv:1703.00410*, 2017.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), Advances in Neural Information Processing Systems, volume 36, pp. 7361–7376. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper_files/paper/ 2023/file/1700ad4e6252e8f2955909f96367b34d-Paper-Conference.pdf.

Shengyuan Hu, Tao Yu, Chuan Guo, Wei-Lun Chao, and Kilian Q Weinberger. A new defense against adversarial images: Turning a weakness into a strength. *Advances in neural information processing* systems, 32, 2019.

Haitham Khedr and Yasser Shoukry. Deepbern-nets: Taming the complexity of certifying neural networks using bernstein polynomial activations and precise bound propagation. In *Proceedings of* the AAAI Conference on Artificial Intelligence, volume 38, pp. 21232–21240, 2024.

Xiangyin Kong, Xiaoyu Jiang, Zhihuan Song, and Zhiqiang Ge. Data id extraction networks for unsupervised class- and classifier-free detection of adversarial examples. IEEE Transactions on Pattern Analysis and Machine Intelligence, 47(9):7428–7442, 2025. doi: 10.1109/TPAMI.2025. 3572245.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009. Boyi Lee, Jhao-Yin Jhang, Lo-Yao Yeh, Ming-Yi Chang, Chia-Mei Chen, and Chih-Ya Shen. Detecting targets of graph adversarial attacks with edge and feature perturbations. *IEEE Transactions on* Computational Social Systems, 11(3):3218–3231, 2024. doi: 10.1109/TCSS.2023.3344642.

Kimin Lee, Kibok Lee, Honglak Lee, and Jinwoo Shin. A simple unified framework for detecting out-of-distribution samples and adversarial attacks. *Advances in neural information processing* systems, 31, 2018.

Changliu Liu, Tomer Arnon, Christopher Lazarus, Christopher Strong, Clark Barrett, Mykel J
Kochenderfer, et al. Algorithms for verifying deep neural networks. Foundations and Trends® in Optimization, 4(3-4):244–404, 2021.

Shiqing Ma and Yingqi Liu. Nic: Detecting adversarial samples with neural network invariant checking. In *Proceedings of the 26th network and distributed system security symposium (NDSS* 2019), 2019.

Xingjun Ma, Bo Li, Yisen Wang, Sarah M Erfani, Sudanthi Wijewickrema, Grant Schoenebeck, Dawn Song, Michael E Houle, and James Bailey. Characterizing adversarial subspaces using local intrinsic dimensionality. *arXiv preprint arXiv:1801.02613*, 2018.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.

Towards deep learning models resistant to adversarial attacks. *arXiv preprint arXiv:1706.06083*, 2017a.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu.

Towards deep learning models resistant to adversarial attacks. *arXiv preprint arXiv:1706.06083*,
2017b.

Dongyu Meng and Hao Chen. Magnet: a two-pronged defense against adversarial examples. In Proceedings of the 2017 ACM SIGSAC conference on computer and communications security, pp.

135–147, 2017.

Jan Hendrik Metzen, Tim Genewein, Volker Fischer, and Bastian Bischoff. On detecting adversarial perturbations. *arXiv preprint arXiv:1702.04267*, 2017.

Raymond Muller, Yanmao Man, Ming Li, Ryan M. Gerdes, Jonathan Petit, and Z. Berkay Celik. VOGUES: Validation of object guise using estimated components. In *Proceedings of* the 33rd USENIX Security Symposium (USENIX Security '24), Philadelphia, PA, USA, August 2024. USENIX Association. ISBN 978-1-939133-44-1. URL https://www.usenix.org/ system/files/usenixsecurity24-muller.pdf.

Fuseini Mumuni and Alhassan Mumuni. Improving deep learning with prior knowledge and cognitive models: A survey on enhancing explainability, adversarial robustness and zero-shot learning. Cognitive Systems Research, 84:101188, March 2024. ISSN 1389-0417. doi: 10.1016/j.cogsys. 2023.101188. URL http://dx.doi.org/10.1016/j.cogsys.2023.101188.

Ali Shafahi, Mahyar Najibi, Mohammad Amin Ghiasi, Zheng Xu, John Dickerson, Christoph Studer, Larry S Davis, Gavin Taylor, and Tom Goldstein. Adversarial training for free! *Advances in neural* information processing systems, 32, 2019.

Jake Snell, Kevin Swersky, and Richard Zemel. Prototypical networks for few-shot learning. *Advances* in neural information processing systems, 30, 2017.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. *arXiv preprint arXiv:1312.6199*, 2013.

Chaowei Xiao, Bo Li, Jun yan Zhu, Warren He, Mingyan Liu, and Dawn Song. Generating adversarial examples with adversarial networks. In *Proceedings of the Twenty-Seventh International Joint* Conference on Artificial Intelligence, IJCAI-18, pp. 3905–3911. International Joint Conferences on Artificial Intelligence Organization, 7 2018a. doi: 10.24963/ijcai.2018/543. URL https:
//doi.org/10.24963/ijcai.2018/543.

Chaowei Xiao, Jun-Yan Zhu, Bo Li, Warren He, Mingyan Liu, and Dawn Song. Spatially transformed adversarial examples. *arXiv preprint arXiv:1801.02612*, 2018b.

Weilin Xu, David Evans, and Yanjun Qi. Feature squeezing: Detecting adversarial examples in deep neural networks. In *Proceedings 2018 Network and Distributed System Security Symposium*, NDSS 2018. Internet Society, 2018. doi: 10.14722/ndss.2018.23198. URL http://dx.doi. org/10.14722/ndss.2018.23198.

Kaiwen Yang, Tianyi Zhou, Yonggang Zhang, Xinmei Tian, and Dacheng Tao. Classdisentanglement and applications in adversarial detection and defense. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 16051–16063. Curran Associates, Inc.,
2021. URL https://proceedings.neurips.cc/paper_files/paper/2021/ file/8606f35ec6c77858dfb80a385d0d1151-Paper.pdf.

Jiawei Zhang, Linyi Li, Ce Zhang, and Bo Li. Care: Certifiably robust learning with reasoning via variational inference. In 2023 IEEE Conference on Secure and Trustworthy Machine Learning
(SaTML), pp. 554–574. IEEE, 2023.

Andy Zhou, Xiaojun Xu, Ramesh Raghunathan, Alok Lal, Xinze Guan, Bin Yu, and Bo Li. Knowgraph: Knowledge-enabled anomaly detection via logical reasoning on graph data. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, pp. 168–182, 2024.

Tengwei Song, Xudong Ma, Yang Liu, and Jie Luo. Robust knowledge graph embedding via denoising. *arXiv preprint arXiv:2505.18171*, 2025.