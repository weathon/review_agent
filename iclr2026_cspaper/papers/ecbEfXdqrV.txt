000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 Deep generative models with tractable and analytically computable likelihoods, exemplified by normalizing flows, offer an effective basis for anomaly detection through likelihood-based scoring. We demonstrate that, unlike in the image domain where deep generative models frequently assign higher likelihoods to anomalous data, such counterintuitive behavior occurs far less often in tabular settings. We first introduce a domain-agnostic formulation that enables consistent detection and evaluation of the counterintuitive phenomenon, addressing the absence of precise definition. Through extensive experiments on 47 tabular datasets and 10 CV/NLP embedding datasets in ADBench, benchmarked against 12 baseline models, we demonstrate that the phenomenon, as defined, is consistently rare in general tabular data. We further investigate this phenomenon from both theoretical and empirical perspectives, focusing on the roles of data dimensionality and feature correlation difference. We find that likelihood-only detection with normalizing flows offers a practical and reliable approach for anomaly detection in tabular domains.

## 1 Introduction

Generative models, including variational autoencoders (VAEs) (Kingma & Welling, 2014), normalizing flows (NFs) (Dinh et al., 2015), and generative adversarial networks (GANs) (Goodfellow et al., 2020), are widely used to model complex data distributions across diverse applications such as industrial diagnostics, medical imaging, and financial risk assessment. Among these, normalizing flows are particularly well-suited for anomaly detection due to their ability to compute estimated likelihoods, providing a straightforward mechanism for detecting out-of-distribution (OOD) samples.

1 The simplest approach of anomaly detection using normalizing flows is to assume that normal data x ∈ R
dfollows the distribution P of normal data, and anomalous data x
′ ∈ R
dfollows a distribution Q ̸= P, and to determine that a given data x*test* ∈ R
dis an anomaly if its likelihood ϕP (x) is lower than a predefined threshold α when tested. We refer to this methodology using normalizing flow as NF-SLT (Normalizing Flow with Simple Likelihood Test). This methodology is based on the intuition that anomalous data are less likely to be observed in the distribution of normal data. However, the image domain illustrates that in-distribution data utilized as training data in models that can obtain the likelihood of the input data indirectly or directly, such as normalizing flow, exhibit similar or even lower likelihoods than out-of-distribution data. Nalisnick et al. (2019a) demonstrates that when CIFAR-10 (Alex, 2009) is used as training data (In-distribution)
and SVHN (Netzer et al., 2011) is used as the test data (Out-of-distribution) of a model that can obtain the likelihood of input data, SVHN has a higher likelihood than CIFAR-10. This is counterintuitive because the likelihood of OOD data is higher than that of in-distribution data. Therefore, it can be inferred that if anomaly detection is performed using only the likelihood of the input data, detection may fail in certain cases (i.e., occurrence of a counterintuitive phenomenon). Refer to Section 2.2 for more details about counterintuitive phenomenon. However, the following question arises: does this phenomenon also occur in tabular data anomaly detection? Kirichenko et al. (2020) demonstrates that although the likelihood of in-distribution/OOD
1Although the two tasks slightly differ, we consider OOD detection and anomaly detection to be the same task, and we will utilize the term anomaly detection. Task definitions are presented in Appendix A.

# Why Is The Counterintuitive Phenomenon Of Likelihood Rare In Tabular Anomaly Detec- Tion With Deep Generative Models?

Anonymous authors Paper under double-blind review

## Abstract

1 054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 data overlaps for the normalizing flow in the tabular data anomaly detection, it is limited by the fact that only two datasets are shown by setting each as in-distribution data/OOD data. In addition, there is no comparison with other comparison models. A common argument is that assigning likelihoods higher than that of normal data to anomalies is sufficient to demonstrate a conterintuitive phenomenon. Regardless, the interpretation has its limitations. First, the view is contradictory since the argument would consider any result outside 100% AUROC as counterintuitive. Second, likelihood inversion can arise from intrinsic dataset difficulty, for example, when normal and abnormal samples are hard to distinguish, rather than from the phenomenon itself. This calls for a more sophisticated approach for determining counterintuitive phenomenon, such as by comparing the generative model's performance against other models (e.g., DeepSVDD, OCSVM), as limitation arises when simple approach (such as direct comparison between abnormal/normal data's likelihood) is made. Hence, it is not yet identified whether the counterintuitive likelihood phenomenon occurs in the tabular domain. To address this gap, for currently vaguely-defined counterintuitive phenomenon, we first propose a clearer definition based on the observation in likelihood-based tests for models with estimated likelihoods, allowing the concept to be applied across different domains.

Building on this definition, we conduct an extensive set of experiments to examine whether the simple likelihood test, previously criticized for its limitations in image anomaly detection, remains effective in the context of tabular anomaly detection. Consequently, we empirically demonstrate that almost all datasets in ADBench (Han et al., 2022), a tabular AD benchmark dataset, do not exhibit counterintuitive phenomena in the tabular domain, and even NF-SLT outperform comparison models in simple likelihood tests. Furthermore, we demonstrate its success in the tabular domain theoretically and empirically from the perspective of dimensionality and feature correlation. To explain why this counterintuitive phenomenon does not occur in the tabular domain, we use the following two facts:
Fact 1.1 (Lower Dimension). Images typically have three dimensions: height, width, and channel, while tabular data generally have lower dimensionality, consisting of a single feature vector without spatial structure. Fact 1.2 (Correlation of Features). Images exhibit strong local pixel correlations, which allows models like CNN to effectively capture spatial relationships between neighboring pixels. In contrast, tabular data does not assume any specific structural relationship between features. Taking ADBench, as an example, most of the datasets have a dimension lower than 100. However, CIFAR-10, one of the image datasets with small dimensions, has a dimension of 3072. This shows that the curse of dimensionality may be more severe in the image domain than in the tabular domain, and we analyze how this affects the likelihood test using normalizing flow. Additionally, Kirichenko et al. (2020) argued that in image OOD detection, normalizing flows fail to capture semantic information effectively because images exhibit local pixel correlations. Based on Fact 1.2, we extend this discussion to the tabular domain and claim that normalizing flows are less affected by feature correlation in this setting. To justify this claim, we quantify overall feature correlation by measuring the reduction of intrinsic dimension (ID) relative to the ambient dimension. We then explain why this reduction reflects the effect of correlation, and compare the degree of ID reduction observed in tabular and image data. Although there are datasets in the tabular domain that have higher dimensions than images or strong correlation (e.g., genomics, see Appendix C.4), these have very different characteristics from typical tabular datasets, so it is reasonable to assume that the trends of the two domains follow the examples described above. In conclusion, the contribution of our study can be described as threefold.

- We provide a **domain-agnostic definition of the counterintuitive phenomenon** in simple likelihood tests and empirically show that simple likelihood testing with normalizing flows in the tabular domain rarely leads to this phenomenon, outperforming comparison models.

- We verify our results using all **47 tabular datasets and 10 CV/NLP embedding datasets**
from ADBench without selection bias (Shwartz-Ziv & Armon, 2022) and compare against 12 anomaly detection baselines.

- We demonstrate **a theoretical and empirical analysis** of why the counterintuitive phenomenon does not occur in the tabular domain, unlike in images, by linking it to the difference in dimension and feature correlation.

## 2 Related Work 2.1 Normalizing Flow

Normalizing flow is one of the generative models that converts input data x ∈ R
d, which follows an unknown distribution called px, into z ∈ R
d; in addition, it follows a simple distribution pz that is typically chosen as standard Gaussian N (0, Id) (Dinh et al., 2017), using an invertible function f : R
d → R
dthat consists of complex functions such as neural networks (Dinh et al., 2015), such that px can be written as a formula in terms of pz. At this point, px is expressed as the determinant of Jacobian of x and z and pz by the change-of-variable rule, and is expressed as Equation 1.

108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161

$$\log p_{\mathbf{x}}(\mathbf{x})=\log p_{\mathbf{z}}(\mathbf{z})+\log|J|,J=\operatorname*{det}{\frac{\partial}{\partial t}}$$

## ∂Z

∂x(1)
In general, it learns in the direction of maximizing the likelihood log px(x) of the learning input data, and approximates the distribution of the input data (Caterini & Loaiza-Ganem, 2022). Normalizing flows can be categorized by whether the determinant of the Jacobian (i.e., the volume term) is fixed (Dinh et al., 2015) or varies with the input (Rezende & Mohamed, 2015; Dinh et al., 2017; Kingma & Dhariwal, 2018; Behrmann et al., 2019; Chen et al., 2019; Durkan et al., 2019). When sampling new data, sampling is performed by extracting it from the pre-defined pz and inputting it as the input of f
−1. The normalizing flow has the advantage of being able to obtain the estimated likelihood of the input data, unlike models such as variational autoencoder and generative adversarial network.

Additionally, the normalizing flow has the advantage of not requiring the additional likelihood approximate inference techniques (Nalisnick et al., 2019a). However, normalizing flow has two constraints: (1) the computational amount of Jacobian must not become too large, and (2) the inverse of f must exist. Therefore, the following methodologies were utilized to ensure the ease of Jacobian calculation and the existence of the inverse f
−1: methods such as a coupling layer (Dinh et al., 2015; 2017; Kingma & Dhariwal, 2018), special-form transformations (Rezende & Mohamed, 2015), and power-series approximations with Lipschitz constraint (Behrmann et al., 2019; Chen et al., 2019) are commonly used.

## 2.2 Counterintuitive Phenomenon Of Likelihood

Nalisnick et al. (2019a) reported that a counterintuitive phenomenon regarding likelihood assignment occurs in models that can obtain estimated likelihood, such as normalizing flow, in the image domain. This study lays the foundation for identifying the cause of this phenomenon or suggesting solutions. Kirichenko et al. (2020); Schirrmeister et al. (2020) improved anomaly detection performance by modifying flow architectures. In particular, the latter introduced an approach that reflects the hierarchical data structure, thereby improving detection performance. Serrà et al. (2020) quantified complexity through a general compression algorithm such as PNG, based on experimental results, demonstrating that simple images exhibit higher likelihood, and presented an anomaly score combining the likelihood and complexity terms. Kamkari et al. (2024) used Local Intrinsic Dimension (LID)
to measure an image's simplicity and proposed a dual thresholding method for LID and likelihood to improve anomaly detection performance. Morningstar et al. (2021); Osada et al. (2024); Ahmadian et al. (2021) mitigated the drawback of using only a single likelihood score by estimating the density of a vector that combines the likelihood with several auxiliary statistics (e.g., complexity, the logdeterminant of the Jacobian). Nalisnick et al. (2019b) demonstrated the perspective that detection may fail because in-distribution data are located in the typicality set (Cover, 1999) and OOD data is in the high density set. Zhang et al. (2021) presented the view that the counterintuitive phenomenon occurs due to misestimation of the model. Le Lan & Dinh (2021) demonstrated that even with a perfect model, simple likelihood-based methods can fail due to variants in the representation. Ren et al. (2019) improved detection performance by using the likelihood ratio between the background and semantic models and Caterini & Loaiza-Ganem (2022) explained the cause of this phenomenon from an entropic perspective and why the likelihood ratio model works well.

## 3 Definition Of Counterintuitive Phenomenon

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 Earlier research (Kirichenko et al., 2020) noted instances where in-distribution and OOD data had overlapping likelihoods in tabular datasets, but these findings were limited to only a few datasets and lacked comprehensive comparisons with other anomaly detection models. To address limitations in prior work's explanations of the counterintuitive phenomenon, we propose a generalized definition of the counterintuitive phenomenon that applies to diverse domains. To formalize this phenomenon, we begin by establishing two core assumptions: Assumption 3.1 (Relatively Low Performance). *If a counterintuitive phenomenon occurs, most* comparison models should outperform the generative model on an anomaly detection task. Assumption 3.2 (High Performance Gap). *Even if the above condition is satisfied, the performance* gap between the generative model and comparison models must be significant to qualify as a counterintuitive phenomenon. If the gap is small, it cannot be considered counterintuitive. We now formalize this phenomenon using these assumptions.

Definition 3.3 (Occurrence of Counterintuitive Phenomenon). Let AUROC0 denote the AUROC of the likelihood-only test using the generative model Pθ0*on a normal/abnormal dataset pair*
(P, Q)*, and let* AUROCi denote that of the i*-th comparison model for* i = 1, . . . , k*. We say that a* counterintuitive phenomenon occurs if both conditions hold:

$$\frac{1}{k}\sum_{i=1}^{k}\mathbb{1}\{\mathrm{AUROC}_{i}>\mathrm{AUROC}_{0}\}>\beta,$$  $$\min_{i:\mathrm{AUROC}_{i}>\mathrm{AUROC}_{0}}(\mathrm{AUROC}_{i}-\mathrm{AUROC}_{0})>\gamma.$$
$$(2)$$
$$({\mathfrak{I}})$$

Definition 3.3 states that a counterintuitive phenomenon occurs when the proportion of the comparison models whose AUROC exceeds that of the generative model Pθ0exceeds β, and the minimum AUROC difference between Pθ0and the models that outperform Pθ0is greater than γ. Consequently, Definition 3.3 enables performance comparisons using relative AUROC, allowing us to determine whether a counterintuitive phenomenon has occurred, rather than merely inferring its presence from a low AUROC. The fully rigorous formulation of Definition 3.3 is provided in Appendix B. Consider the CIFAR-10 (in-distribution) vs. SVHN (out-of-distribution). According to Morningstar et al. (2021), a simple likelihood test using the Glow (Kingma & Dhariwal, 2018) yielded an AUROC of 6.4%. In contrast, Sun et al. (2022) achieved AUROC scores exceeding 90% with their proposed method and comparison models. Based on Definition 3.3, this case clearly demonstrates a counterintuitive phenomenon, as the generative model performs significantly worse than the comparison models. To explore whether this phenomenon occurs in tabular data, we conducted experiments to test if a counterintuitive phenomenon, as defined in Definition 3.3, appears in tabular anomaly detection datasets.

## 4 Experiment

Dataset and Preprocessing The experiment was conducted using the data split protocol in Zong et al. (2018). To explain this protocol, 50% of normal data is used for training, and the remaining 50% of normal and abnormal data are used as test data. We used **all 47 tabular and 10 CV/NLP** embedding datasets presented in ADBench. Using the entire dataset was motivated by Shwartz-Ziv & Armon (2022), who criticized that researchers often introduced selection bias by choosing specific datasets to inflate performance. To address this, we included all proposed benchmark datasets without exclusion. All models except the NeuTraLAD model utilized RobustScaler provided by the Python library Scikit-learn (Pedregosa et al., 2011) to standardize the input data. The reason for excluding NeuTraLAD is that a significant performance decrease was observed when scaling.

Models We compared the performance of 6 shallow AD models and 6 deep AD models. We implemented the shallow models using PyOD (Zhao et al., 2019) and Scikit-learn (Pedregosa et al., 2011). The compared shallow models are PCA (Shyu et al., 2003), LOF (Breunig et al., 2000), IF (Liu et al., 2008), OCSVM (Schölkopf et al., 1999), COPOD (Li et al., 2020), and ECOD (Li et al., 2022). The compared deep models are DAGMM (Zong et al., 2018), DeepSVDD (Ruff et al., 2018), GOAD (Bergman & Hoshen, 2020), NeuTraLAD (Qiu et al., 2021), ICL (Shenkar & Wolf, 2022),
216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269 Table 1: (Top): Evaluation performance of 47 tabular data. (Bottom): Evaluation performance of 10 CV/NLP embedding data. The Top2 Ratio indicates the proportion of datasets where model ranked within the top2 for AUROC, the Fail Ratio shows the proportion of datasets where a model's AUROC rank was 9th or lower.

| Method       | AUROC ↑   | AUPRC ↑   | Avg. Rank ↓   | Top2 Ratio ↑   | Fail Ratio ↓   |        |
|--------------|-----------|-----------|---------------|----------------|----------------|--------|
| PCA          | 0.7715    | 0.5209    | 6.51          | 0.17           | 0.40           |        |
| LOF          | 0.8169    | 0.5606    | 5.53          | 0.21           | 0.23           |        |
| IF           | 0.8014    | 0.5060    | 5.62          | 0.19           | 0.13           |        |
| OCSVM        | 0.6562    | 0.3833    | 9.47          | 0.06           | 0.72           |        |
| COPOD        | 0.7471    | 0.4419    | 7.68          | 0.11           | 0.49           |        |
| ECOD         | 0.7425    | 0.4530    | 7.98          | 0.09           | 0.49           |        |
| DAGMM        | 0.6467    | 0.3468    | 10.51         | 0.00           | 0.85           |        |
| DeepSVDD     | 0.7687    | 0.5388    | 6.96          | 0.06           | 0.34           |        |
| GOAD         | 0.6086    | 0.4114    | 9.72          | 0.04           | 0.60           |        |
| NeuTraLAD    | 0.8081    | 0.5694    | 5.57          | 0.26           | 0.26           |        |
| ICL          | 0.8208    | 0.6170    | 5.17          | 0.32           | 0.23           |        |
| MCM          | 0.7864    | 0.5383    | 6.70          | 0.11           | 0.23           |        |
| NF-SLT       | 0.8575    | 0.6398    | 3.43          | 0.45           | 0.02           |        |
| Dataset      | DeepSVDD  | GOAD      | NeuTraLAD     | ICL            | MCM            | NF-SLT |
| CIFAR-10     | 0.9103    | 0.9335    | 0.9405        | 0.9254         | 0.9381         | 0.9527 |
| FashionMNIST | 0.9117    | 0.9060    | 0.9360        | 0.9267         | 0.9380         | 0.9455 |
| MNIST-C      | 0.8348    | 0.7741    | 0.8519        | 0.8257         | 0.8836         | 0.8950 |
| MVTecAD      | 0.7543    | 0.7960    | 0.8874        | 0.8874         | 0.8408         | 0.9100 |
| SVHN         | 0.5466    | 0.5366    | 0.5774        | 0.5626         | 0.5770         | 0.5842 |
| 20news       | 0.5547    | 0.5438    | 0.6001        | 0.6087         | 0.5995         | 0.6547 |
| agnews       | 0.6630    | 0.5857    | 0.6510        | 0.6697         | 0.7252         | 0.7591 |
| amazon       | 0.5833    | 0.5613    | 0.6010        | 0.6022         | 0.6050         | 0.6194 |
| imdb         | 0.5090    | 0.5398    | 0.5393        | 0.5098         | 0.5090         | 0.5013 |
| yelp         | 0.6490    | 0.6138    | 0.6620        | 0.6690         | 0.6750         | 0.6971 |

MCM (Yin et al., 2024), and NF-SLT with NICE (Dinh et al., 2015). For NF-SLT with NICE, we used 10 coupling layer and trained the model for 200 epochs with weight decay 1e-4. We optimized the negative log-likelihood of the latent variables using AdamW (Loshchilov & Hutter, 2019) with a CosineAnnealingWarmRestarts learning rate scheduler (Loshchilov & Hutter, 2017). The batch size was set to 512, and the latent prior was fixed to N (0, Id). Overall hyperparameter settings and implementation details are provided in Appendix F. Evaluation We evaluate these AD models using AUROC and AUPRC. We performed 10 repeated experiments on the tabular datasets and recorded the average AUROC scores and the relative rank of each model as summarized in Table 1. For each dataset, after experimenting with all combinations in the hyperparameter searching space with 10 repeated experiments, the hyperparameter combination with the highest average AUROC for all datasets is selected as the representative hyperparameter combination to demonstrate the performance of the model. The hyperparameter search space for each model and hyperparameter sensitivity experiment is recorded in Appendix F. Additionally, the results of applying other flows to NF-SLT are included in Appendix G. Experiment Result Consider Definition 3.3; if a counterintuitive phenomenon is also frequent in the tabular domain, it should have a high fail ratio even if it works well on a particular dataset resulting in a high top2 ratio. In addition, the failed dataset should have a large minimum performance difference from the other models. However, based on the results in Table 1, we can observe that NF-SLT has a lower fail ratio than the shallow and deep models, and outperforms other metrics. Furthermore, on the 'yeast' dataset where NF-SLT exhibited relatively low performance, the minimum performance difference between MCM and AUROC is 0.02; hence, we cannot assume that it exhibited low performance due to a counterintuitive phenomenon. Furthermore, NF-SLT outperforms deep models on ADBench's CV/NLP embedding datasets, excluding the 'imdb' dataset. Although it shows worse performance than other models on the 'imdb' dataset, the difference in performance with the comparison model is very small, so it cannot be judged that a counterintuitive phenomenon has occurred because it does not satisfy the second condition of Definition 3.3. Furthermore, we report the detection performance on datasets dominated by categorical features and various anomaly types in Appendix E, and the results of the experiments also demonstrate the superiority of NF-SLT. To verify the consistency, we compared its performance with other test methodologies such that typicality test (Nalisnick et al., 2019b), with results in Appendix H.

## 5 Why Is The Simple Likelihood Test Successful In Tabular Data? 5.1 High Dimension Perspective

Based on Fact 1.1, we explain why tabular data can be successful in likelihood testing because of their lower dimensionality. It has been reported that the case where the likelihood of normal and anomaly data is inverted in the image domain usually occurs when the normal data has a more complex texture than the anomaly data, that is, when the complexity of the normal data is higher than that of the anomaly data (Serrà et al., 2020). Additionally, it can be thought that the high complexity of data sampled from a specific distribution means that the entropy of the distribution is high. Hence, to explain why the counterintuitive phenomenon rarely occurs in the tabular domain, we extend the likelihood-gap expression of Caterini & Loaiza-Ganem (2022), which characterizes the expected likelihood difference between normal and abnormal data in terms of entropy, and link it to Fact 1.1.

Let the distribution of normal data be P, the distribution of abnormal data be Q, and let Pθ be a model such as normalizing flow that estimates the density of P. Then, the gap of the likelihood of each distribution estimated by Pθ can be expressed as follows:
Ex∼P [log Pθ(x)] − Ex∼Q[log Pθ(x)] = DKL(Q||Pθ) − DKL(P||Pθ) + H(Q) − H(P) (4)
where H(P) is entropy of distribution P, and DKL(Q||Pθ) is the KL-divergence of distribution Q
and density estimation model like normalizing flow Pθ. In Equation 4, if the difference in entropy between the two distributions H(Q) − H(P) is a very small negative number , the expectation gap of the likelihood can become negative. However, in the previous study in Caterini & Loaiza-Ganem (2022), the effect of the dimension in expectation of likelihood gap was not analyzed, so we analyzed how the dimension can affect the expectation of likelihood gap and included it in Theorem 5.4 and the proof of this is reported in Appendix D.

Theorem 5.4 (Impact of Dimensionality on Likelihood Gap). Let P =Qd i=1 pi(xi) and Q = 
Qd i=1 qi(xi) be independent d*-dimensional continuous probability density models in* R
d *with same* conditions as Lemma 5.1. Let Pθ be a well-trained density estimation model approximates P *(i.e.,*
pθ(x) → p(x) pointwisely as θ → θ0). If H(P) − H(Q) > DKL(Q||P)*, the lower bound of gap* between the expectation of the likelihood for P and Q *decreases linearly with respect to* d.

270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 According to Theorem 5.4, even when Pθ is almost perfect model, if P and Q are d-dimensional independent distributions and the difference in entropy between the two distributions is greater than a DKL(Q||P), it can be verified that the lower bound of gap between the expectation of the likelihood for P and Q is negative and decreases linearly with the dimension. This shows that as the dimension increases, the phenomenon of inversion of the likelihood expectation of data sampled from each distribution can become more severe. Additionally, we show in Corollary 5.6 that under additional assumptions on the entropy of the distribution, not only the likelihood gap but also the upper bound of the AUROC, which is a practical and widely used evaluation metric, is inversely proportional to the dimension. The proof of this result is provided in Appendix D. Corollary 5.6 (Dimensionality and AUROC Upper Bound). *Building on the assumptions of Corollary* 5.5, suppose the n*-th absolute central moment of the log-likelihood difference,* log pθ(Y )−log pθ(X),
scales as O(d k) for some n > 1 and k < n. In this case, if the average log-likelihood gap becomes negative, the maximum achievable AUROC for distinguishing samples from P and Q is inversely related to the dimensionality d. This indicates that as the dimension increases, the likelihood test becomes fundamentally less effective at separating normal and abnormal samples. According to Corollary 5.6, the upper bound on the achievable AUROC decreases. This is consistent with Definition 3.3: a smaller AUROC implies a higher likelihood that the counterintuitive phenomenon occurs. To validate this prediction, we conducted dimensionality-reduction experiments. Specifically, we applied ICA (Hyvärinen & Oja, 2000) to high-dimensional image data and retained a varying number of independent components. Using RealNVP, we then measured the AUROC as Table 2: AUROC scores for likelihood tests as a function of dimensionality (number of PCs) using RealNVP with MLP (image preprocessed by ICA). The region to the left of the bold vertical line indicates cases where H(P) > H(Q), and the region to the right is the opposite.

In-dist (P) / Out-dist (Q) 1024 512 256 30 In-dist (P) / Out-dist (Q) 1024 512 256 30

CIFAR-10 / SVHN 0.2311 0.2924 0.2984 0.3143 SVHN / CIFAR-10 0.9917 0.9843 0.9486 0.8520

CIFAR-100 / SVHN 0.0843 0.1160 0.2036 0.3490 SVHN / CIFAR-100 0.9933 0.9536 0.9137 0.8622

CelebA / SVHN 0.1207 0.1782 0.2745 0.4711 SVHN / CelebA 0.9976 0.9811 0.9722 0.9481

a function of the retained dimension in Table 2. This setup isolates the effect of dimensionality on likelihood ranking and provides empirical support for our theoretical claims.

The results in Table 2 show that, when H(P) > H(Q) holds, the AUROC increases as the dimensionality decreases. Notably, the improvement remains substantial even when the dimensionality is reduced to almost 1% of the original dimension. In contrast, the cases to the right of the bold vertical line show decreasing AUROC as the dimension decreases. This matches the trivial behavior obtained by reversing the entropy condition, i.e., H(P) < H(Q), in Theorem 5.4 and Corollary 5.6. Therefore, even if the H(P) > H(Q) condition is satisfied, tabular data can be considered more advantageous in the simple likelihood test because they are less exposed to the problems that arise in high dimensions, as indicated by Fact 1.1. We also adjusted the dimension of the image using the bilinear interpolation resize method provided by torchvision (Marcel & Rodriguez, 2010) for the raw image, and performed a likelihood test after obtaining the likelihood of the image through Glow which is consist of a CNN, the results are included in Table 3. Since this experiment uses raw images, independence between pixels is not guaranteed, so the theorem presented in Appendix D cannot be applied. However, this experiment was conducted to check the performance trend according to dimension in a situation where there is no independence assumption. Table 3: AUROC scores for likelihood tests as a function of image size using Glow (image resized by bilinear interpolation). The region to the left of the bold vertical line indicates cases where H(P) > H(Q), and the region to the right is the opposite.

In-dist (P) / Out-dist (Q) 32x32 16x16 8x8 In-dist (P) / Out-dist (Q) 32x32 16x16 8x8

CIFAR-10 / SVHN 0.0716 0.3586 0.4512 SVHN / CIFAR-10 0.9902 0.9777 0.9195

CIFAR-100 / SVHN 0.0846 0.4448 0.3918 SVHN / CIFAR-100 0.9900 0.9798 0.9481

CelebA / SVHN 0.1541 0.3056 0.7037 SVHN / CelebA 0.9850 0.9968 0.9982

CIFAR-100 / CIFAR-10 0.4857 0.4933 0.5016 CIFAR-10 / CIFAR-100 0.5259 0.5446 0.5567

CelebA / CIFAR-10 0.7481 0.7137 0.7557 CIFAR-10 / CelebA 0.5087 0.6181 0.6751

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 Surprisingly, we can see that there are cases where the AUROC exceeds 0.5 when reducing the size of the two images in Table 3 (see CelebA vs SVHN case). In addition, when SVHN is set to in-distribution and CelebA is set to out-of-distribution, the performance tends to increase as the dimension decreases, which is a result that conflicts with the theorems in Appendix D. We argue that this is because resizing an image via bilinear interpolation strengthens the correlation between image pixels, which significantly reduces the entropy of the image distribution where each texture is complex (i.e., high entropy). Therefore, although it is difficult to confirm the effect of dimension on AUROC through the methodology, it can be seen that not only can the performance be improved by the simple image resize methodology for cases where likelihood inversion occurs, but also it is possible to increase the AUROC to more than 0.5. We reported experimental setting of Table 2 and the impact of dimensionality when simply applying PCA, and the effect of dimension in real tabular data in Appendix C.4. In addition, we conducted an experiment to distinguish between two Gaussian distributions with different means using a likelihood test using a NICE and RealNVP consisting of ReLU-like functions, and found that as the dimension increases, the AUROC approaches 0.5. Since this phenomenon is also a case where AUROC seriously decreases simply as the dimension increases, we present experiments and our theoretical analysis about flow's latent vector in high dimensional space in Appendix C.

![7_image_0.png](7_image_0.png)

## 5.2 Feature Correlation Perspective

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431 In prior work, Kirichenko et al. (2020) showed that likelihood inversion can arise in normalizing flows on image data due to strong local pixel correlations. Meanwhile, Schirrmeister et al. (2020)
reported that, in the image domain, OOD detection performance can improve when flow architectures use multi-layer perceptron (MLP) rather than convolutional neural network (CNN)in certain settings. As noted by Battaglia et al. (2018), CNNs exhibit an inductive bias known as locality, making them particularly effective for image data where local pixel correlations are strong. In contrast, MLPs have a weak inductive bias and are thus more suitable for tabular data, where no strong correlation between features is assumed. Leveraging Fact 1.2 and noting that architectural inductive bias is shaped by domain-specific characteristics, we argue that the counterintuitive phenomenon is infrequent in tabular data. This is because tabular features are inherently heterogeneous, unlike the homogeneous features of image data. Images have homogeneity because all features are composed of the same type of "pixel" and have values in the same interval from 0 to 255 and local pixels have strong correlations. However, tabular features are composed of many types such as continuous, discrete, and categorical, and the interval (or category) of these values can be set arbitrarily, and no assumptions are made between features so it has heterogeneity. Hence, we quantify heterogeneity and homogeneity to interpret the differences between the two domains from the perspective of correlation. To quantify feature heterogeneity and homogeneity, we measure overall feature correlation, which captures the strength of relationships between features. This is because correlation indicates the strength of the relationship between features, and if this strength is high, it can be interpreted as a strong tendency to follow a specific pattern (e.g., non-linear relationship), making it possible to determine whether the features are heterogeneous or homogeneous. However, since common feature correlation methodologies (e.g., Pearson correlation coefficient) are based on the relationship between two features, they do not capture global correlation structure.

Hence, we quantify correlation indirectly by showing that for data x ∈ R
das the strength of the correlation between features increases, the intrinsic dimension (ID) d
′ becomes smaller relative to the ambient dimension d. Please refer to Camastra & Staiano (2016) for a definition of ID. Since it is impossible to know the exact ID, we estimate the ID using MLE (Levina & Bickel, 2004) and TwoNN (Facco et al., 2017), which are popular methods for estimating the ID based on fractal theory. First, we demonstrate our claimed relationship between correlation and ID using random variables that follow a Gaussian distribution as a toy example. Let X ∼ N (0, Σ), Σ can be organized into an 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 Table 4: (Top) : ID estimates for real dataset. (Bottom) : The ratio of datasets with a rank more than or equal to 3 according to the d Ratio. d Ratio is the ratio of intrinsic dimension estimated by TwoNN to ambient dimension. Additionally, the results about image dataset were recorded with reference to the results recorded in Pope et al. (2021).

| Dataset                | MNIST   | CIFAR-10   | CIFAR-100   | SVHN   | magicgamma   | satellite   | landsat   | waveform   |
|------------------------|---------|------------|-------------|--------|--------------|-------------|-----------|------------|
| MLE                    | 13      | 26         | 23          | 19     | 7            | 12          | 11        | 16         |
| TwoNN                  | 15      | 11         | 9           | 7      | 7            | 15          | 14        | 17         |
| d Ratio                | 0.019   | 0.003      | 0.002       | 0.002  | 0.700        | 0.417       | 0.389     | 0.810      |
| d Ratio Threshold      | 0.1     | 0.2        | 0.3         | 0.4    | 0.5          | 0.6         | 0.7       | 0.8        |
| Rank ≥ 3 Dataset Ratio | 0.160   | 0.440      | 0.640       | 0.760  | 0.840        | 0.840       | 0.92      | 1.000      |

d-dimensional autoregressive covariance structure and represented by a formula as follows

$$\Sigma=1$$
$$\left[\begin{array}{l l l l l}{{1}}&{{}}&{{\rho}}&{{\rho^{2}}}&{{\cdots}}&{{\rho^{d-1}}}\\ {{\rho}}&{{}}&{{1}}&{{\rho}}&{{\cdots}}&{{\rho^{d-2}}}\\ {{}}&{{}}&{{}}&{{\vdots}}&{{}}\\ {{\rho^{d-1}}}&{{\rho^{d-2}}}&{{\rho^{d-3}}}&{{\cdots}}&{{1}}\end{array}\right],\;\rho\in[0,1].$$
$$({\boldsymbol{\Sigma}})$$
, ρ ∈ [0, 1]. (5)
We set the covariance Σ as in Equation 5 so that adjusting ρ controls the strength of correlations among variables. If ρ is closer to 1, the correlation between each variable will be stronger. Then, we use TwoNN and MLE (k = 10) ID estimators to describe the change in ID according to ρ in left and center subplot of Figure 1 for X with an ambient dimension of 10 and 50. Through left and center subplot of Figure 1, we can interpret that both ID estimators estimate smaller ID values when ρ increases. Therefore, it can be seen that stronger correlation between variables leads the ID to take values considerably smaller than the ambient dimension. The plot on the center of Figure 1 shows that even when ρ = 0, it underestimates the ID, but this is because the estimator tends to underestimate the ID when it has a large truth ID, so it is reasonable to look at the estimate as a lower bound when dataset has a large truth ID (Ansuini et al., 2019; Sharma & Kaplan, 2022). Based on the results obtained from the preceding experiments, we define the ratio of the intrinsic dimension to the ambient dimension as the d Ratio to measure the overall feature correlation. A higher degree of feature correlation results in a lower intrinsic dimension estimate, and thus a smaller d Ratio, whereas weaker correlation yields a larger d Ratio. Subsequently, to validate the findings from synthetic data experiments on real datasets, we estimate the ID of real-world image and tabular datasets and compare these estimates to their corresponding ambient dimensions. To compare this, we report ID estimates in Table 4 using MNIST, CIFAR-10, CIFAR-100, SVHN image datasets, which are mainly used in image benchmarks and ADBench's dataset, using MLE (k = 20) and TwoNN. According to Table 4, all four image datasets have a d Ratio of about 1%, whereas the tabular datasets exhibit substantially higher d Ratio values compared to the images. Additionally, we recorded a log-scale plot with each dimension as the axis in Figure 1 to check the tendency of the ID estimation values and ambient dimensions of the tabular and image datasets. As a result of comparing the average distance between the green line in Figure 1, it is perceived that the image has a larger average distance than the tabular. In addition to the numerical results, it can be seen visually that the blue points (tabular dataset) are much closer from the green line than the red points (image dataset), and we can see that the yellow points follow N (0, Id) which are theoretically perfectly uncorrelated data points are formed close to the green line. Through these results, it can be seen that tabular data has an ID closer to the ambient dimension than image data, so it can be concluded that they exhibit a lower correlation between features than image data.

Furthermore, for 25 datasets where NF-SLT does not achieve top performance (rank ≥ 3), we show the fraction of datasets with a d Ratio below a certain threshold in Table 4. These experimental results show that NF-SLT fails to achieve high performance on most datasets with low d Ratio, even within the tabular domain. Therefore, we conclude that one factor behind the high detection performance of tabular data is the heterogeneous nature of its features. We further argue that this effect may act 486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 in combination with the improvement in anomaly detection performance obtained using MLPs, as reported in (Schirrmeister et al., 2020). Moreover, to account for the absence of counterintuitive phenomena on CV/NLP embeddings in Section 4, we estimate the intrinsic dimension of the ADBench CIFAR-10 and SVHN embedding representations using TwoNN. The estimated intrinsic dimensions are 23 and 18, respectively, whereas the ambient embedding dimension is 1000 (smaller than the 3072-dimensional raw pixel space). Despite this reduced ambient dimension, the embeddings exhibit higher intrinsic dimensionality than the original images, implying a larger d Ratio. This suggests that the embedding features are less strongly correlated and span a higher-dimensional manifold than raw pixels. Consequently, the high-dimensional issues that degrade likelihood ranking are mitigated, making NF-SLT effective on these embeddings. This explanation is consistent with Kirichenko et al. (2020), which reported that the counterintuitive phenomenon is alleviated when using semantic embedding representations instead of raw pixels. Hence, unlike images, the tabular domain generally has low feature correlation, which contributes to its heterogeneous nature and makes it difficult to satisfy the assumption in Definition 3.3 that the generative model's relative performance should be low.

## 6 Conclusion

This paper examined whether the counterintuitive phenomenon in image anomaly detection also appears in tabular data. We first provided a domain-agnostic definition of this phenomenon, allowing it to be analyzed consistently across different data types. Using theoretical and empirical analyses with extensive experiments, we showed that this phenomenon rarely occurs in tabular data with simple likelihood tests using normalizing flows. Our results show that flow-based likelihood tests effectively detect tabular anomalies, outperforming traditional models without facing image domain challenges. For future work, we hope to see the development of flow architectures that can better capture semantic information in tabular data, as well as theoretical and empirical studies that extend these methods to high-dimensional tabular datasets with correlation structures comparable to those in image data.

## Reproducibility Statement

To ensure the reproducibility of our work, we provide details on our experimental setup. Appendix C.4 and F provide information necessary for reproducibility, including the sources of the comparison models and the hyperparameters used in the experiments. We also provide code of NF-SLT for reproducing our experiments in the supplementary materials.

## References

Kjersti Aas, Claudia Czado, Arnoldo Frigessi, and Henrik Bakken. Pair-copula constructions of multiple dependence. *Insurance: Mathematics and economics*, 44(2):182–198, 2009.

Krizhevsky Alex. Learning multiple layers of features from tiny images. https://www. cs. toronto.

edu/kriz/learning-features-2009-TR. pdf, 2009.

Alessio Ansuini, Alessandro Laio, Jakob H Macke, and Davide Zoccolan. Intrinsic dimension of data representations in deep neural networks. *Advances in Neural Information Processing Systems*, 32, 2019.

Milla Anttila, Keith Ball, and Irini Perissinaki. The central limit problem for convex bodies. Transactions of the American Mathematical Society, 355(12):4723–4735, 2003.

Amirhossein Ahmadian, Fredrik Lindsten, and Zhi-Hua Zhou. Likelihood-free out-of-distribution detection with invertible generative models. In *IJCAI*, pp. 2119–2125, 2021.

Peter W Battaglia, Jessica B Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, et al. Relational inductive biases, deep learning, and graph networks. arxiv 2018. arXiv preprint arXiv:1806.01261, 2018.

Jens Behrmann, Will Grathwohl, Ricky TQ Chen, David Duvenaud, and Jörn-Henrik Jacobsen.

Invertible residual networks. In *International conference on machine learning*, pp. 573–582. PMLR, 2019.

Liron Bergman and Yedid Hoshen. Classification-based anomaly detection for general data. In International Conference on Learning Representations, 2020. URL https://openreview. net/forum?id=H1lK_lBtvS.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Markus M Breunig, Hans-Peter Kriegel, Raymond T Ng, and Jörg Sander. Lof: identifying densitybased local outliers. In *Proceedings of the 2000 ACM SIGMOD international conference on* Management of data, pp. 93–104, 2000.

Francesco Camastra and Antonino Staiano. Intrinsic dimension estimation: Advances and open problems. *Information Sciences*, 328:26–41, 2016.

Anthony L Caterini and Gabriel Loaiza-Ganem. Entropic issues in likelihood-based ood detection.

In *I (Still) Can't Believe It's Not Better! Workshop at NeurIPS 2021*, pp. 21–26. PMLR, 2022.

Ricky TQ Chen, Jens Behrmann, David K Duvenaud, and Jörn-Henrik Jacobsen. Residual flows for invertible generative modeling. *Advances in Neural Information Processing Systems*, 32, 2019.

Yuansi Chen. An almost constant lower bound of the isoperimetric coefficient in the kls conjecture.

Geometric and Functional Analysis, 31:34–61, 2021.

Thomas M Cover. *Elements of information theory*. John Wiley & Sons, 1999. Laurent Dinh, David Krueger, and Yoshua Bengio. Nice: Non-linear independent components estimation. In *International Conference on Learning Representations (ICLR) Workshop*, 2015. URL https://arxiv.org/abs/1410.8516.

Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density estimation using real nvp. In International Conference on Learning Representations, 2017.

Felix Draxler, Stefan Wahl, Christoph Schnoerr, and Ullrich Koethe. On the universality of volumepreserving and coupling-based normalizing flows. In International Conference on Machine Learning, pp. 11613–11641. PMLR, 2024.

Conor Durkan, Artur Bekasov, Iain Murray, and George Papamakarios. Neural spline flows. Advances in neural information processing systems, 32, 2019.

Ronen Eldan. Thin shell implies spectral gap up to polylog via a stochastic localization scheme.

Geometric and Functional Analysis, 23(2):532–569, 2013.

Elena Facco, Maria d'Errico, Alex Rodriguez, and Alessandro Laio. Estimating the intrinsic dimension of datasets by a minimal neighborhood information. *Scientific reports*, 7(1):12140, 2017.

Bruno Fleury, Olivier Guédon, and Grigoris Paouris. A stability result for mean width of lp-centroid bodies. *Advances in Mathematics*, 214(2):865–877, 2007.

Izhak Golan and Ran El-Yaniv. Deep anomaly detection using geometric transformations. Advances in neural information processing systems, 31, 2018.

Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020.

Olivier Guédon. Concentration phenomena in high dimensional geometry. In *ESAIM: Proceedings*,
volume 44, pp. 47–60. EDP Sciences, 2014.

Olivier Guédon and Emanuel Milman. Interpolating thin-shell and sharp large-deviation estimates for lsotropic log-concave measures. *Geometric and Functional Analysis*, 21(5):1043–1068, 2011.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 Songqiao Han, Xiyang Hu, Hailiang Huang, Minqi Jiang, and Yue Zhao. Adbench: Anomaly detection benchmark. *Advances in Neural Information Processing Systems*, 35:32142–32159, 2022.

Sonja Hänzelmann, Robert Castelo, and Justin Guinney. Gsva: gene set variation analysis for microarray and rna-seq data. *BMC bioinformatics*, 14:1–15, 2013.

Trevor Hastie. The elements of statistical learning: data mining, inference, and prediction, 2009. Aapo Hyvärinen and Erkki Oja. Independent component analysis: algorithms and applications.

Neural networks, 13(4-5):411–430, 2000.

Hamidreza Kamkari, Brendan Leigh Ross, Jesse C. Cresswell, Anthony L. Caterini, Rahul Krishnan, and Gabriel Loaiza-Ganem. A geometric explanation of the likelihood OOD detection paradox. In Forty-first International Conference on Machine Learning, 2024. URL https://openreview.

net/forum?id=EVMzCKLpdD.

Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. In *International Conference* on Learning Representations (ICLR), Conference Track Proceedings, 2014. URL https:// arxiv.org/abs/1312.6114.

Durk P Kingma and Prafulla Dhariwal. Glow: Generative flow with invertible 1x1 convolutions.

Advances in neural information processing systems, 31, 2018.

Polina Kirichenko, Pavel Izmailov, and Andrew G Wilson. Why normalizing flows fail to detect out-of-distribution data. *Advances in neural information processing systems*, 33:20578–20589, 2020.

Bo'az Klartag. A central limit theorem for convex sets. *Inventiones mathematicae*, 168(1):91–131, 2007.

Charline Le Lan and Laurent Dinh. Perfect density models cannot guarantee anomaly detection.

Entropy, 23(12):1690, 2021.

Elizaveta Levina and Peter Bickel. Maximum likelihood estimation of intrinsic dimension. Advances in neural information processing systems, 17, 2004.

Zheng Li, Yue Zhao, Nicola Botta, Cezar Ionescu, and Xiyang Hu. Copod: copula-based outlier detection. In *2020 IEEE international conference on data mining (ICDM)*, pp. 1118–1123. IEEE, 2020.

Zheng Li, Yue Zhao, Xiyang Hu, Nicola Botta, Cezar Ionescu, and George H Chen. Ecod: Unsupervised outlier detection using empirical cumulative distribution functions. IEEE Transactions on Knowledge and Data Engineering, 35(12):12181–12193, 2022.

Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou. Isolation forest. In *2008 eighth ieee international* conference on data mining, pp. 413–422. IEEE, 2008.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017. URL https://openreview. net/forum?id=Skq89Scxx.

Sébastien Marcel and Yann Rodriguez. Torchvision the machine-vision package of torch. In Proceedings of the 18th ACM international conference on Multimedia, pp. 1485–1488, 2010.

Glenn W Milligan. An algorithm for generating artificial test clusters. *Psychometrika*, 50:123–127, 1985.

Warren Morningstar, Cusuh Ham, Andrew Gallagher, Balaji Lakshminarayanan, Alex Alemi, and Joshua Dillon. Density of states estimation for out of distribution detection. In *International* Conference on Artificial Intelligence and Statistics, pp. 3232–3240. PMLR, 2021.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.