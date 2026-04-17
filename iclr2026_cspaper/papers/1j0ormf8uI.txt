# Conformalized Survival Counterfactual Pre- Diction For General Right-Censored Data

Sijie Ren1,2, Meng Yan4,5, Zhen Zhang3,5, Yinghui Xu1**, Xinwei Sun**2 ∗ 1Artificial Intelligence Innovation and Incubation Institute, Fudan University 2School of Data Science, Fudan University 3Zhejiang Key Laboratory of Particle Radiotherapy Equipment, Zhejiang Cancer Hospital 4Department of Radiation Oncology, Tianjin Medical University Cancer Institute & Hospital 5Department of Radiation Oncology (Maastro), Maastricht University Medical Centre+
sjren23@m.fudan.edu.cn,yanmeng1999@tmu.edu.cn, zhen.zhang@maastro.nl,{xuyinghui,sunxinwei}@fudan.edu.cn,

## Abstract

This paper aims to develop a lower prediction bound (LPB) for survival time across different treatments in the general right-censored setting. Although previous methods have utilized conformal prediction to construct the LPB, their resulting prediction sets provide only probably approximately correct (PAC)–type miscoverage guarantees rather than exact ones. To address this problem, we propose a new calibration procedure under the potential outcome framework. Under the strong ignorability assumption, we propose a reweighting scheme that can transform the problem into a weighted conformal inference problem, allowing an LPB to be obtained via quantile regression with an exact miscoverage guarantee. Furthermore, our procedure is doubly robust against model misspecification. Empirical evaluations on synthetic and real-world clinical data demonstrate the validity and informativeness of our constructed LPBs, which indicate the potential of our analytical benchmark for comparing and selecting personalized treatments.

## 1 Introduction

Predicting survival time under specific treatment is of great importance in making correct predictions in high-stakes domains for healthcare Obermeyer et al. (2019); Navarro et al. (2021). For example, predicting the survival time of lung cancer under different radiochemotherapy regimens is crucial for tailoring therapies to individual patients Wilson et al. (2021); Horne et al. (2024), optimizing outcomes while minimizing side effects. This problem can be formulated as predicting the conditional expectations of potential outcomes under different treatment regimes Rubin (2005). However, survival data are often right-censored Klein & Moeschberger (2006); Wilson et al. (2021), meaning the true survival time is not fully observed for some patients. Unlike traditional causal inference, where outcomes are fully observed, right-censored data pose challenges due to the partial information on survival times. While many previous methods have been proposed to predict the survival function (Cox, 1972; Murphy et al., 1997; Tibshirani, 1997; Gui & Li, 2005; Steyerberg, 2016), they often rely on model assumptions that are difficult to verify, limiting their ability to provide reliable uncertainty quantification. In contrast, providing a Lower Predictive Bound (LPB) for survival analysis, instead of predicting the entire survival function, offers enhanced reliability and robustness. LPBs are particularly effective at handling censoring and provide a conservative estimate, making them more suitable for high-risk decision-making. This is especially crucial in clinical contexts, where overly optimistic predictions can lead to suboptimal or even harmful treatment choices. To ensure reliable Lower Prediction Bounds (LPBs), conformal prediction methods Vovk et al. (2005) have been applied to right-censored survival data Candes et al. (2023). Recent works have im- ` proved the calibration process by incorporating additional data points. For instance, Qi et al. (2024)
∗Corresponding author.

assigned estimated values to censored data based on their true event times, while Gui et al. (2024) introduced a covariate-dependent, data-adaptive censoring time to account for the heterogeneity in the censoring mechanism for Type-I right-censored data with a PAC-type guarantee. Additionally, Davidov et al. (2025) extended this framework to handle general right-censored data. However, these works have two main limitations. First, they do not provide an LPB for the treatment effect on survival time. For example, while Candes et al. (2023) predicted counterfactual ` survival times under different conditions, this was only applicable to cases where the censoring time exceeded a specific threshold. Furthermore, the LPB provided by these works Gui et al. (2024); Davidov et al. (2025) only offers a *Probably-Approximately-Correct* (PAC)-type guarantee, which does not ensure that the prediction is accurate for the entire population. In contrast, the marginal coverage guarantees more reliable and safer predictions across the entire population, including rare and extreme cases, which is crucial in high-stakes clinical scenarios MacDonald et al. (2025). To bridge this gap, this paper introduces *conformalized survival counterfactuals prediction*, a novel approach providing exact marginally valid LPB for counterfactual survival outcomes for general right-censored data. At its core, this approach provides an upper bound of a pre-specified nominal level α, through a non-conformity score defined via the counterfactual quantile regression function.

This upper bound is identifiable, and can be further calibrated using weighted conformal prediction (Lei & Candes, 2021; Jin et al., 2023). We can show that such an LPB can achieve the marginal ` coverage, provided that the weight function can be well estimated. We also provide the doubly robustness property. We demonstrate the validity and the effectiveness of our method on synthetic data and an in-house lung cancer dataset. We observe that the LPB provided by our method is valid and less conservative than other methods. Besides, the LPB under different radiochemotherapy regimens varies across diverse patient populations in lung cancer, offering valuable insights into personalized treatment strategies. To summarize, the contributions of this paper can be listed as follows:
- *Survival counterfactual prediction:* We propose a new procedure for quantifying the uncertainty of counterfactual survival time predictions under different treatments in general right-censored data. Our procedure establishes an upper bound for the miscoverage rate that can be reliably identified and further calibrated through weighted conformal prediction.

- *Theoretical guarantee:* We provide a distribution-free exact guarantee for the counterfactual prediction set and quantify the error from weight estimation. We also provide the doubly robustness property.

- *Empirical validation:* We validate our procedure using both synthetic and real clinical data.

On the synthetic data, we show that our calibration process yields less conservative LPBs while maintaining the desired coverage guarantee. On the clinical data, we demonstrate its effectiveness in identifying optimal treatments across diverse populations.

## 2 Related Work

Conformal prediction Vovk et al. (2005) was widely used for providing a reliable LPB in survival analysis Candes et al. (2023); Qi et al. (2024); Meixide et al. (2024); Qin et al. (2025) and counterfac- ` tual inference Lei & Candes (2021); Jin et al. (2023); Cand ` es et al. (2023); Deshpande & Kuleshov ` (2024). The conformal prediction for survival analysis was first considered by Candes et al. (2023) ` for the Type-I right-censored data. Qi et al. (2024) assigned a "best-guess" (BG) value as a surrogate of censored data for their true event times, of which performance is heavily affected by the quality of the imputation. Although distribution-free methods have been proposed in Meixide et al. (2024) and Qin et al. (2025) for constructing LPBs in the general right-censored setting, the validity of their bootstrap-based approaches depends on asymptotic results under specific regularity conditions. Gui et al. (2024) offered more informative LPBs for Type-I right-censored data by employing the adaptive cutoff method by tuning a hyperparameter for quantile regression using holdout calibration data to attain the desired coverage rate, and Davidov et al. (2025) extended this framework to the general right-censored setting with a well-designed data selection strategy for calibration, but both with PAC-type guarantees. However, none of the above works achieves an exact guarantee on general right-censored data. Candes et al. (2023); Gui et al. (2024) addressed survival analysis under a Type-I right-censored `
setting, which assumes that the censoring time Ciis known and relies on the availability of Ci.

Davidov et al. (2025) considered the general right-censored setting, but with a PAC-type guarantee. In this work, we *firstly* establish a conformalized procedure for survival counterfactuals prediction for general right-censored data with exact coverage guarantee of the LPB.

## 3 Preliminary

Throughout the paper, we focus on the potential outcome framework Rubin (1974); Splawa-Neyman et al. (1990) with different treatments. In particular, this paper focuses on the under-treated general right-censored data setting. Specifically, suppose we have {Wi, Xi, T̃i, ei}
N
i=1, where W, X, *T , e* ̃
denotes the treatment, the vector of covariates, the observed censored survival time T̃ ∶= min(*T, C*), and the binary indicator e = 1{T < C}, with *T, C* respectively denoting the true survival time and the censoring time. This means that for each subject, we can observe either the censoring time or the survival time, but not both, depending on which event occurs first. For simplicity, we consider binary treatments, *i.e.*, W ∈ {0, 1}.

Denote by (T(1), T(0)) the pair of potential outcomes. We assume that
(Wi, Xi, Ti(1), Ti(0), ei)
i.i.d.

∼ 
P(W, X, T(1), T(0), e).

Under the *stable unit treatment value assumption* (SUTVA) condition Rubin (1990), the observed Ti equals Ti(1) if Wi = 1, and Ti(0) if Wi = 0 for each i. To proceed, we further require the ignorability and overlap assumptions, which are standard in causal survival analysis Kalbfleisch & Prentice (2002); Candes et al. (2023). `
Assumption 3.1 (Ignorability). {T(1), T(0)} ⊥⊥ (*W, C*)∣X.

Remark 3.2. In addition to the treatment W, we assume that the potential outcome (*a.k.a*, true survival time) is also independent of the censoring time. This has been similarly assumed in Kalbfleisch & Prentice (2002) to achieve identifiability.

Assumption 3.3 (Overlap). 0 < P(Wi = 1∣X) < 1.

Given a nominal coverage level α ∈ (0, 1), our goal is to provide the lower predictive bound (LPB)
Lˆ(w)
N,n(X) for any treatment w, ensuring that it satisfies the marginal coverage.

$\langle\pi T\ell\rangle$
PX,T(w)(T(w) ≥ Lˆ(w)
N,n(X)) ≥ 1 − α, where *N, n* are the sample sizes of the training and calibration data. To provide the LPB, previous works Gui et al. (2024); Davidov et al. (2025) introduced adaptive cut-off methods. These methods offered LPBs that achieve probably-approximately-correct (PAC)- type coverage, which is approximately marginal coverage based on the available data. However, their approaches may fail to achieve exact marginal coverage that considers the average across the whole population, including extreme cases that are omitted by PAC-type coverage, yet are crucial in survival analysis. Specifically, Gui et al. (2024) considered the type-I censoring scenario, where we can observe
{Xi, T̃i, Ci}i. Given the estimated quantile regression function q̂τ (X), they propose to search ̂τ to meet the coverage guarantee, where qτ is the true τ -th quantile we want to estimate. First, they define α(τ ) such that

$\alpha(\tau):=\mathbb{P}(T\leq\widehat{q}_{\tau}(X))$  $$\approx\frac{\mathbb{E}\left[1\{T<\widehat{q}_{\tau}(X)\leq C\}\widehat{w}_{\tau}(X)\right]}{\mathbb{E}\left[1\{\widehat{q}_{\tau}(X)\leq C\}\widehat{w}_{\tau}(X)\right]}$$ $$\stackrel{{(1)}}{{\approx}}\frac{\sum_{i\in\mathcal{I}_{\mathrm{cai}}}\widehat{w}_{\tau}(X_{i})1\{T_{i}<\widehat{q}_{\tau}(X_{i})\leq C_{i}\}}{\sum_{i\in\mathcal{I}_{\mathrm{cai}}}\widehat{w}_{\tau}(X_{i})1\{\widehat{q}_{\tau}(X_{i})\leq C_{i}\}}\stackrel{{\Delta}}{{=}}\widehat{\alpha}(\tau),$$
where Ical denotes the index set of calibration data, and ŵτ (x) is chosen to be approximately equal to 1/P(q̂τ (X) ≤ C∣X = x). Then, they adaptively choose a cut-off value ̂τ , such that
̂τ ∶= sup{τ ∈ [0, 1] ∶ supτ
′≤τ α̂(τ
′) ≤ α}. Later, Davidov et al. (2025) extended this adaptive Algorithm 1 Conformal Survival Counterfactual Prediction.

Input: : Data D = {Wi, Xi, T̃i, ei}, counterfactual quantile regression estimator q̂
(w)
τ of T(w)∣X =
x and function ω̂(x) to fit the weight function; level α, testing point x.

1: Split the data into two folds: the training fold Dtr and the calibration fold Dcal.

2: Define the non-conformity score function: V
(w)
i∶= V
(w)(Xi, T̃i) = q̂
(w)
τ − T̃i.

3: Define I
(w)
cal = {i ∶ (Xi,Wi, T̃i, ei) ∈ Dcal with Wi = *w, e*i = 1}.

4: Compute the V
(w)
ifor each i ∈ I
(w)
cal 
.

5: Compute the weight ω̂(Xi) = 1/̂γ(Xi) for each i ∈ I
(w)
cal .

6: Compute the weights p̂i(x) and p̂∞(x) from (2).

7: Compute the c
(w)
1−α(τ ) = Quantile {1 − α; ∑i∈I
(w)
cal p̂i(x)δV
(w)
i+ p̂∞(x)δ∞}
Output: : The calibrated Conformal Survival Counterfactual Prediction LPB: L̂(w)
N,n(*X, τ* ) =
q̂
(w)
τ (X) − c
(w)
1−α(τ ).

cut-off to a general right-censored setting. However, this method can only achieve approximate marginal coverage due to "(1)" that the empirical average can approximate the population average with high probability. This motivates us to provide a new calibration procedure, which can achieve exact marginal coverage.

4 METHOD
In this section, we introduce our calibration procedure for survival counterfactual prediction. The core idea of our procedure lies in transforming the coverage probability into a reweighted expectation, and then applying a reweighting scheme for calibration. Section 4.1 introduces the procedure, and Section 4.2 provides the coverage guarantee.

## 4.1 Conformal Calibration

To begin, we split the dataset into a training data Dtr and a holdout calibration data Dcal. Let N = ∣Dtr∣, n = ∣Dcal∣. Correspondingly, we use Itr and Ical to respectively denote the index set of the training and calibration data. Counterfactual quantile regression function. Denote by q
(w)
τ (x) the true τ -th quantile of T(w)∣X = x and L
(w)
α (x) the oracle LPB for the corresponding α-th counterfactual conditional quantile function. Under the ignorability condition, we have P(T(w)∣X = x) = P(T(w)∣X = x,W = w) = P(T∣X = x,W = w).

That means, the goal is to estimate the τ -th quantile of T∣X = x,W = w from observed samples {T̃i, Xi,Wi} with T̃i∶= min{Ti, Ci}. We can then apply censored quantile regression (CQR) methods to estimate q
(w)
τ (x), including CQR Peng & Huang (2008), CQR forest Li & Bradic (2020),
and CQR neural networks Pearce et al. (2022). Non-conformity score. We follow Romano et al. (2019) to define the *non-conformity* score as V
(w)
i∶= V
(w)(Xi, T̃i) = q̂
(w)
τ (Xi) − T̃i on Dcal, indicating how atypical a value of the outcome is given observed covariate values, and a large value indicates a lack of conformity to training data. Let c
(w)
1−α(τ ) denotes the 1 − α quantile of V
(w), *i.e.*, c
(w)
1−α(τ ) ∶= infc{P(V
(w)(Xi, T̃ ≤ c) ≥ 1 − α}.

Calibration procedure. Note that the adaptive cut-off method Gui et al. (2024) adaptively adjusted the τ such that α(τ ) ∶= P(T < q̂τ (X)), because α(τ ) may not necessarily equal (or even close)
to α when q̂τ (X) does not estimate the τ -th quantile of T∣X (T∣X,W = w in our scenario) well.

Therefore, they used α̂(τ ), the empirical version of α(τ ) to learn the cut-off value. The gap between α̂(τ ) and α(τ ) makes their procedures achieve only approximate coverage.

To achieve the exact marginal coverage, instead of α(τ ) ∶= P(T < q̂τ (X)), we provide the upper bound for P(V
(w)(X, T̃) ≤ c
(w)
1−α(τ )), which is exactly equals to α and therefore avoid the need for

Table 1: Experiment results with different α within a reasonable range, with the corresponding optimized τ

∗, average coverage rate , and the corresponding optimal LPB. The results are evaluated

on 10 independent trials on setting 4 with the same ratio as shown in Figure 1.

α 0.05 0.10 0.15 0.20 τ

∗0.16 0.16 0.26 0.21

Average coverage rate 0.958 0.914 0.872 0.845

LPB with τ = α 0.411 0.778 1.19 1.57

Optimal LPB 0.503 0.803 1.25 1.64

adaptively adjustment. Specifically, we have:

α = P(V
(w)(X, T(w)) ≥ c
(w)
1−α(τ ))
= P(T(w) ≤ q̂
(w)
α (X) − c
(w)
1−α(τ ))
= E[P(T(w) ≤ q̂
(w)
α (x) − c
(w)
1−α(τ )∣X = x)]
(i)
= EX[P(T ≤ q̂
(w)
α (x) − c
(w)
1−α(τ )∣X = x,W = w)]
(ii)
= EX [P(T ≤ q̂
(w)
α (x) − c
(w)
1−α(τ )∣X = x,W = w)p(e = 1∣x,W = w)
1
p(e = 1∣x,W = w)]
(iii)
≤ 
EX [P(T ≤ q̂
(w)
α (x) − c
(w)
1−α(τ ), e = 1∣X = x,W = w)
1
p(e = 1∣x,W = w)
]
(iv)
= 
EX [P(T ≤ q̂
(w)
α (x) − c
(w)
1−α(τ )∣e = 1,W = *w, X* = x]
= E[P(T ≤ q̂
(w)
α (x) − c
(w)
1−α(τ ), e = 1,W = w∣X = x)
1
p(e = 1,W = w∣x)]
def
= E[I (T ≤ q̂
(w)
α (x) − c
(w)
1−α(τ ),W = *w, e* = 1)
1
γ(x)
]
= E[I (V (*T , X* ̃ ) ≥ c
(w)
1−α(τ )) 
p(W = *w, e* = 1)
γ(x)∣W = *w, e* = 1] , (1)
where (i) follows from Assumption 3.1 and the SUTVA condition, (ii) comes from the tower property, (iii) is derived by the proof of Lemma A.1 conditional on X = x,W = w, and γ(x) ∶= p(W =
w, e = 1∣x). Through the upper bound in (iv), note that it is sufficient for the LPB L̂(w)
N,n(*X, τ* )
to satisfy the coverage guarantee for PX × PT̃∣W=w,e=1,X (since T = T̃ given e = 1). Denote ω(x) =
p(W=w,e=1)
γ(x). Since EX∣W=w,e=1 [ω(X)∣W = *w, e* = 1] = 1, we can employ the weighted conformal prediction Lei & Candes (2021) to ensure that `

$$\mathrm{(1)}$$
$$\mathbb{E}\left[\mathbb{I}\left(V^{(w)}(\widetilde{T},X)\geq c_{1-\alpha}^{(w)}(\tau)\right)\omega(X)|W=w,e=1\right]\approx\alpha.$$

To this end, for each i ∈ Ical with W = w and e = 1, we compute ω̂(Xi) =
p(W=w,e=1)
̂γ(Xi). For the test data x, we compute ω̂(x) ∶=
p(W=w,e=1)
̂γ(x). Following Lei & Candes (2021), we can take `

$$c_{1-\alpha}^{(w)}(\tau)=\mathrm{Quantile}\left\{1-\alpha;\;\sum_{i\in{\mathcal{I}}_{2}^{(w)}}\widehat{p}_{i}(x)\delta_{V_{i}^{(w)}}+\widehat{p}_{\infty}(x)\delta_{\infty}\right\},$$

where

$$\overline{p}_{i}(x)=\frac{\overline{\omega}(X=x_{i})}{\sum_{i\in\mathcal{I}_{\text{col}}^{(v)}}\overline{\omega}(X=x_{i})+\overline{\omega}(x)},\ \overline{p}_{\infty}(x)=\frac{\overline{\omega}(x)}{\sum_{i\in\mathcal{I}_{\text{col}}^{(v)}}\overline{\omega}(X=x_{i})+\overline{\omega}(x)}\tag{2}$$

![5_image_0.png](5_image_0.png)

Since p(W = *w, e* = 1) can be canceled out, it is sufficient to only estimate ̂γ(x). Therefore, with a bit of abuse of notation, we denote ω̂(x) ∶=1
̂γ(x)
. The LPB can be then given by: L̂(w)
N,n(*X, τ* ) ∶=
q̂
(w)
τ (X) − c
(w)
1−α(τ ). The whole procedure is summarized in Algorithm 1.

LPB optimization. As we will demonstrate in the next subsection, our procedure yields a prediction set that satisfies the coverage guarantee for any τ ∈ (0, 1). To ensure the LPB is as informative as possible, we choose τ to maximize L̂(w)
N,n(X, τ ) for any test data X. Specifically, for each X = x, we obtain:

$$\tau^{*}(x):=\arg\operatorname*{max}_{\tau\in(0,1)}\left(\tilde{q}_{\tau}^{(w)}(x)-c_{1-\alpha}^{(w)}(\tau)(x)\right),$$

The optimized LPB on the test data x is given by L̂(w)
N,n(*X, τ* ∗(x)) = q̂

$$\cdot:\widetilde{q}_{\tau^{\star}}^{(w)}(x)-c_{1-\alpha}^{(w)}(\tau)(x).$$

4.2 THEORETICAL GUARANTEE

Our theoretical analysis builds on Lei & Candes (2021); Cand ` es et al. (2023). Specifically, by (iv) `
in (1), the problem reduces to constructing the LPB for the distribution PX × PT̃∣W=w,e=1,X, from
the data
$\left\{X_{i},\widetilde{T}_{i}\right\}_{i\in\mathcal{I}_{\mathrm{cal}},W_{i}=w,e_{i}=1}\sim\mathbb{P}_{X|W=w,e=1}\times\mathbb{P}_{\widetilde{T}|W=w,e=1,X}$  In, the weight function $\omega(x)$ is introduced such that 
 To achieve $\epsilon$. 
To achieve calibration, the weight function ω(x) is introduced such that
$\left\{X_i,\widetilde{T}_i\right\}_{i\in\mathcal{I}_{\mathrm{cal}},W_i=u}$  a, the weight function. 
$\omega(x)=\frac{d\mathbb{P}_{X}}{d\mathbb{P}_{X|W=w,e=1}}(x)=\frac{p(W=w,e=1)}{\gamma(x)}$.  
. (3)
We then propose to estimate ω̂(x) from the training data. Theorem 4.1 establishes a distributionfree exact guarantee for counterfactual prediction intervals under covariate shift, in which ω̂(x) is the estimated density ratio quantifying distribution shift via Radon-Nikodym derivative
(dPX/dPX∣W=w,e=1)(x).

Theorem 4.1. Let (Xi, T̃i)i∈Ical ,Wi=w,ei=1 i.i.d
∼ 
PX∣W=w,e=1 × PT(w)∣X,e=1. Set N = ∣Dtr∣ and n =
∣Dcal∣*. Further, let* q̂
(w)
α (x) = q̂
(w)
α (x;Dtr) be an estimate of the α*-th conditional quantile* q
(w)
α (x)
of T(w)∣X = x, ω̂(x) = ω̂(x;Dtr) be an estimate of ω(x) = (dPX/dPX∣W=w,e=1)(x)*, and* L̂(w)
N,n(x)
be the counterfactual LPB resulting from Algorithm 1. Assume that E[ω̂∣Dtr] < ∞, where E *denotes* expectation over X ∼ PX∣W=w,e=1*. Redefine* ω̂(x) as ω̂(x)/E[ω̂(x)∣Dtr] *so that* E[ω̂(X)∣Dtr] = 1.

Then we have:

_then we have:_  $$\mathbb{P}_{(X,T(w))\sim\mathbb{P}_{X\times\mathbb{P}_{T(w)\setminus X,w+1}}}\left(T(w)\geq\widehat{T}_{N,n}^{(w)}(X)\right)\geq1-\alpha-\frac{1}{2}\mathbb{E}_{X\sim\mathbb{P}_{X|W\sim w,w+1}}\left[|\varnothing(X)-\omega(X)|\right].\tag{4}$$
$$(3)$$

![6_image_0.png](6_image_0.png) 

This bound quantifies how estimation error in the density ratio (ω̂N (x)−ω(x)) affects the coverage
probability, with the error term diminishing as the density ratio estimation improves. The weight
estimator is normalized to ensure E[ω̂(X)∣Dtr] = 1, which stabilizes the variance and enables
tractable theoretical analysis. The result holds without strong distributional assumptions, maintaining the non-parametric spirit of conformal methods while extending them to counterfactual settings. The following theorem provides the doubly robust property for counterfactual prediction intervals. This theoretical guarantee ensures that our method maintains valid coverage even when either the
weights function ̂γ(x) or the counterfactual quantile estimator q̂
(w)(x) is misspecified, provided
one of them is consistently estimated. The doubly robust mechanism provides mutual compensation between the two estimation approaches: when the weights function is inaccurate, the quantile estimation compensates through Assumption A1, and vice versa, when the quantile estimation is misspecified, the weights function provides robustness through Assumption A2.
Theorem 4.2. Let N = ∣Dtr∣, n = ∣Dcal∣*, and* q̂
(w)
α (x) = q̂
(w)
α (x∣Dtr) denote the estimate of α-th
conditional quantile q
(w)
α (x) of T(w)∣X = x. Further, let ̂γ(x) = ̂γ(x∣Dtr) denote the estimate
of γ(x)*, and* L̂(w)
N,n(x) *denote the corresponding counterfactual LPB resulting from Algorithm 1.*
Assume that E[1/̂γ(X)∣Dtr] < ∞ and E[1/γ(X)] < ∞*. Assume that one of the following holds:*
A1 lim
N→∞
E[∣ 1
̂γ(x)−
1
γ(x)∣] = 0;
A2 (i) there exists r, b1, b2 > 0 such that P(T(w) = t∣X = x) ∈ [b1, b2] uniformly over all (*x, t*)
with t ∈ [q
(w)
α (x) − *r, q*
(w)
α (x) + r],
$$(i i)\;l e t\;{\mathcal{E}}_{N}(X)=|\widehat{q}_{\beta,N}^{(w)}(x)|$$
β,N (x) − q
(w)
β(x)∣, there exist δ > 0 *such that*
$$\mathbb{E}\left[\frac{1}{\widehat{\gamma}(x)^{1+\delta}}|\mathcal{D}_{\mathrm{tr}}\right]<\infty,\quad\lim_{N\to\infty}\left[\frac{\mathcal{E}_{N}(X)}{\widehat{\gamma}_{N}(x)}\right]=\lim_{N\to\infty}\left[\frac{\mathcal{E}_{N}(X)}{\gamma(x)}\right].$$  _Then under SUTVA and the strong ignorability assumption,_

$\mathcal{L}$
$$\left(S\right)$$
$$(\mathbf{6})$$
$$\operatorname*{lim}_{N,n\to\infty}\mathbb{P}(X,T(w)){\sim}\mathbb{P}_{X\times\mathbb{P}_{T(w)|X,\,n=1}}\left(T(w)\geq{\widehat{L}}_{N,n}^{(w)}(x)\right)\geq1-\alpha.$$
_Ids, then for any $\epsilon>0$,_
N,n(x)) ≥ 1 − α. (6)
Furthermore, if A*2 holds, then for any* ϵ > 0,
$$\operatorname*{lim}_{N,n\to\infty}\mathbb{P}_{X\sim\mathbb{P}_{X|W=w,\,\epsilon=1}}\left(\mathbb{P}(T(w)\geq\tilde{L}_{N,n}^{(w)}(x)|X)\leq1-\alpha-\epsilon\right)=0$$
N,n(x)∣X) ≤ 1 − α − ϵ) = 0 (7)
Theorem 4.2 is a special case of Corollary B.4 in Appendix B. It is easy to see that Theorem 4.2 applies to counterfactual predictions across all treatments. Moreover, property (7) implies that conformal survival counterfactual analysis has approximately guaranteed conditional coverage for counterfactuals if the conditional quantiles are estimated well.

## 5 Experiment

We evaluate both the coverage rate and the average LPB over the test set on synthetic data and real clinical data, with the desired coverage level 1−α to 90%. In Section 5.1, we compare different calibration procedures on different scenarios with simulation for censoring and treatment rates in real-world clinical trials. We further validate our method using some real clinical cases collected from a cancer hospital in Section 5.2.

$$(7)$$

![7_image_0.png](7_image_0.png) 

![7_image_1.png](7_image_1.png) 

![7_image_2.png](7_image_2.png) 

![7_image_3.png](7_image_3.png) 

## 5.1 Simulation

Data. Simulation is crucial for verifying the efficiency of performance, as the survival time T is often lacking. We test our method on different settings designed to mimic the censorship in realworld clinical trials, as in Candes et al. (2023); Gui et al. (2024); Davidov et al. (2025). With the `
details described in Appendix C.1.

Models. The model utilized is a multilayer perceptron (MLP) with only one hidden layer to reduce the potential overfitting, implemented in PyTorch. And we fit Random Forest classifiers to estimate the weights function ω(x). For more details, please refer to Appendix D.

Results. All methods aim to achieve a coverage rate of 1 − α = 90%, corresponding to the red dashed line shown in Figure 1. The larger the relative LPB, the more informative it is. The results of synthetic data are shown in Figure 1. In comparison, we evaluate our method against the *uncab* method applied without calibration, the *naive* calibrated method, the *focused* calibration method Davidov et al. (2025), and the *fused* calibration method in Davidov et al. (2025). Our method consistently achieves a more informative median-value LPB while ensuring it is closest to the desired coverage rates across all six experimental settings. Although the average coverage rate of our method slightly falls below 1 − α in setting 6, it remains remarkably close to the target level.

It simultaneously demonstrates the highest LPB among all methods that guarantee the desired coverage rates (including naive and focused methods). In settings 3, 4, and 5, our method achieves valid coverage rates comparable to the fused approach while yielding significantly larger LPB values. Although the resulting prediction intervals are wider, our method provides exact statistical guarantees for the coverage probability. Our method also achieves desired coverage with outliers. To verify the robustness of our method, we introduce outlier data (details in Appendix D.2) into Setting 4 and report the resulting coverage rates and LPB in Figure 3. The results show that our method consistently maintains the desired coverage guarantee, whereas the compared baselines fail to do so. In particular, those methods-"Focus" and "Fused" Davidov et al. (2025)-with PAC-type guarantee do not necessarily achieve the marginal coverage in the presence of outlier data.

Additional Experiment. Besides the comparative experiments above demonstrating the effectiveness and robustness of our method, some additional experiments are conducted in Appendix E. We first verify in Appendix E.1 that once the sample sizes reach a certain threshold, our method can stably achieve the coverage guarantee while producing less conservative LPB. In Appendix E.2, we further validate that the LPB of our method effectively captures adaptiveness to different covariates.

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

![8_image_2.png](8_image_2.png)

![8_image_3.png](8_image_3.png)

In Appendix E.3, we explore the impact of p(W = *w, e* = 1) on our method, with results consistent with the expectations derived from (3). Besides, Appendix E.4 and Appendix E.5 present sensitivity analysis with respect to the regression algorithm and weight function, respectively. The results show that our procedure consistently attains the desired coverage. In addition, we report the value of τ
∗selected by our optimization procedure, along with the corresponding coverage rate and LPB
in Table 1 and Figure 11. As shown in Figure 11, the LPB achieved at τ
∗is comparable to that at τ = α, indicating that the quantile regression model is well trained. Finally, we expand setting 4 to a multi-treatment scenario. Figure 2 shows that the LPB varies across treatments but consistently satisfies the coverage guarantee. Please refer to Appendix D.1 for additional details.

## 5.2 Application On Real Data

Data. We evaluate our method on a real-world dataset of 541 non-small cell lung cancer patients from a cancer hospital. Four different radiochemotherapy regimens in real data are examined, and the details of radiochemotherapy and the proportion of patients receiving different treatments is shown in Table 4. The dataset includes 124 clinical and quantitative radiomic features of lung cancer. Detail of the dataset is provided in Appendix C.2. Models. The model trained for real data is a multilayer perceptron (MLP) with three hidden layers, implemented in PyTorch. And we also fit Random Forest classifiers to estimate the weights function ω(x). For more details, please refer to Appendix D.

Results. To validate our method's performance and reliability in reality, we apply it to a realworld dataset stratified by different treatment regimens. First, we investigate the effects of four distinct radiochemotherapy regimens on survival time. As shown in the Figure 4, the result shows a higher median LPB than those treated under intensity modulated radiation therapy (IMRT), which is consistent with the VMAT's better clinical benefits in lung cancer Hunte et al. (2022). Besides, as for the three chemotherapies, the addition of induction chemotherapy and concurrent chemotherapy shows higher LPBs, consistent with prior studies Curran et al.; Aguado et al. (2022). Our analysis also reveals a higher LPB under consolidation chemotherapy, potentially due to more favorable baseline characteristics Liu et al.. Then, we explore the adaptiveness of the LPB on cases under the VMAT technique. We select a set of covariates from three categories of known prognostic factors for survival time, including three

![9_image_0.png](9_image_0.png)

tumor-related clinical factors (i.e., overall stage, T-stage, and N-stage), one host-related clinical factor (i.e., Karnofsky Performance Status (KPS)), and two quantitative radiomic features of the tumor (i.e., Max3D-Diameter, Voxel-Volume). To better assess how these covariates influence LPB, all covariates are binarized, as listed in Table 5. As shown in the Figure 5, patients with more advanced stages, larger quantitative radiomic features tend to have larger tumor burdens and greater lymph node involvement Aerts et al. (2014); Amin et al. (2017), and shorter survival time. The patients with better body performance status (i.e., larger KPS value) usually have better functional status with longer survival time Quinten et al. (2009). Additionally, Appendix E.6 implements baseline methods that satisfy coverage guarantees in the simulation settings, and reports the corresponding results in Figure 10. As shown, our method can produce higher LPB, suggesting the informativeness of our procedure.

The LPB of our method in real data consistently correlates with important factors while maintaining the desired coverage rate, which demonstrates the applicability of our approach in complex, heterogeneous real-world scenarios and the potential in supporting personalized clinical decision-making for tumor treatment.

## 6 Discussion

We introduce an uncertainty quantification procedure of counterfactuals with exact LPB coverage guarantees for general right-censored survival data. Under the SUTVA and strong ignorability assumptions, we achieve counterfactual prediction for a new test point by training a counterfactual quantile regressor and performing a weighted adjustment of the non-conformity score using the counterfactual calibration set. Although the assumptions employed in this paper are commonly used in causal inference, it isn't easy to guarantee that all these assumptions hold completely during real-world data collection processes. Therefore, in practical applications, appropriately constraining data that violate these assumptions in real datasets could enhance the robustness of our method in counterfactual prediction Oliveira et al. (2024); Feldman & Romano (2024). In actual follow-up studies, extreme scenarios such as imbalanced treatment usage proportions and high censoring rates may lead to inaccurate estimation of γ(x), consequently increasing the uncertainty in counterfactual outcome predictions under such treatments. Hence, mitigating the impact of data imbalance is crucial Gui et al. (2024). Furthermore, beyond demonstrating differences between treatments, accurate quantitative estimation of the causal effects between different treatment outcomes holds significant importance for decision-making Lopez & Gutman (2017); Hu & Gu (2021); Lei & Candes (2021). `

## Acknowledgement

This work was supported in part by the State Key Program of National Natural Science Foundation of China (Grant No.12331009), Young Scientists Fund of the National Natural Science Foundation of China (Grant No.KRH2305058), Funded by Tianjin Key Medical Discipline Construction Project (Grant No.TJYXZDXK-3-004B), and National Natural Science Foundation of China (No.82303672, No.82573437).

## Ethics Statement

The institutional ethics committee approves the usage of the clinical dataset, and all the cases are anonymized in this work. No specific information is provided to preserve anonymity.

## Reproducibility Statement

The proof of Theorem 4.1 and Theorem 4.2 is provided in Appendix A.1 and Appendix B.3 respectively. The generation details of synthetic data are provided in Appendix C.1. The model parameters and implementation are provided in Appendix D. Our code is available at conformalized survival counterfactual code.

## References

Hugo J. W. L. Aerts, Emmanuel Rios Velazquez, Ralph T. H. Leijenaar, Chintan Parmar, Patrick Grossmann, Sara Carvalho, Johan Bussink, Rene Monshouwer, Benjamin Haibe-Kains, Derek ´ Rietveld, Frank Hoebers, Michelle M. Rietbergen, C. Rene Leemans, Andre Dekker, John Quack- ´ enbush, Robert J. Gillies, and Philippe Lambin. Decoding Tumour Phenotype by Noninvasive Imaging Using a Quantitative Radiomics Approach. *Nature Communications*, 5(1):4006, 2014.

Carlos Aguado, Luis Chara, Monica Anto ´ nanzas, Jose Maria Matilla Gonzalez, Unai Jim ˜ enez, ´
Raul Hernanz, Xabier Mielgo-Rubio, Juan Carlos Trujillo-Reyes, and Felipe Counago. Neoad- ˜ juvant Treatment in Non-small Cell Lung Cancer: New Perspectives with the Incorporation of Immunotherapy. *World Journal of Clinical Oncology*, 13(5):314, 2022.

Mahul B. Amin, Frederick L. Greene, Stephen B. Edge, Carolyn C. Compton, Jeffrey E. Gershenwald, Robert K. Brookland, Laura Meyer, Donna M. Gress, David R. Byrd, and David P. Winchester. The Eighth Edition AJCC Cancer Staging Manual: Continuing to Build a Bridge from a Population-based to a More "personalized" Approach to Cancer Staging. CA: A Cancer Journal for Clinicians, 67(2):93–99, 2017.

Thomas B Berrett, Yi Wang, Rina Foygel Barber, and Richard J Samworth. The Conditional Permutation Test for Independence While Controlling for Confounders. *Journal of the Royal Statistical* Society Series B: Statistical Methodology, 82(1):175–197, 2020.

Emmanuel Candes, Lihua Lei, and Zhimei Ren. Conformalized Survival Analysis. ` Journal of the Royal Statistical Society Series B: Statistical Methodology, 85(1):24–45, 2023.

David R Cox. Regression Models and Life-tables. *Journal of the Royal Statistical Society: Series B*
(Methodological), 34(2):187–202, 1972.

Walter J. Curran, Rebecca Paulus, Corey J. Langer, Ritsuko Komaki, Jin S. Lee, Stephen Hauser, Benjamin Movsas, Todd Wasserman, Seth A. Rosenthal, Elizabeth Gore, Mitchell Machtay, William Sause, and James D. Cox. Sequential Vs. Concurrent Chemoradiation for Stage III Non-small Cell Lung Cancer: Randomized Phase III Trial RTOG 9410. *Journal of the National* Cancer Institute, 103(19):1452–1460.

Hen Davidov, Shai Feldman, Gil Shamai, Ron Kimmel, and Yaniv Romano. Conformalized Survival Analysis for General Right-Censored Data. In The Thirteenth International Conference on Learning Representations, 2025.

Shachi Deshpande and Volodymyr Kuleshov. Calibrated and Conformal Propensity Scores for Causal Effect Estimation. In *Conference on Uncertainty in Artificial Intelligence*, volume 2024, pp. 1083, 2024.

Shai Feldman and Yaniv Romano. Robust Conformal Prediction Using Privileged Information.

Advances in Neural Information Processing Systems, 37:117813–117852, 2024.

Jiang Gui and Hongzhe Li. Penalized Cox Regression Analysis in the High-dimensional and Lowsample Size Settings, with Applications to Microarray Gene Expression Data. *Bioinformatics*, 21 (13):3001–3008, 2005.

Yu Gui, Rohan Hore, Zhimei Ren, and Rina Foygel Barber. Conformalized Survival Analysis with Adaptive Cut-offs. *Biometrika*, 111(2):459–477, 2024.

Ashley Horne, Ken Harada, Katherine D. Brown, Kevin Lee Min Chua, Fiona McDonald, Gareth Price, Paul Martin Putora, Dominic G. Rothwell, and Corinne Faivre-Finn. Treatment Response Biomarkers: Working Toward Personalized Radiotherapy for Lung Cancer. Journal of Thoracic Oncology, 19(8):1164–1185, 2024.

Liangyuan Hu and Chenyang Gu. Estimation of Causal Effects of Multiple Treatments in Healthcare Database Studies with Rare Outcomes. *Health Services and Outcomes Research Methodology*, 21(3):287–308, 2021.