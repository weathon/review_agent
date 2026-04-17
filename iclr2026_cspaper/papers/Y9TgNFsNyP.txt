000 001 002 003 004 005 006 007 008 009 010 011 012 013 014 015 016 017 018 019 020 021 022 023 024 025 026 027 028 029 030 031 032 033 034 035 036 037 038 039 040 041 042 043 044 045 046 047 048 049 050 051 052 053 The Forward-Forward (FF) algorithms present promising and biologically plausible alternatives to backpropagation (BP), enabling efficient model training through layer-wise greedy optimization. However, the critical task of machine unlearning for FF models, which involves efficiently removing specific training data's influence without full retraining, remains a foundational yet unexplored problem. The inherent characteristics of FF models, such as their sensitivity to parameter tuning and layer-wise independent training, pose unique challenges, often causing catastrophic model collapse when applying conventional unlearning methods. To fill this gap, we introduce a novel unlearning framework specifically for FF models, which employs a goodness-guided strategy. This method proposes a stable guidance model to generate target goodness distributions, steering the original model to unlearn forgetting data by shifting its layer-wise goodness scores, thereby effectively adapting gradient-based unlearning for the FF architecture. To enable robust verification on unlearning performance, we also propose a novel goodnessbased membership inference attack (G-MIA), a powerful and lightweight blackbox attack that leverages the unique properties of FF models' goodness scores.

Our experiments demonstrate that our proposed method effectively removes the influence of target forgetting data on FF models while preserving model utility on the remaining data. Critically, our approach accomplishes 1.9 to 3.1× faster than retraining from scratch, establishing an efficient foundation for FF unlearning.

## 1 Introduction

The Forward-Forward (FF) Hinton (2022) algorithms have emerged as a promising alternative to backpropagation (BP) for training deep learning models. This approach updates model parameters by greedily optimizing a layer-wise "goodness" score, which reflects the activation level of neurons in a layer. By maximizing this score for positive data (i.e., valid training data with correct labels) and minimizing it for negative data (e.g., invalid data or incorrectly labeled data) during forwarding, the FF algorithms effectively train model parameters without requiring a backward pass that blocks all layers. This BP-free nature is considered more biologically plausible and brings significant practical advantages, including reduced memory overhead from stored activations and the potential for efficient training using pipeline parallelism. These features make FF particularly well-suited for training on resource-constrained scenarios, such as in edge computing. However, the critical task of machine unlearning for FF models remains a foundational yet unexplored problem. Usually, machine learning applications involve analyzing sensitive individuals' data. Their owners require the "right to be forgotten" (RTBF), which has been explicitly stated in the European Union General Data Protection Regulation (GDPR)Voigt & Von dem Bussche (2017) and the California Consumer Privacy Act (CCPA)Harding et al. (2019). Moreover, the model owners also need to remove outdated or poisoned data to promote model performanceWang et al. (2025b); Zhang et al. (2023). Machine unlearning achieves these data erasing goals by removing the influence of specific training samples from a trained model (*i.e.*, effectiveness) while preserving the model performance on the remaining data (*i.e.*, model utility).

Existing machine unlearning methods are not feasible for FF models. The most straightforward approach, retraining the model from scratch on the remaining data, is computationally prohibitive and impractical. Other unlearning methods calibrate the model parameters by either directly performing gradient ascent (GA) on the forgetting data Tarun et al. (2023a); Sekhari et al. (2021a) or estimating the parameters tuning Qiao et al. (2024); Liu et al. (2022b). As illustrated in Figure 1, they are also not applicable due to the unique challenges posed by the BP-free nature and layer-wise training of FF models. The specific details are as follows.

# Ff-Erase : Machine Unlearning And Verifica- Tion For Forward-Forward Models

Anonymous authors Paper under double-blind review

## Abstract

1

![1_image_0.png](1_image_0.png) 

Firstly, FF models exhibit heightened sensitivity to parameter tuning due to their BP-free nature. BP methods utilize backpropagation to ensure consistent parameter update directions, thereby enhancing robustness to tuning variations. In contrast, FF algorithms use greedy and layer-wise training approaches, where each layer is independently optimized on its local goodness objective until the overall goodness scores converge to a specific distribution. In this process, the parameters in the previous layers do not strictly update towards a consistent direction with the subsequent layers, nor compress everything "useful" for the final output layer. Therefore, without careful design to prevent goodness from shifting to invalid distributions, layers may diverge in update directions during unlearning, risking model collapse. However, determining the validity of a goodness distribution in advance remains challenging, making it difficult to reliably guide layer updates during unlearning.

Secondly, the independent layer-wise training of FF models further complicates the unlearning process. In BP models, a common unlearning strategy is to perform gradient ascent on the loss function of the data to be removed, updating all layers jointly through the chain rule Gupta et al. (2021); Tarun et al. (2023b); Sekhari et al. (2021b); Chundawat et al. (2023b). In contrast, FF models optimize separate objective functions at each layer, with varying degrees of goodness improvement. This independence creates a key difficulty: it is unclear how much each layer's goodness should be penalized given a forgetting data sample. As a result, some layers may continue to over-forget while others only partially retain residual effects, thereby complicating the trade-off between effective unlearning and preserving the overall model utility. The above discussion motivates us to answer the first key question: How to design an efficient machine unlearning method for FF models to ensure both effectiveness and model utility? Moreover, it is also challenging to verify the effectiveness of an unlearning algorithm on FF models, especially for the data owners who do not have full access to the models. Membership inference attacks (MIAs) Shokri et al. (2017) have been widely adopted as an empirical verification method for machine unlearning Gao et al. (2024), since other methods either sacrifice the model utility Sommer et al. (2022); Guo et al. (2023); Han et al. (2025) or necessitate full access Jagielski et al. (2022). However, current white-box MIAs are impractical for FF unlearning, as the data owners may not have full access to model parameters and gradients. Our experiments find that the existing blackbox attacks are not accurate enough for FF models. Their effectiveness is often compromised by standard regularization techniques (e.g., dropout, batch normalization), which inherently decrease the attack success rate. This leads to the second key question in this paper: How to design an accurate and practical verification method for FF unlearning algorithms?

To address these challenges, we make the following contributions:
- *Problem Identification*: To the best of our knowledge, we are the first to formalize the problem and identify the unique challenges of machine unlearning for FF models. Direct gradient ascent induces optimization instability and frequent model collapse due to the sensitivity of FF models to parameter tuning. Layer-wise independent training further complicates the effectiveness-utility trade-off during unlearning.

054 055 056 057 058 059 060 061 062 063 064 065 066 067 068 069 070 071 072 073 074 075 076 077 078 079 080 081 082 083 084 085 086 087 088 089 090 091 092 093 094 095 096 097 098 099 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148 149 150 151 152 153 154 155 156 157 158 159 160 161
- *Novel FF unlearning Framework*: We propose FF-Erase, the first unlearning framework specific to FF models. It introduces a novel goodness-guided approach where a dedicated guidance model directs layer-wise updates. We also propose two practical strategies to efficiently generate this guidance model, mini-retraining and fast-distillation, for a large amount and a small amount of remaining data, respectively.

- *Accurate Black-Box Unlearning Verification*: We propose a new black-box verification method for FF models, the goodness-based MIA (G-MIA). G-MIA leverages the unique properties of the FF models' goodness scores to achieve superior accuracy, providing a reliable tool for unlearning verification. We empirically demonstrate that G-MIA is effective when other black-box attacks fail with regulation techniques applied and even matches the performance of white-box attacks with deep networks and complex datasets.

- *Extensive Evaluation*: We demonstrate through extensive experiments that our method effectively unlearns target data while preserving model utility. FF-Erase achieves unlearning 1.9-3.1× faster than retraining from scratch, with only a minor 1.6-3.3% degradation in accuracy.

## 2 Related Work

Forward-Forward Algorithm: The Forward-Forward algorithm (FF) Hinton (2022) was recently proposed as a novel training method to solve the bio-implausibility problem of backpropagation (BP) Rumelhart et al. (1986) methods, which are the dominant training methods for deep learning models. By eliminating the backward pass, FF models avoid storing intermediate activations and allow layers to process the next data batch immediately, thereby reducing memory consumption and enabling efficient pipeline parallelism. Therefore, numerous works have recently explored different FF algorithms. Initial efforts, such as Symba and Deeperforward Lee & Song (2023); Sun et al. (2025), focused on refining the core goodness function to support deeper networks and faster convergence. Building on these foundational improvements, subsequent work has expanded the FF training methods to more complex domains like convolutional (CwComp Papachristodoulou et al. (2024)), recurrent (FF-LSTM Gautham et al. (2024)), and graph-based (FORWARDGNN Park et al. (2024)) neural networks. As these FF algorithms investigate more complex tasks and architectures, the computational cost of retraining from scratch becomes increasingly prohibitive, creating an urgent need for efficient FF unlearning methods.

Machine Unlearning: Machine unlearning aims to remove the data impact of specific training samples from a trained model, while being efficient and preserving the utility of the unlearned model. Retraining the model from scratch is the gold standard for effectiveness and model utility, but it lacks efficiency. Existing works can be categorized into two types: exact and approximate unlearning. Exact unlearning methods seek to produce a model identical to the retrained model. However, current approaches are incompatible with general FF models, as they either rely on specific sharded architectures Bourtoule et al. (2021); Tao et al. (2024) or are restricted to linear models Guo et al. (2020). Approximate unlearning methods tune the model parameters to achieve fast forgetting. The dominant approaches perform gradient ascent (GA) on the forgetting data Tarun et al. (2023a); Sekhari et al. (2021a), while Qiao et al. (2024); Liu et al. (2022b); Wu et al. (2023b) refine this process by using techniques such as influence functions and Hessian matrix to estimate the parameter calibration. However, as discussed in §1 and Appendix §A, these methods were designed for BP- based models and are not suited for FF models due to their sensitivity to parameter tuning and risk of optimization instability. This leaves a clear gap for developing unlearning methods for FF models. Membership Inference Attacks: Membership inference attacks (MIAs) Shokri et al. (2017); Nasr et al. (2019); Melis et al. (2019) are an empirical method for verifying the effectiveness of machine unlearning, particularly for complex, non-convex models Tu et al. (2024). The goal of an MIA is to determine if a given sample was in a model's training set. If an unlearning method is effective, MIAs should not successfully inference the forgetting samples as members. The more accurate an MIA is, the more reliable it is as a verification metric. MIAs are classified by their required level of access. White-box MIAs Wu et al. (2023a); Hamidouche et al. (2022) assume full access to model parameters and gradients, making them powerful but impractical for real-world verification, where data owners typically lack such privileged access or hardware resources for running full models. Black-box MIAs Liu et al. (2023); Cifuentes et al. (2021), which only use the model's final prediction output, are more practical but less accurate as a reliable verification metric. To fill this gap, we propose the Goodness-based MIA (G-MIA), a novel attack that leverages the unique layerwise goodness scores of FF models. G-MIA achieves superior accuracy under a strict black-box constraint, being accurate and practical for verification.

162 163 164 165 166 167 168 169 170 171 172 173 174 175 176 177 178 179 180 181 182 183 184 185 186 187 188 189 190 191 192 193 194 195 196 197 198 199 200 201 202 203 204 205 206 207 208 209 210 211 212 213 214 215 where η is the learning rate. When optimizing Lff(·), the distribution of layers' goodness vectors is shifting towards a direction where the goodness score of the correct class gy is significantly higher than others. For example, after training on data sample (x, y), the goodness distribution moves towards g = [g
∼
1, g∼
2*, . . . , g*↑
y*, . . . , g*∼
J], where the uparrow ↑indicates significant increase and waves
∼ indicate moderate adjustments. As the average of goodness scores usually increases during FF
training, we call this distribution shifting on goodness vectors as goodness increase for brevity.

Model Inference: FF models output the goodness vectors from all layers g 1, g 2*, . . . ,* g L for inference. It is common to take a fully-connected layer on them as the predictor. We employ this predictor as our default setting in experiments due to its superior performance. We provide more details of the above FF training process using an illustration in Figure 2(a) for better understanding.

## 3 Preliminaries

In this section, we begin by reviewing the training and inference process of FF models in §3.1, and then formalize the machine unlearning problem and its notation in §3.2.

## 3.1 Forward-Forward Training Algorithms

Data Forwarding and Goodness Calculation: Consider a neural network model with L layers for a J-class classification task. The objective of FF training is to optimize each layer l's parameters θ l, so that every layer's goodness can better predict the correct class label y for given input x.

Specifically, the function f lfor each layer l first computes its output h l using its input z l−1from layer l − 1. Then it computes the goodness vector g l based on h l, which reflects the activation degree of the neurons in a layer and is the key design for FF training and inference. After that, the layer simultaneously updates its parameters θ land forward z l, which is the normalization of h l1, to the next layer. Specially, the raw input x is considered as z 0. This process is formalized as follows:

$$\mathbf{z}^{0}=\mathbf{x},\quad\forall l\in\{1,2,\ldots,L\},\quad\mathbf{h}^{l}=f^{l}(\mathbf{z}^{l-1};\theta^{l}),\quad\mathbf{g}^{l}=\|\mathbf{h}^{l}\|_{1},\quad\mathbf{z}^{l}={\frac{\mathbf{h}^{l}-\mathbf{g}^{l}}{\sqrt{\sigma^{2}}+\epsilon}},$$
, (1)
where σ 2are the variance of h lfor layer normalization, and ϵ is a small constant to avoid dividing by zero. The goodness vector g l = [g l1, gl2*, . . . , g*lJ] contains J scores for each class, respectively.

Loss Function and Optimization: FF training aims to increase the goodness score g ly of the correct class y while suppressing the other goodness scores g lj,j̸=y
. The loss function Lff is formalized as:

$$\forall l\in\{1,2,\ldots,L\},\quad{\mathcal{L}}_{\mathrm{fl}}(g^{l}(\mathbf{x},y;\theta^{l}))=-\log\left({\frac{\exp\left(g_{y}^{l}\right)}{\sum_{j=1}^{J}\exp\left(g_{j}^{l}\right)}}\right).$$
. (2)
As this is a layer-wise loss function, FF training optimizes each layer's parameters independently:

$$(1)$$
$$(2)$$

∀*l, θ*l ← θ l − η∇θ lLff(g l(x, y; θ l)), (3)
The purpose of a machine unlearning process is to remove the influence of forgetting data Dforget from an original model θo (the model to unlearn) while maintaining the utility of unlearned model θu on the remaining data Dremain = Dtrain \ Dforget, where Dtrain is the training dataset of θo. Specifically, we denote the model retrained on Dremain as θr. This objective can be formalized as:

$$\operatorname*{min}_{\theta^{u}\in\Theta}{\mathcal{L}}(\theta_{u};\mathbb{D}_{\mathrm{forget}})-\lambda{\mathcal{L}}(\theta_{u};\mathbb{D}_{\mathrm{remain}}),$$
L(θu; Dforget) − λL(θu; Dremain), (4)
$$(4)$$

## 3.2 Machine Unlearning Notations

where λ is a hyper-parameter to balance the trade-off between effectiveness, *i.e.*, loss value on forgetting data L(θu; Dforget) and model utility, *i.e.*, loss value on remaining data L(θu; Dremain).

216 217 218 219 220 221 222 223 224 225 226 227 228 229 230 231 232 233 234 235 236 237 238 239 240 241 242 243 244 245 246 247 248 249 250 251 252 253 254 255 256 257 258 259 260 261 262 263 264 265 266 267 268 269

![4_image_1.png](4_image_1.png)

![4_image_0.png](4_image_0.png) 
Figure 2: Illustrations for FF learning (a) and FF-Erase unlearning (b). We elaborate the layer-wise training at the lower left corner and illustrate the multi-class goodness design at the lower right corner. For example, when training on images of number 2, the corresponding goodness score g 32 increases while others are suppressed. We also describe each step of unlearning at the upper corner.

## 4 Methodology

In this section, we first introduce the workflow of our proposed FF-Erase unlearning algorithm in
§4.1. Then we present two practical strategies to efficiently acquire the guidance model required for performing FF-Erase unlearning in §4.2. Finally, we discuss the efficiency of FF-Erase in §4.3.

## 4.1 Fast Forward-Forward Unlearning

The key idea of FF-Erase unlearning is to decrease the goodness score on the forgetting data while maintaining the goodness score on the remaining data. The goodness decrease is the opposite process of learning, *i.e.*, g = [g
∼
1, g∼
2, . . . , g↓y*, . . . , g*∼
J] for forgetting data sample (x, y), which is named as "forgetting forward". To address the instability challenge during parameter tuning, we decrease the goodness under the guidance goodness g∗ from a guidance model θg, which is ignorant of the forgetting data but has the same architecture as the original model. Besides, we also run "*recovering forward*" to maintain the goodness score on the remaining data by repeating the learning process every K epochs.2 The overall workflow of FF-
Erase unlearning is summarized as follows. Forgetting Forward: 1) Every epoch, we forward the forgetting data samples through the original model and collect the goodness vector g(x; θ); 2) we forward the same forgetting data samples through the guidance model to acquire the guidance goodness vector g∗(x; θg); 3) we decrease the goodness of forgetting data on the original model by minimizing the KL-loss between them:
∀(x, y) ∈ Dforget, ∀l ∈ 1, 2*, . . . , L, θ*l ← θ l − η∇θlDKL(g l(x, y; θ l)∥g l∗(x, y; θ lg)), (5)
Algorithm 1 FF-Erase Unlearning Algorithm Input: Models θo and θg, epoch E, thresholds ϵ1 and ϵ2, datasets Dforget and Dremain.

Parameter: FF model depth L, learning rate η, recovery step K, hyper-parameter λ.

Output: Unlearned model θu.

1: for e = 1, 2*, . . . , E*:
2: for x in Dforget:
3: ℓ1=**FFwd**(x,θo,θg) // *forgetting forward* 4: for (x, y) in Dremain, if e%K == 0:
5: ℓ2=**RFwd**(x,y,θo) // *recovering forward* 6: if ℓ1 < ϵ1 or ℓ2 > ϵ2: **break**
Return: θu=θo FFwd(z 0 = x, θo, θg):
1: for l = 1, 2, *. . .* , L: 2: h l=f l(z l−1;θ lo), h lg=f l(z l−1 g;θ lg)
3: z l=**LayerNorm**(h l), z l g=**LayerNorm**(h l g)
4: g l=**Norm**(h l), g l∗=**Norm**(h lg)
5: ℓ1[l] = ∇DKL([g l], [g l∗
]), θ lo = θ lo − ηℓ1[l]
6: **return** PL
l=1 ℓ1[l]
RFwd(z 0 = x, y, θo):
1: for l = 1, 2, *. . .* , L:
2: h l=f l(**LayerNorm**(h l−1);θ lo),g l=**Norm**(h l)
3: ℓ2[l] = ∇Lff([g l], y), θ lo = θ lo − ηλℓ2[l]
4: **return** PL
l=1 ℓ2[l]
270 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290 291 292 293 294 295 296 297 298 299 300 301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320 321 322 323 which leverages a distillation-like manner for moderate parameter tuning during goodness decrease. Recovering Forward: 1) Every K epochs, we forward the remaining data samples through the original model and collect the goodness vector g(x; θ); 2) we update the parameters layer-wise to increase the goodness of remaining data. We summarize these two steps as:
∀(x, y) ∈ Dremain, ∀l ∈ [1, L], θl ← θ l − η∇θ lλLff(g l(x, y; θ l)) (6)
We provide more details to help understand the two forwards with corresponding steps including an illustration in Figure 2(b) and pseudocode in Algorithm 1. The functions **FFwd** and **RFwd** refer to the forgetting forward and recovering forward processes, respectively. We use **LayerNorm** and Norm to denote the layer normalization and L1-norm operation for computing goodness in Equation
(1), respectively. Rather than directly minimizing the goodness score of the correct class, FF-Erase decreases the goodness by shifting the goodness distribution towards the guidance goodness g∗ using the Kullback-Leibler divergence for stable and moderate parameter tuning: DKL(g∥g∗) =
PJ
i=1 gˆilog (ˆgi/gˆ∗i), where gˆi = exp gi/PJ
j=1 exp gj is the softmaxed goodness of the i-th class.

Termination Conditions. The unlearning process in FF-Erase will halt if the model fails to converge after a maximum number of epochs E. Besides, FF-Erase also employs an early stopping mechanism as commonly used in machine unlearning. Specifically, if the loss value update on the forgetting data Dforget drops below a threshold ϵ1 or the loss value on the remaining data Dremain exceeds a threshold ϵ2, FF-Erase will terminate unlearning and return the current model as the unlearned model θu.

## 4.2 Training Guidance Models

To ensure both the efficiency and unlearning performance for the FF-Erase algorithm, we require a stable and accurate guidance model. That is to say, the guidance models need to provide stable guidance goodness distributions and be ignorant of the forgetting data. This is important for stabilizing the parameter calibration and avoiding model collapse during unlearning. Besides, the efficiency of generating the guidance model is also important. To this end, we propose two practical strategies to efficiently obtain accurate guidance models in different scenarios: mini-retrained and fast-distilled. Mini-retrained models are faster to obtain. However, when there are not enough remaining samples for retraining, we can still obtain fast-distilled models as slower alternatives, as they can be generated using fewer data samples. Mini-Retrained Strategy. An ideal guidance model is one retrained from scratch on the remaining data, which is naturally stable and accurate. However, it is computationally prohibitive. As we do not demand guidance models' accuracy on the remaining data, we accelerate this process through two approximations: retraining α1 =|Dref|/|Dremain|∈ (0, 1) proportion of the remaining samples using α2 ∈ (0, 1) proportion of the epochs, where Dref ⫋ Dremain is the selected subset:
θ g,t ← θ g,t−1 − η∇θ g,t−1L(Dref; θ g,t−1). (7)
Fast-Distilled Strategy. The knowledge distillation Hinton et al. (2015a); Gou et al. (2021) is a well-known approach to rapidly train a new model using existing models. Here, the original model θo acts as the "teacher". The goal is to train a "student" guidance model, θg, to mimic the teacher's output on the remaining data. We use a simplified objective for fast distillation as follows:
θ g,t ← θ g,t−1 − η∇θ g,t−1DKL(Dref; θ g,t−1∥θo). (8)
This strategy can also be accelerated using α1 and α2 as the mini-retrained strategy does. 4.3 EFFICIENCY OF FF-ERASE
The unlearning time of FF-Erase algorithm tunl contains two parts: the time to obtain the guidance model t0 and the time for goodness decrease t1. When unlearning β = |Dforget|/|Dtrain| ∈ (0, 1)
proportion of the training samples, the total time for FF-Erase using mini-retrained strategy is:
tunl = t0 + t1 ≈ α1 · α2 · tret + (K−1 + β) · tret, (9)
where tret is the time for retraining from scratch. According to the experimental results in §6, we can achieve satisfactory unlearning performance using guidance models with α1 = 0.3 and α2 = 0.5, indicating an acceptable overhead of obtaining the guidance model (about 15% of tret). Empirically, t1 usually takes another 10 to 20% of tret, leading to an overall tunl of 25 to 35% of tret for FF-Erase to achieve effective unlearning. FF-Erase using fast-distilled strategy takes similar time.

## 5 Goodness-Based Membership Inference Attack (G-Mia)

324 325 326 327 328 329 330 331 332 333 334 335 336 337 338 339 340 341 342 343 344 345 346 347 348 349 350 351 352 353 354 355 356 357 358 359 360 361 362 363 364 365 366 367 368 369 370 371 372 373 374 375 376 377 In this section, we introduce the workflow of G-MIA and describe how to use G-MIA for quantitative verification of FF unlearning algorithms. We consider that the attacker can synthesize data that has a similar distribution to the training data, which is a common setting in related works (*e.g.*, Shokri et al. (2017); Liu et al. (2022a); Nasr et al. (2019)) and can be realized by model inversion techniques Fredrikson et al. (2015). It is also noted that the attacker can obtain the output of the target model of attack, *i.e.*, the goodness vectors from all layers. With the above information, a complete G-MIA contains the following four steps:
1) **Shadow Model Training.** The attackers first generate a synthetic dataset Dsyn and trains shadow models θshadow on it. They also generate another separate synthetic dataset D
′syn for testing.

2) **Goodness Feature Extraction.** The attacker collects the goodness vectors from all layers when member data (Dsyn) and non-member data (D
′syn) of θshadow forward the network.

3) **Attack Model Training.** The attacker uses the collected goodness vectors to train a binary classifier fG−MIA(·) that predicts whether a given sample is a member or non-member:

$$m e m b e r$$ $$n o n\l{\mathrm{-}}m e m b e r$$
$$f_{\mathrm{G-MIA}}(\mathbf{g}^{1},\mathbf{g}^{2},\ldots,\mathbf{g}^{L})={\begin{cases}1,\\ 0,\end{cases}}$$
$$(10)^{\frac{1}{2}}$$
L) = 1, *member*
0, *non-member* (10)
4) **Membership Inference.** Given a specific data d, the attacker first forwards d on the model under attack and obtains the goodness vectors, then predicts its membership by fG−MIA(·).

G-MIA Verification. We quantify the unlearning using the attack accuracy (ACC) and the area under the curve (AUC). A lower ACC or AUC score indicates fewer forgetting samples are identified as members, implying the unlearning is more effective. We provide more details in Appendix B.1.

## 6 Experiments

In this section, we first present the effects of G-MIA in §6.1. Then we show the experimental results of FF-Erase unlearning regarding efficiency, effectiveness, and model utility in §6.2. In §6.3, we further explore classical unlearning methods under different parameters to robustly support our findings in §1. Lastly in §6.4, we present an ablation study to show the necessity and trade-offs of the guidance models. We evaluate FF unlearning on 4 standard image benchmarks: CIFAR-10, CIFAR-100 Krizhevsky et al. (2009), MNIST LeCun et al. (2010), and Fashion-MNIST Xiao et al. (2017), which are consistent with prior work on FF algorithms regarding the dataset complexity. We test on various FF models, including a 2-layer tiny CNN, AlexNet Krizhevsky et al. (2012), and VGG Simonyan et al. (2014) using state-of-the-art FF algorithms: CwComp and Deeperforward.

## 6.1 G-Mia Performance

As an effective and reliable verification metric for FF unlearning, G-MIA should be accurate and present high ACC and AUC scores. To this end, we compare the attack accuracy (ACC) and area under the curve (AUC) of G-MIA with several state-of-the-art MIAs, including black-box final-layer MIA (FL) Shokri et al. (2017), white-box MIA using intermediate layer gradient (GR) Nasr et al. (2019), and white-box MIA using all layer outputs, including global average pooling (GAP) and statistics (ST). The statistics include mean, variance, maximum, and L2 norm of all layer outputs.

Our target models have employed basic MIA-defending techniques, including dropout, batch normalization, and weight decay. For each model, we randomly select 5000 pieces of data samples from the training set and test set, respectively, as the member and non-member data. The attack model for every type of MIAs is a standard multilayer perceptron with six hidden layers.

Our results shown in Figure 3 (using ACC as the metric3) indicate that G-MIA is an accurate and practical verification metric for FF unlearning. Firstly, G-MIA consistently outperforms the classical black-box final-layer MIA (FL) on all datasets and models. This indicates that the goodness from all layers provides more membership information than the final-layer output alone. Moreover, G- MIA even presents a better performance than white-box MIAs under deeper models and complex datasets. For example, G-MIA achieves the best accuracy under VGG13 and CIFAR-100. This is because deeper models and complex datasets amplify the impact of layer-wise independent training, making the goodness vectors from all layers more informative for membership inference.

3Due to space limitations, we show the results using AUC in the appendix §B.2.

378 379 380 381 382 383 384 385 386 387 388 389 390 391 392 393 394 395 396 397 398 399 400 401 402 403 404 405 406 407 408 409 410 411 412 413 414 415 416 417 418 419 420 421 422 423 424 425 426 427 428 429 430 431

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png)

## 6.2 Machine Unlearning On Ff Models

In this experiment, we present the efficiency, effectiveness, and model utility of different unlearning methods using time versus accuracy curves. For the setup, we randomly sample 20% of the training data Dtrain as forgetting Dforget and use a separate test set, Dtest, from the same data distribution.

As these three datasets share the same data distribution, effective unlearning algorithms will produce models that their accuracy on Dforget are similar to the original model's accuracy on Dtest. We further leverage G-MIA scores to rigorously quantify the extent of effective information removal. For model utility, a desirable algorithm should preserve performance of the unlearned model, meaning its accuracy on Dtest should remain close to that of the original model.

We compare FF-Erase with retraining from scratch (RE) and direct gradient ascent (GA). RE is the gold standard for unlearning performance on effectiveness and model utility, while GA is a representative method for classical unlearning methods. We use FF-Erase(D) and FF-Erase(R) to denote FF-Erase unlearning using fast-distillation and retrain-based guidance models, respectively. Due to space limitations, we only show the results of VGG13 models trained on the CIFAR-10 dataset in the main text and put other results in Appendix §C. As shown in Figure 4, our proposed FF-Erase efficiently realizes both effective and model utility. For effectiveness, FF-Erase(D) presents as a low G-MIA score (0.5245) as RE (0.532). It also achieves the same (81.31) accuracy on Dforget using only 38.52% of the RE time. For model utility, FF-
Erase(D) retains similar accuracy as retaining on test data (80.85 and 77.87, respectively). Compared with FF-Erase(D), the FF-Erase(R) is more efficient (29.19% of the RE time) with tradeoffs on effectiveness (0.526 and 80.72 for G-MIA score and accuracy on Dforget, respectively) and model utility (77.77). We also investigate the trade-off of different guidance models on FF-Erase in §6.4. Noted that the GA method (when λ = 10) fails to converge and leads to model collapse. We will further explore its performance under different λ choices in section §6.3. 6.3 FURTHER EXPLORATION ON CLASSICAL UNLEARNING METHODS
In this experiment, we further explore the impact of different λ = 101, 100, 10−1, 10−2, 10−3, 0 in Equation (4) on the performance of classical unlearning methods using gradient ascent as a representative. Our results on VGG13 models trained on the CIFAR-10 dataset are shown in Figure 5, indicating that the model will either collapse (λ = 101, 100, 10−1) or cannot unlearn the forgetting 432 433 434 435 436 437 438 439 440 441 442 443 444 445 446 447 448 449 450 451 452 453 454 455 456 457 458 459 460 461 462 463 464 465 466 467 468 469 470 471 472 473 474 475 476 477 478 479 480 481 482 483 484 485 data (λ = 10−2, 10−3, 0). As shown in Figure 5(a), GA (when λ = 10−2, 10−3, 0) presents significantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.

Figure 5(c) shows a more precise result, where GA (λ = 10−2, 10−3, 0) gets G-MIA scores of 0.6, 0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA
(when λ = 101, 100, 10−1) shows low accuracy Dtest (below 60), failing to preserve model utility.

6.4 ABLATION STUDY ON GUIDANCE MODELS

In this experiment, we explore the efficiency-performance trade-off of different guidance models in FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may sacrifice the unlearning effectiveness and model utility. We test different proportions of selected data

α1 and of epochs α2 for acquiring guidance models and utilize them for FF-Erase unlearning.

Methods Efficiency Effectiveness Utility

tunl (/s)↓ t0 (/s)↓ tunl − t0 (/s)↓ Accf (%)↓ G-MIA ACC↓ G-MIA AUC↓ Acct (%)↑

RE 1107 0 1107 81.61 0.551 0.571 80.85

D-(0.5,0.5) 583.5 410.5 173 81.58 0.556 0.577 78.34 D-(0.5,0.2) 426.7 158.9 267.8 80.76 0.561 0.583 77.95 D-(0.5,0.1) 353.7 83.7 270 80.48 0.587 0.612 77.54 D-(0.3,0.5) 569.6 288.5 281.1 81.12 0.568 0.595 77.87 D-(0.3,0.2) 391.1 111.7 279.4 80.73 0.571 0.606 77.81 R-(0.5,0.5) 518.5 155.1 363.4 81.53 0.562 0.583 79.16 R-(0.5,0.2) 443.7 60.7 383 81.02 0.573 0.609 79.00 R-(0.3,0.5) 474.4 107.8 366.6 81.51 0.569 0.598 78.86

R-(0.3,0.2) 429.6 41.8 387.6 81.03 0.577 0.621 78.58

R.G.M 950.8 0 950.8 51.18 0.553 0.575 55.53

Table 1: FF-Erase unlearning using different guidance models. Accf and Acct respectively denote

the accuracy on Dforget and Dtest. D (R) refers to fast-distilled (mini-retrained) strategy, followed by

α1 and α2, *e.g.*, D-(0.5,0.1) refers to FF-Erase guided by a fast-distilled guidance model on α1=50% data for α2=10% epochs. The tunl is the total unlearning time containing two parts: guidance model obtaining t0 (if any) and goodness decrease tunl −t0. R.G.M in the last line refers to FF-Erase using

randomly initialized guidance model. The ↓ (↑) indicates that a lower (higher) is better.

Firstly, as shown in Table 1, a stable and accurate guidance model is crucial for FF-Erase unlearning. FF-Erase using a randomly initialized model as guidance model (denoted as R.G.M in the last line) leads to unacceptable performance degradation: the ACCt drops to 55.53%. Such a guidance model could not provide stable guidance goodness for goodness decreasing, leading to a situation similar to the direct gradient ascent (GA) method. Secondly, using more data samples for generating the guidance model (a larger α1) leads to better unlearning performance (lower G-MIA ACC, *i.e.*, more effective, and higher ACCt, *i.e.*, better model utility) but requires longer unlearning time tunl. Using more training epochs (a larger α2) also leads to the same trend. Our ablation study demonstrates that FF-Erase can flexibly achieve different efficiency-performance trade-offs by choosing different guidance strategies and hyperparameters, making it adaptable to various application scenarios.

## 7 Conclusion

In this paper, we propose FF-Erase, the first machine unlearning method for FF models. We identify the problem that existing unlearning methods designed for BP-based models are infeasible for FF models due to the sensitivity of FF models to parameter changes. To address this challenge, we design FF-Erase, a novel FF-specific gradient ascent method to effectively erase the data impact of

![8_image_0.png](8_image_0.png)

(b) Time vs. Dtest Accuracy
486 487 488 489 490 491 492 493 494 495 496 497 498 499 500 501 502 503 504 505 506 507 508 509 510 511 512 513 514 515 516 517 518 519 520 521 522 523 524 525 526 527 528 529 530 531 532 533 534 535 536 537 538 539 forgetting samples. FF-Erase uses a goodness-based regularization to stabilize the parameter calibration and a layer-wise unlearning scheme to promote the unlearning efficiency. Moreover, we propose two flexible strategies to acquire the guidance model for FF-Erase. Accordingly, we propose G-MIA, a goodness-based membership inference attack, to quantitatively verify the unlearning effectiveness of FF-Erase. Extensive experiments on various datasets and model architectures demonstrate that FF-Erase is effective and efficient, achieving comparable unlearning effectiveness as retraining while being 1.9-3.1× faster.

## References

Amirhossein Bagheri, Radmehr Karimian, and Gholamali Aminian. f-scrub: Unbounded machine unlearning via f-divergences. In ICLR 2025 Workshop on Navigating and Addressing Data Problems for Foundation Models, 2025.

Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In 2021 IEEE symposium on security and privacy (SP), pp. 141–159. IEEE, 2021.

Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles. In 2022 IEEE symposium on security and privacy (SP), pp. 1897–1914. IEEE, 2022.

Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. Can bad teaching induce forgetting? unlearning in deep networks using an incompetent teacher. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 7210–7217, 2023a.

Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. Zero-shot machine unlearning. *IEEE Transactions on Information Forensics and Security*, 18:2345–2354, 2023b.

Nicolas Cifuentes, Mingyu Sun, Robin Gupta, and Bikash C Pal. Black-box impedance-based sta- ´
bility assessment of dynamic interactions between converters and grid. *IEEE Transactions on* Power Systems, 37(4):2976–2987, 2021.

Tom Fawcett. An introduction to roc analysis. *Pattern recognition letters*, 27(8):861–874, 2006. Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In Proceedings of the 22nd ACM SIGSAC conference on computer and communications security, pp. 1322–1333, 2015.

Xiangshan Gao, Xingjun Ma, Jingyi Wang, Youcheng Sun, Bo Li, Shouling Ji, Peng Cheng, and Jiming Chen. Verifi: Towards verifiable federated unlearning. IEEE Transactions on Dependable and Secure Computing, 21(6):5720–5736, 2024.

Suraj R Gautham, Swapnil Nair, Suresh Jamadagni, Mridul Khurana, and Md Assadi. Exploring the feasibility of forward forward algorithm in neural networks. In 2024 International Conference on Advances in Modern Age Technologies for Health and Engineering Science (AMATHE), pp. 1–6. IEEE, 2024.

Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A
survey. *International journal of computer vision*, 129(6):1789–1819, 2021.

Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens Van Der Maaten. Certified data removal from machine learning models. In Proceedings of the 37th International Conference on Machine Learning, ICML'20. JMLR.org, 2020.

Yu Guo, Yu Zhao, Saihui Hou, Cong Wang, and Xiaohua Jia. Verifying in the dark: Verifiable machine unlearning by using invisible backdoor triggers. IEEE Transactions on Information Forensics and Security, 19:708–721, 2023.

Varun Gupta, Christopher Jung, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, and Chris Waites.

Adaptive machine unlearning. *Advances in Neural Information Processing Systems*, 34:16319– 16330, 2021.

Mounia Hamidouche, Reda Bellafqira, Gwenole Quellec, and Gouenou Coatrieux. White-box ´
membership attack against machine learning based retinopathy classification. arXiv preprint arXiv:2206.03584, 2022.

Mengde Han, Tianqing Zhu, Lefeng Zhang, Huan Huo, and Wanlei Zhou. Vertical federated unlearning via backdoor certification. *IEEE Transactions on Services Computing*, 2025.

540 541 542 543 544 545 546 547 548 549 550 551 552 553 554 555 556 557 558 559 560 561 562 563 564 565 566 567 568 569 570 571 572 573 574 575 576 577 578 579 580 581 582 583 584 585 586 587 588 589 590 591 592 593 Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015b.

Matthew Jagielski, Om Thakkar, Florian Tramer, Daphne Ippolito, Katherine Lee, Nicholas Carlini, Eric Wallace, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, et al. Measuring forgetting of memorized training examples. *arXiv preprint arXiv:2207.00099*, 2022.

Yongwoo Kim, Sungmin Cha, and Donghyun Kim. Are we truly forgetting? a critical reexamination of machine unlearning evaluation protocols. *arXiv preprint arXiv:2503.06991*, 2025.

Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In *International conference on machine learning*, pp. 3519– 3529. PMlR, 2019.

Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.

2009.

Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. *Advances in neural information processing systems*, 25, 2012.

Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded machine unlearning. *Advances in neural information processing systems*, 36:1957–1987, 2023.

Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. *ATT Labs [Online].*
Available: http://yann.lecun.com/exdb/mnist, 2, 2010.

Heung-Chang Lee and Jeonggeun Song. Symba: Symmetric backpropagation-free contrastive learning with forward-forward algorithm for optimizing convergence, 2023.

Gaoyang Liu, Tianlong Xu, Rui Zhang, Zixiong Wang, Chen Wang, and Ling Liu. Gradient-leaks:
Enabling black-box membership inference attacks against machine learning models. IEEE Transactions on Information Forensics and Security, 19:427–440, 2023.

Lan Liu, Yi Wang, Gaoyang Liu, Kai Peng, and Chen Wang. Membership inference attacks against machine learning models via prediction sensitivity. IEEE Transactions on Dependable and Secure Computing, 20(3):2341–2347, 2022a.

Yi Liu, Lei Xu, Xingliang Yuan, Cong Wang, and Bo Li. The right to be forgotten in federated learning: An efficient realization with rapid retraining. In IEEE INFOCOM 2022-IEEE conference on computer communications, pp. 1749–1758. IEEE, 2022b.

Yiyong Liu, Zhengyu Zhao, Michael Backes, and Yang Zhang. Membership inference attacks by exploiting loss trajectory. In Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security, pp. 2085–2098, 2022c.

Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. Exploiting unintended feature leakage in collaborative learning. In *2019 IEEE symposium on security and privacy (SP)*, pp. 691–706. IEEE, 2019.

Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. *arXiv* preprint arXiv:1503.02531, 2015a.

Geoffrey Hinton. The forward-forward algorithm: Some preliminary investigations, 2022. Elizabeth Liz Harding, Jarno J Vanto, Reece Clark, L Hannah Ji, and Sara C Ainsworth. Understanding the scope and impact of the california consumer privacy act of 2018. Journal of Data Protection & Privacy, 2(3):234–253, 2019.

Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning:
Passive and active white-box inference attacks against centralized and federated learning. In 2019 IEEE symposium on security and privacy (SP), pp. 739–753. IEEE, 2019.

Andreas Papachristodoulou, Christos Kyrkou, Stelios Timotheou, and Theocharis Theocharides.

Convolutional channel-wise competitive learning for the forward-forward algorithm. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pp. 14536–14544, 2024.

Namyong Park, Xing Wang, Antoine Simoulin, Shuai Yang, Grey Yang, Ryan Rossi, Puja Trivedi, and Nesreen Ahmed. Forward learning of graph neural networks. arXiv preprint arXiv:2403.11004, 2024.

Xinbao Qiao, Meng Zhang, Ming Tang, and Ermin Wei. Hessian-free online certified unlearning.

arXiv preprint arXiv:2404.01712, 2024.

594 595 596 597 598 599 600 601 602 603 604 605 606 607 608 609 610 611 612 613 614 615 616 617 618 619 620 621 622 623 624 625 626 627 628 629 630 631 632 633 634 635 636 637 638 639 640 641 642 643 644 645 646 647 David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. *nature*, 323(6088):533–536, 1986.

Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems, 34:18075–18086, 2021a.

Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what you want to forget: Algorithms for machine unlearning. Advances in Neural Information Processing Systems, 34:18075–18086, 2021b.

Nazanin Mohammadi Sepahvand, Eleni Triantafillou, Hugo Larochelle, Doina Precup, James J
Clark, Daniel M Roy, and Gintare Karolina Dziugaite. Selective unlearning via representation erasure using domain adversarial training. In The Thirteenth International Conference on Learning Representations, 2025.

Haonan Shi, Tu Ouyang, and An Wang. Learning-based difficulty calibration for enhanced membership inference attacks. In 2024 IEEE 9th European Symposium on Security and Privacy (EuroS&P), pp. 62–77. IEEE, 2024.

Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In *2017 IEEE symposium on security and privacy (SP)*, pp. 3–18. IEEE, 2017.

Karen Simonyan, Andrew Zisserman, et al. Very deep convolutional networks for large-scale image recognition. *arXiv preprint arXiv:1409.1556*, 2014.

David M Sommer, Liwei Song, Sameer Wagh, and Prateek Mittal. Athena: Probabilistic verification of machine unlearning. *Proceedings on Privacy Enhancing Technologies*, 2022.

Liang Sun, Yang Zhang, Weizhao He, Jiajun Wen, Linlin Shen, and Weicheng Xie. Deeperforward: Enhanced forward-forward training for deeper and better performance. In The Thirteenth International Conference on Learning Representations, 2025.

Youming Tao, Cheng-Long Wang, Miao Pan, Dongxiao Yu, Xiuzhen Cheng, and Di Wang. Communication efficient and provable federated unlearning. *Proc. VLDB Endow.*, 17(5):1119–1131, January 2024. ISSN 2150-8097. doi: 10.14778/3641204.3641220. URL https://doi.org/ 10.14778/3641204.3641220.

Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. Fast yet effective machine unlearning. *IEEE Transactions on Neural Networks and Learning Systems*, 35(9): 13046–13055, 2023a.

Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. Fast yet effective machine unlearning. *IEEE Transactions on Neural Networks and Learning Systems*, 35(9): 13046–13055, 2023b.

Yiwen Tu, Pingbang Hu, and Jiaqi Ma. Towards reliable empirical machine unlearning evaluation:
A game-theoretic view, 2024. URL https://arxiv.org/abs/2404.11577.