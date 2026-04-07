# METAOCDN: A COGNITION-INSPIRED META OPTIMIZED COMPLEMENTARY DUAL NETWORKS FOR ONLINE CONTINUAL CONCEPT DRIFT ADAPTATION


**Anonymous authors**
Paper under double-blind review


ABSTRACT


The _Complementary Learning Systems_ (CLS) theory points that humans can continuously and efficiently adapt to new tasks through the collaboration between
the hippocampus and the neocortex: the former rapidly encodes new knowledge,
while the latter extracts structured knowledge by abstract learning. Their synergy
enables humans not only to quickly learn new tasks in the short term but also to
transfer acquired knowledge across different tasks. Inspired by this theory, we
address the challenge of streaming data mining under open environment with concept drift by proposing a cognition-inspired meta optimized complementary dual
networks architecture (MetaOCDN), which consists of the Adaptive Fine Tuning
Network (AFT-Net) and the Meta Representation Network (MRN-Net). AFT-Net
is similar to the hippocampus, selectively fine-tunes key layers based on gradient variations to achieve rapid adaptation to novel concepts; MRN-Net is similar
to the neocortex, we design self-supervised duality loss to continuously enhance
its deep representation capability, thereby improving generalization to unknown
distributions; furthermore, we design MAML-based multi-scale knowledge distillation strategy to facilitate dynamic information flow and knowledge transfer
between the two networks. In summary, MetaOCDN provides a brain-inspired
collaborative architecture that integrates the rapid responsiveness of AFT-Net with
the abstract generalization capacity of MRN-Net, and enhances their interaction
through knowledge distillation, thereby achieving a dynamic balance between fast
adaptation and stable generalization in non-stationary data streams with concept
drift. Extensive experiments demonstrate that MetaOCDN consistently outperforms state-of-the-art baselines across various drift scenarios.


1 INTRODUCTION


In open environment streaming data mining tasks, concept drift limits model performance. Models
trained with traditional batch learning paradigms struggle to quickly adapt to new distribution after
concept drift (Lu et al., 2019). At present, researchers expect to train models through online learning
approach (Cano & Krawczyk, 2022) (such as active drift detection online learning and adaptive
online learning) to capture the dynamic changes in streaming data. The former actively monitors
data distribution changes (e.g., via statistical tests or sliding-window error rates) to detect concept
drift and performs the targeted update, such as ROALE-DI (Zhang et al., 2020). However, during
the process of actively detecting concept drift, the setting of the threshold can significantly affect
model performance (Gama et al., 2004). Although the Delayed Detection Index (Liu et al., 2022)
alleviates this issue, the false positives, false negatives, and delayed detection remain challenging.
The latter overcomes these challenges by adapting models in real time without relying on drift
detection, e.g., DDG-DA (Li et al., 2022). However, most of these methods rely on supervised or
semi-supervised training strategies, models are difficult to efficiently learn robust features from the
limited samples available after concept drift (Liu et al., 2021). They also tend to optimize a single
objective, restricting the balance between fast adaptation and generalization.


**How to design model that can quickly adapt after concept drift while having a strong gener-**
**alization ability to cope with the impact of changes in data distribution?** The _Complementary_
_Learning_ _Systems_ (CLS) theory (McClelland et al., 1995; Kumaran et al., 2016) offers new inspi

1


ration for us. Humans can quickly extract patterns and adapt to new environments from a limited
number of samples, primarily due to the unique structure of the brain: specifically, the neocortex and
hippocampus. The CLS theory suggests that the neocortex and hippocampus collaborate to enable
efficient learning: the neocortex gradually acquires structured knowledge by alternating between
different tasks, and the hippocampus is better at encoding new information quickly. When facing
new and complex tasks, the hippocampus retrieves structured knowledge stored in the neocortex
to promote rapid learning, and the neocortex encodes the new knowledge from the hippocampus
into structured knowledge, it enhances the stability of knowledge and improves the ability to learn
quickly. Recent studies have introduced the CLS theory into continual learning and have shown
its potential to mitigate catastrophic forgetting (Pham et al., 2023). However, how to transfer this
mechanism into open environments for concept drift adaptation remains an open challenge for further exploration. Therefore, to alleviate limitations in existing works, we propose a meta optimized
complementary dual network strategy (MetaOCDN). The connection between MetaOCDN and CLS
theory is shown in Fig. 1:


Figure 1: Meta optimized complementary dual network strategy inspired by the CLS theory.


Specifically, we construct Adaptive Fine Tuning Network (AFT-Net) to simulate the hippocampus and design gradient-aware selective fine-tuning strategy to selectively fine-tune its key layers,
thereby forming a sparse network. AFT-Net learns task-specific knowledge from the current samples in an online learning manner, to ensure the model rapidly adapts to new distribution. And we
construct Meta Representation Network (MRN-Net) to simulate the neocortex, the self-supervised
duality loss is designed to continuously refine its feature extraction ability, and offline learning
is employed to acquire more robust representations from historical samples. Finally, we design
MAML-based multi-scale knowledge distillation strategy to facilitate knowledge transfer from the
MRN-Net to the AFT-Net. In conclusion, MetaOCDN achieves rapid adaptation to new distribution
while maintaining strong generalization capability. The main contributions of this paper are:


1. Inspired by CLS theory, we propose the MetaOCDN, it includes the AFT-Net and MRN-Net to
emulate the hippocampus for rapid learning new knowledge and the neocortex for extracting structured knowledge. The MAML-based multi-scale knowledge distillation strategy further enhances
knowledge transfer, balances fast convergence with stable generalization.


2. We analysis why selective fine-tuning the critical layer in the face of different distribution
changes has a better effect than fully fine-tuning the network, and at the same time we prove that the
MetaOCDN has an excellent sublinear regret bound.


3. The actual performance of MetaOCDN was verified in classification and regression tasks involving concept drift. Compared with the baseline methods, MetaOCDN achieves good results in terms
of model convergence speed and generalization after concept drift.


2 RELATED WORK


**Active drift detection online learning.** This type of approach mainly relies on dynamic monitoring
of model performance or data distribution to determine whether drift has occurred. Typical methods
include: Type-LDD (Yu et al., 2023), a pre-trained framework for drift localization and type identification using knowledge distillation; and Targeted EL (Guo et al., 2024), which identifies drift types


2


**Hippocampus**
**(AFT-Net)**


**Rapid encoding** **Interleaved learning** **Storage Knowledge**


**Neocortex** **Hippocampus** **Rapid encoding**
**(MRN-Net)** **(AFT-Net)** **(Rapid learning)**


**Interleaved learning**
**(Knowledge transfer)** **(Feedback)**


and selects base classifiers accordingly to improve diversity, among others. Most of these methods
are error-rate–based, relying on window mechanisms and manually set parameters, which often lead
to unstable performance (Bifet & Gavalda, 2007). Compared with error rate–based detectors, these
methods identify drift timing and location more accurately by comparing data distributions or representation spaces (Liu et al., 2022). Representative approaches include MCDDD (Wan et al., 2024)
(contrastive concept embedding), PERCESS (Cai et al., 2025) (latent representation estimation for
online prediction), and AMSL (Zhang et al., 2022) (self-supervised adaptive memory). They offer
finer-grained detection but rely heavily on representation quality, making them prone to false alarms
or delays in real-time streaming scenarios.


**Adaptive online learning.** Adaptive online learning under concept drift bypasses explicit drift detection by assuming that data distribution may change at any time and adapting models through
real-time updates. Representative methods include: HBP (Sahoo et al., 2017), which dynamically
re-weights network layers to adjust depth during training; OneNet (Wen et al., 2023), which integrates reinforcement learning into online convex optimization to enhance robustness but with limited
fast adaptation; ReCDA (Yang et al., 2024), which introduces drift-aware perturbation and representation alignment to learn more stable features; and memory-aware approaches that update parameter
importance for continual adaptation (Aljundi et al., 2018). Overall, these methods improve adaptability and robustness under drift through dynamic adjustment, yet most rely on supervised or semisupervised training and struggle to efficiently learn from limited post-drift samples, with objectives
often biased toward either fast adaptation or generalization, but not both.


3 METAOCDN: COGNITION-INSPIRED ONLINE LEARNING ALGORITHM


Concept drift is a phenomenon in which the statistical properties of a target domain change over
time in an arbitrary way (Lu et al., 2019). Given a time period [0 _, t_ ], there is a set of streaming data
_DS_ = ( _Xt, yt_ ), _Xt_ denote the feature vector at the timestamp _t_, _yt_ denote the corresponding label.
The streaming data follow a certain distribution _F_ 0 _,t_ ( _X, y_ ), concept drift occurs at timestamp _t_ + 1,
if _F_ 0 _,t_ ( _X, y_ ) _̸_ = _Ft_ +1 _,_ + _∞_ ( _X, y_ ), denoted as _∃t_ : _Pt_ ( _X, y_ ) _̸_ = _Pt_ +1( _X, y_ ). In addition, we denote the
current samples as _D_ _[t]_ = ( _x_ _[t]_ _i_ _[, y]_ _i_ _[t]_ [)][, the historical samples as] _[ D][m]_ [= (] _[x][m]_ _i_ _[, y]_ _i_ _[m]_ [)][, and] _[ {][i]_ [ = 1] _[,]_ [ 2] _[, . . ., n][}]_ [.]


3.1 ADAPTIVE FINE TUNING NETWORK


According to the CLS theory, the hippocampus’s rapid learning ability primarily stems from two
aspects: (1) its synapses exhibit strong plasticity, can quick adjust after one or a few learning trials;
and (2) it encodes new information through sparse neuronal activation patterns. To simulate this
mechanism, we enhance the plasticity of AFT-Net via online learning and design a gradient-aware
selective fine-tuning strategy to construct a sparse network.


Similar to the hippocampus, online learning incrementally learns from streaming data, updates parameters in real time and adapts to current samples distribution within a few iterations. Accordingly,
the AFT-Net is trained under the online learning paradigm (Bartlett et al., 2007), with its parameters
are updated via online gradient descent: _θt_ +1 = _θt_ _−_ _η∇θL_ _[AF T]_ ( _θt_ ; _D_ _[t]_ ), _η_ denotes the learning
rate, and _L_ _[AF T]_ represents the total loss of AFT-Net. Relying solely on online learning to enhance
rapid adaptation to new distribution is insufficient. As indicated by online gradient descent, processing each current sample requires updating all parameters, resulting in a computational complexity
of _O_ ( _d_ ). This not only increases the computational burden but also leads to overfitting to new
distribution and forgetting of previously learned knowledge.


To better simulate the hippocampus and accelerate model convergence, we conduct lots of experiments on three standard concept drift datasets. As a tool for loss minimization, gradients can
more intuitively and precisely reveal the model’s sensitivity to changes in data distribution. The results show that gradients provide a more accurate characterization of the model’s state after concept
drift—different types and degrees of drift exert significantly different impacts on various layers of
the model (see Fig. 2). So we design a gradient-aware selective fine-tuning strategy that freezes
parameters insensitive to the new distribution, thereby constructing a sparse AFT-Net.


Firstly, when the AFT-Net is trained at timestamp _t_, the gradient of the _l_ -th layer is denoted as _gt_ _[l]_ [,]
in this paper, we use the gradient norm �� _gtl_ ��2 [to represent the changes of the] _[ l]_ [-th layer.] [To capture]
the long-term gradient variation patterns of the model, we design a historical gradient variation rate


3


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||


|1e 5|(c)|) Hyperp|plane|Col5|
|---|---|---|---|---|
||||||
||||||


_ℓ_ [sim] = _−I_ Low - _z_ [+] ; _z_ _[t]_ [�] = _−_ [1]

_n_


4


0 200 400 600 800 1000
**Timestamp**


4


2


0


2


4


6


0 200 400 600 800 1000
**Timestamp**


0


1


2


6


4


2


0


0 200 400 600 800 1000
**Timestamp**


Figure 2: _Gradient changes of network layers._ _We analyze gradient changes of ResNet on datasets_
_with_ _different_ _drift_ _types:_ _abrupt_ _(RBFBlips),_ _gradual_ _(Sea),_ _and_ _incremental_ _(Hyperplane),_ _with_
_drifts occurring at timesteps 250, 500, and 750._


matrix **G** _∈_ R _[m][×][L]_ to store the model’s historical gradient variation rates of all _L_ layers over the
last _m_ timestamps, its element _rt_ _[L]_ [=] �� _gtL_ ��2 _[−]_ �� _gtL−_ 1��2 [is] [the] [rate] [of] [change] [of] [the] [gradient.] [On]
this basis, we design layer gradient sensitivity index _R_ _[l]_ _t_ [to reveal the influence intensity of different]
layers:


_R_ _[l]_ _t_ [=]


�� _gtl_ �� _· f_   - _rt_ _[l][, σ][l]_ [�]

- _Li_ =1 �� _gti_ �� _· f_ - _rt_ _[i][, σ][i]_ [�] (1)


Among them, _σ_ _[l]_ is the standard deviation of historical gradient variation rate and it is used to automatically “balance” the contribution of each layer to the overall measurement. Adaptively adjust the
weights _f_ ( _rt_ _[l][, σ][l]_ [) = exp(] _[r]_ _t_ _[l][/σ][l]_ [)][, a larger value of] _[ r]_ _t_ _[l][/σ][l]_ [ indicates that the] _[ l]_ [-th layer is more sensitive]
to changes in the new distribution, conversely, the more stable it is.


Finally, a drift-aware threshold is dynamically generated for each layer to determine whether the
layer should be frozen: _τt_ _[l]_ [=] _R_ ¯ _[L]_ _t_ [+] _[σ]_ _t_ [2][,] [and] _R_ ¯ _[L]_ _t_ [=] [1] _[/L]_ _[·]_ [�] _l_ _[L]_ =1 _[R]_ _t_ _[l]_ [.] [When] _[R][l]_ _t_ _[<]_ _[τ][ l]_ _t_ [,] [the] _[l]_ [-th]
layer is well-adapted to current samples, thus is frozen to avoid unnecessary resource consumption;
otherwise, the layer is regarded as more sensitive to the new distribution and is activated for local
updates. By retaining only the layers sensitive to distribution changes, the model forms a sparse
network. When concept drift occurs, only these key layers need to be fine-tuned, thereby improving
response efficiency while effectively mitigating overfitting.


3.2 META REPRESENTATION NETWORK


Similarly, MetaOCDN constructs a Meta Representation Network (MRN-Net) that learns structured
knowledge from historical samples, analogous to the neocortex. Neocortex relies on slow and cumulative synaptic adjustments, allowing it to extract stable patterns through long-term, cross-task
learning and form task-agnostic structured knowledge. Inspired by this, we design a self-supervised
duality loss (Silva et al., 2024) to optimize the model’s representation ability, thereby building MRNNet capable of “learn to learn extract features”.


Firstly, we use the Wasserstein distance to measure the similarity between current and historical
samples in order to select appropriate training instances (Chizat et al., 2020). Based on this we
divide them into positive samples _D_ _[m]_ [+] and negative samples _D_ _[m][−]_ . We design self-supervised
duality loss to optimize the representation capability of the MRN-Net. The self-supervised duality
loss does not rely on samples’ labels, which is crucial for label-scarce streaming data. It helps the
model learn more discriminative and robust feature representations, enables the MRN-Net to better
capture the underlying structure of the data.


Specifically, the self-supervised duality loss consists of similarity loss and difference loss. We leverage MRN-Net to jointly represent the positive samples _z_ [+] and the current samples _z_ _[t]_, and approximate the similarity loss by maximizing the mutual information max _I_ ( _z_ [+] ; _z_ _[t]_ ) between them. We
_φ_

approximate the maximization of mutual information by maximizing its lower bound (Oord et al.,


_p_ ( _z_ [+] _|z_ _[t]_ )
2018), denoted as: _I_ Low ( _z_ [+] ; _z_ _[t]_ ) = E _p_ ( _z_ + _,zt_ ) log _p_ ( _z_ [+] )


_z_ [+] _|z_ _[t]_ ) _q_ ( _z_ [+] _|z_ _[t]_ )

_p_ ( _z_ [+] ) _≥_ E _p_ ( _z_ + _,zt_ ) log _p_ ( _z_ [+] )


2018), denoted as: _I_ Low ( _z_ ; _z_ ) = E _p_ ( _z_ + _,zt_ ) log _p_ ( _z_ [+] ) _≥_ E _p_ ( _z_ + _,zt_ ) log _p_ ( _z_ [+] ) [.] [Since] [com-]

puting the lower bound of mutual information is challenging, we adopt InfoNCE as a surrogate
objective for mutual information maximization. We have:


_n_

- log _e_ _[ψ]_ [(] _[z]_ _j_ _[t][,z]_ _j_ [+][)] (2)

_j_ =1 - _ni_ =1 _[e][ψ]_ [(] _[z]_ _j_ _[t][,z]_ _i_ [+][)] + _ξ_


_ψ_ ( _·_ ) denotes the similarity function, _n_ is the number of samples, and _ξ_ is a stability term that
smooths the loss function. The proof is in Appendix A.1.


To further enhance MRN-Net’s ability to discriminate irrelevant features, we construct the difference loss by minimizing the mutual information between negative samples and the current samples
representations min [Similarly,] [we] [use] [an] [upper] [bound] [on] [mutual] [information] [as] [an] [ap-]
_φ_ _[I]_ [(] _[z][−]_ [;] _[ z][t]_ [)][.]

proximation for this minimization (Zhang et al., 2023). By introducing a random variable _N_ ( _N_ is
sampled from the original input of negative samples), we can derive the upper bound of mutual information, which is expressed as: _I_ ( _z_ _[−]_ ; _z_ _[t]_ ) = _I_ ( _z_ _[−]_ ; _z_ _[t]_ ; _N_ ) + _I_ ( _z_ _[−]_ ; _z_ _[t]_ _| N_ ). From the derivation
in Appendix A.1, the difference loss is given by:


_ℓ_ _[diff]_ _≈_ _DKL_         - _p_         - _z_ _[−]_ _| N_         - _∥q_         - _z_ _[−]_ [��] + _DKL_         - _p_         - _z_ _[t]_ _| N_         - _∥q_         - _z_ _[t]_ [��] (3)


In conclusion, the total loss of the MRN-Net is: _L_ [MRN] = _βℓ_ _[sim]_ +(1 _−β_ ) _ℓ_ _[diff]_ . _β_ is a hyperparameter
that balances the two losses.


3.3 MAML-BASED MULTI-SCALE KNOWLEDGE DISTILLATION


Finally, the CLS theory suggests that the human brain integrates rapid learning and resistance to forgetting through the synergy between the hippocampus and the neocortex: the hippocampus rapidly
encodes information and replays it during sleep, while the neocortex repeatedly extracts structured
knowledge and feeds it back to the hippocampus to accelerate learning. Inspired by this, we design MAML-based multi-scale knowledge distillation strategy (Finn et al., 2017): AFT-Net adapts
via inner-loop updates with replayed historical samples and transfers knowledge to the MRN-Net,
which extracts cross-task stable patterns and feeds them back, completing the outer loop. This “replay–extract–transfer–feedback” synergy enables MetaOCDN to achieve both fast adaptation and
long-term generalization in dynamic environment.


Specifically, we divide the feature maps extracted by the AFT-Net and the MRN-Net (denoted as
_F_ [AFT] _, F_ [MRN] _∈_ R _[H][×][W][ ×][C]_ ) into multi-scale units, and aggregate the knowledge within each unit
through average pooling:


_pi_ _∈{p_ 1 _, p_ 2 _, . . ., pK}_ represents a set of different scales, [�] _p_ _[MRN]_ _i_ _∈_ R _[p][i][×][p][i][×][C]_ represents the
aggregated features at different scales. Then, we concatenate the aggregated features from different
scales along the channel dimension to form the final multi-scale knowledge representation:


Π _[AF T]_ fused [= Concat] �Π _[AF T]_ _p_ 1 _, . . .,_ Π _[AF T]_ _pK_  - _,_ Π _[MRN]_ fused = Concat �Π _[MRN]_ _p_ 1 _, . . .,_ Π _[MRN]_ _pK_  - (5)


Distillation loss is expressed as follows: _ℓ_ _[KD]_ = KL �softmax �Π _[AT F]_ fused - _,_ softmax �Π _[MRN]_ fused ��. The
interaction between the neocortex and hippocampus relies not only on knowledge transfer but also
on memory replay and structured knowledge extraction. Inspired by this, we introduce MAML to
optimize the knowledge distillation process and better simulate their synergistic mechanism.


Specifically, we map the AFT-Net and the MRN-Net into the bi-level optimization framework of
MAML. The AFT-Net serves as the inner-loop optimizer, trains on replayed information provided
by the MRN-Net; meanwhile, the MRN-Net acts as the outer-loop optimizer, extracts structured
knowledge based on the update dynamics of the AFT-Net and feeding it back. Through this dualloop process, the MRN-Net can perceive and adapt to the learning state of the AFT-Net, distilling
more tailored knowledge to enhance its adaptability.


The initialization parameters of AFT-Net are _θ_ . For the _i_ -th inner-loop optimization, the parameter
update of AFT-Net is denoted as _θ_ _[i]_ . Specifically, support sets _D_ _[s]_ are randomly sampled from
historical samples, and AFT-Net is iteratively updated via stochastic gradient descent. For example,
with a single gradient update: _θ_ _[i]_ = _θ −_ _αin_ _∂ℓ_ ¯ _[KD]_ ( _∂θD_ _[s]_ ; _θ,φ_ ), _αin_ denotes the learning rate of the AFT
Net. After multiple rounds of information replay, the MRN-Net serves as the outer-loop optimizer
to extract structured knowledge. We employ a regularization term as an approximate gradient to
transfer the knowledge encoded in the AFT-Net parameters to the MRN-Net, as follows: _φ_ =


5


Π _[AT F]_ _pi_ = _p_ [1][2] _i_


_h,w∈_ �( _H,W_ ) _F_ _[AT F]_ ( _h, w_ ) _,_ Π _[MRN]_ _pi_ = _p_ [1][2] _i_


_p_ [2] _i_


 - _F_ _[MRN]_ ( _h, w_ ) (4)


_h,w∈_ ( _H,W_ )


_θt_ +1 = _θt −_ _λθ∇θ_ (� _ℓ_ _[cross]_ ( _D_ _[t]_ _, f_ ( _θt_ )) + _ℓ_ _[KD]_ ( _D_ _[t]_ ; _θt, φt_ ) + _R_ ( _φt, θt_ )) (6)


Here, _ℓ_ _[cross]_ ( _·_ ) denotes the loss of the model on the current samples after multiple rounds of information replay, and _R_ ( _φt, θt_ ) represents the regularization term. Since the parameters of the MRN-Net
contain a large amount of meta knowledge and exhibit strong adaptability to changes in data distribution, we align the parameter spaces of the two networks and introduce a regularization penalty to
constrain the boundaries of the AFT-Net’s parameters. By incorporating this parameter alignment
mechanism, the model complexity is reduced while effectively mitigating instability during online
training, thereby enhancing the model’s ability to rapidly adapt to distribution changes.


4 MODEL PERFORMANCE ANALYSIS


To better understand how the gradient-aware selective fine-tuning strategy can accelerate the adaptation speed of MetaOCDN, we conduct a theoretical analysis of it. At the same time, we prove the
efficiency of MetaOCDN through its regret bound.


4.1 ANALYSIS OF GRADIENT-AWARE SELECTIVE FINE TUNING


For MetaOCDN, there are two main update strategies: (1) selectively adjusting the key layers with
significant gradient fluctuations, and (2) full fine-tuning all model parameters. However, full finetuning not only tends to cause overfitting on the limited number of target samples and catastrophic
forgetting, but also hinders knowledge transfer, while reducing the model’s ability to rapidly adapt
to current samples (Lee et al.), so we analyze it.


The parameters of AFT-Net are denoted as _θ_ . On stationary streaming data (historical samples),
the model loss approaches zero, i.e., _L_ _[AF T]_ ( _θt, D_ _[m]_ ) _→_ 0. We set the selective fine-tuning’s loss
is _L_ _[ft]_ ( _θt, D_ _[t]_ ), for gradient-aware selective fine-tuning, adaptation to current samples is achieved
primarily by updating the layers with large fluctuations, and update process is expressed as follows:


_∂tθ_ _[sle]_ = _−∇θsle_ _L_ _[ft]_ [ �] _θ_ _[sle]_ _, D_ _[t]_ [�] _, ∂tθ_ _[oth]_ = 0 (7)


Let _θ_ _[sle]_ denote the network parameters selected, and _θ_ _[oth]_ denote the parameters that remain unchanged. For full fine-tuning, all layer parameters are updated in:


_∂tθ_ _[sle]_ = _−∇θsle_ _L_ _[ful]_ ( _θ_ _[sle]_ _, D_ _[t]_ ) _, ∂tθ_ _[oth]_ = _−∇θothL_ _[ful]_ ( _θ_ _[oth]_ _, D_ _[t]_ ) (8)


**Theorem 1.** When facing concept drift of varying degrees and types, for any _δ_ _>_ 0, there exists at
least a probability such 1 _−_ _δ_ that the convergence loss of selective fine-tuning the chosen layers is 0,
while the loss caused by full fine-tuning is greater than that of selective fine-tuning. In Appendix A.2,
we will prove this conclusion.


4.2 ANALYSIS OF THE REGRET BOUNDARY


We primarily focus on the performance of the AFT-Net. Let _θ_ 1 and _θ_ 2 denote the parameters of the
AFT-Net at two arbitrary timestamps. For notational convenience, we use _f_ ( _θ_ ) to represent the loss
function _L_ _[AF T]_ and impose the following assumptions on it.


**Assumption 1** ( _Lipschitz Continuity_ ): The loss function _f_ ( _θ_ ) is Lipschitz continuous with respect
to the parameter _θ_ . According to the bounded gradient criterion, _|| ∇f_ ( _θ_ ) _||≤_ _l_ .
**Assumption 2** ( _Bounded Parameter Domain_ ): The parameter domain _W_ has a diameter of Γ, i.e.,
for arbitrary AFT-Net and MRN-Net parameters _φ_ and _θ_ : _|| φ −_ _θ_ _||≤_ Γ _, ∀φ, θ_ _∈W_ .


These assumptions are largely standard in online learning Cesa-Bianchi & Lugosi (2006), and they
are particularly applicable to model adaptation problems in dynamic environment. Specifically, **As-**
**sumption 1** avoids the optimization instability caused by changes in data distribution, ensuring that
the gradient does not explode due to sudden distribution changes when the model is updated, while
**Assumption** **2** provides a feasible framework for theoretical analysis (such as the upper bound of


6


_φ_ _−_ _T_ _[α][out][out]_ - _i∈T_ _[out]_ ���� _φ −_ _θi_ ����2, where _αout_ denotes the learning rate of the MRN-Net, and _T out_

represents the training epoch. Finally, the knowledge of the MRN-Net is fedback to the AFT-Net:


_φ_ _−_ _[α][out][out]_


_T_ _[α][out][out]_ 

Regret). In the context of strong convex functions, these assumptions lead to sublinear convergence
rates, so in the Appendix A.3 we prove that the loss function _f_ ( _θ_ ) is strong convex.


The regret bound is often used to measure the performance of online learning and is defined as the
difference between the cumulative loss of the algorithm in round decision-making and the cumulative loss of the optimal model in the assumption space. Since the AFT-Net uses online gradient
descent to update parameters _θt_ +1 = _θt_ _−_ _η∇θf_ ( _θt_ ), we analyzed it’s regret bound. The regret
boundary of the AFT-Net can be expressed as (Demˇsar, 2006):


_l_ 1 is the boundary of the gradient, _θt_ is the AFT-Net parameter at the current moment, _θ_ represents
the optimal model parameters within the hypothesis space, min _θ∈W_ - _Tt_ =1 _[f][t]_ [(] _[θ]_ [)] [is] [the] [cumulative]
loss in the decision-making of the optimal model round. The proof of Equation 9 is given in the
Appendix A.4, we prove that the AFT-Net has a regret bound approximately equal to _O_ (ln _T/_ 2 _δ_ ).
It indicates that it can converge to a very good effect within step _T_ .


5 EXPERIMENTS


**Experiment Setting.** To comprehensively evaluate the MetaOCDN model, we validated its performance on both classification and regression tasks. For the classification task, we used six datasets,
comprising standard concept drift benchmarks ( _RBFblips_, _Sea_, _Hyperplane_ ) and real-world datasets
( _Kddcup99_, _MIRS_, _Yoga_ ). For the regression task, we utilized three real-world datasets: _ETTH2_,
_Ettm1_, and _WTH_ . Detailed information on all datasets and comparison methods are provided in Appendix B.3. Notably, the AFT-Net and MRN-Net models, used for comparison in this paper, are
both built upon a ResNet12 backbone. Further experimental settings, such as model parameters, are
detailed in Appendix B.1.


5.1 COMPARISONS WITH PRIOR WORK


We compared the performance of MetaOCDN and other methods on the classification task and the
regression task. For the classification task, the average real-time accuracy ( _Avgracc_ ) and cumulative
accuracy ( _Fincacc_ ) were used as evaluation indicators (see Appendix B.4). For the regression task,
we used MSE and MAE as evaluation indicators. The results are as shown in Table 1:


Table 1: Comparison of different methods on classifcation and regression tasks.


**Classification (** _**Avgracc**_ **)** **Regression (** _**MSE**_ **)** **AvgRank**
_**RBFblips**_ _**Sea**_ _**Hyperplane**_ _**Kddcup99**_ _**MIRS**_ _**Yoga**_ _**ETTH2**_ _**ETtm1**_ _**WTH**_


DWM 55.40(16) 69.07(11) 87.20(3) 83.60(5) 44.71(15) 52.54(4) 9.596(9) 7.949(9) 0.904(4) 8 _._ 44
OBC 88.05(7) 60.68(15) 74.59(13) 96.41(2) 48.94(14) 47.04(15) 8.478(8) 5.073(10) _−_ 10 _._ 5
RUS 90.58(6) 61.00(14) 73.37(14) 15.98(17) 61.51(2) 48.92(12) 43.69(10) 67.403(12) 10.664(11) 11 _._ 11
LEV 93.27(5) 60.51(16) 71.25(16) 96.03(3) 58.00(8) 43.45(17) 54.548(11) 25.013(11) 10.00(10) 11 _._ 22
ARF 83.27(12) 67.06(12) 77.33(11) **99.38(1)** 59.92(6) 51.14(7) 50.9(12) 22.54(10) 4.11(8) 8 _._ 78
DNN 87.16(8) 71.55(10) 85.78(6) 71.86(9) 50.13(13) 49.84(11) 178.8(13) 91.59(14) 90.69(15) 11 _._ 33
ResNet 83.00(13) 74.48(8) 86.37(5) 65.35(10) 37.75(17) 46.32(16) 801.9(14) 225.1(15) 47.58(13) 12 _._ 44
Highway 84.82(9) 76.84(5) 88.41(2) 75.37(8) 53.48(11) 51.54(5) 775.6(16) 81.94(13) 2875.1(16) 9 _._ 33
HBP 93.50(4) 77.71(3) 86.92(4) 76.70(7) 54.13(10) 53.60(3) 685.4(15) 232.63(17) 40.56(12) 8 _._ 22
DenseNet 94.42(2) 75.44(6) **89.05(1)** 87.56(4) 60.87(4) 54.13(2) 801.92(17) 225.11(16) 47.58(14) 7 _._ 22
Informer 57.67(15) 72.43(9) 76.11(12) 23.31(11) 52.64(12) 48.85(13) 1.69(7) 1.18(7) 1.10(6) 10 _._ 56
ER 84.15(10) 76.89(4) 81.47(10) 23.01(15) 60.87(5) 50.84(8) 0.264(6) 0.149(5) 1.074(5) 7 _._ 44
DER++ 83.45(11) 74.48(8) 71.79(15) 23.27(12) 58.72(7) 50.47(9) 0.1742(4) 0.092(3) 4.156(9) 8 _._ 89
FsNet 93.99(3) 78.21(2) 84.23(7) 22.56(16) 61.07(3) 50.35(10) 0.069(2) 0.163(6) 1.732(7) 6 _._ 44
Time-TCN 58.63(14) 61.11(13) 84.23(7) 23.24(13) 57.93(9) 51.27(6) 0.234(5) 0.101(4) 0.553(3) 8 _._ 44
PatchTST 26.75(17) 39.8(17) 49.8(17) 23.2(14) 44.38(16) 48.52(14) 0.138(3) 0.077(2) 0.224(1) 11 _._ 22


**MetaOCDN** **97.62(1)** **79.28(1)** 82.64(9) 82.11(6) **61.92(1)** **54.24(1)** **0.039(1)** **0.031(1)** **0.27(2)** **2.55**


As shown in Table 1, our proposed method performs well on synthetic datasets exhibiting abrupt and
gradual concept drift, but performs relatively poorly on the incremental drift dataset _Hyperplane_ .
This is because incremental drift spans a long duration and changes only slightly over time without
clear drift points. As a result, during the model update process, the AFT-Net tends to freeze more
layers, preventing timely updates that would allow it to capture subtle distribution shifts, thereby
degrading performance. Meanwhile, on real-world datasets, our method achieves good results on
_MIRS_ and _Yoga_, but performs less effectively on _Kddcup99_ . This is primarily because _Kddcup99_


7


- _ft_ ( _θ_ ) = _O_ ( [(] _[l]_ [1][ +] _[ β]_ [1][Γ)][2]

2 _δ_

_t_ =1


ln _T_ ) (9)
2 _δ_


_regret_ =


_T_

- _ft_ ( _θt_ ) _−_ min

_θ∈W_
_t_ =1


_T_


consists of discrete features, while neural networks are black-box models and often struggle to interpret such discrete attributes. In contrast, ARF, based on the recursive splitting mechanism of random
forests, can naturally adapt to the partitioning of discrete feature spaces. Its information gain criterion is inherently compatible with categorical variables, enabling it to achieve superior performance
on such datasets. In the regression task, MetaOCDN demonstrates strong performance. ResNet
enhances the training of deep models through its residual structure, enabling it to capture complex
patterns in time series data. Additionally, the MRN-Net extracts rich structural representations from
historical samples, providing a significant advantage when modeling time series data.


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|~~Informer~~<br>ER|~~Pa~~<br>Hi|~~tchTST~~<br>hway||~~Resnet~~<br>DenseNe|t|~~OBC~~<br>RUS|


|Critical Distance 17 16 15 14 13|= 6.7792 12 11 10 9 8 7 6 5 4 3 2 1|
|---|---|
|12.44|2.5<br>6.4<br>7.2<br>7.4<br>8.2<br>8.4<br>8.4<br>8.7<br>|
|11.33<br>|11.33<br>|
|11.22<br>|11.22<br>|
|11.22<br>|11.22<br>|
|11.11<br>|11.11<br>|
|10.56<br>|10.56<br>|
|10.5<br>|10.5<br>|
|9.33<br>|9.33<br>|


To evaluate the convergence speed of MetaOCDN after concept drift occurs, this section compares and analyzes the recovery performance at different drift points. During the determination of convergence points, the
convergence threshold is set to _ε_ = 0 _._ 8. Table 2 presents the convergence performance of various
algorithms on five datasets with known drift points. In the table, each row lists three values representing the recovery scores of each algorithm at the early, middle, and late drift points, respectively.
“-” indicates that the model fails to learn features to fit the data at that drift point.

|Col1|Table 2: RSA comparison of different methods|
|---|---|
|Datasets|DWM<br>OBC<br>RUS<br>LEV<br>ARF<br>DNN<br>ResNet<br>Highway<br>HBP|
|_RBFblips_<br><br>_Sea_<br>|2.16/0.15/0.12<br>0.54/0.49/0.26<br>0.87/0.56/0.82<br>0.68/0.27/0.31<br>-/0.28/0.71<br>0.14/0.11/0.10<br>0.63/0.11/-<br>0.10/0.12/-<br>0.13/0.10/0.06<br>1.10/1.16/0.30<br>1.37/1.15/0.38<br>1.37/1.13/0.35<br>1.4/0.38/1.17<br>0.5/1.0/0.33<br>1.93/1.50/0.31<br>1.78/0.63/0.22<br>1.70/1.00/0.21<br>-/2.17/0.30|
|Datasets|DenseNet<br>Informer<br>ER<br>DER++<br>FsNet<br>Time-TCN<br>PatchTST<br>**Ours**|
|_RBFblips_<br><br>_Sea_<br>|0.46/0.11/0.25<br>0.51/0.40/1.01<br>0.11/0.17/0.83<br>0.45/0.15/0.03<br>0.05/0.07/0.31<br>-/0.77/1.45<br>-/0.76/1.45<br>**0.13**/**0.03**/**0.02**<br>0.23/0.56/0.30<br>0.54/0.23/0.23<br>0.21/0.64/0.21<br>0.23/0.21/0.19<br>0.22/0.48/0.20<br>0.63/0.61/0.60<br>0.63/0.60/0.59<br>**0.21**/**0.43**/**0.17**|


Table 2 shows that MetaOCDN converges well on datasets with two known drift points, quickly
regaining high accuracy after drift. This benefit stems from the gradient-aware selective fine-tuning
strategy, which focuses updates on distribution-sensitive layers and thus achieves faster convergence.


5.2 ABLATION EXPERIMENT


**Gradient-aware** **selective** **fine-tuning** **analysis.** Fig. 5 illustrates the gradient variations of
the four residual blocks in AFT-Net on benchmark datasets with concept drift. Based on
this, we evaluate the convergence speed of AFT-Net under different residual block freezing settings to validate the effectiveness of gradient-aware selective fine-tuning analysis.


8


**(c)** _**Hyperplane**_


0 25 50 75 100 125 150 175 200
Timestamp


1.0


0.8


0.6


0.4


0.2


0.0


0 25 50 75 100 125 150 175 200
Timestamp


**(b)** _**Sea**_


0 25 50 75 100 125 150 175 200
Timestamp


0.9


0.8


0.7


0.6


0.5


0.4


0.8


0.7


0.6


0.5


0.4


0.3


Figure 3: Comparison of _Fincacc_ of different methods


Fig. 3 shows the _Fincacc_ of each algorithm over different time steps. Similarly, MetaOCDN performs poorly on the _Hyperplane_ but achieves good results on the remaining datasets. The remaining
experimental results are in Appendix B.5.


**Statistical Analysis.** This paper also employs the BonferroniDunn test to evaluate the statistical significance of differences
(Critical Difference) among all methods. According to the
calculation, under the significance level _α_ = 0 _._ 05, the critical difference (CD) is 6.72. The statistical analysis results
are shown in Fig. 4. In the figure, methods that do not show
a significant difference are connected with red lines. The results indicate that, from a statistical perspective, the method
proposed in this chapter demonstrates a clear advantage.


Figure 4: Bonferroni-Dunn test of
all methods


We present the results of the model on the _RBFBlips_, with
the remaining datasets provided in Appendix B.6. The line
plots depict the gradient variations of the four residual blocks
around three different drift points; the green bars illustrate the
convergence speed when different residual blocks are frozen;
and the blue bars compare model performance and parameter updates between selective fine-tuning and full fine-tuning.
Experimental results show that freezing residual blocks with
large gradient fluctuations diminishes the model’s rapid adaptation ability, whereas gradient-aware selective fine-tuning
not only achieves higher accuracy than full fine-tuning but
also significantly reduces parameter overhead.


We also compared the convergence speed and parameter scale
of MetaOCDN’s gradient-aware selective fine-tuning strategy
with full fine-tuning on real-world datasets.


Figure 6: A partial ablation study results figure.


Fig. 6(a) presents a comparison of the model under two update strategies in terms of convergence
speed and parameter overhead. The gradient-aware selective fine-tuning strategy enables the model
to converge to superior performance within a shorter time while significantly reducing the number
of parameters required for updates, thereby improving training efficiency and resource utilization
without sacrificing accuracy.


**Robustness Analysis of MRN-Net.** We compared the adaptability of MetaOCDN under MRN-Net
and AFT-Net collaboration versus AFT-Net alone on three datasets with explicit drift points. The
evaluation metrics include _RSA_ (Recovery Speed after Adaptation), which measures the model’s
real-time convergence ability during drift, and _DCE_ (Drift Cumulative Error), which captures the
accumulated error during the drift adaptation phase. Partial results are shown in Fig. 6(b), with
the remaining results provided in Appendix B.6. Experimental results indicate that MetaOCDN
with both networks collaborating exhibits significantly smaller overall accuracy fluctuations. During
changes in data distribution, MRN-Net provides more robust initialization or adjustment signals for
the online adaptation process, enabling the model to converge more quickly to the new distribution
while substantially reducing accumulated error during the drift adaptation phase.


6 CONCLUSION


Inspired by the theory of _Complementary Learning Systems_, we propose MetaOCDN. This approach
constructs a meta optimized complementary dual network architecture consisting of an Adaptive
Fine-Tuning Network (AFT-Net) and a Meta-Representation Network (MRN-Net), analogous to the
cooperative mechanism between the hippocampus and neocortex in the human brain. To address the
challenge of concept drift in open environments, we focus on enhancing the model’s rapid adaptation
capability and improving its robustness, which effectively mitigates instability during online training
and boosts overall performance under dynamic data distributions.


9


Figure 5: Gradient variation and result analysis


REPRODUCIBILITY STATEMENT


For reproducibility, we elaborate on the overall pipeline of our work in Section 3. And in Appendix B.1, we provide a description of the model architecture and key parameter settings. In the
future, we will upload the source code to a public GitHub repository.


ETHICS STATEMENT


MetaOCDN aims to improve the robustness and adaptability of models in streaming data mining
tasks with concept drift, which could be beneficial in the real world, such as financial analysis and
anomaly detection as described. All experiments were based on publicly available standard datasets
and did not involve any personal privacy or sensitive information. They also did not involve human
or animal experiments and did not require additional ethical approval.


REFERENCES


Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars.
Memory aware synapses: Learning what (not) to forget. In _Proceedings of the European Confer-_
_ence on Computer Vision (ECCV)_, pp. 139–154, 2018.


Shaojie Bai, J Zico Kolter, and Vladlen Koltun. An empirical evaluation of generic convolutional
and recurrent networks for sequence modeling. _arXiv preprint arXiv:1803.01271_, 2018.


Peter Bartlett, Elad Hazan, and Alexander Rakhlin. Adaptive online gradient descent. _Advances in_
_Neural Information Processing Systems_, 20, 2007.


Albert Bifet and Ricard Gavalda. Learning from time-changing data with adaptive windowing. In
_Proceedings_ _of_ _the_ _2007_ _SIAM_ _International_ _Conference_ _on_ _Data_ _Mining_, pp. 443–448. SIAM,
2007.


Albert Bifet, Geoff Holmes, and Bernhard Pfahringer. Leveraging bagging for evolving data streams.
In _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_, pp.
135–150. Springer, 2010.


Dariusz Brzezinski and Jerzy Stefanowski. Prequential auc for classifier evaluation and drift detection in evolving data streams. In _International_ _Workshop_ _on_ _New_ _Frontiers_ _in_ _Mining_ _Complex_
_Patterns_, pp. 87–101. Springer, 2014.


Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, and Simone Calderara. Dark experience for general continual learning: a strong, simple baseline. _Advances in Neural Information_
_Processing Systems_, 33:15920–15930, 2020.


Ruichu Cai, Haiqin Huang, Zhifan Jiang, Zijian Li, Changze Zhou, Yuequn Liu, Yuming Liu, and
Zhifeng Hao. Disentangling long-short term state under unknown interventions for online time
series forecasting. In _Proceedings of the AAAI Conference on Artificial Intelligence_, volume 39,
pp. 15641–15649, 2025.


Alberto Cano and Bartosz Krawczyk. Rose: Robust online self-adjusting ensemble for continual
learning on imbalanced drifting data streams. _Machine Learning_, 111(7):2561–2599, 2022.


Nicolo Cesa-Bianchi and G´abor Lugosi. _Prediction,_ _learning,_ _and_ _games_ . Cambridge university
press, 2006.


Arslan Chaudhry, Marcus Rohrbach, Mohamed Elhoseiny, Thalaiyasingam Ajanthan, Puneet K
Dokania, Philip HS Torr, and Marc’Aurelio Ranzato. On tiny episodic memories in continual
learning. _arXiv preprint arXiv:1902.10486_, 2019.


Lenaic Chizat, Pierre Roussillon, Flavien L´eger, Franc¸ois-Xavier Vialard, and Gabriel Peyr´e. Faster
wasserstein distance estimation with the sinkhorn divergence. _Advances_ _in_ _Neural_ _Information_
_Processing Systems_, 33:2257–2269, 2020.


Janez Demˇsar. Statistical comparisons of classifiers over multiple data sets. _Journal_ _of_ _Machine_
_Learning Research_, 7(Jan):1–30, 2006.


10


Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation
of deep networks. In _International_ _Conference_ _on_ _Machine_ _Learning_, pp. 1126–1135. PMLR,
2017.


Joao Gama, Pedro Medas, Gladys Castillo, and Pedro Rodrigues. Learning with drift detection. In
_Brazilian symposium on artificial intelligence_, pp. 286–295. Springer, 2004.


Heitor M Gomes, Albert Bifet, Jesse Read, Jean Paul Barddal, Fabr´ıcio Enembreck, Bernhard
Pfharinger, Geoff Holmes, and Talel Abdessalem. Adaptive random forests for evolving data
stream classification. _Machine Learning_, 106(9):1469–1495, 2017.


Husheng Guo, Yang Zhang, and Wenjian Wang. Dynamical targeted ensemble learning for streaming data with concept drift. _IEEE Transactions on Knowledge and Data Engineering_, 2024.


Yiwen Guo, Anbang Yao, and Yurong Chen. Dynamic network surgery for efficient dnns. _Advances_
_in Neural Information Processing Systems_, 29, 2016.


Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In _Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition_, pp.
770–778, 2016.


Gao Huang, Zhuang Liu, Geoff Pleiss, Laurens Van Der Maaten, and Kilian Q Weinberger. Convolutional networks with dense connectivity. _IEEE Transactions on Pattern Analysis and Machine_
_Intelligence_, 44(12):8704–8716, 2019.


J Zico Kolter and Marcus A Maloof. Dynamic weighted majority: An ensemble method for drifting
concepts. _The Journal of Machine Learning Research_, 8:2755–2790, 2007.


Bj¨orn Kr¨uger, Anna V¨ogele, Tobias Willig, Angela Yao, Reinhard Klein, and Andreas Weber. Efficient unsupervised temporal segmentation of motion data. _IEEE Transactions on Multimedia_, 19
(4):797–812, 2016.


Dharshan Kumaran, Demis Hassabis, and James L McClelland. What learning systems do intelligent
agents need? complementary learning systems theory updated. _Trends in Cognitive Sciences_, 20
(7):512–534, 2016.


Yoonho Lee, Annie S Chen, Fahim Tajwar, Ananya Kumar, Huaxiu Yao, Percy Liang, and Chelsea
Finn. Surgical fine-tuning improves adaptation to distribution shifts (2023). _URL_ _https://arxiv._
_org/abs/2210.11466_ .


Wendi Li, Xiao Yang, Weiqing Liu, Yingce Xia, and Jiang Bian. Ddg-da: Data distribution generation for predictable concept drift adaptation. In _Proceedings of the AAAI Conference on Artificial_
_Intelligence_, volume 36, pp. 4092–4100, 2022.


Anjin Liu, Jie Lu, Yiliao Song, Junyu Xuan, and Guangquan Zhang. Concept drift detection delay
index. _IEEE Transactions on Knowledge and Data Engineering_, 35(5):4585–4597, 2022.


Xiao Liu, Fanjin Zhang, Zhenyu Hou, Li Mian, Zhaoyu Wang, Jing Zhang, and Jie Tang. Selfsupervised learning: Generative or contrastive. _IEEE Transactions on Dnowledge and Data En-_
_gineering_, 35(1):857–876, 2021.


Jie Lu, Anjin Liu, Fan Dong, Feng Gu, Jo˜ao Gama, and Guangquan Zhang. Learning under concept
drift: A review. _IEEE_ _Transactions_ _on_ _Knowledge_ _and_ _Data_ _Engineering_, 31(12):2346–2363,
2019.


James L McClelland, Bruce L McNaughton, and Randall C O’Reilly. Why there are complementary
learning systems in the hippocampus and neocortex: insights from the successes and failures of
connectionist models of learning and memory. _Psychological Review_, 102(3):419, 1995.


Q NIEY, NH NGUYEN, et al. A time series is worth 64 words: Long-term forecasting with transformers, 2023.


Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. _arXiv preprint arXiv:1807.03748_, 2018.


11


Nikunj C Oza and Stuart Russell. Experimental comparisons of online and batch versions of bagging and boosting. In _Proceedings_ _of_ _the_ _seventh_ _ACM_ _SIGKDD_ _International_ _Conference_ _on_
_Knowledge Discovery and Data Mining_, pp. 359–364, 2001.


Quang Pham, Chenghao Liu, Doyen Sahoo, and Steven CH Hoi. Learning fast and slow for online
time series forecasting. _arXiv preprint arXiv:2202.11672_, 2022.


Quang Pham, Chenghao Liu, and Steven CH Hoi. Continual learning, fast and slow. _IEEE Transac-_
_tions on Pattern Analysis and Machine Intelligence_, 46(1):134–149, 2023.


Doyen Sahoo, Quang Pham, Jing Lu, and Steven CH Hoi. Online deep learning: Learning deep
neural networks on the fly. _arXiv preprint arXiv:1711.03705_, 2017.


Thalles Silva, Helio Pedrini, and Ad´ın Ram´ırez Rivera. Learning from memory: Non-parametric
memory augmented self-supervised learning of visual features. _arXiv preprint arXiv:2407.17486_,
2024.


Alessandro Sordoni, Nouha Dziri, Hannes Schulz, Geoff Gordon, Philip Bachman, and Remi Tachet
Des Combes. Decomposed mutual information estimation for contrastive representation learning.
In _International Conference on Machine Learning_, pp. 9859–9869. PMLR, 2021.


Rupesh K Srivastava, Klaus Greff, and J¨urgen Schmidhuber. Training very deep networks. _Advances_
_in Neural Information Processing Systems_, 28, 2015.


Ke Wan, Yi Liang, and Susik Yoon. Online drift detection with maximum concept discrepancy. In
_Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining_,
pp. 2924–2935, 2024.


Boyu Wang and Joelle Pineau. Online bagging and boosting for imbalanced data streams. _IEEE_
_Transactions on Knowledge and Data Engineering_, 28(12):3353–3366, 2016.


Qingsong Wen, Weiqi Chen, Liang Sun, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan, et al.
Onenet: Enhancing time series forecasting models under concept drift by online ensembling.
_Advances in Neural Information Processing Systems_, 36:69949–69980, 2023.


Shuo Yang, Xinran Zheng, Jinze Li, Jinfeng Xu, Xingjun Wang, and Edith CH Ngai. Recda: Concept
drift adaptation with representation enhancement for network intrusion detection. In _Proceedings_
_of_ _the_ _30th_ _ACM_ _SIGKDD_ _Conference_ _on_ _Knowledge_ _Discovery_ _and_ _Data_ _Mining_, pp. 3818–
3828, 2024.


Hang Yu, Jinpeng Li, Jie Lu, Yiliao Song, Shaorong Xie, and Guangquan Zhang. Type-ldd: A typedriven lite concept drift detector for data streams. _IEEE_ _Transactions_ _on_ _Knowledge_ _and_ _Data_
_Engineering_, 36(12):9476–9489, 2023.


Hang Zhang, Weike Liu, and Qingbao Liu. Reinforcement online active learning ensemble for
drifting imbalanced data streams. _IEEE_ _Transactions_ _on_ _Knowledge_ _and_ _Data_ _Engineering_, 34
(8):3971–3983, 2020.


Hao Zhang, Chenglin Li, Wenrui Dai, Junni Zou, and Hongkai Xiong. Fedcr: Personalized federated learning based on across-client common representation with conditional mutual information
regularization. In _International_ _Conference_ _on_ _Machine_ _Learning_, pp. 41314–41330. PMLR,
2023.


Yuxin Zhang, Jindong Wang, Yiqiang Chen, Han Yu, and Tao Qin. Adaptive memory networks with
self-supervised learning for unsupervised anomaly detection. _IEEE Transactions on Knowledge_
_and Data Engineering_, 35(12):12068–12080, 2022.


H Zhou, S Zhang, J Peng, S Zhang, J Li, H Xiong, and W Zhang Informer. Beyond efficient
transformer for long sequence time-series forecasting., 2021. _DOI: https://doi. org/10.1609/aaai._
_v35i12_, 17325, 2023.


Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang.
Informer: Beyond efficient transformer for long sequence time-series forecasting. In _Proceedings_
_of the AAAI Conference on Artificial Intelligence_, volume 35, pp. 11106–11115, 2021.


12


A APPENDIX


A.1 PROOF OF SECTION 3.2


In Section 3.2, we construct a Meta Representation Network (MRN-Net) to learn structured knowledge from historical samples. To enhance the model’s representation capability, we design a selfsupervised duality loss, which consists of similarity loss and difference loss. The similarity loss
reinforces representation consistency among similar samples, while the difference loss pushes apart
representations of unrelated samples. This dual mechanism ensures semantic clustering while improving feature discriminability, leading to more robust and generalizable representations.


**Self-supervised** **similarity** **loss.** We estimate a lower bound of mutual information to enable the
model to capture shared features. The mutual information lower bound is expressed as follows:


**Self-supervised difference loss.** Similarly, we construct the difference loss by minimizing mutual
information. In practice, researchers often use an upper bound of mutual information as an approximation for this minimization. By introducing a random variable _N_ (negative samples representations
in this paper) and applying the definition of mutual information, we obtain:
_I_ ( _z_ _[−]_ ; _z_ _[t]_ ) = _I_ ( _z_ _[−]_ ; _z_ _[t]_ ; _N_ ) + _I_ ( _z_ _[−]_ ; _z_ _[t]_ _|N_ ) (14)
and _I_ ( _z_ _[−]_ ; _z_ _[t]_ _|N_ ) is the conditional mutual information. Since _z_ _[−]_ is sampled from the set _N_ and is
conditionally independent of the current samples feature set _z_ _[t]_, we can deduce that:
_I_ ( _z_ _[−]_ ; _z_ _[t]_ _|N_ ) = _H_ ( _z_ _[−]_ _| N_ ) _−_ _H_ ( _z_ _[−]_ _| N_ _, z_ _[t]_ )

(15)
= _H_ ( _z_ _[−]_ _| N_ ) _−_ _H_ ( _z_ _[−]_ _| N_ ) = 0


13


_[|][ z][t]_ [)]
_ILow_ ( _z_ [+] ; _z_ _[t]_ ) = E _p_ ( _z_ + _,zt_ ) log _[p]_ [(] _[z]_ [+]


(10)
_p_ ( _z_ [+] )


_[z]_ [+] _[|][ z][t]_ [)] _[|][ z][t]_ [)]

_≥_ E _p_ ( _z_ + _,zt_ ) log _[q]_ [(] _[z]_ [+]
_p_ ( _z_ [+] ) _p_ ( _z_ [+] )


where _p_ ( _z_ [+] _|_ _z_ _[t]_ ) is the conditional distribution of _z_ [+] given _z_ _[t]_, E _p_ ( _z_ + _,zt_ ) is the expectation under
the joint distribution _p_ ( _z_ [+] _, z_ _[t]_ ), and _q_ ( _z_ [+] _|_ _z_ _[t]_ ) denotes the variational distribution that _p_ ( _z_ [+] _, z_ _[t]_ ).
Specifically, we independently sample a set of samples _{z_ 1 [+] _[,][ · · ·]_ _[, z]_ _n_ [+] _[}]_ [ from the proposal distribution]

_e_ _[ψ]_ [(] _[zt,z]_ [+)]
_π_ ( _z_ [+] ), and assign the importance weight _wz_ + = - _ni_ _[e][ψ]_ [(] _[zt,z]_ _i_ [+][)] [.] _[ψ]_ [(] _[z][T][, z]_ _i_ [+][)][ is the cosine similarity.]

Given the sample set and the target sample, _q_ ( _z_ [+] _| z_ _[t]_ ) can be replaced by:


_n · e_ _[ψ]_ [(] _[z][t][,z]_ [+][)]
1: [+] _n_ [) =] _[ π]_ [(] _[z]_ [+][)] _[ ·]_ _[t]_ [+] _[n]_


_q_ ( _z_ [+] _| z_ _[t]_ _, z_ [+]


(11)
_e_ _[ψ]_ [(] _[z][t][,z]_ [+][)] + [�] _i_ _[n]_ =2 _[e][ψ]_ [(] _[z][t][,z]_ _i_ [+][)]


In summary, the mutual information lower bound is given by (Sordoni et al., 2021):


_ILow_ ( _z_ [+] ; _z_ _[t]_ ) _≥_ E _p_ ( _z_ + _,zt_ ) log _[q]_ [(] _[z]_ [+] _[|][ z][t][, z]_ 1: [+] _n_ [)]
_p_ ( _z_ [+] )


_≥_ E _p_ ( _z_ + _,zt_ )


E _π_ ( _z_ 1:+ _n_ [)][ log] _[np]_ [(] _[z]_ _p_ [+] ( _z_ [)][+] _[ ·][ w]_ ) _[z]_ [+]


_n · e_ _[ψ]_ [(] _[z][t][,z]_ [+][)]
= E _p_ ( _z_ + _,zt_ ) _⌊_ E _π_ ( _z_ 1:+ _n_ [)][ log] _e_ _[ψ]_ [(] _[z][t][,z]_ 1 [+][)] + [�] _i_ _[n]_ =2 _[e][ψ]_ [(] _[z][t][,z]_ _i_ [+][)] _[⌋]_


(12)


= E _p_ ( _z_ + _,zt_ ) _π_ ( _z_ 1:+ _n_ [)]


_e_ _[ψ]_ [(] _[z][t][,z]_ [+][)]
log


1 - _n_ _i_ [)]
_n_ _i_ =1 _[e][ψ]_ [(] _[z][t][,z]_ [+]


the second step is derived from the Jensen’s inequality, where _p_ ( _z_ [+] ) approximates _π_ ( _z_ [+] ). We construct the similarity loss by maximizing mutual information, which is implemented by minimizing
a negative lower bound of mutual information. The similarity loss can be expressed as:


_ℓ_ [sim] = _−I_ Low - _z_ [+] ; _z_ _[t]_ [�] = _−_ [1]

_n_


_ℓ_ [sim] = _−I_ Low - _z_ [+] ; _z_ _[t]_ [�] = _−_ [1]


_n_

- log _e_ _[ψ]_ [(] _[z]_ _j_ _[t][,z]_ _j_ [+][)] (13)

_j_ =1 - _ni_ =1 _[e][ψ]_ [(] _[z]_ _j_ _[t][,z]_ _i_ [+][)] + _ξ_


_H_ ( _·_ ) denotes the information entropy. By combining Eq. 14 and Eq. 15, we can derive:


_I_ ( _z_ _[−]_ ; _z_ _[t]_ ) = _I_ ( _z_ _[−]_ ; _z_ _[t]_ ; _N_ )

= _I_ ( _N_ ; _z_ _[−]_ ) _−_ _I_ ( _N_ ; _z_ _[−]_ _|z_ _[t]_ )

= _I_ ( _N_ ; _z_ _[−]_ ) + _I_ ( _N_ ; _z_ _[t]_ ) _−_ _I_ ( _N_ ; _z_ _[−]_ _, z_ _[t]_ )


The Eq. 22 is a convex function with respect to _θ∞_ _[sle]_ [.] [This] [means] [the] [final] [parameters] [will] [con-]
verge to a global minimum, i.e., _L_ _[ft]_ ( _θ∞_ _[sle][, θ]_ [oth][;] _[ D][t]_ [)] [=] [0][.] [Moreover,] [since] _[θ]_ _∞_ [oth] = _θ_ 0 [oth], we have
_L_ _[ft]_ ( _θ∞_ _[sle][, θ]_ 0 [oth] ; _D_ _[t]_ ) = 0.


**Lemma 2:** With at least probability 1, full fine-tuning yields a non-zero loss at all times.


_L_ _[ful]_ ( _**θ**_ _∞_ _[sle][,]_ _**[ θ]**_ _∞_ _[oth]_ [;] _[ D][t]_ [)] _[ >]_ [ 0] (23)


_**Proof:**_ Suppose the model function space is _F_ = _{fθ_ : _θ_ _∈_ Θ _}_, and the current samples _D_ _[t]_ satisfy
the mapping _yi_ _[t]_ [=] _[ f]_ _∗_ _[ t]_ [(] _[x][t]_ _i_ [)][, with a probability distribution of] _[ P][t]_ [(] _[x, y]_ [)][.] [The mapping between features]
and labels in historical samples is given by _yi_ _[m]_ = _f∗_ _[m]_ [(] _[x][m]_ _i_ [)][,] [after] [concept] [drift] [occurs,] [we] [have]
_yi_ _[t]_ [=] _[f]_ _∗_ _[ t]_ [(] _[x][t]_ _i_ [)] [=] _[y]_ _i_ _[m]_ = _f∗_ _[m]_ [(] _[x][m]_ _i_ [)][,] [due] [to] [the] [limited] [representation] [capacity] [of] [the] [model,] [which]
cannot adapt to the new data distribution in time. Therefore, _f∗_ _[t]_ [(] _[x][t]_ _i_ [)] _[ ̸∈F]_ [.] [The expected squared loss]
of the model under the new distribution is:


_L_ _[ful]_ ( _θ_ [sle] _, θ_ [oth] ; _D_ _[t]_ ) = E _x−Pt_ ( _x_ )[( _fθ_ ( _x_ ) _−_ _f∗_ _[t]_ [(] _[x]_ [))][2][]] (24)


14


(16)


The third step is derived using the chain rule of mutual information. Based on the variational information bottleneck theory, the variational upper bounds of _I_ ( _N_ ; _z_ _[−]_ ) and _I_ ( _N_ ; _z_ _[t]_ ) can be obtained,
yielding an upper bound of the mutual information _I_ ( _z_ _[−]_ ; _z_ _[t]_ ) as follows:


_IUp_ ( _z_ _[−]_ ; _z_ _[t]_ ) = _I_ ( _N_ ; _z_ _[−]_ ) + _I_ ( _N_ ; _z_ _[t]_ ) _−_ _I_ ( _N_ ; _z_ _[−]_ _, z_ _[t]_ )


_≤_ _DKL_ ( _p_ ( _z_ _[−]_ _| N_ ) _|| q_ ( _z_ _[−]_ )) + _DKL_ ( _p_ ( _z_ _[t]_ _| N_ ) _|| q_ ( _z_ _[t]_ ))

_−_ E _q_ ( _z−|N_ ) _q_ ( _zt|N_ )[log _p_ ( _N_ _| z_ _[−]_ _, z_ _[t]_ )]


The difference loss is approximated as:


(17)


_ℓ_ _[diff]_ _≈_ _DKL_ ( _p_ ( _z_ _[−]_ _| N_ ) _|| q_ ( _z_ _[−]_ )) + _DKL_ ( _p_ ( _z_ _[T]_ _| N_ ) _|| q_ ( _z_ _[t]_ )) (18)


A.2 PROOF OF THEOREM 1


**Lemma 1:** For any _δ_ _>_ 0, assuming the model has previously converged on a stationary distribution
_n_ _>_ 10 _d_ [orth] log [2] _δ_ [,] [there] [exists] [at] [least a] [probability] [1] _[ −]_ _[δ]_ [such] [that the loss] [under selective] [fine-]

tuning becomes zero (Lee et al.):


_L_ _[ft]_ ( _θ∞_ _[sle][, θ]_ _∞_ _[oth]_ [;] _[ D][t]_ [) = 0] (19)


_**Proof:**_ When only the selected layers are updated, the remaining frozen layers remain unchanged,
so we have _θ∞_ _[oth]_ = _θ_ 0 _[oth]_ . These layers stay frozen during the fine-tuning process. The loss function
of the model on the current samples _D_ _[t]_ is defined as:


_L_ _[ft]_ ( _**θ**_ _∞_ _[sle][,]_ _**[ θ]**_ 0 _[oth]_ ; _D_ _[t]_ ) = [1]

_n_


_n_

- _ℓ_ ( _f_ ( _x_ _[t]_ _i_ [;] _**[ θ]**_ _∞_ _[sle][,]_ _**[ θ]**_ 0 _[oth]_ ) _, yi_ _[t]_ [)] (20)


_i_ =1


_f_ ( _·_ ; _θ_ ) denotes the forward propagation function, and _ℓ_ ( _·_ ) is the squared loss. We set the model’s
output layer as a linear layer and freeze the model parameters _θ_ 0 _[oth]_, so the loss function becomes a
convex function with respect to _θ∞_ _[slϵ]_ [.] [The model output is expressed as:]


_f_ ( _x_ _[t]_ _i_ [;] _**[ θ]**_ _∞_ _[sle][,]_ _**[ θ]**_ 0 _[oth]_ ) = _**θ**_ _∞_ _[sle]_ _[·]_ _**[ ϕ]**_ [(] _[x]_ _i_ _[t]_ [;] _**[ θ]**_ 0 _[oth]_ ) (21)
_ϕ_ ( _x_ _[t]_ _i_ [;] _[ θ]_ 0 _[oth]_ ) is the nonlinear transformation from the frozen layers, then:


_L_ _[ft]_ ( _θ∞_ _[sle]_ [) =]


_n_
�( _θ∞_ _[sle]_ _[·]_ _**[ ϕ]**_ [(] _[x][t]_ _i_ [)] _[ −]_ _[y]_ _i_ _[t]_ [)][2] (22)


_i_ =1


Full fine-tuning means adjusting all parameters _θ_ _[all]_ = _θ_ _[sle]_ + _θ_ _[oth]_ to minimize the loss:

_L_ _[ful]_ _∗_ ( _θ_ [all] ; _D_ _[t]_ ) = inf _∗_ ( _θ_ [sle] _, θ_ [oth] ; _D_ _[t]_ ) (25)
_θ∈_ Θ _[L][ful]_

since _f∗_ _[t]_ [(] _[x][t]_ _i_ [)] _[ ̸∈F]_ [, the model incurs an approximation error:]

_ϵapprox_ := _θ_ inf _∈_ Θ [E] _[x][∼][P][t]_ [(] _[x]_ [)][[(] _[f][θ]_ [(] _[x]_ [)] _[ −]_ _[f]_ _∗_ _[ t]_ [(] _[x]_ [))][2][]] _[ >]_ [ 0] (26)


Therefore _Lft∞_ [(] _[θ]_ [sle] _[, θ]_ [oth][;] _[ D][t]_ [)] _[ >]_ [ 0][, so Lemma 2 holds.] [Based on Lemma 1 and Lemma 2, we have:]
_L_ _[ful]_ ( _θ_ _[all]_ _, D_ _[t]_ ) _≥L_ _[ft]_ ( _θ_ _[sle]_ _, D_ _[t]_ ) = 0 _, ∀t_ . Therefore, Theorem 1 holds.


A.3 ANALYSIS OF THE REGRET BOUNDARY


**Proposition** **1:** The loss function _f_ ( _θ_ ) of ATF-Net is strongly convex and satisfies the following
inequality for any parameters _θ_ 1, _θ_ 2: _f_ ( _θ_ 1) _≥_ _f_ ( _θ_ 2) + _∇f_ ( _θ_ 1) _[T]_ ( _θ_ 2 _−_ _θ_ 1).


_**Proof**_ _:_ As can be seen from the last paragraph of Section 3.3, the loss function _f_ ( _θ_ ) can be expressed
as: _f_ ( _θ_ ) = _L_ _[KD]_ + _R_ ( _φ, θ_ ). The loss function consists of KL divergence and regularization terms,
the regularization term is the _L_ 2 norm, and it is well known that the _L_ 2 norm is a strong convex
function. When _L_ _[KD]_ is a convex function, it can be proved that _f_ ( _θ_ ) is strongly convex. We use
the _P_ and _Q_ to represent the probability distributions, from KL divergence:


- _P_ ( _x_ _[T]_ ) log( _[P]_ [(] _[x][T]_ [ )]

_Q_ ( _x_ _[T]_ )

_x_ _[T]_


_L_ _[KD]_ = _DKL_ ( _P_ _∥Q_ ) = 


(27)
_Q_ ( _x_ _[T]_ ) [)]


_x_ _[T]_ represents current samples. Assuming _DKL_ ( _P, Q_ ) is a convex function, since KL divergence
does not satisfy triangular symmetry, and we use MRN-Net to help fit the AFT-Net, so let _Q_ be a
fixed term. From the properties of convex functions, we know:


_DKL_ (( _λP_ 1 + (1 _−_ _λ_ ) _P_ 2) _|| Q_ ) _≤_ _λDKL_ ( _P_ 1 _|| Q_ ) + (1 _−_ _λ_ ) _DKL_ ( _P_ 2 _|| Q_ ) (28)


where _λ ∈_ [0 _,_ 1] is the weight factor and _P_ 1, _P_ 2 are arbitrary distributions. If Eq. 28 holds, it can be
proved that _L_ _[KD]_ is a convex function. Let _Pλ_ = _λP_ 1 + (1 _−_ _λ_ ) _P_ 2, expand the left side of Eq. 28
to:


- _Pλ_ ( _x_ _[T]_ ) _·_ log _[P][λ]_ [(] _[x][T]_ [ )]

_Q_ ( _x_ _[T]_ )

_x_ _[T]_


_DKL_ ( _Pλ_ _|| Q_ ) = 


(29)
_Q_ ( _x_ _[T]_ )


For ease of calculation, we use _F_ ( _Pλ_ ) = _Pλ ·_ log _[P]_ _Q_ _[λ]_ [,] _[ Q]_ [ is a fixed term, so] _[ F]_ [(] _[P][λ]_ [)][ is a function about]

_Pλ_, its second derivative is:


_F_ _[′]_ ( _Pλ_ ) = log _Pλ_ + 1 _−_ log _Q, F_ _[′′]_ ( _Pλ_ ) = [1] (30)

_Pλ_


_P_ 1 _λ_ [is] [the] [probability] [distribution] [for] _[z][AF T]_ [,] [so] [1] _[/P][λ]_ _[>]_ [0][,] [therefore] _[F][ ′′]_ [(] _[P][λ]_ [)] _[>]_ [0] [and] _[F]_ [(] _[P][λ]_ [)] [is] [a]
convex function. From Jensen’s inequality we know:


_F_ ( _Pλ_ ( _x_ _[T]_ )) = _Pλ_ ( _x_ _[T]_ ) _·_ log _[P][λ]_ [(] _[x][T]_ [ )]

_Q_ ( _x_ _[T]_ )


_≤_ _λP_ 1( _x_ _[T]_ ) log _[P]_ [1][(] _[x][T]_ [ )]


[1][(] _[x][T]_ [ )]

_[P]_ [2][(] _[x][T]_ [ )]
_Q_ ( _x_ _[T]_ ) [+ (1] _[ −]_ _[λ]_ [)] _[P]_ [2][(] _[x][T]_ [ ) log] _Q_ ( _x_ _[T]_ )


_Q_ ( _x_ _[T]_ )


(31)


(32)


The sum of all samples is known:

  - _[T]_ _[P]_


- _Pλ_ ( _x_ _[T]_ ) _·_ log _[P][λ]_ [(] _[x][T]_ [ )]

_Q_ ( _x_ _[T]_ )

_x_ _[T]_


- _P_ 1( _x_ _[T]_ ) log _[P]_ [1][(] _[x][T]_ [ )]

_Q_ ( _x_ _[T]_ )

_x_ _[T]_


  _Q_ ( _x_ _[T]_ ) _[≤]_ _[λ]_


_Q_ ( _x_ _[T]_ )


- _P_ 2( _x_ _[T]_ ) log _[P]_ [2][(] _[x][T]_ [ )]

_Q_ ( _x_ _[T]_ )

_x_ _[T]_


+ (1 _−_ _λ_ ) 


_Q_ ( _x_ _[T]_ )


= _λDKL_ ( _P_ 1 _|| Q_ ) + (1 _−_ _λ_ ) _DKL_ ( _P_ 2 _|| Q_ )


15


Eq. 28 holds, i.e. _L_ _[KD]_ = _DKL_ ( _P_ _||Q_ ) is a convex function of _P_ . And because _R_ ( _φ, θ_ ) is a
strong convex function, so the loss _f_ ( _θ_ ) of the AFT-Net is a strong convex function. It satisfies
all properties of strong convex functions and provides a guarantee for the proof of sublinear regret
bounds.


A.4 PROOF OF REGRET BOUNDARY


From **Assumption 1**, we know that the gradient of the AFT-Net is bounded, i.e. _gt_ = _|| ∇f_ ( _θ_ ) _||≤_
_l_ . And according to **Assumption** **2**, the diameter of the parameter domain is Γ, so the gradient
boundary of _R_ ( _φ, θ_ ) is:


_β_ 1 is the weight factor of the regularization penalty term in the loss function, and _δ_ is the initial
learning rate adjustment factor, which decreases with time. _regret/T_ is 0 as _T_ approaches infinity,
meaning that our model converges within _T_ steps.


16


_|| ∇R_ ( _φ, θ_ ) _||_ = _[β]_ [1]


[1] _[≤]_ _[β]_ [1][Γ] (33)

2 _[∇||][ φ][ −]_ _[θ][ ||]_ [2]


Then _gt_ = _l_ = _l_ 1 + _β_ 1Γ, _l_ 1 is the boundary of _|| ∇L_ _[KD]_ _||_ . Eq. 9 can be transformed into:


_T_

- _ft_ ( _θ_ ) =


_t_ =1


_T_

- _ft_ ( _θt_ ) _−_


_t_ =1


_T_

- _ft_ ( _θ∗_ )


_t_ =1


(34)


_regret_ =


=


_T_

- _ft_ ( _θt_ ) _−_ min

_θ∈W_
_t_ =1


_T_

- ( _ft_ ( _θt_ ) _−_ _ft_ ( _θ∗_ ))


_t_ =1


According to (Cesa-Bianchi & Lugosi, 2006), we set the learning rate to _ηt_ = 1 _/_ ( _δt_ ), from the
previous section, we can see that _f_ ( _θ_ ) is a strong convex function, according to its nature, it can be
obtained:
_ft_ ( _θt_ ) _−_ _ft_ ( _θ∗_ ) _≤⟨∇ft_ ( _θt_ ) _, θt −_ _θ∗⟩−_ _[δ]_

2 _[||][θ][t][ −]_ _[θ][∗][||]_ [2]


1
_≤_ ( _||θt −_ _θ∗||_ [2] _−||θt_ +1 _−_ _θ∗||_ [2]
2 _ηt_


+ _[η][t]_


2 _[||][ θ][t][ −]_ _[θ][∗]_ _[||]_ [2][)]


_[t]_

2 [(] _[l]_ [1][ +] _[ β]_ [1][Γ)][2] _[ −]_ 2 _[δ]_


When we sum them over the _T_ -round iteration, we get:


_T_


2 _[||][θ]_ [1] _[ −]_ _[θ][∗][||]_ [2]


- ( _ft_ ( _**θ**_ _t_ ) _−_ _ft_ ( _**θ**_ _∗_ )) _≤_ 1 _||θ_ 1 _−_ _θ∗||_ [2] _−_ _[δ]_

2 _η_ 1 2
_t_ =1


1

_−_ _|| θT_ +1 _−_ _θ∗_ _||_ [2]
2 _ηT_


+ [1]

2


_T_


_t_ =2


- 1 1 
_−_ _−_ _δ_ _||θt −_ _θ∗||_ [2]
_ηt_ _ηt−_ 1


+ [(] _[l]_ [1][ +] _**[ β]**_ [1][Γ)][2]

2


_T_

- _**η**_ _t_


_t_ =1


Substituting _ηt_ into Eq. 36 yields:


_T_


_t_ =1


- _ft_ ( _θt, D_ _[T]_ ) _−_ _ft_ ( _θ∗, D_ _[T]_ )� _≤_ [(] _[l]_ [1][ +] _[ β]_ [1][Γ)][2]

2 _δ_


_T_


_t_ =1


1


_t_


_≤_ [(] _[l]_ [1][ +] _[ β]_ [1][Γ)][2] (ln _T_ + 1)

2 _δ_


(35)


(36)


(37)


(38)


Thus, the regret boundary can be expressed as:


_T_

- _ft_ ( _θ_ )


_t_ =1


_regret_ =


_T_

- _ft_ ( _θt_ ) _−_ min

_θ∈W_
_t_ =1


_≤_ [(] _[l]_ [1][ +] _[ β]_ [1][Γ)][2]


_[ β]_ [1][Γ)][2]

(ln _T_ + 1) _≈_ _O_ ( [(] _[l]_ [1][ +] _[ β]_ [1][Γ)][2]
2 _δ_ 2 _δ_


ln _T_ )
2 _δ_


B ADDITIONAL EXPERIMENTAL RESULTS


B.1 EXPERIMENTAL SETTINGS


MetaOCDN is implemented using the deep learning framework PyTorch. The experimental environment is as follows: Intel(R) Xeon(R) Platinum 8468V, 1.0TB memory and NVIDIA H100
graphics card. Furthermore, all of our experiments follow the standard setting of stream data prequential (Brzezinski & Stefanowski, 2014), that is, the data of each batch is first used to test the
model and then to train the model, and each dataset passes through the model only once.


In this paper, ResNet with 12 layers is adopted as baseline, dense blocks are constructed by using
two-layer one-dimensional convolution Conv1d and ReLU, and channel attention and spatial attention modules are added after each dense block to improve the perception ability of the model for
key information. In addition, considering the limitation of memory resources, we set the size of
historical samples to _m_ = 20, which means that the samples of the last 20 batches are stored in the
memory module, the constant offset term of similarity loss _ξ_ is set to 0.001, and the initial value of
the weight factor of regularization penalty term _β_ 1 is 1e-4.


B.2 DATASETS


In order to verify the performance of MetaOCDN under different tasks, we investigated the classical
datasets of concept drift in classification task and regression task, respectively.


**Classification** **Datasets:** We used the data flow generator in the Massive Online Analysis (MOA)
platform (Bifet & Gavalda, 2007) to generate three abrupt, gradual, and incremental concept drift
datasets: _RBFBlips_, _Sea_ and _Hyperplane_ . For convenience of testing, we set the drift sites as 25K,
50K and 75K. Furthermore, we also selected three real datasets: _Kddcup99_, _MIRS_ (Kr¨uger et al.,
2016) and _Yoga_ (Kr¨uger et al., 2016).


**Regression Datasets:** For the regression task, we tested MetaOCDN and other metheds on a series
of time series prediction datasets: _ETTH2_, _ETTm1_ and _WTH_ (Zhou et al., 2023). These datasets are
real datasets, and the details of the datasets are shown in Table B.2.


Table 3: Characteristics of Datasets
**Datasets** **Instances** **Features** **Target variable** **Types** **Number Of drift**


B.3 COMPARISON METHODS


Furthermore, we compare OCF with various methods, including traditional concept drift adaptive
method: DWM (Kolter & Maloof, 2007): Dynamic Weighted Majority (DWM) is an ensemble
method for handling concept drift. It continuously trains online learners, dynamically adjusts their
weights based on performance. OBC (Oza & Russell, 2001): Bagging and boosting are ensemble methods that combine multiple base learners to improve performance. RUS (Wang & Pineau,
2016): RUS combines online ensemble techniques with cost-sensitive strategies from batch learning,
resulting in theoretically sound algorithms with guaranteed convergence under certain conditions.
LEV (Bifet et al., 2010): LEV adapts classical ensemble methods like bagging, boosting, and Random Forests to evolving data streams by introducing additional randomization to inputs and outputs
while preserving bagging’s simplicity. ARF (Gomes et al., 2017): Adaptive Random Forest (ARF)
extends Random Forests to data streams by introducing adaptive mechanisms and resampling strategies to handle concept drift effectively.


17


**Class.**


**Reg.**


_RBFblips_ 100K 20 4 Abrupt 3
_Sea_ 100K 3 2 Gradual 3
_Hyperplane_ 100K 10 2 Incremental _Kddcup99_ 4.94M 23 23 Unknown _MIRS_ 4260 3600 2 Abrupt _Yoga_ 3300 426 2 Unknown 

_ETTH2_ 17420 6 1 Unknown _ETTm1_ 69680 6 1 Unknown _WTH_ 35065 11 1 Unknown 

And some deep neural networks: DNN (Guo et al., 2016): The DNN is the most common network.
ResNet (He et al., 2016): ResNet alleviates the vanishing gradient problem in deep networks by introducing skip connections and allowing cross-layer information transmission. Highway (Srivastava
et al., 2015): Highway networks introduce adaptive gating units to regulate information flow across
many layers, enabling the direct training of extremely deep networks using simple gradient descent.
HBP (Sahoo et al., 2017): Hedge Backpropagation (HBP) for effectively updating DNN parameters
in online learning settings. DenseNet (Huang et al., 2019): DenseNet promotes feature reuse and
alleviates the vanishing gradient problem by connecting the outputs of each layer with those of all
the previous layers.


We have also introduced the latest time series prediction methods: Informer (Zhou et al., 2021): Informer is an efficient Transformer model. By introducing the ProbSparse self-attention mechanism,
self-attention distillation and generative decoder, it solves the computational and structural bottleneck problems of Transformer in long sequence time series prediction. ER (Chaudhry et al., 2019):
ER stores the previous data in the buffer and interweaves it with newer samples during the learning
period. DER++ (Buzzega et al., 2020): DER++ adds the knowledge distillation strategy on the basis
of ER. FsNet (Pham et al., 2022): FSNet is an online time series prediction framework inspired by
the complementary learning system theory. By introducing layer-by-layer adaptors and associative
memory mechanisms. Time-TCN (Bai et al., 2018): Time-TCN is a convolutional neural network
structure in the time dimension. PatchTST (NIEY et al., 2023): PatchTST is an efficient modeling method for Transformer time series. It is independently designed by using time series slices as
input tokens and channels to improve the prediction of long sequences and the learning effect of
self-supervised representations, while reducing the computational cost of attention.


In all of these methods, the batch size is uniformly set to 100 and the hidden node is 100, using the
ReLU activation function and a fixed learning rate of 0.01.


B.4 EVALUATION INDICATORS


To measure OCF performance on different datasets, we use Average Real Accuracy ( _Avgracc_ ) and
Final Cumulative Accuracy ( _Fincacc_ ) on the categorical datasets, and Mean Square Error (MSE) and
Mean Absolute Error (MAE) on the regression datasets, respectively. And on all types of datasets,
we adopted the Bonferroni-Dunn test to compare the differences among different methods. On the
dataset with known drift sites, we used Recovery speed under accuracy ( _RSA_ ) to test the convergence performance of different methods. Since MSE and MAE adopt common settings, we will not
introduce them here. The specific details of the evaluation indicators are as follows:


(1) Average real accuracy ( _Avgracc_ ): The average of the real-time accuracy of the model at each
time step, which reflects the real-time performance of the model:


where _n_ represents the size of samples obtained at each timestamp, _nt_ represents the number of
samples for which the classifier predicts the correct label at the _t_ th timestamp.


18


_Avgracc_ = [1]

_T_


_T_

- _acct_ (39)


_t_ =1


where _acct_ is the real-time accuracy of the _t_ -step time. The real-time accuracy of the model in this
paper is adopted Class Balance Accuracy (CBA).

_acc_ = _CBA_ = Σ _[k]_ _i_ =1 _max_ ( _cciii∗,c∗i_ ) (40)

_k_

where _k_ is the total number of categories, _cii_ is the _i_ th element on the main diagonal of the prediction
result confusion matrix, _ci∗_ and _c∗i_ represent one element in row _i_ and column _i_ . The performance
metric bias caused by class imbalance is mitigated by calculating class balance accuracy.


(2) Final cumulative accuracy ( _Fincacc_ ): The ratio of the number of samples cumulatively predicted
correctly to the number of samples cumulatively acquired up to the current time, which reflects the
population of the model performance:


1
_Fincacc_ =
_T_ _∗_ _n_


_T_

- _nt_ (41)


_t_ =1


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


(3) Recovery speed under accuracy ( _RSA_ ): An online learning model with good convergence can
not only converge to the stable state of the new distribution in a short time after concept drift but
also maintain the minimum real-time error during the convergence process. Therefore, the _RSA_ is
defined in the following way to measure the convergence performance of the model:
_RSA_ = _step ∗_ _ϵavg_ (42)

where the _step_ denotes the number of time steps required from the concept drift site to the convergence site, and _ϵavg_ denotes the average real-time error rate of the convergence process. For the
definition of a convergence site, on the one hand, the amplitude of data fluctuation should not be too
large, and at the same time, the randomness of data fluctuation should be considered. Therefore, this
paper adopts the testing results of 20 subsequent reference sites of a certain site to define whether
the site is a convergence site. If the accuracy difference between this site and subsequent reference
sites is less than the given threshold, and the average accuracy of the first and last 10 reference sites
of the reference sites is also less than the threshold, then the site is considered the convergence site.
_∀i, i ∈{_ 1 _, · · ·_ _,_ 20 _}_,


Figure 7: Comparison of _Fincacc_ of different methods on real-world datasets


**Results on regression datasets.** Since traditional classification methods perform poorly on time series regression datasets, we only compare the methods with relatively better performance. The _MAE_
results are shown in Fig. 8. As illustrated in the figure, MetaOCDN achieves strong performance
across all three datasets. This is because MetaOCDN is capable of learning structured knowledge
from historical samples. Time series data often contain global patterns within historical observations, and MetaOCDN leverages the MRN-Net to effectively capture long-term dependencies in the
data, leading to superior results.


B.6 SUPPLEMENTARY RESULTS OF THE ABLATION STUDY


**Gradient-aware Selective Fine-tuning analysis.** From Fig. 9, we observe that on the _Sea_ dataset,
when concept drift occurs, the gradient norms of Residual Block 1 and Residual Block 2 fluctu

19


_< ε_ (43)
������


10


- _acct_ + _j_ _−_ [1]

10

_j_ =1


10


_|acct −_ _acct_ + _i| < ε_ and


������


1

10


20

- _acct_ + _k_


_k_ =1


here, _ε_ is the convergence threshold parameter.


(4) In addition, the critical difference (CD) of all methods was calculated by the Bonferroni-Dunn
test [31] to show the relative performance between the proposed and the comparison method. The
performance of two classifiers is significantly different if the corresponding average rank sum differs
by at least the critical difference:


_CD_ = _qα_


- _k_ ( _k_ + 1)

(44)
6 _N_


where _qα_ is the critical value at significance level _α_ .


B.5 ANALYSIS OF EXPERIMENTAL RESULTS


**Results on classification datasets.** Fig. 7 presents the _Fincacc_ results of all methods on real-world
datasets, showing that MetaOCDN achieves superior predictive accuracy.


**(b)** _**MIRS**_


0 5 10 15 20 25 30 35 40 45
Timestamp


1.0


0.8


0.6


0.4


0.2


0.0


**(a)** _**Kddcup99**_


0 200 400 600 800 1000
Timestamp


0.7


0.6


0.5


0.4


0.3


0.2


0.1


0.0


0.6


0.5


0.4


0.3


0.2


0.1


**(c)** _**Yoga**_


0 5 10 15 20 25 30 35
Timestamp


**1026**

**1027**


**1028**

**1029**

**1030**

**1031**

**1032**

**1033**


**1034**

**1035**

**1036**

**1037**

**1038**

**1039**


**1040**

**1041**

**1042**

**1043**

**1044**


**1045**

**1046**

**1047**

**1048**

**1049**

**1050**


**1051**

**1052**

**1053**

**1054**

**1055**

**1056**


**1057**

**1058**

**1059**

**1060**

**1061**

**1062**


**1063**

**1064**

**1065**

**1066**

**1067**


**1068**

**1069**

**1070**

**1071**

**1072**

**1073**


**1074**

**1075**

**1076**

**1077**

**1078**

**1079**


Figure 9: Gradient dynamics and result analysis


ate most significantly, while those of Residual Block 3 and Residual Block 4 remain nearly zero,
showing almost no impact. This indicates that the first two residual blocks are more sensitive to distributional shifts and primarily contribute to adapting and representing drift patterns. Furthermore,
combining this with the convergence speed results (bottom-left subfigure), we find that freezing
Residual Blocks 1 and 2 leads to a significant decline in convergence speed, with the effect being
particularly pronounced when Residual Block 2 is frozen; in contrast, freezing Residual Block 3
has almost no negative impact on convergence. This phenomenon further validates the critical role
of Residual Blocks 1 and 2 in adapting to concept drift. On the other hand, the top-right subfigure
shows that under the selective fine-tuning strategy, the model achieves accuracy performance (in
terms of both average real-time accuracy and cumulative accuracy) comparable to full fine-tuning,
while significantly reducing parameter overhead. This demonstrates that the strategy achieves a
better trade-off between accuracy and efficiency, thereby enhancing resource utilization and deployment flexibility.


**(a)** _**MIRS**_ **(b)** _**Yoga**_


Figure 10: Convergence Speed and Parameter Scale Comparison


Fig. 10 illustrates the convergence speed and parameter scale of MetaOCDN on the _MIRS_ and _Yoga_
datasets. The experimental results show that the selective fine-tuning strategy helps MetaOCDN
achieve faster convergence while requiring fewer parameters, thereby reducing computational overhead to some extent.


20


**(b)** _**ETTm1**_


2.5


2.0


1.5


1.0


0.5


0.0


0 20 40 60 80 100 120 140 160
Timestamp


**(a)** _**ETTH2**_


0 20 40 60 80 100 120 140 160
Timestamp


0 20 40 60 80 100 120 140 160
Timestamp


2.5


2.0


1.5


1.0


0.5


0.0


4.0


3.5


3.0


2.5


2.0


1.5


1.0


0.5


0.0


Figure 8: Comparison of _MAE_ of different methods on real-world datasets


**(b)** _**Hyperplane**_


**1080**

**1081**


**1082**

**1083**

**1084**

**1085**

**1086**

**1087**


**1088**

**1089**

**1090**

**1091**

**1092**

**1093**


**1094**

**1095**

**1096**

**1097**

**1098**


**1099**

**1100**

**1101**

**1102**

**1103**

**1104**


**1105**

**1106**

**1107**

**1108**

**1109**

**1110**


**1111**

**1112**

**1113**

**1114**

**1115**

**1116**


**1117**

**1118**

**1119**

**1120**

**1121**


**1122**

**1123**

**1124**

**1125**

**1126**

**1127**


**1128**

**1129**

**1130**

**1131**

**1132**

**1133**


**Robustness** **Analysis** **of** **MRN-Net.** Specifically, we selected three datasets with clearly defined
drift points and compared the performance of MetaOCDN with and without MRN-Net assistance
after concept drift occurred. The evaluation metrics include _RSA_ (Recovery Speed after Adaptation),
which measures the model’s real-time convergence ability during drift, and _DCE_ (Drift Cumulative
Error), which quantifies the accumulated error during the drift adaptation phase. The experimental
results are shown in Fig. 11.


Figure 11: Comparison of _MAE_ of different methods on real-world datasets


As shown in the figures, on the _RBFBlips_, _Sea_, and _Hyperplane_ datasets with known drift points,
MetaOCDN with the collaboration of AFT-Net and MRN-Net exhibits significantly smaller overall
accuracy fluctuations compared to MetaOCDN relying solely on AFT-Net. When concept drift occurs, MRN-Net provides more stable initialization or adjustment signals for the online adaptation
process, enabling the model to converge more rapidly to the new data distribution while substantially reducing error accumulation during the drift adaptation phase. Furthermore, this mechanism
not only enhances the model’s dynamic responsiveness and error suppression ability but also demonstrates consistent and notable advantages in stability and adaptability across multiple non-stationary
environments, thereby validating the critical role of the MRN-Net in strengthening model robustness.


B.7 THE USE OF LLMS


No large language models were used in the experiments or in writing this paper.


21


**(a)** _**Sea**_