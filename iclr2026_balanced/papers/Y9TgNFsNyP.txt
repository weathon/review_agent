### FF-ERASE : MACHINE UNLEARNING AND VERIFICA#### TION FOR FORWARD-FORWARD MODELS


**Anonymous authors**
Paper under double-blind review


ABSTRACT


The Forward-Forward (FF) algorithms present promising and biologically plausible alternatives to backpropagation (BP), enabling efficient model training through
layer-wise greedy optimization. However, the critical task of machine unlearning
for FF models, which involves efficiently removing specific training data’s influence without full retraining, remains a foundational yet unexplored problem. The
inherent characteristics of FF models, such as their sensitivity to parameter tuning
and layer-wise independent training, pose unique challenges, often causing catastrophic model collapse when applying conventional unlearning methods. To fill
this gap, we introduce a novel unlearning framework specifically for FF models,
which employs a goodness-guided strategy. This method proposes a stable guidance model to generate target goodness distributions, steering the original model
to unlearn forgetting data by shifting its layer-wise goodness scores, thereby effectively adapting gradient-based unlearning for the FF architecture. To enable
robust verification on unlearning performance, we also propose a novel goodnessbased membership inference attack (G-MIA), a powerful and lightweight blackbox attack that leverages the unique properties of FF models’ goodness scores.
Our experiments demonstrate that our proposed method effectively removes the
influence of target forgetting data on FF models while preserving model utility on
the remaining data. Critically, our approach accomplishes 1.9 to 3.1 _×_ faster than
retraining from scratch, establishing an efficient foundation for FF unlearning.


1 INTRODUCTION


The Forward-Forward (FF) Hinton (2022) algorithms have emerged as a promising alternative to
backpropagation (BP) for training deep learning models. This approach updates model parameters
by greedily optimizing a layer-wise “goodness” score, which reflects the activation level of neurons
in a layer. By maximizing this score for positive data ( _i.e._, valid training data with correct labels)
and minimizing it for negative data ( _e.g._, invalid data or incorrectly labeled data) during forwarding,
the FF algorithms effectively train model parameters without requiring a backward pass that blocks
all layers. This BP-free nature is considered more biologically plausible and brings significant practical advantages, including reduced memory overhead from stored activations and the potential for
efficient training using pipeline parallelism. These features make FF particularly well-suited for
training on resource-constrained scenarios, such as in edge computing.


However, the critical task of machine unlearning for FF models remains a foundational yet unexplored problem. Usually, machine learning applications involve analyzing sensitive individuals’
data. Their owners require the “right to be forgotten” (RTBF), which has been explicitly stated in the
European Union General Data Protection Regulation (GDPR)Voigt & Von dem Bussche (2017) and
the California Consumer Privacy Act (CCPA)Harding et al. (2019). Moreover, the model owners
also need to remove outdated or poisoned data to promote model performanceWang et al. (2025b);
Zhang et al. (2023). Machine unlearning achieves these data erasing goals by removing the influence of specific training samples from a trained model ( _i.e._, effectiveness) while preserving the
model performance on the remaining data ( _i.e._, model utility).


Existing machine unlearning methods are not feasible for FF models. The most straightforward approach, retraining the model from scratch on the remaining data, is computationally prohibitive and
impractical. Other unlearning methods calibrate the model parameters by either directly performing
gradient ascent (GA) on the forgetting data Tarun et al. (2023a); Sekhari et al. (2021a) or estimating
the parameters tuning Qiao et al. (2024); Liu et al. (2022b). As illustrated in Figure 1, they are also
not applicable due to the unique challenges posed by the BP-free nature and layer-wise training of
FF models. The specific details are as follows.


1


and unlearn forgetting data on BP models (a). However, those methods result in model collapse and
fail in FF models due to unique challenges as shown in illustration (b).


Firstly, FF models exhibit heightened sensitivity to parameter tuning due to their BP-free nature. BP
methods utilize backpropagation to ensure consistent parameter update directions, thereby enhancing robustness to tuning variations. In contrast, FF algorithms use greedy and layer-wise training
approaches, where each layer is independently optimized on its local goodness objective until the
overall goodness scores converge to a specific distribution. In this process, the parameters in the
previous layers do not strictly update towards a consistent direction with the subsequent layers, nor
compress everything “useful” for the final output layer. Therefore, without careful design to prevent goodness from shifting to invalid distributions, layers may diverge in update directions during
unlearning, risking model collapse. However, determining the validity of a goodness distribution in
advance remains challenging, making it difficult to reliably guide layer updates during unlearning.


Secondly, the independent layer-wise training of FF models further complicates the unlearning process. In BP models, a common unlearning strategy is to perform gradient ascent on the loss function
of the data to be removed, updating all layers jointly through the chain rule Gupta et al. (2021);
Tarun et al. (2023b); Sekhari et al. (2021b); Chundawat et al. (2023b). In contrast, FF models optimize separate objective functions at each layer, with varying degrees of goodness improvement.
This independence creates a key difficulty: it is unclear how much each layer’s goodness should be
penalized given a forgetting data sample. As a result, some layers may continue to over-forget while
others only partially retain residual effects, thereby complicating the trade-off between effective
unlearning and preserving the overall model utility.


The above discussion motivates us to answer the first key question: _How_ _to_ _design_ _an_ _efficient_
_machine unlearning method for FF models to ensure both effectiveness and model utility?_


Moreover, it is also challenging to verify the effectiveness of an unlearning algorithm on FF models,
especially for the data owners who do not have full access to the models. Membership inference
attacks (MIAs) Shokri et al. (2017) have been widely adopted as an empirical verification method for
machine unlearning Gao et al. (2024), since other methods either sacrifice the model utility Sommer
et al. (2022); Guo et al. (2023); Han et al. (2025) or necessitate full access Jagielski et al. (2022).
However, current white-box MIAs are impractical for FF unlearning, as the data owners may not
have full access to model parameters and gradients. Our experiments find that the existing blackbox attacks are not accurate enough for FF models. Their effectiveness is often compromised by
standard regularization techniques (e.g., dropout, batch normalization), which inherently decrease
the attack success rate. This leads to the second key question in this paper: _How_ _to_ _design_ _an_
_accurate and practical verification method for FF unlearning algorithms?_


To address these challenges, we make the following contributions:


 - _Problem Identification_ : To the best of our knowledge, we are the first to formalize the problem
and identify the unique challenges of machine unlearning for FF models. Direct gradient ascent
induces optimization instability and frequent model collapse due to the sensitivity of FF models
to parameter tuning. Layer-wise independent training further complicates the effectiveness-utility
trade-off during unlearning.


2


- _Novel FF unlearning Framework_ : We propose FF-Erase, the first unlearning framework specific
to FF models. It introduces a novel goodness-guided approach where a dedicated guidance model
directs layer-wise updates. We also propose two practical strategies to efficiently generate this
guidance model, mini-retraining and fast-distillation, for a large amount and a small amount of
remaining data, respectively.

 - _Accurate_ _Black-Box_ _Unlearning_ _Verification_ : We propose a new black-box verification method
for FF models, the goodness-based MIA (G-MIA). G-MIA leverages the unique properties of the
FF models’ goodness scores to achieve superior accuracy, providing a reliable tool for unlearning
verification. We empirically demonstrate that G-MIA is effective when other black-box attacks
fail with regulation techniques applied and even matches the performance of white-box attacks
with deep networks and complex datasets.

 - _Extensive Evaluation_ : We demonstrate through extensive experiments that our method effectively
unlearns target data while preserving model utility. FF-Erase achieves unlearning 1.9-3.1 _×_ faster
than retraining from scratch, with only a minor 1.6-3.3% degradation in accuracy.

2 RELATED WORK


**Forward-Forward Algorithm:** The Forward-Forward algorithm (FF) Hinton (2022) was recently
proposed as a novel training method to solve the bio-implausibility problem of backpropagation
(BP) Rumelhart et al. (1986) methods, which are the dominant training methods for deep learning
models. By eliminating the backward pass, FF models avoid storing intermediate activations and
allow layers to process the next data batch immediately, thereby reducing memory consumption
and enabling efficient pipeline parallelism. Therefore, numerous works have recently explored different FF algorithms. Initial efforts, such as Symba and Deeperforward Lee & Song (2023); Sun
et al. (2025), focused on refining the core goodness function to support deeper networks and faster
convergence. Building on these foundational improvements, subsequent work has expanded the FF
training methods to more complex domains like convolutional (CwComp Papachristodoulou et al.
(2024)), recurrent (FF-LSTM Gautham et al. (2024)), and graph-based (FORWARDGNN Park et al.
(2024)) neural networks. As these FF algorithms investigate more complex tasks and architectures,
the computational cost of retraining from scratch becomes increasingly prohibitive, creating an urgent need for efficient FF unlearning methods.


**Machine** **Unlearning:** Machine unlearning aims to remove the data impact of specific training
samples from a trained model, while being efficient and preserving the utility of the unlearned
model. Retraining the model from scratch is the gold standard for effectiveness and model utility,
but it lacks efficiency. Existing works can be categorized into two types: exact and approximate
unlearning. Exact unlearning methods seek to produce a model identical to the retrained model.
However, current approaches are incompatible with general FF models, as they either rely on specific
sharded architectures Bourtoule et al. (2021); Tao et al. (2024) or are restricted to linear models Guo
et al. (2020). Approximate unlearning methods tune the model parameters to achieve fast forgetting.
The dominant approaches perform gradient ascent (GA) on the forgetting data Tarun et al. (2023a);
Sekhari et al. (2021a), while Qiao et al. (2024); Liu et al. (2022b); Wu et al. (2023b) refine this
process by using techniques such as influence functions and Hessian matrix to estimate the parameter
calibration. However, as discussed in §1 and Appendix §A, these methods were designed for BPbased models and are not suited for FF models due to their sensitivity to parameter tuning and risk of
optimization instability. This leaves a clear gap for developing unlearning methods for FF models.


**Membership Inference Attacks:** Membership inference attacks (MIAs) Shokri et al. (2017); Nasr
et al. (2019); Melis et al. (2019) are an empirical method for verifying the effectiveness of machine
unlearning, particularly for complex, non-convex models Tu et al. (2024). The goal of an MIA is
to determine if a given sample was in a model’s training set. If an unlearning method is effective,
MIAs should not successfully inference the forgetting samples as members. The more accurate an
MIA is, the more reliable it is as a verification metric. MIAs are classified by their required level
of access. White-box MIAs Wu et al. (2023a); Hamidouche et al. (2022) assume full access to
model parameters and gradients, making them powerful but impractical for real-world verification,
where data owners typically lack such privileged access or hardware resources for running full models. Black-box MIAs Liu et al. (2023); Cifuentes et al. (2021), which only use the model’s final
prediction output, are more practical but less accurate as a reliable verification metric. To fill this
gap, we propose the Goodness-based MIA (G-MIA), a novel attack that leverages the unique layerwise goodness scores of FF models. G-MIA achieves superior accuracy under a strict black-box
constraint, being accurate and practical for verification.


3


3 PRELIMINARIES


In this section, we begin by reviewing the training and inference process of FF models in §3.1, and
then formalize the machine unlearning problem and its notation in §3.2.


3.1 FORWARD-FORWARD TRAINING ALGORITHMS


**Data Forwarding and Goodness Calculation:** Consider a neural network model with _L_ layers for
a _J_ -class classification task. The objective of FF training is to optimize each layer _l_ ’s parameters
_θ_ _[l]_, so that every layer’s goodness can better predict the correct class label _y_ for given input _**x**_ .
Specifically, the function _f_ _[l]_ for each layer _l_ first computes its output _**h**_ _[l]_ using its input _**z**_ _[l][−]_ [1] from
layer _l_ _−_ 1. Then it computes the goodness vector _**g**_ _[l]_ based on _**h**_ _[l]_, which reflects the activation
degree of the neurons in a layer and is the key design for FF training and inference. After that, the
layer simultaneously updates its parameters _θ_ _[l]_ and forward _**z**_ _[l]_, which is the normalization of _**h**_ _[l]_ [1], to
the next layer. Specially, the raw input _**x**_ is considered as _**z**_ [0] . This process is formalized as follows:


As this is a layer-wise loss function, FF training optimizes each layer’s parameters independently:


_∀l,_ _θ_ _[l]_ _←_ _θ_ _[l]_ _−_ _η∇θl_ _L_ ff ( _**g**_ _[l]_ ( _**x**_ _, y_ ; _θ_ _[l]_ )) _,_ (3)


where _η_ is the learning rate. When optimizing _L_ ff ( _·_ ), the distribution of layers’ goodness vectors is
shifting towards a direction where the goodness score of the correct class _gy_ is significantly higher
than others. For example, after training on data sample ( _**x**_ _, y_ ), the goodness distribution moves
towards _**g**_ = [ _g_ 1 _[∼][, g]_ 2 _[∼][, . . ., g]_ _y_ _[↑][, . . ., g]_ _J_ _[∼]_ []][, where the uparrow] _[↑]_ [indicates significant increase and waves]
_∼_ indicate moderate adjustments. As the average of goodness scores usually increases during FF
training, we call this distribution shifting on goodness vectors as goodness increase for brevity.


**Model** **Inference:** FF models output the goodness vectors from all layers _**g**_ [1] _,_ _**g**_ [2] _, . . .,_ _**g**_ _[L]_ for inference. It is common to take a fully-connected layer on them as the predictor. We employ this
predictor as our default setting in experiments due to its superior performance. We provide more
details of the above FF training process using an illustration in Figure 2(a) for better understanding.


3.2 MACHINE UNLEARNING NOTATIONS


The purpose of a machine unlearning process is to remove the influence of forgetting data Dforget
from an original model _θo_ (the model to unlearn) while maintaining the utility of unlearned model
_θu_ on the remaining data Dremain = Dtrain _\_ Dforget, where Dtrain is the training dataset of _θo_ .
Specifically, we denote the model retrained on Dremain as _θr_ . This objective can be formalized as:


min (4)
_θ_ _[u]_ _∈_ Θ _[L]_ [(] _[θ][u]_ [;][ D][forget][)] _[ −]_ _[λ][L]_ [(] _[θ][u]_ [;][ D][remain][)] _[,]_


where _λ_ is a hyper-parameter to balance the trade-off between effectiveness, _i.e._, loss value on
forgetting data _L_ ( _θu_ ; Dforget) and model utility, _i.e._, loss value on remaining data _L_ ( _θu_ ; Dremain).


1It is noted that _**h**_ _l_ = [ _**h**_ _l_ 1 _[,]_ _**[ h]**_ _[l]_ 2 _[, . . .,]_ _**[ h]**_ _[l]_ _J_ []][ is a vector of vector, where each element] _**[ h]**_ 1 _[l]_ [presents for the output]
vector of one class. The _**h**_ _[l]_ is also denoted by the alias _**H**_ _[l]_ _∈_ R _[J][×][d][l][J]_, where _d_ _[l]_ is the dimension of _l_ -th layer’s
output. The _**g**_ _[l]_ is calculated by the _column-wise L1 norm_ of _**h**_ _[l]_ .


4


_**z**_ [0] = _**x**_ _,_ _∀l ∈{_ 1 _,_ 2 _, . . ., L},_ _**h**_ _[l]_ = _f_ _[l]_ ( _**z**_ _[l][−]_ [1] ; _θ_ _[l]_ ) _,_ _**g**_ _[l]_ = _∥_ _**h**_ _[l]_ _∥_ 1 _,_ _**z**_ _[l]_ = ~~_√_~~ _**[h]**_ _[l][ −]_ _**[g]**_ _[l]_


~~_√_~~


_,_ (1)
_σ_ [2] + _ϵ_


where _σ_ [2] are the variance of _**h**_ _[l]_ for layer normalization, and _ϵ_ is a small constant to avoid dividing
by zero. The goodness vector _**g**_ _[l]_ = [ _g_ 1 _[l]_ _[, g]_ 2 _[l]_ _[, . . ., g]_ _J_ _[l]_ []][ contains] _[ J]_ [scores for each class, respectively.]

**Loss Function and Optimization:** FF training aims to increase the goodness score _gy_ _[l]_ [of the correct]
class _y_ while suppressing the other goodness scores _gj,j_ _[l]_ = _y_ [.] [The loss function] _[ L]_ [ff] [is formalized as:]


_∀l ∈{_ 1 _,_ 2 _, . . ., L},_ _L_ ff ( _**g**_ _[l]_ ( _**x**_ _, y_ ; _θ_ _[l]_ )) = _−_ log


- exp - _gy_ _[l]_ 
 - _Jj_ =1 [exp] - _gj_ _[l]_ 


_._ (2)


|Col1|Predictor<br>1<br>3|
|---|---|
|**1**<br>**2**|**1**<br>**2**|


|2<br>1|Col2|Col3|Col4|
|---|---|---|---|
|**2**<br>**1**|**2**<br>||**2**<br>|


_∀_ ( _**x**_ _, y_ ) _∈_ Dforget _, ∀l ∈_ 1 _,_ 2 _, . . ., L,_ _θ_ _[l]_ _←_ _θ_ _[l]_ _−_ _η∇θl_ _D_ KL( _**g**_ _[l]_ ( _**x**_ _, y_ ; _θ_ _[l]_ ) _∥_ _**g**_ _∗_ _[l]_ [(] _**[x]**_ _[, y]_ [;] _[ θ]_ _g_ _[l]_ [))] _[,]_ (5)


2 _K_ is an empirical hyper-parameter for model utility maintenance determined by the dataset. A smaller _K_
indicates more frequent recovering forwards, which usually leads to better model utility and worse efficiency.


5


(a) FF training process


(b) FF-Erase unlearning process


Figure 2: Illustrations for FF learning (a) and FF-Erase unlearning (b). We elaborate the layer-wise
training at the lower left corner and illustrate the multi-class goodness design at the lower right
corner. For example, when training on images of number 2, the corresponding goodness score _g_ 2 [3]
increases while others are suppressed. We also describe each step of unlearning at the upper corner.


4 METHODOLOGY


In this section, we first introduce the workflow of our proposed FF-Erase unlearning algorithm in
§4.1. Then we present two practical strategies to efficiently acquire the guidance model required for
performing FF-Erase unlearning in §4.2. Finally, we discuss the efficiency of FF-Erase in §4.3.


4.1 FAST FORWARD-FORWARD UNLEARNING


The key idea of FF-Erase unlearning is to decrease the goodness score on the forgetting
data while maintaining the goodness score
on the remaining data. The goodness decrease is the opposite process of learning, _i.e._,
_**g**_ = [ _g_ 1 _[∼][, g]_ 2 _[∼][, . . ., g]_ _y_ _[↓][, . . ., g]_ _J_ _[∼]_ []] [for] [forgetting]
data sample ( _**x**_ _, y_ ), which is named as “ _for-_
_getting_ _forward_ ”. To address the instability
challenge during parameter tuning, we decrease the goodness under the guidance goodness _**g**_ _∗_ from a guidance model _θg_, which
is ignorant of the forgetting data but has the
same architecture as the original model. Besides, we also run “ _recovering_ _forward_ ” to
maintain the goodness score on the remaining data by repeating the learning process every _K_ epochs. [2] The overall workflow of FFErase unlearning is summarized as follows.


**Forgetting** **Forward** : 1) Every epoch, we
forward the forgetting data samples through
the original model and collect the goodness
vector _**g**_ ( _**x**_ ; _θ_ ); 2) we forward the same forgetting data samples through the guidance
model to acquire the guidance goodness vector _**g**_ _∗_ ( _**x**_ ; _θg_ ); 3) we decrease the goodness of
forgetting data on the original model by minimizing the KL-loss between them:


**Algorithm 1** FF-Erase Unlearning Algorithm
**Input** : Models _θo_ and _θg_, epoch _E_, thresholds _ϵ_ 1
and _ϵ_ 2, datasets Dforget and Dremain.
**Parameter** : FF model depth _L_, learning rate _η_, recovery step _K_, hyper-parameter _λ_ .
**Output** : Unlearned model _θu_ .

1: **for** _e_ **=** 1 _,_ 2 _, . . ., E_ :
2: **for** _**x**_ **in** Dforget:
3: _ℓ_ 1= **FFwd** ( _**x**_, _θo_, _θg_ ) // _forgetting forward_
4: **for** ( _**x**_ _, y_ ) **in** Dremain, **if** _e_ % _K_ == 0:
5: _ℓ_ 2= **RFwd** ( _**x**_, _y_, _θo_ ) // _recovering forward_
6: **if** _ℓ_ 1 _< ϵ_ 1 **or** _ℓ_ 2 _> ϵ_ 2: **break**
**Return** : _θu_ = _θo_
**FFwd** ( _**z**_ [0] = _**x**_, _θo_, _θg_ ):

1: **for** _l_ **=** 1, 2, _. . ._, _L_ :
2: _**h**_ _[l]_ = _f_ _[l]_ ( _**z**_ _[l][−]_ [1] ; _θo_ _[l]_ [),] _**[ h]**_ _[l]_ _g_ [=] _[f][ l]_ [(] _**[z]**_ _g_ _[l][−]_ [1] ; _θg_ _[l]_ [)]
3: _**z**_ _[l]_ = **LayerNorm** ( _**h**_ _[l]_ ), _**z**_ _g_ _[l]_ [=] **[LayerNorm]** [(] _**[h]**_ _[l]_ _g_ [)]
4: _**g**_ _[l]_ = **Norm** ( _**h**_ _[l]_ ), _**g**_ _∗_ _[l]_ [=] **[Norm]** [(] _**[h]**_ _[l]_ _g_ [)]
5: _ℓ_ 1[ _l_ ] = _∇D_ KL([ _**g**_ _[l]_ ], [ _**g**_ _∗_ _[l]_ []),] _[ θ]_ _o_ _[l]_ [=] _[ θ]_ _o_ _[l]_ _[−]_ _[ηℓ]_ [1][[] _[l]_ []]
6: **return** [�] _l_ _[L]_ =1 _[ℓ]_ [1][[] _[l]_ []]
**RFwd** ( _**z**_ [0] = _**x**_, _y_, _θo_ ):

1: **for** _l_ **=** 1, 2, _. . ._, _L_ :
2: _**h**_ _[l]_ = _f_ _[l]_ ( **LayerNorm** ( _**h**_ _[l][−]_ [1] ); _θo_ _[l]_ [),] _**[g]**_ _[l]_ [=] **[Norm]** [(] _**[h]**_ _[l]_ [)]
3: _ℓ_ 2[ _l_ ] = _∇L_ ff ([ _**g**_ _[l]_ ], _y_ ), _θo_ _[l]_ [=] _[ θ]_ _o_ _[l]_ _[−]_ _[ηλℓ]_ [2][[] _[l]_ []]
4: **return** [�] _l_ _[L]_ =1 _[ℓ]_ [2][[] _[l]_ []]


which leverages a distillation-like manner for moderate parameter tuning during goodness decrease.


**Recovering** **Forward** : 1) Every _K_ epochs, we forward the remaining data samples through the
original model and collect the goodness vector _**g**_ ( _**x**_ ; _θ_ ); 2) we update the parameters layer-wise to
increase the goodness of remaining data. We summarize these two steps as:

_∀_ ( _**x**_ _, y_ ) _∈_ Dremain _, ∀l ∈_ [1 _, L_ ] _,_ _θ_ _[l]_ _←_ _θ_ _[l]_ _−_ _η∇θl_ _λL_ ff ( _**g**_ _[l]_ ( _**x**_ _, y_ ; _θ_ _[l]_ )) (6)


We provide more details to help understand the two forwards with corresponding steps including an
illustration in Figure 2(b) and pseudocode in Algorithm 1. The functions **FFwd** and **RFwd** refer
to the forgetting forward and recovering forward processes, respectively. We use **LayerNorm** and
**Norm** to denote the layer normalization and _L_ 1-norm operation for computing goodness in Equation
(1), respectively. Rather than directly minimizing the goodness score of the correct class, FF-Erase
decreases the goodness by shifting the goodness distribution towards the guidance goodness _**g**_ _∗_
using the Kullback-Leibler divergence for stable and moderate parameter tuning: _D_ KL( _**g**_ _∥_ _**g**_ _∗_ ) =

- _Ji_ =1 _[g]_ [ˆ] _[i]_ [ log (ˆ] _[g][i][/][g]_ [ˆ] _[∗][i]_ [)][, where][ ˆ] _[g][i]_ [= exp] _[ g][i][/]_ [�] _[J]_ _j_ =1 [exp] _[ g][j]_ [is the softmaxed goodness of the] _[ i]_ [-th class.]


**Termination** **Conditions.** The unlearning process in FF-Erase will halt if the model fails to converge after a maximum number of epochs _E_ . Besides, FF-Erase also employs an early stopping
mechanism as commonly used in machine unlearning. Specifically, if the loss value update on the
forgetting data Dforget drops below a threshold _ϵ_ 1 or the loss value on the remaining data Dremain
exceeds a threshold _ϵ_ 2, FF-Erase will terminate unlearning and return the current model as the unlearned model _θu_ .


4.2 TRAINING GUIDANCE MODELS


To ensure both the efficiency and unlearning performance for the FF-Erase algorithm, we require a
stable and accurate guidance model. That is to say, the guidance models need to provide stable guidance goodness distributions and be ignorant of the forgetting data. This is important for stabilizing
the parameter calibration and avoiding model collapse during unlearning. Besides, the efficiency of
generating the guidance model is also important. To this end, we propose two practical strategies to
efficiently obtain accurate guidance models in different scenarios: mini-retrained and fast-distilled.
Mini-retrained models are faster to obtain. However, when there are not enough remaining samples for retraining, we can still obtain fast-distilled models as slower alternatives, as they can be
generated using fewer data samples.


**Mini-Retrained Strategy.** An ideal guidance model is one retrained from scratch on the remaining
data, which is naturally stable and accurate. However, it is computationally prohibitive. As we do
not demand guidance models’ accuracy on the remaining data, we accelerate this process through
two approximations: retraining _α_ 1 = _|_ Dref _|/|_ Dremain _|∈_ (0 _,_ 1) proportion of the remaining samples
using _α_ 2 _∈_ (0 _,_ 1) proportion of the epochs, where Dref ⫋Dremain is the selected subset:

_θ_ _[g,t]_ _←_ _θ_ _[g,t][−]_ [1] _−_ _η∇θg,t−_ 1 _L_ (Dref ; _θ_ _[g,t][−]_ [1] ) _._ (7)


**Fast-Distilled** **Strategy.** The knowledge distillation Hinton et al. (2015a); Gou et al. (2021) is a
well-known approach to rapidly train a new model using existing models. Here, the original model
_θo_ acts as the “teacher”. The goal is to train a “student” guidance model, _θg_, to mimic the teacher’s
output on the remaining data. We use a simplified objective for fast distillation as follows:

_θ_ _[g,t]_ _←_ _θ_ _[g,t][−]_ [1] _−_ _η∇θg,t−_ 1 _D_ KL(Dref ; _θ_ _[g,t][−]_ [1] _∥θo_ ) _._ (8)

This strategy can also be accelerated using _α_ 1 and _α_ 2 as the mini-retrained strategy does.


4.3 EFFICIENCY OF FF-ERASE


The unlearning time of FF-Erase algorithm _t_ unl contains two parts: the time to obtain the guidance
model _t_ 0 and the time for goodness decrease _t_ 1. When unlearning _β_ = _|_ Dforget _|/|_ Dtrain _|_ _∈_ (0 _,_ 1)
proportion of the training samples, the total time for FF-Erase using mini-retrained strategy is:

_t_ unl = _t_ 0 + _t_ 1 _≈_ _α_ 1 _· α_ 2 _· t_ ret + ( _K_ _[−]_ [1] + _β_ ) _· t_ ret _,_ (9)

where _t_ ret is the time for retraining from scratch. According to the experimental results in §6, we can
achieve satisfactory unlearning performance using guidance models with _α_ 1 = 0 _._ 3 and _α_ 2 = 0 _._ 5,
indicating an acceptable overhead of obtaining the guidance model (about 15% of _t_ ret). Empirically,
_t_ 1 usually takes another 10 to 20% of _t_ ret, leading to an overall _t_ unl of 25 to 35% of _t_ ret for FF-Erase
to achieve effective unlearning. FF-Erase using fast-distilled strategy takes similar time.


6


5 GOODNESS-BASED MEMBERSHIP INFERENCE ATTACK (G-MIA)


In this section, we introduce the workflow of G-MIA and describe how to use G-MIA for quantitative verification of FF unlearning algorithms. We consider that the attacker can synthesize data
that has a similar distribution to the training data, which is a common setting in related works ( _e.g._,
Shokri et al. (2017); Liu et al. (2022a); Nasr et al. (2019)) and can be realized by model inversion
techniques Fredrikson et al. (2015). It is also noted that the attacker can obtain the output of the
target model of attack, _i.e._, the goodness vectors from all layers. With the above information, a
complete G-MIA contains the following four steps:


1) **Shadow Model Training.** The attackers first generate a synthetic dataset Dsyn and trains shadow
models _θ_ shadow on it. They also generate another separate synthetic dataset D _[′]_ syn [for testing.]
2) **Goodness Feature Extraction.** The attacker collects the goodness vectors from all layers when
member data (Dsyn) and non-member data (D _[′]_ syn [) of] _[ θ]_ [shadow] [forward the network.]
3) **Attack** **Model** **Training.** The attacker uses the collected goodness vectors to train a binary
classifier _f_ G _−_ MIA( _·_ ) that predicts whether a given sample is a member or non-member:


�1 _,_ _member_
_f_ G _−_ MIA( _**g**_ [1] _,_ _**g**_ [2] _, . . .,_ _**g**_ _[L]_ ) = (10)
0 _,_ _non-member_


4) **Membership Inference.** Given a specific data _**d**_, the attacker first forwards _**d**_ on the model under
attack and obtains the goodness vectors, then predicts its membership by _f_ G _−_ MIA( _·_ ).


**G-MIA** **Verification** . We quantify the unlearning using the attack accuracy (ACC) and the area
under the curve (AUC). A lower ACC or AUC score indicates fewer forgetting samples are identified
as members, implying the unlearning is more effective. We provide more details in Appendix B.1.


6 EXPERIMENTS


In this section, we first present the effects of G-MIA in §6.1. Then we show the experimental
results of FF-Erase unlearning regarding efficiency, effectiveness, and model utility in §6.2. In §6.3,
we further explore classical unlearning methods under different parameters to robustly support our
findings in §1. Lastly in §6.4, we present an ablation study to show the necessity and trade-offs
of the guidance models. We evaluate FF unlearning on 4 standard image benchmarks: CIFAR-10,
CIFAR-100 Krizhevsky et al. (2009), MNIST LeCun et al. (2010), and Fashion-MNIST Xiao et al.
(2017), which are consistent with prior work on FF algorithms regarding the dataset complexity.
We test on various FF models, including a 2-layer tiny CNN, AlexNet Krizhevsky et al. (2012), and
VGG Simonyan et al. (2014) using state-of-the-art FF algorithms: CwComp and Deeperforward.


6.1 G-MIA PERFORMANCE


As an effective and reliable verification metric for FF unlearning, G-MIA should be accurate and
present high ACC and AUC scores. To this end, we compare the attack accuracy (ACC) and area
under the curve (AUC) of G-MIA with several state-of-the-art MIAs, including black-box final-layer
MIA (FL) Shokri et al. (2017), white-box MIA using intermediate layer gradient (GR) Nasr et al.
(2019), and white-box MIA using all layer outputs, including global average pooling (GAP) and
statistics (ST). The statistics include mean, variance, maximum, and _L_ 2 norm of all layer outputs.
Our target models have employed basic MIA-defending techniques, including dropout, batch normalization, and weight decay. For each model, we randomly select 5000 pieces of data samples
from the training set and test set, respectively, as the member and non-member data. The attack
model for every type of MIAs is a standard multilayer perceptron with six hidden layers.


Our results shown in Figure 3 (using ACC as the metric [3] ) indicate that G-MIA is an accurate and
practical verification metric for FF unlearning. Firstly, G-MIA consistently outperforms the classical
black-box final-layer MIA (FL) on all datasets and models. This indicates that the goodness from
all layers provides more membership information than the final-layer output alone. Moreover, GMIA even presents a better performance than white-box MIAs under deeper models and complex
datasets. For example, G-MIA achieves the best accuracy under VGG13 and CIFAR-100. This is
because deeper models and complex datasets amplify the impact of layer-wise independent training,
making the goodness vectors from all layers more informative for membership inference.

3Due to space limitations, we show the results using AUC in the appendix §B.2.


7


|NIST CIFAR-10 CIFAR-10<br>MNIST|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|||||||||||
|A<br>FL<br>(a) <br> : Acc<br>und t<br>  usin|Ti<br>ur<br>o i<br>  g|G<br>ny<br>ac<br>nd<br>   a|R<br>CN<br>y<br>ic<br>    cir|N<br> of<br>at<br>    cl|GA<br>  d<br>e <br>    e|P<br>  if<br>b<br>     a|fe<br>la<br>     nd|r<br>c<br>|ST<br>  en<br>k-b<br>      the|
|||||||||||
|||||||RE<br>|RE<br>|RE<br>||
|||||||A<br>~~FF-Eras~~<br>FF-Eras|A<br>~~FF-Eras~~<br>FF-Eras|A<br>~~FF-Eras~~<br>FF-Eras|~~e(D)~~<br>e(R)|


|NIST CIFAR-10 CIFAR-100<br>MNIST|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||
|A<br>FL<br>(b) A<br>     various<br> ite-box<br>         of all t|G<br>lex<br>      F<br>  M<br>           yp|R<br>N<br>      F<br>  I<br>           es|e<br>       m<br>  A<br>            u|t<br>       o<br>  s<br>            s|GA<br>       d<br> , <br>            i|P<br>       e<br> re<br>            ng|ls<br>s<br>|. <br>p<br>             a|ST<br> F<br>e<br>              st|or<br>ct<br>              a|
||||||||||||
||||||||||||
||||||||||||
|||||||RE<br>|RE<br>|RE<br>|||
|||||||A<br>~~FF-Eras~~<br>FF-Eras|A<br>~~FF-Eras~~<br>FF-Eras|A<br>~~FF-Eras~~<br>FF-Eras|~~e(D)~~<br>e(R)|~~e(D)~~<br>e(R)|
||||||||||||


6.2 MACHINE UNLEARNING ON FF MODELS


its accuracy on Dtest should remain close to that of the original model.


dataset in the main text and put other results in Appendix §C.

|MNIST CIFAR-10 CIFAR-10<br>FMNIST|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||
|G<br> f<br>W|-M<br> gu<br>e|IA<br> re<br> h|FL<br>(c) V<br>, we u<br> ighlig|FL<br>(c) V<br>, we u<br> ighlig|FL<br>(c) V<br>, we u<br> ighlig|G<br>  s<br> h|GR<br>G<br>  e<br> t t|13<br>   a b<br>he|GA<br>   lu<br> be|P<br>   e<br>s|an<br>t|ST<br>    d r<br> blac|
|RE<br>FF-Erase<br>(D)<br>FF-Erase<br>(R)<br>GA<br>( =10)<br>0<br>2<br>4<br>6<br>0.5320<br>0.5245<br>0.5260<br>0.5520<br>(c) G-MIA score (ACC)<br>   ed lines of FF-Erase prese<br>   the solid lines that follows<br>   sing G-MIA scores.<br>     tility of different unlearni<br>  y sample 20% of the traini<br> the same data distributio<br>     ning algorithms will produ<br>   s accuracy on Dtest. We f<br>     ve information removal. F<br>     e unlearned model, meani<br>gradient ascent (GA). RE <br>odel utility, while GA is<br>rase(D) and FF-Erase(R) <br>    idance models, respective<br>ls trained on the CIFAR-<br>      fective and model utility. F<br>     RE (0.532). It also achiev<br>ime. For model utility, F<br>      87, respectively). Compar<br> RE time) with tradeoffs <br>get, respectively) and mod<br>    odels on FF-Erase in §6.4<br>   to model collapse. We w<br>  3.<br>ODS<br>_,_ 100_,_ 10_−_1_,_ 10_−_2_,_ 10_−_3_,_ 0|||||||||||||
|RE<br>FF-Erase<br>(D)<br>FF-Erase<br>(R)<br>GA<br>( =10)<br>0<br>2<br>4<br>6<br>0.5320<br>0.5245<br>0.5260<br>0.5520<br>(c) G-MIA score (ACC)<br>   ed lines of FF-Erase prese<br>   the solid lines that follows<br>   sing G-MIA scores.<br>     tility of different unlearni<br>  y sample 20% of the traini<br> the same data distributio<br>     ning algorithms will produ<br>   s accuracy on Dtest. We f<br>     ve information removal. F<br>     e unlearned model, meani<br>gradient ascent (GA). RE <br>odel utility, while GA is<br>rase(D) and FF-Erase(R) <br>    idance models, respective<br>ls trained on the CIFAR-<br>      fective and model utility. F<br>     RE (0.532). It also achiev<br>ime. For model utility, F<br>      87, respectively). Compar<br> RE time) with tradeoffs <br>get, respectively) and mod<br>    odels on FF-Erase in §6.4<br>   to model collapse. We w<br>  3.<br>ODS<br>_,_ 100_,_ 10_−_1_,_ 10_−_2_,_ 10_−_3_,_ 0|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|
|RE<br>FF-Erase<br>(D)<br>FF-Erase<br>(R)<br>GA<br>( =10)<br>0<br>2<br>4<br>6<br>0.5320<br>0.5245<br>0.5260<br>0.5520<br>(c) G-MIA score (ACC)<br>   ed lines of FF-Erase prese<br>   the solid lines that follows<br>   sing G-MIA scores.<br>     tility of different unlearni<br>  y sample 20% of the traini<br> the same data distributio<br>     ning algorithms will produ<br>   s accuracy on Dtest. We f<br>     ve information removal. F<br>     e unlearned model, meani<br>gradient ascent (GA). RE <br>odel utility, while GA is<br>rase(D) and FF-Erase(R) <br>    idance models, respective<br>ls trained on the CIFAR-<br>      fective and model utility. F<br>     RE (0.532). It also achiev<br>ime. For model utility, F<br>      87, respectively). Compar<br> RE time) with tradeoffs <br>get, respectively) and mod<br>    odels on FF-Erase in §6.4<br>   to model collapse. We w<br>  3.<br>ODS<br>_,_ 100_,_ 10_−_1_,_ 10_−_2_,_ 10_−_3_,_ 0|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|0.5320<br>0.5245<br>0.5260<br>0.5520|||
|RE<br>FF-Erase<br>(D)<br>FF-Erase<br>(R)<br>GA<br>( =10)<br>0<br>2<br>4<br>6<br>0.5320<br>0.5245<br>0.5260<br>0.5520<br>(c) G-MIA score (ACC)<br>   ed lines of FF-Erase prese<br>   the solid lines that follows<br>   sing G-MIA scores.<br>     tility of different unlearni<br>  y sample 20% of the traini<br> the same data distributio<br>     ning algorithms will produ<br>   s accuracy on Dtest. We f<br>     ve information removal. F<br>     e unlearned model, meani<br>gradient ascent (GA). RE <br>odel utility, while GA is<br>rase(D) and FF-Erase(R) <br>    idance models, respective<br>ls trained on the CIFAR-<br>      fective and model utility. F<br>     RE (0.532). It also achiev<br>ime. For model utility, F<br>      87, respectively). Compar<br> RE time) with tradeoffs <br>get, respectively) and mod<br>    odels on FF-Erase in §6.4<br>   to model collapse. We w<br>  3.<br>ODS<br>_,_ 100_,_ 10_−_1_,_ 10_−_2_,_ 10_−_3_,_ 0|||||||||||||
|RE<br>FF-Erase<br>(D)<br>FF-Erase<br>(R)<br>GA<br>( =10)<br>0<br>2<br>4<br>6<br>0.5320<br>0.5245<br>0.5260<br>0.5520<br>(c) G-MIA score (ACC)<br>   ed lines of FF-Erase prese<br>   the solid lines that follows<br>   sing G-MIA scores.<br>     tility of different unlearni<br>  y sample 20% of the traini<br> the same data distributio<br>     ning algorithms will produ<br>   s accuracy on Dtest. We f<br>     ve information removal. F<br>     e unlearned model, meani<br>gradient ascent (GA). RE <br>odel utility, while GA is<br>rase(D) and FF-Erase(R) <br>    idance models, respective<br>ls trained on the CIFAR-<br>      fective and model utility. F<br>     RE (0.532). It also achiev<br>ime. For model utility, F<br>      87, respectively). Compar<br> RE time) with tradeoffs <br>get, respectively) and mod<br>    odels on FF-Erase in §6.4<br>   to model collapse. We w<br>  3.<br>ODS<br>_,_ 100_,_ 10_−_1_,_ 10_−_2_,_ 10_−_3_,_ 0||RE<br> G-M<br>   lines<br>   soli<br>   g G<br>     ty o<br>   mpl<br> e sa<br>     g alg<br>   ccura<br>      infor<br>     nlear<br>dient<br>el u<br>e(D)<br>    nce<br>rain<br>      tive<br>      (0.<br>e. F<br>      resp<br> tim<br>, res<br>    els o<br>    mod<br>S<br> 0_,_ 1|RE<br> G-M<br>   lines<br>   soli<br>   g G<br>     ty o<br>   mpl<br> e sa<br>     g alg<br>   ccura<br>      infor<br>     nlear<br>dient<br>el u<br>e(D)<br>    nce<br>rain<br>      tive<br>      (0.<br>e. F<br>      resp<br> tim<br>, res<br>    els o<br>    mod<br>S<br> 0_,_ 1|I<br>    o<br>   d<br>    -M<br>      f d<br>   e<br> m<br>      o<br>   c<br>      m<br>     n<br> a<br>til<br> a<br>    m<br>ed<br>      an<br>      53<br>or<br>      e<br>e<br>pe<br>     n<br>    e<br> 0_−_|FF-Erase<br>(D)<br>A sco<br>    f FF<br>    lines<br>    IA s<br>      iffer<br>   20%<br> e dat<br>      rithm<br>   y on <br>      ation<br>     ed m<br>scen<br>ity, <br>nd F<br>    odels<br> on <br>      d mo<br>      2). I<br> mod<br>      ctive<br>) wit<br>ctive<br>     FF-E<br>    l coll<br>1_,_ 10|FF-Erase<br>(D)<br>A sco<br>    f FF<br>    lines<br>    IA s<br>      iffer<br>   20%<br> e dat<br>      rithm<br>   y on <br>      ation<br>     ed m<br>scen<br>ity, <br>nd F<br>    odels<br> on <br>      d mo<br>      2). I<br> mod<br>      ctive<br>) wit<br>ctive<br>     FF-E<br>    l coll<br>1_,_ 10|FF-Erase<br>(D)<br>A sco<br>    f FF<br>    lines<br>    IA s<br>      iffer<br>   20%<br> e dat<br>      rithm<br>   y on <br>      ation<br>     ed m<br>scen<br>ity, <br>nd F<br>    odels<br> on <br>      d mo<br>      2). I<br> mod<br>      ctive<br>) wit<br>ctive<br>     FF-E<br>    l coll<br>1_,_ 10|FF-Erase<br>(R)<br> re (A<br>    -Eras<br>    that f<br>    core<br>      ent u<br>    of th<br>  a dis<br>      s wil<br> Dtest<br>      rem<br>      odel,<br>t (G<br>while <br>F-Er<br>   , res<br>the C<br>       del u<br>t also<br>el ut<br>      ly). C<br>h tra<br>ly) a<br>     rase<br>    apse.<br> _−_2_,_ 1|FF-Erase<br>(R)<br> re (A<br>    -Eras<br>    that f<br>    core<br>      ent u<br>    of th<br>  a dis<br>      s wil<br> Dtest<br>      rem<br>      odel,<br>t (G<br>while <br>F-Er<br>   , res<br>the C<br>       del u<br>t also<br>el ut<br>      ly). C<br>h tra<br>ly) a<br>     rase<br>    apse.<br> _−_2_,_ 1|C<br>    e<br>     o<br>    s.<br>       nl<br>    e<br>  tr<br>      l<br>. <br>      o<br> <br>A)<br> <br>as<br>     pe<br>I<br>       ti<br> a<br>il<br>       o<br>de<br>n<br>      i<br>  <br> 0|GA<br>( =10)<br> C)<br>     prese<br>     llows<br>       earni<br>     traini<br>  ibutio<br>       produ<br> We f<br>      val. F<br>      meani<br>. RE <br>GA is<br>e(R) <br>     ctive<br>FAR-<br>       lity. F<br> chiev<br>ity, F<br>       mpar<br>offs <br>d mod<br>      n §6.4<br>We w<br>_−_3_,_ 0|GA<br>( =10)<br> C)<br>     prese<br>     llows<br>       earni<br>     traini<br>  ibutio<br>       produ<br> We f<br>      val. F<br>      meani<br>. RE <br>GA is<br>e(R) <br>     ctive<br>FAR-<br>       lity. F<br> chiev<br>ity, F<br>       mpar<br>offs <br>d mod<br>      n §6.4<br>We w<br>_−_3_,_ 0|


Equation (4) on the performance of classical unlearning methods using gradient ascent as a representative. Our results on VGG13 models trained on the CIFAR-10 dataset are shown in Figure 5,
indicating that the model will either collapse ( _λ_ = 10 [1] _,_ 10 [0] _,_ 10 _[−]_ [1] ) or cannot unlearn the forgetting


8


0.9


0.8


0.7


0.6


0.9


0.8


0.7


0.6


0.5


0.5


Time (/s)


(a) Time vs. Dforget Accuracy


(b) Time vs. Dtest Accuracy


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|
|---|---|---|---|---|---|---|
|||||~~RE~~<br>=1<br>=1<br>|~~RE~~<br>=1<br>=1<br>|0<br>|
||||||~~=0~~<br>=0<br>=0<br>=0|~~1~~<br>.01<br>.001|


|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||||||~~RE~~<br>=1<br>=1<br>|~~RE~~<br>=1<br>=1<br>|0<br>|
|||||||=0<br>=0<br>=0<br>=0|1<br>.01<br>.001|


|) 80 0.625|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|
|---|---|---|---|---|---|---|---|---|---|---|---|
|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0.598<br>0.608 0.605|0.598<br>0.608 0.605|0.598<br>0.608 0.605|0.598<br>0.608 0.605|0.598<br>0.608 0.605|0.598<br>0.608 0.605|0.598<br>0.608 0.605|0.598<br>0.608 0.605|0.598<br>0.608 0.605|
|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|<br>0.554|<br>0.554|<br>0.554|<br>0.554|<br>0.554|||||
|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0.550 0.552<br>~~0.541~~<br>|0.550 0.552<br>~~0.541~~<br>|0.550 0.552<br>~~0.541~~<br>|0.550 0.552<br>~~0.541~~<br>||||||
|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br>80<br>Acc on D_forget(%)<br>~~RE~~<br>=10<br>=1<br>~~=0.1~~<br>=0.01<br>=0.001<br>=0<br>(a) Time vs. Dforget Accuracy<br>0<br>200<br>400<br>600<br>800<br>1000<br>Time (/s)<br>20<br>40<br>60<br><br>Acc on D_test(%)<br>~~RE~~<br>=10<br>=1<br>=0.1<br>=0.01<br>=0.001<br>=0<br>(b) Time vs. Dtest Accuracy<br>RE<br>=10<br>=1<br>=0.1<br>=<br>0.01<br>=<br>0.001<br>=0<br>0.500<br>0.525<br>0.550<br>0.575<br>0.600<br>0.625<br>G-MIA Accuracy<br>0.550 0.552<br>~~0.541~~<br>0.554<br>0.598<br>0.608 0.605<br>(c) G-MIA score (ACC)<br>Figure 5: GA performance under different_ λ_. Some GA methods(_λ_ = 0_._01_,_ 0_._001_,_ 0) are ineffective.<br>data (_λ_ = 10_−_2_,_ 10_−_3_,_ 0). As shown in Figure 5(a), GA (when _λ_ = 10_−_2_,_ 10_−_3_,_ 0) presents sig-<br>nifcantly high accuracy on forgetting data Dforget (84.43, 83.3, and 83.32, respectively) compared<br>with RE (81.61) and FF-Erase(D) (81.31 in Figure 4(a)), indicating poor unlearning effectiveness.<br>Figure 5(c) shows a more precise result, where GA (_λ_ = 10_−_2_,_ 10_−_3_,_ 0) gets G-MIA scores of 0.6,<br>0.61, and 0.6, respectively, being much higher than RE (0.55). For model utility in Figure 5(b), GA<br>(when_ λ_ = 101_,_ 100_,_ 10_−_1) shows low accuracy Dtest (below 60), failing to preserve model utility.<br>6.4<br>ABLATION STUDY ON GUIDANCE MODELS<br>In this experiment, we explore the effciency-performance trade-off of different guidance models in<br>FF-Erase. While a faster acquisition of guidance model can accelerate the unlearning process, it may<br>sacrifce the unlearning effectiveness and model utility. We test different proportions of selected data<br>_α_1 and of epochs_ α_2 for acquiring guidance models and utilize them for FF-Erase unlearning.|0.550 0.552<br>~~0.541~~<br>|RE<br> G-<br>_,_ 0_._<br>_−_2_,_ <br> , re<br> unl<br> ets<br> util<br>  o pr<br>     ere<br>      nle<br>     opo<br>     F-E|=10<br>MIA<br>001_,_<br> 10_−_<br>  spe<br> ear<br> G-<br> ity i<br>  ese<br>     nt g<br>      arni<br>     rtio<br>     rase|=1<br> sc<br> 0)<br>3_,_ <br>  cti<br> nin<br> MI<br>  n F<br>  rve<br>     uid<br>      ng<br>     ns<br>     u|=0.1<br><br> ore (<br> are<br> 0) p<br>  vely<br> g ef<br> A sc<br>  igu<br>   mo<br>     anc<br>      pro<br>      of s<br>     nlea|=<br>0.01<br>0<br> AC<br> ine<br>res<br>  ) co<br>  fect<br>  ore<br>  re 5<br>   del<br>     e m<br>      ces<br>      elec<br>     rnin|=<br>.001<br> C)<br> ffe<br>ents<br>   m<br>  ive<br>  s o<br>   (b)<br>   uti<br>      od<br>      s, it<br>      ted<br>     g.|=0<br> cti<br> s<br>   par<br>  ne<br>  f 0<br>  , G<br>   lit<br>      els<br>       m<br>       d|=0<br> cti<br> s<br>   par<br>  ne<br>  f 0<br>  , G<br>   lit<br>      els<br>       m<br>       d|
|Method|s<br>Effciency|Effectiveness|Effectiveness|||||U|til|ity||
||_t_unl (/s)_↓t_0 (/s)_↓t_unl_ −t_0 (/s)_↓_|Accf (%)_↓_G-MIA ACC_↓_G|Accf (%)_↓_G-MIA ACC_↓_G|-M|IA|AU|C_↓_|Ac|ct (|%|)_↑_|
|RE|<br>1107<br>0<br>1107|<br>81.61<br>0.551|<br>81.61<br>0.551||0.57|1|||<br>80.|85||
|D-(0.5,0.<br>D-(0.5,0.<br>D-(0.5,0.<br>D-(0.3,0.<br>D-(0.3,0.|5)<br>583.5<br>410.5<br>173<br>2)<br>426.7<br>158.9<br>267.8<br>1)<br>353.7<br>83.7<br>270<br>5)<br>569.6<br>288.5<br>281.1<br>2)<br>391.1<br>111.7<br>279.4|81.58<br>0.556<br>0.577<br>80.76<br>0.561<br>0.583<br>80.48<br>0.587<br>0.612<br>81.12<br>0.568<br>0.595<br>80.73<br>0.571<br>0.606|81.58<br>0.556<br>0.577<br>80.76<br>0.561<br>0.583<br>80.48<br>0.587<br>0.612<br>81.12<br>0.568<br>0.595<br>80.73<br>0.571<br>0.606||.5|7|||8.|4|4|
|R-(0.5,0.<br>R-(0.5,0.<br>R-(0.3,0.<br>R-(0.3,0.|5)<br>518.5<br>155.1<br>363.4<br>2)<br>443.7<br>60.7<br>383<br>5)<br>474.4<br>107.8<br>366.6<br>2)<br>429.6<br>41.8<br>387.6|81.53<br>0.562<br>0.583<br>81.02<br>0.573<br>0.609<br>81.51<br>0.569<br>0.598<br>81.03<br>0.577<br>0.621|81.53<br>0.562<br>0.583<br>81.02<br>0.573<br>0.609<br>81.51<br>0.569<br>0.598<br>81.03<br>0.577<br>0.621|81.53<br>0.562<br>0.583<br>81.02<br>0.573<br>0.609<br>81.51<br>0.569<br>0.598<br>81.03<br>0.577<br>0.621|81.53<br>0.562<br>0.583<br>81.02<br>0.573<br>0.609<br>81.51<br>0.569<br>0.598<br>81.03<br>0.577<br>0.621|81.53<br>0.562<br>0.583<br>81.02<br>0.573<br>0.609<br>81.51<br>0.569<br>0.598<br>81.03<br>0.577<br>0.621|81.53<br>0.562<br>0.583<br>81.02<br>0.573<br>0.609<br>81.51<br>0.569<br>0.598<br>81.03<br>0.577<br>0.621|79.16<br>79.00<br>78.86<br>78.58|79.16<br>79.00<br>78.86<br>78.58|79.16<br>79.00<br>78.86<br>78.58|79.16<br>79.00<br>78.86<br>78.58|
|R.G.M|950.8<br>0<br>950.8|51.18<br>0.553<br>0.575|51.18<br>0.553<br>0.575|51.18<br>0.553<br>0.575|51.18<br>0.553<br>0.575|51.18<br>0.553<br>0.575|51.18<br>0.553<br>0.575|55.53|55.53|55.53|55.53|


Table 1: FF-Erase unlearning using different guidance models. Accf and Acct respectively denote
the accuracy on Dforget and Dtest. D (R) refers to fast-distilled (mini-retrained) strategy, followed by
_α_ 1 and _α_ 2, _e.g._, D-(0.5,0.1) refers to FF-Erase guided by a fast-distilled guidance model on _α_ 1=50%
data for _α_ 2=10% epochs. The _t_ unl is the total unlearning time containing two parts: guidance model
obtaining _t_ 0 (if any) and goodness decrease _t_ unl _−_ _t_ 0. R.G.M in the last line refers to FF-Erase using
randomly initialized guidance model. The _↓_ ( _↑_ ) indicates that a lower (higher) is better.


Firstly, as shown in Table 1, a stable and accurate guidance model is crucial for FF-Erase unlearning.
FF-Erase using a randomly initialized model as guidance model (denoted as R.G.M in the last line)
leads to unacceptable performance degradation: the ACCt drops to 55.53%. Such a guidance model
could not provide stable guidance goodness for goodness decreasing, leading to a situation similar
to the direct gradient ascent (GA) method. Secondly, using more data samples for generating the
guidance model (a larger _α_ 1) leads to better unlearning performance (lower G-MIA ACC, _i.e._, more
effective, and higher ACCt, _i.e._, better model utility) but requires longer unlearning time _t_ unl. Using
more training epochs (a larger _α_ 2) also leads to the same trend. Our ablation study demonstrates
that FF-Erase can flexibly achieve different efficiency-performance trade-offs by choosing different
guidance strategies and hyperparameters, making it adaptable to various application scenarios.


7 CONCLUSION


In this paper, we propose FF-Erase, the first machine unlearning method for FF models. We identify the problem that existing unlearning methods designed for BP-based models are infeasible for
FF models due to the sensitivity of FF models to parameter changes. To address this challenge, we
design FF-Erase, a novel FF-specific gradient ascent method to effectively erase the data impact of


9


forgetting samples. FF-Erase uses a goodness-based regularization to stabilize the parameter calibration and a layer-wise unlearning scheme to promote the unlearning efficiency. Moreover, we
propose two flexible strategies to acquire the guidance model for FF-Erase. Accordingly, we propose G-MIA, a goodness-based membership inference attack, to quantitatively verify the unlearning effectiveness of FF-Erase. Extensive experiments on various datasets and model architectures
demonstrate that FF-Erase is effective and efficient, achieving comparable unlearning effectiveness
as retraining while being 1.9-3.1 _×_ faster.


REFERENCES


Amirhossein Bagheri, Radmehr Karimian, and Gholamali Aminian. _f_ -scrub: Unbounded machine
unlearning via _f_ -divergences. In _ICLR 2025 Workshop on Navigating and Addressing Data Prob-_
_lems for Foundation Models_, 2025.


Lucas Bourtoule, Varun Chandrasekaran, Christopher A Choquette-Choo, Hengrui Jia, Adelin
Travers, Baiwu Zhang, David Lie, and Nicolas Papernot. Machine unlearning. In _2021_ _IEEE_
_symposium on security and privacy (SP)_, pp. 141–159. IEEE, 2021.


Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramer. Membership inference attacks from first principles. In _2022 IEEE symposium on security and privacy_
_(SP)_, pp. 1897–1914. IEEE, 2022.


Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. Can bad teaching
induce forgetting? unlearning in deep networks using an incompetent teacher. In _Proceedings of_
_the AAAI Conference on Artificial Intelligence_, volume 37, pp. 7210–7217, 2023a.


Vikram S Chundawat, Ayush K Tarun, Murari Mandal, and Mohan Kankanhalli. Zero-shot machine
unlearning. _IEEE Transactions on Information Forensics and Security_, 18:2345–2354, 2023b.


Nicol´as Cifuentes, Mingyu Sun, Robin Gupta, and Bikash C Pal. Black-box impedance-based stability assessment of dynamic interactions between converters and grid. _IEEE_ _Transactions_ _on_
_Power Systems_, 37(4):2976–2987, 2021.


Tom Fawcett. An introduction to roc analysis. _Pattern recognition letters_, 27(8):861–874, 2006.


Matt Fredrikson, Somesh Jha, and Thomas Ristenpart. Model inversion attacks that exploit confidence information and basic countermeasures. In _Proceedings of the 22nd ACM SIGSAC confer-_
_ence on computer and communications security_, pp. 1322–1333, 2015.


Xiangshan Gao, Xingjun Ma, Jingyi Wang, Youcheng Sun, Bo Li, Shouling Ji, Peng Cheng, and
Jiming Chen. Verifi: Towards verifiable federated unlearning. _IEEE Transactions on Dependable_
_and Secure Computing_, 21(6):5720–5736, 2024.


Suraj R Gautham, Swapnil Nair, Suresh Jamadagni, Mridul Khurana, and Md Assadi. Exploring the
feasibility of forward forward algorithm in neural networks. In _2024 International Conference on_
_Advances in Modern Age Technologies for Health and Engineering Science (AMATHE)_, pp. 1–6.
IEEE, 2024.


Jianping Gou, Baosheng Yu, Stephen J Maybank, and Dacheng Tao. Knowledge distillation: A
survey. _International journal of computer vision_, 129(6):1789–1819, 2021.


Chuan Guo, Tom Goldstein, Awni Hannun, and Laurens Van Der Maaten. Certified data removal
from machine learning models. In _Proceedings of the 37th International Conference on Machine_
_Learning_, ICML’20. JMLR.org, 2020.


Yu Guo, Yu Zhao, Saihui Hou, Cong Wang, and Xiaohua Jia. Verifying in the dark: Verifiable
machine unlearning by using invisible backdoor triggers. _IEEE_ _Transactions_ _on_ _Information_
_Forensics and Security_, 19:708–721, 2023.


Varun Gupta, Christopher Jung, Seth Neel, Aaron Roth, Saeed Sharifi-Malvajerdi, and Chris Waites.
Adaptive machine unlearning. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_, 34:16319–
16330, 2021.


10


Mounia Hamidouche, Reda Bellafqira, Gwenol´e Quellec, and Gouenou Coatrieux. White-box
membership attack against machine learning based retinopathy classification. _arXiv_ _preprint_
_arXiv:2206.03584_, 2022.


Mengde Han, Tianqing Zhu, Lefeng Zhang, Huan Huo, and Wanlei Zhou. Vertical federated unlearning via backdoor certification. _IEEE Transactions on Services Computing_, 2025.


Elizabeth Liz Harding, Jarno J Vanto, Reece Clark, L Hannah Ji, and Sara C Ainsworth. Understanding the scope and impact of the california consumer privacy act of 2018. _Journal_ _of_ _Data_
_Protection & Privacy_, 2(3):234–253, 2019.


Geoffrey Hinton. The forward-forward algorithm: Some preliminary investigations, 2022.


Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv_
_preprint arXiv:1503.02531_, 2015a.


Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. _arXiv_
_preprint arXiv:1503.02531_, 2015b.


Matthew Jagielski, Om Thakkar, Florian Tramer, Daphne Ippolito, Katherine Lee, Nicholas Carlini,
Eric Wallace, Shuang Song, Abhradeep Thakurta, Nicolas Papernot, et al. Measuring forgetting
of memorized training examples. _arXiv preprint arXiv:2207.00099_, 2022.


Yongwoo Kim, Sungmin Cha, and Donghyun Kim. Are we truly forgetting? a critical reexamination of machine unlearning evaluation protocols. _arXiv preprint arXiv:2503.06991_, 2025.


Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural
network representations revisited. In _International_ _conference_ _on_ _machine_ _learning_, pp. 3519–
3529. PMlR, 2019.


Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images.
2009.


Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. _Advances in neural information processing systems_, 25, 2012.


Meghdad Kurmanji, Peter Triantafillou, Jamie Hayes, and Eleni Triantafillou. Towards unbounded
machine unlearning. _Advances in neural information processing systems_, 36:1957–1987, 2023.


Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. _ATT Labs [Online]._
_Available:_ _http://yann.lecun.com/exdb/mnist_, 2, 2010.


Heung-Chang Lee and Jeonggeun Song. Symba: Symmetric backpropagation-free contrastive learning with forward-forward algorithm for optimizing convergence, 2023.


Gaoyang Liu, Tianlong Xu, Rui Zhang, Zixiong Wang, Chen Wang, and Ling Liu. Gradient-leaks:
Enabling black-box membership inference attacks against machine learning models. _IEEE Trans-_
_actions on Information Forensics and Security_, 19:427–440, 2023.


Lan Liu, Yi Wang, Gaoyang Liu, Kai Peng, and Chen Wang. Membership inference attacks against
machine learning models via prediction sensitivity. _IEEE Transactions on Dependable and Secure_
_Computing_, 20(3):2341–2347, 2022a.


Yi Liu, Lei Xu, Xingliang Yuan, Cong Wang, and Bo Li. The right to be forgotten in federated learning: An efficient realization with rapid retraining. In _IEEE INFOCOM 2022-IEEE conference on_
_computer communications_, pp. 1749–1758. IEEE, 2022b.


Yiyong Liu, Zhengyu Zhao, Michael Backes, and Yang Zhang. Membership inference attacks by
exploiting loss trajectory. In _Proceedings of the 2022 ACM SIGSAC Conference on Computer and_
_Communications Security_, pp. 2085–2098, 2022c.


Luca Melis, Congzheng Song, Emiliano De Cristofaro, and Vitaly Shmatikov. Exploiting unintended
feature leakage in collaborative learning. In _2019 IEEE symposium on security and privacy (SP)_,
pp. 691–706. IEEE, 2019.


11


Milad Nasr, Reza Shokri, and Amir Houmansadr. Comprehensive privacy analysis of deep learning:
Passive and active white-box inference attacks against centralized and federated learning. In _2019_
_IEEE symposium on security and privacy (SP)_, pp. 739–753. IEEE, 2019.


Andreas Papachristodoulou, Christos Kyrkou, Stelios Timotheou, and Theocharis Theocharides.
Convolutional channel-wise competitive learning for the forward-forward algorithm. In _Proceed-_
_ings of the AAAI Conference on Artificial Intelligence_, volume 38, pp. 14536–14544, 2024.


Namyong Park, Xing Wang, Antoine Simoulin, Shuai Yang, Grey Yang, Ryan Rossi, Puja
Trivedi, and Nesreen Ahmed. Forward learning of graph neural networks. _arXiv_ _preprint_
_arXiv:2403.11004_, 2024.


Xinbao Qiao, Meng Zhang, Ming Tang, and Ermin Wei. Hessian-free online certified unlearning.
_arXiv preprint arXiv:2404.01712_, 2024.


David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. Learning representations by backpropagating errors. _nature_, 323(6088):533–536, 1986.


Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what
you want to forget: Algorithms for machine unlearning. _Advances_ _in_ _Neural_ _Information_ _Pro-_
_cessing Systems_, 34:18075–18086, 2021a.


Ayush Sekhari, Jayadev Acharya, Gautam Kamath, and Ananda Theertha Suresh. Remember what
you want to forget: Algorithms for machine unlearning. _Advances_ _in_ _Neural_ _Information_ _Pro-_
_cessing Systems_, 34:18075–18086, 2021b.


Nazanin Mohammadi Sepahvand, Eleni Triantafillou, Hugo Larochelle, Doina Precup, James J
Clark, Daniel M Roy, and Gintare Karolina Dziugaite. Selective unlearning via representation
erasure using domain adversarial training. In _The Thirteenth International Conference on Learn-_
_ing Representations_, 2025.


Haonan Shi, Tu Ouyang, and An Wang. Learning-based difficulty calibration for enhanced membership inference attacks. In _2024 IEEE 9th European Symposium on Security and Privacy (Eu-_
_roS&P)_, pp. 62–77. IEEE, 2024.


Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In _2017 IEEE symposium on security and privacy (SP)_,
pp. 3–18. IEEE, 2017.


Karen Simonyan, Andrew Zisserman, et al. Very deep convolutional networks for large-scale image
recognition. _arXiv preprint arXiv:1409.1556_, 2014.


David M Sommer, Liwei Song, Sameer Wagh, and Prateek Mittal. Athena: Probabilistic verification
of machine unlearning. _Proceedings on Privacy Enhancing Technologies_, 2022.


Liang Sun, Yang Zhang, Weizhao He, Jiajun Wen, Linlin Shen, and Weicheng Xie. Deeperforward: Enhanced forward-forward training for deeper and better performance. In _The Thirteenth_
_International Conference on Learning Representations_, 2025.


Youming Tao, Cheng-Long Wang, Miao Pan, Dongxiao Yu, Xiuzhen Cheng, and Di Wang. Communication efficient and provable federated unlearning. _Proc. VLDB Endow._, 17(5):1119–1131,
January 2024. ISSN 2150-8097. doi: 10.14778/3641204.3641220. [URL https://doi.org/](https://doi.org/10.14778/3641204.3641220)
[10.14778/3641204.3641220.](https://doi.org/10.14778/3641204.3641220)


Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. Fast yet effective machine unlearning. _IEEE Transactions on Neural Networks and Learning Systems_, 35(9):
13046–13055, 2023a.


Ayush K Tarun, Vikram S Chundawat, Murari Mandal, and Mohan Kankanhalli. Fast yet effective machine unlearning. _IEEE Transactions on Neural Networks and Learning Systems_, 35(9):
13046–13055, 2023b.


Yiwen Tu, Pingbang Hu, and Jiaqi Ma. Towards reliable empirical machine unlearning evaluation:
A game-theoretic view, 2024. [URL https://arxiv.org/abs/2404.11577.](https://arxiv.org/abs/2404.11577)


12


Paul Voigt and Axel Von dem Bussche. The eu general data protection regulation (gdpr). _A practical_
_guide, 1st ed., Cham:_ _Springer International Publishing_, 10(3152676):10–5555, 2017.


Jiali Wang, Hongxia Bie, Zhao Jing, and Yichen Zhi. Scrub-and-learn: Category-aware weight
modification for machine unlearning. _AI_, 6(6):108, 2025a.


Tinghua Wang, Dongyan Zhao, and Shengfeng Tian. An overview of kernel alignment and its
applications. _Artificial Intelligence Review_, 43(2):179–192, 2015.


Wenbin Wang, Qiwen Ma, Zifan Zhang, Yuchen Liu, Zhuqing Liu, and Minghong Fang. Poisoning
attacks and defenses to federated unlearning. In _Companion_ _Proceedings_ _of_ _the_ _ACM_ _on_ _Web_
_Conference 2025_, pp. 1365–1369, 2025b.


Chen Wu, Sencun Zhu, and Prasenjit Mitra. Federated unlearning with knowledge distillation. _arXiv_
_preprint arXiv:2201.09441_, 2022.


Di Wu, Saiyu Qi, Yong Qi, Qian Li, Bowen Cai, Qi Guo, and Jingxian Cheng. Understanding and
defending against white-box membership inference attack in deep learning. _Knowledge-Based_
_Systems_, 259:110014, 2023a.


Jiancan Wu, Yi Yang, Yuchun Qian, Yongduo Sui, Xiang Wang, and Xiangnan He. Gif: A general
graph unlearning strategy via influence function. In _Proceedings_ _of_ _the_ _ACM_ _Web_ _Conference_
_2023_, pp. 651–661, 2023b.


Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms. _arXiv preprint arXiv:1708.07747_, 2017.


Xiaoyu Xu, Xiang Yue, Yang Liu, Qingqing Ye, Huadi Zheng, Peizhao Hu, Minxin Du, and Haibo
Hu. Unlearning isn’t deletion: Investigating reversibility of machine unlearning in llms. _arXiv_
_preprint arXiv:2505.16831_, 2025.


Haibo Zhang, Toru Nakamura, Takamasa Isohara, and Kouichi Sakurai. A review on machine
unlearning. _SN Computer Science_, 4(4):337, 2023.


13


A SUPPLEMENTARY ON RELATED WORK


In this appendix section, we specifically discuss more details of related works on conventional approximate unlearning methods for BP models and explain why they are not suitable for FF models
as mentioned in §1 and §2. Moreover, in the main text, we utilize direct gradient ascent (GA) as
the representative of the approximate unlearning methods. Here in this appendix section, we also
explain the rationale of the GA’s representativeness.


**Robust Gradient Ascent:** While directly applying GA by increasing the loss on forgetting data can
be too brute-force and damages the model’s utility, advanced GA methods use different techniques
for realizing a more robust GA, and those methods are proven to be successful in BP models. Tarun
et al. (2023b) introduces an impair-and-repair process, adding gradient descent on remaining data
after all GA epochs to repair the model utility. Gao et al. (2024) balances the utility using a scale-tounlearn technique. Sepahvand et al. (2025) leverages adversarial training to unlearn representations
of forgetting data. However, all those methods face the same failure for FF models as direct GA: the
model collapse takes place immediately, making those repairing techniques ineffective, as experimental results in Appendix §C.3 have demonstrated. This is because all those methods only limit
how much the gradient ascent is; however, the FF models additionally require avoiding invalid goodness distribution, which is not considered in those methods. Therefore, those robust GA methods
still can not achieve effective FF unlearning without model collapse.


**Distillation-based** **Gradient** **Ascent:** The teacher-student approaches, or distillation-based GA
methods Kurmanji et al. (2023); Chundawat et al. (2023a); Bagheri et al. (2025); Wang et al. (2025a);
Wu et al. (2022), are another type of advanced GA methods leveraging knowledge distillation Hinton et al. (2015b) techniques for a gentle gradient ascent. However, these methods are also not
feasible to be directly applied to FF models for two specific reasons. First, they depend on the
final-layer logits for unlearning, which is not effective enough, as FF models also persist knowledge
in their previous layers during their layer-wise greedy learning. Second, literature such as SCRUB
Kurmanji et al. (2023) utilizes the original model as the teacher model and unlearns by increasing
the divergence between the original and unlearned model outputs on forgetting data. Although this
achieves successful unlearning for BP models, it can lead to an away-but-invalid goodness distribution for FF models, risking model collapse. Therefore, those methods still can not achieve effective
FF unlearning without model collapse.


B GOODNESS-BASED MEMBERSHIP INFERENCE ATTACK (G-MIA)


In this appendix section, we will first provide supplementary descriptions of G-MIA verification
as mentioned in §5. Then we will show more experimental results of G-MIA to demonstrate its
accuracy and lightweight as mentioned in §6.1. Finally, we provide the G-MIA score (AUC) of
different unlearning methods under various settings as mentioned in §6.2 and §6.3.


B.1 G-MIA VERIFICATION


Besides attack accuracy (ACC), the area under the receiver operating characteristic curve (AUC) is
also widely used in related work Shokri et al. (2017); Carlini et al. (2022); Liu et al. (2022c); Shi
et al. (2024) to quantify the accuracy of an MIA. AUC describes the probability that a randomly
chosen member receives a higher score than a randomly chosen non-member. Let _f_ MIA( _**x**_ ) _∈_ [0 _,_ 1]
be a continuous MIA prediction score, _i.e._, the attack model outputs a float number from 0 to 1
to measure the likelihood that _**x**_ is a member, rather than directly 0 as a non-member and 1 as a
member. The AUC can be calculated by:


AUC = _P_ [ _f_ MIA( _XM_ ) _> f_ MIA( _XN_ )] + 0 _._ 5 _× P_ [ _f_ MIA( _XM_ ) = _f_ MIA( _XN_ )] _,_ (11)


where _XM_ and _XN_ denote member and non-member data, respectively. The result from the above
definition is equivalent to the area under the receiver operating characteristic (ROC) curve Fawcett
(2006), where ROC plots True Positive Rate (TP) vs. False Positive Rate (FP) as the threshold varies
across all real values. For verification, a lower ACC or AUC score on the forgetting data indicates
that these data are less likely to be detected as members, thereby demonstrating a more effective
unlearning method. Existing literature has rigorously analyzed the relationship between MIA ACC


14


(AUC) scores and unlearning effectiveness Tu et al. (2024), which is orthogonal and can be applied
to our work.


In our experiments, we use both of them to quantify the unlearning. We utilize ACC because it is
more intuitive and easier to understand. However, it is sensitive to the member/non-member ratio
in the evaluation and requires a specific threshold to classify the membership. Therefore, we also
leverage the AUC score, which is invariant to the ratio, thereby being easier to compare across
datasets, splits, or output interfaces. Due to the space limitation, we only give the ACC results in the
main text. Then we will show the corresponding AUC results in §B.2.


B.2 G-MIA PERFORMANCE


Firstly, we give the AUC score of different MIAs on various FF models in Figure 6, which corresponds to the ACC results in Figure 3. The AUC results further demonstrate that G-MIA is
an accurate and practical verification metric for FF unlearning. G-MIA consistently outperforms
the classical black-box final-layer MIA (FL) on all datasets and models. Moreover, G-MIA even
presents a better performance than white-box MIAs under deeper models and complex datasets. For
example, G-MIA achieves the best accuracy under VGG13 and CIFAR-100.


|MNIST CIFAR-10 CIFAR-100<br>FMNIST|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
|G-<br>re<br> an<br> bl|MIA<br> 6: <br> d<br> ac|(<br> <br> re<br> k-|a<br>A<br> <br> b|F<br>)<br>U<br> d<br>|L<br> <br> <br>  b<br> o|T<br>C<br> <br> x|in<br> s<br>  ac<br>|y<br> c<br>  k<br>|G<br><br> <br> <br> I|R<br>C<br> o<br>  g<br> A|N<br> r<br>  r<br>|N<br> e<br>  o<br>  u|G<br><br> of<br>  un<br>  s|AP<br>  d<br>  d<br>  in|i<br>  t<br>  g|f<br>  o<br>|f<br> <br>  a|S<br>  er<br>   i<br>   c|T<br>  e<br>   n<br>   i|n<br>   di<br>   rc|


|MNIST CIFAR-10 CIFAR-100<br>FMNIST|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||||
||||||||||||||||||||||
||||||||||||||||||||||
|G-<br>  s<br>   la<br>   d t|MIA<br>   on<br>   ck<br>    he|v<br>   -b<br>    b|(<br> <br> <br>    e|F<br>b<br>   ar<br>   o<br>    s|L<br>) <br>   i<br>   x<br>    t|<br>   o<br>    a<br>|A<br>   u<br>    n<br>     M|le<br>   s<br> <br>     I|G<br>x<br> <br>    d<br>     A|R<br><br>    F<br> <br>|N<br>    F<br>    w<br>|et<br> <br>    h<br>     of|G<br><br>    m<br>    it<br>      a|AP<br>    o<br>    e-<br>      ll|d<br>    b<br>|e<br> <br>      t|ls<br>    ox<br>      yp|S<br> <br> <br>      e|T<br>     in<br> <br>      s|<br> <br>|


|MNIST CIFAR-10 CIFAR-100<br>FMNIST|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||
||||||||||||||||||||
||||||||||||||||||||
|G-<br>     e 3<br>     esp<br>       a s|MIA<br>      . F<br>     ec<br>        ta|o<br>     t<br>        r.|(<br>r<br>     iv|FL<br>c) <br> e<br>     el|<br> a<br>     y|V<br> c<br>     .|G<br> h<br>|GR<br>G<br> f<br>W|1<br> g<br>e|3<br> u<br> h|G<br> re<br> ig|AP<br>,<br> h|w<br> l|e<br> ig|<br> h|S<br>  u<br> t|T<br>  s<br>|e<br> th|


Besides, we also give a detailed comparison of the input data size required by different types of
MIAs in Table 2, which demonstrates that G-MIA is a lightweight attack. FL, G-MIA, and GD only
require small amounts of input data (10 [1] _∼_ 10 [3] ), while GAP and ST need much larger input data
(10 [3] _∼_ 10 [5] ). The input data size directly determines the size of the attack model, including training
efficiency and memory consumption. Therefore, G-MIA is not only accurate but also lightweight.

|Setting|FL|G-MIA|GAP|ST|GR|
|---|---|---|---|---|---|
|C10/MN/FMN-VGG16<br>CIFAR100-VGG16|10<br>100|170<br>1700|12880<br>18400|38657<br>55217|103<br>103|
|C10/MN/FMN-VGG13<br>CIFAR100-VGG13|10<br>100|140<br>1400|4760<br>6800|14294<br>20414|85<br>85|
|C10/MN/FMN-AlexNet<br>CIFAR100-AlexNet|10<br>100|50<br>500|1300<br>1300|3905<br>3905|31<br>31|
|C10/MN/FMN-TinyCNN<br>CIFAR100-TinyCNN|10<br>100|40<br>400|1100<br>1100|3304<br>3304|25<br>25|


Table 2: Input data size required by different types of MIAs. We use C10/MN/FMN to refer to
CIFAR10, MNIST, or FMNIST for brevity.
B.3 G-MIA AUC SCORE IN EXPERIMENTS


In §6.2 and §6.3, we have shown the G-MIA ACC score of different unlearning methods under
various settings in Figure 4(c) and 5(c), respectively. Here we provide the corresponding AUC score
in Figure 7(a) and (b), respectively. The AUC results are consistent with the ACC results, which
further demonstrate that our proposed method is effective for FF unlearning, and demonstrate that
GA methods (when _λ_ = 0 _._ 01 _,_ 0 _._ 001 _,_ 0) can not effectively unlearn the forgetting data.


C SUPPLEMENTARY EXPERIMENTAL RESULTS


15


0.9


0.8


0.7


0.6


0.9


0.8


0.7


0.6


0.9


0.8


0.7


0.6


0.5


0.5


0.5


.

|0.671 0.671<br>0.652|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
|0.652<br>0.671 0.671|0.652<br>0.671 0.671|0.652<br>0.671 0.671|0.652<br>0.671 0.671|0.652<br>0.671 0.671|0.652<br>0.671 0.671||||
|0.608<br>0.594<br><br>0.597<br>|0.608<br>0.594<br><br>0.597<br>|0.608<br>0.594<br><br>0.597<br>|0.608<br>0.594<br><br>0.597<br>|0.608<br>0.594<br><br>0.597<br>|||||
||||~~.583~~<br>||||||
||||||||||
||||||||||
|e <br> A<br>  t<br> ,<br>  d<br>   e<br>  e<br> d<br> <br>|RE<br> (A<br> UC<br>  hod<br>  we<br>  FF<br>   sult<br>  eff<br> e e<br>  of o<br>|=10<br>UC)<br> sc<br>  wit<br>   pro<br>   mo<br>   s of<br>  ecti<br> xtra<br>  ur p<br>|=1<br>(a)<br>. F<br> ore<br>  h a<br>   vid<br>   de<br>    a<br>  ven<br>  ex<br>   ro<br>|=0.1<br>0<br>igur<br>  of t<br>   low<br>   e m<br>   ls to<br>    new<br>  ess<br>  peri<br>   pose<br>|=<br>.01<br>0<br>e (a<br>  he o<br>   er<br>   ore<br>    de<br>     exp<br>   lay<br>  men<br>   d F<br>|=<br>.001<br>) a<br>   rig<br>    G-<br>    exp<br>    mo<br>     eri<br>   er-<br>  ts<br>   F-E<br>|=0<br>nd<br>   in<br>    MI<br>    er<br>    ns<br>     m<br>   wis<br>  us<br>   ra<br>|<br>   a<br>    A<br>    i<br>    tr<br>     e<br>   e<br>  in<br>   s<br>|


demonstrate the generalizability of our findings.

|0.5740|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
|~~0.5345~~<br><br>|~~0.5345~~<br><br>|~~0.5345~~<br><br>|~~0.5345~~<br><br>|~~0.5345~~<br><br>|~~0.5345~~<br><br>|~~0.5345~~<br><br>|||
|~~0.5309~~<br>~~0.5335~~|~~0.5309~~<br>~~0.5335~~|~~0.5309~~<br>~~0.5335~~|~~0.5309~~<br>~~0.5335~~|~~0.5309~~<br>~~0.5335~~|~~0.5309~~<br>~~0.5335~~|~~0.5309~~<br>~~0.5335~~|||
|~~0.5309~~<br>~~0.5335~~|||||||||
||||||||||
|i<br>    (<br>     c<br>    e<br>     g<br>     h<br>   ti<br>   u<br>ly<br>X<br>n <br>i<br>    a<br>tl<br> <br>  q<br> <br>   e<br>o|RE<br>de th<br>    0.618<br>     ore is<br>    sults.<br>     enera<br>      utili<br>   ng th<br>   nlea<br>, in §<br>PERI<br> CIF<br>ffere<br>    nd F<br>y mo<br>More<br>  uired<br>with <br>   poch<br>n co|e <br>    4<br> <br> <br>     l<br>      z<br>    e<br>   r<br> <br>M<br>A<br>nt<br>     a<br>r<br>o<br> <br> t<br>   s<br>m|FF-Erase<br>(D)<br>(b)<br> GIA<br>    ) in<br>      regar<br>     Firstl<br>     izabi<br>      es ce<br>    effe<br>   ning<br> C.4,<br>ENT<br>R-10<br> data<br>     shion<br>e eff<br>ver, <br>   unle<br>he m<br>    on si<br>plex|<br>     F<br>      d<br>     y<br>     li<br>      n<br>    ct<br>    m<br>  w<br>S <br> <br>s<br>     -<br>c<br>G<br>   ar<br>o<br>    m<br> d|FF-Erase<br>(R)<br>AUC <br>     igure<br>      ed as<br>    , in §<br>     ty of<br>      tered<br>    iven<br>    etho<br>  e exp<br> IN M<br>due <br>ets a<br>     MNI<br>ient <br>A-ba<br>   ning<br>re co<br>    pler<br>ata l|s<br> <br> <br>      C<br>      o<br> <br>    es<br>    d<br>  l<br> <br>to<br>n<br>     S<br> t<br>s<br>   e<br><br>     d<br>e|GA<br>( =10)<br>core <br>     (b) p<br>       effec<br>      .1, w<br>      ur f<br>       kerne<br>    s of<br>    s as<br>  ore t<br>AIN <br> spa<br>d va<br>     T. T<br>han <br>ed m<br>   poch<br>mple<br>     atas<br>arn|<br>      ro<br>       ti<br>      e<br>      n<br>       l<br>     o<br>     b<br>   h<br> P<br>c<br>ri<br>     h<br>R<br> e<br>   s<br>x <br>     et<br>m|


making the information removal of forgetting data more difficult.


Similarly, Figure 9 presents results across different FF model architectures (TinyCNN, AlexNet, and
VGG16). These experiments confirm that the conclusions regarding the efficiency, effectiveness, and
utility of our proposed method hold irrespective of the underlying model architecture.


C.2 LAYER-WISE UNLEARNING


As FF models are trained using a layer-wise and greedy optimization, different layers may retain
different aspects of knowledge from the training data. To empirically show the effectiveness of
our proposed FF-Erase in removing residual knowledge from all layers, we further evaluate the
layer-wise unlearning performance using Centered Kernel Alignment (CKA) Wang et al. (2015);
Kornblith et al. (2019) similarity as the metric.


CKA is a similarity measure between two sets of representations, which has been widely used in
analyzing the unlearning effectiveness of different layers in neural networks Xu et al. (2025); Kim
et al. (2025). Given two representation matrices _Xi_ _[o]_ _[∈]_ [R] _[n][×][p]_ [from original model and] _[ X]_ _i_ _[u]_ _[∈]_ [R] _[n][×][p]_
from unlearned model, where _n_ is the number of samples, and _p_ are the dimensions of the output
representations of layer _i_, the linear CKA similarity between them is calculated by:


HSIC( _Xi_ _[o][, X]_ _i_ _[u]_ [)]
CKA( _Xi_ _[o][, X]_ _i_ _[u]_ [) =] ~~�~~ HSIC( _Xi_ _[o][, X]_ _i_ _[o]_ [)HSIC(] _[X]_ _i_ _[u][, X]_ _i_ _[u]_ [)] _,_ (12)


where HSIC( _·, ·_ ) is the Hilbert-Schmidt Independence Criterion. Detailed calculations of HSIC can
be found in Wang et al. (2015). The CKA similarity ranges from 0 to 1, where a lower value indicates


16


0.70


0.65


0.60


0.55


0.60


0.58


0.56


0.54


0.52


0.50


0.50


|Col1|Col2|
|---|---|
|||
|R<br>G<br>|A<br>|
|F<br>FF|-Erase(D<br>-Erase(R)|


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
|||R<br>|E<br>~~A~~||
|||F<br>~~F~~|F-Erase(D<br>~~F-Erase(R~~|)<br>~~)~~|


|Col1|Col2|Col3|
|---|---|---|
||||
||~~R~~<br>G<br>|~~E~~<br>A<br>|
||~~F~~<br>F|~~-Erase(D)~~<br>-Erase(R)|


|Col1|Col2|
|---|---|
|||
|||
|RE<br>~~G~~||
|FF<br>FF|-Erase(D)<br>-Erase(R)|


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
|||R<br>~~G~~|||
|||FF<br>~~FF~~|-Erase(D<br>~~-Erase(R~~|)<br>|


|Col1|Col2|Col3|
|---|---|---|
||||
||RE<br>GA<br>||
||~~FF~~<br>FF|~~-Erase(D)~~<br>-Erase(R)|


|Col1|Col2|
|---|---|
|||
|R<br>G||
|FF<br>|-Erase(D)<br>|
|FF|-Erase(R)|


|Col1|Col2|Col3|Col4|
|---|---|---|---|
|||||
||~~R~~<br>G<br>~~F~~<br>|~~E~~<br>A<br>~~F-Erase(D~~<br>|~~)~~<br>|
||F|F-Erase(R|)|


|Col1|Col2|Col3|
|---|---|---|
||||
||~~R~~<br>G<br>~~F~~<br>|A<br>~~F-Erase(D)~~<br>|
||F|-Erase(R)|


|Col1|Col2|
|---|---|
|RE<br>||
|~~G~~<br>FF<br>|-Erase(D)<br>|
|FF|-Erase(R)|


|Col1|Col2|Col3|Col4|
|---|---|---|---|
||R<br>G<br>~~FF~~<br>|E<br>A<br>~~-Erase(D~~<br>|~~)~~<br>|
||FF|-Erase(R||


|Col1|Col2|Col3|
|---|---|---|
||RE<br>GA<br>FF<br>|-Erase(D)<br>|
||FF|-Erase(R)|


Figure 9: Comparison among different FF unlearning methods under different FF models. Figures
in the first line (a), (b), and (c) show the model accuracy on forgetting data Dforget and the figures in
the second line (d), (e), and (f) show the model accuracy on test data Dtest.


a larger difference between the two representations. We compare the CKA similarity on forgetting
data between the original and unlearned models, including FF-Erase(D), FF-Erase(E), retraining
(RE), and direct gradient ascent (GA), following the same setting of Figure 4. We additionally add
extra baselines: Bad Teacher (BT) Chundawat et al. (2023a), an advanced distillation-based gradient
ascent method of approximate unlearning; FATS Tao et al. (2024), an advanced retraining-based
exact unlearning; FYE Tarun et al. (2023b), a robust gradient ascent method. The CKA scores of RE
show an unlearning standard. The results are shown in Table 3 as follows: Firstly, for the retraining
model (RE), the CKA similarity scores first decrease and then increase along the layer depth. For
shallow layers, CKA scores are highest, indicating that those layers learn general features shared
by both forgetting and remaining data, making them hard to unlearn even by retraining. For middle
layers, CKA scores are lowest, implying that those layers learn more specific features of forgetting


17


60


40


20


Time (/s)


(e) MNIST VGG13


50

40

30

20

10


Time (/s)


(d) CIFAR100 VGG13


0 200 400 600 800
Time (/s)


(b) CIFAR10 AlexNet


0 200 400 600 800
Time (/s)


(e) CIFAR10 AlexNet


(c) CIFAR10 VGG16


80


60


40


80


60


40


20


(d) CIFAR10 TinyCNN


|Layer|1|2|3|4|5|6|7|
|---|---|---|---|---|---|---|---|
|RE<br>FF-Erase(D)<br>FF-Erase(R)<br>GA(_λ_=10)<br>GA(_λ_=10_−_2)<br>BT<br>FYE<br>FATS|0.9918<br>0.9958<br>0.9961<br>0.5809_ ↓_<br>0.9952<br>0.9998<br>0.9827<br>0.9723|0.9249<br>0.9658<br>0.9721<br>0.3886_ ↓_<br>0.9704<br>0.9975<br>0.6004_ ↓_<br>0.9118|0.8004<br>0.9066<br>0.9199<br>0.2915_ ↓_<br>0.9219<br>0.9915<br>0.2147_ ↓_<br>0.7971|0.6319<br>0.8100<br>0.8279<br>0.2454_ ↓_<br>0.8368_ ↑_<br>0.9638_ ↑_<br>0.1242_ ↓_<br>0.6772|0.6286<br>0.7584<br>0.7816<br>0.2308_ ↓_<br>0.7801<br>0.9572_ ↑_<br>0.1003_ ↓_<br>0.6478|0.5700<br>0.7309<br>0.7454<br>0.1892_ ↓_<br>0.7404<br>0.9008_ ↑_<br>0.0571_ ↓_<br>0.5677|0.3619<br>0.6168_ ↑_<br>0.6292_ ↑_<br>0.1476_ ↓_<br>0.6152_ ↑_<br>0.7571_ ↑_<br>0.0331_ ↓_<br>0.3827|
|Layer|8|9|10|11|12|13|-|
|RE<br>FF-Erase(D)<br>FF-Erase(R)<br>GA(_λ_=10)<br>GA(_λ_=10_−_2)<br>BT<br>FYE<br>FATS|0.3472<br>0.5737_ ↑_<br>0.5919_ ↑_<br>0.1317_ ↓_<br>0.5562_ ↑_<br>0.6368_ ↑_<br>0.0295_ ↓_<br>0.3312|0.4981<br>0.5462<br>0.5930<br>0.1495_ ↓_<br>0.5888<br>0.7053_ ↑_<br>0.0262_ ↓_<br>0.4502|0.4200<br>0.4548<br>0.5128<br>0.1255_ ↓_<br>0.4809<br>0.5766<br>0.0150_ ↓_<br>0.3564|0.4923<br>0.5071<br>0.5748<br>0.1116_ ↓_<br>0.5568<br>0.6097<br>0.0075_ ↓_<br>0.3713|0.6223<br>0.5742<br>0.6490<br>0.0867_ ↓_<br>0.6309<br>0.7573<br>0.0072_ ↓_<br>0.5776|0.6144<br>0.5488<br>0.6358<br>0.0656_ ↓_<br>0.6144<br>0.0049_ ↓_<br>0.0056_ ↓_<br>0.5944|-<br>-<br>-<br>-<br>-<br>-<br>-<br>-|


Table 3: Layer-wise CKA similarity between original and unlearned models on forgetting data under
the VGG13 and CIFAR-10 settings of Figure 4. We use _↑_ ( _↓_ ) to denote CKA scores that are 20%
higher (lower) than the RE scores, which suggests ineffective unlearning (over forgetting).


data, which can be effectively unlearned by retraining. It might also suggest unique characteristics
of FF models, where middle layers are more specialized for each training process. For deep layers,
CKA scores slightly increase again, which may be because deep layers learn high-level features
shared by both forgetting and remaining data, as our forgetting data are randomly sampled rather
than unlearning specific classes of data.


Secondly, for FF-Erase methods, the CKA scores show a clear trend of decreasing similarity across
all layers, indicating their effectiveness in unlearning the forgetting data. This is particularly evident
in FF-Erase(D), which consistently outperforms FF-Erase(R) in all layers. The results suggest that
FF-Erase(D) is more effective in erasing the specific features learned from the forgetting data, while
FF-Erase(R) retains some of those features, leading to higher CKA scores.


Thirdly, for conventional approximate unlearning methods, the layer-wise CKA scores are consistent with our conclusion: some lead to model collapse ( _e.g._, GA( _λ_ = 10) and FYE) and some lead
to ineffective unlearning ( _e.g._, GA( _λ_ = 10 _[−]_ [2] ) and BT). For GA, a direct gradient ascent method
relying on loss on forgetting data, the CKA scores (except the first layer) decrease sharply, which is
reasonable since a collapsed model (model unlearned by GA) is not supposed to extract similar representations as a well-trained model (the original model). For BT, a distillation-based gradient ascent
method, its CKA scores show why it is ineffective for FF models in two aspects: the shallow and
middle layers retain too much information on forgetting data (a significantly higher score compared
to RE for each layer), while the last layer extracts totally different representations. Such a model
state, where residual knowledge persists in shallow/middle layers while over-forgetting happens in
deep layers, will be easily captured by G-MIA attacks.


Lastly, for FATS, a retraining-based exact unlearning method, its CKA scores are close to RE, indicating its effectiveness. However, as shown in §C.3, FATS still requires a large amount of unlearning
time, making it inefficient compared to our proposed FF-Erase.


C.3 SUPPLEMENTARY EXPERIMENTS


In §6, we use direct gradient ascent (GA) as the representative of conventional BP unlearning methods. Here we provide supplementary experimental results covering additional baselines, including
robust GA methods FYE Tarun et al. (2023b) and SURE Sepahvand et al. (2025), distillation-based
GA method Bad Teacher (BT) Chundawat et al. (2023a), and advanced retraining-based method
FATS Tao et al. (2024). FATS utilizes incremental learning to bypass unnecessary retraining epochs;
its expectation of unlearning time for a single data point is half of RE. It is noted that those new
baselines often trigger our termination conditions in §4.1, including timeout (exceeds a maximum


18


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


|Col1|Col2|Col3|Col4|RE|Col6|
|---|---|---|---|---|---|
|||||~~GA~~<br>FF-Eras<br>FF-Eras<br>|e(D)<br>e(R)|
|||||BT<br>FYE<br>~~SURE~~<br>||


|Col1|Col2|Col3|Col4|RE|Col6|
|---|---|---|---|---|---|
|||||GA<br>F-Era|e(D)|
|||||FF-Eras<br>BT<br>FYE<br>SURE<br>|e(R)|


“Orig. Model” for “Original Model” and “FE” for “FF-Erase”.


as follows:

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|
|---|---|---|---|---|---|---|---|---|---|---|
|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|0.58<br>0.59|
|||~~0.55~~|~~0.55~~|~~0.55~~|~~0.55~~||0.56<br>|0.56<br>|0.56<br>|0.56<br>|
|||~~0.53~~<br>0.530.53<br>|~~0.53~~<br>0.530.53<br>|~~0.53~~<br>0.530.53<br>|~~0.53~~<br>0.530.53<br>|||~~0.55~~<br>0.53|~~0.55~~<br>0.53|~~0.55~~<br>0.53|
|M<br>c)<br>      s<br> in<br>el<br>      o<br>  e<br>a<br>      d<br>      e<br>      0<br>     .<br>ut<br> <br>l<br>a<br>     n<br>     l|Orig.<br>ode<br> G<br>      ar<br> es<br>, s<br>       th<br>  as<br>ta <br>      §2<br>      t),<br>      0s<br>ili<br>dir<br>ea<br>lid<br>     g,<br>     s.|l<br>RE<br>-M<br>      e p<br>  re<br>o <br>       e s<br>  e o<br> lo<br>     , s<br>       th<br>      ),<br>ty<br>ec<br>rni<br> g<br>      the|FE<br>(D)<br>IA<br>       res<br>  pr<br>the<br>       pa<br>   n r<br>we<br>       ho<br>       e a<br>      wh<br> co<br>t G<br>ng <br>oo<br>      y a|FE<br>(R)<br>( <br> sc<br>       en<br>  ese<br>y <br>       ce<br>   em<br>r t<br>       wi<br>       dva<br>      ich<br> lla<br>A <br> re<br>dne<br>      re|GA<br> =10<br> ore<br>       t f<br>  nt<br>onl<br>        lim<br>   ai<br>han<br>       ng<br>       nc<br>       is<br> pse<br> do<br>qu<br>ss <br>       no|)<br>BT<br> (A<br>       or<br>  th<br>y <br>        it,<br>   nin<br> a<br>       in<br>       ed<br>       cl<br> )<br>es<br>ire<br> di<br>       t f|FYE<br><br> C<br>        ge<br>  e a<br>ha<br>        in<br>   g<br> th<br>        Fi<br>        re<br>       os<br> and<br>. T<br>s n<br>str<br>       eas|SURE<br> C)<br>        ner<br>   cc<br>ve <br>        clu<br>    dat<br>re<br>        gu<br>        tra<br>       e t<br>  S<br>hi<br>ot <br>ibu<br>       ib|FATS<br>        ati<br>   ura<br> so<br>        di<br>    a<br>sh<br>        re<br>        ini<br>        o<br>  U<br>s <br> o<br>ti<br>       le|n<br>   c<br>li<br>        n<br>    f<br>ol<br>        1<br>        n<br>        R<br>  R<br>e<br>nl<br>o<br>        f|


Thirdly, the distillation-based GA method, Bad Teacher (BT)(terminated at around 330s by loss
plateau), still leads to ineffective unlearning as direct GA does (when _λ_ is small). The G-MIA
accuracy on BT method (0.59) is significantly higher than others ( _≤_ 0.58), demonstrating its ineffectiveness in quantitative. This experimental result is consistent with our explanation in §A, that
those distillation-based GA methods are not suitable for FF unlearning due to their dependence on
final-layer logits.


Overall, these supplementary experiments confirm the representativeness of direct GA as the baseline of conventional BP unlearning methods in §6. Comparisons with those additional baselines
further validate our analysis in §1 and §2: exact unlearning methods are too time-consuming, while
approximate unlearning methods (represented by direct gradient ascent, GA) lead to either model
collapse or ineffective unlearning.


C.4 DETAILS OF GUIDANCE MODEL


In this section, we explore the training curve of guidance model training used in §6.4. All the
guidance models have a much lower accuracy on various datasets regardless the training time, and
thus being not suitable to directly serve as the unlearned model. However, when we apply our
strategies in §4.2, we can efficiently obtain guidance models for FF-Erase unlearning. The details
are as follows:


Firstly, even given sufficient training time, all mini-retrained guidance models ( _≤_ 75%) and fastdistilled guidance models ( _≤_ 70%) have a significantly lower accuracy on test set compared to the
retraining methods RE (81%). The guidance models used in FF-Erase have a worse accuracy (60%71%) than corresponding sufficiently-trained cases, because they are picked before converging to
reduce the overall unlearning time.


19


0 200 400 600 800 1000
Time (/s)


0.60

0.58

0.56

0.54

0.52


80


60


40


80


60


40


0 200 400 600 800 1000
Time (/s)


0.50


(a) Time vs. Dforget Accuracy


(b) Time vs. Dtest Accuracy


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


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||R<br>|E<br>||D-(0.5<br>|0.5)<br>|
||~~R~~<br>R<br>~~R~~|~~-(0.5,0.~~<br>-(0.5,0.<br>~~(0.3,0.~~|~~)~~<br>2)<br>~~)~~|~~D-(0.5,~~<br>D-(0.5,<br>~~D-(0.3~~|~~0.2)~~<br>0.1)<br>~~0.5)~~|
||R|-(0.3,0.|)|D-(0.3,|0.2)|
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||R<br>|E<br>||D-(0.5<br>|0.5)<br>|
||~~R~~<br>R<br>~~R~~|~~-(0.5,0.~~<br>-(0.5,0.<br>~~(0.3,0.~~|~~)~~<br>2)<br>~~)~~|~~D-(0.5,~~<br>D-(0.5,<br>~~D-(0.3~~|~~0.2)~~<br>0.1)<br>~~0.5)~~|
||R|-(0.3,0.|)|D-(0.3,|0.2)|
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
||~~R~~<br>|~~E~~<br>||~~D-(0.5~~<br>|~~0.5)~~<br>|
||R-<br>R-<br>~~R~~|(0.5,0.<br>(0.5,0.<br>~~(0.3,0.~~|)<br>2)<br>~~)~~|D-(0.5,<br>D-(0.5,<br>~~D-(0.3~~|0.2)<br>0.1)<br>~~0.5)~~|
||R-|(0.3,0.|2)|D-(0.3,|0.2)|


Figure 11: Training curve of guidance models used in Table 1. The curve shows the accuracy of the
models on different datasets over time. Figures (a), (b), and (c) shows the accuracy on forgetting
data, test data and remaining data, respectively. The five lines respectively presents retraining, miniretraining using 0.5 of remaining data, mini-retraining using 0.3 of remaining data, fast-distilling
using 0.5 of remaining data, and fast-distilling using 0.3 of remaining data. The points on the curve
indicates the guidance model we pick up. For example, the blue triangle represents taking 20% of
the training epochs for fast-distilling using 0.5 of the remaining data.


Secondly, the guidance models are efficient to obtain. Compared with retraining from stratch
(around 1100s), it takes only about 5% to obtain an R-(0.3,0.2) guidance model (around 40s). With
efficient guidance model obtaining, the overall FF-Erase unlearning cost is significantly lower than
retraining while forgetting as effective as retraining, as shown in §6 and §C.


D USAGE OF LLM


In this paper, we use the Large Language Model (LLM) to aid or polish writing. Details are described
as follows:


In the interest of full transparency, we utilized LLMs (including Gemini-2.5-pro and GPT-5) as a
writing assistant to enhance the clarity, conciseness, and overall readability of this manuscript. The
LLM’s role was strictly limited to improving sentence structure, grammar, and flow. All scientific
contributions, experimental results, and core arguments were conceived and articulated exclusively
by the authors, who reviewed and approved every revision to ensure the integrity and accuracy of
the final text.


20


80

70

60

50

40

30


80


60


40


80

70

60

50

40

30


0 200 400 600 800 1000
Time (/s)


(c) Time vs. Dremain Accuracy


0 200 400 600 800 1000
Time (/s)


(a) Time vs. Dforget Accuracy


0 200 400 600 800 1000
Time (/s)


(b) Time vs. Dtest Accuracy