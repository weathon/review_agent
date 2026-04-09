# SUSI: SEMI-STRUCTURED PRUNING FOR LLMS VIA DIFFERENTIABLE SUBSET SAMPLING


**Anonymous authors**
Paper under double-blind review


ABSTRACT


The rapid growth of large language models (LLMs) has driven the need for efficient post-training optimization techniques for reducing computational and memory demands while preserving performance. Semi-structured pruning, which enforces hardware-compatible sparsity patterns like N:M sparsity, offers a balanced
approach for accelerating inference. In this study, we introduce SUSI [1] (Semistructured prUning via Subset samplIng), a novel semi-structured pruning method
that leverages the weighted reservoir and differentiable subset sampling to learn
high-quality N:M sparsity masks with minimal computational cost. Compared to
other learnable mask methods (i.e., MaskLLM), which increase parameter complexity, SUSI reduces trainable parameters by up to 1.5× for the 2:4 sparsity, enabling efficient deployment on hardware optimized for sparse computation. We
evaluate SUSI on three OPT model variants (125M, 350M, and 1.3B parameters)
using benchmarks including Wikitext-2 for perplexity and zero-shot NLP tasks
(e.g., ARC, HellaSwag, PIQA, RACE, SciQ). SUSI outperforms baselines such
as SparseGPT, Wanda, and MaskLLM in perplexity while maintaining competitive zero-shot accuracy across various benchmarks. These results establish SUSI
as a robust and practical solution for compressing LLMs, facilitating efficient deployment in resource-constrained environments.


1 INTRODUCTION


With the rapid development of large language models (LLMs), post-training techniques have
emerged as critical methodologies for optimizing model efficiency while preserving performance
(Wan et al., 2024). Among these techniques, two primary approaches to network compression have
gained prominence: model quantization (Egashira et al., 2024; Liu et al., 2025b) and network pruning (Cheng et al., 2024; Mu˜noz et al., 2025). While model quantization focuses on representing
weights with reduced precision (e.g., 8-bit, 4-bit, or lower), pruning techniques aim to eliminate redundant parameters to accelerate inference while preserving task performance (Williams & Aletras,
2024). This study focuses on pruning techniques to develop sparse LLMs, thereby reducing memory
footprint and enhancing inference speed.


Current post-training pruning methods can be categorized into three distinct approaches: (i) unstructured pruning, which removes individual weight parameters without regard to network architecture
(Sun et al., 2024); (ii) structured pruning, which eliminates entire network components such as neurons, attention heads, or layers (Xia et al., 2024; Le et al., 2025); and (iii) Semi-structured pruning,
which combines the flexibility of unstructured methods with the regularity of structured patterns
(Fang et al., 2024; Huang et al., 2025). This research focuses on semi-structured pruning, as it
efficiently removes redundant weights while enforcing regular sparsity patterns that are hardwarecompatible and effective for acceleration. Specifically, semi-structured pruning strikes an optimal
balance by keeping regular sparsity patterns (e.g., N:M sparsity (Hubara et al., 2021)), which is
optimized for hardware. Modern approaches in this field are generally categorized into two types:
i) _importance-based_ : with several typical methods such as SparseGPT (Frantar & Alistarh, 2023)
and Wanda (Sun et al., 2024) using a small dataset, typically a subset of the pretraining data, to approximate the knowledge encoded in the language model. They define an importance score for each
weight (or group of weights) based on this dataset, which guides the pruning process. Importance


1https://anonymous.4open.science/r/susi-2E2C


1


|Col1|Col2|Col3|re|
|---|---|---|---|
|||||


Figure 1: Learnable semi-structured N:M sparsity methods: a) modeling the mask selection process
using a categorical distribution over feasible masks, and b) our proposed method by learning to
sample subsets without replacement of model parameters. The proposed method is more memory
efficient than previous works for most practical N:M sparsity patterns. The memory advantage
becomes more pronounced as _M_ increases or when _N_ is around _M/_ 2.


scores may be based on weight magnitude, gradients, or the Hessian matrix. However, these criteria
are often chosen heuristically, leading to potentially sub-optimal results. Additionally, the limited
dataset may not adequately capture the model’s rich knowledge; ii) _learnable_ _masks_ : focusing on
the direct optimization of pruning masks through a retraining process. Recently, MaskLLM (Fang
et al., 2024) proposed a novel method that models N:M sparsity patterns as learnable categorical distributions, employing Gumbel-Softmax sampling (Jang et al., 2017). This approach demonstrates
robust pruning performance and strong generalization across diverse tasks. However, it introduces
significant computational overhead due to an increased number of trainable parameters (Huang et al.,
2025). Specifically, for a model with _W_ parameters under N:M sparsity, MaskLLM requires learning - _MN_ - _×_ _[W]_ _M_ [parameters, which consistently equals or surpasses the original model parameter count,]

as illustrated in Figure 1(a). For instance, with the commonly utilized 2:4 semi-structured sparsity
pattern, the number of parameters to be learned is 1 _._ 5 _× W_ . This substantial parameter overhead
poses considerable challenges during the training of large-scale language models.


To address this limitation, we propose an effective semi-structured pruning method, termed SUSI
(Semi-structured prUning via Subset samplIng). SUSI systematically selects _N_ weights from
each group of _M_ consecutive parameters, enabling the enforcement of N:M sparsity with minimal
degradation in model accuracy. The main idea is to utilize Weighted Reservoir Sampling (WRS)
(Efraimidis & Spirakis, 2006) as an efficient alternative for learning high-quality sparsity masks.
WRS enables selective sampling of mask configurations based on importance weights, reducing
computational overhead while maintaining the ability to identify effective N:M sparsity patterns.
The proposed lightweight pruning mask learning technique significantly reduces the number of trainable parameters, thereby facilitating efficient deployment on hardware optimized for N:M sparsity,
as depicted in Figure 1(b).


2 PRELIMINARIES


2.1 WEIGHTED RESERVOIR SAMPLING


Weighted Reservoir Sampling (WRS) (Efraimidis & Spirakis, 2006) is an extension of the Reservoir Sampling class of algorithms (Vitter, 1985), which aims to sample _K_ items from a set
of _N_ . In WRS, each item is assigned a non-negative weight, and items with larger weights
compared to others are more likely to appear in the sampled subset. Given a population set
_X_ = _{x_ 1 _, x_ 2 _, . . ., xN_ _}_ with corresponding weights _**w**_ = [ _w_ 1 _, w_ 2 _, . . ., wN_ ], WRS produces an
ordered subset _Y_ = _{y_ 1 _, y_ 2 _, . . ., yK}_, which is drawn from following distribution:


_P_ WRS( _Y|_ _**w**_ ) = _[w][y]_ [1] _wy_ 2 _× . . . ×_ _wyK_ (1)

_W_ _[×]_ _W_ _−_ _wy_ 1 _W_ _−_ [�] _[K]_ _j_ =1 _[−]_ [1] _[w][y]_ _j_


2


**Memory Comparison**


**Pattern N:M (K)** **2:4 (6)** **1:4(4)** **4:8(70)** **2:8(28)**


**Ratio** **6:4 = 1.5×** **4:4 = 1.0×** **70:8 = 8.75×** **28:8 = 3.5×**


Categorical
distribution
over K =


Categorical distribution over
M model parameters


Sample
without


**a) Conventional Approach (e.g., MaskLLM)**


**b) SUSI**


where _W_ = [�] _i_ _[N]_ =1 _[w][i]_ [is the total weight and] _[ w][y]_ _i_ [is the weight of the corresponding item] _[ y][i]_ [.] [Sam-]
pling from the above distribution resembles the sampling without replacement process, where the
probability of selecting a subset is proportional to the item weights.


2.2 GUMBEL-TOP- _K_ TRICK


Gumbel-Max (Gumbel, 1954) is a monotonic transformation of the WRS technique, a reparameterization trick to sample from a categorical distribution by perturbing the distribution’s logprobabilities with Gumbel noise. Given a categorical distribution over _N_ items _{x_ 1 _, . . ., xN_ _}_ parameterized by _N_ logit parameters _**ϕ**_ = [ _ϕ_ 1 _, . . ., ϕN_ ], the probability of an arbitrary item’s selection
is _πi_ = exp( _ϕi_ ) _/_ [�] _[N]_ _j_ =1 [exp(] _[ϕ][j]_ [)][.] [The Gumbel-Max trick performs sampling from such a distribu-]
tion by first generating random keys corresponding to each item via Gumbel perturbations:


i.i.d
_κi_ = _ϕi_ + _gi,_ _gi_ _∼_ Gumbel(0 _,_ 1) (2)


where _gi_ s are noise independently drawn from the Gumbel(0 _,_ 1) distribution. Finally, the output of
this sampling process is achieved by taking the item _xj_ having the largest key _κj_ . The index _j_ is the
output of taking argmax over key values ( _j_ = argmax _i_ _κi_ ).


Gumbel-Top- _K_ is a generalization of the Gumbel-Max trick, where instead of selecting the item
with the largest random key, the top- _K_ items with the highest keys are selected (Xie & Ermon,
2019). This corresponds to sampling _K_ items without replacement from a categorical distribution
over _N_ items. By relaxing the argtop _K_ operator using successive softmaxes (Pl¨otz & Roth, 2018),
this sampling process becomes differentiable, thereby allowing for learning with backpropagation.


To sample a subset of _K_ items with the Gumbel-Top- _K_ trick, logits are first independently perturbed
with Gumbel noise to create random keys _κi_, similar to the Gumbel-Max trick. Sequentially, a
chain of softmax is applied to produce approximated one-hot representations of selected items. Let
_**α**_ [(] _[k]_ [)] = [ _α_ 1 _, . . ., αN_ ] denote adjusted keys at the sampling step _k_ . These adjusted keys are defined
recursively as follows:


_**α**_ [(1)] := [ _κ_ 1 _, . . ., κN_ ]; _**α**_ [(] _[k]_ [)] := _**α**_ [(] _[k][−]_ [1)] + log(1 _−_ _**µ**_ [(] _[k][−]_ [1)] ) (3)


where _**µ**_ [(] _[k][−]_ [1)] = [ _µ_ [(] 1 _[k][−]_ [1)] _, . . ., µ_ [(] _N_ _[k][−]_ [1)] ] is the one-hot approximation indicating the item selected at
the previous sampling step. This representation is achieved by applying softmax over adjusted keys
at the sampling step _k −_ 1 with a pre-defined temperature _τ_ :

_µ_ [(] _i_ _[k][−]_ [1)] =            - _Nj_ exp(=1 [exp(] _αi_ [(] _[k][α][−]_ _j_ [(][1)] _[k][−]_ _/τ_ [1)] ) _/τ_ ) (4)


After applying softmaxes _K_ times, we attain an ordered subset of _K_ approximated one-hot representing selected items _S_ = _{_ _**µ**_ [(1)] _, . . .,_ _**µ**_ [(] _[K]_ [)] _}_ . The sum of elements in this subset yields a soft
_K_ -hot vector, and the mapping from the logits _ϕi_ to this vector is differentiable, enabling usage of
gradient-based optimization methods.


3 METHODOLOGY


3.1 PROBLEM STATEMENT


The problem of finding the optimal N:M sparsity can be formulated as selecting, for each group of _M_
consecutive parameters, a binary mask of length _M_ with exactly _N_ non-zero entries that minimizes
the loss on a calibration set. Let _G_ denote the number of weight groups, **W** = _{_ **w** 1 _, . . .,_ **w** _G}_
the corresponding weight groups, and **M** = _{_ **m** 1 _, . . .,_ **m** _G}_ the associated binary masks. The
optimization problem is then defined as follows:


**M** _[∗]_ = argmin _L_ CE( _D_ ; **W** _⊙_ **M** ) (5)
**M**


where _L_ CE is the cross-entropy loss for language modeling, _D_ denotes the calibration set, and _⊙_
represents the element-wise product between each weight group and its corresponding binary mask.


3


Figure 2: Overview of the SUSI Framework for Semi-Structured Pruning via Differentiable Subset
Sampling, illustrating the training and inference phases.


However, such an optimization problem is NP-hard due to the vast search space, where there exists - _MN_ - _G_ feasible solutions. In the context of Large Language Models, the number of weight
groups _G_ is gargantuan, making this combinatorial optimization problem impractical to brute-force.
Therefore, in the following section, we reformulate the above problem as a stochastic variational
optimization variant to gain tractability and improve efficiency.


3.2 SUSI: SEMI-STRUCTURED PRUNING VIA DIFFERENTIABLE SUBSET SAMPLING


The overview of SUSI is illustrated in Figure 2. Accordingly, stochastic variational optimization
(Bird et al., 2018) is based on an observation that given an arbitrary distribution _q_ ( _x_ ) the expectation
of a function _f_ ( _x_ ) provides an upper bound on its minimum:


min (6)
_x_ _[f]_ [(] _[x]_ [)] _[ ≤]_ [E] _[q]_ [(] _[x]_ [)][[] _[f]_ [(] _[x]_ [)]]


By treating pruning masks as random variables, the optimization problem in the Equation 5 can be
reframed as minimizing the variational upper bound of the objective with respect to the variational
distribution parameters. Formally, we seek to find:

**Φ** _[∗]_ = argmin E _P_ ( **M** _|_ **Φ** )[ _L_ CE( _D_ ; **W** _⊙_ **M** )] (7)
**Φ**


where **Φ** = _{_ _**ϕ**_ 1 _, . . .,_ _**ϕ**_ _G}_ is a set of parameters, corresponding to variational distributions
_P_ ( **m** 1 _|_ _**ϕ**_ 1), ..., _P_ ( **m** _G|_ _**ϕ**_ _G_ ), with joint distribution _P_ ( **M** _|_ **Φ** ) = [�] _i_ _[G]_ =1 _[P]_ [(] **[m]** _[i][|]_ _**[ϕ]**_ _[i]_ [)][.] [Through] [this]
formulation as a stochastic variational optimization problem, the sampling of masks can be reparameterized and relaxed to be a differentiable function with respect to variational distributions’
parameters, making it possible to learn via gradient-based optimization.


3.2.1 VARIATIONAL DISTRIBUTION SELECTION


Since pruning masks are _N_ -hot vectors of length _M_, each mask can take one of - _MN_ - possible
values. Modeling such a distribution over possible values requires - _MN_ - _−_ 1 free parameters, which
grows combinatorially as _M_ increases. To efficiently learn masks with a reasonable number of
parameters, we propose using the WRS distribution (Equation 1) over ordered subsets to model
mask distributions _P_ ( **m** _i|_ _**ϕ**_ _i_ ). Let _Si_ = _{_ _**µ**_ [(1)] _i_ _[, . . .,]_ _**[ µ]**_ _i_ [(] _[N]_ [)] _}_ be a set of _N_ one-hot vectors representing
selected weights within the _i_ -th group, sampled from the WRS distribution using the Gumbel-Top- _K_
trick, the probability of a mask **m** _i_ is then:


_P_ ( **m** _i|_ _**ϕ**_ _i_ ) =          - _P_ WRS( _S_ **m** _i|_ exp( _**ϕ**_ _i_ )) (8)

_S_ **m** _i_


where _S_ **m** _i_ denotes the set of elements whose sum equals **m**, where the _N_ -hot mask **m** _i_ can be
obtained by summing up _**µ**_ [(] _i_ _[j]_ [)][s.] [Exactly] [computing] [this] [probability] [is] [expensive] [and] [unnecessary]
since constructing **m** _i_ ignores the order of items in the sampled subset, and the expected loss can


4


❄️ : Frozen

Training 𝑓 : Language model Inference


be computed via Monte Carlo sampling. The original problem then turns into learning to select
important weights, with importance score exp( _ϕij_ ), in order to minimize the objective. Denoting
the WRS distribution of an arbitrary subset _Si_ as _P_ WRS( _Si|_ _**ϕ**_ _i_ ) for brevity, the optimization problem
(Equation 7) is then reformulated as follows:

**Φ** _[∗]_ = argmin E _P_ WRS( _S|_ **Φ** )[ _L_ CE( _D_ ; **W** _⊙_ **M** )] (9)
**Φ**


where _S_ = _{S_ 1 _, . . ., SG}_ is a collection of _G_ subsets, generated from the joint distribution
_P_ WRS( _S|_ **Φ** ) = [�] _i_ _[G]_ =1 _[P]_ [WRS][(] _[S][i][|]_ _**[ϕ]**_ _[i]_ [)][. Each] _[ N]_ [-hot pruning mask] **[ m]** _[i]_ [ in the collection] **[ M]** [ is constructed]
as **m** _i_ = [�] _**µ**_ [(] _i_ _[j]_ [)] _∈Si_ _**[µ]**_ _i_ [(] _[j]_ [)][.] [Parameterizing] [masks] [as] [sums] [of] [subsets] [sampled] [from] [WRS-restricted]

distributions yields the same expected loss as sampling masks from the exact distributions, as proved
in Theorem 1. Our approach reduces the parameter complexity by reformulating the _N_ -hot mask
sampling process as a sequential sampling without replacement paradigm. Instead of maintaining a
full categorical distribution over - _MN_ - configurations, we model only a single categorical distribution
over every _M_ model parameters, requiring exactly _M_ parameters regardless of _N_ . The proposed
�� _M_                         - [�]
method achieves a reduction in parameter complexity from _O_ _N_ to _O_ ( _M_ ), representing an
exponential improvement in memory efficiency.


3.2.2 MASK SELECTION RELAXATION


To make the objective differentiable with respect to variational distributions’ parameters, we relax
the sampling process using the Gumbel-Top- _K_ trick. Given logits _**ϕ**_ _i_ = [ _ϕi_ 1 _, . . ., ϕiM_ ] forming
a categorical distribution over _M_ consecutive model weights within the _i_ -th group, the probability
of selecting the _j_ -th weight is achieved via softmax: _πij_ = exp( _ϕij_ ) _/_ [�] _k_ _[M]_ =1 [exp(] _[ϕ][ik]_ [)][.] [To sample]
a subset _Si_ without replacement from this distribution, we first perturb logits with Gumbel noise
independently to attain random keys:


i.i.d
_κij_ = _ϕij_ + _gij,_ _gij_ _∼_ Gumbel(0 _,_ 1) (10)


We define the adjusted keys of the _i_ -th group at sampling step _k_ as _**α**_ [(] _i_ _[k]_ [)] = [ _αi_ [(] 1 _[k]_ [)] _[, . . ., α]_ _iM_ [(] _[k]_ [)] []][,] [The]
update rule follows the Gumbel-Top-K procedure, except that we incorporate a power term _p >_ 1 to
amplify the impact of removing the selected item in the previous sampling step. This modification
improves stability during training. Formally:


_**α**_ [(1)] _i_ := [ _κi_ 1 _, . . ., κiM_ ] _,_ _**α**_ [(] _i_ _[k]_ [)] := _**α**_ [(] _i_ _[k][−]_ [1)] _−|_ log(1 _−_ _**µ**_ [(] _i_ _[k][−]_ [1)] ) _|_ _[p]_ (11)


Finally, an approximated relaxed one-hot vector _**µ**_ [(] _i_ _[k]_ [)] = [ _µ_ [(] _i_ 1 _[k]_ [)] _[, . . ., µ]_ _iM_ [(] _[k]_ [)] []] [representing] [the] [selected]
item at the _k_ -th sampling step is achieved by taking softmax over adjusted keys with temperature _τ_ :


exp( _αij_ [(] _[k]_ [)] _[/τ]_ [)]
_µ_ [(] _ij_ _[k]_ [)] [=]             - _M_ (12)
_k_ =1 [exp(] _[α]_ _ik_ [(] _[k]_ [)] _[/τ]_ [)]


After _N_ sampling steps, a set of soft one-hot vectors representing selected weights is attained. By
summing up these vectors, a relaxation of the _N_ -hot pruning mask can be constructed, enabling
gradient-based training.


3.2.3 TEMPERATURE ANNEALING


The temperature _τ_ is mentioned as a hyperparameter controlling the hardness of one-hot approximations. Additionally, we define a hyperparameter _λ_, which regulates the degree of randomness in the
sampling process. Subsequently, the Gumbel-Top- _K_ trick is applied to the scaled logits, denoted as
_**ϕ**_ _i_ = _**ϕ**_ _i/λ_ . In our experiments, we implement an annealing schedule for _τ_ and _λ_ to guide the mask
learning process, beginning with high randomness to promote solution exploration and converging
to a small set of optimal solutions by training’s end. We adopt a linear annealing schedule, where at
the _t_ -th training step, the temperatures are defined as follows:


_τt_ = _τ_ init _×_ (1 _−_ _[t]_


_[t]_ _[t]_

_T_ [) +] _[ τ]_ [end] _[ ×]_ _T_


_[t]_ _λt_ = _λ_ init _×_ (1 _−_ _[t]_

_T_ [;] _T_


_[t]_ _[t]_

_T_ [) +] _[ λ]_ [end] _[ ×]_ _T_


(13)
_T_


5


Table 1: Comparative evaluation of zero-shot accuracy across multiple benchmark datasets for various pruning methods applied to OPT models of different sizes with 2:4 sparsity pattern. Bold values
denote the highest performance in each metric. The column ’W/U’ indicates whether weight updates
are applied during pruning.


**Method** **W/U** **ARC-C** **ARC-E** **HellaS.** **PIQA** **RACE** **SciQ** **Average**


**Base Model:** **OPT-125M** - 19.03 43.52 29.19 62.95 30.05 75.20 43.32
Magnitude ✗ 17.66 32.28 27.14 57.67 22.78 44.00 33.59
Wanda (Sun et al., 2024) ✗ 18.69 36.03 27.55 59.09 23.54 64.70 38.27
SparseGPT (Frantar & Alistarh, 2023) ✓ **19.71** 38.09 **27.60** 59.74 25.55 69.00 39.95
MaskLLM (Fang et al., 2024) ✗ 18.34 39.73 27.50 61.26 26.79 **70.30** 40.65
**SUSI** (Ours) ✗ 19.02 **40.19** 27.33 **61.97** **28.04** 69.80 **41.06**


**Base Model:** **OPT-350M** - 20.82 44.02 32.02 64.58 29.95 74.90 44.38
Magnitude ✗ 16.72 31.52 27.09 57.40 22.87 51.30 34.48
Wanda (Sun et al., 2024) ✗ **19.71** 34.64 **28.76** 60.34 26.79 64.70 39.16
SparseGPT (Frantar & Alistarh, 2023) ✓ 18.52 34.89 28.43 59.58 26.89 **66.60** 39.15
MaskLLM (Fang et al., 2024) ✗ 18.26 37.71 27.52 **61.48** 26.99 **66.60** 39.76
**SUSI** (Ours) ✗ 18.17 **38.42** 27.85 61.15 **27.85** 66.20 **39.94**


**Base Model:** **OPT-1.3B** - 23.29 57.03 41.54 71.76 34.16 84.30 52.01
Magnitude ✗ 17.83 39.31 31.15 61.81 26.22 65.40 40.29
Wanda (Sun et al., 2024) ✗ 20.82 47.60 33.85 65.72 30.53 **79.40** 46.32
SparseGPT (Frantar & Alistarh, 2023) ✓ **21.93** 45.62 **34.19** 63.98 32.06 78.90 46.11
MaskLLM (Fang et al., 2024) ✗ 19.53 47.39 33.29 **66.76** 31.87 76.40 45.87
**SUSI** (Ours) ✗ 21.67 **47.68** 33.50 66.70 **32.15** 77.20 **46.48**


4 EXPERIMENT


4.1 EXPERIMENTAL SETTING


The proposed method is evaluated on three OPT models (Zhang et al., 2022) of increasing sizes
(e.g., 125M, 350M, and 1.3B parameters) to assess its stability and scalability under semi-structured
pruning. All main experiments adopt a 2:4 sparsity pattern, compatible with NVIDIA Ampere
hardware. The detailed hyperparameters are listed in the Appendix A.3. Training runs for 2,000
steps with a batch size of 256 and a sequence length of 2048, processing 1B tokens in total. To
ensure robust generalization, training data is collected on 1B tokens sampled from the C4 corpus
(Raffel et al., 2020), a cleaned English dataset aligned with OPT’s pretraining data.


To assess the effectiveness of the proposed approach, four representative semi-structured pruning
methods are selected, covering a range of popular strategies from classical to recent advancements:
i) **Magnitude** is a simple, data-free method that removes parameters with the smallest absolute values. While this method is easy to implement, it often yields subpar results due to the limitations of
parameter sensitivity and model dynamics; ii) **Wanda** (Sun et al., 2024) combines parameter magnitudes with activation statistics at each layer, achieving better performance than pure magnitude
pruning, especially at higher sparsity, while maintaining computational efficiency; iii) **SparseGPT**
(Frantar & Alistarh, 2023) incorporates activation outputs and Hessian information to estimate parameter importance, followed by parameter updates to reduce output error further. This method
yields high accuracy but is more computationally demanding; and iv) **MaskLLM** (Fang et al., 2024)
is quite similar to the proposed method in this study, by learning pruning masks with minimizing
calibration loss under an N:M sparsity constraint, modeled via a multinomial distribution. It delivers
strong performance across benchmarks but suffers from high computational cost.


4.2 MAIN RESULTS


Table 1 presents zero-shot accuracy results across various benchmark datasets. SUSI consistently
achieves the highest or near-highest average accuracy across all evaluated OPT models (41.06%
for OPT-125M, 39.94% for OPT-350M, and 46.48% for OPT-1.3B), outperforming baselines such
as Magnitude, Wanda, SparseGPT, and MaskLLM. Notably, SUSI exhibits a minimal performance
drop compared to unpruned models (e.g., 19.02% vs. 19.03% on ARC-C for OPT-125M), highlighting its ability to preserve model quality through differentiable subset sampling. This advantage over
heuristic-based methods like Magnitude becomes more pronounced as model size increases (e.g.,


6


46.48% for SUSI vs. 45.87% for MaskLLM on OPT-1.3B), highlighting its scalability. Additionally,
SUSI shows strong performance across diverse tasks (e.g., 27.85% on RACE for OPT-350M), effectively balancing sparsity and accuracy while maintaining lower computational overhead compared
to MaskLLM. Table 2 reports the PPL performance on WikiText-2, highlighting the advantages of
the proposed method as follows:


**(i)** **Effectiveness** **of** **Differentiable** **Subset** **Sampling:** SUSI consistently outperforms other baselines across all model scales, suggesting that the proposed differentiable subset sampling mechanism effectively learns performant sparsity patterns, better preserving model quality post-pruning.


**(iii) Robustness of Learnable Mask Approaches:** While Traditional magnitude pruning performs
poorly (e.g., 655.87 PPL on OPT-350M), reaffirming that naive pruning strategies significantly degrade language modeling performance. The promising performance of SUSI and MaskLLM emphasizes the importance of structured and learnable pruning mechanisms.


4.3 DETAILED ANALYSIS


4.3.1 EFFICIENT TRAINING


Figure 3 presents a comparative analysis of the parameter efficiency and data efficiency achieved
by the proposed SUSI method under both 2:4 and 2:8 sparsity settings. Figure 3(a) reports the
number of trainable parameters across multiple OPT model sizes. Under the 2:4 pattern, SUSI consistently requires about 1 _._ 5 _×_ fewer parameters than MaskLLM, effectively lowering optimization
costs. More importantly, the advantage of SUSI becomes even more evident in the 2:8 setting: while
MaskLLM requires up to 4 _._ 2B parameters for OPT-1.3B, SUSI reduces this to 1 _._ 2B, achieving a
3 _._ 5 _×_ reduction. Such parameter efficiency directly translates into substantial computational and
memory savings, which is critical for deployment in resource-constrained environments. Sequentially, Figure 3(b) reports the perplexity on WikiText-2 as a function of the number of training tokens
for the OPT-350M model. Under the 2:4 pattern, SUSI consistently achieves lower perplexity than
MaskLLM across all token budgets, demonstrating superior data efficiency. In the 2:8 pattern, although the number of trainable parameters is drastically reduced compared to 2:4 (Figure 3a), SUSI
maintains competitive perplexity with MaskLLM, reaching 144 _._ 9 at 1B tokens. These results highlight the robustness of SUSI: it not only improves parameter efficiency but also sustains competitive
modeling performance under more aggressive sparsity constraints. The detailed performance of 2:8
sparsity patterns is shown in the Appendix A.7.


4.3.2 ABLATION STUDY


Figure 4 summarizes the ablation study on the proposed SUSI model, focusing on the contributions
of two critical design choices: (i) the power term _p_, which amplifies the effect of removing a selected
weight (Equation 11); and (ii) the temperature annealing schedule that gradually sharpens the sampling distribution (Equation 13). Figure 4(a) illustrates the training loss trajectories under different
configurations. Without the power term ( _p_ = 1 _._ 0), convergence is noticeably slower and less stable,
with higher final loss compared to _p_ = 3 _._ 0. Increasing _p_ strengthens the penalization on selected
weights, which accelerates convergence and consistently lowers the final loss, suggesting that this


7


Furthermore, the small gap between
the perplexity of the pruned SUSI
models and the unpruned baseline indicates that SUSI maintains competitive performance even under aggressive pruning settings.


**w/o Pruning** 31.95 25.42 16.41
Magnitude 407.66 655.87 245.75
Wanda 92.50 134.26 34.09
SparseGPT 72.80 61.23 29.27
MaskLLM 50.91 55.86 28.56
**SUSI** (Ours) **50.24** **54.14** **28.05**


Table 2: Perplexity scores on WikiText-2.


**Method** **OPT-125M** **OPT-350M** **OPT-1.3B**


**(ii)** **Scalability** **across** **Model** **Sizes:**

Wanda 92.50

SUSI demonstrates consistent im
SparseGPT 72.80

provements across increasing model

MaskLLM 50.91

scales, showing especially strong re
**SUSI** (Ours) **50.24**

sults for medium (OPT-350M) and
large (OPT-1.3B) models. This indicates that the method generalizes well
and maintains scalability, which is often a limitation of recent pruning methods.


Figure 3: Comparison of sparsity and perplexity performance: (a) Learnable parameter counts under
the 2:4 and 2:8 sparsity settings across multiple OPT model sizes; and (b) Perplexity versus number
of training tokens on Wikitext-2 for the OPT-350M model.


(a) (b)


Figure 4: Comparison of training dynamics and ablation results: (a) shows the training loss convergence across different configurations (with/without _p_ and annealing). (b) ablation study on OPT350M showing PPL (log-scale) and average accuracy.


mechanism facilitates escaping suboptimal mask distributions. On the other hand, removing the
annealing mechanism leads to rapid divergence, underscoring the necessity of temperature scheduling for maintaining a stable optimization process. Figure 4(b) reports the downstream performance
on OPT-350M in terms of perplexity (log scale) and average zero-shot accuracy. As _p_ increases
from 1 _._ 0 to 3 _._ 0, perplexity drops dramatically (from 998 _._ 33 to 28 _._ 05) and accuracy improves significantly (from 33 _._ 82% to 39 _._ 94%), validating the importance of the power term for effective mask
learning. In contrast, disabling annealing results in infinite perplexity and a severe accuracy drop
(27 _._ 07%), highlighting that annealing is indispensable for stable training and generalization. The
results demonstrate that both components are synergistic: the power term enhances selection sharpness, while annealing ensures convergence stability, which improves the performance.


4.3.3 ROBUSTNESS ANALYSIS


To further evaluate the stability and robustness of SUSI, we trained the variational mask parameters
using three distinct random seeds (42, 123, 1812) and measured the overlap of learned pruning masks
across key layers. Figure 5 reports the probability of mask overlap between runs for representative
modules such as self ~~a~~ ttn.q ~~p~~ roj, self ~~a~~ ttn.k ~~p~~ roj, and mlp.up ~~p~~ roj. Accordingly, the learned pruning
masks show high overlap across seeds (e.g., 0.88 for q ~~p~~ roj, 0.83 for k ~~p~~ roj, 0.94 for mlp.up ~~p~~ roj),
and downstream performance varies by less than 0.5%. These results confirm that SUSI consistently
converges to similar sparsity patterns with minimal variation across initializations, demonstrating
strong robustness and reproducibility.


8


7.5


7.0


6.5


6.0


5.5


5.0


4.5


4.0


seed=42 seed=123 seed=1812


self_attn.q_proj
P(overlap)=0.88


self_attn.k_proj
P(overlap)=0.83


mlp.up_proj
P(overlap)=0.94


Figure 5: The learned masks from the query projection, key projection, and MLP up-projection in
the first transformer block exhibit high similarity across different random seeds.


4.4 RELATED WORKS


Pruning LLMs is a critical optimization technique that removes less significant or redundant parameters, such as weights or neurons, from the neural network architecture. This process reduces model
size, computational complexity, and memory requirements, thereby improving inference speed and
enabling deployment on resource-constrained devices. Pruning methods for LLMs are broadly categorized into structured, unstructured, and semi-structured approaches, each with distinct characteristics and trade-offs (Cheng et al., 2024).


**Structured** **pruning** involves the elimination of entire architectural components, such as layers
or attention heads, to improve computational efficiency (Ashkboos et al., 2024; Xia et al., 2024;
An et al., 2024; Liu et al., 2025a; Le et al., 2025). This approach simplifies the model structure,
making it more amenable to hardware optimization. However, it frequently results in substantial
performance degradation, necessitating extensive retraining to restore model functionality.


**Unstructured** **pruning** targets individual weights based on their significance, enabling high performance even at elevated sparsity levels (Dong et al., 2024; Sun et al., 2024). Despite its efficacy
in preserving model accuracy, the irregular sparsity patterns produced are often incompatible with
hardware acceleration, limiting its practical applicability in deployment scenarios.


**Semi-structured** **pruning** has emerged as a promising approach, striking a balance between the
benefits of structured and unstructured methods. By enforcing regular sparsity patterns, such as
_N_ : _M_ sparsity, this technique optimizes models for hardware acceleration while maintaining performance (Hubara et al., 2021). Methods like SparseGPT (Frantar & Alistarh, 2023) and Wanda (Sun
et al., 2024) employ training-free pruning, achieving efficiency without retraining. More recent
methods, such as MaskLLM (Fang et al., 2024) and AST (Huang et al., 2025), focus on retraining
sparse LLMs, which achieve promising performances while maintaining hardware compatibility.
Nonetheless, the significant computational overhead associated with the number of trainable parameters remains a critical challenge, warranting further investigation. Building on this foundation, our
proposed method leverages weighted reservoir sampling to enhance semi-structured pruning with
_N_ : _M_ sparsity, aiming to enable the retraining of semi-structured sparse LLM with minimal training
costs.


5 CONCLUSION


This study introduced SUSI, a novel semi-structured pruning technique for LLMs, utilizing differentiable subset sampling to efficiently derive N:M sparsity masks. Compared to existing methods,
SUSI reduces the number of trainable parameters and associated memory overhead while maintaining strong performance. Experiments on OPT models (125M, 350M, 1.3B parameters) show that
SUSI outperforms existing methods in perplexity on the Wikitext-2 dataset and maintains competitive zero-shot accuracy across a range of benchmarks. Additionally, SUSI exhibits enhanced data
efficiency and scalability as calibration data increases. These results establish SUSI as a promising
solution for compressing LLMs, effectively balancing performance retention with the demands of
resource-constrained deployment environments.


9


REPRODUCIBILITY STATEMENT


We have taken several steps to ensure the reproducibility of our work:


**Datasets** . All training and calibration data used in our experiments are publicly available. We
follow prior work by sampling 1B tokens from the cleaned English portion of the C4 corpus for
calibration and training, ensuring alignment with OPT’s pretraining distribution. For evaluation, we
employ well-known open-source benchmarks, including WikiText-2 for perplexity evaluation and
ARC (Easy/Challenge), HellaSwag, PIQA, SciQ, and RACE for zero-shot task accuracy. Dataset
statistics and details are provided in Appendix A.1 to facilitate replication.


**Code and Implementation** . We provide an anonymous, fully reproducible implementation of SUSI,
including (i) training scripts for variational mask optimization, (ii) hyperparameter configurations
(see Appendix A.2), and (iii) evaluation scripts leveraging the LM-Evaluation-Harness toolkit. All
results reported in this paper can be reproduced using the provided codebase.


**Availability** . To encourage transparency and facilitate verification of our findings, we submit
the source code and experiment configuration files as supplementary material. An anonymous
and reproducible version of the repository can be accessed at the following link: [https://](https://anonymous.4open.science/r/susi-2E2C)
[anonymous.4open.science/r/susi-2E2C.](https://anonymous.4open.science/r/susi-2E2C)


This repository contains all necessary scripts, instructions, and environment configuration files (including requirements.txt) for reproducing our results end-to-end on standard hardware.


REFERENCES


Yongqi An, Xu Zhao, Tao Yu, Ming Tang, and Jinqiao Wang. Fluctuation-based adaptive structured
pruning for large language models. In Michael J. Wooldridge, Jennifer G. Dy, and Sriraam Natarajan (eds.), _Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Con-_
_ference_ _on_ _Innovative_ _Applications_ _of_ _Artificial_ _Intelligence,_ _IAAI_ _2024,_ _Fourteenth_ _Symposium_
_on_ _Educational_ _Advances_ _in_ _Artificial_ _Intelligence,_ _EAAI_ _2014,_ _February_ _20-27,_ _2024,_ _Vancou-_
_ver,_ _Canada_, pp. 10865–10873. AAAI Press, 2024. doi: 10.1609/AAAI.V38I10.28960. URL
[https://doi.org/10.1609/aaai.v38i10.28960.](https://doi.org/10.1609/aaai.v38i10.28960)


Saleh Ashkboos, Maximilian L. Croci, Marcelo Gennari Do Nascimento, Torsten Hoefler, and
James Hensman. Slicegpt: Compress large language models by deleting rows and columns. In
_The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria,_
_May_ _7-11,_ _2024_ . OpenReview.net, 2024. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=vXxardq6db)
[vXxardq6db.](https://openreview.net/forum?id=vXxardq6db)


Thomas Bird, Julius Kunze, and David Barber. Stochastic variational optimization. _CoRR_,
abs/1809.04855, 2018. [URL http://arxiv.org/abs/1809.04855.](http://arxiv.org/abs/1809.04855)


Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. PIQA: reasoning
about physical commonsense in natural language. In _The_ _Thirty-Fourth_ _AAAI_ _Conference_ _on_
_Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intel-_
_ligence_ _Conference,_ _IAAI_ _2020,_ _The Tenth AAAI Symposium on Educational Advances in Artifi-_
_cial_ _Intelligence,_ _EAAI_ _2020,_ _New_ _York,_ _NY,_ _USA,_ _February_ _7-12,_ _2020_, pp. 7432–7439. AAAI
Press, 2020. doi: 10.1609/AAAI.V34I05.6239. [URL https://doi.org/10.1609/aaai.](https://doi.org/10.1609/aaai.v34i05.6239)
[v34i05.6239.](https://doi.org/10.1609/aaai.v34i05.6239)


Hongrong Cheng, Miao Zhang, and Javen Qinfeng Shi. A survey on deep neural network pruning:
Taxonomy, comparison, analysis, and recommendations. _IEEE Trans. Pattern Anal. Mach. Intell._,
46(12):10558–10578, 2024. doi: 10.1109/TPAMI.2024.3447085. [URL https://doi.org/](https://doi.org/10.1109/TPAMI.2024.3447085)
[10.1109/TPAMI.2024.3447085.](https://doi.org/10.1109/TPAMI.2024.3447085)


Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the AI2 reasoning challenge.
_CoRR_, abs/1803.05457, 2018. [URL http://arxiv.org/abs/1803.05457.](http://arxiv.org/abs/1803.05457)


Peijie Dong, Lujun Li, Zhenheng Tang, Xiang Liu, Xinglin Pan, Qiang Wang, and Xiaowen
Chu. Pruner-zero: Evolving symbolic pruning metric from scratch for large language models. In _Forty-first International Conference on Machine Learning,_ _ICML 2024,_ _Vienna,_ _Austria,_


10


_July 21-27,_ _2024_ . OpenReview.net, 2024. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=1tRLxQzdep)
[1tRLxQzdep.](https://openreview.net/forum?id=1tRLxQzdep)


Pavlos S. Efraimidis and Paul G. Spirakis. Weighted random sampling with a reservoir. _Inf. Process._
_Lett._, 97(5):181–185, 2006. doi: 10.1016/J.IPL.2005.11.003. [URL https://doi.org/10.](https://doi.org/10.1016/j.ipl.2005.11.003)
[1016/j.ipl.2005.11.003.](https://doi.org/10.1016/j.ipl.2005.11.003)


Kazuki Egashira, Mark Vero, Robin Staab, Jingxuan He, and Martin T. Vechev. Exploiting LLM quantization. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), _Advances_ _in_
_Neural_ _Information_ _Processing_ _Systems_ _38:_ _Annual_ _Conference_ _on_ _Neural_ _Information_
_Processing_ _Systems_ _2024,_ _NeurIPS_ _2024,_ _Vancouver,_ _BC,_ _Canada,_ _December_ _10_ _-_ _15,_
_2024_, 2024. URL [http://papers.nips.cc/paper_files/paper/2024/hash/](http://papers.nips.cc/paper_files/paper/2024/hash/496720b3c860111b95ac8634349dcc88-Abstract-Conference.html)
[496720b3c860111b95ac8634349dcc88-Abstract-Conference.html.](http://papers.nips.cc/paper_files/paper/2024/hash/496720b3c860111b95ac8634349dcc88-Abstract-Conference.html)


Gongfan Fang, Hongxu Yin, Saurav Muralidharan, Greg Heinrich, Jeff Pool, Jan Kautz,
Pavlo Molchanov, and Xinchao Wang. Maskllm: Learnable semi-structured sparsity for
large language models. In Amir Globersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang (eds.), _Advances_ _in_
_Neural_ _Information_ _Processing_ _Systems_ _38:_ _Annual_ _Conference_ _on_ _Neural_ _Information_
_Processing_ _Systems_ _2024,_ _NeurIPS_ _2024,_ _Vancouver,_ _BC,_ _Canada,_ _December_ _10_ _-_ _15,_
_2024_, 2024. URL [http://papers.nips.cc/paper_files/paper/2024/hash/](http://papers.nips.cc/paper_files/paper/2024/hash/0e9a05f5ce62284c91e4a33498899124-Abstract-Conference.html)
[0e9a05f5ce62284c91e4a33498899124-Abstract-Conference.html.](http://papers.nips.cc/paper_files/paper/2024/hash/0e9a05f5ce62284c91e4a33498899124-Abstract-Conference.html)


Elias Frantar and Dan Alistarh. Sparsegpt: Massive language models can be accurately pruned
in one-shot. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan
Sabato, and Jonathan Scarlett (eds.), _International Conference on Machine Learning, ICML 2023,_
_23-29_ _July_ _2023,_ _Honolulu,_ _Hawaii,_ _USA_, volume 202 of _Proceedings_ _of_ _Machine_ _Learning_
_Research_, pp. 10323–10337. PMLR, 2023. URL [https://proceedings.mlr.press/](https://proceedings.mlr.press/v202/frantar23a.html)
[v202/frantar23a.html.](https://proceedings.mlr.press/v202/frantar23a.html)


Leo Gao, Jonathan Tow, Baber Abbasi, Stella Biderman, Sid Black, Anthony DiPofi, Charles Foster, Laurence Golding, Jeffrey Hsu, Alain Le Noac’h, Haonan Li, Kyle McDonell, Niklas Muennighoff, Chris Ociepa, Jason Phang, Laria Reynolds, Hailey Schoelkopf, Aviya Skowron, Lintang
Sutawika, Eric Tang, Anish Thite, Ben Wang, Kevin Wang, and Andy Zou. The language model
evaluation harness, 07 2024. [URL https://zenodo.org/records/12608602.](https://zenodo.org/records/12608602)


Emil Julius Gumbel. _Statistical theory of extreme values and some practical applications:_ _a series_
_of lectures_, volume 33. US Government Printing Office, 1954.


Weiyu Huang, Yuezhou Hu, Guohao Jian, Jun Zhu, and Jianfei Chen. Pruning large language models
with semi-structural adaptive sparse training. In Toby Walsh, Julie Shah, and Zico Kolter (eds.),
_AAAI-25,_ _Sponsored by the Association for the Advancement of Artificial Intelligence,_ _February_
_25 - March 4, 2025, Philadelphia, PA, USA_, pp. 24167–24175. AAAI Press, 2025. doi: 10.1609/
AAAI.V39I23.34592. [URL https://doi.org/10.1609/aaai.v39i23.34592.](https://doi.org/10.1609/aaai.v39i23.34592)


Itay Hubara, Brian Chmiel, Moshe Island, Ron Banner, Joseph Naor, and Daniel Soudry. Accelerated sparse neural training: A provable and efficient method to find N: M transposable masks. In
Marc’Aurelio Ranzato, Alina Beygelzimer, Yann N. Dauphin, Percy Liang, and Jennifer Wortman Vaughan (eds.), _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _34:_ _Annual_ _Confer-_
_ence_ _on_ _Neural_ _Information_ _Processing_ _Systems_ _2021,_ _NeurIPS_ _2021,_ _December_ _6-14,_ _2021,_
_virtual_, pp. 21099–21111, 2021. URL [https://proceedings.neurips.cc/paper/](https://proceedings.neurips.cc/paper/2021/hash/b0490b85e92b64dbb5db76bf8fca6a82-Abstract.html)
[2021/hash/b0490b85e92b64dbb5db76bf8fca6a82-Abstract.html.](https://proceedings.neurips.cc/paper/2021/hash/b0490b85e92b64dbb5db76bf8fca6a82-Abstract.html)


Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with gumbel-softmax. In _5th_
_International Conference on Learning Representations, ICLR 2017, Toulon, France, April 24-26,_
_2017, Conference Track Proceedings_ . OpenReview.net, 2017. [URL https://openreview.](https://openreview.net/forum?id=rkE3y85ee)
[net/forum?id=rkE3y85ee.](https://openreview.net/forum?id=rkE3y85ee)


Guokun Lai, Qizhe Xie, Hanxiao Liu, Yiming Yang, and Eduard H. Hovy. RACE: large-scale
reading comprehension dataset from examinations. In Martha Palmer, Rebecca Hwa, and Sebastian Riedel (eds.), _Proceedings_ _of_ _the_ _2017_ _Conference_ _on_ _Empirical_ _Methods_ _in_ _Natural_


11


_Language_ _Processing,_ _EMNLP_ _2017,_ _Copenhagen,_ _Denmark,_ _September_ _9-11,_ _2017_, pp. 785–
794. Association for Computational Linguistics, 2017. doi: 10.18653/V1/D17-1082. URL
[https://doi.org/10.18653/v1/d17-1082.](https://doi.org/10.18653/v1/d17-1082)


Qi Le, Enmao Diao, Ziyan Wang, Xinran Wang, Jie Ding, Li Yang, and Ali Anwar. Probe pruning:
Accelerating llms through dynamic pruning via model-probing. In _The Thirteenth International_
_Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2025,_ _Singapore,_ _April_ _24-28,_ _2025_ . OpenReview.net, 2025. [URL https://openreview.net/forum?id=WOt1owGfuN.](https://openreview.net/forum?id=WOt1owGfuN)


Yijiang Liu, Huanrui Yang, Youxin Chen, Rongyu Zhang, Miao Wang, Yuan Du, and Li Du. PAT:
pruning-aware tuning for large language models. In Toby Walsh, Julie Shah, and Zico Kolter
(eds.), _AAAI-25,_ _Sponsored_ _by_ _the_ _Association_ _for_ _the_ _Advancement_ _of_ _Artificial_ _Intelligence,_
_February_ _25_ _-_ _March_ _4,_ _2025,_ _Philadelphia,_ _PA,_ _USA_, pp. 24686–24695. AAAI Press, 2025a.
doi: 10.1609/AAAI.V39I23.34649. URL [https://doi.org/10.1609/aaai.v39i23.](https://doi.org/10.1609/aaai.v39i23.34649)
[34649.](https://doi.org/10.1609/aaai.v39i23.34649)


Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. Spinquant: LLM
quantization with learned rotations. In _The_ _Thirteenth_ _International_ _Conference_ _on_ _Learning_
_Representations,_ _ICLR_ _2025,_ _Singapore,_ _April_ _24-28,_ _2025_ . OpenReview.net, 2025b. URL
[https://openreview.net/forum?id=ogO6DGE6FZ.](https://openreview.net/forum?id=ogO6DGE6FZ)


Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. Pointer sentinel mixture
models. In _5th_ _International_ _Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2017,_ _Toulon,_
_France, April 24-26, 2017, Conference Track Proceedings_ [. OpenReview.net, 2017. URL https:](https://openreview.net/forum?id=Byj72udxe)
[//openreview.net/forum?id=Byj72udxe.](https://openreview.net/forum?id=Byj72udxe)


Juan Pablo Mu˜noz, Jinjie Yuan, and Nilesh Jain. Mamba-shedder: Post-transformer compression
for efficient selective structured state space models. In Luis Chiruzzo, Alan Ritter, and Lu Wang
(eds.), _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Associ-_
_ation for Computational Linguistics:_ _Human Language Technologies, NAACL 2025 - Volume 1:_
_Long Papers,_ _Albuquerque,_ _New Mexico, USA, April 29 - May 4, 2025_, pp. 3851–3863. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.NAACL-LONG.195. URL
[https://doi.org/10.18653/v1/2025.naacl-long.195.](https://doi.org/10.18653/v1/2025.naacl-long.195)


Tobias Pl¨otz and Stefan Roth. Neural nearest neighbors networks. In Samy Bengio, Hanna M.
Wallach, Hugo Larochelle, Kristen Grauman, Nicol`o Cesa-Bianchi, and Roman Garnett (eds.),
_Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ _31:_ _Annual_ _Conference_ _on_ _Neural_ _Infor-_
_mation_ _Processing_ _Systems_ _2018,_ _NeurIPS_ _2018,_ _December_ _3-8,_ _2018,_ _Montr´eal,_ _Canada_, pp.
1095–1106, 2018. URL [https://proceedings.neurips.cc/paper/2018/hash/](https://proceedings.neurips.cc/paper/2018/hash/f0e52b27a7a5d6a1a87373dffa53dbe5-Abstract.html)
[f0e52b27a7a5d6a1a87373dffa53dbe5-Abstract.html.](https://proceedings.neurips.cc/paper/2018/hash/f0e52b27a7a5d6a1a87373dffa53dbe5-Abstract.html)


Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi
Zhou, Wei Li, and Peter J. Liu. Exploring the limits of transfer learning with a unified text-totext transformer. _J. Mach. Learn. Res._, 21:140:1–140:67, 2020. [URL https://jmlr.org/](https://jmlr.org/papers/v21/20-074.html)
[papers/v21/20-074.html.](https://jmlr.org/papers/v21/20-074.html)


Mingjie Sun, Zhuang Liu, Anna Bair, and J. Zico Kolter. A simple and effective pruning approach for large language models. In _The_ _Twelfth_ _International_ _Conference_ _on_ _Learning_ _Rep-_
_resentations,_ _ICLR_ _2024,_ _Vienna,_ _Austria,_ _May_ _7-11,_ _2024_ . OpenReview.net, 2024. URL
[https://openreview.net/forum?id=PxoFut3dWW.](https://openreview.net/forum?id=PxoFut3dWW)


Jeffrey Scott Vitter. Random sampling with a reservoir. _ACM_ _Trans._ _Math._ _Softw._, 11(1):37–57,
1985. doi: 10.1145/3147.3165. [URL https://doi.org/10.1145/3147.3165.](https://doi.org/10.1145/3147.3165)


Zhongwei Wan, Xin Wang, Che Liu, Samiul Alam, Yu Zheng, Jiachen Liu, Zhongnan Qu, Shen Yan,
Yi Zhu, Quanlu Zhang, Mosharaf Chowdhury, and Mi Zhang. Efficient large language models: A
survey. _Trans. Mach. Learn. Res._, 2024, 2024. [URL https://openreview.net/forum?](https://openreview.net/forum?id=bsCCJHbO8A)
[id=bsCCJHbO8A.](https://openreview.net/forum?id=bsCCJHbO8A)


Johannes Welbl, Nelson F. Liu, and Matt Gardner. Crowdsourcing multiple choice science questions. In Leon Derczynski, Wei Xu, Alan Ritter, and Tim Baldwin (eds.), _Proceedings_ _of_


12


_the_ _3rd_ _Workshop_ _on_ _Noisy_ _User-generated_ _Text,_ _NUT@EMNLP_ _2017,_ _Copenhagen,_ _Den-_
_mark,_ _September_ _7,_ _2017_, pp. 94–106. Association for Computational Linguistics, 2017. doi:
10.18653/V1/W17-4413. [URL https://doi.org/10.18653/v1/w17-4413.](https://doi.org/10.18653/v1/w17-4413)


Miles Williams and Nikolaos Aletras. On the impact of calibration data in post-training quantization
and pruning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), _Proceedings of the 62nd_
_Annual Meeting of the Association for Computational Linguistics (Volume 1:_ _Long Papers), ACL_
_2024, Bangkok, Thailand, August 11-16, 2024_, pp. 10100–10118. Association for Computational
Linguistics, 2024. doi: 10.18653/V1/2024.ACL-LONG.544. [URL https://doi.org/10.](https://doi.org/10.18653/v1/2024.acl-long.544)
[18653/v1/2024.acl-long.544.](https://doi.org/10.18653/v1/2024.acl-long.544)


Mengzhou Xia, Tianyu Gao, Zhiyuan Zeng, and Danqi Chen. Sheared llama: Accelerating language
model pre-training via structured pruning. In _The Twelfth International Conference on Learning_
_Representations,_ _ICLR_ _2024,_ _Vienna,_ _Austria,_ _May_ _7-11,_ _2024_ . OpenReview.net, 2024. URL
[https://openreview.net/forum?id=09iOdaeOzp.](https://openreview.net/forum?id=09iOdaeOzp)


Sang Michael Xie and Stefano Ermon. Reparameterizable subset sampling via continuous relaxations. In Sarit Kraus (ed.), _Proceedings_ _of_ _the_ _Twenty-Eighth_ _International_ _Joint_ _Confer-_
_ence on Artificial Intelligence, IJCAI 2019, Macao, China, August 10-16, 2019_, pp. 3919–3925.
ijcai.org, 2019. doi: 10.24963/IJCAI.2019/544. URL [https://doi.org/10.24963/](https://doi.org/10.24963/ijcai.2019/544)
[ijcai.2019/544.](https://doi.org/10.24963/ijcai.2019/544)


Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine really finish your sentence? In Anna Korhonen, David R. Traum, and Llu´ıs M`arquez
(eds.), _Proceedings_ _of_ _the_ _57th_ _Conference_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics,_
_ACL_ _2019,_ _Florence,_ _Italy,_ _July_ _28-_ _August_ _2,_ _2019,_ _Volume_ _1:_ _Long_ _Papers_, pp. 4791–
4800. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1472. URL
[https://doi.org/10.18653/v1/p19-1472.](https://doi.org/10.18653/v1/p19-1472)


Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona T. Diab, Xian Li, Xi Victoria Lin, Todor Mihaylov, Myle Ott, Sam Shleifer,
Kurt Shuster, Daniel Simig, Punit Singh Koura, Anjali Sridhar, Tianlu Wang, and Luke Zettlemoyer. OPT: open pre-trained transformer language models. _CoRR_, abs/2205.01068, 2022.
doi: 10.48550/ARXIV.2205.01068. URL [https://doi.org/10.48550/arXiv.2205.](https://doi.org/10.48550/arXiv.2205.01068)
[01068.](https://doi.org/10.48550/arXiv.2205.01068)


A APPENDIX


A.1 WRS YIELDS EQUIVALENT VARIATIONAL OBJECTIVE


**Theorem 1.** _Let P_ ( **m** _i|_ _**ϕ**_ _i_ ) _be the exact distribution of each mask_ **m** _i defined as in Equation 8._ _The_
_expected loss when sampling each mask from its exact distribution is equivalent to the expected loss_
_obtained_ _when_ _each_ _mask_ _is_ _parameterized_ _as_ _a_ _sum_ _of_ _elements_ _in_ _an_ _ordered_ _subset_ _Si_ _sampled_
_from the corresponding restricted distribution PWRS_ ( _Si|_ _**ϕ**_ _i_ ) _._


_Proof._ Without loss of generality, we prove the following terms are equivalent:


where _f_ ( **m** ) is an objective function depending on **m**, _P_ ( **m** _|_ _**ϕ**_ ) = [�] _S_ **m** _[P]_ [WRS][(] _[S]_ **[m]** _[|]_ _**[ϕ]**_ [)][ is the exact]
distribution with _S_ **m** s are sets that the sum of elements in _S_ **m** equals **m** .


13





 [�]


_**µ**_

_**µ**_ _∈S_








E _P_ ( **m** _|_ _**ϕ**_ )[ _f_ ( **m** )] = E _P_ WRS( _S|_ _**ϕ**_ )





 _f_





 (14)


Given _M_, the set of binary masks satisfying the N:M sparsity, the expected loss when sampling **m**
from the exact distribution is then:


The final expression is precisely the expectation of _f_ under the distribution _P_ WRS( _S|_ _**ϕ**_ ), proving
the claim.


A.2 EVALUATION METRICS AND BENCHMARK DATASETS


Following previous works in this research field, three automated metrics are considered for the evaluation, including both quantitative and qualitative metrics to capture the full impact of pruning: i)
_Task Accuracy (ACC)_ : on common NLP tasks such as question answering in reading comprehension,
mathematics, and science. These tasks are typically assessed in zero-shot or few-shot settings using
benchmark datasets; _Perplexity (PPL)_ : is a standard metric for assessing language model quality. It


Table 3: Statistics of datasets used for zero-shot evaluation.


**Dataset** **Questions** **Task Type**


ARC-Easy 2,376 Multiple-choice science
ARC-Challenge 1,172 Multiple-choice science
HellaSwag 10,042 Sentence completion
PIQA 1,838 Physical interaction QA
RACE 1,045 Multiple-choice comprehension
SciQ 1,000 Multiple-choice science


measures how well the model predicts the next word in a sequence, with lower values indicating
better predictive performance. The benchmark datasets used to assess the effectiveness of pruning methods include WikiText-2 (Merity et al., 2017) for perplexity evaluation and a range of NLP
benchmark datasets for zero-shot evaluation, which cover diverse task types and reasoning requirements, including ARC (Clark et al., 2018), HellaSwag (Zellers et al., 2019), PIQA (Bisk et al.,
2020), SciQ (Welbl et al., 2017), and RACE (Lai et al., 2017). These evaluations are conducted
using the LM-Evaluation-Harness toolkit (Gao et al., 2024).


Table 3 provides a comprehensive summary of the datasets used for zero-shot evaluation across
multiple tasks. These datasets span a range of domains, including commonsense reasoning, science
question answering, and reading comprehension, thereby ensuring a rigorous and diverse assessment
of pruning performance.


A.3 HYPERPARAMETER SETTING


The hyperparameters used for training SUSI are listed in Table 4. These settings were carefully
chosen to balance convergence stability and computational efficiency across all evaluated models.
Specifically, model weights remain frozen during training. The variational distribution is initialized
from a standard normal ( _µ_ = 0 _._ 0, _σ_ = 0 _._ 01), and a simulated annealing process gradually reduces
randomness. Temperatures _τ_ and _λ_ linearly decay from 1.0 to 0.05 and from 1.0 to 0.002, respectively. Optimization uses AdamW-8bit with a learning rate decaying from 1 _×_ 10 _[−]_ [3] to 1 _×_ 10 _[−]_ [4],


14





_f_





- _P_ ( **m** _|_ _**ϕ**_ ) _f_ ( **m** ) = 

**m** _∈M_ **m**


��


   E _P_ ( **m** _|_ _**ϕ**_ )[ _f_ ( **m** )] =


 [�]


_**µ**_

_**µ**_ _∈S_ **m**





**m** _∈M_


_P_ WRS( _S_ **m** _|_ _**ϕ**_ )

_S_ **m**














 [�]


_**µ**_

_**µ**_ _∈S_


 = 


= 

**m** _∈M_


- _P_ WRS( _S_ **m** _|_ _**ϕ**_ ) _f_


_S_ **m**


 [�]


_**µ**_

_**µ**_ _∈S_ **m**


_P_ WRS( _S|_ _**ϕ**_ ) _f_

**m** _∈M_





(15)








_**µ**_

_**µ**_ _∈S_








= E _P_ WRS( _S|_ _**ϕ**_ )





 _f_


 [�]





Table 4: Hyperparameter configuration used in training.


**Parameter** **Values**


Initialization distribution _N_ (0 _,_ 0 _._ 01)
Gumbel-Softmax temperature _τ_ = 1 _._ 0 _→_ 0 _._ 05
Sampling temperature _λ_ = 1 _._ 0 _→_ 0 _._ 002
Weight decay 0 _._ 05
Learning rate 10 _[−]_ [3] _→_ 10 _[−]_ [4]
Strengthening power term _p_ = 3 _._ 0
AdamW parameters _β_ 1 = 0 _._ 9 _, β_ 2 = 0 _._ 95
Batch size 256
Sequence length 2048
Training steps 2000


weight decay of 0.05, and _β_ 1 = 0 _._ 9, _β_ 2 = 0 _._ 95, matching the OPT pretraining setup. The power
term _p_ (Equation 11) is selected from _{_ 1 _,_ 2 _,_ 3 _}_, where _p_ = 1 corresponds to no power-term scaling,
and larger values of _p_ (e.g., _p_ = 2 or 3) progressively emphasize the impact of removing higherimportance elements.


A.4 GUMBEL-TOP-K ALGORITHM


The algorithm for Gumbel-Top-K is illustrated in the Algorithm 1. Specifically, we provide a clear
description of the Gumbel-Top-K sampling procedure employed to enable differentiable mask learning. This formulation allows for efficient sampling of K items without replacement while preserving
differentiability for gradient-based optimization.


**Algorithm 1** Gumbel-Top- _K_ Sampling Algorithm (Differentiable)
**Input** : Set of candidates _X_ = _{x_ 1 _, . . ., xN_ _}_ with corresponding logits _**ϕ**_ = [ _ϕ_ 1 _, . . ., ϕN_ ], number
of samples _K_, temperature _τ_ _>_ 0
**Output** : Soft _K_ -hot selection vector **S** _∈_ R _[N]_

1: **for** _i ←_ 1 to _N_ **do**
2: _ui_ _∼_ Uniform(0 _,_ 1)
3: _gi_ _←−_ log( _−_ log( _ui_ )) _// Sample Gumbel noise_
4: _κi_ _←_ _ϕi_ + _gi_ _// Compute perturbed key_
5: **end for**
6: _**α**_ [(1)] _←_ [ _κ_ 1 _, . . ., κN_ ]
7: **for** _k_ _←_ 1 to _K_ **do**
8: _**µ**_ [(] _[k]_ [)] _←_ softmax� _**α**_ [(] _[k]_ [)] _/τ_ 
9: _**α**_ [(] _[k]_ [+1)] _←_ _**α**_ [(] _[k]_ [)] + log�1 _−_ _**µ**_ [(] _[k]_ [)][�]

10: **end for**
11: **S** _←_ [�] _[K]_ _k_ =1 _**[µ]**_ [(] _[k]_ [)] _// Soft K-hot vector_
12: **return S** =0


A.5 COMPARISON TO STRAIGHT-THROUGH GUMBEL-TOP- _K_


We further examined whether adopting a straight-through (ST) Gumbel-Top-K estimator benefits
pruning performance. In this variant, the forward pass generates discrete masks by directly applying
an argtopK over Gumbel-perturbed logits, while the backward pass propagates gradients through
the continuous Gumbel-softmax relaxation. This strategy enforces discretization earlier in training,
which is able to improve mask interpretability. However, our empirical results in Table 5 show that
ST Gumbel-Top-K leads to slightly inferior performance compared to the pure soft relaxation. For
instance, on OPT-125M, ST achieves 51.20 perplexity and 40.04% average accuracy, while the soft
approach reaches 50.24 perplexity and 41.06% accuracy. Similarly, on OPT-350M, the gap widens
(60.49 vs. 54.14 perplexity). These observations suggest that the bias introduced by the ST estimator
hampers generalization, outweighing the potential benefits of earlier discretization. Overall, the soft


15


Table 5: Comparison of pruning results with 2:4 sparsity, both with and without ST Gumbel-Top- _K_
estimator (denoted as ”w STE” and ”w/o STE”).


**OPT-125M** **OPT-350M**
**Metric** w STE w/o STE (ours) w STE w/o STE (ours)


PPL (↓) 51.20 **50.24** 60.49 **54.14**


Avg. Acc (↑) 40.04 **41.06** 39.11 **39.94**


Gumbel-Top-K relaxation used in SUSI provides a more effective balance between trainability and
performance.


A.6 MASK DIFFERENCE ANALYSIS


To investigate how different pruning strategies select weights, we measure the overlap between
masks produced by various methods on the same model. Figure 6 shows that SUSI’s learned masks
achieve much higher cross-seed similarity (82%) compared to one-shot pruning methods such as
Magnitude (63%), Wanda (66%), and SparseGPT (75%), which produce substantially different sparsity patterns.


Figure 6: Mask difference analysis between SUSI and previous works. Besides the name of each
baseline, place an overlapping percentage indicating the similarity of the produced masks between
that baseline and SUSI.


Interestingly, the mask similarity of SUSI closely matches that of other mask-learning approaches
like MaskLLM, suggesting that iterative mask optimization converges toward a stable and consistent
subset of important weights. Combined with the main results, higher mask similarity is correlated
with better perplexity and zero-shot accuracy, underscoring that stable mask learning plays a key
role in achieving superior downstream performance.


A.7 EXTEND TO OTHER SPARSITY PATTERN


To further examine the generality of SUSI, we extend our evaluation beyond the commonly studied
2:4 configuration. These alternative settings introduce more aggressive pruning constraints and
exacerbate the challenges faced by learnable mask methods such as MaskLLM, whose parameter
overhead grows quadratically. In contrast, SUSI preserves linear complexity in _M_, enabling efficient
scalability to larger group sizes.


16


Table 6: Performance on 2:8 sparsity pattern.

|Method W/U ARC-C ARC-E HellaS. PIQA RACE SciQ Average ↑|PPL ↓|
|---|---|
|**Base Model: OPT-125M**<br>-<br>19.03<br>43.52<br>29.19<br>62.95<br>30.05<br>75.20<br>43.32<br>Magnitude<br>✗<br>**21.25**<br>27.26<br>25.90<br>53.65<br>21.82<br>21.80<br>28.61<br>Wanda<br>✗<br>18.86<br>29.12<br>26.19<br>54.35<br>21.44<br>28.80<br>29.79<br>SparseGPT<br>✓<br>19.88<br>28.28<br>26.43<br>55.01<br>23.73<br>32.40<br>30.96<br>MaskLLM<br>✗<br>18.77<br>**35.19**<br>26.86<br>58.16<br>**23.44**<br>61.20<br>**37.27**<br>**SUSI** (Ours)<br>✗<br>18.17<br>33.80<br>**26.91**<br>**58.27**<br>**23.44**<br>**62.70**<br>37.22|32<br>13431<br>5195<br>986<br><br>110|
|**Base Model: OPT-350M**<br>-<br>20.82<br>44.02<br>32.02<br>64.58<br>29.95<br>74.90<br>44.38<br>Magnitude<br>✗<br>**19.97**<br>28.07<br>26.31<br>53.54<br>22.20<br>30.00<br>30.02<br>Wanda<br>✗<br>18.52<br>27.61<br>26.51<br>53.48<br>22.11<br>29.00<br>29.54<br>SparseGPT<br>✓<br>17.49<br>28.83<br>26.50<br>54.30<br>23.44<br>37.00<br>31.26<br>MaskLLM<br>✗<br>16.55<br>**31.61**<br>26.38<br>**57.51**<br>**24.78**<br>**58.60**<br>**35.91**<br>**SUSI** (Ours)<br>✗<br>16.30<br>29.50<br>**26.61**<br>57.02<br>24.69<br>57.20<br>35.22|25.42<br>9805<br>2956<br>1358<br><br>145|


Table 7: Performance on the 4:8 sparsity pattern. Note that experimenting on MaskLLM could not
be executed on our infrastructure in this setting due to the excessive number of trainable parameters.

|Method W/U ARC-C ARC-E HellaS. PIQA RACE SciQ Average ↑|PPL ↓|
|---|---|
|**Base Model: OPT-125M**<br>-<br>19.03<br>43.52<br>29.19<br>62.95<br>30.05<br>75.20<br>43.32<br>Magnitude<br>✗<br>18.09<br>34.72<br>27.55<br>58.32<br>23.25<br>57.4<br>36.56<br>Wanda<br>✗<br>**19.11**<br>37.42<br>27.74<br>59.85<br>26.22<br>67.10<br>39.57<br>SparseGPT<br>✓<br>18.77<br>39.06<br>**27.94**<br>61.15<br>27.75<br>71.10<br>40.96<br>MaskLLM<br>✗<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**SUSI** (Ours)<br>✗<br>**19.11**<br>**40.36**<br>27.67<br>**62.25**<br>**29.37**<br>**72.30**<br>**41.84**|32<br>205<br>61<br>54<br>-<br>**41**|
|**Base Model: OPT-350M**<br>-<br>20.82<br>44.02<br>32.02<br>64.58<br>29.95<br>74.90<br>44.38<br>Magnitude<br>✗<br>16.81<br>33.12<br>27.74<br>58.32<br>22.78<br>56.90<br>35.95<br>Wanda<br>✗<br>17.83<br>35.86<br>28.81<br>60.83<br>25.17<br>66.30<br>39.13<br>SparseGPT<br>✓<br>**18.52**<br>36.57<br>**29.56**<br>61.32<br>27.85<br>**69.10**<br>40.49<br>MaskLLM<br>✗<br>-<br>-<br>-<br>-<br>-<br>-<br>-<br>**SUSI** (Ours)<br>✗<br>18.16<br>**38.85**<br>28.79<br>**62.51**<br>**29.33**<br>68.51<br>**41.03**|25<br>221<br>71<br>46<br>-<br>**42**|


As shown in Figure 3 and Table 6, under the 2:8 sparsity pattern, SUSI achieves a 3 _._ 5 _×_ reduction in trainable parameters relative to MaskLLM, while maintaining competitive perplexity. This
demonstrates that even with substantially fewer learnable parameters than in the 2:4 case, SUSI continues to deliver robust language modeling performance. These results underscore the efficiency of
differentiable subset sampling in handling larger sparsity patterns.


The 4:8 sparsity pattern (Table 7) presents an even more demanding setting. Here, MaskLLM fails to
execute due to the prohibitive number of trainable parameters. By contrast, SUSI remains tractable,
successfully completing training and yielding stable evaluation results. This highlights a distinct
advantage of SUSI: its parameter efficiency not only improves training feasibility but also makes
previously impractical sparsity patterns accessible to large-scale language models.


A.8 EXTEND SUSI TO RECENT LLMS


We further extend SUSI to recent LLM architectures, including Qwen2.5-0.5B and Llama3.2-1B,
to examine its generality beyond the OPT family. As shown in Table 8, SUSI remains feasible
and efficient under these modern settings. While the performance gap relative to dense models is
more pronounced than in the OPT series (e.g., Qwen2.5-0.5B drops from 55.33% accuracy at 22
PPL to 43.75% at 46 PPL after pruning), SUSI still achieves competitive results. Compared to
the OPT family, where SUSI nearly matches the dense baseline, these results highlight that SUSI
scales consistently to diverse architectures, maintaining tractable training and offering substantial
efficiency gains even when accuracy trade-offs are larger in more recent models.


A.9 LIMITATIONS


Despite the promising performance and efficiency demonstrated by SUSI, several limitations remain:


17


Table 8: Performance of SUSI on recent LLMs (Qwen2.5-0.5B and Llama3.2-1B). SUSI remains
tractable, demonstrating scalability across architectures. Although the performance gap to dense
models is larger than in the OPT family, SUSI preserves competitive accuracy with favorable
perplexity-efficiency trade-offs.


First, the deployment of semi-structured sparsity is inherently hardware-dependent. At present,
substantial throughput gains are realized only on select platforms (e.g., AMD ROCm and certain
NVIDIA Ampere and Hopper GPUs) where 2:4 structured sparsity is natively supported and accelerated at the kernel level. Although SUSI can, in principle, be extended to arbitrary _N_ : _M_ sparsity
patterns, its practical utility is constrained by the absence of hardware kernels and vendor-optimized
libraries for ratios other than 2:4. On accelerators or CPUs lacking such specialized support, pruning yields only marginal reductions in memory footprint and fails to deliver meaningful inference
speedup. This hardware dependency poses a significant challenge for widespread adoption in heterogeneous production environments, where deployment targets may vary.


Second, the current evaluation focuses exclusively on English-centric OPT models and a limited set
of standard NLP benchmarks. Future research should investigate the applicability of SUSI to multilingual LLMs, larger-scale models, and domain-specific tasks (e.g., code generation, reasoningintensive applications) to assess its generalization and scalability comprehensively.


18