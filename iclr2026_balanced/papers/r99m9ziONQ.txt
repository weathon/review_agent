# IA2 : ALIGNMENT WITH ICL ACTIVATIONS IMPROVES SUPERVISED FINE-TUNING


**Aayush Mishra, Daniel Khashabi** _[♡]_ **& Anqi Liu** _[♡]_
Department of Computer Science, Johns Hopkins University


ABSTRACT


Supervised Fine-Tuning (SFT) is used to specialize model behavior by training
weights to produce intended target responses for queries. In contrast, In-Context
Learning (ICL) adapts models during inference with instructions or demonstrations in the prompt. ICL can offer better generalizability and more calibrated
responses compared to SFT in data scarce settings, at the cost of more inference
compute. In this work, we ask the question: _Can ICL’s internal computations be_
_used_ _to_ _improve_ _the_ _qualities_ _of_ _SFT?_ We first show that ICL and SFT produce
distinct activation patterns, indicating that the two methods achieve adaptation
through different functional mechanisms. Motivated by this observation and to
use ICL’s rich functionality, we introduce **I** CL **A** ctivation **A** lignment ( **IA2** ), a
self-distillation technique which aims to replicate ICL’s activation patterns in SFT
models and incentivizes ICL-like internal reasoning. Performing **IA2** as a priming step before SFT significantly improves the accuracy and calibration of model
outputs, as shown by our extensive empirical results on 12 popular benchmarks
and two model families. This finding is not only practically useful, but also offers
a conceptual window into the inner mechanics of model adaptation.


1 INTRODUCTION


LLMs are general purpose models but are often used in specialized applications. For example, a
news aggregator app may need to classify articles into predefined categories. A popular approach
for adapting LLMs is **Supervised** **Fine-Tuning** **(SFT)** which uses a dataset of labeled samples to
train and adapt LLMs on narrow downstream tasks. With the power of parameter efficient fine tuning
(PEFT) techniques (Hu et al., 2021; Liu et al., 2022), SFT models are often just a small set of parameters which can be efficiently loaded on/off GPU memory making them extremely useful. However,
SFT typically requires a large set of labeled samples to generalize well on new tasks (Le Scao &
Rush, 2021), which can be expensive to collect.


In contrast, **In-Context** **Learning** **(ICL)** (Brown et al., 2020) is used to adapt and steer LLM
behavior during inference time. The model is given a query preceded by in-context demonstrations or instructions, which help it “learn” the demonstrated task and answer accordingly. Prior
work (Duan et al., 2024) shows (and our experiments concur (§5)) that _ICL_ _generalizes_ _well_ _in_ _a_
_few-shot setting and typically produces well calibrated responses_ . However, using ICL comes at a
cost. ICL demonstrations/instructions use up valuable context space (increasing the cost of running
each query) which could otherwise be used for processing more query and response tokens.


These subtle but important differences between ICL and SFT motivate us to investigate the difference in their functional behavior. We find that while their token space behavior may look similar
on the surface, **models** **produce** **disparate** **internal** **activations** **under** **ICL** **and** **SFT** (§3). As
activations are a footprint of the model’s internal processing before it produces the next token, divergent activation patterns highlight a difference in how the two methods achieve adaptation, an idea
supported by prior work (Shen et al., 2024). We hypothesize that when the model performs ICL,
its activations contain rich information about how to extract generalizable patterns from the context.
This information may be absent in SFT models especially in a few-shot setting, where they are prone
to shortcut learning/overfitting on target responses. Hence, we ask the research question:
_Can the information rich ICL activations be used to improve the quality of SFT?_


_♡_ Equal advising. Correspondence to: _{_ amishr24,danielk,aliu.cs _}_ @jhu.edu


1


Figure 1: An overview of our improved SFT pipeline. Standard SFT only enforces output space
alignment (between the model’s response and a target response—only step 3 above), resulting in
subpar performance and mis-calibration in low-data settings. In contrast, our method **IA2** enforces
functional alignment with ICL by matching the rich activation patterns produced when performing
ICL. **IA2** _priming_ before SFT boosts the quality of adaptation. We show this improvement through
performance comparison charts aggregated across models and datasets. See section 5 for details.


Recent works have tried to distill context in the weights of LLMs (Snell et al., 2022; Chen et al.,
2024b), but they use training signals only from response texts, which may suffer from the same SFT
issues highlighted above. Simply training a model to reproduce the outputs of an ICL-conditioned
model does not ensure that it functions like an ICL-conditioned model.


To address this, we propose **I** CL **A** ctivation **A** lignment ( **IA2** ), a self-distillation method to enforce
alignment with the model’s own functional behavior when performing ICL (§4). See Figure 1 for
an overview of our proposed method: (1) collect information rich ICL activations and (2) enforce
functional alignment with ICL. Then (3) perform SFT on this primed model. We show that **priming**
**models with IA2 before SFT** _**using the same data**_ **, drastically improves the performance** of the
adapted model on a variety of text classification and generation tasks (§5). In addition, we show
that **IA2** **provides** **an** **important** **training** **signal** **that** **is** **unavailable** **with** **SFT** **only** **training** .
We trained over 13,000 models spanning 12 benchmarks, to validate our findings, which not only
signify the practical benefits of **IA2**, but offer a conceptual window into the inner mechanics of
model adaptation. In summary:


_⋆_ We show that ICL and SFT with the same data, do not align in the model’s activation space,
highlighting a gap in their functional behavior (§3).
_⋆_ We propose **IA2** —ICL Activation Alignment, to enforce alignment with ICL’s functional behavior. This priming step drastically improves the performance of SFT models (see Figure 1, §4).
_⋆_ We show that the **IA2** training signal is not present in SFT only training, highlighting its importance in improving the quality of adaptation (§5).


2 BACKGROUND AND NOTATION


**Transformer Language Models:** A standard decoder-only transformer based LM _M_ Θ with parameters Θ consists of a stack of self-attention (SA) and linear layers. A sequence of token vectors
_T_ = [ _ti_ ] _[R]_ _i_ =1 [is processed by applying the SA and linear projections on each token at each layer, until]
the last token projection at the final layer is passed through the LM head to predict the most likely
next token in the sequence. The SA operation is special because it is affected by other tokens in the
sequence. For a standard SA operation which uses 4 weight matrices, _WQ, WK, WV_ and _WO_ _∈_ Θ
(we will call it the set _WQKV O_ for brevity), the output at each token position is given by:


SA( _T_ ; _WQKV O_ ) = _Z_ = - _zi_ = _σ_ - _qi_ ~~_√_~~ _K_ _[T]_

_d_


- _V_ _· WO_ - _Ri_ =1 (1)


where _σ_ ( _._ ) denotes Softmax, _qi_ = _ti · WQ, K_ = _t_ : _i · WK_, _V_ = _t_ : _i · WV_ ( _t_ : _i_ means first _i_ tokens).
Note that _Z_ has the same shape ( _R × d_ ) as the input _T_ . In this work, we study the interplay of
changes in model behavior induced by changing the context _T_ or the weights _WQKV O_ .


2


**Supervised Fine-Tuning:** Given a set _DT_ = _{_ ( _Xi, Yi_ ) _}_ _[N]_ _i_ =1 [of] _[ N]_ [input (] _[X]_ [)–output (] _[Y]_ [ ) pairs illus-]
trating a task _T_, we can fine-tune the model weights Θ _→_ Θ _[′]_ to produce relevant responses for new
inputs (say _Xt_ ), i.e., _M_ Θ _′_ ( _Xt_ ) = _y_ ¯ _t_ . Here, _y_ ¯ _t_ is the first token of the whole response _Y_ [¯] _t_ which can
be extended by repeated application of _M_ Θ _′_ on the growing sequence. This process is performed
as follows. If the ground truth response _Yi_ contains _G_ tokens, we generate _G_ new tokens with the
model to get _Y_ [¯] _i_ and minimize the cross-entropy loss over all generated tokens. The SFT loss is
defined as:


_G_


_L_ SFT =


_N_


_i_ =1


- cross-entropy( _Y_ [¯] _ij, Yij_ ) (2)


_j_ =1


**In-Context** **Learning:** Using _DT_, _M_ Θ can also be prompted with the concatenated sequence of
demonstration tokens _I_ = [ _X_ 1 _◦_ _Y_ 1 _◦_ _. . ._ _◦_ _Xn_ _◦_ _Yn_ ] along with a new test sample _Xt_ to get a
task-appropriate response without any weight updates. _M_ Θ processes the examples in its prompt to
understand the task and produces a response _M_ Θ( _I ◦_ _Xt_ ) = _y_ ˆ _t_ accordingly. Remarkably, _Y_ [ˆ] _t_ is often
similar to the expected response _Yt_ . Hence, ICL serves as a useful inference-time adaptation method.
ICL was first illustrated in GPT-3 (Brown et al., 2020) and is widely used to adapt generic LMs at
the the inference-time. Today’s frontier models have the ability to follow instructions directly, so a
generic instruction could be considered a form of ICL demonstrations. In this work, we will focus
on the classic ICL setting with a demonstration set of input-output pairs.


**Activations:** As the SA operation uses every token in the context, it produces different outputs for
different sequences of tokens. We denote the hidden SA outputs as activations. Activations are a
footprint of the model’s internal processing which can be used to study model behavior changes with
changing contexts. If _M_ Θ consists of _L_ layers, and it processes _R_ tokens, we get an activation tensor
_A_ of size _L × R × d_ . In this work, we study sequences of tokens _T_ = [ _I_ _◦_ _X_ ] for ICL; and _T_ = _X_
without ICL under different model weights, where _I_ denotes tokens of ICL demos and _X_ denotes
tokens of the query. Here, we also define activation similarity (asim ( _A_ 1 _, A_ 2)) as the token-wise
cosine similarity between two activation tensors. Note that asim is of size _L × R_ .


3 ARE ICL AND SFT PRACTICALLY THE SAME?

Layerwise Similarity with ICL Activations


1.0


0.8


0.6


0.4


0.2


Recent studies (Von Oswald et al., 2023; Aky¨urek
et al., 2022) claim that ICL in transformer models
works by an internal gradient descent mechanism.
They show that the self-attention operation can produce outputs as if the model parameters were updated
using ICL demos in context. This implies a functional
equivalence between the two. Under this equivalence,
we should expect the base model with ICL demos to
produce similar activations at the output token positions as the SFT model produces without ICL demos.
We investigated whether this phenomenon is actually
exhibited by LLMs.


Qwen-4B
mean:0.52

Llama-3B
mean:0.61

Llama-1B
mean:0.50


Qwen-4B
mean:0.80

Llama-3B
mean:0.91

Llama-1B
mean:0.88


Qwen-4B
mean:0.67

Llama-3B
mean:0.83

Llama-1B
mean:0.70


**ICL** **and** **SFT** **produce** **different** **activations:** We

mean:0.70

0.0


for 100 test samples from multiple datasets (SST2,
AGNews, etc.) in a variety of models (Qwen3 (Yang Figure 2: Layerwise Similarity with ICL
et al., 2025) and Llama-3.2 (Grattafiori et al., 2024) Activations. SFT models have small asim
family, 1B _↔_ 16 layers, 3B _↔_ 28 layers, and 4B _↔_ 36 with ICL. This indicates differing functional
layers). We then trained SFT models using the same behavior for adaptation. Our recommended
ICL demos until convergence, and collected activa- pipeline ( **IA2** → SFT) aligns much more
tions _A_ SFT (using _T_ = _X_ ). Experiment details can be with ICL than using only SFT, and performs
found in §B. Then, we calculated asim ( _A_ ICL _, A_ SFT) much better as a result (see Figure 1).
at output token positions, and plotted the average across tokens, samples, and datasets (see Figure 2–
SFT only in the first row). We see that ICL and SFT activations are not aligned across different
models. It is noteworthy that the activations align better near the initial and final layers where we
expect the tokens to be processed at an individual level, but are misaligned in the middle where we
expect the whole demonstration set to be processed at an abstract level.


0.0


Figure 2: Layerwise Similarity with ICL
Activations. SFT models have small asim
with ICL. This indicates differing functional
behavior for adaptation. Our recommended
pipeline ( **IA2** → SFT) aligns much more
with ICL than using only SFT, and performs
much better as a result (see Figure 1).


3


**Why** **should** **we** **care** **about** **alignment** **with** **ICL?** The difference in functional behavior of ICL
and SFT is not just hidden in activation patterns. It also surfaces in the form of expected calibration
error (Guo et al., 2017) as seen in Figure 1. ICL is much more calibrated in its responses than SFT at
similar (slightly better) accuracy levels. We suspect that this is due to the output-oriented nature of
the SFT training signal, which can allow it to learn shortcuts that can fail on new data. In contrast, as
ICL relies on complex circuits (Elhage et al., 2021) that need to work for a variety of tasks, it extracts
generalizable patterns from the demos and performs in a well-calibrated manner. This experiment
demonstrates that ICL and SFT achieve adaptation through different means, and SFT should not
be expected to work as a drop-in replacement for ICL. Motivated by these differences, we ask the
following research question:


_Can the information rich ICL activations be used to improve the quality of SFT?_


Next, we will show that this is indeed the case and describe our proposed SFT procedure.


4 INDUCING ICL-LIKE BEHAVIOR IN SFT MODELS


In this section, we will discuss our two-step SFT pipeline. First, we propose our priming step ( **IA2** )
which incentivizes ICL-like functional behavior. Then, we discuss how performing further SFT on
these ICL-aligned weights results in much better adaptation qualities.


**IA2—ICL Activation Alignment:** In §3, we found SFT to have little asim with ICL. To increase
asim, and hence the functional similarity with ICL, we design the following goal ( **IA2** ) for each
layer of _M_ Θ:

Find _W_ [˜] _QKV O_ = ( _W_ [˜] _Q,_ _W_ [˜] _K,_ _W_ [˜] _V,_ _W_ [˜] _O_ ) such that for every newly generated token (at index: _−_ 1):

SA([ _I_ _◦_ _X_ ]; _WQKV O_ ) _≈_ SA( _X_ ; _W_ [˜] _QKV O_ ) _∀X_ _∈T ._ (3)


Prior work (Chen et al., 2024a) has shown a closed-form solution for the above on linearized attention models (using a kernel approximation for softmax). We aim to find a practical general solution
in real non-linear transformers. Our goal tries to achieve the qualities of ICL directly in the weights
of the SFT model. These modified weights are not incentivized to produce similar outputs, but to
process all inputs (queries) similar to how ICL does at each layer of the base model.


To perform **IA2**, we first generate model responses using ICL ( _Ti_ = [ _I_ _◦_ _Xi_ ]), giving us _Y_ [ˆ] _i_ for all
training samples _Xi_ . We use the remaining samples in the dataset to construct _I_ for each sample. We
collect activations at each output position giving us _A_ _[i]_ ICL _[∈]_ [R] _[L][×][G][×][d]_ [assuming] _[ G]_ [ response tokens.]
Next, we provide the response attached only with the query _Ti_ = [ _Xi_ _◦_ _Y_ [ˆ] _i_ ], as if the model had
produced the ICL response with only the query in context. This gives an unaligned activation tensor
_A_ _[i]_ at the output token positions. Then, we simply train the model to minimize the mean squared
error between the two activations w.r.t. model parameters:


_L_ **IA2** =


_N_

- _||A_ _[i]_ _−_ _A_ _[i]_ ICL _[||][.]_ (4)

_i_ =1


**IA2 →** **SFT:** After **IA2**, we continue to train the model with the standard cross-entropy loss for SFT
(Equation 2) on target tokens. Say **IA2** training updates the model parameters from Θ to Θ _[′]_ . We
collect output responses _Y_ [¯] _i_ for all _Xi_ using _M_ Θ _′_ and minmize the SFT loss between _Y_ [¯] _i_ and ground
truth _Yi_ for all training samples. This loss further aligns the model’s outputs with our intended
targets. Overall, our proposed training pipeline has the following two steps:


1. Collect target ICL activations _A_ ICL and perform **IA2** using _L_ **IA2** until convergence. This
aligns the model’s functional behavior with ICL.
2. Switch to _L_ SFT with ground truth target responses _Y_, and train until convergence to align
with target output behavior.


5 EXPERIMENTAL RESULTS


Our study includes experiments in two different settings: single-token and multi-token, with different characteristics. We will highlight any differences as we present our results.


4


**Adaptation** **Tasks:** In the single-token setting, the output _Yi_ for each sample is a single token,
typically a category/class label, True/False, or MCQ choice. In the multi-token setting, the output
_Yi_ can be multiple tokens long and variable in length. This is the general case for open-ended
generation. For both settings, we test and compare our proposed SFT pipeline on various tasks
using multiple models. In many cases, we also test the adaptation methods on OOD datasets to
study their behavior under distribution shift. We enlist the tasks under each category below:


**Single-token tasks:**


- _Sentiment Classification:_ We use SST2 (Socher et al., 2013), Financial Phrasebank [FinS] (Malo
et al., 2014) and Poem Sentiment [PoemS] (Sheng & Uthus, 2020) datasets. We use SST2 and FinS
for training models and use all three for evaluation on each set. This creates one in-distribution
(ID) and 2 out-of-distribution (OOD) evaluation datasets.

- _True/False:_ We use StrategyQA [STF] (Geva et al., 2021) which involves multi-hop reasoning
before coming to a True/False conclusion.

- _News_ _Categorization:_ We use AGNews [AGN] (Zhang et al., 2015) and BBCNews

[BBCN] (Greene & Cunningham, 2006). We train on AGN and evaluate on both.

- _MCQ:_ We use SciQ (Johannes Welbl, 2017), and QASC (Khot et al., 2019). To create additional
complexity, we remap the MCQ choices [A, B, C ...] to an unrelated token set. Hence, we call the
datasets SciQr and QASCr. We train on SciQr and evaluate on both.


**Multi-token tasks:**


- _Grade-school_ _Math:_ We use GSM8K (Cobbe et al., 2021), and GSM Symbolic [GSM8Ks]
(Mirzadeh et al., 2024). We train on GSM8K and evaluated on both.

- _Advanced_ _Math:_ We use MATH Algebra [HMathA] (Hendrycks et al., 2021) which consists of
higher grade Algebra problems.

- _Scientific QA:_ We used SciQ again, but this time for generating the tokens in the the answer instead
of choosing between options.


**Data** **Setup:** As our focus is on the data-scarce few-shot setting where ICL is typically used, we
create multiple training datasets _DT_ = _{_ ( _Xi, Yi_ ) _}_ _[N]_ _i_ =1 [for] [each] [task] [by] [varying] _[N]_ [in] [orders] [of]
2 : [2 _,_ 4 _,_ 8 _,_ 16 _, ..._ ]. We create 5 different sets at each _N_ value to average out outlier effects on our
final quantitative analysis. Importantly, _we collect ICL activation tensors A_ _[i]_ _ICL_ _[∀][X][i][ for]_ _**[ IA2]**_ _[ training]_
_by using the remaining N −_ 1 _samples as ICL demonstrations (in random order), effectively reusing_
_training samples_ . This makes sure that **we use the exact same data for all adaptation methods**,
keeping the comparison fair. For evaluation, we sampled a different set of 500 samples for each
task. Additional details about the datasets and how they are processed can be found in §A.


**Multi-token** **details:** Here, the model outputs a sequence of _reasoning_ _tokens_ (not inside the
_<_ think _>_ tags, just standard text) before the answer in a specific format illustrated in the demonstrations. The demonstrations and expected output are both chain-of-thought (Wei et al., 2022) style.
The actual answer needs to be extracted from the generated text to evaluate model performance and
different datasets have different method for parsing the answer from response tokens.


- For GSM8K and GSM8Ks, the ground truth answers have the following pattern: “ _<_ reasoning
steps _>_ #### _<_ numerical answer _>_ ”. We do not evaluate the correctness or conciseness of the
reasoning steps and only parse the numerical answer out from the response and match it exactly
with the ground truth answer for calculating accuracy. We also use other common answer parsing
techniques like matching with “The answer is _<_ value _>_ ”, to allow for slight variations in output
format. Our exact parsing function can be seen in our code.

- For HMathA, the answers are expected to be inside a _\_ boxed _{}_ element and parsed accordingly.

- For SciQ, we generate the ground truth format similar to the GSM8K pattern, i.e., “ _<_ support
text _>_ #### _<_ text answer _>_ ”. We parsed the answers according to this format and considered an
exact string match as the only correct answer.


**Training Setup:** We use two model families to test each training method across all tasks. We use
the Qwen3-4B-Base model on every task (single/multi-token). In addition, we also use the Llama3.2-1B model on every single-token task except SciQr for which we use Llama-3.2-3B. This is
because the 1B Llama model was unable to improve above random baseline for any _N_ upto 128.
For the multi-token tasks, we use the Llama-3.2-1B-Instruct model as the secondary model. Using


5


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|AGN<br>AGN<br>30.0 (02.3)<br>0.13 (0.03)<br>BBCN_∗_<br>28.8 (00.6)<br>0.12 (0.01)|27.9 (04.3)<br>0.52 (0.12)<br>24.0 (00.3)<br>**0.11** (0.01)<br>**31.8** (03.8)<br>0.35 (0.24)<br>**31.8** (07.1)<br>0.56 (0.14)<br>24.3 (01.3)<br>**0.10** (0.02)<br>27.9 (04.3)<br>0.40 (0.21)|
|FinS<br>FinS<br>63.6 (01.5)<br>0.12 (0.01)<br>PoemS_∗_<br>56.9 (02.2)<br>0.12 (0.02)<br>SST2_∗_<br>70.1 (01.3)<br>0.17 (0.01)|67.4 (14.1)<br>0.31 (0.14)<br>63.1 (15.0)<br>0.24 (0.06)<br>**78.7** (13.7)<br>**0.16** (0.11)<br>49.9 (04.6)<br>0.48 (0.04)<br>52.8 (05.6)<br>**0.15** (0.04)<br>**60.6** (15.7)<br>0.36 (0.15)<br>59.4 (06.5)<br>0.20 (0.10)<br>69.8 (13.7)<br>0.30 (0.10)<br>**71.4** (18.9)<br>**0.20** (0.15)|
|SciQr<br>QASCr_∗_<br>56.5 (01.4)<br>0.08 (0.01)<br>SciQr<br>87.7 (01.1)<br>0.07 (0.01)|76.5 (03.8)<br>0.09 (0.05)<br>71.2 (03.0)<br>0.12 (0.01)<br>**79.4** (01.8)<br>**0.09** (0.01)<br>88.3 (04.6)<br>0.07 (0.05)<br>90.4 (01.5)<br>0.09 (0.01)<br>**91.7** (01.7)<br>**0.05** (0.01)|
|SST2<br>FinS_∗_<br>41.9 (00.3)<br>0.19 (0.01)<br>PoemS_∗_<br>65.1 (01.2)<br>0.11 (0.02)<br>SST2<br>85.4 (00.4)<br>0.13 (0.00)|68.4 (02.3)<br>0.30 (0.04)<br>71.3 (02.1)<br>0.21 (0.03)<br>**82.4** (12.2)<br>**0.11** (0.08)<br>56.5 (13.2)<br>0.33 (0.14)<br>62.4 (09.2)<br>**0.19** (0.06)<br>**68.4** (08.4)<br>0.30 (0.08)<br>65.2 (10.7)<br>0.22 (0.10)<br>82.7 (06.4)<br>0.28 (0.03)<br>**90.4** (02.3)<br>**0.06** (0.01)|
|STF<br>STF<br>66.0 (02.1)<br>0.07 (0.01)|52.4 (04.4)<br>0.29 (0.10)<br>62.2 (04.1)<br>**0.16** (0.04)<br>**62.4** (04.3)<br>0.29 (0.08)|


Table 1: Performance report for _N_ = 4 on Qwen3-4B-Base model, showing accuracy (acc) and
Expected Calibration Error (ece). Numbers in parentheses show standard deviations across 5 runs
for the best performing learning rate. Best training method shown in **bold** . ( _[∗]_ ) highlights OOD
evaluations. _Our proposed_ _**IA2**_ _→_ _SFT training method outperforms standard SFT across the board._


an Instruct model tests our method’s efficacy on post-trained models. For training models, we use
LoRAs with rank 8 to modify the _WQ, WK_ and _WO_ matrices of self-attention layers. We chose
these models and weight combinations reasonably on the basis of available resources, as we trained
more than 13,000 models in total for our experiments. Training in multi-token setting is much more
resource extensive. As each sample has a long answer, we decided to only use _N_ upto 16 for Llama
and _N_ upto 8 for Qwen models in multi-token setting. We also fix the maximum generated tokens
( _G_ = 200) during training for efficiency. This is used in **IA2** for collecting the activation tensor
_A_ ˆ _i_ _∈_ R _[L][×]_ [200] _[×][d]_ _∀xi_ . For SFT, _G_ is equal to the number of tokens in the ground truth answer. Each
(method, dataset) combination is trained until convergence (upto 50 epochs) with 3 learning rates
(slow:1e-4, medium:3e-4, fast:1e-3), and the best performing one is chosen to neutralize the
effect of hyperparameter selection on method performance. Additional details in §B.


**Evaluation:** In the single-token setting, we use accuracy and expected calibration error (ECE) as
performance metrics. Given that the model’s answer is a single token, we calculate the accuracy and
ECE on the basis of the first generated token’s probability distribution. First, we evaluate the base
model with ICL demos to set the ICL performance benchmark. For this, we evaluate the validation
set of each task 5 times with different ICL demos to report their average metrics. Then, we evaluate
each training method without ICL demos and report their best-performing average metrics. In the
multi-token setting, we restricted the maximum generated tokens during evaluation to judiciously
use compute resources. We set the limit to 200 for GSM8K and GSM8Ks datasets. This is on the
basis of their 95 percentile answer lengths (184 and 154 respectively). We relax this limit to 400 for
HMathA, as the 95 percentile answer lengths are 307. For performance, we consider only accuracy,
i.e., number of questions answered correctly in the required format (see answer parsing and other
details in §A). As the answer confidence is hard to measure for open-ended generation tasks, we do
not measure calibration.


5.1 **IA2** PRIMING BOOSTS SFT PERFORMANCE


**Single-Token results:** In Table 1, we report the performance metrics of adapted Qwen models for
the practical setting of _N_ = 4 for single-token tasks. We find that _our proposed_ _**IA2**_ _→_ _SFT method_
_outperforms_ _SFT_ _only_ _training_ in almost all cases in terms of _both_ _accuracy_ _and_ _calibration_, all
while using the same amount of data. In most cases, **IA2** → SFT also outperforms ICL in terms of
accuracy, with slightly less calibration. Notice that **IA2** only training reaches high accuracy in many
cases, while being reasonably calibrated like ICL, _without being trained on target response tokens_ .
This highlights the richness of **IA2** training signal. While this table shows the results for one setup,
Figure 1 captures the aggregate performance trends across models and datasets. Detailed tabulated
results for some other model/ _N_ combinations, and a statistical significance test of our improvement
can be seen in §C.


6


Dataset Adaptation Method

|w/o ICL ICL<br>Source Eval<br>acc ↑ acc ↑|SFT only IA2 only IA2 →<br>acc ↑ acc ↑ acc|
|---|---|
|GSM8K<br>GSM8K<br>56.4 (00.0)<br>76.4 (01.1)<br>GSM8Ks_∗_<br>45.4 (00.0)<br>68.4 (01.1)|70.9 (02.8)<br>**77.4** (02.8)<br>73.6 (<br>64.5 (04.0)<br>66.2 (03.7)<br>**68.8** (|
|HMathA<br>HMathA<br>21.0 (00.0)<br>60.4 (02.1)|50.4 (01.8)<br>47.8 (06.0)<br>**55.3** (|
|SciQ<br>SciQ<br>15.4 (00.0)<br>37.5 (01.0)|35.0 (07.1)<br>06.9 (13.8)<br>**40.8** (|


Table 2: Performance report for _N_ = 4 on Qwen3-4B-Base models, on multi-token datasets. Best
training method shown in **bold** . _**IA2**_ _→_ _SFT outperforms standard SFT across all datasets._


**Multi-Token results:** In Table 2, we report the accuracy of adapted Qwen models in the multi-token
setting. Like single-token, our proposed _**IA2**_ _→_ _SFT method outperforms SFT only training, in all_
_cases_ . It is noteworthy that compared to the single-token setting, **IA2** → SFT performs worse than
ICL sometimes. We suspect that this is due to imperfect compression of longer contexts in the same
small LoRA weight space, which should improve with larger ranks. More tabulated results for other
model/ _N_ combinations are present in §C.


5.2 WHY IS **IA2** PRIMING IMPORTANT?


We investigate several properties of the three training methods: 1) SFT only, 2) **IA2** only and 3)
**IA2** → SFT models, and make the following observations.


**Activation Similarity vs Performance:** Figure 2 shows the activation similarity of all 3 methods
with ICL. **IA2** drastically increases this similarity as expected. Importantly, **IA2** → SFT retains
higher similarity with ICL activations even after SFT training. In Figure 3 we show the relationship
between ICL activation similarity and performance metrics. **IA2** → SFT sits comfortably between
**IA2** and SFT in terms of activation similarity, while achieving better accuracy and calibration.
Importantly, note that SFT only and **IA2** → SFT training only differs in the start state. This implies
that _**IA2**_ _offers_ _a_ _rich_ _training_ _signal_ _which_ _is_ _unavailable_ _via_ _SFT_ _only_ _training_ . It is clear that
more activation similarity implies better calibration. However, extreme activation similarity may
not be the optimal for achieving the best accuracy, as the ICL signal is not always right. In fact,
in §C, we show one case where **IA2** models follow ICL performance even if becomes worse due
to overfitting. Therefore, it is best to combine the two training signals: **IA2** to improve the internal
functional alignment with ICL and SFT on ground truth responses to align with expected output
behavior.


**Subspace Overlap:** Although activation patterns are a footprint of the model’s functional behavior,
it does not isolate changes in behavior induced by different training methods. This is because we
use LoRA adaptors and the activation patterns have significant influence from the unchanged base
model weights. To isolate the change in behavior, we analyze LoRA weights directly.


90


80


70


60


50


40


30


20


|Col1|Col2|Col3|Col4|
|---|---|---|---|
||SFT o<br>IA2 o<br>IA2|nly<br> nly<br> SFT||


0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Activation Similarity


0.5


0.4


0.3


0.2


0.1


0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Activation Similarity


Figure 3: We scatter asim ( _A, A_ ICL) vs Accuracy/ECE for all training methods to show the impact
of asim on performance metrics. Each point corresponds to one training experiment. With increasing similarity, ECE goes down smoothly. But extreme ICL activation alignment may leave some
accuracy gains on the table that can be sourced using SFT on ground truth responses.


7


Figure 4 shows the weight subspace overlap distribution between pairs of models trained using different methods. We use
models from all single/multi token experiments. We first perform SVD on the weight matrices to find basis vectors. Then
we calculate subspace overlap by the formula: [1] _r_ _[∗||][U]_ 1 _[ T]_ _[·]_ _[U]_ [2] _[||]_ _F_ [2]

where _U_ 1 and _U_ 2 are the basis vectors corresponding to the two
methods being compared, and _r_ is the rank of LoRA weights.
Subspace overlap measures how much of the weight space is
shared between two methods, in effect isolating and comparing the difference in their functional behavior. We find that
_SFT_ _only_ _training_ _results_ _in_ _weight_ _updates_ _which_ _are_ _almost_
_completely orthogonal to the other two methods._ Meanwhile,
the most performant **IA2** → SFT models share around 39% of
their spanned weight space on average with **IA2** only trained
models. This implies that a lot of the performance gains of
**IA2** → SFT models are achieved because of **IA2**, and the subspaces identified by **IA2** are practically unreachable by SFT
only training. The high spread of subspace overlap between
**IA2** only and **IA2** → SFT models is because in many cases,
the **IA2** model itself was already output aligned, so the secondary SFT training did not change the model much before
convergence, resulting in very high subspace overlap.


6 DISCUSSION


1.0


0.8


0.6


0.4


0.2


0.0


|=0.055 =0.067 =0.388<br>=0.016 =0.015 =0.260|Col2|Col3|
|---|---|---|
|med=0.061<br><br>med=0.064<br><br>med=0.277|med=0.061<br><br>med=0.064<br><br>med=0.277|med=0.061<br><br>med=0.064<br><br>med=0.277|
|med=0.061<br><br>med=0.064<br><br>med=0.277|||
||||
||||
||||
||||


SFT only

vs
IA2 only


SFT only

vs
IA2 SFT


IA2 only

vs
IA2 SFT


Figure 4: **IA2** → SFT models have
significantly high subspace overlap
with **IA2** only models, while SFT
only models do not align with either. This indicates a training signal in **IA2** that is absent from SFT
only training and, which is important for performance.


**IA2** **+** **SFT:** One natural alternative to **IA2** → SFT training is **IA2** + SFT, i.e., using both _L_ **IA2**
and _L_ SFT at the same time rather than sequentially. This approach poses a practical challenge. _A_ [ˆ]
collected as the target activation tensor for **IA2** uses the model’s own generated response tokens for
any _G_ _>_ 1. We call it the ICL response, which could be very different (even in length) from the
ground truth response in the dataset. This makes the two objectives incompatible to train together.
However, we could perform self-distillation using ICL responses, similar to Snell et al. (2022) but
including an additional **IA2** signal. We perform this training with a unified loss _L_ **IA2** + _β_ _· L_ SFT
and vary _β_ with 4 values spanning the spectrum between **IA2** only and SFT only training smoothly
(details in §B). After choosing the best performing (learning rate, _β_ ) combination across 5 random
seeds, we report the performance for multi-token experiments in Table 3. Note that the performance
of SFT only and **IA2** → SFT training drops when compared to training on ground truth responses
(from Table 2), but **IA2** + SFT extracts significantly more performance out of ICL responses when
compared to SFT only training. This highlights the richness of the **IA2** training signal. Additional
results of **IA2** + SFT training can be found in §C.


Dataset Adaptation Method

|w/o ICL ICL<br>Source Eval<br>acc ↑ acc ↑|SFT only IA2 only IA2 →SFT IA2 + SFT<br>acc ↑ acc ↑ acc ↑ acc ↑|
|---|---|
|GSM8K<br>GSM8K<br>56.4 (00.0)<br>76.4 (01.1)<br>GSM8Ks_∗_<br>45.4 (00.0)<br>68.4 (01.1)|66.4 (18.1)<br>**77.4** (02.8)<br>67.2 (22.1)<br>77.0 (07.8)<br>59.8 (20.2)<br>66.2 (03.7)<br>61.6 (24.0)<br>**69.0** (09.6)|
|HMathA<br>HMathA<br>21.0 (00.0)<br>60.4 (02.1)|49.0 (03.0)<br>47.8 (06.0)<br>**54.0** (01.7)<br>53.6 (02.7)|
|SciQ<br>SciQ<br>15.4 (00.0)<br>37.5 (01.0)|27.2 (12.0)<br>06.9 (13.8)<br>32.8 (10.0)<br>**34.5** (10.6)|


Table 3: Performance report for _N_ = 4 on Qwen3-4B-Base models when trained with ICL responses instead of ground truth responses. _**IA2**_ _+ SFT performs significantly better than SFT only_
_highlighting the significance of_ _**IA2**_ _._


**Knowledge Distillation baseline:** A popular approach to distill the behavior of a stronger teacher
into the student model is through soft label matching (Hinton et al., 2015). The teacher model
provides a denser training signal through the probability mass associated with each token. In our
work, we treat the same model (enhanced with ICL context) as the teacher. To test how **IA2** based
training performs in comparison to soft label matching, we train and evaluate Llama models on


8


GSM8K (1B-Instruct) and SST2 (1B) with soft label matching on ICL responses. In Table 4, we
observe that soft-label matching performs significantly better than standard SFT in the multi-token
case almost reaching **IA2** + SFT performance, but lacks in the single-token case. This highlights
the consistency of **IA2** based training in achieving high performance.

|Dataset N ICL|SFT only SFT (soft labels) IA2 only IA2 →SFT IA2 + SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|GSM8K<br>4<br>34.6 (00.9)<br>–|28.2 (04.5)<br>–<br>36.1 (03.5)<br>–<br>29.3 (05.0)<br>–<br>30.3 (03.2)<br>–<br>**37.0** (01.9)<br>–|
|SST2<br>8<br>77.0 (02.2)<br>0.10 (0.02)|52.3 (04.3)<br>0.38 (0.14)<br>55.9 (08.8)<br>**0.16** (0.06)<br>64.9 (15.3)<br>0.30 (0.05)<br>64.7 (19.3)<br>0.33 (0.19)<br>**66.9** (18.4)<br>0.28 (0.21)|


Table 4: Comparison with Knowledge Distillation (KD) baseline (SFT on soft labels). Although
KD baseline performs better than standard SFT in the multi-token case, **IA2** based training provides
consistently high performance for single/multi token cases.


**Why** **IA2** **based** **training** **does** **not** **consistently** **beat** **ICL?** In some multi-token cases, like the
Qwen model on math datasets (GSM8K and HMathA), we find that ICL performance exceeds all
training methods including **IA2** based training. We suspect that this is due to mid-training of Qwen
on STEM data, making ICL extremely sample efficient (surprisingly, ICL performance for GSM8K
at _N_ = 2 is larger than that at _N_ = 4 _,_ 8). In Table 5, we illustrate the evolution of performance with
shot count. We also highlight the gap between ICL and **IA2** → SFT performance, which closes
quickly with more shots. Our key takeaway— **IA2** → SFT performs better than SFT only—remains
consistent as highlighted by the last column.

|Dataset N ICL|SFT only IA2 only IA2 →SFT Gap wrt ICL Gap wrt SFT only|
|---|---|
|GSM8K<br>2<br>81.2 (00.4)<br>4<br>76.4 (01.1)<br>8<br>77.2 (00.9)|65.2 (01.0)<br>70.7 (01.8)<br>74.6 (02.4)<br>-06.6<br>+09.4<br>70.9 (02.8)<br>77.4 (02.8)<br>73.6 (03.7)<br>-02.8<br>+02.7<br>73.5 (01.6)<br>78.8 (02.5)<br>76.0 (02.0)<br>-01.2<br>+02.5|
|HMathA<br>2<br>58.6 (02.0)<br>4<br>60.4 (02.1)<br>8<br>62.1 (01.0)|47.8 (04.0)<br>33.9 (06.4)<br>48.7 (07.0)<br>-09.9<br>+00.9<br>50.4 (01.8)<br>47.8 (06.0)<br>55.3 (02.1)<br>-05.1<br>+04.9<br>52.0 (00.8)<br>57.3 (03.7)<br>59.4 (00.8)<br>-02.7<br>+07.4|


Table 5: ICL is more sample efficient than training methods for Qwen models in math datasets. But
**IA2** _→_ SFT quickly closes the gap in performance with increasing number of shots, and consistently
beats SFT only training.


**Other fine-tuning methods:** LoRA is not the only method to fine-tune models. One could perform
full fine tuning (full-rank) or other parameter efficient methods like prompt/prefix tuning (Lester
et al., 2021; Li & Liang, 2021) and activation scaling ((IA) [3] Liu et al. (2022)). Our proposed
pipeline is not in competition, rather an improvement over these methods as it provides better training signals. To establish this, we repeat our experiments on a few datasets with the popular PEFT
method (IA) [3] . We followed the exact same methodology from §5, just replaced LoRA training
with (IA) [3] (details about setup in §B). We report the metrics for one setting in Table 6 which
shows trends similar to LoRA. Other results can be seen in §C. We defer the exploration of improvement in other SFT methods through **IA2** for future work. On a side note, **IA2** also marks a major
improvement in method naming compared to (IA) [3] .


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|SciQr<br>QASCr_∗_<br>35.6 (01.4)<br>0.06 (0.01)<br>SciQr<br>60.0 (01.8)<br>0.07 (0.01)|32.8 (03.4)<br>0.20 (0.08)<br>34.2 (02.7)<br>**0.08** (0.01)<br>**48.6** (05.5)<br>0.25 (0.11)<br>64.5 (05.0)<br>0.12 (0.05)<br>63.0 (04.6)<br>0.10 (0.04)<br>**70.6** (04.6)<br>**0.08** (0.02)|
|SST2<br>FinS_∗_<br>56.6 (01.6)<br>0.17 (0.01)<br>PoemS_∗_<br>52.3 (03.0)<br>0.19 (0.02)<br>SST2<br>60.3 (02.8)<br>0.19 (0.02)|58.2 (10.7)<br>0.25 (0.03)<br>**61.9** (09.5)<br>**0.18** (0.06)<br>59.8 (07.8)<br>0.18 (0.03)<br>48.1 (03.2)<br>0.44 (0.11)<br>48.1 (03.4)<br>**0.11** (0.04)<br>**59.8** (13.4)<br>0.33 (0.13)<br>52.4 (02.8)<br>0.35 (0.05)<br>54.2 (04.2)<br>**0.15** (0.06)<br>**62.6** (09.1)<br>0.28 (0.11)|


Table 6: Performance report for _N_ = 4 on Llama-3.2 models trained using (IA) [3] . _**IA2**_ _→_ _SFT_
_achieves significant gains over SFT only training, exceeding even ICL accuracy in all cases._


9


**Computational Overhead: IA2** → SFT requires a data (activations) collection step before training,
where we perform ICL inference on the model with the given dataset. This is a one-time cost to
collect the rich ICL activations, which greatly benefits the model’s capabilities during inference that
runs at the same cost as SFT only models.


**Potential improvements:** We made several design choices, which were fixed due to resource constraints, that have potential for exploration and improvements. The most important ones include
ablations around LoRA parameters (rank, target modules, etc.) and selective **IA2** . We showed that
extreme alignment can hurt performance and previous work (Todd et al., 2023) has shown that some
layers are more sensitive to ICL than others. Therefore, selectively aligning layers can be more
effective. Another minor improvement could include prompt optimization before collecting target
activations, as ICL performance is sensitive to the order of demos (Lu et al., 2022; Zhao et al.,
2021). Lastly, we perform the **IA2** → SFT pipeline with fixed learning rates, but the two signals
could benefit from finer grained control. We defer these explorations to future work.


7 RELATED WORK


**ICL vs SFT:** Prior works have studied the performance differences between ICL and SFT (Mosbach
et al., 2023; Duan et al., 2024), but the verdict is unclear. In our work, we found SFT to perform
worse than ICL. Other works have studied the functional similarity (Dai et al., 2022) and differences (Wang et al., 2023) of ICL with SFT. Some works show how adaptation through ICL/SFT
can be miscalibrated and how to improve it (Zhang et al., 2024; Li et al., 2025; 2024; Xiao et al.,
2025). Sia et al. (2024) identify “where” translation happens in LLMs during ICL through context
masking, using activations as a signal for functional behavior similar to our work. Importantly, recent work (Chu et al., 2025; Shenfeld et al., 2025) has shown SFT to be prone to memorization/overfitting and catastrophic forgetting. Our improved SFT pipeline aims to address these problems.


**Context** **Distillation:** A large body of work targets to improve LLM adaptation using some form
of compression of explicit context signals like us. These range from using intermediate _contempla-_
_tion_ tokens for compression (Cheng & Van Durme, 2024), internalizing context through generating
adapters on the fly (Chen et al., 2025; Charakorn et al., 2025), using related text to enhance distillation (Zhu et al., 2025; Choi et al., 2025), or compressing context directly into weights (Shen
et al., 2025; Deng et al., 2024; Yu et al., 2024; Snell et al., 2022; Chen et al., 2024b; Shin et al.,
2024). These works aim to distill the effect of given contexts on model outputs using token response
signals. Our work is complementary to these works and improves the distillation effects through the
model’s own ICL processing signals, an idea with supporting prior work (Aguilar et al., 2020; Jin
et al., 2024; Yang et al., 2024).


**Activation Steering:** Lastly, a growing line of work attempts to extract “steering” vectors or weights
to apply on targeted locations inside the model to create an intended effect on the model output (Postmus & Abreu, 2024; Stolfo et al., 2025; Caccia et al., 2025; Fleshman & Van Durme, 2024; Arditi
et al., 2024). These methods are similar to our work in that they intervene in the activation space
to influence the model’s functionality. In contrast to their targeted approach and small application
domain, our method aims to incite a large behavioral change in the model which avoids potential
brittleness and reliability issues (Queiroz Da Silva et al., 2025).


8 CONCLUSION AND FUTURE WORK


In this work, we used model activations to probe the functional behavior of LLMs and found a
key distinction: ICL and SFT achieve adaptation through different mechanisms, and ICL often
encodes richer, more generalizable patterns. Motivated by this, we introduced **IA2**, an SFT priming
technique that enables ICL-like internal behavior in the model. This simple step shifts models into
a more adaptable weight subspace, which is inaccessible to SFT alone. Extensive experiments show
that **IA2** consistently improves the accuracy and calibration of SFT. In the future, we aim to test
other effects like impact on diversity and catastrophic forgetting of **IA2** based SFT to measure how
**IA2** can help in post-training LLMs. We also aim to study the effect of **IA2** on post-trained (RLtuned) models.


10


REPRODUCIBILITY STATEMENT


We provide codes used for data download and processing, training, evaluation, analysis, plotting,
etc. [on github.](https://github.com/aamixsh/ia2) All details in this paper (§5 and the appendix) along with a detailed README file
should be sufficient to fully reproduce our results as we used repeatable random seeds.


ACKNOWLEDGMENTS


This work is supported by ONR (N00014-24-1-2089), an Amazon Research Award, and a grant
from Open Philanthropy. We thank Andrea Wynn for their constructive feedback on earlier versions
of this document.


REFERENCES


Gustavo Aguilar, Yuan Ling, Yu Zhang, Benjamin Yao, Xing Fan, and Chenlei Guo. Knowledge
distillation from internal representations. In _Proceedings_ _of_ _the_ _AAAI_ _conference_ _on_ _artificial_
_intelligence_, volume 34, pp. 7350–7357, 2020. URL [https://arxiv.org/pdf/1910.](https://arxiv.org/pdf/1910.03723)
[03723.](https://arxiv.org/pdf/1910.03723)


Ekin Aky¨urek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, and Denny Zhou. What learning algorithm is in-context learning? investigations with linear models. In _International Conference on_
_Learning Representations_ (ICLR), 2022. [URL https://arxiv.org/abs/2211.15661.](https://arxiv.org/abs/2211.15661)


Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel
Nanda. Refusal in language models is mediated by a single direction. _Advances in Neural Infor-_
_mation Processing Systems_, 37:136037–136083, 2024.


Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are
few-shot learners. _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ (NeurIPS), 2020. URL
[https://arxiv.org/abs/2005.14165.](https://arxiv.org/abs/2005.14165)


Lucas Caccia, Alan Ansell, Edoardo Ponti, Ivan Vuli´c, and Alessandro Sordoni. Training plug-nplay knowledge modules with deep context distillation. _arXiv preprint arXiv:2503.08727_, 2025.
[URL https://arxiv.org/pdf/2503.08727.](https://arxiv.org/pdf/2503.08727)


Rujikorn Charakorn, Edoardo Cetin, Yujin Tang, and Robert Tjarko Lange. Text-to-lora: Instant
transformer adaption. _arXiv preprint arXiv:2506.06105_, 2025. [URL https://arxiv.org/](https://arxiv.org/pdf/2506.06105)
[pdf/2506.06105.](https://arxiv.org/pdf/2506.06105)


Brian K Chen, Tianyang Hu, Hui Jin, Hwee Kuan Lee, and Kenji Kawaguchi. Exact conversion
of in-context learning to model weights in linearized-attention transformers. _ArXiv_ _preprint_,
abs/2406.02847, 2024a. [URL https://arxiv.org/abs/2406.02847.](https://arxiv.org/abs/2406.02847)


Tong Chen, Qirun Dai, Zhijie Deng, and Dequan Wang. Demonstration distillation for efficient
in-context learning, 2024b. [URL https://openreview.net/forum?id=Y8DClN5ODu.](https://openreview.net/forum?id=Y8DClN5ODu)


Tong Chen, Hao Fang, Patrick Xia, Xiaodong Liu, Benjamin Van Durme, Luke Zettlemoyer, Jianfeng Gao, and Hao Cheng. Generative adapter: Contextualizing language models in parameters
with a single forward pass. In _International_ _Conference_ _on_ _Learning_ _Representations_ (ICLR),
2025. [URL https://arxiv.org/abs/2411.05877.](https://arxiv.org/abs/2411.05877)


Jeffrey Cheng and Benjamin Van Durme. Compressed chain of thought: Efficient reasoning through
dense representations. _arXiv preprint arXiv:2412.13171_, 2024. [URL https://arxiv.org/](https://arxiv.org/abs/2412.13171)
[abs/2412.13171.](https://arxiv.org/abs/2412.13171)


Younwoo Choi, Muhammad Adil Asif, Ziwen Han, John Willes, and Rahul Krishnan. Teaching
llms how to learn with contextual fine-tuning. In _International_ _Conference_ _on_ _Learning_ _Repre-_
_sentations_ (ICLR), 2025. [URL https://arxiv.org/pdf/2503.09032.](https://arxiv.org/pdf/2503.09032)


11


Tianzhe Chu, Yuexiang Zhai, Jihan Yang, Shengbang Tong, Saining Xie, Dale Schuurmans, Quoc V
Le, Sergey Levine, and Yi Ma. Sft memorizes, rl generalizes: A comparative study of foundation
model post-training. _arXiv preprint arXiv:2501.17161_, 2025.


Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
Schulman. Training verifiers to solve math word problems. _arXiv_ _preprint_ _arXiv:2110.14168_,
2021. [URL https://arxiv.org/pdf/2110.14168.](https://arxiv.org/pdf/2110.14168)


Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei. Why can gpt learn incontext? language models secretly perform gradient descent as meta optimizers. _arXiv preprint_
_arXiv:2212.10559_, 2022. [URL https://arxiv.org/abs/2212.10559.](https://arxiv.org/abs/2212.10559)


Yuntian Deng, Yejin Choi, and Stuart Shieber. From explicit cot to implicit cot: Learning to internalize cot step by step. _arXiv preprint arXiv:2405.14838_, 2024.


Yifei Duan, Liu Li, Zirui Zhai, and Jinxia Yao. In-context learning distillation for efficient fewshot fine-tuning. _arXiv preprint arXiv:2412.13243_, 2024. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2412.13243)
[2412.13243.](https://arxiv.org/abs/2412.13243)


Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann,
Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Deep Ganguli, Zac Hatfield-Dodds, Danny Hernandez, Andy Jones, Jackson Kernion, Liane Lovitt, Kamal
Ndousse, Dario Amodei, Tom Brown, Jack Clark, Jared Kaplan, Sam McCandlish, and Chris
Olah. A mathematical framework for transformer circuits. _Transformer_ _Circuits_ _Thread_, 2021.
[URL https://transformer-circuits.pub/2021/framework/index.html.](https://transformer-circuits.pub/2021/framework/index.html)


William Fleshman and Benjamin Van Durme. RE-Adapt: Reverse Engineered Adaptation of Large
Language Models. _arXiv_ _preprint_ _arXiv:2405.15007_, 2024. URL [https://arxiv.org/](https://arxiv.org/abs/2405.15007)
[abs/2405.15007.](https://arxiv.org/abs/2405.15007)


Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. Did aristotle
use a laptop? a question answering benchmark with implicit reasoning strategies. _Transactions of_
_the Association for Computational Linguistics_, 9:346–361, 2021.


Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad
Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd
of models. _arXiv preprint arXiv:2407.21783_, 2024. [URL https://ai.meta.com/blog/](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
[llama-3-2-connect-2024-vision-edge-mobile-devices/.](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)


Derek Greene and P´adraig Cunningham. Practical solutions to the problem of diagonal dominance
in kernel document clustering. In _Proceedings of the 23rd international conference on Machine_
_learning_, pp. 377–384, 2006.


Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. On calibration of modern neural
networks. In _ICML_, 2017.


Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _NeurIPS_,
2021.


Geoffrey Hinton, Oriol Vinyals, Jeff Dean, et al. Distilling the knowledge in a neural network.
In _Advances in Neural Information Processing Systems_ (NeurIPS) _Workshop on Deep Learning_,
2015. [URL https://arxiv.org/abs/1503.02531.](https://arxiv.org/abs/1503.02531)


Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen,
et al. Lora: Low-rank adaptation of large language models. In _International_ _Conference_ _on_
_Learning Representations_ (ICLR), 2021. [URL https://arxiv.org/abs/2106.09685.](https://arxiv.org/abs/2106.09685)


Heegon Jin, Seonil Son, Jemin Park, Youngseok Kim, Hyungjong Noh, and Yeonsoo Lee. Alignto-distill: Trainable attention alignment for knowledge distillation in neural machine translation.
In _International_ _Conference_ _on_ _Computational_ _Linguistics_ (COLING), 2024. URL [https://](https://arxiv.org/abs/2403.01479)
[arxiv.org/abs/2403.01479.](https://arxiv.org/abs/2403.01479)


12


Matt Gardner Johannes Welbl, Nelson F. Liu. Crowdsourcing multiple choice science questions.
_arXiv:1707.06209v1_, 2017.


Tushar Khot, Peter Clark, Michal Guerquin, Peter Jansen, and Ashish Sabharwal. QASC: A dataset
for question answering via sentence composition. In _Conference on Artificial Intelligence_ (AAAI),
2019.


Teven Le Scao and Alexander Rush. How many data points is a prompt worth? In _Conference_
_of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ (NAACL),
2021. doi: 10.18653/v1/2021.naacl-main.208. URL [https://aclanthology.org/](https://aclanthology.org/2021.naacl-main.208)
[2021.naacl-main.208.](https://aclanthology.org/2021.naacl-main.208)


Brian Lester, Rami Al-Rfou, and Noah Constant. The power of scale for parameter-efficient prompt
tuning. In _Conference on Empirical Methods in Natural Language Processing_ (EMNLP), 2021.
[URL https://arxiv.org/abs/2104.08691.](https://arxiv.org/abs/2104.08691)


Chengzu Li, Han Zhou, Goran Glavaˇs, Anna Korhonen, and Ivan Vuli´c. Large language models
are miscalibrated in-context learners. In _Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_
_Linguistics_ (ACL) _- Findings_, 2025. [URL https://arxiv.org/abs/2312.13772.](https://arxiv.org/abs/2312.13772)


Jinyang Li, Binyuan Hui, Ge Qu, Jiaxi Yang, Binhua Li, Bowen Li, Bailin Wang, Bowen Qin,
Ruiying Geng, Nan Huo, et al. Can LLM already serve as a database interface? A big bench for
large-scale database grounded text-to-SQLs. _Advances in Neural Information Processing Systems_,
36, 2024.


Xiang Lisa Li and Percy Liang. Prefix-tuning: Optimizing continuous prompts for generation. In
_Annual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ (ACL), 2021. URL [https:](https://arxiv.org/pdf/2101.00190.pdf)
[//arxiv.org/pdf/2101.00190.pdf.](https://arxiv.org/pdf/2101.00190.pdf)


Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and
Colin A Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context
learning. _Advances in Neural Information Processing Systems_, 35:1950–1965, 2022.


Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, and Pontus Stenetorp. Fantastically ordered prompts and where to find them: Overcoming few-shot prompt order sensitivity. In _An-_
_nual_ _Meeting_ _of_ _the_ _Association_ _for_ _Computational_ _Linguistics_ (ACL), 2022. URL [https:](https://arxiv.org/pdf/2104.08786.pdf)
[//arxiv.org/pdf/2104.08786.pdf.](https://arxiv.org/pdf/2104.08786.pdf)


P. Malo, A. Sinha, P. Korhonen, J. Wallenius, and P. Takala. Good debt or bad debt: Detecting
semantic orientations in economic texts. _Journal of the Association for Information Science and_
_Technology_, 65, 2014.


Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad
Farajtabar. GSM-Symbolic: Understanding the limitations of mathematical reasoning in large
language models. _arXiv preprint arXiv:2410.05229_, 2024.


Marius Mosbach, Tiago Pimentel, Shauli Ravfogel, Dietrich Klakow, and Yanai Elazar. Few-shot
fine-tuning vs. in-context learning: A fair comparison and evaluation. In _Annual_ _Meeting_ _of_
_the Association for Computational Linguistics_ (ACL) _- Findings_, 2023. doi: 10.18653/v1/2023.
findings-acl.779. [URL https://aclanthology.org/2023.findings-acl.779.](https://aclanthology.org/2023.findings-acl.779)


Joris Postmus and Steven Abreu. Steering large language models using conceptors: Improving
addition-based activation engineering. _arXiv_ _preprint_ _arXiv:2410.16314_, 2024. URL [https:](https://arxiv.org/abs/2410.16314)
[//arxiv.org/abs/2410.16314.](https://arxiv.org/abs/2410.16314)


Patrick Queiroz Da Silva, Hari Sethuraman, Dheeraj Rajagopal, Hannaneh Hajishirzi, and Sachin
Kumar. Steering off course: Reliability challenges in steering language models. _arXiv preprint_
_arXiv:2504.04635_, 2025. [URL https://arxiv.org/abs/2504.04635.](https://arxiv.org/abs/2504.04635)


Lingfeng Shen, Aayush Mishra, and Daniel Khashabi. Do pretrained transformers learn in-context
by gradient descent? In _International_ _Conference_ _on_ _Machine_ _Learning_ (ICML), 2024. URL
[https://arxiv.org/abs/2310.08540.](https://arxiv.org/abs/2310.08540)


13


Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing
chain-of-thought into continuous space via self-distillation. _arXiv_ _preprint_ _arXiv:2502.21074_,
2025. [URL https://arxiv.org/abs/2502.21074.](https://arxiv.org/abs/2502.21074)


Idan Shenfeld, Jyothish Pari, and Pulkit Agrawal. Rl’s razor: Why online reinforcement learning
forgets less. _arXiv preprint arXiv:2509.04259_, 2025.


Emily Sheng and David Uthus. Investigating societal biases in a poetry composition system, 2020.


Haebin Shin, Lei Ji, Yeyun Gong, Sungdong Kim, Eunbi Choi, and Minjoon Seo. Generative prompt
internalization. _arXiv_ _preprint_ _arXiv:2411.15927_, 2024. URL [https://arxiv.org/abs/](https://arxiv.org/abs/2411.15927)
[2411.15927.](https://arxiv.org/abs/2411.15927)


Suzanna Sia, David Mueller, and Kevin Duh. Where does in-context translation happen in large
language models. _arXiv preprint arXiv:2403.04510_ [, 2024. URL https://arxiv.org/pdf/](https://arxiv.org/pdf/2403.04510)
[2403.04510.](https://arxiv.org/pdf/2403.04510)


Charlie Snell, Dan Klein, and Ruiqi Zhong. Learning by distilling context. _arXiv_ _preprint_
_arXiv:2209.15189_, 2022. [URL https://openreview.net/pdf?id=am22IukDiKf.](https://openreview.net/pdf?id=am22IukDiKf)


Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng,
and Christopher Potts. Recursive deep models for semantic compositionality over a sentiment
treebank. In _Conference on Empirical Methods in Natural Language Processing_ (EMNLP), 2013.
[URL https://aclanthology.org/D13-1170.pdf.](https://aclanthology.org/D13-1170.pdf)


Alessandro Stolfo, Vidhisha Balachandran, Safoora Yousefi, Eric Horvitz, and Besmira Nushi. Improving instruction-following in language models through activation steering. In _International_
_Conference on Learning Representations_ (ICLR), 2025. [URL https://openreview.net/](https://openreview.net/forum?id=wozhdnRCtw)
[forum?id=wozhdnRCtw.](https://openreview.net/forum?id=wozhdnRCtw)


Eric Todd, Millicent L Li, Arnab Sen Sharma, Aaron Mueller, Byron C Wallace, and David Bau.
Function vectors in large language models. _ArXiv preprint_ [, abs/2310.15213, 2023. URL https:](https://arxiv.org/abs/2310.15213)
[//arxiv.org/abs/2310.15213.](https://arxiv.org/abs/2310.15213)


Johannes Von Oswald, Eyvind Niklasson, Ettore Randazzo, Jo˜ao Sacramento, Alexander Mordvintsev, Andrey Zhmoginov, and Max Vladymyrov. Transformers learn in-context by gradient descent. In _International_ _Conference_ _on_ _Learning_ _Representations_ (ICLR), 2023. URL
[https://arxiv.org/abs/2212.07677.](https://arxiv.org/abs/2212.07677)


Xindi Wang, Yufei Wang, Can Xu, Xiubo Geng, Bowen Zhang, Chongyang Tao, Frank Rudzicz,
Robert E Mercer, and Daxin Jiang. Investigating the learning behaviour of in-context learning: a
comparison with supervised learning. In _European Conference on Artificial Intelligence_ (ECAI),
2023. [URL https://arxiv.org/abs/2307.15411.](https://arxiv.org/abs/2307.15411)


Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny
Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. _Advances in_
_Neural Information Processing Systems_ (NeurIPS), 2022. [URL https://arxiv.org/abs/](https://arxiv.org/abs/2201.11903)
[2201.11903.](https://arxiv.org/abs/2201.11903)


Jiancong Xiao, Bojian Hou, Zhanliang Wang, Ruochen Jin, Qi Long, Weijie J Su, and Li Shen.
Restoring calibration for aligned large language models: A calibration-aware fine-tuning approach. In _International_ _Conference_ _on_ _Machine_ _Learning_ (ICML), 2025. URL [https:](https://arxiv.org/abs/2505.01997)
[//arxiv.org/abs/2505.01997.](https://arxiv.org/abs/2505.01997)


An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu,
Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. _arXiv_ _preprint_
_arXiv:2505.09388_, 2025.


Zhaorui Yang, Tianyu Pang, Haozhe Feng, Han Wang, Wei Chen, Minfeng Zhu, and Qian Liu.
Self-distillation bridges distribution gap in language model fine-tuning. In _Annual Meeting of the_
_Association_ _for_ _Computational_ _Linguistics_ (ACL), 2024. URL [https://aclanthology.](https://aclanthology.org/2024.acl-long.58/)
[org/2024.acl-long.58/.](https://aclanthology.org/2024.acl-long.58/)


14


Ping Yu, Jing Xu, Jason Weston, and Ilia Kulikov. Distilling system 2 into system 1. _arXiv preprint_
_arXiv:2407.06023_, 2024. [URL https://arxiv.org/abs/2407.06023.](https://arxiv.org/abs/2407.06023)


Hanlin Zhang, Yifan Zhang, Yaodong Yu, Dhruv Madeka, Dean Foster, Eric Xing, Himabindu
Lakkaraju, and Sham Kakade. A study on the calibration of in-context learning. In _Conference of_
_the North American Chapter of the Association for Computational Linguistics_ (NAACL), 2024.
[URL https://arxiv.org/pdf/2312.04021.](https://arxiv.org/pdf/2312.04021)


Xiang Zhang, Junbo Zhao, and Yann LeCun. Character-level convolutional networks for
text classification. In _Advances_ _in_ _Neural_ _Information_ _Processing_ _Systems_ (NeurIPS),
2015. URL [https://proceedings.neurips.cc/paper/2015/file/](https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf)
[250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf.](https://proceedings.neurips.cc/paper/2015/file/250cf8b51c773f3f8dc8b4be867a9a02-Paper.pdf)


Zihao Zhao, Eric Wallace, Shi Feng, Dan Klein, and Sameer Singh. Calibrate before use: Improving
few-shot performance of language models. In _International_ _Conference_ _on_ _Machine_ _Learning_
(ICML), 2021. [URL http://proceedings.mlr.press/v139/zhao21c/zhao21c.](http://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)
[pdf.](http://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)


Xingyu Zhu, Abhishek Panigrahi, and Sanjeev Arora. On the power of context-enhanced learning
in llms. In _International_ _Conference_ _on_ _Machine_ _Learning_ (ICML), 2025. URL [https://](https://arxiv.org/abs/2503.01821)
[arxiv.org/abs/2503.01821.](https://arxiv.org/abs/2503.01821)


15


A ADDITIONAL DATASET DETAILS


For all tasks, we shuffled and filtered out at most 2000 samples that would be used for creating
training datasets (including ICL demonstrations). For the evaluation datasets, we filtered out at
most 500 different samples (all datasets had 500 samples, apart from PoemS which had only 288
samples). These evaluation samples are never seen by the model in any adaptation method. Some
details about datasets are presented below.


    - **SST2** contains a binary positive/negative sentiment classification of phrases from movie
reviews. The prompt was structured as: “Text: _<_ _Xi_ _>\_ n Label:”. The model is supposed
to output the sentiment _Yi_ as a single token from the set [0 _,_ 1]. We created training datasets
for _N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ . We trained with Llama-3.2-1B on all _N_ settings, and
with Qwen3-4B-Base on _N_ _∈{_ 2 _,_ 4 _,_ 8 _}_ (as the ICL performance quickly saturates).

    - **FinS** contains positive/negative/neutral sentiment for financial news sentences. It is annotated by multiple humans so we choose the subset of samples where at least 50% of
annotators agreed on the classification and only included positive/negative samples to keep
the label set consistent with SST2. Prompt and output structure was the same as SST2. We
created training datasets for _N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ . We trained with Llama-3.2-1B
on all _N_ settings, and with Qwen3-4B-Base on _N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _}_ .

    - **PoemS** is a small dataset of sentiment analysis on poem verses. We filtered for positive/negative sentiments and only used this dataset for OOD evaluation. Prompt and output structure was the same as SST2.

    - **STF** contains True/False questions that require logical reasoning based on common sense
and general knowledge facts. The prompt was then structured as: “Question: _<_ _Xi_ _>\_ n
Answer:”. The model is supposed to output T/F ( _Yi_ ) as a single token from the set [0 _,_ 1]. We
created training datasets for _N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _}_ . We trained only with Qwen3-4B-Base
on _N_ _∈{_ 2 _,_ 4 _,_ 8 _}_ (as the ICL performance on Llama models did not improve much).

    - **AGN** contains 4 class (World, Business, Sports, Sci/Tech) classification of News articles. Prompt structure was the same as SST2. The model is supposed to output the
class _Yi_ as a single token from the set [0 _,_ 1 _,_ 2 _,_ 3]. We created training datasets for
_N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ . We trained with both Llama-3.2-1B and Qwen3-4B-Base on
all _N_ settings (for this data, performance increase with _N_ was very slow in both models).

    - **BBCN** is slightly shifted containing 5 classes. We mapped the 5 BBCN classes to AGN
classes as follows: Entertainment, Politics _→_ World; Business _→_ Business, Sports _→_
Sports, Tech _→_ Sci/Tech. Prompt and output structure was the same as AGN.

    - **SciQr** contains 3 distractors and 1 correct choice for Science exam questions. The choice
set is hence [0 _,_ 1 _,_ 2 _,_ 3]. However we remapped the labels to different tokens (discussed
below). The choices were randomly shuffled and combined to give a choice ~~s~~ tring like
“[label0]choice0 _\_ n ... [label3]choice3”. The prompt was structured as: “Question: _<_
_Xi_ _>\_ n Choices: _<_ choice ~~s~~ tring _>_ Answer:”. The model is supposed to output
the choice _Yi_ as a single token from the choice set. We created training datasets for
_N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _,_ 32 _,_ 64 _,_ 128 _}_ . We trained with Llama-3.2-3B on all _N_ settings, and with
Qwen3-4B-Base on _N_ _∈{_ 2 _,_ 4 _,_ 8 _}_ . We chose Llama 3B model instead of 1B because the
latter’s ICL performance did not improve above random baseline for any _N_ .

    - **QASCr** contains 8 choices instead of 4. This creates a unique label shift in the data therefore we evaluated SciQr models on QASCr questions as well. Prompt and output structure
was the same as SciQr.

    - **GSM8K** contains grade school Math problems. Prompt structure is the same as STF. The
model is supposed to output a long response ending with the proposed answer in a specific
format (discussed below). We created training datasets for _N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _}_ . We trained
with Llama-3.2-1B-Instruct on all _N_ settings, and with Qwen3-4B-Base on _N_ _∈{_ 2 _,_ 4 _,_ 8 _}_ .

    - **GSM8Ks** is the same quality of problems from GSM8K but with numbers and symbols
replaced, to minimize influence of data contamination in the pretraining stages. The prompt
and output format is the same as GSM8K.

    - **HMathA** consists of more advanced Algebra questions and has the same prompt/output
structure as GSM8K. We created training datasets for _N_ _∈{_ 2 _,_ 4 _,_ 8 _}_, and trained only
Qwen3-4B-Base model on these settings (because of longer evaluations).


16


- **SciQ** uses the same questions from SciQr but instead of posing it as a MCQ problem,
we use it as longer form QA problem. We use the support text column in the dataset as
reasoning tokens before using the correct answer text as the ground truth answer. The
model is still given the same prompt structure as GSM8K, and it is supposed to output the
support text followed by the answer text in a specific format. We created training datasets
for _N_ _∈{_ 2 _,_ 4 _,_ 8 _,_ 16 _}_ . We trained with Llama-3.2-1B-Instruct on all _N_ settings, and with
Qwen3-4B-Base on _N_ _∈{_ 2 _,_ 4 _,_ 8 _}_ .


**OOD** **evaluations:** For ICL OOD evaluations, we use ICL demos from the source dataset and
append the query from the OOD dataset at the end to evaluate how the model handles distribution
shift in the prompt. The trained models are trained with only the source data and during evaluation,
as we do not use ICL demos, so the prompt only contains the query from the OOD dataset.


**Label remapping for SciQr and QASCr:** To add additional complexity to the task and reduce any
influence from pretraining memorization, we remapped the MCQ choices (A, B, C ... were mapped
to 0, 1, 2 ... first) using the following remapping dictionary:


1 "remap_dict": {

2 "0": "apple",

3 "1": "Friday",

4 "2": "banana",

5 "3": "Saturday",

6 "4": "Thursday",

7 "5": "Sunday",

8 "6": "Wednesday",

9 "7": "Monday"

10 }


**Multi-Token evaluation details:** During evaluation, we used a stop string “ _\_ n _\_ n” to stop generations early as this usually indicated the end of the answer for the given question. If we do not use
this stop string, models often continue generating a new query and its answer without generating the
end of text token. This trick saves significant evaluation time for our experiments. We did not use
publically avaliable evaluation frameworks like [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) on known benchmarks like
GSM8K because we wanted more fine-grained control over which samples were used in the evaluation process. As our parsing functions remain consistent across all training/adaptation methods,
the relative performance between methods still gives a good estimate of how they would perform on
standard evaluations.


B ADDITIONAL EXPERIMENTAL DETAILS


**Training details:** We stop all training runs for convergence when the loss on a held out dev set (of
size _N/_ 2 to reflect the data scarce setting) does not decrease for 5 steps. Note that in total, we train
and evaluate 15 models for every _N_ value for reliable performance numbers.


**LoRA details:** We used LoRA rank = 8, _α_ = 8 and target modules are attention weights _WQ, WK_
and _WO_ for all our experiments. We settled on these parameters on the basis of generally used values for small targeted adaptations, as additional explorations require substantial compute resources
for the scale of our experiments. The target modules capture the cross-attention parts ( _Q, K_ ) and
final projection _O_ to allow for better adaptability. We did preliminary ablations to make sure our
results were not biased against SFT only (next paragraph). However, these selections are subject to
improvement with more exploration.


**LoRA** **ablations:** As our LoRA parameters are consistent, no training method has an unfair advantage. However, some methods may benefit from particular parameter choices. Therefore, we
conducted small scale ablation studies to measure the impact of LoRA parameters on performance.
In Table 7, we present these results for Llama 3.2-1B models trained on the SST2 dataset (evaluated
on 3 datasets: SST2, FinS, PoemS) and Qwen3-4B-base models on the GSM8K dataset (evaluated on itself). Each numerical column contains [acc-mean (acc-std), ece-mean (ece-std)] across 5
re-runs for the best performing hyperparameters (learning rate, beta). Although training only QK
matrices results in overall worse performance for all methods, there is still a healthy gap between
SFT only and **IA2** _→_ SFT. Rank increase from 8 to 16 does not have a huge impact on performance,


17


0.7


0.6


0.5


0.4


0.3


0.2


0.1


|SF|T only|Col3|
|---|---|---|
|<br>IA<br>IA|<br>2 only<br>2<br> SFT||
||<br>||
||||


0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Activation Similarity


90


80


70


60


50


40


30


20


0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
Activation Similarity


Figure 5: We scatter asim ( _A, A_ ICL) vs Accuracy/ECE for all training methods to show its the
impact of asim pn performance metrics. Each point corresponds to one training experiment. With
increasing similarity, ECE goes down smoothly. But extreme ICL activation alignment may leave
some accuracy gains on the table that can be sourced using SFT on ground truth responses.


and the choice between V and O matrices does not impact the results much either. Overall, our main
takeaway remains consistent: **IA2** _→_ SFT outperforms SFT only.

|Dataset LoRA params ICL|SFT only IA2 only IA2 →SFT|
|---|---|
|<br>acc_ ↑_|ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|FinS<br>r8, qko<br>60.2 (03.0)<br>0<br>PoemS<br>r8, qko<br>57.5 (01.8)<br>0<br>SST2<br>r8, qko<br>77.0 (02.2)<br>0<br>FinS<br>r8, qk<br>60.2 (03.0)<br>0<br>PoemS<br>r8, qk<br>57.5 (01.8)<br>0<br>SST2<br>r8, qk<br>77.0 (02.2)<br>0<br>FinS<br>r8, qkv<br>60.2 (03.0)<br>0<br>PoemS<br>r8, qkv<br>57.5 (01.8)<br>0<br>SST2<br>r8, qkv<br>77.0 (02.2)<br>0<br>FinS<br>r16, qko<br>60.2 (03.0)<br>0<br>PoemS<br>r16, qko<br>57.5 (01.8)<br>0<br>SST2<br>r16, qko<br>77.0 (02.2)<br>0|.10 (0.02)<br>60.0 (13.6)<br>0.33 (0.11)<br>53.2 (20.4)<br>0.33 (0.04)<br>89.8 (09.0)<br>0.09 (0.08)<br>.11 (0.02)<br>50.7 (03.7)<br>0.41 (0.07)<br>61.2 (12.5)<br>0.22 (0.06)<br>84.8 (08.1)<br>0.14 (0.08)<br>.10 (0.02)<br>53.5 (05.0)<br>0.34 (0.11)<br>64.9 (15.3)<br>0.30 (0.05)<br>90.4 (01.3)<br>0.09 (0.02)<br>.10 (0.02)<br>48.8 (16.1)<br>0.21 (0.12)<br>51.9 (12.6)<br>0.14 (0.02)<br>50.4 (21.6)<br>0.27 (0.07)<br>.11 (0.02)<br>48.9 (03.1)<br>0.24 (0.14)<br>42.6 (06.2)<br>0.13 (0.04)<br>56.7 (12.7)<br>0.29 (0.13)<br>.10 (0.02)<br>50.0 (06.4)<br>0.33 (0.13)<br>61.6 (08.3)<br>0.17 (0.09)<br>65.7 (15.0)<br>0.27 (0.14)<br>.10 (0.02)<br>54.2 (15.8)<br>0.34 (0.16)<br>66.2 (01.9)<br>0.16 (0.06)<br>87.3 (05.4)<br>0.06 (0.02)<br>.11 (0.02)<br>53.4 (05.5)<br>0.36 (0.11)<br>57.4 (08.0)<br>0.20 (0.08)<br>80.0 (07.2)<br>0.19 (0.07)<br>.10 (0.02)<br>53.5 (03.4)<br>0.34 (0.08)<br>65.3 (18.6)<br>0.34 (0.03)<br>88.7 (02.8)<br>0.10 (0.03)<br>.10 (0.02)<br>55.4 (13.0)<br>0.25 (0.12)<br>51.7 (19.4)<br>0.33 (0.04)<br>86.8 (09.4)<br>0.13 (0.09)<br>.11 (0.02)<br>54.5 (10.9)<br>0.26 (0.16)<br>57.2 (09.1)<br>0.27 (0.06)<br>79.5 (10.0)<br>0.13 (0.05)<br>.10 (0.02)<br>55.8 (13.9)<br>0.29 (0.14)<br>65.5 (14.9)<br>0.30 (0.05)<br>87.7 (05.8)<br>0.12 (0.06)|
|GSM8K<br>r8, qko<br>76.4 (01.1)<br>GSM8K<br>r16, qko<br>76.4 (01.1)|–<br>70.9 (02.8)<br>–<br>77.4 (02.8)<br>–<br>73.6 (03.7)<br>–<br>–<br>70.3 (05.5)<br>–<br>77.8 (06.4)<br>–<br>74.1 (03.3)<br>–|


Table 7: Ablation results on LoRA parameters (rank, target modules).


**Collecting activations:** We inserted hooks in the attention modules of the models and stacked activations from multiple token positions and layers together to get the activation tensors. For the asim
experiment, we used a small subset of validation samples (100 out of the total 500) to collect activations. Also, we restricted the trained models used in asim comparison to a small set corresponding
to _N_ values of _{_ 2 _,_ 4 _,_ 8 _,_ 16 _}_ . This is because collecting activations with ICL on higher _N_ values
takes a long time due to longer context. We used all datasets and model combinations to calculate
the asim in Figure 2. However, we removed the multi-token experiments and AGN dataset experiments from Figure 3. This is because 1) we don’t have ECE for multi-token experiments so it would
make the two plots incompatible, and 2) most AGN models did not improve in performance much
by _N_ = 16 mark, so it introduces noisy points in the scatter plot at baseline accuract and high ECE.
We show the plots including AGN in Figure 5, which still show similar trends with a few noisy
points from AGN models.


**IA2 + SFT details:** The _β_ parameter used in **IA2** + SFT experiments depends on the magnitude of
activation tensors generated by the model. We found that the Llama family had a small activation
magnitude, hence the MSE loss _L_ **IA2** at the start of training is typically an order of magnitude
smaller than the cross-entropy (CE) loss _L_ SFT, therefore we choose to weight the losses accordingly
and experiment with _β_ _∈_ [0 _._ 001 _,_ 0 _._ 01 _,_ 0 _._ 05 _,_ 0 _._ 5]. Here a _β_ = 0 _._ 5 makes the CE loss outweigh the
MSE loss almost completely. Qwen models typically have a similar order of magnitude between
MSE and CE loss. Therefore, we experiment with _β_ _∈_ [0 _._ 1 _,_ 0 _._ 5 _,_ 0 _._ 7 _,_ 0 _._ 9] to weight the CE loss
uniformly over the spectrum. Overall, these experiments were 4 times more compute extensive
because for each learning rate, we train the models with 4 different values of _β_ . We also notice that


18


**IA2** _→_ SFT _>_ SFT only **IA2** _→_ SFT _<_ SFT only
(acc) (ece)
_N_
**t-statistic** **p-value** **t-statistic** **p-value**


**IA2** _→_ SFT _>_ SFT only
(acc)
_N_
**t-statistic** **p-value**


Table 8: We show the statistical significance of improvement in performance of **IA2** _→_ SFT over
SFT only. The stats for single-token experiments are on the left and multi-token experiments on
the right. Statistically significant ( _p <_ 0 _._ 05) improvements are marked with green and insignificant
improvements with red.


in some cases, **IA2** + SFT or even **IA2** → SFT models trained with ICL response tokens outperform
those trained with ground truth tokens (see tables in §C). We suspect that this may be due to ICL
response being more “natural” (with respect to pre-trained weights) to the model and easily aligned
compared to potentially surprising ground truth responses.


**(IA)** [3] **details:** We used the same target modules, _WQ, WK_ and _WO_ as LoRA and used a different learning rate range [0 _._ 1 _,_ 0 _._ 01 _,_ 0 _._ 001]. This is because we experimented and found that (IA) [3]
typically needs a faster learning rate than LoRA to work. (IA) [3] works differently from LoRA – it
does not modify model weights, rather introduces attention vectors to scale the output of the target
modules. But our training procedure works seamlessly with this as well, as it finds better scaling
vectors through the rich **IA2** training signal.


**Performance numbers in Figure 1:** To create the bar plots in figure 1, we only considered singletoken experiments (because of ECE) and included results from both Qwen/Llama models across
datasets. The performance numbers for different datasets have signficant variations at a single _N_
value but we show the average across all datasets. To make sure that these results are statistically
significant, we also performed paired t-tests between SFT only and **IA2** → SFT performance metrics
across all datasets and training runs. We report this statistical significance result in Table 8, which
shows that almost all cases of improvements were statistically significant.


**ICL overfitting:** ICL is not perfect and is prone to overfitting. We see this in our OOD label shift
scenario QASCr evaluation based on SCiQr data. In Figure 6, we show the trend of performance of
all adaptation methods over variation in dataset size _N_, and notice the impact of ICL overfitting.


70


60


50


40


30


20


10


Accuracy vs N, Eval Dataset: QASCr, Source Dataset: SciQr

|Col1|ICL SFT only IA2 only|Col3|Col4|Col5|Col6|Col7|Col8|
|---|---|---|---|---|---|---|---|
||~~IA2 only~~<br>IA2<br> SFT|||||||
|||||||||
|||||||||
|||||||||
|||||||||


N (Number of Demonstrations)


Figure 6: The performance of ICL drops with increasing _N_ when the query is OOD to the demonstration data. **IA2** only performance follows the ICL curve indicating a strong correlation in its
functional behavior with ICL. However, this does not impact **IA2** → SFT performance.


19


C ADDITIONAL RESULTS


We show more performance tables here for the reader’s perusal.


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT IA2 + SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|AGN<br>AGN<br>30.0 (02.3)<br>0.13 (0.03)<br>BBCN_∗_<br>28.8 (00.6)<br>0.12 (0.01)|21.3 (01.8)<br>0.66 (0.06)<br>24.0 (00.3)<br>**0.11** (0.01)<br>23.9 (01.5)<br>0.70 (0.08)<br>**24.2** (00.9)<br>0.26 (0.10)<br>30.8 (05.6)<br>0.52 (0.08)<br>24.3 (01.3)<br>**0.10** (0.02)<br>30.1 (05.3)<br>0.47 (0.19)<br>**31.3** (05.6)<br>0.39 (0.16)|
|FinS<br>FinS<br>63.6 (01.5)<br>0.12 (0.01)<br>PoemS_∗_<br>56.9 (02.2)<br>0.12 (0.02)<br>SST2_∗_<br>70.1 (01.3)<br>0.17 (0.01)|61.9 (11.4)<br>0.35 (0.05)<br>**63.1** (15.0)<br>**0.24** (0.06)<br>60.8 (13.5)<br>0.35 (0.05)<br>62.8 (09.7)<br>0.31 (0.04)<br>47.7 (03.1)<br>0.50 (0.06)<br>**52.8** (05.6)<br>**0.15** (0.04)<br>47.4 (02.5)<br>0.47 (0.13)<br>47.8 (03.2)<br>0.25 (0.08)<br>53.0 (01.7)<br>0.34 (0.11)<br>**69.8** (13.7)<br>**0.30** (0.10)<br>51.3 (01.8)<br>0.49 (0.02)<br>54.1 (03.8)<br>0.37 (0.09)|
|SciQr<br>QASCr_∗_<br>56.5 (01.4)<br>0.08 (0.01)<br>SciQr<br>87.7 (01.1)<br>0.07 (0.01)|66.6 (15.1)<br>0.13 (0.09)<br>71.2 (03.0)<br>0.12 (0.01)<br>**72.8** (06.6)<br>0.14 (0.03)<br>68.8 (02.7)<br>**0.08** (0.02)<br>80.6 (14.3)<br>0.12 (0.10)<br>**90.4** (01.5)<br>0.09 (0.01)<br>86.6 (07.5)<br>**0.08** (0.05)<br>89.5 (00.8)<br>0.08 (0.01)|
|SST2<br>FinS_∗_<br>41.9 (00.3)<br>0.19 (0.01)<br>PoemS_∗_<br>65.1 (01.2)<br>0.11 (0.02)<br>SST2<br>85.4 (00.4)<br>0.13 (0.00)|57.4 (12.6)<br>0.40 (0.11)<br>71.3 (02.1)<br>**0.21** (0.03)<br>66.2 (18.1)<br>0.26 (0.17)<br>**71.5** (22.2)<br>0.26 (0.23)<br>53.4 (07.0)<br>0.40 (0.07)<br>62.4 (09.2)<br>**0.19** (0.06)<br>63.3 (11.7)<br>0.35 (0.13)<br>**66.1** (15.0)<br>0.23 (0.16)<br>62.3 (12.6)<br>0.34 (0.13)<br>**82.7** (06.4)<br>0.28 (0.03)<br>72.1 (18.4)<br>0.27 (0.19)<br>76.8 (14.3)<br>**0.27** (0.04)|
|STF<br>STF<br>66.0 (02.1)<br>0.07 (0.01)|53.5 (06.5)<br>0.41 (0.12)<br>**62.2** (04.1)<br>**0.16** (0.04)<br>60.0 (07.0)<br>0.36 (0.10)<br>58.5 (06.4)<br>0.27 (0.13)|


Table 9: Performance report for _N_ = 4 on Qwen3-4B-Base models trained using LoRA with ICL
responses on single-token datasets. With ICL responses, SFT signal from tokens is not very helpful
in increasing performance of **IA2** only models.


Dataset Adaptation Method

|w/o ICL ICL<br>Source Eval<br>acc ↑ acc ↑|SFT only IA2 only IA2 →SFT<br>acc ↑ acc ↑ acc ↑|
|---|---|
|GSM8K<br>GSM8K<br>56.4 (00.0)<br>81.2 (00.4)<br>GSM8Ks_∗_<br>45.4 (00.0)<br>71.3 (01.4)|65.2 (01.0)<br>70.7 (01.8)<br>**74.6** (02.4)<br>58.6 (04.3)<br>62.4 (02.5)<br>**67.2** (03.4)|
|HMathA<br>HMathA<br>21.0 (00.0)<br>58.6 (02.0)|47.8 (04.0)<br>33.9 (06.4)<br>**48.7** (07.0)|
|SciQ<br>SciQ<br>15.4 (00.0)<br>36.6 (01.6)|36.7 (06.8)<br>05.7 (09.5)<br>**40.4** (06.2)|


Table 10: Performance report for _N_ = 2 on Qwen3-4B-Base models trained using LoRA with
ground truth tokens on multi-token datasets.


Dataset Adaptation Method

|w/o ICL ICL<br>Source Eval<br>acc ↑ acc ↑|SFT only IA2 only IA2 →SFT IA2 + SFT<br>acc ↑ acc ↑ acc ↑ acc ↑|
|---|---|
|GSM8K<br>GSM8K<br>56.4 (00.0)<br>81.2 (00.4)<br>GSM8Ks_∗_<br>45.4 (00.0)<br>71.3 (01.4)|72.8 (02.0)<br>70.7 (01.8)<br>74.6 (02.5)<br>**78.8** (02.0)<br>68.0 (03.1)<br>62.4 (02.5)<br>65.4 (04.8)<br>**71.4** (03.8)|
|HMathA<br>HMathA<br>21.0 (00.0)<br>58.6 (02.0)|41.6 (06.4)<br>33.9 (06.4)<br>**45.4** (04.3)<br>43.9 (03.3)|
|SciQ<br>SciQ<br>15.4 (00.0)<br>36.6 (01.6)|26.1 (07.8)<br>05.7 (09.5)<br>32.9 (09.1)<br>**33.9** (07.4)|


Table 11: Performance report for _N_ = 2 on Qwen3-4B-Base models trained using LoRA with ICL
responses on multi-token datasets.


20


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|AGN<br>AGN<br>29.1 (01.0)<br>0.12 (0.01)<br>BBCN_∗_<br>32.6 (03.8)<br>0.11 (0.03)|**38.2** (05.3)<br>0.30 (0.06)<br>24.4 (00.4)<br>**0.11** (0.04)<br>35.8 (06.1)<br>0.39 (0.15)<br>**34.6** (06.2)<br>0.31 (0.07)<br>32.1 (07.0)<br>**0.11** (0.06)<br>29.6 (05.1)<br>0.43 (0.18)|
|FinS<br>FinS<br>57.7 (03.1)<br>0.13 (0.01)<br>PoemS_∗_<br>47.9 (02.6)<br>0.15 (0.03)<br>SST2_∗_<br>55.7 (02.7)<br>0.10 (0.01)|69.0 (01.5)<br>0.22 (0.08)<br>61.4 (12.4)<br>**0.14** (0.05)<br>**76.6** (11.3)<br>0.17 (0.09)<br>52.0 (04.9)<br>0.29 (0.14)<br>48.2 (04.0)<br>**0.23** (0.11)<br>**54.3** (14.9)<br>0.34 (0.16)<br>56.8 (04.5)<br>0.27 (0.15)<br>52.0 (00.5)<br>**0.19** (0.07)<br>**65.3** (15.9)<br>0.26 (0.13)|
|SciQr<br>QASCr_∗_<br>36.0 (01.1)<br>0.08 (0.01)<br>SciQr<br>68.5 (01.3)<br>0.07 (0.01)|49.0 (04.5)<br>0.35 (0.10)<br>32.4 (05.3)<br>**0.10** (0.01)<br>**58.1** (01.9)<br>0.25 (0.06)<br>70.6 (07.8)<br>0.23 (0.03)<br>72.6 (04.0)<br>0.15 (0.06)<br>**82.4** (02.1)<br>**0.13** (0.04)|
|SST2<br>FinS_∗_<br>60.2 (03.0)<br>0.10 (0.02)<br>PoemS_∗_<br>57.5 (01.8)<br>0.11 (0.02)<br>SST2<br>77.0 (02.2)<br>0.10 (0.02)|60.0 (13.6)<br>0.33 (0.11)<br>53.2 (20.4)<br>0.33 (0.04)<br>**89.8** (09.0)<br>**0.09** (0.08)<br>50.7 (03.7)<br>0.41 (0.07)<br>61.2 (12.5)<br>0.22 (0.06)<br>**84.8** (08.1)<br>**0.14** (0.08)<br>53.5 (05.0)<br>0.34 (0.11)<br>64.9 (15.3)<br>0.30 (0.05)<br>**90.4** (01.3)<br>**0.09** (0.02)|


Table 12: Performance report for _N_ = 8 on Llama-3.2 models trained using LoRA with ground
truth tokens on single-token datasets.


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT IA2 + SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|AGN<br>AGN<br>29.1 (01.0)<br>0.12 (0.01)<br>BBCN_∗_<br>32.6 (03.8)<br>0.11 (0.03)|23.4 (02.6)<br>0.65 (0.14)<br>24.4 (00.4)<br>**0.11** (0.04)<br>**24.4** (00.5)<br>0.57 (0.15)<br>24.2 (00.4)<br>0.43 (0.11)<br>32.4 (07.0)<br>0.66 (0.06)<br>32.1 (07.0)<br>**0.11** (0.06)<br>31.9 (06.9)<br>0.42 (0.20)<br>**32.6** (06.4)<br>0.61 (0.10)|
|FinS<br>FinS<br>57.7 (03.1)<br>0.13 (0.01)<br>PoemS_∗_<br>47.9 (02.6)<br>0.15 (0.03)<br>SST2_∗_<br>55.7 (02.7)<br>0.10 (0.01)|65.2 (03.0)<br>0.31 (0.02)<br>61.4 (12.4)<br>**0.14** (0.05)<br>67.2 (00.8)<br>0.29 (0.04)<br>**67.4** (00.5)<br>0.26 (0.08)<br>47.3 (02.2)<br>0.49 (0.09)<br>48.2 (04.0)<br>**0.23** (0.11)<br>**49.3** (06.2)<br>0.45 (0.15)<br>48.8 (05.1)<br>0.36 (0.12)<br>52.6 (01.0)<br>0.42 (0.10)<br>52.0 (00.5)<br>**0.19** (0.07)<br>54.3 (04.2)<br>0.42 (0.09)<br>**54.8** (05.2)<br>0.32 (0.10)|
|SciQr<br>QASCr_∗_<br>36.0 (01.1)<br>0.08 (0.01)<br>SciQr<br>68.5 (01.3)<br>0.07 (0.01)|41.5 (09.0)<br>0.42 (0.09)<br>32.4 (05.3)<br>**0.10** (0.01)<br>**53.4** (05.2)<br>0.25 (0.08)<br>42.6 (02.6)<br>0.30 (0.04)<br>60.5 (12.0)<br>0.34 (0.12)<br>72.6 (04.0)<br>**0.15** (0.06)<br>**73.6** (09.9)<br>0.21 (0.10)<br>60.1 (03.0)<br>0.29 (0.04)|
|SST2<br>FinS_∗_<br>60.2 (03.0)<br>0.10 (0.02)<br>PoemS_∗_<br>57.5 (01.8)<br>0.11 (0.02)<br>SST2<br>77.0 (02.2)<br>0.10 (0.02)|46.7 (16.8)<br>0.49 (0.20)<br>53.2 (20.4)<br>**0.33** (0.04)<br>**64.4** (28.3)<br>0.33 (0.27)<br>60.7 (24.2)<br>0.35 (0.26)<br>52.8 (03.4)<br>0.42 (0.07)<br>61.2 (12.5)<br>**0.22** (0.06)<br>**66.7** (19.2)<br>0.32 (0.19)<br>63.1 (15.5)<br>0.29 (0.11)<br>52.3 (04.3)<br>0.38 (0.14)<br>64.9 (15.3)<br>0.30 (0.05)<br>64.7 (19.3)<br>0.33 (0.19)<br>**66.9** (18.4)<br>**0.28** (0.21)|


Table 13: Performance report for _N_ = 8 on Llama-3.2 models trained using LoRA with ICL
response tokens on single-token datasets.


Dataset Adaptation Method

|w/o ICL ICL<br>Source Eval<br>acc ↑ acc ↑|SFT only IA2 only IA2 →SFT<br>acc ↑ acc ↑ acc ↑|
|---|---|
|GSM8K<br>GSM8K<br>14.6 (00.0)<br>34.6 (00.9)<br>GSM8Ks_∗_<br>14.8 (00.0)<br>27.8 (01.0)|22.8 (02.8)<br>29.3 (05.0)<br>**31.5** (01.9)<br>20.4 (03.3)<br>24.2 (01.9)<br>**25.1** (03.4)|
|SciQ<br>SciQ<br>02.2 (00.0)<br>10.0 (00.7)|06.3 (04.3)<br>00.0 (00.1)<br>**12.0** (04.7)|


Table 14: Performance report for _N_ = 4 on Llama-3.2 models trained using LoRA using ground
truth tokens on multi-token datasets.


Dataset Adaptation Method


w/o ICL ICL SFT only **IA2** only **IA2** _→_ SFT **IA2** + SFT
Source Eval
acc _↑_ acc _↑_ acc _↑_ acc _↑_ acc _↑_ acc _↑_


GSM8K 14.6 (00.0) 34.6 (00.9) 28.2 (04.5) 29.3 (05.0) 30.3 (03.2) **37.0** (01.9)
GSM8K
GSM8Ks _[∗]_ 14.8 (00.0) 27.8 (01.0) 23.9 (03.0) 24.2 (01.9) 25.5 (02.9) **27.9** (03.1)


SciQ SciQ 02.2 (00.0) 10.0 (00.7) 09.5 (05.8) 00.0 (00.1) 13.2 (07.8) **14.8** (08.2)


Table 15: Performance report for _N_ = 4 on Llama-3.2 models trained using LoRA with ICL
response tokens on multi-token datasets.


21


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT IA2 + SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|SciQr<br>QASCr_∗_<br>35.6 (01.4)<br>0.06 (0.01)<br>SciQr<br>60.0 (01.8)<br>0.07 (0.01)|25.9 (03.7)<br>0.23 (0.09)<br>34.2 (02.7)<br>0.08 (0.01)<br>**41.5** (03.6)<br>0.28 (0.13)<br>33.8 (03.0)<br>**0.08** (0.00)<br>53.4 (06.6)<br>0.16 (0.07)<br>**63.0** (04.6)<br>**0.10** (0.04)<br>54.2 (07.4)<br>0.18 (0.08)<br>61.9 (04.2)<br>0.11 (0.03)|
|SST2<br>FinS_∗_<br>56.6 (01.6)<br>0.17 (0.01)<br>PoemS_∗_<br>52.3 (03.0)<br>0.19 (0.02)<br>SST2<br>60.3 (02.8)<br>0.19 (0.02)|49.2 (15.2)<br>0.32 (0.15)<br>**61.9** (09.5)<br>**0.18** (0.06)<br>50.7 (12.8)<br>0.30 (0.13)<br>52.0 (16.4)<br>0.39 (0.24)<br>50.1 (03.4)<br>0.48 (0.04)<br>48.1 (03.4)<br>**0.11** (0.04)<br>50.8 (03.8)<br>0.40 (0.14)<br>**50.9** (06.3)<br>0.17 (0.06)<br>50.4 (02.1)<br>0.40 (0.10)<br>**54.2** (04.2)<br>**0.15** (0.06)<br>50.4 (02.1)<br>0.45 (0.09)<br>52.0 (02.5)<br>0.20 (0.09)|


Table 16: Performance report for _N_ = 4 on Llama-3.2 models trained using (IA) [3] with ICL
response tokens on single-token datasets.


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|SciQr<br>QASCr_∗_<br>36.0 (01.1)<br>0.08 (0.01)<br>SciQr<br>68.5 (01.3)<br>0.07 (0.01)|43.4 (06.7)<br>0.31 (0.09)<br>36.2 (01.9)<br>**0.09** (0.01)<br>**57.7** (03.8)<br>0.17 (0.04)<br>70.0 (02.7)<br>**0.09** (0.03)<br>69.5 (04.3)<br>0.14 (0.06)<br>**77.4** (07.5)<br>0.17 (0.08)|
|SST2<br>FinS_∗_<br>60.2 (03.0)<br>0.10 (0.02)<br>PoemS_∗_<br>57.5 (01.8)<br>0.11 (0.02)<br>SST2<br>77.0 (02.2)<br>0.10 (0.02)|58.5 (11.2)<br>0.26 (0.10)<br>59.6 (05.2)<br>0.15 (0.04)<br>**82.1** (07.0)<br>**0.10** (0.08)<br>51.6 (04.5)<br>0.37 (0.06)<br>54.7 (06.2)<br>0.18 (0.04)<br>**81.8** (09.1)<br>**0.14** (0.07)<br>53.5 (02.8)<br>0.25 (0.08)<br>66.5 (10.1)<br>0.24 (0.06)<br>**81.3** (12.1)<br>**0.17** (0.12)|


Table 17: Performance report for _N_ = 8 on Llama-3.2 models trained using (IA) [3] with ground
truth tokens on single-token datasets.


Dataset Adaptation Method

|Source Eval ICL|SFT only IA2 only IA2 →SFT IA2 + SFT|
|---|---|
|acc_ ↑_<br>ece_ ↓_|acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_<br>acc_ ↑_<br>ece_ ↓_|
|SciQr<br>QASCr_∗_<br>36.0 (01.1)<br>0.08 (0.01)<br>SciQr<br>68.5 (01.3)<br>0.07 (0.01)|34.8 (09.3)<br>0.37 (0.11)<br>36.2 (01.9)<br>**0.09** (0.01)<br>**49.3** (08.2)<br>0.39 (0.07)<br>36.0 (00.9)<br>0.09 (0.01)<br>52.0 (07.8)<br>0.26 (0.08)<br>**69.5** (04.3)<br>0.14 (0.06)<br>66.7 (16.1)<br>0.28 (0.16)<br>67.0 (02.1)<br>**0.10** (0.03)|
|SST2<br>FinS_∗_<br>60.2 (03.0)<br>0.10 (0.02)<br>PoemS_∗_<br>57.5 (01.8)<br>0.11 (0.02)<br>SST2<br>77.0 (02.2)<br>0.10 (0.02)|50.0 (15.7)<br>0.38 (0.21)<br>**59.6** (05.2)<br>**0.15** (0.04)<br>54.0 (19.9)<br>0.38 (0.20)<br>51.8 (17.9)<br>0.36 (0.22)<br>54.1 (04.6)<br>0.38 (0.11)<br>54.7 (06.2)<br>**0.18** (0.04)<br>**60.6** (12.0)<br>0.32 (0.17)<br>57.3 (10.4)<br>0.27 (0.06)<br>55.4 (07.6)<br>0.36 (0.15)<br>**66.5** (10.1)<br>**0.24** (0.06)<br>64.5 (18.7)<br>0.31 (0.21)<br>63.6 (17.3)<br>0.26 (0.06)|


Table 18: Performance report for _N_ = 8 on Llama-3.2 models trained using (IA) [3] with ICL
response tokens on single-token datasets.


Dataset Adaptation Method

|w/o ICL ICL<br>Source Eval<br>acc ↑ acc ↑|SFT only IA2 only IA2 →SFT<br>acc ↑ acc ↑ acc ↑|
|---|---|
|GSM8K<br>GSM8K<br>14.6 (00.0)<br>35.9 (01.1)<br>GSM8Ks_∗_<br>14.8 (00.0)<br>30.1 (00.7)|24.6 (02.2)<br>30.9 (02.3)<br>**31.0** (02.5)<br>19.1 (02.0)<br>23.5 (01.2)<br>**24.1** (02.0)|


Table 19: Performance report for _N_ = 8 on Llama-3.2 models trained using (IA) [3] with ground
truth tokens on multi-token datasets.


Dataset Adaptation Method

|w/o ICL ICL<br>Source Eval<br>acc ↑ acc ↑|SFT only IA2 only IA2 →SFT IA2 + SFT<br>acc ↑ acc ↑ acc ↑ acc ↑|
|---|---|
|GSM8K<br>GSM8K<br>14.6 (00.0)<br>35.9 (01.1)<br>GSM8Ks_∗_<br>14.8 (00.0)<br>30.1 (00.7)|26.6 (10.1)<br>30.9 (02.3)<br>34.2 (01.3)<br>**34.3** (02.7)<br>21.4 (06.6)<br>23.5 (01.2)<br>27.4 (01.8)<br>**27.4** (02.2)|


Table 20: Performance report for _N_ = 8 on Llama-3.2 models trained using (IA) [3] with ICL
response tokens on multi-token datasets.


22