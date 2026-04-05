# LARGE LANGUAGE MODELS DO NOT MAKE COMPLETE USE OF MATH REASONING DATA


**Anonymous authors**
Paper under double-blind review


ABSTRACT


In deep learning, increasing dataset size has been shown to improve the performance of deep neural networks. However, it is unclear if these models are able
to make complete use of the data that they are trained on. Understanding this is
especially important in the current large language model era, where data scarcity
has become a pressing issue. We discover that when performing fine-tuning on
mathematical reasoning tasks, adding more training data causes the model to incorrectly answer a large portion of previously correctly answered test samples.
This remains true even with popular test-time scaling techniques, which can iron
out inconsistencies in model predictions. To better understand this phenomenon,
we show both empirically and theoretically that models trained using Supervised
Fine-Tuning and Reinforcement Learning are incapable of making complete use
of the data that they are trained on, where models trained on the same data learn
very different functions across different random seeds, exhibiting high predictive
multiplicity. This work contains novel insights that can aid in improving a model’s
ability to effectively scale its performance with more data.


1 INTRODUCTION


It is generally understood that increasing training data can improve the performance of deep neural
networks (Kaplan et al., 2020; Sorscher et al., 2022). To take advantage of this, an active effort has
been applied to procure more data or generate it synthetically, specifically for settings involving
large language models. However, many recent studies have shown that we are approaching the limits
of human-generated text data (Muennighoff et al., 2023; Tirumala et al., 2023) and suggest that
synthetic data may be a promising direction. Recent works have even attempted to study scaling laws
of synthetically generated data to confirm this hypothesis, where most studies suggest that current
generation techniques produce data that do not match human-generated samples. In an effort to
procure more data, however, a crucial factor has been understudied: _Do deep neural networks make_
_complete use of the data they are provided?_


To answer this question, we begin by performing a deeper analysis of how model performance scales
with an increase in training data. More specifically, rather than simply observing a net increase in
test performance with increasing training data, we study how individual test samples are impacted
by increasing data. We note that our scope is defined to cover large language model training on
math reasoning tasks with supervised fine-tuning and reinforcement learning, a setting where data
scarcity has become a concern. Interestingly, we observe that on addition of more training data, while
test performance rises, a large portion (10-15%) of previously correctly answered test samples are
now incorrectly answered with more data. This behavior is observed even with test-time scaling
techniques such as majority voting, which are used to overcome inconsistencies in models’ outputs
produced through non-deterministic behaviors and elicit better performances.


To better understand this phenomenon, we investigate incorrectly answered sample groups and
perform a fixed set analysis to learn that these models are simply not capable of making complete
use of the training data that they are provided. Large language models trained on the same data
but with different random seeds can learn vastly different functions, where all models have similar
test performances, but the intersection of correctly answered test samples by these models is very
small. Observing that large language models exhibit notably high predictive multiplicity, we link
our empirical findings to established theory and show that this behavior is attributed to an inability


1


Newly Correctly Answered Newly Incorrectly Answered


20


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||


(c) Gemma3-4B on GSM8k (Supervised Fine-tuning)


Figure 1: With additional training data, there exist many previously correctly answered test samples
that become incorrectly answered when trained using supervised fine-tuning.


to make full use of the training data provided. We provide an in-depth understanding of this novel
failure mode of the current large language model training paradigm.


2 BACKGROUND


**Reasoning** **in** **Deep** **Learning.** Recent works aim to understand if deep networks are capable
of drawing out underlying rules from training instances. Early works show that deep networks
instead rely on approximate rules/heuristics or statistical features to solve reasoning tasks, making
them unable to adapt to newer domains that rely on the same set of rules that can correctly solve
the original task (Zhang et al., 2023; Liu et al., 2023; Nikankin et al., 2025). However, with the
advent of more robust reasoning models, it is becoming increasingly evident that these models are
capable of performing reasoning by drawing out rules and learning when to apply them to solve a
problem (DeepSeek-AI et al., 2025), but still struggle to perform certain tasks.


**Mathematical** **Reasoning.** Assessing and understanding the mathematical reasoning ability of
deep neural networks has become an important area of study within the field of deep learning (Zhou
et al., 2024; Lee et al., 2024). (Lample & Charton, 2020) show that deep networks attain good
performances in slightly complex mathematical tasks such as symbolic integration and the solving
of differential equations. Following this, Cobbe et al. (2021) showed that heavily parameterized


2


60


55


50


45


90


80


70


60


52


48


44


40


Train Subset Size (%)


Train Subset Size (%)


15


10


5


0


(a) Llama3-8B on GSM8K (Supervised Fine-tuning)


20


15


10


Train Subset Size (%)


5


0


(b) Llama3-8B on MAWPS (Supervised Fine-tuning)

20


15


10


5


0


Newly Correctly Answered Newly Incorrectly Answered


|Col1|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
||||||
||||||
||||||
||||||


|Col1|Col2|Col3|
|---|---|---|
||||
||||
||<br>||


(b) Qwen2.5-0.5B on MATH8K (ZeroRL)


Figure 2: With additional training data, there exist many previously correctly answered test samples
that become incorrectly answered when trained using reinforcement learning.


models fail when solving math word problems with simple arithmetic tasks and that attaining good
performance generally requires the utilization of additional compute, especially during test time.
Hendrycks et al. (2021) further reinforced this claim and showed that in word problems with more
complex mathematical operations, deep networks are unable to perform. Today’s reasoning models,
although less parameterized than the ones discussed in (Cobbe et al., 2021), are better at solving these
tasks but still struggle to solve many problems within these tasks, even with test-time scaling.


**Math Reasoning Datasets.** We consider 3 math reasoning datasets in this work: MAWPS (KoncelKedziorski et al., 2016), which comprises simple math word problems, GSM8K (Cobbe et al., 2021)
which comprises harder grade school math word problems, and MATH (Hendrycks et al., 2021; Zeng
et al., 2025), which comprises much harder high school math word problems. These datasets are
representative of most math reasoning tasks studied in current literature.


**Inclusion of Reasoning Steps in Training Data.** To improve the mathematical reasoning ability of
deep networks, most datasets comprise of chain-of-thought reasoning steps, where instead of standard
question-answer pairs, solutions often also contain intermediate reasoning steps that eventually arise
at the answer. Inclusion of such intermediate reasoning steps has been shown to improve performance
significantly. In addition to these, many works propose augmenting or transforming training instances
to induce new ways of reasoning about mathematical word problems. Weng et al. (2023); Jiang et al.
(2024) introduce backward reasoning questions for training instances, where models are taught to
reason about a question better by starting from the answer and working back towards an initially
provided value. Yu et al. (2024) aim to augment a training set with rephrased questions that arrive
at the same answer, thereby allowing models to draw out similarities between questions that seem
different on the surface.


**Scaling Test-time Compute.** Practitioners have shown that scaling test-time compute helps improve
performance significantly on math reasoning tasks. Cobbe et al. (2021) show that by training verifiers
and performing ranking of many candidate solutions, deep networks were capable of attaining much
better performances on mathematical reasoning benchmarks. Wang et al. (2023) follow a simple
approach of sampling many candidate solutions and simply picking the most common answer, also
known as majority voting. Others use reward models to score candidate solutions and simply pick
answers with the highest score (Snell et al., 2025).


3


47.5

45.0

42.5

40.0

37.5

35.0


30


25


20


15


Train Subset Size (%)


20


15


10


5


0


(a) Qwen2.5-0.5B on GSM8K (ZeroRL)

20


15


10


Test Subset Size (%)


5


0


Final Model Union


_yi,<t_ = ( _yi,_ 1 _, . . ., yi,t−_ 1).


makes use of the data it is provided.


**Reinforcement** **Learning** **(RL)** **with** **verifiable** **rewards.** Recent work has shown that simple
reinforcement learning with verifiable rewards on base language models can elicit strong reasoning
behaviors (DeepSeek-AI et al., 2025; Zeng et al., 2025; Gandhi et al., 2025). Large language models
are commonly trained with reinforcement learning through Group Relative Policy Optimization
(GRPO) (Shao et al., 2024) and thus, it is the technique used for RL training in this paper. More
specifically, we perform ZeroRL, which is RL training on the base model without any prior supervised
fine-tuning step (Zeng et al., 2025).


**Parameter-Efficient Fine-Tuning.** Parameter-efficient fine-tuning techniques were created to make
it more efficient to fit a model to a particular task. More specifically, it freezes the weights of a model
and trains a much smaller set of extra parameters to reduce memory requirements. LoRA (Hu et al.,
2022), QLoRA (Dettmers et al., 2023), and LoftQ (Li et al., 2024) are the primary PEFT techniques
used in practice. In this work, we make use of LoRA and LoftQ.


3 STANDARD SCALING LAW ANALYSES OBSCURE LOSS OF IMPORTANT
INFORMATION.


We begin our analysis by performing supervised fine-tuning of large language models (Llama3 (Dubey
et al., 2024), Gemma3 (Team et al., 2025)) on math reasoning tasks (GSM8K (Cobbe et al., 2021),
MAWPS (Koncel-Kedziorski et al., 2016). To understand how additional data improves test-time
performance, we fine-tune the model on increasing subsets of the dataset, where every time we
increase the subset size, the new subset is always a superset of the previous subset. Note that the
subsets are consistent across different random seeds. Consistent with previous analyses, as one scales
the size of the train dataset, test-time performance improves, with large initial increases that plateau
over time (Fig. 1 (Left)), thereby mimicking standard scaling laws (Kaplan et al., 2020; Sorscher
et al., 2022). Note that for initial experimentation, we perform greedy decoding in our evaluation.


4


80

70

60

50

40

30


80


60


40


20


0


95


90


85


80


90

80

70

60

50

40

30


80


60


40


20


0


(c) Gemma3-4B on GSM8K


the entire dataset.


forcing):


Newly Correctly Answered Newly Incorrectly Answered


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||


Figure 4: Even with majority voting, the addition of new training samples causes previously correctly
answered samples to be incorrectly answered. Sn denotes Step number n when increasing the train
set size.


We observe, however, that for every step at which we increase the subset-size considered, a significant
number of samples that were previously correctly answered are now incorrectly answered (Fig. 1
(right column)). In fact, at every step in most settings, more than 10% of the test dataset that was
previously correctly answered becomes incorrectly answered ( **Newly Incorrectly Answered** ). More
interestingly, as we continue to increase the size of the dataset used to fine-tune the model, the number
of samples that are newly correctly answered becomes very similar to the number of samples that
are newly incorrectly answered, thereby resulting in marginal improvements in testing accuracy.
This occurs even with RL-based training, where we perform ZeroRL of Qwen2.5-0.5B (Qwen et al.,
2025) on GSM8K and Math8K (Hendrycks et al., 2021; Zeng et al., 2025) datasets using GRPO
(Fig. 2). Note that we perform greedy decoding in all of these experiments during inference, such
that there is no randomness during evaluation. To further demonstrate how much information is
lost by increasing the size of the train data used, we compare the testing accuracy obtained by the
model trained on the entire dataset (Final Model) against a testing metric where we assume a
test sample is correctly answered if at least one of the models (trained on increasing subsets of the
dataset) answers it correctly (Union) (see Fig. 3). We observe that the testing accuracy obtained by
Final Model is significantly lower than that by Union.


3.1 TEST-TIME SCALING DOES NOT RESOLVE THE PROBLEM


Recent work has shown that during inference, models exhibit a high degree of non-determinism,
where small changes in the sampling strategy can cause big changes in model output. Thus, it is
generally recommended to sample multiple outputs using temperature sampling from the same model
and perform majority voting across these outputs to get the model’s prediction (Wang et al., 2023).
Such sampling enhances consistency in a model’s prediction, overcoming minor fluctuations that
can lead to incorrect answers. Such a technique falls under a broader pool of strategies known as
test-time scaling techniques, used to overcome inconsistencies and even boost performance at times.


The results in Fig. 1 are presented with naive single-sample greedy decoding. To confirm that our
insights are not observed simply due to these fluctuations during inference, we repeat our evaluation
with majority voting (Wang et al., 2023). Consistent with standard practice (Wang et al., 2023; Brown
et al., 2024), we sample 15 outputs and perform majority voting and experiment with temperatures
of 0.6 and 1 with no nucleus sampling. Since we observe worse performance when sampling with
temperature 1 (sampling with high temperature has been observed to hurt performance before (Song
et al., 2025)), we only present results with temperature 0.6. We observe that even with majority
voting, samples continue to be incorrectly answered with the addition of more training data, and
previously learned information is lost (Figs. 4 & 3).


We note that we only stick to majority voting for test-time scaling. Our sole objective is to overcome
inconsistencies in model prediction, to understand if these inconsistencies and minor fluctuations
are the root cause behind the phenomenon of **Newly Incorrectly Answered** samples in Fig. 1. We
learned above, however, that this is not the case. There exist many test-time scaling techniques that
can be used to enhance model performance, but Brown et al. (2024) show that these often provide
competitive or worse performance than other test-time scaling techniques in the settings concerning
math reasoning.


5


15

10

5

0


20


10


0


15

10

5

0


3.2 EXPERIMENTAL DETAILS


3.2.1 SUPERVISED FINE-TUNING


**Llama3-8b** **GSM8K Setting.** We fine-tune a Llama3-8B on the GSM8K dataset (Cobbe et al.,
2021) using LoftQ (Li et al., 2024). Consistent with their implementation, we train for 3 epochs
using a learning rate of 5e-4 with cosine decay and weight decay 0.1. Fine-tuning was performed on
a machine with two Nvidia RTX4090s for 3 seeds.


**Llama3-8b MAWPS Setting.** We fine-tune a Llama3-8B on the MAWPS dataset (Koncel-Kedziorski
et al., 2016) using LoftQ. We use the same hyperparameters used for the Llama3-8b-GSM8K Setting.
Fine-tuning was performed on a machine with two Nvidia RTX4090s for 3 seeds.


**Gemma3-4b GSM8K Setting.** We fine-tune a Gemma3-4B on the GSM8K dataset using LoRA (Hu
et al., 2022). We train for 5 epochs using a learning rate of 5e-4 with cosine decay and weight decay
of 0.1. Fine-tuning was performed on a machine with two Nvidia RTX4090s for 3 seeds.


3.2.2 REINFORCEMENT LEARNING


**Qwen2.5-0.5B GSM8K Setting.** We fine-tune a Qwen2.5-0.5B on the GSM8K dataset in ZeroRL
fashion (Zeng et al., 2025) using GRPO (Shao et al., 2024). We train for 1 epoch using a learning
rate of 5e-6 with cosine decay and weight decay of 0.1. Training was performed on a machine with
one NVIDIA H100 for 1 seed. We do not use any PEFT techniques.


**Qwen2.5-0.5B** **MATH8K** **Setting.** We fine-tune a Qwen2.5-0.5B on the MATH8K
dataset (Hendrycks et al., 2021; Zeng et al., 2025) in ZeroRL fashion using GRPO. We train
for 1 epoch using a learning rate of 5e-6 with cosine decay and weight decay of 0.1. Training
was performed on a machine with one NVIDIA H100 for 1 seed. We evaluate on the MATH500
dataset (Lightman et al., 2024; Zeng et al., 2025), which is a representative subset of the MATH test
set. We do not use any PEFT techniques.


4 LARGE LANGUAGE MODELS MAKE INCOMPLETE USE OF MATH
REASONING DATA


At first glance, the results in Fig. 1
can be perceived as though additional

Per Seed Intersection

training data is causing previously cor
rectly answered due to some seem- 12

ever, that this is not the case. When
we perform the same experiment over Figure 5: The intersection of the newly incorrectly answered
multiple different seeds, the newly in- samples at step 5 across three seeds is less than one-fifth of
correctly answered samples are very the average number of newly incorrectly answered samples
different across models. In other at step 5 in all settings.
words, adding the _same training sam-_
_ples_ causes different sets of test samples to be incorrectly answered. In our setting, the intersection of all the newly incorrectly answered
sets by models obtained at Step 5 in Fig. 1 is less than one-fifth of all the samples that are newly
incorrectly answered, as shown in Fig. 5. Due to the randomness in the newly incorrectly answered
samples, it becomes evident that this occurs not because of sample-to-sample conflicts. This raises
the following question: _If additional training data does not conflict with previously attained data,_
_then why are so many previously correctly answered samples now incorrectly answered?_


Per Seed Intersection


3.0

2.5

2.0

1.5

1.0

0.5

0.0


15.0

12.5

10.0


7.5

5.0

2.5

0.0


12

10


8

6

4

2

0


Figure 5: The intersection of the newly incorrectly answered
samples at step 5 across three seeds is less than one-fifth of
the average number of newly incorrectly answered samples
at step 5 in all settings.


6


Per Seed Intersection


similar testing accuracies but correctly answer very diverse test samples.


4.1 FIXED SET ANALYSIS


Since additional training samples are not causing any test samples to be newly incorrectly answered
due to conflicts, we hypothesize that the primary reason behind this is due to the information richness
of math reasoning datasets, causing these models to learn a wide variety of different strategies and
solutions. To confirm this, we train the same Llama3-8B on the entire train dataset but with different
random seeds. More specifically, we train these models on the same sample set of the GSM8K dataset
across three random seeds. We hypothesize and then observe that even if one were to train these
models on the same fixed dataset but across _different random seeds_, they would correctly answer
widely different samples within the test set, as shown in Fig. 6. In other words, the model encodes
different functions over different training runs, even if it is trained on the same data. This remains
true even if the sample set used to train is only a small fraction (3.3-3.5%) of the entire train set, as
shown in Fig. 6. Note that we provide the intersection of correctly classified samples across the three
seeds. **Thus, while the training data can be used to correctly answer many samples within the**
**test set, each model only correctly answers a small subset of the test set.** **This shows that these**
**models do not make complete use of their training data.**


4.2 LARGE LANGUAGE MODELS EXHIBIT HIGH PREDICTIVE MULTIPLICITY ON MATH
REASONING TASKS.


In this section, we aim to better understand why these models, trained on the same data, correctly
answer a very diverse set of test samples. First, we show how this phenomenon ties into existing
theory and explain that these models learn different strategies per sample for math reasoning tasks.
Next, we present a simple framework to show why there exist multiple models that have similar
testing accuracies but correctly answer very different sets of test samples. Finally, we perform
ablation studies, which show that simple changes to the training recipe can cause significant changes
in the final learned function.


**Training on Math Reasoning Tasks Leads to Large Rashomon Sets.** We observe that all models
trained on the same data but with different random seeds obtain almost the same testing accuracies
but differ significantly on per-sample correctness, leading to a low intersection on correctly answered
samples. Such predictive multiplicity has been well documented in literature, commonly referred to
as the _Rashomon effect_ (Breiman, 2001; Marx et al., 2020; Semenova et al., 2023; Rudin et al., 2024).
The Rashomon effect is a phenomenon where multiple models obtain similar empirical risk but differ
significantly in per-sample predictions. A Rashomon set (also known as _ε_ -level set) is the set of all
models that obtain similar empirical risk constrained by some slack _ε_ .


**Definition 1 (Rashomon set)** _Let_ _S_ = _{_ ( _xj, yj_ ) _}_ _[n]_ _j_ =1 _[be]_ _[a]_ _[dataset]_ _[and]_ _[R]_ [ˆ] _[S]_ [(] _[h]_ [)] _[the]_ _[empirical]_ _[risk.]_
_For a baseline empirical risk minimizer h_ 0 _∈_ arg min _h∈H_ _R_ [ˆ] _S_ ( _h_ ) _and ε_ _≥_ 0 _, the Rashomon set is_
_defined as_
_Sε_ ( _h_ 0) = _{ h ∈H_ : _R_ [ˆ] _S_ ( _h_ ) _≤_ _R_ [ˆ] _S_ ( _h_ 0) + _ε }._


where _H_ is the hypothesis space for that task and _h_ is any model within the hypothesis space.


Note that in our setting, we define the Risk _R_ [ˆ] as 0-1 error on the test dataset _S_ as follows:


7


60


50


40


30


20


100

90

80

70

60

50

40


50


40


30


20


50


40


30


20


90

80

70

60

50

40


70


60


50


40


30


20


for _n_ (1 _−_ _P_ ) _≥_ _δε/_ 2 and _n_ ( _P_ ) _≥_ _δε/_ 2, where _M_ _⊂_ _K_ is the set of incorrect strategies per sample.


8


_R_ ˆ _S_ ( _h_ ) = [1]

_n_


_n_

- **1** [ _h_ ( _xi_ ) _̸_ = _yi_ ] _._


_i_ =1


**Definition 2 (Discrepancy)** _For every model h ∈_ _Sε_ ( _h_ 0) _, we define discrepancy as_


_δε_ ( _h_ ) =


_n_

- **1** [ _h_ ( _xi_ ) _̸_ = _h_ 0( _xi_ ) ] _._


_i_ =1


**Existence of Multiple Learnable Strategies.** Math reasoning problems are generally multi-step
reasoning problems, where individual steps can be swapped or replaced entirely, while still resulting
in the same correct prediction. This alone can lead to a large strategy space for a given test set. This
space blows up even further when one considers the number of incorrect strategies that can be learned
per sample.


We observe that across 10 different runs in the Llama3B-GSM8K setting, the average number of
unique strategies per test sample was **5.32**, with the average number of unique incorrect strategies
equal to **3.15** . We define strategies as the sequence of mathematical operations within the model’s
reasoning trace. Note that we used greedy decoding for all 10 runs.


**Definition 3 (Strategy Set)** _For every sample, we define the strategy set K_ _as the set of all unique_
_operation sequences that yield completion (correct or incorrect) of the question._


We extract strategies from a model’s generated reasoning trace by simply extracting the operations
in their appeared sequence while discarding other information. We now consider two settings that
present a simple framework to show how the number of permissibly allowed models within the
Rashomon set explodes with only a small increase in _|K|_ .


**Setting 1 (Budget Based** _ε_ **-Permissibility)** _We assume the simplified setting where every model h_
_within the Rashomon set makes δε_ _mistakes on baseline-correct items but does not correct baseline-_
_incorrect items._


Given _R_ [ˆ] _S_ ( _h_ 0) = _P_, the number of baseline-correct samples which can be flipped becomes - _n_ (1 _δ−ε_ _P_ )�.
Thus, the number of ways in which _h ∈_ _Sε_ ( _h_ 0) can flip baseline-correct samples becomes:


- _n_ (1 _−_ _P_ )
_δε_


_|M_ _|_ _[δ][ε]_


for _n_ (1 _−_ _P_ ) _≥_ _δε_, where _M_ _⊂_ _K_ is the set of incorrect strategies per sample. This gives us the
number of permissible models within the Rashomon set, based on the budget _ε_ .


**Setting 2 (Trades Based** _ε_ = 0 **)** _Here, we assume a setting where ε tends to zero, but models dis-_
_agree significantly on predictions._ _This is what we empirically observe in Fig. 6 (Final Model), with_
_models attaining similar or the same testing accuracies but low intersection on correctly classified_
_samples._


For _ε_ = 0, the necessary condition is that the number of baseline-correct flips made by any _h ∈_ _Sε_ ( _h_ 0)
must be equal to the number of baseline-incorrect flips made by that model _h_ . For a given discrepancy
_δε_, the number of baseline-correct and baseline-incorrect samples which can be flipped becomes

- _n_ (1 _δε−/_ 2 _P_ )� and - _nδε_ ( _P/_ 2 )�, respectively. Thus, the number of permissible models within _Sε_ ( _h_ 0) and equal

risk _R_ [ˆ] _S_ ( _h_ 0) = _P_ becomes:


- _n_ (1 _−_ _P_ )
_δε/_ 2


�� _n_ ( _P_ )�

_|M_ _|_ _[δ][ε][/]_ [2] _|K −_ _M_ _|_ _[δ][ε][/]_ [2]

_δε/_ 2


The GSM8K dataset has 1319 test samples, and Llama3-8B in our setting attains a Risk _R_ [ˆ] of 0.38
on average. Considering these, it is evident that in both settings, minor increases in _|K|_ can cause a
significant increase in _|Sε_ ( _h_ 0) _|_, thereby exacerbating predictive multiplicity significantly. Note that
we assume that per-sample strategies are independent of each other in that a change in strategy for
one sample does not impact a change in strategy for another sample.


We observe that on fixing the order in which the model observes samples during training across three
seeds, while also removing the LoRA dropout, the model learns the same function, resulting in a
large intersection of correctly classified samples, as shown in Fig. 7 (the size of intersection set is
equal to the size of correctly answered sample set of any of the three models). **These experiments**
**show how small changes in the training recipe (sample order and LoRA dropout application)**
**can cause models to learn widely different functions.**


5 CONCLUSION


In this work, we answer an important question: _Do large language models make complete use of the_
_training data that they are provided?_ Through comprehensive experiments across a range of models
and math reasoning datasets trained using both supervised fine-tuning and reinforcement learning,
we learn that they do not. First, we observe that model performance scales poorly with increasing
dataset sizes, where adding more data causes smaller improvements over time due to the incorrect
prediction of previously correctly answered samples. We then show that this happens not because
of sample conflicts but because these models exhibit high predictive multiplicity on math reasoning
tasks, where the same models trained on the same data but across different seeds learn very different
functions. These models attain similar testing accuracies, but they correctly answer very different sets
of test samples. Thus, while the training data can be used to correctly answer many samples within
the test set, each model only correctly answers a small subset of the test set. This shows that these
models do not make complete use of their training data. Finally, we tie this observation to existing
theory and explain why Large Language Model training on math reasoning tasks is bound to have
high predictive multiplicity.


9


**Ablations.** Given the large Rashomon set, it must
follow that even minor changes to the training recipe
while keeping the dataset and all training relevant parameters the same should result in models that differ
significantly. We confirm this experimentally. Through
extensive ablations in the Llama-GSM8K setting, we
discover that there are two factors due to which we
observe the learning of very different functions:


    - **Sample Order.** It is well documented within
the deep learning literature that the order in
which samples are processed during training
plays a very important role in the final function map that is eventually learned. We observe that changing this order across runs but
keeping other factors constant encourages the
learning of very different functions.


    - **LoRA** **Dropout.** In our LoftQ (Li et al.,
2024) setup, we maintain a LoRA dropout ratio of 0.1, whose application varies from one
run to another based on the random seed. Such
variance causes training to be driven in different directions, ultimately resulting in very
different functions being learned.


Standard
Fixing Order


No LoRA
Dropout
Combined


70


60


50


40


30


20


10


0


Figure 7: Discarding or fixing factors contributing to randomness in the training process makes the model learn the same function across seeds. This is evident by the
high intersection of correctly answered samples of Combined. Combined represents
fixing the sample order and discarding the
LoRA dropout. Intersection is taken across
3 seeds.


REFERENCES


Leo Breiman. Statistical modeling: The two cultures. _Statistical Science_, 16(3):199–215, 2001. ISSN
08834237, 21688745.


Bradley C. A. Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré,
and Azalia Mirhoseini. Large language monkeys: Scaling inference compute with repeated
sampling. _CoRR_, abs/2407.21787, 2024. doi: 10.48550/ARXIV.2407.21787. URL [https:](https://doi.org/10.48550/arXiv.2407.21787)
[//doi.org/10.48550/arXiv.2407.21787.](https://doi.org/10.48550/arXiv.2407.21787)


Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi
Wang, Mostafa Dehghani, Siddhartha Brahma, Albert Webson, Shixiang Shane Gu, Zhuyun Dai,
Mirac Suzgun, Xinyun Chen, Aakanksha Chowdhery, Sharan Narang, Gaurav Mishra, Adams
Yu, Vincent Y. Zhao, Yanping Huang, Andrew M. Dai, Hongkun Yu, Slav Petrov, Ed H. Chi, Jeff
Dean, Jacob Devlin, Adam Roberts, Denny Zhou, Quoc V. Le, and Jason Wei. Scaling instructionfinetuned language models. _CoRR_, abs/2210.11416, 2022. doi: 10.48550/ARXIV.2210.11416.
[URL https://doi.org/10.48550/arXiv.2210.11416.](https://doi.org/10.48550/arXiv.2210.11416)


Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser,
Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John
Schulman. Training verifiers to solve math word problems. _arXiv preprint arXiv:2110.14168_,
2021.


DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu,
Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu,
Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao
Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan,
Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao,
Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding,
Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang
Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong,
Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao,
Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang,
Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang,
Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L.
Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shiyu Wang,
Shuiping Yu, Shunfeng Zhou, Shuting Pan, S. S. Li, Shuang Zhou, Shaoqing Wu, Shengfeng
Ye, Tao Yun, Tian Pei, Tianyu Sun, T. Wang, Wangding Zeng, Wanjia Zhao, Wen Liu, Wenfeng
Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, W. L. Xiao, Wei An, Xiaodong Liu, Xiaohan
Wang, Xiaokang Chen, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xinyu Yang,
Xinyuan Li, Xuecheng Su, Xuheng Lin, X. Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen,
Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Xianzu Wang, Xinxia Shan, Y. K. Li,
Y. Q. Wang, Y. X. Wei, Yang Zhang, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang,
Yi Yu, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Yishi Piao, Yisong Wang, Yixuan Tan,
Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yuan Ou, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia
He, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Y. X. Zhu, Yanhong
Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Ying Tang, Yukun Zha,
Yuting Yan, Z. Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang,
Zhewen Hao, Zhicheng Ma, Zhigang Yan, Zhiyu Wu, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li,
Ziwei Xie, Ziyang Song, Zizheng Pan, Zhen Huang, Zhipeng Xu, Zhongyu Zhang, and Zhen
Zhang. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning, 2025.
[URL https://arxiv.org/abs/2501.12948.](https://arxiv.org/abs/2501.12948)


Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. Qlora: Efficient finetuning of
quantized llms. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and
Sergey Levine (eds.), _Advances in Neural Information Processing Systems 36:_ _Annual Conference_
_on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December_
_10_ _-_ _16,_ _2023_, 2023. URL [http://papers.nips.cc/paper_files/paper/2023/](http://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)
[hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html.](http://papers.nips.cc/paper_files/paper/2023/hash/1feb87871436031bdc0f2beaa62a049b-Abstract-Conference.html)


Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha
Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn,


10


Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston
Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron,
Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris
McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton
Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David
Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes,
Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip
Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme
Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu,
Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan
Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet
Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng
Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park,
Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya
Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. The llama 3 herd
of models. _CoRR_, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL [https:](https://doi.org/10.48550/arXiv.2407.21783)
[//doi.org/10.48550/arXiv.2407.21783.](https://doi.org/10.48550/arXiv.2407.21783)


Kanishk Gandhi, Ayush Chakravarthy, Anikait Singh, Nathan Lile, and Noah D. Goodman. Cognitive
behaviors that enable self-improving reasoners, or, four habits of highly effective stars. _CoRR_,
abs/2503.01307, 2025. doi: 10.48550/ARXIV.2503.01307. URL [https://doi.org/10.](https://doi.org/10.48550/arXiv.2503.01307)
[48550/arXiv.2503.01307.](https://doi.org/10.48550/arXiv.2503.01307)


Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song,
and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. _NeurIPS_,
2021.


Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International_
_Conference on Learning Representations_, 2022. [URL https://openreview.net/forum?](https://openreview.net/forum?id=nZeVKeeFYf9)
[id=nZeVKeeFYf9.](https://openreview.net/forum?id=nZeVKeeFYf9)


Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James T. Kwok.
Forward-backward reasoning in large language models for mathematical verification. In Lun-Wei
Ku, Andre Martins, and Vivek Srikumar (eds.), _Findings of the Association for Computational_
_Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024_, pp. 6647–6661.
Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS-ACL.397.
[URL https://doi.org/10.18653/v1/2024.findings-acl.397.](https://doi.org/10.18653/v1/2024.findings-acl.397)


Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child,
Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models.
_CoRR_, abs/2001.08361, 2020. [URL https://arxiv.org/abs/2001.08361.](https://arxiv.org/abs/2001.08361)


Rik Koncel-Kedziorski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS:
A math word problem repository. In Kevin Knight, Ani Nenkova, and Owen Rambow (eds.),
_NAACL_ _HLT_ _2016,_ _The_ _2016_ _Conference_ _of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_
_for_ _Computational_ _Linguistics:_ _Human_ _Language_ _Technologies,_ _San_ _Diego_ _California,_ _USA,_
_June_ _12-17,_ _2016_, pp. 1152–1157. The Association for Computational Linguistics, 2016. doi:
10.18653/V1/N16-1136. [URL https://doi.org/10.18653/v1/n16-1136.](https://doi.org/10.18653/v1/n16-1136)


Guillaume Lample and François Charton. Deep learning for symbolic mathematics. In _International_
_Conference on Learning Representations_, 2020. [URL https://openreview.net/forum?](https://openreview.net/forum?id=S1eZYeHFDS)
[id=S1eZYeHFDS.](https://openreview.net/forum?id=S1eZYeHFDS)


Nayoung Lee, Kartik Sreenivasan, Jason D. Lee, Kangwook Lee, and Dimitris Papailiopoulos.
Teaching arithmetic to small transformers. In _The Twelfth International Conference on Learning_
_Representations_, 2024. [URL https://openreview.net/forum?id=dsUB4bst9S.](https://openreview.net/forum?id=dsUB4bst9S)


Yixiao Li, Yifan Yu, Chen Liang, Nikos Karampatziakis, Pengcheng He, Weizhu Chen, and Tuo
Zhao. Loftq: LoRA-fine-tuning-aware quantization for large language models. In _The Twelfth_
_International Conference on Learning Representations_, 2024. [URL https://openreview.](https://openreview.net/forum?id=LzPWWPAdY4)
[net/forum?id=LzPWWPAdY4.](https://openreview.net/forum?id=LzPWWPAdY4)


11


Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan
Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In _The Twelfth_
_International Conference on Learning Representations_, 2024. [URL https://openreview.](https://openreview.net/forum?id=v8L0pN6EOi)
[net/forum?id=v8L0pN6EOi.](https://openreview.net/forum?id=v8L0pN6EOi)


Bingbin Liu, Jordan T. Ash, Surbhi Goel, Akshay Krishnamurthy, and Cyril Zhang. Transformers learn shortcuts to automata. In _The_ _Eleventh_ _International_ _Conference_ _on_ _Learning_
_Representations,_ _ICLR_ _2023,_ _Kigali,_ _Rwanda,_ _May_ _1-5,_ _2023_ . OpenReview.net, 2023. URL
[https://openreview.net/pdf?id=De4FYqjFueZ.](https://openreview.net/pdf?id=De4FYqjFueZ)


Charles Marx, Flavio Calmon, and Berk Ustun. Predictive multiplicity in classification. In Hal Daumé
III and Aarti Singh (eds.), _Proceedings of the 37th International Conference on Machine Learning_,
volume 119 of _Proceedings of Machine Learning Research_, pp. 6765–6774. PMLR, 13–18 Jul
2020.


Niklas Muennighoff, Alexander M Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra
Piktus, Sampo Pyysalo, Thomas Wolf, and Colin Raffel. Scaling data-constrained language
models. In _Thirty-seventh Conference on Neural Information Processing Systems_, 2023. URL
[https://openreview.net/forum?id=j5BuTrEj35.](https://openreview.net/forum?id=j5BuTrEj35)


Yaniv Nikankin, Anja Reusch, Aaron Mueller, and Yonatan Belinkov. Arithmetic without algorithms:
Language models solve math with a bag of heuristics. In _The Thirteenth International Confer-_
_ence on Learning Representations_, 2025. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=O9YTt26r2P)
[O9YTt26r2P.](https://openreview.net/forum?id=O9YTt26r2P)


Qwen, :, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan
Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang,
Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin
Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tianyi
Tang, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan,
Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report, 2025. URL
[https://arxiv.org/abs/2412.15115.](https://arxiv.org/abs/2412.15115)


Cynthia Rudin, Chudi Zhong, Lesia Semenova, Margo I. Seltzer, Ronald Parr, Jiachang Liu, Srikar
Katta, Jon Donnelly, Harry Chen, and Zachery Boner. Position: Amazing things come from having
many good models. In _Forty-first International Conference on Machine Learning, ICML 2024,_
_Vienna, Austria, July 21-27, 2024_ . OpenReview.net, 2024. [URL https://openreview.net/](https://openreview.net/forum?id=oFDFGd9Age)
[forum?id=oFDFGd9Age.](https://openreview.net/forum?id=oFDFGd9Age)


Lesia Semenova, Harry Chen, Ronald Parr, and Cynthia Rudin. A path to simpler models starts
with noise. In _Thirty-seventh Conference on Neural Information Processing Systems_, 2023. URL
[https://openreview.net/forum?id=Uzi22WryyX.](https://openreview.net/forum?id=Uzi22WryyX)


Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu,
and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language
models. _CoRR_ [, abs/2402.03300, 2024. URL https://doi.org/10.48550/arXiv.2402.](https://doi.org/10.48550/arXiv.2402.03300)
[03300.](https://doi.org/10.48550/arXiv.2402.03300)


Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute
optimally can be more effective than scaling parameters for reasoning. In _The Thirteenth Inter-_
_national Conference on Learning Representations_, 2025. [URL https://openreview.net/](https://openreview.net/forum?id=4FWAwZtd2n)
[forum?id=4FWAwZtd2n.](https://openreview.net/forum?id=4FWAwZtd2n)


Yifan Song, Guoyin Wang, Sujian Li, and Bill Yuchen Lin. The good, the bad, and the greedy:
Evaluation of llms should not ignore non-determinism. In Luis Chiruzzo, Alan Ritter, and
Lu Wang (eds.), _Proceedings of the 2025 Conference of the Nations of the Americas Chapter of_
_the Association for Computational Linguistics:_ _Human Language Technologies, NAACL 2025 -_
_Volume 1:_ _Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025_, pp. 4195–4206.
Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025.NAACL-LONG.211.
[URL https://doi.org/10.18653/v1/2025.naacl-long.211.](https://doi.org/10.18653/v1/2025.naacl-long.211)


12


Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari S. Morcos. Beyond neural
scaling laws: beating power law scaling via data pruning. In Alice H. Oh, Alekh Agarwal, Danielle
Belgrave, and Kyunghyun Cho (eds.), _Advances in Neural Information Processing Systems_, 2022.
[URL https://openreview.net/forum?id=UmvSlP-PyV.](https://openreview.net/forum?id=UmvSlP-PyV)


Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej,
Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, Louis Rouillard, Thomas
Mesnard, Geoffrey Cideron, Jean bastien Grill, Sabela Ramos, Edouard Yvinec, Michelle Casbon,
Etienne Pot, Ivo Penchev, Gaël Liu, Francesco Visin, Kathleen Kenealy, Lucas Beyer, Xiaohai
Zhai, Anton Tsitsulin, Robert Busa-Fekete, Alex Feng, Noveen Sachdeva, Benjamin Coleman,
Yi Gao, Basil Mustafa, Iain Barr, Emilio Parisotto, David Tian, Matan Eyal, Colin Cherry, JanThorsten Peter, Danila Sinopalnikov, Surya Bhupatiraju, Rishabh Agarwal, Mehran Kazemi,
Dan Malkin, Ravin Kumar, David Vilar, Idan Brusilovsky, Jiaming Luo, Andreas Steiner, Abe
Friesen, Abhanshu Sharma, Abheesht Sharma, Adi Mayrav Gilady, Adrian Goedeckemeyer, Alaa
Saade, Alex Feng, Alexander Kolesnikov, Alexei Bendebury, Alvin Abdagic, Amit Vadi, András
György, André Susano Pinto, Anil Das, Ankur Bapna, Antoine Miech, Antoine Yang, Antonia
Paterson, Ashish Shenoy, Ayan Chakrabarti, Bilal Piot, Bo Wu, Bobak Shahriari, Bryce Petrini,
Charlie Chen, Charline Le Lan, Christopher A. Choquette-Choo, CJ Carey, Cormac Brick, Daniel
Deutsch, Danielle Eisenbud, Dee Cattle, Derek Cheng, Dimitris Paparas, Divyashree Shivakumar
Sreepathihalli, Doug Reid, Dustin Tran, Dustin Zelle, Eric Noland, Erwin Huizenga, Eugene
Kharitonov, Frederick Liu, Gagik Amirkhanyan, Glenn Cameron, Hadi Hashemi, Hanna KlimczakPluci´nska, Harman Singh, Harsh Mehta, Harshal Tushar Lehri, Hussein Hazimeh, Ian Ballantyne,
Idan Szpektor, Ivan Nardini, Jean Pouget-Abadie, Jetha Chan, Joe Stanton, John Wieting, Jonathan
Lai, Jordi Orbay, Joseph Fernandez, Josh Newlan, Ju yeong Ji, Jyotinder Singh, Kat Black, Kathy
Yu, Kevin Hui, Kiran Vodrahalli, Klaus Greff, Linhai Qiu, Marcella Valentine, Marina Coelho,
Marvin Ritter, Matt Hoffman, Matthew Watson, Mayank Chaturvedi, Michael Moynihan, Min Ma,
Nabila Babar, Natasha Noy, Nathan Byrd, Nick Roy, Nikola Momchev, Nilay Chauhan, Noveen
Sachdeva, Oskar Bunyan, Pankil Botarda, Paul Caron, Paul Kishan Rubenstein, Phil Culliton,
Philipp Schmid, Pier Giuseppe Sessa, Pingmei Xu, Piotr Stanczyk, Pouya Tafti, Rakesh Shivanna,
Renjie Wu, Renke Pan, Reza Rokni, Rob Willoughby, Rohith Vallu, Ryan Mullins, Sammy Jerome,
Sara Smoot, Sertan Girgin, Shariq Iqbal, Shashir Reddy, Shruti Sheth, Siim Põder, Sijal Bhatnagar,
Sindhu Raghuram Panyam, Sivan Eiger, Susan Zhang, Tianqi Liu, Trevor Yacovone, Tyler Liechty,
Uday Kalra, Utku Evci, Vedant Misra, Vincent Roseberry, Vlad Feinberg, Vlad Kolesnikov,
Woohyun Han, Woosuk Kwon, Xi Chen, Yinlam Chow, Yuvein Zhu, Zichuan Wei, Zoltan Egyed,
Victor Cotruta, Minh Giang, Phoebe Kirk, Anand Rao, Kat Black, Nabila Babar, Jessica Lo,
Erica Moreira, Luiz Gustavo Martins, Omar Sanseviero, Lucas Gonzalez, Zach Gleicher, Tris
Warkentin, Vahab Mirrokni, Evan Senter, Eli Collins, Joelle Barral, Zoubin Ghahramani, Raia
Hadsell, Yossi Matias, D. Sculley, Slav Petrov, Noah Fiedel, Noam Shazeer, Oriol Vinyals, Jeff
Dean, Demis Hassabis, Koray Kavukcuoglu, Clement Farabet, Elena Buchatskaya, Jean-Baptiste
Alayrac, Rohan Anil, Dmitry, Lepikhin, Sebastian Borgeaud, Olivier Bachem, Armand Joulin,
Alek Andreev, Cassidy Hardin, Robert Dadashi, and Léonard Hussenot. Gemma 3 technical report,
2025. [URL https://arxiv.org/abs/2503.19786.](https://arxiv.org/abs/2503.19786)


Kushal Tirumala, Daniel Simig, Armen Aghajanyan, and Ari S. Morcos. D4: Improving LLM
pretraining via document de-duplication and diversification. In _Thirty-seventh_ _Conference_ _on_
_Neural Information Processing Systems Datasets and Benchmarks Track_, 2023. [URL https:](https://openreview.net/forum?id=CG0L2PFrb1)
[//openreview.net/forum?id=CG0L2PFrb1.](https://openreview.net/forum?id=CG0L2PFrb1)


Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha
Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language
models. In _The_ _Eleventh_ _International_ _Conference_ _on_ _Learning_ _Representations_, 2023. URL
[https://openreview.net/forum?id=1PL1NIMMrw.](https://openreview.net/forum?id=1PL1NIMMrw)


Jason Wei, Maarten Bosma, Vincent Y. Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan
Du, Andrew M. Dai, and Quoc V. Le. Finetuned language models are zero-shot learners. In
_The_ _Tenth_ _International_ _Conference_ _on_ _Learning_ _Representations,_ _ICLR_ _2022,_ _Virtual_ _Event,_
_April 25-29, 2022_ . OpenReview.net, 2022. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=gEZrGCozdqR)
[gEZrGCozdqR.](https://openreview.net/forum?id=gEZrGCozdqR)


Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun
Zhao. Large language models are better reasoners with self-verification. In _The 2023 Conference_


13


_on Empirical Methods in Natural Language Processing_, 2023. [URL https://openreview.](https://openreview.net/forum?id=s4xIeYimGQ)
[net/forum?id=s4xIeYimGQ.](https://openreview.net/forum?id=s4xIeYimGQ)


Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo
Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for
large language models. In _The Twelfth International Conference on Learning Representations_,
2024. [URL https://openreview.net/forum?id=N8N0hgNDRt.](https://openreview.net/forum?id=N8N0hgNDRt)


Weihao Zeng, Yuzhen Huang, Qian Liu, Wei Liu, Keqing He, Zejun Ma, and Junxian He. Simplerlzoo: Investigating and taming zero reinforcement learning for open base models in the wild. In
_Second Conference on Language Modeling_, 2025.


Honghua Zhang, Liunian Harold Li, Tao Meng, Kai-Wei Chang, and Guy Van den Broeck. On the
paradox of learning to reason from data. In _Proceedings of the Thirty-Second International Joint_
_Conference on Artificial Intelligence, IJCAI 2023, 19th-25th August 2023, Macao, SAR, China_, pp.
3365–3373. ijcai.org, 2023. doi: 10.24963/IJCAI.2023/375.


Hattie Zhou, Arwen Bradley, Etai Littwin, Noam Razin, Omid Saremi, Joshua M. Susskind, Samy
Bengio, and Preetum Nakkiran. What algorithms can transformers learn? a study in length
generalization. In _The Twelfth International Conference on Learning Representations_, 2024. URL
[https://openreview.net/forum?id=AssIuHnmHX.](https://openreview.net/forum?id=AssIuHnmHX)


14


A APPENDIX


A.1 ROLE OF CAPACITY


To better understand the impact of capacity on incomplete use of training data, we conduct the
Gemma3-4B GSM8K experiments with Gemma3-1B and Gemma3-12B. We observe that samples
continue to be newly incorrectly answered with the addition of more data, despite increase in capacity
(Fig. 8)


Newly Correctly Answered Newly Incorrectly Answered


7.5


5.0


2.5


0.0

|Col1|Col2|N<br>N|ewly Corre<br>ewly Incor|ctly Classi<br>rectly Clas|fied<br>sified|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|||||||


Figure 9: Samples are still newly incorrectly answered, even without the use of PEFT techniques
(Llama3.2-3B on GSM8K).


15


20


15


10


5


20


15


10


5


0
Step 1 Step 2 Step 3 Step 4 Step 5


20


15


10


5


0
Step 1 Step 2 Step 3 Step 4 Step 5


0
Step 1 Step 2 Step 3 Step 4 Step 5


Figure 8: Samples continue to be newly incorrectly answered, despite increase in capacity


A.2 FULL SUPERVISED FINE-TUNING


We observe that even without the use of PEFT techniques, samples are still newly incorrectly answered
with the addition of more data, as shown in Fig. 9.


15.0


12.5


10.0