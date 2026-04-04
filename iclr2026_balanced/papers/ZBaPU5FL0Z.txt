# OP-LORA: THE BLESSING OF DIMENSIONALITY WITH OVERPARAMETERIZED LOW-RANK ADAPTATION


**Anonymous authors**
Paper under double-blind review


|Col1|Col2|
|---|---|
|Pretrained<br>Weight|Weight<br>B<br>Weight<br>A|


|Out|tput Generated|
|---|---|
||MLP<br>Weight<br>B<br><br>Weights<br>Weight<br>A|
|Pretrained<br>Weight|Pretrained<br>Weight|


an MLP and a learned embedding. (c) Qualitative image generation results when adapting Stable
Diffusion XL to Naruto (Cervenka, 2022), with OP-LoRA demonstrating higher quality and more
faithful reconstruction of the text prompt than standard LoRA.


ABSTRACT


Low-rank adapters (LoRA) enable finetuning of large models with only a small
number of parameters, reducing storage costs and minimizing the risk of catastrophic forgetting. However, they often suffer from an ill-conditioned loss landscape, leading to difficult optimization. Prior work addresses these challenges by
aligning adapter updates with full finetuning gradients via custom optimizers, but
these methods lack the flexibility to accommodate new adapter architectures and
are computationally expensive. We instead introduce OP-LoRA, a novel method
which replaces each LoRA adapter with weights predicted by an extra MLP, which
is discarded after training. This temporarily allows additional parameters during
training to improve optimization, yet requires less wall time than custom optimizers
and zero extra cost at inference time because the MLP is discarded. Crucially,
extending OP-LoRA to other adapters is as simple as modifying the size of the
prediction head for each new adapter type. Since the additional parameters are used
only during training and thrown away before inference, there is no risk of overfitting
due to increased representational capacity, unlike simply raising the LoRA rank.
Instead, we show that this approach allows the optimization to adaptively increase
or decrease step size, improving performance and decreasing sensitivity to learning
rate. On both small and large-scale LoRA tuning tasks, we observe consistent
performance gains of OP-LoRA relative to LoRA and its variants. We achieve
especially notable improvements in image generation, with OP-LoRA CMMD
scores improving by up to 15 points relative to LoRA.


1 INTRODUCTION


Finetuning large foundation models for specific tasks can provide significant performance gains
but is computationally intensive, with risks of catastrophic forgetting (Ruiz et al., 2024; Cho et al.,
2021a; Biderman et al., 2024). Methods utilizing low-rank adapters (LoRA) (Hu et al., 2022; Hayou


1


et al., 2024; Zhang et al., 2023; Liu et al., 2024; Nikdan et al., 2024; Meng et al., 2024) address
these challenges by modifying the model in rank-constrained ways. This preserves generalization
and reduces interference with pretrained weights (Figure 1a). However, low-rank adapters can make
optimization harder by creating uneven curvature in the loss landscape (Section 3). Even when
optimized with AdamW (Loshchilov & Hutter, 2019; Kingma & Ba, 2015), which preconditions
gradients, poorly conditioned loss landscapes can still pose a problem (Das et al., 2024). These issues
manifest during LoRA training as high sensitivity to learning rates, as shown by Biderman et al.
(2024) and confirmed in Section 3.3.


The most successful attempts to address this issue are custom optimizers such as LoRA-Pro (Wang
et al., 2025) and ScaledAdamW (Wang et al., 2024), which aim to align the LoRA update with that
of full finetuning. However, in implementation, they are complex and difficult to extend to new
LoRA variants (see Limitations of Wang et al. (2025)). For example, adapting these optimizers to
DoRA (Liu et al., 2024) is non-trivial due to weight normalization, which complicates the projection
of full finetuning gradients. They also tend to be more expensive to run than standard optimizers,
requiring matrix inversions and expensive optimizations. In our testing using authors’ code, wall time
of ScaledAdamW is 15% longer than that of OP-LoRA, and LoRA-Pro is up to 14x longer (Section
4.4) when finetuning LLaMA on Commonsense Reasoning tasks. These limitations highlight the
need for alternative optimization strategies that are both effective and architecture-agnostic.


Instead of relying on specialized optimizers, we propose a fundamentally different and more flexible
approach, which we call OP-LoRA (Overparameterized Low-Rank Adaptation). OP-LoRA uses a
small MLP as a hypernetwork (Ha et al., 2017) to predict the low-rank adapter matrices at **train time**
**only** (Figure 1b). In contrast to other hypernetworks (Ruiz et al., 2024; Ortiz-Barajas et al., 2024),
we do not condition on the input sample in any way. This allows us to discard the MLP at inference
time, making inference and storage costs equal to that of standard LoRA. OP-LoRA also retains
exactly the same representational capacity as standard LoRA: any adapter weights produced by the
MLP can be expressed directly with standard LoRA parameters. Despite the higher parameter count
during training, the extra parameters do not increase model capacity, decreasing risk of overfitting.
We show that this "blessing of dimensionality" is due to an increased ability to navigate complex loss
landscapes by an acceleration mechanism (Section 3.2).


Integration of OP-LoRA takes only a few lines of code and generalizes to any LoRA variant. For
example, our OP-DoRA extension of DoRA simply adds a second MLP head to predict its extra
adapter weights, something which would be difficult to do with custom optimizers. OP-LoRA
outperforms standard variants by 1-6% on natural language tasks and up to 15 CMMD points on
image generation tasks (Section 4).


We summarize our contributions as follows:


    - We introduce OP-LoRA, a novel yet easy-to-implement reparameterization of LoRA that
uses an MLP to predict adapter weights instead of learning them directly. After training, the
MLP is discarded so zero additional storage or inference costs are incurred.

    - We show that OP-LoRA navigates loss landscapes better than standard LoRA due to a
built-in acceleration mechanism (Section 3.2).

    - We empirically validate OP-LoRA on a large range of tasks including image and text
generation and show consistent performance gains on both, and a large improvement in
adapting Stable Diffusion relative to standard LoRA (Section 4).


More generally, we believe that train-time over-parameterization represents a promising yet underexplored paradigm in model training, and we hope that our work will catalyze further work.


2 RELATED WORK


**Low-rank finetuning:** Low-rank finetuning, specifically with LoRA (Low-Rank Adaptation) (Huh
et al., 2023), has emerged as a powerful approach for adapting pre-trained models with minimal additional parameters. A number of follow ups have emerged, improving performance.
AdaLoRA (Zhang et al., 2023) prunes weights during training. DoRA (Liu et al., 2024) adds a
magnitude scaling vector to the updated matrix. RoSA (Nikdan et al., 2024) adds a sparse weight
update to the low-rank update, but requires a full-finetuning pass to compute the weight mask.


2


OLoRA(Büyükakyüz, 2024) and PiSSA (Meng et al., 2024) ease optimization by initializing
LoRA orthogonally. However, they are prone to overfitting due to removing important components
from the frozen base weights. Another approach is to make the LoRA optimization trajectory
similar to that of full finetuning: LoRA-GA (Wang et al., 2025) initializes LoRA parameters to
an SVD approximation of the full-finetuning gradient, while LoRA-Pro (Wang et al., 2024) and
ScaledAdamW (Zhang & Pilanci, 2024a) project full-tuning gradients onto the LoRA subspace. Both
have computational overhead resulting in extended training times: LoRA-Pro requires expensive
computations in gradient projection while LoRA-GA requires a full-finetuning pass similar to
RoSA. Deep LoRA (Yaras et al., 2024) learns an over-parameterized LoRA first before compressing
it, leveraging the over-parameterization for improved training. However, the compression is an
expensive process, and therefore impractical at large scale.
**Reparameterization** **with** **hypernetworks:** Ha et al. (2017) generate weights of an LSTM and
CNN from a neural network, introducing the concept of HyperNetworks. However, their focus is
relaxing weight sharing in LSTMs and reducing parameter count in convolutional networks for
image classification. In contrast, we leverage over-parameterization for improved performance.
HyperDreamBooth (Ruiz et al., 2024) generates initializations for LoRA parameters from an
input image. In contrast, we learn our parameter-generating MLP with a learned parameter vector
as input. HyperLoader (Ortiz-Barajas et al., 2024) uses hypernetwork to generate adapters, but
share parameters between layers and tasks. We show that this shared structure severely reduces
performance in Table 3.
**Convergence Properties of Neural Networks:** Understanding the convergence behavior of neural
networks has been a subject of significant research interest(Du et al., 2019; Nguyen & Hein,
2017). Du et al. (2019) find that gradient descent can find global optima in ResNets. Huang et al.
(2020) find that optimizers are biased to flat minima in overparameterized models, and coin the term
the “blessing of dimensionality”. Most relevant to OP-LoRA, though, is Arora et al. (2018)’s work
showing that stacking linear layers can function as an implicit acceleration mechanism in gradient
descent.


3 METHODOLOGY: OVERPARAMETERIZED LORA (OP-LORA)


Low-Rank Adaptation (LoRA) has become a popular strategy for finetuning large models, allowing
adaptation to new tasks by learning a low-rank matrix factorization of weight updates. With LoRA,
finetuning a model layer’s weight matrix _W_ 0 _∈_ R _[d][×][d]_ is achieved by learning an additive low-rank
update ∆ _W_ such that the adapted weights _W_ are given by:

_W_ = _W_ 0 + ∆ _W_ = _W_ 0 + _BA,_ (1)
where _A ∈_ R _[r][×][d]_ and _B_ _∈_ R _[d][×][r]_ are learned low-rank matrices, with _r_ _≪_ _d_, reducing the number of
parameters to learn. However, LoRA introduces challenges during optimization. While the original
parameter space has curvature defined by the Hessian _HW_, the LoRA _A_ matrix has the transformed
Hessian, show as a composition of functional operators:

_HA_ = _B_ _[⊤]_ _◦_ _HW_ _◦_ _B._ (2)


This transformation affects the condition number of the optimization problem in the _A_ -space. Even if
the original Hessian _HW_ is well-conditioned and symmetric positive definite (a reasonable assumption
near local minima), the reparameterized Hessian can become ill-conditioned depending on the singular
values of _B_ . In particular, the condition number _κ_ ( _HA_ ) satisfies:


where _κ_ ( _·_ ) denotes the spectral condition number. A higher condition number indicates a greater
ratio between the largest and smallest curvatures of the loss landscape. In practice, higher condition
numbers lead to slower convergence and greater sensitivity to learning rates. These bounds imply
that even if _HW_ is well conditioned, poor conditioning in _B_ alone can lead to difficulty in optimizing
_A_, since they drive up both upper and lower bounds. Biderman et al. (2024) observe that LoRA
is sensitive to learning rate, which we confirm in Section 3.3. The full derivation of this result,
and a symmetric one for optimizing _A_, is provided in the Appendix. In Section 3.2 and 3.3, we
show OP-LoRA can dynamically adjust step size and do an adaptive line search, overcoming the
optimization difficulties of standard LoRA.


3


  - _κ_ ( _HW_ ) _κ_ ( _B_ ) [2]
max _κ_ ( _B_ ) [2] _[,]_ _κ_ ( _HW_ )


_≤_ _κ_ ( _HA_ ) _≤_ _κ_ ( _B_ ) [2] _· κ_ ( _HW_ ) (3)


Figure 2: **Optimization behavior of LoRA and OP-LoRA for Rotated MNIST classification** :
**a.)** Training loss achieved by LoRA and OP-LoRA as a function of the learning rate, showing
that OP-LoRA attains lower loss across a wide range of learning rates and remains robust even at
suboptimal step sizes. **(b)** and **(c)** Parameter-update trajectories overlaid on the training loss surface
at (b) the optimal learning rate and (c) a low learning rate. In both cases, OP-LoRA descends more
directly towards lower loss (yellow contours) than LoRA.


3.1 PREDICTING LORA WEIGHTS


In order to avoid the optimization difficulties of training LoRA discussed in Section 3, we avoid
directly optimizing _A_ and _B_ by introducing a two-layer MLP which takes as input _z_ and predicts
the entries of _A_ and _B_ . This is a way of reshaping the optimization landscape, making it easier
to optimize. However, because the additional parameters are discarded after training, they **do**
**not increase finetuning capacity** . This is an important distinction with other methods of adding
parameters to the finetuning procedure like increasing LoRA rank. Concretely, we generate _A_ and _B_
as flattened matrices via:

        - _A_         _B_ = **W** 2(ReLU( **W** 1 _z_ + _c_ 1)) + _c_ 2 (4)


where _z_ is the learned input vector to the MLP, **W** and _c_ correspond to learned weights biases, and
_A ∈_ R _[r][×][d]_ and _B_ _∈_ R _[d][×][r]_ are the generated matrices.


Once finetuning is complete, the MLP can be discarded, retaining only the low-rank matrices _A_
and _B_ for inference and storage. Furthermore, as in standard LoRA, _A_ and _B_ can be merged with
the pre-trained model’s weights by adding ∆ _W_ = _BA_ to the relevant layer weights (Equation 1).
Although the MLP is compact in depth, it predicts a high-dimensional output, the size of the LoRA
parameters, increasing its parameter count. For instance, an MLP with a hidden dimension of 32
scales the number of trainable parameters by approximately 32. This makes OP-LoRA particularly
advantageous in settings where inference resources are constrained, but sufficient memory is available
during training. Further, because the MLP is small relative to a typical base model, wall-time penalties
are not large. For more details on computational cost, see Section 4.4.


3.2 OPTIMIZATION BENEFITS OF OP-LORA


Arora et al. (2018) prove that increasing depth by replacing linear layers with products of matrices,
which has the same expressive power as a single matrix due to the linear nature of the transformation,
leads to faster convergence. Although we focus on re-parameterizing with an MLP instead of
increasing depth, we employ the same theoretical framework to examine OP-LoRA’s enhanced
training dynamics below. While we consider the linear case here for clarity, we extend the analysis to
the MLP case in the Appendix.


Consider the OP-LoRA reformulation of parameter vector _v_ with a two layer MLP, defined as
_v_ = **W** 2(ReLU( **W** 1 _z_ + _c_ 1)) + _c_ 2. We can then assign vector _h_ = ReLU( **W** 1 _z_ + _c_ 1), and for
clarity of derivation, we treat _h_ as a free parameter vector; this corresponds to only updating the
bias in the first layer of the MLP. We also merge bias parameters into the parameter matrices **W**
for ease of notation, assuming a constant value is appended to _z_ and _h_ . This leaves us with simpler


4


reparameterization form _v_ = **W** 2 _h_, where _v_ _∈_ R _[p]_ _,_ **W** 2 _∈_ R _[p][×][k]_ _,_ _h_ _∈_ R _[k]_ and _p_ is the size of
generated parameter vector _v_ and _k_ is the hidden dimension of the MLP re-parameterization.


The update rule for **W** 2 and _h_ with learning rate _η_ is given by:

**W** 2 [(] _[t]_ [+1)] = **W** 2 [(] _[t]_ [)] _−_ _η∇_ **W** 2 _,_ _h_ [(] _[t]_ [+1)] = _h_ [(] _[t]_ [)] _−_ _η∇h._ (5)


Then, the parameter vector _v_ at time step _t_ + 1 becomes:


                - ��                _v_ [(] _[t]_ [+1)] = **W** 2 [(] _[t]_ [+1)] _h_ [(] _[t]_ [+1)] = **W** 2 [(] _[t]_ [)] _−_ _η∇_ **W** 2 _h_ [(] _[t]_ [)] _−_ _η∇h_ _._ (6)


Expanding the product, we have:

_v_ [(] _[t]_ [+1)] = **W** 2 [(] _[t]_ [)] **[h]** [(] _[t]_ [)] _[ −]_ _[η][∇]_ **[W]** 2 _[h]_ [(] _[t]_ [)] _[ −]_ _[η]_ **[W]** 2 [(] _[t]_ [)] _[∇][h]_ [ +] _[ η]_ [2] _[∇]_ **[W]** 2 _[∇][h][.]_ (7)


Following (Arora et al., 2018; Zhang & Pilanci, 2024b), we ignore the higher order term since the
learning rate _η_ is assumed to be small and therefore the term shrinks to 0. By definition, _v_ = **W** 2 _h_,
so:
_v_ [(] _[t]_ [+1)] = _v_ [(] _[t]_ [)] _−_ _η∇_ **W** 2 _h_ [(] _[t]_ [)] _−_ _η_ **W** 2 [(] _[t]_ [)] _[∇][h]_ (8)
By substituting chain-rule expansions of _∇_ **W** 2 and _∇h_, the new value for _v_ becomes:

_∇_ **W** 2 = _∇vh_ _[T]_ _,_ _∇h_ = **W** 2 _[T]_ _[∇][v][.]_ (9)


This reveals two key properties of the optimization trajectory under OP-LoRA. First, OP-LoRA
introduces a dynamic learning rate scaling factor, _∥h_ [(] _[t]_ [)] _∥_ [2] . From the update rule _∇h_ = ( **W** 2 [(] _[t]_ [)][)] _[T][ ∇][v]_ [,]
we see that _∥h_ [(] _[t]_ [)] _∥_ [2] grows when the gradient is positively aligned with **W** 2 [(] _[t]_ [)] and shrinks otherwise.
In the special case where _k_ = 1, the vector _h_ becomes a scaling factor for the single column vector
**W** 2 which represents the direction of parameter vector _v_ . Consequently, if the optimizer overshoots
a minimum, the sign of the gradient update for _h_ flips, and the effective learning rate decreases, but
if consecutive updates align, it increases. In the more general _k_ _>_ 1 setting, there are _k_ such scalar
factors (one for each column in **W** 2 [(] _[t]_ [)][)] [and] [each] [can] [increase] [or] [decrease] [independently,] [and] [the]
overall learning rate becomes scaled by _∥h_ [(] _[t]_ [)] _∥_ [2] . We refer to _∥h_ [(] _[t]_ [)] _∥_ [2] as the _**trainable learning rate**_ .

Second, OP-LoRA adds an extra update term **W** 2 [(] _[t]_ [)] �( **W** 2 [(] _[t]_ [)][)] _[T][ ∇][v]_ �, which shifts parameters along
the already learned directions **W** 2 [(] _[t]_ [)][, in proportion to the gradient’s projection onto those directions.]
When _k_ = 1, this can be seen as a gradient step in a line-search, along the direction of the current
state of _v_, and naturally biases the updates toward directions already taken. However, if _∇v_ suddenly
changes to a new and orthogonal direction, the final term immediately vanishes, causing the effective
step size to shrink right away and the update to suddenly shift towards the current gradient. For _k_ _>_ 1,
_v_ becomes a weighted sum over each column of **W** 2 [(] _[t]_ [)][, and the parameter update is biased toward any]
directions in **W** 2 [(] _[t]_ [)] on which the gradient has nonzero projection. We refer to **W** 2 [(] _[t]_ [)] �( **W** 2 [(] _[t]_ [)][)] _[T][ ∇][v]_ - as
_**adaptive line search**_ . Together, the _**adaptive line search**_ and _**trainable learning rate**_ suggest that
OP-LoRA has an improved ability to navigate complex loss landscapes. While the overall trainable
learning rate dynamically updates to the given problem, the adaptive line search can rapidly search the
subspace spanned by _W_ 2. In Section 3.3, we explore how this improves performance in a small-scale
setting.


3.3 MNIST CASE STUDY


To illustrate the advantages of OP-LoRA, we start with a small scale study on MNIST, showing that
OP-LoRA converges to better loss and also is less sensitive to learning rate.


We use a two-layer MLP _f_ ( _x_ ) with hidden dimension 512. We train _f_ on MNIST for 30 epochs to
nearly 0 training loss. This constitutes our base model, which we freeze before continued LoRA
tuning on Rotated MNIST to create an adaptation task. We finetune the frozen _f_ with LoRA and
OP-LoRA adapters of rank _r_ = 4, but this time on the new task of Rotated MNIST, where each
MNIST sample is randomly rotated before classification.


5


_v_ [(] _[t]_ [+1)] = _v_ [(] _[t]_ [)] _−_ _η ∥h_ [(] _[t]_ [)] _∥_ [2] _∇v_ ( _t_ )

        - ��        trainable learning rate


_−_ _η_ **W** 2 [(] _[t]_ [)] �( **W** 2 [(] _[t]_ [)][)] _[T][ ∇][v]_ 
 - ��  adaptive line search


(10)


Figure 3: Comparison of generated images across different low-rank finetuning methods of SD-XL
for two datasets: Naruto (upper) and Claude Monet-style painting (lower). Each row shows outputs
from a specific model configuration based on ground truth captions. For the Naruto prompt, OP-LoRA
and OP-DoRA effectively capture the presence of a hoodie, and generally generate higher fidelity
images. For the Monet-style paintings, OP-LoRA and OP-Dora offer more realistic scenes.


We find two main results. First, we find that OP-LoRA achieves lower training loss than LoRA. In
Figure 2, we can see that train loss for LoRA reaches 0.59, vs 0.54 for OP-LoRA. Second, we find
that OP-LoRA is much less sensitive to learning rates than LoRA (Figure 2 a.), with losses staying
relatively low even with learning rates two to three orders of magnitude smaller than optimal. The
learning rate sensitivity of standard LoRA corroborates to findings in (Biderman et al., 2024) (See
Figure of Biderman et al. (2024)). Qualitatively, we plot the optimization trajectory of both LoRA
and OP-LoRA in Figure 2 b.) and c.). Interestingly, for learning rates which are three orders of
magnitude lower than optimal, OP-LoRA makes good progress, demonstrating OP-LoRA’s ability to
accelerate training. We highlight that this is despite both methods being optimized with Adam, an
optimizer with adaptive learning rates and a momentum term to help with slow learning.


We do additional analysis on the gradient properties of the Hessian using power iteration in Appendix
B.1, where we find that gradient norm in the direction of highest curvature is much higher for
OP-LoRA than standard LoRA.This empirically suggests that OP-LoRA may be less sensitive to poor
conditioning than LoRA, since it can minimize loss effectively even when high curvature requires a
small learning rate. This is consistent with the view that OP-LoRA adaptively reshapes the LoRA
loss landscape for better and faster optimization through reparameterization.


4 EXPERIMENTS


In this section, we show performance on three different tasks. We start by finetuning Stable Diffusion (Podell et al., 2024) in Section 4.1. Low-rank finetuning offers particular advantages in this
context, especially since individual users often have only a few images for model specialization,
making regularization with low-rank updates particularly valuable. Additionally, storage concerns
are significant, as users may want to save numerous specialized model variants. This means that it is
beneficial for any over-parameterization scheme to be training only, to minimize storage costs for the
end user. In Section 4.2, we move onto visual question answering with a VL-Bart (Cho et al., 2021a)
model, where we demonstrate that OP-LoRA and OP-DoRA show consistent improvements over
their standard counterparts. Finally, we show results of a finetuned LLaMA (Touvron et al., 2023)
model on CommonSense reasoning tasks in Section 4.3. This is another common PEFT use-case,
where full finetuning is too expensive and fully-fine tuned models become cumbersome to store.
Details beyond what is provided below are in the Appendix. Finally, in Appendix B, we show results
for improving VeRA(Kopiczko et al., 2024),Mix-of-Show(Gu et al., 2023), and Matrix Factorization
with MLP over-parameterization.


4.1 FINETUNING STABLE DIFFUSION


We finetune Stable Diffusion XL (Podell et al., 2024) for two datasets, a Claude Monet subset of
Wiki-Art (Face & Huggan, 2023) and Blip-Captioned Naruto Images (Cervenka, 2022), and evaluate
using MMD distances over CLIP embeddings.
**Datasets:** **WikiArt (Face & Huggan, 2023)** is a dataset of around 80000 pieces of artwork; each
labeled with artist, genre, and style. We filter by the artist Claude Monet, leaving 1334 images, and


6


**Method** **Naruto** **WArt** **Avg (** _↓_ **)**


_W/ grad._ _alignment_
LoRA-GA _r_ =4 (Wang et al., 2024) **15.2** 43.7 29.5
LoRA-Pro _r_ =4 (Wang et al., 2025) 20.9 32.4 26.7
SAdamW _r_ =4 (Zhang & Pilanci, 2024b) 21.9 **30.5** **26.2**


_W/out grad._ _alignment_
PiSSA _r_ =4 (Meng et al., 2024) 29.7 43.2 36.5
LoRA _r_ =4 (Hu et al., 2022) 23.7 47.9 35.8
DoRA _r_ =4 (Liu et al., 2024) 17.2 49.8 33.5
OP-DoRA _r_ =4 (ours) 11.9 46.6 29.3
OP-LoRA _r_ =4 (ours) **9.6** **31.7** **20.7**


Table 1: Finetuning Stable Diffusion: CMMD
on Naruto and WikiArt (lower is better).


construct text captions for finetune of the form ‘A painting of genre _⟨⟩_ ’, for example ‘A painting
of genre _portrait_ ’. We found a single low-rank adapter to not be able to model many artists jointly.
Importantly, the artist name Claude Monet is not mentioned in the caption, so the Claude Monet
style has to be learned from the data. **Naruto BLIP Captions (Cervenka, 2022)** is a dataset of 1221
anime images from the Japanese manga series Naruto, which are captioned by BLIP (Li et al., 2022).
**Finetuning protocol:** We train for two epochs and we target only the attention layers in the U-Net in
Stable Diffusion XL 1.0. We use rank _r_ = 4 and OP-LoRA MLP width 32. We choose a number of
baselines in addition to LoRA intended to make optimization easier. LoRA-GA (Wang et al., 2024),
LoRA-Pro (Wang et al., 2025) and (Zhang & Pilanci, 2024b) ScaledAdamW leverage full finetuning
gradient information, while PiSSA (Meng et al., 2024) initializes to principle components. DoRA
adds a weight normalization to LoRA in order to improve the learning ability of LoRA. We extend
OP-LoRA to OP-DoRA by adding an additional prediction head to generate the weight scaling factors
along side the low-rank matrices.
**Evaluation Protocol:** For evaluation, we aim to measure how different the distribution of generated
images is from the ground truth distribution. We generate a new image for each training caption. To
assess the quality of the generation, we compute the CLIP Maximum Mean Discrepancy (Jayasumana
et al., 2024) (CMMD) distance, which computes MMD over clip scores. Lower CMMD values
indicate lower distributional distances between generated samples and the ground truth, and therefore
higher quality generations. Jayasumana et al. (2024) show CMMD to be a better measure of generated
image quality than alternatives such as FID Score or Inception Score, aligning better with human
raters and providing more consistent results with varied sample sizes.
**Results:** We present CMMD scores in Table 1. Two interesting trends emerge. First, OP-LoRA
outperforms OP-DoRA on both the Naruto and WikiArt datasets. We attribute this to the increased
ease of overfitting with DoRA. Second, OP-LoRA and OP-DoRA, achieve substantially improved
scores over their standard counterparts. Specifically, OP-LoRA achieves a CMMD score of 9.6 on
the Naruto dataset, compared to 23.7 for LoRA, and similarly, OP-DoRA scores 11.9 compared to
DoRA’s 17.2. On the WikiArt dataset, OP-LoRA also shows a substantial gain with a score of 31.7
compared to 47.9 for LoRA. Furthermore, OP-LoRA outperforms other baseline methods on average,
including state-of-the-art optimizers such as LoRA-Pro (Wang et al., 2025) and ScaledAdamW (Zhang
& Pilanci, 2024b) which leverage information about the full-finetuning gradient. This suggests that
following the full-finetuning gradient as closely as possible is not the only way for parameter efficient
adapters to perform well, and different approaches are worth exploring.


We also show samples of generated images in Figure 3, where we compare LoRA to OP-LoRA and
DoRA to OP-DoRA. We can overall see much higher quality for the over-parameterized variants.
For example, OP-LoRA and OP-DoRA capture the hoodie, while LoRA and DoRA do not. The
still life setting for OP-LoRA is more complex, with flowers. Finally, DoRA seems to generate a
somewhat degenerate image in the second row, while OP-LoRA and OP-DoRA do not. We provide
many random samples in the Appendix.


4.2 VISUAL QUESTION ANSWERING EXPERIMENTS


**Datasets:** **VQAv2** (Goyal et al., 2017) (113K training images) and **GQA** (Hudson & Manning, 2019)
(82.7K training images) are both visual question answering datasets scored. **NLVRv2** (Suhr et al.,


7


**Method** **VQAv2 GQA NLVR Avg** ( _↑_ )


Full FT 66.9 56.7 73.7 65.8


LoRA _r_ =128 (Hu et al., 2022) 65.5 53.9 72.0 63.8
OP-LoRA _r_ =128 (ours) **65.6** **54.9** **73.0** **64.5**


DoRA _r_ =128 (Liu et al., 2024) 65.8 54.9 72.4 64.4
OP-DoRA _r_ =128 (ours) **66.4** **55.1** **74.0** **65.2**


Table 2: VQA evaluation with VL-BART, measuring accuracy. OP-LoRA and OP-DoRA outperform their non-overparameterized counterparts by around 1%.


**Train** **Inf**
**Method**
**Par.** **Par.** **[BoolQ PIQA SIQA HSwag WinoG ARC-E ARC-C OBQA AVG]**


_W/ grad alignment_
LoRA-GA _r_ =32 (Wang et al., 2024) 0.83 0.83 63.0 73.4 75.5 53.0 74.9 66.4 51.7 70.8 66.1
LoRA-Pro _r_ =32 (Wang et al., 2025) 0.83 0.83 69.6 81.6 77.7 **84.4** **80.3** 81.7 65.5 **80.2** 77.6
SAdamW _r_ =32 (Zhang & Pilanci, 2024b) 0.83 0.83 **70.7** **82.3** **78.2** 83.3 79.6 **82.6** **66.4** 78.6 **77.7**
SAdamW _r_ =16 (Zhang & Pilanci, 2024b) 0.41 0.41 **69.9** 81.5 77.3 82.5 79.2 **81.4** 65.2 76.8 76.7
OP-SAdamW _r_ =16(Ours) 13.5 0.41 **69.9** **81.9** **77.8** **84.6** **81.1** 80.1 **65.5** **80.4** **77.7**


_W/Out grad alignment_
DeepLoRA _r_ =32 (Yaras et al., 2024) 0.83 0.83 66.7 78.8 59.8 39.6 51.7 41.6 32.8 39.8 51.4
HLoader _r_ =32 (Ortiz-Barajas et al., 2024) 0.83 0.83 61.5 48.5 43.5 36.3 54.3 26.2 32.1 31.8 41.8
AdaLoRA _r_ =32 (Zhang et al., 2023) 1.25 0.83 67.4 80.7 77.0 47.3 79.6 81.4 64.8 76.2 71.8
DoRA _r_ =32 (Liu et al., 2024) 0.84 0.84 65.3 65.6 76.9 81.2 78.8 79.4 64.0 78.3 73.7
LoRA _r_ =32 (Hu et al., 2022) 0.83 0.83 67.5 80.8 **78.2** 83.4 80.4 78.0 62.6 79.1 76.3
OP-LoRA(Ours) _r_ =32 27.4 0.83 **69.0** 81.4 77.9 85.7 79.2 80.5 64.4 78.6 77.1
OP-DoRA(Ours) _r_ =32 27.7 0.84 67.2 **82.0** 76.3 **86.5** **81.4** **81.5** **65.3** **80.3** **77.5**
LoRA _r_ =16 (Hu et al., 2022) 0.41 0.41 69.9 77.8 75.1 72.1 55.8 77.1 62.2 78.0 70.9
LoRA _r_ =466 (Hu et al., 2022) 12.1 12.1 53.6 70.1 73.4 52.3 71.3 61.4 46.0 64.6 61.6
OP-LoRA(Ours) _r_ =16 13.5 0.41 **70.3** **83.1** **77.7** **77.7** **79.1** **78.9** **63.9** **78.8** **76.2**


Table 3: Finetuning of LLaMA 7B (Touvron et al., 2023) on commonsense reasoning datasets. OPLoRA and OP-DoRA outperform their standard counterparts, and are on par with the complex custom
optimizers such as LoRA-Pro(Wang et al., 2025) and ScaledAdamW (Zhang & Pilanci, 2024b)
despite not leveraging information about the full FT gradient. When combined with ScaledAdamW,
OP-LoRA can match standard ScaledAdamW performance with half the inference parameters.


2019) (206K training images) is a visual reasoning dataset, answering a True/False question about a
pair of images. All are evaluated with accuracy.
**Finetuning and evaluation protocol:** We follow Liu et al. (2024) in our finetuning protocol. We finetune VL-BART (Cho et al., 2021b) in multi-task way. VL-BART composes a CLIP-ResNET101 (Radford et al., 2021) and Bart Base (Lewis et al., 2020) model, and trains all tasks jointly with a languagemodeling loss. We finetune for 20 epochs with rank _r_ = 128, targeting only _Q_ and _K_ matrices
in attention layers, while also training biases. We use MLP hidden layers size 4, based on our
experiments in Section 4.4.
**Results:** We present results in Table 2. We can reach a similar conclusion for the vision-language
task as for the image generation task; OP-LoRA and OP-DoRA improve over their counterparts. The
roughly 1% improvement achieved by the OP variants matches the gains from LoRA to DoRA.


4.3 COMMONSENSE REASONING EXPERIMENTS


**Datasets:** The Commonsense task consists of 8 sub-tasks, with about 170k training sequences
total. **BoolQ** (Clark et al., 2019) is a yes/no question-answering dataset. **PIQA** (Bisk et al., 2020)
requires physical knowledge to answer. **SIQA** (Sap et al., 2019) is about social reasoning for
humans. **HellaSwag** (Zellers et al., 2019) asks the model to complete the context with a sentence.
**WinoGrande** (Sakaguchi et al., 2021) asks the model to fill in the blank. **ARC-E** and **ARC-C** (Clark
et al., 2018) are easy and hard variants of multiple choice science questions. **OBQA** (Mihaylov et al.,
2018) asks multiple choice questions requiring strong comprehension skills of context.
**Finetuning** **and** **evaluation** **protocol:** We follow Liu et al. (2024) and train with all datasets
jointly for 3 epochs, but evaluate each dataset independently. We use _r_ = 32 and MLP width
32. In addition to baselines intended to make optimization easier to optimize by leveraging full
finetuning gradient information (LoRA-GA (Wang et al., 2024), LoRA-Pro (Wang et al., 2025),
SAdamW (Zhang & Pilanci, 2024b) and OP-SAdamW), we also compare to the AdaLora (Zhang
et al., 2023), DeepLoRA (Yaras et al., 2024), and HyperLoader (Ortiz-Barajas et al., 2024). OPSAdamW adds MLP overparameterization to OP-LoRA by inserting gradient projections from
ScaledAdamW to the backward pass into the generating MLP. AdaLoRA dynamically allocates
parameters to different layers, useful for PEFT training of large scale models. Like OP-LoRA,
DeepLoRA (Yaras et al., 2024) is motivated by overparameterization. In its conception, it trains an
over-parameterized adapter and compresses it, but this process is too expensive for large models.
Therefore, Yaras et al. (2024) simply add a third square matrix to the LoRA adapter, therefore
training a product of three matrices. HyperLoader is similar to OP-LoRA, but shares parameters


8


Method GPU Mem Time


LoRA (Hu et al., 2022) 44 GB 3.5 H
OP-LoRA 69 GB 4 H
ScaledAdamW (Zhang & Pilanci, 2024b) 44 GB 4.5 H
LoRA-Pro (Wang et al., 2025) 46 GB 56 H


Table 4: GPU Memory and wall time
cost, evaluated on CommonSense Reasoning on an H100 HBM3 GPU. The
increased memory usage is manageable,
and Wall Time is faster than alternatives.


between LoRA-generating MLPs, and is used to test the necessity of decoupling parameters. Finally,
we expand the rank of LoRA to 466, to verify that performance gains are not simply from naive
adding of parameters.
**Results** Table 3 presents the results of our experiments, where both OP-LoRA and OP-DoRA
consistently outperform their non-overparameterized counterparts by a margin of 1-4%. Moreover,
OP-LoRA nearly matches recent LoRA variants such which align with full finetuning such as LoRAPro and ScaledAdamW, but with lower training time. LoRA-Pro takes 56 hours to complete compared
to the 4 hours for OP-LoRA. Furthermore, we emphasize that although ScaledAdamW and LoRA-Pro
slightly outperform OP-LoRA on commonsense reasoning tasks, they substantially under perform on
image generation tasks (Section 4.1) and that combining ScaledAdamW with OP-LoRA enables us
to reach the same performance but with **half the inference parameters due to lower rank** . This
enables substantially lower inference storage costs. It also decreases inference costs in scenarios
where many LoRAs are served concurrently(Sheng et al., 2023). Interestingly, we can see that
HyperLoader, which is like OP-LoRA but shares MLP parameters between layers, performs very
poorly in our single task context. This supports our design choice to decouple parameters between
LoRA adapters, and that the subspace spanned by _W_ 2 of the OP-LoRA MLP cannot be the same
between adapters. Finally, we can see that expanding rank to 466 is not as performant as a parameter
equivalent OP-LoRA, verifying that HOW we add parameters is important, and not JUST that we
add parameters.


4.4 OP-LORA ANALYSIS


**Computational** **Costs:** OP-LoRA introduces extra train-time parameters that are thrown away
at inference, so there’s no added deployment cost. We evaluate computational cost finetuning
LLaMA-7B for CommonSense reasoning, for 3 epochs. On an H100 HBM3 GPU with adapter
rank = 32, standard LoRA uses 44 GB of GPU memory, whereas OP-LoRA uses 69 GB. The MLP
reparameterization slows training by only about 15%, raising wall-clock time for the entire training
run from 3.5 h to 4 h on the CommonSense benchmark. By contrast, ScaledAdamW takes around
4.5 h, and LoRA-Pro’s heavier computations extend training to around 56 h. Given that OP-LoRA
achieves consistently higher performance than LoRA, we believe this to be manageable. Meanwhile,
it is around 10% faster than ScaledAdamW and more than 10 times faster than LoRA-Pro.


**Ablating MLP Width:** One natural question is how much to over-parameterize low rank adapters.
We study this question for OP-DoRA on the VL-Bart vision-language and commonsense reasoning
tasks, by varying MLP hidden layer size. In Figure 4 we see an inverted U-shape for VL-BART;
too little over-parameterization is not enough but too much starts degrading performance. For
commonsense benchmarks, we see a low score for hidden layer size 1 but otherwise little variation.


5 CONCLUSION


In this work, we introduce OP-LoRA, an MLP-based reparameterization of LoRA. By leveraging
over-parameterization we accelerate training without additional inference overhead. Our experiments
across diverse tasks demonstrate that OP-LoRA consistently improves performance over LoRA. We
believe that train-time over-parameterization represents a promising yet underexplored paradigm in
model training, and we hope that our work will inspire broader investigation into its applications.


9


Figure 4: Effect of MLP hidden layer size for OPDoRA. Performance follows an inverted U-shape
for VL-Bart. For Commonsense reasoning, size 1
is too little but otherwise the trend is flat.


**Reproducibility Statement:**


We provide code in the supplementary implementing OP-LoRA and additional training details are
presented in the Appendix C.


**Ethics Statement:**


OP-LoRA aims to improve the predictive performance of LoRA. This is a general goal that we believe
does not require any special ethical consideration. Nevertheless, machine learning models are a tool
which can be used for good or ill, and we encourage users of OP-LoRA to consider the implications
of any systems built.


REFERENCES


Sanjeev Arora, Nadav Cohen, and Elad Hazan. On the optimization of deep networks: Implicit
acceleration by overparameterization. In _International conference on machine learning_, pp. 244–
253. PMLR, 2018.


Dan Biderman, Jacob Portes, Jose Javier Gonzalez Ortiz, Mansheej Paul, Philip Greengard, Connor Jennings, Daniel King, Sam Havens, Vitaliy Chiley, Jonathan Frankle, Cody Blakeney, and
John Patrick Cunningham. LoRA learns less and forgets less. _Transactions on Machine Learn-_
_ing_ _Research_, 2024. ISSN 2835-8856. URL [https://openreview.net/forum?id=](https://openreview.net/forum?id=aloEru2qCG)
[aloEru2qCG.](https://openreview.net/forum?id=aloEru2qCG) Featured Certification.


Yonatan Bisk, Rowan Zellers, Jianfeng Gao, Yejin Choi, et al. Piqa: Reasoning about physical
commonsense in natural language. In _Proceedings of the AAAI conference on artificial intelligence_,
volume 34, pp. 7432–7439, 2020.


Kerim Büyükakyüz. Olora: Orthonormal low-rank adaptation of large language models. _arXiv_
_preprint arXiv:2406.01775_, 2024.


Eole Cervenka. Naruto blip captions. [https://huggingface.co/datasets/](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions/)
[lambdalabs/naruto-blip-captions/, 2022.](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions/)


Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text
generation. In _International Conference on Machine Learning_, pp. 1931–1942. PMLR, 2021a.


Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal. Unifying vision-and-language tasks via text
generation. In Marina Meila and Tong Zhang (eds.), _Proceedings_ _of_ _the_ _38th_ _International_
_Conference on Machine Learning_, volume 139 of _Proceedings of Machine Learning Research_, pp.
1931–1942. PMLR, 18–24 Jul 2021b. [URL https://proceedings.mlr.press/v139/](https://proceedings.mlr.press/v139/cho21a.html)
[cho21a.html.](https://proceedings.mlr.press/v139/cho21a.html)


Christopher Clark, Kenton Lee, Ming-Wei Chang, Tom Kwiatkowski, Michael Collins, and Kristina
Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In _Proceedings_
_of_ _the_ _2019_ _Conference_ _of_ _the_ _North_ _American_ _Chapter_ _of_ _the_ _Association_ _for_ _Computational_
_Linguistics:_ _Human Language Technologies, Volume 1 (Long and Short Papers)_, pp. 2924–2936,
2019.


Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and
Oyvind Tafjord. Think you have solved question answering? try arc, the ai2 reasoning challenge.
_arXiv preprint arXiv:1803.05457_, 2018.


Rudrajit Das, Naman Agarwal, Sujay Sanghavi, and Inderjit S Dhillon. Towards quantifying the
preconditioning effect of adam. _arXiv preprint arXiv:2402.07114_, 2024.


Simon Du, Jason Lee, Haochuan Li, Liwei Wang, and Xiyu Zhai. Gradient descent finds global
minima of deep neural networks. In _International conference on machine learning_, pp. 1675–1685.
PMLR, 2019.


Hugging Face and Huggan. Wikiart dataset. [https://huggingface.co/datasets/](https://huggingface.co/datasets/huggan/wikiart)
[huggan/wikiart, 2023.](https://huggingface.co/datasets/huggan/wikiart) Accessed: 2024-11-08.


10


Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the v in vqa
matter: Elevating the role of image understanding in visual question answering. In _Proceedings of_
_the IEEE conference on computer vision and pattern recognition_, pp. 6904–6913, 2017.


Yuchao Gu, Xintao Wang, Jay Zhangjie Wu, Yujun Shi, Yunpeng Chen, Zihan Fan, Wuyou Xiao,
Rui Zhao, Shuning Chang, Weijia Wu, et al. Mix-of-show: Decentralized low-rank adaptation for
multi-concept customization of diffusion models. _Advances in Neural Information Processing_
_Systems_, 36:15890–15902, 2023.


David Ha, Andrew M. Dai, and Quoc V. Le. Hypernetworks. In _International Conference on Learning_
_Representations_, 2017. [URL https://openreview.net/forum?id=rkpACe1lx.](https://openreview.net/forum?id=rkpACe1lx)


Soufiane Hayou, Nikhil Ghosh, and Bin Yu. Lora+: Efficient low rank adaptation of large models. In
_Forty-first International Conference on Machine Learning_, 2024.


Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang,
and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In _International_
_Conference on Learning Representations_, 2022. [URL https://openreview.net/forum?](https://openreview.net/forum?id=nZeVKeeFYf9)
[id=nZeVKeeFYf9.](https://openreview.net/forum?id=nZeVKeeFYf9)


W. Ronny Huang, Zeyad Emam, Micah Goldblum, Liam Fowl, Justin K. Terry, Furong Huang, and
Tom Goldstein. Understanding generalization through visualizations. In Jessica Zosa Forde, Francisco Ruiz, Melanie F. Pradier, and Aaron Schein (eds.), _Proceedings on "I Can’t Believe It’s Not_
_Better!" at NeurIPS Workshops_, volume 137 of _Proceedings of Machine Learning Research_, pp. 87–
97. PMLR, 12 Dec 2020. [URL https://proceedings.mlr.press/v137/huang20a.](https://proceedings.mlr.press/v137/huang20a.html)
[html.](https://proceedings.mlr.press/v137/huang20a.html)


Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning
and compositional question answering. In _Proceedings of the IEEE/CVF conference on computer_
_vision and pattern recognition_, pp. 6700–6709, 2019.


Minyoung Huh, Hossein Mobahi, Richard Zhang, Brian Cheung, Pulkit Agrawal, and Phillip Isola.
The low-rank simplicity bias in deep networks. _Transactions on Machine Learning Research_, 2023.
ISSN 2835-8856. [URL https://openreview.net/forum?id=bCiNWDmlY2.](https://openreview.net/forum?id=bCiNWDmlY2)


Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and
Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In
_Proceedings_ _of_ _the_ _IEEE/CVF_ _Conference_ _on_ _Computer_ _Vision_ _and_ _Pattern_ _Recognition_, pp.
9307–9315, 2024.


Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In _International_
_Conference_ _on_ _Learning_ _Representations_, 2015. URL [https://arxiv.org/abs/1412.](https://arxiv.org/abs/1412.6980)
[6980.](https://arxiv.org/abs/1412.6980)


Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M Asano. Vera: Vector-based random matrix
adaptation. In _The Twelfth International Conference on Learning Representations_, 2024.


Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained
categorization. In _Proceedings of the IEEE international conference on computer vision workshops_,
pp. 554–561, 2013.


Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy,
Veselin Stoyanov, and Luke Zettlemoyer. BART: Denoising sequence-to-sequence pre-training for
natural language generation, translation, and comprehension. In Dan Jurafsky, Joyce Chai, Natalie
Schluter, and Joel Tetreault (eds.), _Proceedings of the 58th Annual Meeting of the Association for_
_Computational Linguistics_, pp. 7871–7880, Online, July 2020. Association for Computational
Linguistics. doi: 10.18653/v1/2020.acl-main.703. URL [https://aclanthology.org/](https://aclanthology.org/2020.acl-main.703)
[2020.acl-main.703.](https://aclanthology.org/2020.acl-main.703)


Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image
pre-training for unified vision-language understanding and generation. In _ICML_, 2022.


Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. In _NeurIPS_,
2023.


11


Shih-yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, KwangTing Cheng, and Min-Hung Chen. Dora: Weight-decomposed low-rank adaptation. In _Forty-first_
_International Conference on Machine Learning_, 2024.


Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike
Lewis, Luke Zettlemoyer, and Veselin Stoyanov. Roberta: A robustly optimized bert pretraining
approach. _arXiv preprint arXiv:1907.11692_, 2019.


Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In _International Confer-_
_ence on Learning Representations_, 2019. [URL https://openreview.net/forum?id=](https://openreview.net/forum?id=Bkg6RiCqY7)
[Bkg6RiCqY7.](https://openreview.net/forum?id=Bkg6RiCqY7)


Fanxu Meng, Zhaohui Wang, and Muhan Zhang. Pissa: Principal singular values and singular vectors
adaptation of large language models. In _Proceedings of the 38th Conference on Neural Information_
_Processing Systems (NeurIPS)_, 2024.


Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal. Can a suit of armor conduct
electricity? a new dataset for open book question answering. In _Proceedings of the 2018 Conference_
_on Empirical Methods in Natural Language Processing_, pp. 2381–2391, 2018.


Quynh Nguyen and Matthias Hein. The loss surface of deep and wide neural networks. In _Interna-_
_tional conference on machine learning_, pp. 2603–2612. PMLR, 2017.


Mahdi Nikdan, Soroush Tabesh, Elvir Crnˇcevi´c, and Dan Alistarh. RoSA: Accurate parameterefficient fine-tuning via robust adaptation. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller,
Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the_
_41st_ _International_ _Conference_ _on_ _Machine_ _Learning_, volume 235 of _Proceedings_ _of_ _Machine_
_Learning Research_, pp. 38187–38206. PMLR, 21–27 Jul 2024. [URL https://proceedings.](https://proceedings.mlr.press/v235/nikdan24a.html)
[mlr.press/v235/nikdan24a.html.](https://proceedings.mlr.press/v235/nikdan24a.html)


Jesus-German Ortiz-Barajas, Helena Gomez-Adorno, and Thamar Solorio. Hyperloader: Integrating
hypernetwork-based lora and adapter layers into multi-task transformers for sequence labelling.
_arXiv preprint arXiv:2407.01411_, 2024.


Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe
Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image
synthesis. In _The Twelfth International Conference on_ _Learning Representations_, 2024. URL
[https://openreview.net/forum?id=di52zR8xgf.](https://openreview.net/forum?id=di52zR8xgf)


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In _International conference on machine learning_, pp.
8748–8763. PMLR, 2021.


Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Wei Wei, Tingbo Hou, Yael Pritch, Neal Wadhwa,
Michael Rubinstein, and Kfir Aberman. Hyperdreambooth: Hypernetworks for fast personalization
of text-to-image models. In _Proceedings of the IEEE/CVF Conference on Computer Vision and_
_Pattern Recognition_, pp. 6527–6536, 2024.


Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. Winogrande: An
adversarial winograd schema challenge at scale. _Communications_ _of_ _the_ _ACM_, 64(9):99–106,
2021.


Maarten Sap, Hannah Rashkin, Derek Chen, Ronan Le Bras, and Yejin Choi. Social iqa: Commonsense reasoning about social interactions. In _Proceedings of the 2019 Conference on Empirical_
_Methods in Natural Language Processing and the 9th International Joint Conference on Natural_
_Language Processing (EMNLP-IJCNLP)_, pp. 4463–4473, 2019.


Ying Sheng, Shiyi Cao, Dacheng Li, Coleman Hooper, Nicholas Lee, Shuo Yang, Christopher Chou,
Banghua Zhu, Lianmin Zheng, Kurt Keutzer, et al. S-lora: Serving thousands of concurrent lora
adapters. _arXiv preprint arXiv:2311.03285_, 2023.


12


Alane Suhr, Stephanie Zhou, Ally Zhang, Iris Zhang, Huajun Bai, and Yoav Artzi. A corpus for
reasoning about natural language grounded in photographs. In _Proceedings of the 57th Annual_
_Meeting of the Association for Computational Linguistics_, pp. 6418–6428, 2019.


Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay
Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation
and fine-tuned chat models. _arXiv preprint arXiv:2307.09288_, 2023.


Patrick von Platen, Suraj Patil, Anton Lozhkov, Pedro Cuenca, Nathan Lambert, Kashif Rasul,
Mishig Davaadorj, Dhruv Nair, Sayak Paul, William Berman, Yiyi Xu, Steven Liu, and Thomas
Wolf. Diffusers: State-of-the-art diffusion models. [https://github.com/huggingface/](https://github.com/huggingface/diffusers)
[diffusers, 2022.](https://github.com/huggingface/diffusers)


Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman. Glue:
A multi-task benchmark and analysis platform for natural language understanding. In _Proceedings_
_of the 2018 EMNLP Workshop BlackboxNLP: Analyzing and Interpreting Neural Networks for_
_NLP_, pp. 353–355, 2018.


Shaowen Wang, Linxi Yu, and Jian Li. Lora-ga: Low-rank adaptation with gradient approximation.
In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_, 2024.


Zhengbo Wang, Jian Liang, Ran He, Zilei Wang, and Tieniu Tan. LoRA-pro: Are low-rank adapters
properly optimized? In _The Thirteenth International Conference on Learning Representations_,
2025. [URL https://openreview.net/forum?id=gTwRMU3lJ5.](https://openreview.net/forum?id=gTwRMU3lJ5)


Can Yaras, Peng Wang, Laura Balzano, and Qing Qu. Compressible dynamics in deep overparameterized low-rank learning & adaptation. In _International Conference on Machine Learning_, pp.
56946–56965. PMLR, 2024.


Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. Hellaswag: Can a machine
really finish your sentence? In _Proceedings of the 57th Annual Meeting of the Association for_
_Computational Linguistics_, pp. 4791–4800, 2019.


Fangzhao Zhang and Mert Pilanci. Riemannian preconditioned LoRA for fine-tuning foundation models. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan
Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Ma-_
_chine Learning_, volume 235 of _Proceedings of Machine Learning Research_, pp. 59641–59669.
[PMLR, 21–27 Jul 2024a. URL https://proceedings.mlr.press/v235/zhang24ax.](https://proceedings.mlr.press/v235/zhang24ax.html)
[html.](https://proceedings.mlr.press/v235/zhang24ax.html)


Fangzhao Zhang and Mert Pilanci. Riemannian preconditioned lora for fine-tuning foundation models.
In _International Conference on Machine Learning_, pp. 59641–59669. PMLR, 2024b.


Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo
Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In _The Eleventh International_
_Conference on Learning Representations_, 2023. [URL https://openreview.net/forum?](https://openreview.net/forum?id=lq62uWRJjiY)
[id=lq62uWRJjiY.](https://openreview.net/forum?id=lq62uWRJjiY)


Yuhui Zhang, Alyssa Unell, Xiaohan Wang, Dhruba Ghosh, Yuchang Su, Ludwig Schmidt, and
Serena Yeung-Levy. Why are visually-grounded language models bad at image classification?
In _The Thirty-eighth Annual Conference on Neural Information Processing Systems_, 2024. URL
[https://openreview.net/forum?id=MwmmBg1VYg.](https://openreview.net/forum?id=MwmmBg1VYg)


A THEORETICAL ANALYSIS


A.1 EXTENSION OF OVERPARMETERIZATION ANALYSIS TO MLP


Although our analysis of the optimization benefits of OP-LoRA is carried out in the linear case
(Section 3.2 of the main paper), the same principles extend to deeper ReLU networks. We formalize
this below.


13


**Lemma A.1** (First-order expansion of ReLU activations) **.** _Let h_ = ReLU( _a_ ) _with a_ = _W_ 1 _z._ _For a_
_perturbation_ ∆ _a, define the activation mask_ **M** = **1** _{W_ 1 _z>_ 0 _}._ _Then a first-order Taylor expansion_
_gives_
_hnew_ _≈_ _h_ + **M** _⊙_ ∆ _a._ (11)
**Lemma A.2** (Expansion of pre-activation perturbations) **.** _Let a_ = _W_ 1 _z._ _For small perturbations_
∆ _W_ 1 _,_ ∆ _z,_
_a_ + ∆ _a_ = ( _W_ 1 + ∆ _W_ 1)( _z_ + ∆ _z_ ) _._ (12)
_Expanding and subtracting a yields_


∆ _a_ = ∆ _W_ 1 _z_ + _W_ 1∆ _z_ + ∆ _W_ 1∆ _z._ (13)


**Lemma A.3** (Form of gradient updates) **.** _With gradients ∇h, ∇z, the updates take the form_

∆ _W_ 1 = _−η_ ( _∇h ⊙_ **M** ) _z_ _[⊤]_ _,_ ∆ _z_ = _−η∇z._ (14)


_Substituting into the expansion of_ ∆ _a and dropping second order terms gives_


∆ _a ≈−η_ (( _∇h ⊙_ **M** ) _z_ _[⊤]_ _z_ + _W_ 1 _∇z_ ) _._ (15)


**Theorem A.4** (Approximation of activation perturbations) **.** _Combining the previous results,_


∆ _h_ _≈−η_ **M** _⊙_ �( _∇h ⊙_ **M** ) _z_ _[⊤]_ _z_ + _W_ 1 _∇z_       - _._ (16)


**Theorem A.5** (Update rule in the ReLU case) **.** _Let v be the parameter vector defined as in the main_
_paper._ _Then the update rule is_


                 -                 - ��
_v_ [(] _[t]_ [+1)] = _v_ [(] _[t]_ [)] _−_ _η∥h_ [(] _[t]_ [)] _∥_ [2] _∇v_ ( _t_ ) _−_ _ηW_ 2 [(] _[t]_ [)] **M** _⊙_ ( _W_ 2 _[⊤][∇][v]_ _[⊙]_ **[M]** [)] _[ z][⊤][z]_ [ +] _[ W]_ [1] _[∇][z]_ _._ (17)


**Corollary A.6** (Comparison to the linear case) **.** _In the linear case (Section 3.2 of the main paper),_
_the update reduces to_


_v_ [(] _[t]_ [+1)] = _v_ [(] _[t]_ [)] _−_ _η∥h_ [(] _[t]_ [)] _∥_ [2] _∇v_ ( _t_ ) _−_ _ηW_ 2 [(] _[t]_ [)][(] _[W]_ [ (] 2 _[t]_ [)] _[⊤]_ _∇v_ ) _._ (18)


_Thus both cases share two key terms:_


    - _**Trainable learning rate:**_ _−∥h_ [(] _[t]_ [)] _∥_ [2] _∇v, unchanged from the linear case._


    - _**Adaptive line search:**_ _an update along the subspace spanned by the current columns of W_ 2 _._

_Remark_ A.7 (Geometric interpretation in the ReLU case) _._ The adaptive line search retains the same
geometric role as in the linear case: shifting updates along the span of _W_ 2. However, the ReLU nonlinearity induces a _composite over-parameterization_ : the values _h_ themselves are generated through
an extra layer with nonlinearity, and each column of _W_ 2 only contributes when its corresponding
ReLU unit is active. This leads to more diverse update directions when different units activate across
inputs or training steps.


A.2 CURVATURE ANALYSIS


A.2.1 LORA HESSIAN


We begin with the LoRA reparameterization


_W_ = _W_ 0 + _BA,_ _B_ _∈_ R _[d][×][r]_ _,_ _A ∈_ R _[r][×][d]_ _,_ _W_ 0 _∈_ R _[d][×][d]_ _._ (19)


Let _L_ ( _W_ ) denote the loss, and let _HW_ be the Hessian of _L_ at _W_ 0, viewed as a linear operator
mapping perturbations in _W_ to second–order variations of the loss.

**Lemma A.8** (Quadratic approximation) **.** _For any small perturbation_ ∆ _W_ _, a second–order Taylor_
_expansion gives_

_L_ ( _W_ 0 + ∆ _W_ ) _≈_ _L_ ( _W_ 0) + _⟨∇W L,_ ∆ _W_ _⟩_ + [1] 2 _[⟨]_ [∆] _[W,]_ _[H][W]_ [ (∆] _[W]_ [)] _[⟩][.]_ (20)


**Theorem A.9** (Effective Hessian with respect to _B_ ) **.** _Fix A and consider variations in B._ _For a_
_perturbation_ ∆ _B, we have_ ∆ _W_ = ∆ _BA._ _The corresponding effective curvature operator is_


_HB_ (∆ _B_ ) = _HW_ (∆ _BA_ ) _A_ _[⊤]_ _._ (21)


14


_Proof._ Substitute ∆ _W_ = ∆ _BA_ into the quadratic form:


12 _[⟨]_ [∆] _[W,]_ _[H][W]_ [ ∆] _[W]_ _[⟩]_ [=] [1] 2 _[⟨]_ [∆] _[BA,]_ _[H][W]_ [ (∆] _[BA]_ [)] _[⟩]_

= [1] 2 _[⟨]_ [∆] _[B,]_ _[H][W]_ [ (∆] _[BA]_ [)] _[ A][⊤][⟩][,]_


which establishes the claim.


**Corollary A.10** (Operator and matrix forms of _HB_ ) **.** _In operator notation,_


_HB_ = ( _· A_ _[⊤]_ ) _◦_ _HW_ _◦_ ( _·A_ ) _,_


_where_ ( _·X_ ) _denotes right multiplication by X._ _In matrix form,_


_HB_ = ( _A_ _[⊤]_ _⊗_ _I_ ) _[T]_ _HW_ ( _A_ _[T]_ _⊗_ _I_ ) _,_ _I_ _∈_ R _[d][×][d]_ _._ (22)


**Theorem A.11** (Effective Hessian with respect to _A_ ) **.** _Fix B_ _and consider variations in A._ _For a_
_perturbation_ ∆ _A, we have_ ∆ _W_ = _B_ ∆ _A._ _The corresponding effective curvature operator is_


_HA_ (∆ _A_ ) = _B_ _[⊤]_ _HW_ ( _B_ ∆ _A_ ) _._ (23)


_Proof._ Substitute ∆ _W_ = _B_ ∆ _A_ into the quadratic form:


21 _[⟨]_ [∆] _[W,]_ _[H][W]_ [ ∆] _[W]_ _[⟩]_ [=] [1] 2 _[⟨][B]_ [∆] _[A,]_ _[H][W]_ [ (] _[B]_ [∆] _[A]_ [)] _[⟩]_

= [1] 2 _[⟨]_ [∆] _[A,]_ _[B][⊤][H][W]_ [ (] _[B]_ [∆] _[A]_ [)] _[⟩][,]_


which establishes the claim.


**Corollary A.12** (Operator and matrix forms of _HA_ ) **.** _In operator notation,_


_HA_ = _B_ _[⊤]_ _◦_ _HW_ _◦_ _B._


_In matrix form,_
_HA_ = ( _I_ _⊗_ _B_ _[⊤]_ ) _HW_ ( _I_ _⊗_ _B_ ) _,_ _I_ _∈_ R _[d][×][d]_ _._ (24)


A.2.2 CONDITION NUMBER BOUNDS


In the main paper we have the following bound of the condition number of _HA_ in Equation (3):


_κ_ ( _XY_ ) _≥_ max


       - _κ_ ( _HW_ ) _κ_ ( _B_ ) [2]
max _κ_ ( _B_ ) [2] _[,]_ _κ_ ( _HW_ )


We show this to be true below.


_≤_ _κ_ ( _HA_ ) _≤_ _κ_ ( _B_ ) [2] _· κ_ ( _HW_ ) _._ (25)


**Lemma A.13** (Upper bound) **.** _For any full-rank X, Y,_


_κ_ ( _XY_ ) _≤_ _κ_ ( _X_ ) _κ_ ( _Y_ ) _._


_Proof._ By definition, _κ_ ( _XY_ ) = _σ_ 1( _XY_ ) _/σ_ min( _XY_ ). Submultiplicativity of the spectral norm gives
_σ_ 1( _XY_ ) _≤_ _σ_ 1( _X_ ) _σ_ 1( _Y_ ), and the inequality _σ_ min( _XY_ ) _≥_ _σ_ min( _X_ ) _σ_ min( _Y_ ) yields the result:


_σ_ 1( _X_ ) _σ_ 1( _Y_ )
_κ_ ( _XY_ ) _≤_
_σ_ min( _X_ ) _σ_ min( _Y_ ) [=] _[ κ]_ [(] _[X]_ [)] _[κ]_ [(] _[Y]_ [ )] _[.]_


**Lemma A.14** (Lower bound) **.** _For any full-rank X, Y,_


_κ_ ( _X_ )


_κ_ ( _X_ ) _[κ]_ [(] _[Y]_ [ )]

_κ_ ( _Y_ ) _[,]_ _κ_ ( _X_ )


_κ_ ( _X_ )


_._


15


_Proof._ Using the min–max characterization of singular values:


_σ_ 1( _XY_ ) = max min [max]
_∥u∥_ =1 _[∥][XY u][∥≥]_ _∥v∥_ =1 _[∥][Xv][∥·]_ _∥u∥_ =1 _[∥][Y u][∥]_ [=] _[ σ]_ [min][(] _[X]_ [)] _[ σ]_ [1][(] _[Y]_ [ )] _[.]_


Similarly,


_σ_ min( _XY_ ) = min [min]
_∥u∥_ =1 _[∥][XY u][∥≤∥][X][∥]_ [2] _[ ·]_ _∥u∥_ =1 _[∥][Y u][∥]_ [=] _[ σ]_ [1][(] _[X]_ [)] _[ σ]_ [min][(] _[Y]_ [ )] _[.]_


Taking the ratio gives


where _A_ and _B_ are LoRA matrices, _W_ are the base weights, _κ_ ( _·_ ) denotes the condition number,
and _H_ are the corresponding Hessians. Equation (26) implies that LoRA can exhibit worse Hessian
conditioning than full finetuning.


A high Hessian condition number indicates very high curvature in some directions relative to others.
This matters because the step size must be small enough to avoid instabilities along the highestcurvature direction; the maximal stable learning rate is then limited by that direction and may be too


16


_σ_ 1( _XY_ )
_κ_ ( _XY_ ) =
_σ_ min( _XY_ ) _[≥]_ _[σ]_ _σ_ [min] 1( _X_ [(] ) _[X]_ _σ_ [)] min _[ σ]_ [1][(] ( _[Y]_ _Y_ [ )] )


_[σ]_ [min][(] _[X]_ [)] _[ σ]_ [1][(] _[Y]_ [ )] _[κ]_ [(] _[Y]_ [ )]

_σ_ 1( _X_ ) _σ_ min( _Y_ ) [=] _κ_ ( _X_ )


_κ_ ( _X_ ) _[.]_


Swapping _X_ and _Y_ yields the other inequality.


**Theorem A.15** (Bounds for products) **.** _For any full-rank X, Y,_


_κ_ ( _X_ )


max


_κ_ ( _X_ ) _[κ]_ [(] _[Y]_ [ )]

_κ_ ( _Y_ ) _[,]_ _κ_ ( _X_ )


_κ_ ( _X_ )


_≤_ _κ_ ( _XY_ ) _≤_ _κ_ ( _X_ ) _κ_ ( _Y_ ) _._


Now let _HW_ _∈_ R _[d][×][d]_ be symmetric positive definite (SPD), associated with a weight matrix _W_ =
_W_ 0 + _AB_ where _B_ _∈_ R _[d][×][r]_ and _A ∈_ R _[r][×][d]_ . Consider the curvature matrices

_HA_ = ( _I_ _⊗_ _B_ ) _[⊤]_ _HW_ ( _I_ _⊗_ _B_ ) _,_ _HB_ = ( _A_ _[⊤]_ _⊗_ _I_ ) _HW_ ( _A ⊗_ _I_ ) _,_


where _⊗_ denotes the Kronecker product.

**Lemma A.16** (Condition number of quadratic form) **.** _If HW_ _is SPD, then_

_HA_ = _Z_ _[⊤]_ _Z,_ _Z_ = _HW_ [1] _[/]_ [2][(] _[I]_ _[⊗]_ _[B]_ [)] _[.]_


_Hence,_
_κ_ ( _HA_ ) = _κ_ ( _Z_ ) [2] _._
**Theorem A.17** (Condition number bounds for _HA_ ) **.** _Assume HW_ _is SPD and B is full-rank._ _Then_


_κ_ ( _HW_ )


max


( _HW_ ) _κ_ ( _B_ ) [2]

_κ_ ( _B_ ) [2] _[,]_ _κ_ ( _HW_ )


_κ_ ( _HW_ )


_≤_ _κ_ ( _HA_ ) _≤_ _κ_ ( _HW_ ) _κ_ ( _B_ ) [2] _._


_Proof._ Apply Theorem A.15 with _X_ = _HW_ [1] _[/]_ [2][,] _[ Y]_ [=] _[ I][ ⊗]_ _[B]_ [.] [Since] _[ κ]_ [(] _[I][ ⊗]_ _[B]_ [) =] _[ κ]_ [(] _[B]_ [)][, the inequali][ty]
follows. Squaring both sides yields the result.


**Corollary A.18** (Condition number bounds for _HB_ ) **.** _Assume HW_ _is SPD and A is full-rank._ _Then_


_κ_ ( _HW_ )


max


( _HW_ ) _κ_ ( _A_ ) [2]

_κ_ ( _A_ ) [2] _[,]_ _κ_ ( _HW_


_κ_ ( _HW_ )


_≤_ _κ_ ( _HB_ ) _≤_ _κ_ ( _HW_ ) _κ_ ( _A_ ) [2] _._


B ADDITIONAL RESULTS


B.1 GRADIENT ANALYSIS LORA AND OP-LORA


Recall that in Eq. (3) of the main paper we show


  - _κ_ ( _HW_ ) _κ_ ( _A_ ) [2]
max _κ_ ( _A_ ) [2] _[,]_ _κ_ ( _HW_ )


_≤_ _κ_ ( _HB_ ) _≤_ _κ_ ( _A_ ) [2] _κ_ ( _HW_ ) _,_ (26)


(a) **Loss Curve** (b) **Effective Rank** (c) **Gradient Consistency** (d) **Gradient Norm**


Figure 5: **Matrix Factorization (MF) Gradient Analysis:** **(a)** Loss Curve shows the reconstruction
error for matrix factorization. OP-MF converges better and faster than standard MF. **(b)** Effective
Rank of BA reveals changes in the rank of the learned solution. OP-MF learns an effective rank
closer to that of the ground-truth SVD solution. **(c)** Gradient Consistency measures the similarity
of gradients across iterations. OP-MF is able to make a sudden change in optimization direction,
while standard MF cannot. **(d)** Gradient Norm illustrates the scale of gradients. OP-MF is able more
quickly adjust optimization step size.


small to make progress in low-curvature directions. This becomes problematic when the highestcurvature direction has already been minimized, but the low-curvature directions still require larger
learning rates.


A useful diagnostic is the magnitude of the gradient in the direction of the principal singular vector of
the Hessian of the trainable parameters. Let _v_ be that direction (estimated by power iteration) and _g_
the gradient. If _|v_ _[⊤]_ _g|_ is relatively large, the loss can still be reduced even with a learning rate small
enough to remain stable in the largest-curvature direction. Conversely, if _|v_ _[⊤]_ _g| ≈_ 0, a large condition
number will prevent further decrease of the loss.


**Setup.** We use the power-iteration method to estimate _v_ and then measure _|v_ _[⊤]_ _g|_ for a small-scale
Rotated-MNIST problem. Pre-training is performed on MNIST and continued training is on Rotated
MNIST (as in Fig. 2 of the main paper). We report the terminal values for LoRA and OP-LoRA.


Method _|v_ _[⊤]_ _g|_


OP-LoRA 0.42
LoRA 0.06


Table 5: _|v_ _[⊤]_ _g|_ at the end of training on Rotated MNIST. Higher is better (indicates remaining descent
along the highest-curvature direction).


**Findings.** OP-LoRA exhibits a much larger _|v_ _[⊤]_ _g|_ in the direction of largest curvature than LoRA
(Table 5). Empirically, this suggests OP-LoRA may be _less_ _sensitive_ to poor conditioning than
LoRA, because it can continue to reduce the loss even when the step size is constrained by the
highest-curvature direction. Thus, even if the OP-LoRA MLP were itself poorly conditioned, this
sensitivity matters less than for LoRA.


These observations are consistent with the view that OP-LoRA adaptively reshapes the LoRA loss
landscape via reparameterization, leading to better and faster optimization.


B.2 A MATRIX FACTORIZATION CASE STUDY


To verify that the gradient properties of MLP over-parameterization results in observable changes, we
design a controlled matrix factorization experiment comparing MLP-generated low-rank matrices _A_
and _B_ with freely learned parameter matrices and measure convergence and gradients.


Matrix factorization decomposes a target matrix _M_ _∈_ R _[m][×][n]_ into two lower-dimensional matrices,
_A ∈_ R _[r][×][n]_ and _B_ _∈_ R _[m][×][r]_, where _r_ is the latent dimension or rank. This decomposition allows us to


17


approximate _M_ by _BA_ . It can be solved exactly with SVD, or as in our study, one can use gradient
descent to minimize the reconstruction error:


_∥M_ _−_ _BA∥_ [2] _F_ _[,]_


where _∥· ∥F_ denotes the Frobenius norm. This resembles LoRA-tuning, where the pre-trained base
weights are set to all zeros and the target matrix _M_ is the full finetuning gradient matrix, making it an
interesting proxy problem to study.


**Experimental setup and training protocol:** We construct a synthetic target matrix _M_ _∈_ R [100] _[×]_ [100]
with entries initialized uniformly at random from 0 to 1. The resulting matrix has a poor condition
number, defined as the ratio between the largest singular value and lowest, making the optimization
difficult and therefore a good test for MLP reparameterization. We train for 1000 steps with SGD,
with linear warmup for 50 steps and linear learning rate decay.


**OP-MF Model:** The OP-MF model generates matrices _A_ and _B_ through two separate MLPs. We
enforce both matrices to be of rank 8. Each MLP receives a learned input vector _z_ _∈_ R [128] and
processes it through two fully connected layers with 32 hidden units and ReLU activations, outputting
the entries for either _A_ or _B_ . The second layer is heavily overparameterized; the parameter count is
number of hidden units in the MLP by the size of the parameter matrix _A_ or _B_ . To align with LoRA’s
initialization strategy, the MLP for matrix _B_ is initialized to output zeros, setting the model close to a
pre-trained state.
**Matrix Factorization(MF) Model:** We train a MF model with freely learnable matrices _A_ and _B_,
initialized with random values for _A_ and zeros for _B_ . Again, both matrices have rank 8.


**Finding 1:** **OP-MF rapidly adapts step size and direction.** We examine the gradients, looking
for evidence that OP-LoRA adaptively changes step size and direction. Our results reveal that as
predicted, OP-MF shows an ability to rapidly adapt step sizes. This can be seen in Figure 5 (d), where
the gradient norm experiences a sharp spike, followed by a collapse, corresponding to acceleration
and slow down in the loss curve in Figure 5 (a). The sudden phase change in gradient norm also
corresponds to a direction change in trajectory, measured by the cosine similarity between gradients
at 10-step intervals in in Figure 5 (c). Therefore, MLP reparameterization rapidly changes step size
and direction, as suggested by the mathematical analysis in Section 3 of the main paper.


**Finding 2:** **OP-MF is more effective at reaching the SVD solution than standard MF trained**
**with SGD.** In Figure 5(a), we study the loss curves for both the MF model and the OP-MF model.
The plot tracks MSE loss over 1000 iterations. The red line represents the SVD solution as a baseline.
Interestingly, OP-MF solutions reach the best-case reconstruction error achieved with SVD, while
Standard-MF cannot.


In addition to reconstruction error, another way to track progress towards a solution in matrix
factorization is plotting the effective rank of the predicted matrix _BA_ over the course of training.
Effective rank _ρ_ is defined as


where _σi_ represents the normalized singular values of _BA_, and _r_ is the rank of the matrix. One would
expect the effective rank to converge towards the effective rank of the ground-truth SVD solution for
successful optimization runs.


In Figure 5 (b), we observe the behavior of effective rank across iterations for both MF and OP-MF.
We find that OP-MF can approximate the effective rank of the best-case SVD solution much more
closely than standard MF.


**Finding** **3:** **OP-MF** **composes** **well** **with** **both** **SGD** **with** **Momentum** **and** **Adam.** A natural
question is if using standard acceleration methods like SGD with Momentum or Adam is enough.
In Figure 6 we present experiments by adding Momentum to SGD and replacing it entirely with
Adam, an optimizer that combines adaptive learning rates with momentum . Both Adam and SGD
with Momentum improve reconstruction error for MF, but neither reach the SVD solution. Moreover,
OP-MF composes with even best-case and advanced optimizers to find the SVD solution even faster,
indicating their complimentary nature.


18


_r_

- _σi_ log _σi_


_i_ =1


_ρ_ = exp


_−_


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


(a) SGD (b) Momentum (c) Adam


Figure 6: **Comparing Optimizers.** MLP reparameterization accelerates matrix factorization, even
with advanced optimizers.


B.3 LLAVA IMAGE CLASSIFICATION


Zhang et al. (2024) recently demonstrated that by applying simple finetuning to adapters, large multimodal models like LLaVA can achieve surprisingly high performance on a variety of classification
tasks. We leverage this finding, replacing full finetuning with LoRA and OP-LoRA.
**Datasets:** We use the Stanford Cars (Krause et al., 2013) dataset, which is a fine-grained dataset of
about 8000 training examples consisting of 196 classes of cars.


**finetuning and evaluation protocol:** We follow (Zhang et al., 2024), and convert image classification
into a captioning task by using the format “ _⟨_ image _⟩_ What type of object is in this photo? _⟨_ class
name _⟩_ ” and training with a language modeling objective. We finetune the visual projector layers of
LLaVA1.5-7B for 50 epochs. We set MLP width to 768, since memory resources are not an issue for
training only the adapter. At evaluation, we parse the generations and search for the correct class
label.


**Results:** We present the results in Table 6. OP-LoRA consistently outperforms LoRA at both rank
levels, achieving 83.6% at rank 16 and 87.1% at rank 64, compared to 82.8% and 86.3% with LoRA.
This result further reinforces the efficacy of MLP re-parameterization.


**Rank** **LoRA** **OP-LoRA**


16 82.8 **83.6**
64 86.3 **87.1**


Table 6: LLaVA1.5-7B Image Classification, Top-1 Accuracy on Stanford Cars (Krause et al., 2013).


B.4 STABILITY OF OP-DORA


We found that OP-DoRA is more stable than DoRA, as shown with standard deviations across 3 runs.
We hypothesize that this is a reflection of decreased learning rate sensitivity.


Method Commonsense


DoRA ( _r_ = 32) 73 _._ 7 _±_ 6 _._ 7
OP-DoRA ( _r_ = 32) 77 _._ 5 _±_ 1 _._ 6


B.5 VERA AND OP-VERA


We extend our method to VERA (Kopiczko et al., 2024), an ultra–low-parameter variant of LoRA, and
evaluate on the GLUE(Wang et al., 2018) benchmark using a RoBERTa-base(Liu et al., 2019) backbone following the setup in (Kopiczko et al., 2024). We report results on SST-2, COLA, and QNLI.


19


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


We exclude MNLI due to computational constraints and therefore also excludeMRPC/RTE/STS-B,
which are commonly initialized from MNLI to mitigate overfitting (Hu et al., 2022).


SST-2 COLA QNLI **Avg.**


VeRA 94.0 59.8 **91.7** 81.8
OP-VeRA **94.3** **62.1** 91.5 **82.6**


Table 7: GLUE dev results with RoBERTa-base.


Averaged over the three tasks, OP-VeRA improves upon VeRA by **1.2** points, indicating the generality
of the proposed optimization.


B.6 IMPROVING MIX-OF-SHOW WITH OP-LORA


Following Zhang & Pilanci (2024b) directly, we apply Mix-of-Show to a small set of 14 training
images of Harry Potter. We compare standard ScaledAdamW and OP-LoRA, keeping all training
settings from Zhang & Pilanci (2024b). In Figure 7, we generate images from the learned <potter>
tokens as a prompt. We see that OP-LoRA better captures the subject of Harry Potter; there are
fewer image of two people (not present in the training set) and the shape of the face is more accurate.
Moreover, OP-LoRA generates Harry Potter in causal clothing less frequently.


ScaledAdamW OP-LoRA Training Data


B.7 QUALITATIVE STABLE DIFFUSION RESULTS


In the main paper, we show the large quantitative gains OP-LoRA/OP-DoRA give in the image
generation task (Table 1). We now present extensive random generations in Figures 8 though 25,
with captions from the dataset as input. Several general trends emerge. First, there is a strong
color bias towards red, however OP-LoRA and OP-DoRA reduce this dramatically. We attibute this
improvement to the over-parameterization easing optimization. Second, over-parameterized LoRA
is generates much more diverse and more complex scenes. Overall, qualitative results match the
quantitative metrics.


C TRAINING DETAILS


In this section, we summarize the training settings of our main experiments.


**Code:** We provide an implementation of OP-LoRA and OP-DoRA in the supplementary zip, packaged
as a drag-and-drop replacement for existing PEFT libraries. We use this implentation in combination
with code from Liu et al. (2024) for CommonSense and VQA benchmarks, von Platen et al. (2022)
for Stable Diffusion finetuning, and Zhang et al. (2024) for LLaVA classification.


**Hardware:** Most experiments were done with a single H100 80GB HB3 GPU.


20


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


Figure 8: GT Naruto(Cervenka, 2022) images


Figure 9: OP-LoRA Naruto(Cervenka, 2022) generated images.


C.1 INITIALIZING THE OP-LORA MLP


In Section 3, we introduce the MLP used to predict low rank parameters as


We initialize _W_ 1 as Kaiming uniform. We also initalize _W_ 2 as Kaiming uniform, except for parameters
predicting the upsampling matrix B, which are initialized to zero to make the initialization not change
the pre-trained model behavior. All biases are initialized to 0.


21


- _A_
_B_


= _W_ 2(ReLU( _W_ 1 _z_ + _c_ 1)) + _c_ 2


**1134**

**1135**


**1136**

**1137**

**1138**

**1139**

**1140**

**1141**


**1142**

**1143**

**1144**

**1145**

**1146**

**1147**


**1148**

**1149**

**1150**

**1151**

**1152**


**1153**

**1154**

**1155**

**1156**

**1157**

**1158**


**1159**

**1160**

**1161**

**1162**

**1163**

**1164**


**1165**

**1166**

**1167**

**1168**

**1169**

**1170**


**1171**

**1172**

**1173**

**1174**

**1175**


**1176**

**1177**

**1178**

**1179**

**1180**

**1181**


**1182**

**1183**

**1184**

**1185**

**1186**

**1187**


Figure 10: OP-DoRA Naruto(Cervenka, 2022) results


Figure 11: DoRA Naruto(Cervenka, 2022) generated images


C.2 TRAINING HYPERPARAMETERS


In Tables 8 through 11 we show training hyperparameters for experiments. These include batch sizes,
learning rates, optimizer settings, and other configurations for each task.


C.3 LLM USAGE


LLM (ChatGPT) was used for polishing during writing, with direct rewrites at the sentence
level/equation and suggestions at the paragraph/theorem level. It was also used for verifying notational
consistency over the whole paper. All LLM output was carefully reviewed by the authors.


22


**1188**

**1189**


**1190**

**1191**

**1192**

**1193**

**1194**

**1195**


**1196**

**1197**

**1198**

**1199**

**1200**

**1201**


**1202**

**1203**

**1204**

**1205**

**1206**


**1207**

**1208**

**1209**

**1210**

**1211**

**1212**


**1213**

**1214**

**1215**

**1216**

**1217**

**1218**


**1219**

**1220**

**1221**

**1222**

**1223**

**1224**


**1225**

**1226**

**1227**

**1228**

**1229**


**1230**

**1231**

**1232**

**1233**

**1234**

**1235**


**1236**

**1237**

**1238**

**1239**

**1240**

**1241**


Figure 12: LoRA Naruto(Cervenka, 2022) generated images


23


**1242**

**1243**


**1244**

**1245**

**1246**

**1247**

**1248**

**1249**


**1250**

**1251**

**1252**

**1253**

**1254**

**1255**


**1256**

**1257**

**1258**

**1259**

**1260**


**1261**

**1262**

**1263**

**1264**

**1265**

**1266**


**1267**

**1268**

**1269**

**1270**

**1271**

**1272**


**1273**

**1274**

**1275**

**1276**

**1277**

**1278**


**1279**

**1280**

**1281**

**1282**

**1283**


**1284**

**1285**

**1286**

**1287**

**1288**

**1289**


**1290**

**1291**

**1292**

**1293**

**1294**

**1295**


Figure 14: LoRA-GA Naruto(Cervenka, 2022) generated images


Figure 15: LoRA-PRO Naruto(Cervenka, 2022) generated images


24


**1296**

**1297**


**1298**

**1299**

**1300**

**1301**

**1302**

**1303**


**1304**

**1305**

**1306**

**1307**

**1308**

**1309**


**1310**

**1311**

**1312**

**1313**

**1314**


**1315**

**1316**

**1317**

**1318**

**1319**

**1320**


**1321**

**1322**

**1323**

**1324**

**1325**

**1326**


**1327**

**1328**

**1329**

**1330**

**1331**

**1332**


**1333**

**1334**

**1335**

**1336**

**1337**


**1338**

**1339**

**1340**

**1341**

**1342**

**1343**


**1344**

**1345**

**1346**

**1347**

**1348**

**1349**


Figure 16: ScaledAdamW Naruto(Cervenka, 2022) generated images


Figure 17: GT Monet WikiArt(Face & Huggan, 2023) Images


25


**1350**

**1351**


**1352**

**1353**

**1354**

**1355**

**1356**

**1357**


**1358**

**1359**

**1360**

**1361**

**1362**

**1363**


**1364**

**1365**

**1366**

**1367**

**1368**


**1369**

**1370**

**1371**

**1372**

**1373**

**1374**


**1375**

**1376**

**1377**

**1378**

**1379**

**1380**


**1381**

**1382**

**1383**

**1384**

**1385**

**1386**


**1387**

**1388**

**1389**

**1390**

**1391**


**1392**

**1393**

**1394**

**1395**

**1396**

**1397**


**1398**

**1399**

**1400**

**1401**

**1402**

**1403**


Figure 18: OP-LoRA Monet WikiArt(Face & Huggan, 2023) Generated Images


Figure 19: OP-DoRA Monet WikiArt(Face & Huggan, 2023) Generated Images


26


**1404**

**1405**


**1406**

**1407**

**1408**

**1409**

**1410**

**1411**


**1412**

**1413**

**1414**

**1415**

**1416**

**1417**


**1418**

**1419**

**1420**

**1421**

**1422**


**1423**

**1424**

**1425**

**1426**

**1427**

**1428**


**1429**

**1430**

**1431**

**1432**

**1433**

**1434**


**1435**

**1436**

**1437**

**1438**

**1439**

**1440**


**1441**

**1442**

**1443**

**1444**

**1445**


**1446**

**1447**

**1448**

**1449**

**1450**

**1451**


**1452**

**1453**

**1454**

**1455**

**1456**

**1457**


Figure 20: DoRA Monet WikiArt(Face & Huggan, 2023) Generated Images


Figure 21: LoRA Monet WikiArt(Face & Huggan, 2023) Generated Images


27


**1458**

**1459**


**1460**

**1461**

**1462**

**1463**

**1464**

**1465**


**1466**

**1467**

**1468**

**1469**

**1470**

**1471**


**1472**

**1473**

**1474**

**1475**

**1476**


**1477**

**1478**

**1479**

**1480**

**1481**

**1482**


**1483**

**1484**

**1485**

**1486**

**1487**

**1488**


**1489**

**1490**

**1491**

**1492**

**1493**

**1494**


**1495**

**1496**

**1497**

**1498**

**1499**


**1500**

**1501**

**1502**

**1503**

**1504**

**1505**


**1506**

**1507**

**1508**

**1509**

**1510**

**1511**


Figure 22: PiSSA Monet WikiArt(Face & Huggan, 2023) Generated Images


Figure 23: LoRA-GA Monet WikiArt(Face & Huggan, 2023) Generated Images


28


**1512**

**1513**


**1514**

**1515**

**1516**

**1517**

**1518**

**1519**


**1520**

**1521**

**1522**

**1523**

**1524**

**1525**


**1526**

**1527**

**1528**

**1529**

**1530**


**1531**

**1532**

**1533**

**1534**

**1535**

**1536**


**1537**

**1538**

**1539**

**1540**

**1541**

**1542**


**1543**

**1544**

**1545**

**1546**

**1547**

**1548**


**1549**

**1550**

**1551**

**1552**

**1553**


**1554**

**1555**

**1556**

**1557**

**1558**

**1559**


**1560**

**1561**

**1562**

**1563**

**1564**

**1565**


Figure 24: LoRA-Pro Monet WikiArt(Face & Huggan, 2023) Generated Images


**1566**

**1567**


**1568**

**1569**

**1570**

**1571**

**1572**

**1573**


**1574**

**1575**

**1576**

**1577**

**1578**

**1579**


**1580**

**1581**

**1582**

**1583**

**1584**


**1585**

**1586**

**1587**

**1588**

**1589**

**1590**


**1591**

**1592**

**1593**

**1594**

**1595**

**1596**


**1597**

**1598**

**1599**

**1600**

**1601**

**1602**


**1603**

**1604**

**1605**

**1606**

**1607**


**1608**

**1609**

**1610**

**1611**

**1612**

**1613**


**1614**

**1615**

**1616**

**1617**

**1618**

**1619**


Hyperparameters All


Base Model Stable Diffusion XL 1.0(Podell et al., 2024)
Rank _r_ 4
_α_ 4
Dropout 0.0
Optimizer AdamW
LR 1e-4
LR Scheduler Constant
Batch size 1
Warmup Steps 0
Epochs 2
Where U-Net Q, K, V, Out
MLP-Width(OP-LoRA/OP-DoRA) 32


Table 8: Training Details for Stable Diffusion Finetuning Experiments.


Hyperparameters All


Base Model VL-Bart(Cho et al., 2021a)
Rank _r_ 128
_α_ 128
Dropout 0.0
Optimizer AdamW
LR 1e-3
LR Scheduler Linear
Batch size 300
Warmup ratio 0.1
Epochs 20
Where Q,K (Bias Also Trained)
MLP-Width(OP-LoRA/OP-DoRA) 32(OP-LoRA)/4(OP-DoRA)


Table 9: Training Details for VL-Bart Finetuning Experiments.


Hyperparameters All


Base Model LLaVA1.5-7B(Liu et al., 2023)
Rank _r_ 64/16
_α_ 128/32
Dropout 0.0
Optimizer AdamW
LR 5e-5
LR Scheduler Cosine
Batch size 64
Warmup Ratio 0 .03
Epochs 50
Where Multimodal Projector
MLP-Width(OP-LoRA/OP-DoRA) 768


Table 10: Training Details for LLaVA Classification Finetuning Experiments.


30


**1620**

**1621**


**1622**

**1623**

**1624**

**1625**

**1626**

**1627**


**1628**

**1629**

**1630**

**1631**

**1632**

**1633**


**1634**

**1635**

**1636**

**1637**

**1638**


**1639**

**1640**

**1641**

**1642**

**1643**

**1644**


**1645**

**1646**

**1647**

**1648**

**1649**

**1650**


**1651**

**1652**

**1653**

**1654**

**1655**

**1656**


**1657**

**1658**

**1659**

**1660**

**1661**


**1662**

**1663**

**1664**

**1665**

**1666**

**1667**


**1668**

**1669**

**1670**

**1671**

**1672**

**1673**


Hyperparameters All


Base Model LLaMA-7B(Touvron et al., 2023)
Rank _r_ 32
_α_ 64
Dropout 0.05
Optimizer AdamW
LR 1e-4
LR Scheduler Linear
Batch size 16
Warmup ratio 0.03
Epochs 3
Where Q,K, V, Up, Down
MLP-Width(OP-LoRA/OP-DoRA) 32


Table 11: Training Details for CommonSense Finetuning Experiments.


31