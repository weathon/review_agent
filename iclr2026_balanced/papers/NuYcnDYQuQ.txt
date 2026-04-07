# Evil in the Pairing Assumption : MULTIMODAL ATTRIBU- TION VIA ADAPTIVE INFORMATION BOTTLENECK


**Anonymous authors**
Paper under double-blind review


ABSTRACT


Multimodal attribution methods such as M2IB aim to interpret vision-language
models without requiring task-specific labels, but they often rely on the assumption
of accurate semantic alignment between image-text pairs. This assumption does not
hold in open-world settings, where noisy or mismatched inputs are common. Under
such conditions, existing attribution methods tend to overfit and generate forced
explanations, compromising the reliability and trustworthiness of interpretability.
To address this, we observe that a well-balanced trade-off between the compression
and prediction terms in the information bottleneck objective can mitigate overfitting.
Based on this insight, we introduce an attribution framework that leverages an
adaptive information bottleneck optimisation objective. Our method dynamically
adjusts the bottleneck constraints without assuming reliable cross-modal alignment.
Extensive experiments on large-scale image-text datasets show that our approach
consistently outperforms existing attribution methods in both quantitative metrics
and qualitative interpretability, providing more robust and trustworthy explanations
while relaxing the requirement for aligned image-text pairs.


1 INTRODUCTION


Multimodal learning, particularly in vision-language models (VLMs), has made remarkable progress
in recent years, enabling powerful capabilities across tasks such as image captioning, visual question
answering, and retrieval. However, as these models become increasingly complex and ubiquitous, the
demand for interpretable and trustworthy explanations has grown accordingly. Multimodal attribution
methods aim to fill this gap by identifying which parts of the input (e.g. image regions or text
tokens) contribute most to a model’s decision. Among these, recent works such as the Multi-Modal
Information Bottleneck (M2IB) (Wang et al., 2023) and the Narrowing Information Bottleneck
(NIB) (Zhu et al., 2025) propose to utilise information bottleneck to generate faithful and compact
explanations without task-specific supervision.


However, a critical assumption underlying these methods is the semantic alignment between paired
modalities—that is, the image and text pairs are assumed to share the same concept or scene. While
this assumption holds in well-curated datasets, it often breaks down in open-world scenarios, where
noisy, mismatched, or loosely related pairs are common. Under such conditions, existing attribution
methods tend to overfit to spurious correlations, resulting in misleading or uninformative explanations,
as illustrated in Fig. 1. This undermines the interpretability and usability in real-world applications.
These challenges highlight the need for attribution methods that can adapt to varying degrees of
cross-modal alignment and remain robust in the presence of noise and semantic mismatch.


In this work, we challenge the pairing assumption and ask: _Can we build robust attribution methods_
_that remain effective even when input modalities are weakly aligned or even completely mismatched?_
To this end, we introduce a new framework that adapts the standard modality-matching objective in
information bottleneck attribution. We do not assume clear alignment between modalities and propose
Adaptive Multimodal Information Bottleneck (AdaIB), which extends the traditional Information
Bottleneck (IB) objective by introducing an adaptive weighting mechanism between the compression
term and the fitting term on the objective without requiring strict one-to-one alignment. Our key
contributions are summarised as follows:


1


**NIB**


**M2IB**


**FastIG** **Chefer et al.**


**GradCAM** **MFABA**


Figure 1: Visualisation results of various multimodal explanation methods on _completely_ _noisy_
_image-text pairs_ . The example caption is “monsters will absorb your wisdom, be careful to protect
your poster and your brain!”. Existing approaches (e.g. M2IB, GradCAM, NIB, etc.) tend to produce
forced or misleading alignments even when the modalities do not semantically match (First three
columns). In contrast, our proposed AdaIB method is able to suppress responses under a completely
mismatched pair (Top of the last column).


    - We propose an adaptive information bottleneck optimisation framework (AdaIB) that dynamically adjusts the objective based on the relationship between paired image-text data,
improving the attribution method for interpretability analysis.


    - We formally analyse the functional properties of the proposed AdaIB, including its gradient
behaviour and information ratio structure. We show that AdaIB enables sample-aware
control overfitting and compression terms and recovers the standard IB as a special case.


    - We empirically validate our approach across diverse metric conditions, showing much better
robustness and interpretability compared to state-of-the-art methods.


2 RELATED WORK


2.1 UNIMODAL INTERPRETABILITY METHODS


Traditional interpretability methods were primarily developed for unimodal deep learning models
and face significant limitations when applied to multimodal architectures. Early gradient-based
approaches, such as Saliency Maps, compute input-output gradients to highlight important regions in
the input. However, they are highly sensitive to noise and often produce low-resolution or unstable
explanations. Grad-CAM Selvaraju et al. (2017) improves upon this by using gradients of classspecific activation maps in convolutional layers, resulting in more focused and human-interpretable
heatmaps. LIME (Ribeiro et al., 2016) adopts a black-box approach by perturbing the input and
fitting a local surrogate model, offering model-agnostic explanations. Despite its generality, LIME
may fail on complex models due to its reliance on local linearity assumptions. RISE (Petsiuk et al.,
2018) also introduces a model-agnostic strategy by applying random masks and measuring output
changes to build relevance maps. While effective globally, its sampling-based nature leads to high
computational cost and potential noise. More recent attribution methods, such as AGI (Pan et al.,
2021) and MFABA (Zhu et al., 2024), utilise adversarial perturbations to generate more robust saliency
maps. These approaches satisfy formal interpretability criteria like Sensitivity and Implementation
Invariance (Sundararajan et al., 2017), offering theoretical guarantees.


Nevertheless, most of these methods are designed for unimodal tasks and require access to downstream
task labels or internal gradients, making them less suitable for large-scale vision-language models.
Attempts to extend such methods to the multimodal setting—such as CLIP—face challenges due to
architectural differences and lack of task-specific supervision. As demonstrated in our experiments,
these methods often fail to provide meaningful explanations when applied directly to multimodal
models.


2


**AdaIB (Ours)**


**Image**


2.2 MULTIMODAL INTERPRETABILITY METHODS


CLIP (Radford et al., 2021) learns joint image-text representations by training separate encoders
on large-scale image-text pairs, aligning the two modalities in a shared embedding space. This
design enables zero-shot transfer, allowing the model to perform various tasks based solely on
natural language prompts, without requiring task-specific annotations. However, the complexity
of multimodal reasoning raises the need for interpretability to ensure that the model’s predictions
are based on semantically meaningful features. Investigating CLIP’s interpretability is thus crucial
to determine whether it captures genuine vision-language associations or merely exploits spurious
dataset correlations.


A range of methods have been proposed to interpret vision-language pre-trained models, yet many
introduce limitations in fidelity, scalability, or practicality. M2IB (Wang et al., 2023) applies a
multimodal information bottleneck to filter irrelevant features, but increases architectural complexity.
COCOA(Lin et al., 2022) modifies Integrated Gradients with contrastive learning, requiring additional
positive and negative samples that may introduce irrelevant context. TEXTSPAN (Gandelsman et al.,
2023) and (Hossain et al.) rely on constructing sample-specific sets or selecting neighbours in
embedding space, making them dependent on external data and less generalizable. LICO (Lei et al.,
2024) retrains models to maintain cross-modal alignment, but its explanations apply to the altered
model, not the original one, and are affected by training randomness. FALCON (Kalibhat et al., 2023)
explains individual features via highly activating examples, but lacks per-instance interpretability.


These existing interpretability methods for vision-language models often rely on strong assumptions,
such as clean image-text alignment, access to contrastive examples, or task-specific supervision. Many
of these approaches require sampling additional inputs, retraining surrogate models, or modifying
the model architecture—factors that reduce their robustness and practicality in open-world settings.
In particular, when image-text pairs are noisy or entirely mismatched, these methods often still
generate forced explanations, undermining the reliability and trustworthiness of the interpretability
results, as shown in the Fig. 1. In contrast, our method avoids these limitations by introducing an
adaptive information bottleneck objective that does not assume reliable modality alignment. Instead
of enforcing a strict pairwise correspondence between images and text, we use a soft alignment
mechanism that adapts to different degrees of semantic consistency, dynamically adjusting the
optimisation of the compression and fitting terms in the information bottleneck theory for each
image-text pair, resulting in more effective and credible interpretability.


3 THE INFORMATION BOTTLENECK PRINCIPLE IN MULTI-MODAL
INTERPRETABILITY


In this section, we provide a detailed explanation of how the Information Bottleneck (IB) principle
can be applied to multimodal interpretability. We further identify key limitations arising from the
optimisation objective of IB in the multimodal setting, which motivate the design of our proposed
method.


The Information Bottleneck (IB) principle offers a theoretically grounded framework for interpretability by balancing the trade-off between compression and relevance. Specifically, it seeks to encode
an input variable _X_ into a latent representation _Z_ that preserves maximal information about a target
variable _Y_, while minimising the information retained about _X_ itself.


In the context of multimodal interpretability, this framework can be naturally extended by considering
_X_ and _Y_ as different modalities, such as text and image. For text-to-image attribution, the textual
input _xT_ is compressed while maximising the mutual information _I_ ( _Z_ ; _xI_ ) with the corresponding
image _xI_ . Conversely, for image-to-text attribution, the image _xI_ is compressed with the objective
of retaining information relevant to the text _xT_ . Under this formulation, attribution maps can be
derived by assessing the extent to which each part of the input survives the bottleneck. **Dimensions**
**that undergo strong compression (i.e.** **contribute less to** _I_ ( _Z_ ; _Y_ ) **) are deemed less informative**
**for cross-modal alignment, whereas those that retain more information are considered more**
**relevant.** Thus, the mutual information term _I_ ( _Z_ ; _Y_ ) not only enforces semantic alignment between
modalities but also provides a principled measure of feature importance based on information
retention, without relying on gradient-based or perturbation-based explanations.


3


**Underfitting** **Moderate Fit** **Overfitting**


𝑰𝒋


𝜷= 𝟎. 𝟎𝟏 𝜷= 𝟎. 𝟏 𝜷= 𝟏 𝜷= 𝟏𝟎


𝑰𝑲


**Underfitting** **Moderate Fit** **Overfitting**


Figure 2: Varying _β_ in the Information Bottleneck controls the trade-off between compression and
fitting. As the optimal _β_ for achieving good multimodal interpretability differs across samples (e.g.
the ‘Moderate Fit’ for _Ij_ and _Ik_ ), a fixed _β_ is suboptimal. This diversity in optimal _β_ directly
challenges the common practice in prior work, where _β_ is treated as a fixed hyperparameter for
all samples. Our observation shows that this one-size-fits-all assumption is inappropriate. This
observation motivates us to explore a dynamic adjustment of the IB trade-off parameter _β_ based on
the semantic relationship between the input image-text pair.


As shown in Fig. 2, this suggests, 1) **an inappropriate choice of** _β_ **can lead to either overfitting or**
**underfitting in the explanation process, resulting in overly specific or overly vague interpreta-**
**tions** . 2) **The optimal** _β_ **varies across different images; an adaptive, image-specific selection is**
**essential for generating reliable explanations** . Based on this motivation, we design our method,
which will be introduced in the following section.


4 METHOD


In this section, we introduce the _Adaptive_ _Information_ _Bottleneck_ _(AdaIB)_, a novel framework
that provides a sample-specific trade-off between sufficiency and representation compression. We
first present the variational inference formulation that makes AdaIB tractable for deep learning
optimisation. We then establish theoretical guarantees showing how AdaIB dynamically balances the
two core principles of the Information Bottleneck (IB) theory—sufficiency and minimality.


4.1 ADAPTIVE INFORMATION BOTTLENECK (ADAIB)


The Information Bottleneck (IB) principle (Tishby et al., 2000) formulates representation learning as
a trade-off between predictive sufficiency and input compression:


_L_ IB = _I_ ( _Z_ ; _Y_ ) _−_ _β · I_ ( _Z_ ; _X_ ) _,_ (1)


where _X_ denotes the input, _Y_ the target, and _Z_ a latent representation of _X_ . The hyperparameter
_β_ _>_ 0 controls the compression strength. The first term enforces that _Z_ retains task-relevant
information about _Y_, while the second term limits the amount of information _Z_ carries from _X_ .


Despite its elegance, the IB objective has two key limitations: (i) the coefficient _β_ is fixed across all
samples, preventing adaptation to the varying relevance of data between _X_ and _Y_ ; and (ii) in noisy or
multimodal settings, a fixed _β_ can result in underfitting of useful features or overfitting to irrelevant
signals.


To address these issues, we propose the _Adaptive Information Bottleneck (AdaIB)_ .


**Definition 1 (adaptive information bottleneck objective)** _Given_ _an_ _input_ _X,_ _a_ _target_ _Y,_ _and_ _a_
_latent representation Z, the AdaIB objective is defined as_


_L_ AdaIB = _f_ ( _X, Y_ ) _· I_ ( _Z_ ; _Y_ ) _−_ _g_ ( _f_ ( _X, Y_ )) _· I_ ( _Z_ ; _X_ ) _,_ (2)


_where_ _f_ : _X_ _×_ _Y_ _→_ (0 _, ∞_ ) _is_ _a_ _relevance_ _function_ _that_ _quantifies_ _the_ _statistical_ _dependence_
_between X_ _and Y, and g_ : (0 _, ∞_ ) _→_ (0 _, ∞_ ) _is a monotone non-increasing function that assigns the_
_corresponding compression weight._


4


In AdaIB, the relevance score _f_ ( _X, Y_ ) adaptively balances sufficiency and minimality. Large _f_
emphasizes _sufficiency_ by scaling up _I_ ( _Z_ ; _Y_ ), while small _f_ emphasizes _minimality_ via _g_ ( _f_ ). Unlike
classical IB with fixed _β_, AdaIB allows a sample-specific trade-off. See Section 4.3 for further
discussion.


**Proposition 1 (classical IB as a special case)** _For_ _any_ _relevance_ _value_ _f_ ( _X, Y_ ) _>_ 0 _,_ _define_ _the_
_effective coefficient β_ eff ( _t_ ) := _g_ ( _f_ ) _/f_ _._ _Then in_ ( _X, Y_ ) _, the AdaIB objective rewrites exactly as_

_L_ AdaIB = _f_ ( _X, Y_ )       - _I_ ( _Z_ ; _Y_ ) _−_ _β_ eff� _f_ ( _X, Y_ )� _I_ ( _Z_ ; _X_ )� _._


_In_ _particular,_ _if_ _f_ _≡_ _c_ _>_ 0 _and_ _g_ _≡_ _βc_ _are_ _constant_ _across_ _samples,_ _then_ _β_ eff _≡_ _β_ _and_ _AdaIB_
_reduces to the classical IB objective._


Proposition 1 above shows that AdaIB is a sample-wise reweighted IB: _f_ ( _X, Y_ ) acts as an importance
weight on each pair ( _x, y_ ), while the trade-off parameter _β_ eff ( _f_ ) adapts with relevance. Hence, unless
_f_ and _β_ eff are constant, AdaIB is a strict generalisation of IB.


4.2 VARIATIONAL OBJECTIVE AND OPTIMISATION FOR ADAIB


The objective in Definition 1 contains mutual-information terms that are intractable in general, as in
(Tishby et al., 2000). We derive a tractable variational formulation and an empirical training loss.


**Variational objective.** Using a standard lower bound for _I_ ( _Z_ ; _Y_ ) and a variational substitution for
_I_ ( _Z_ ; _X_ ), we obtain

_L_ [var] AdaIB [=][ E] _p_ ( _x,y_ )� _f_ ( _X, Y_ ) E _p_ ( _z|x_ ) log _q_ ( _y|z_ ) _−_ _g_   - _f_ ( _X, Y_ )� KL� _p_ ( _z|x_ ) _∥_ _r_ ( _z_ )� [�] _,_ (3)


where _q_ ( _y|z_ ) and _r_ ( _z_ ) are variational distributions. The additive term _H_ ( _Y_ ) E _p_ ( _x,y_ )[ _f_ ( _X, Y_ )] is constant w.r.t. _q, p, r_ and is dropped during optimization. Full derivations are provided in Appendix B.1.


**Empirical objective.** We estimate the expectations via Monte Carlo with the reparameterization
trick, _zi_ _∼_ _pψ_ ( _z|xi_ ), which yields


This estimator is sample-specific, end-to-end differentiable, and avoids explicit MI estimation.


4.3 SUFFICIENCY AND MINIMALITY BALANCE


In the classical Information Bottleneck (IB) theory, the two central desiderata of a representation
are _sufficiency_ and _minimality_ . Sufficiency requires that the learned representation _Z_ preserves all
task-relevant information about the target _Y_ . Minimality requires that _Z_ discards all task-irrelevant
information from the input _X_ .


The Information Bottleneck principle aims to learn representations that are both _sufficient_ and _minimal_ .
The adaptive relevance function _f_ ( _X, Y_ ) promotes sufficiency by amplifying the fitting term when
_X_ and _Y_ are strongly correlated, while the inverse weighting _g_ ( _f_ ( _X, Y_ )) enforces minimality by
strengthening compression when _X_ and _Y_ are weakly correlated. In what follows, we establish 3
theoretical properties of AdaIB:


(i) sufficiency when _f_ is large (Theorem 1);
(ii) minimality when _f_ is small (Theorem 2);
(iii) an adaptive trade-off balancing the sufficiency and minimality (Theorem 3).


**Theorem 1 (sufficiency at high relevance)** _Let f_ : _X_ _× Y_ _→_ (0 _, ∞_ ) _and g_ : (0 _, ∞_ ) _→_ (0 _, ∞_ ) _be_
_locally_ _Lipschitz_ _with_ _g_ _non-increasing._ _Assume_ _that_ _for_ _the_ _considered_ _class_ _of_ _representations_
_Z, the mutual informations satisfy I_ ( _Z_ ; _Y_ ) _<_ _∞_ _and I_ ( _Z_ ; _X_ ) _<_ _∞._ _Then as f_ ( _X, Y_ ) _→∞, the_
_AdaIB objective satisfies_
_L_ AdaIB _∼_ _f_ ( _X, Y_ ) _· I_ ( _Z_ ; _Y_ ) _,_


5


_L_ ˆ = [1]

_N_


_N_
�� - - - [�]

_f_ ( _xi, yi_ ) log _q_ ( _yi|zi_ ) _−_ _g_ ( _f_ ( _xi, yi_ ) KL _p_ ( _z|xi_ ) _∥_ _r_ ( _z_ ) _._ (4)


_i_ =1


_i.e._ _L_ AdaIB _/_ ( _f_ ( _X, Y_ ) _I_ ( _Z_ ; _Y_ )) _→_ 1 _._ _Consequently, the optimisation places dominant emphasis on_
_maximising I_ ( _Z_ ; _Y_ ) _, driving Z_ _toward sufficiency._


Theorem 1 is proved in B.2 Proof 1. It indicates that when _X_ and _Y_ are highly correlated (e.g. a
clear image with a matching caption), _f_ ( _X, Y_ ) is large; AdaIB thus prioritises preserving predictive
information about _Z_ and _Y_ .


**Theorem 2 (minimality at low relevance)** _Let f_ : _X ×Y_ _→_ (0 _, ∞_ ) _and g_ : (0 _, ∞_ ) _→_ (0 _, ∞_ ) _with_
_g nonincreasing._ _Assume I_ ( _X_ ; _Y_ ) _< ∞_ _and that_ lim _u→_ 0+ _[g]_ [(] _u_ _[u]_ [)] = + _∞_ _._ _Then for every ε >_ 0 _there_

_exists η_ _>_ 0 _such that whenever_ 0 _< f_ ( _X, Y_ ) _< η, any maximizer Z_ _[∗]_ _of L_ AdaIB _satisfies_

_I_ ( _Z_ _[∗]_ ; _X_ ) _≤_ inf [+] _[ε.]_
_Z_ _[I]_ [(] _[Z]_ [;] _[ X]_ [)]

_Thus, as f_ _→_ 0 [+] _(hence g_ ( _f_ ) _/f_ _→∞), the objective is dominated by the compression term and the_
_solution approaches a minimal-information representation._


Theorem 2 is proved in Appendix B.2 Proof 2. It indicates that in the low-relevance regime, the
AdaIB objective is dominated by the compression term, _−g_ ( _f_ ) _I_ ( _Z_ ; _X_ ). Hence, any maximiser _Z_ _[⋆]_
approaches a representation that minimises _I_ ( _Z_ ; _X_ ) within the model class, avoiding overfitting to
noise or mismatched pairs. This behaviour reflects the IB principle of minimality.


**Theorem 3 (adaptive sufficiency–minimality trade-off)** _Let f_ : _X ×Y_ _→_ (0 _, ∞_ ) _be the relevance_
_score_ _and_ _g_ : (0 _, ∞_ ) _→_ (0 _, ∞_ ) _be_ _positive_ _and_ _non-increasing._ _Assume_ _I_ ( _Z_ ; _X_ ) _<_ _∞_ _and_
_I_ ( _Z_ ; _Y_ ) _<_ _∞_ _for all admissible Z, and define the effective compression weight λ_ ( _f_ ) := _g_ ( _f_ ) _/f_ _._
_Then λ_ ( _f_ ) _is non-increasing in f_ _(see Lemma B.1), and the AdaIB objective factors as_


                  -                   _L_ AdaIB = _f_ _I_ ( _Z_ ; _Y_ ) _−_ _λ_ ( _f_ ) _I_ ( _Z_ ; _X_ ) _._


Consequently, as _f_ increases, _λ_ ( _f_ ) decreases monotonically while the sufficiency term is scaled by
_f_, implementing a sample-wise trade-off that shifts emphasis from compression to sufficiency. The
limiting behaviours as _f_ _→_ 0 [+] and _f_ _→∞_ are characterised by Theorems 1 and 2, respectively.
The proof of Theorem 3 (Appendix B.2, Proof 4) confirms that between these extremes, the objective
follows the adaptive trade-off principle.


In addition, AdaIB also satisfies a bounded leakage property, ensuring it never incentivises gratuitous
dependence on _X_ at fixed predictive power. (see Appendix B.3).


At any fixed sufficiency level _I_ ( _Z_ ; _Y_ ), AdaIB strictly prefers representations with smaller _I_ ( _Z_ ; _X_ ),
and even in the high-relevance limit, it never increases _I_ ( _Z_ ; _X_ ) without also improving _I_ ( _Z_ ; _Y_ ).
Detailed arguments are deferred to Appendix B.2.


4.4 LEARNABLE RELEVANCE AND COMPRESSION FUNCTIONS


We now extend AdaIB by allowing both the relevance function _f_ and the compression function _g_ to
be learned from data as independent functions _fθ_ and _gϕ_ . In Sections 4.1–4.3, we assumed _g_ was a
nonincreasing function of _f_, enforcing a specific monotonic relationship.


**Definition 2 (decoupled learnable functions)** _We_ _parameterise_ _independent_ _relevance_ _and_ _com-_
_pression functions:_

_fθ_ ( _X, Y_ ) = _ϵf_ + act� _hθ_ ( _X, Y_ )� _,_ _gϕ_ ( _X, Y_ ) = _ϵg_ + act� _uϕ_ ( _X, Y_ )� _,_

_where ϵf_ _, ϵg_ _>_ 0 _ensure strict positivity, and_ act( _·_ ) _is a nonnegative activation._ _The AdaIB objective_
_becomes:_
_L_ AdaIB = _fθ_ ( _X, Y_ ) _I_ ( _Z_ ; _Y_ ) _−_ _gϕ_ ( _X, Y_ ) _I_ ( _Z_ ; _X_ ) _._


Under Definition 2, we establish in Appendix B.4 that stationary optimal points still exist. Moreover,
the properties developed in Sections 4.1–4.3 continue to hold after decoupling (see Appendix B.5).
In particular, the adaptive sufficiency–minimality trade-off remains valid (Appendix B.6).


In practice, different degrees of flexibility can be considered, such as using a fixed _f_ with learnable
_gϕ_ ( _f_ ), a learnable _fθ_ ( _X, Y_ ) with fixed _g_ ( _f_ ), or jointly learning both _fθ_ ( _X, Y_ ) and _gϕ_ ( _X, Y_ ). These
variants extend the applicability of AdaIB across heterogeneous settings.


6


5 EXPERIMENTS


5.1 DATASETS AND BASELINES


In this study, we adopt the experimental setup from M2IB (Wang et al., 2023) and NIB (Zhu et al.,
2025), leveraging the pre-trained CLIP model with a Vision Transformer (ViT-B/32) (Radford et al.,
2021) as the visual encoder. CLIP’s ability to jointly align visual and textual modalities has shown
remarkable performance across various multimodal tasks.


While prior work often focuses on relatively small datasets such as Flickr8k (Hodosh et al., 2013),
which includes 8,000 images paired with natural language descriptions. Additionally, we aim to evaluate the model’s generalisation ability as thoroughly as possible. To this end, we conduct experiments
on larger and more diverse datasets, specifically Conceptual Captions 3M (CC3M) (Sharma et al.,
2018) and LAION-400M (Schuhmann et al., 2021), both of which provide large-scale image-text pairs
suitable for learning robust multimodal representations. CC3M consists of automatically generated
image-text alignments from the web, offering a rich training signal for vision-language learning.
LAION-400M further expands this scale with hundreds of millions of image-text pairs, enabling
comprehensive evaluation of the model’s generalisation capabilities across diverse domains. More
details about the datasets can be found in Appendix A.1.


For baselines, we compare against several well-established attribution techniques to evaluate their
effectiveness. The baseline methods include NIB (Zhu et al., 2025), M2IB (Wang et al., 2023),
RISE (Petsiuk et al., 2018), Grad-CAM (Selvaraju et al., 2017), the method by (Chefer et al., 2021),
Saliency Maps (Simonyan, 2013), MFABA (Zhu et al., 2024), and FastIG (Hesse et al., 2021).


5.2 EXPERIMENTAL SETTINGS


Following the approach of MI2B and NIB, we insert an information bottleneck into a specified
layer of both the text and image encoders within the CLIP model for each “image-caption” pair.
To train the bottleneck, we adopt the same procedure as the Per-Sample Bottleneck method from
IBA (Schulz et al., 2020), repeating each sample 10 times to stabilise optimisation. We optimise by
10 steps using Adam with a learning rate of 1. To further enhance stability, we also apply gradient
clipping during optimisation, specifically capping the global L2-norm of the gradients at 1.0. For the
function _f_ ( _X, Y_ ), we choose the L2 distance by default. The function _g_ ( _f_ ( _X, Y_ )) is implemented
as a learnable shallow MLP with a 1→32→1 architecture and a ReLU activation function. We
keep _f_ to be heuristically chosen and _g_ to be trainable, which we found to be optimal based on our
experimental results. Detailed ablation studies concerning these choices for _f_ and _g_ are discussed in
the Appendix E. All results are reported as the mean ± standard deviation over 5 independent runs to
ensure statistical reliability. All experiments were performed on a single 4090 GPU. The detailed
experimental setup can be found in Appendix A.2.


5.3 EVALUATION METRICS


Consistent with prior work (Wang et al., 2023; Zhu et al., 2025), we evaluate the quality of our
generated attribution maps using a comprehensive suite of metrics from (Chattopadhay et al., 2018;
Hooker et al., 2019).


**Confidence drop.** This metric quantifies the drop in model confidence when only the most salient
features are preserved. A high-quality attribution method should identify features that are sufficient
for the model’s prediction; consequently, their preservation should lead to only a minimal drop in
confidence. It is computed as: Drop = _N_ [1] - _Ni_ =1 [max(0] _[, o][i][ −]_ _[s][i]_ [)][ where] _[ o][i]_ [and] _[ s][i]_ [denote the original]

and post-masking image-text cosine similarities, respectively. Lower values are better.


**Confidence** **increase.** Conversely, this metric evaluates whether removing irrelevant features
reduces noise and thereby _increases_ the model’s confidence. It is defined as the proportion of samples
for which the confidence improves after masking: Incr. = _N_ 1 - _Ni_ =1 [I][(] _[o][i]_ _[<]_ _[s][i]_ [)] [where] [I][(] _[·]_ [)] [is] [the]
indicator function. Higher values are better.


7


Table 1: Comparison of interpretability methods across multiple vision-language datasets. We report
the mean and standard deviation ( _mean ± std_ ) on both image and text modalities using two metrics:
Drop (Confidence Drop) and Incr. (Confidence Increase)


.


**Dataset** **Method** **M2IB** **Grad-CAM** **Chefer et al.** **MFABA** **FastIG** **NIB** **AdaIB (Ours)**


cc3M-I Drop _↓_ **1.11** _±_ 0.10 8.07 _±_ 0.23 6.78 _±_ 0.11 2.38 _±_ 0.11 1.02 _±_ 0.07 1.03 _±_ 0.10 **1.01** _±_ 0.08
cc3M-I Incr. _↑_ 37.20 _±_ 3.75 7.20 _±_ 1.52 11.00 _±_ 2.26 21.90 _±_ 2.63 32.90 _±_ 2.48 38.80 _±_ 3.63 **40.70** _±_ 2.60
cc3M-T Drop _↓_ **0.90** _±_ 0.08 1.79 _±_ 0.14 1.02 _±_ 0.08 1.60 _±_ 0.18 1.42 _±_ 0.15 1.13 _±_ 0.10 1.07 _±_ 0.11


Table 2: The quantitative comparison of our method against several baselines on the Refcoco dataset,
using various metrics including pointing-game IoU and Drop/Incr.


**Metric** **M2IB** **Grad-CAM** **Chefer et al.** **MFABA** **FastIG** **NIB** **AdaIB (Ours)**


mIoU _↑_ 14.20 _±_ 0.57 8.89 _±_ 0.12 10.97 _±_ 0.01 4.90 _±_ 0.13 9.47 _±_ 0.10 11.96 _±_ 0.33 **16.46** _±_ 0.32


**Pointing-game** **IoU.** To evaluate the spatial grounding performance of the attribution map, we
employ the Pointing-Game Intersection over Union (IoU) on the RefCOCO dataset (Kazemzadeh
et al., 2014). This metric quantifies the overlap between a binary mask, generated by thresholding the
attribution map, and the ground-truth bounding box of the referenced object. Higher IoU is better.
The threshold used to generate the binary mask from the attribution map is set to 0 _._ 5 by default.


**Remove and retrain (ROAR).** We also adapt the computationally efficient Remove and Retrain
(ROAR) benchmark (Hooker et al., 2019). The purpose of this benchmark is to assess how critical
the features identified by our method are to the model’s predictions. This is achieved by removing the
most salient features and measuring the subsequent drop in zero-shot image-text retrieval performance.
The ROAR score is calculated by the formula _[ACC]_ _Acc_ _[o][−][ACC]_ _o_ _[c]_, where _ACCo_ is the accuracy on the

original data and _ACCc_ is the accuracy on the data after feature removal. A higher score indicates
a more effective attribution method. A detailed description of our implementation is available in
Appendix F.


5.4 EXPERIMENTAL RESULTS


**Quantitative results:** Table 1, Table 2 and Table 3 present a comparison of attribution performance
across multiple vision-language datasets using three different metrics. On the Refcoco dataset, AdaIB
achieves the highest pointing-game mIoU of 16.46 _±_ 0.32, significantly outperforming all baselines.
This result highlights its ability to generate attribution maps that are not only accurate but also
spatially precise. Furthermore, on the ROAR metric, AdaIB consistently leads on diverse datasets
like cc3M, Flickr8k, and Laion400m for both image-to-text (i2t) and text-to-image (t2i) retrieval
tasks. For instance, on the Flickr8k dataset, AdaIB achieves the best scores for both i2t-oc (66.95 _±_
2.23) and t2i-oc (71.85 _±_ 2.03). Additionally, AdaIB shows strong performance on the Drop and
Increase metrics, securing the lowest Drop scores in multiple tasks (e.g. 1.01 ± 0.08 for cc3M-I) and
the highest Increase scores (e.g. 29.80 _±_ 1.44 for Flickr8k-I). These comprehensive results confirm
that AdaIB not only generates reliable and accurate attributions but also exhibits strong robustness
and generalizability across various complex datasets and tasks. **Qualitative results:** The comparison
of the attribution method visualisation results of different methods can be seen in the Appendix G.


8


Table 3: Quantitative results of the ROAR metric, which evaluates zero-shot performance on imageto-text (i2t) and text-to-image (t2i) retrieval tasks under different corruption settings. For _i2t-oc_,
the image features are original (o) while the text features are corrupted (c). For _i2t-co_, the image is
corrupted (c) and the text is original (o). The same logic applies to the t2i metrics. An upward arrow
( _↑_ ) indicates that higher scores are better. The best result in each row is highlighted in **bold** .


**Dataset** **Metric** **M2IB** **Grad-CAM** **Chefer et al.** **MFABA** **FastIG** **NIB** **AdaIB (Ours)**


6 ANALYSIS OF THE ADAIB


**Misalignment image-caption analysis.** We discussed the performance of AdaIB on misalignment
image-caption pairs in Appendix C. Our method maintains leading performance under artificially set
misalignment of image and caption. While the other baseline models fail to distinguish the matched
and mismatched image-caption pairs (Fig. 3 in Appendix).


**Dynamic** _β_ **of AdaIB.** In Appendix D, we study and demonstrate the changing relationship between
_f_ and _g_ in AdaIB under different data samples. Interestingly, we found that for different samples
under the similar _f_ will produce different _g_ values, indicating that AdaIB’s adaptability is not solely
based on the image-caption distance calculated by _f_, but can also be adjusted based on the intrinsic
information of the data itself. This is consistent with the different optimal beta values for different
samples shown in Fig. 2.


We propose AdaIB, a powerful attribution framework that relaxes the common assumption of
strict semantic alignment in multimodal explanations. While many existing methods can overfit or
produce misleading attributions when faced with misaligned inputs, AdaIB takes a different approach.
Grounded in the Information Bottleneck (IB) principle, it dynamically balances compression and
prediction terms based on the relationship between modalities. Without enforcing strict pairwise
alignment, AdaIB adapts automatically to various input conditions, delivering more reliable and
interpretable explanations. This provides a more robust and dependable approach to multimodal
interpretability in real-world settings.


However, despite its strengths, our framework has limitations. We focus on attribution under
semantic misalignment in large-scale web-scraped datasets (e.g. CC3M, LAION-400M). While
AdaIB addresses such cases effectively, we do not examine its performance on subtler mismatches,
such as sarcasm, puns, or metaphors. These non-literal relationships remain a significant challenge
for attribution, and future work will explore extending AdaIB to handle them.


9


**Computational** **cost.** We report the computational
cost and memory consumption in Table 4. The core
of our method introduces only a shallow MLP with a
single layer of additional parameters and feature distance calculation, resulting in no significant increase in
computational load or memory consumption compared
to the original CLIP model.


7 LIMITATIONS AND CONCLUSION


Table 4: Computational efficiency of AdaIB
compared to baselines M2IB and NI


**Metric** **M2IB** **NIB** **AdaIB (Ours)**


FPS 2.47 12.5 2.27
Memory 3.28GB 2.32GB 3.28GB


REFERENCES


Luitzen Egbertus Jan Brouwer. Über abbildung von mannigfaltigkeiten. _Mathematische annalen_, 71
(1):97–115, 1911.


Aditya Chattopadhay, Anirban Sarkar, Prantik Howlader, and Vineeth N Balasubramanian. Gradcam++: Generalized gradient-based visual explanations for deep convolutional networks. In _2018_
_IEEE winter conference on applications of computer vision (WACV)_, pp. 839–847. IEEE, 2018.


Hila Chefer, Shir Gur, and Lior Wolf. Generic attention-model explainability for interpreting bi-modal
and encoder-decoder transformers. In _Proceedings of the IEEE/CVF International Conference on_
_Computer Vision_, pp. 397–406, 2021.


Yossi Gandelsman, Alexei A Efros, and Jacob Steinhardt. Interpreting clip’s image representation via
text-based decomposition. _arXiv preprint arXiv:2310.05916_, 2023.


Robin Hesse, Simone Schaub-Meyer, and Stefan Roth. Fast axiomatic attribution for neural networks.
_Advances in Neural Information Processing Systems_, 34:19513–19524, 2021.


Micah Hodosh, Peter Young, and Julia Hockenmaier. Framing image description as a ranking task:
Data, models and evaluation metrics. _Journal_ _of_ _Artificial_ _Intelligence_ _Research_, 47:853–899,
2013.


Sara Hooker, Dumitru Erhan, Pieter-Jan Kindermans, and Been Kim. A benchmark for interpretability
methods in deep neural networks. _Advances in neural information processing systems_, 32, 2019.


M Shifat Hossain, Chase Walker, Sumit Kumar Jha, and Rickard Ewetz. Explaining contrastive
models using exemplars: Explanation, confidence, and knowledge limits.


Neha Kalibhat, Shweta Bhardwaj, C Bayan Bruss, Hamed Firooz, Maziar Sanjabi, and Soheil Feizi.
Identifying interpretable subspaces in image representations. In _International_ _Conference_ _on_
_Machine Learning_, pp. 15623–15638. PMLR, 2023.


Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to
objects in photographs of natural scenes. In Alessandro Moschitti, Bo Pang, and Walter Daelemans
(eds.), _Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing_
_(EMNLP)_, pp. 787–798, Doha, Qatar, October 2014. Association for Computational Linguistics.
doi: 10.3115/v1/D14-1086. [URL https://aclanthology.org/D14-1086.](https://aclanthology.org/D14-1086)


Yiming Lei, Zilong Li, Yangyang Li, Junping Zhang, and Hongming Shan. Lico: explainable models
with language-image consistency. _Advances in Neural Information Processing Systems_, 36, 2024.


Chris Lin, Hugh Chen, Chanwoo Kim, and Su-In Lee. Contrastive corpus attribution for explaining
representations. In _The Eleventh International Conference on Learning Representations_, 2022.


Deng Pan, Xin Li, and Dongxiao Zhu. Explaining deep neural network models with adversarial
gradient integration. In _Thirtieth International Joint Conference on Artificial Intelligence (IJCAI)_,
2021.


Vitali Petsiuk, Abir Das, and Kate Saenko. Rise: Randomized input sampling for explanation of
black-box models. In _Proceedings of the British Machine Vision Conference (BMVC)_, 2018.


Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal,
Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual
models from natural language supervision. In _International conference on machine learning_, pp.
8748–8763. PmLR, 2021.


Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. " why should i trust you?" explaining the
predictions of any classifier. In _Proceedings of the 22nd ACM SIGKDD international conference_
_on knowledge discovery and data mining_, pp. 1135–1144, 2016.


Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis,
Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of
clip-filtered 400 million image-text pairs. _arXiv preprint arXiv:2111.02114_, 2021.


10


Karl Schulz, Leon Sixt, Federico Tombari, and Tim Landgraf. Restricting the flow: Information
bottlenecks for attribution. _arXiv preprint arXiv:2001.00396_, 2020.


Ramprasaath R Selvaraju, Michael Cogswell, Abhishek Das, Ramakrishna Vedantam, Devi Parikh,
and Dhruv Batra. Grad-cam: Visual explanations from deep networks via gradient-based localization. In _Proceedings of the IEEE international conference on computer vision_, pp. 618–626,
2017.


Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned,
hypernymed, image alt-text dataset for automatic image captioning. In _Proceedings of the 56th_
_Annual Meeting of the Association for Computational Linguistics (Volume 1:_ _Long Papers)_, pp.
2556–2565, 2018.


Karen Simonyan. Deep inside convolutional networks: Visualising image classification models and
saliency maps. _arXiv preprint arXiv:1312.6034_, 2013.


Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In
_International conference on machine learning_, pp. 3319–3328. PMLR, 2017.


Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. _arXiv_
_preprint physics/0004057_, 2000.


Ying Wang, Tim GJ Rudner, and Andrew G Wilson. Visual explanations of image-text representations
via multi-modal information bottleneck attribution. _Advances in Neural Information Processing_
_Systems_, 36:16009–16027, 2023.


Zhiyu Zhu, Huaming Chen, Jiayu Zhang, Xinyi Wang, Zhibo Jin, Minhui Xue, Dongxiao Zhu, and
Kim-Kwang Raymond Choo. Mfaba: A more faithful and accelerated boundary-based attribution
method for deep neural networks. In _Proceedings of the AAAI Conference on Artificial Intelligence_,
volume 38, pp. 17228–17236, 2024.


Zhiyu Zhu, Zhibo Jin, Jiayu Zhang, Nan Yang, Jiahao Huang, Jianlong Zhou, and Fang Chen.
Narrowing information bottleneck theory for multimodal image-text representations interpretability.
_arXiv preprint arXiv:2502.14889_, 2025.


11


THE USE OF LARGE LANGUAGE MODELS (LLMS)


In the preparation of this manuscript, we employed large language models (LLMs) to assist with
improving the clarity and readability of the text. Specifically, LLMs were used to refine grammar,
enhance fluency, and polish the overall presentation of ideas, while all conceptual contributions,
analyses, and experimental results remain the sole work of the authors.


REPRODUCIBILITY STATEMENT


We have made every effort to ensure the reproducibility of our work. All training protocols, model
architectures, hyperparameters, and evaluation settings are described in detail in the main text and
appendix. Upon acceptance of this paper, we will release our implementation and experimental code
in a public GitHub repository to further facilitate reproducibility and future research.


A IMPLEMENTATION DETAILS.


A.1 DATASETS


In our study, we evaluate the model’s performance and generalisation capabilities on large-scale,
diverse datasets. Below are the detailed descriptions of the primary datasets used for training and
evaluation.


**Flickr8k** Flickr8k (Hodosh et al., 2013) is a classic and widely-used benchmark dataset in the field
of image captioning. It consists of 8,000 images collected from the Flickr website. The defining
characteristic of this dataset is its high-quality, human-generated annotations. Each image is paired
with five independent, descriptive sentences written by human annotators. Compared to large-scale,
web-crawled datasets, Flickr8k is significantly smaller but features clean, reliable captions. In this
work, it serves as a baseline to contrast with the large-scale, noisy data environments where we aim
to test our model’s generalisation capabilities.


**Conceptual Captions 3M (CC3M)** The Conceptual Captions 3M dataset (Sharma et al., 2018) is
a large-scale collection of approximately 3.3 million image-URL and caption pairs, automatically
sourced from alt-text attributes of images on the web. Unlike curated datasets like Flickr8k, the
captions in CC3M are not human-generated annotations but are naturally occurring descriptions.
While this results in a higher level of noise and less descriptive detail, the sheer scale and diversity of
the data provide a rich training signal for learning robust vision-language alignments.


**LAION-400M Subset** LAION-400M (Schuhmann et al., 2021) is an open-source, massive-scale
dataset containing approximately 400 million image-text pairs scraped from the web. The pairs
were filtered using CLIP to ensure a baseline level of semantic alignment between the image and its
corresponding text.


Due to the immense size of the full dataset and computational constraints, our experiments were
conducted on a specific subset of LAION-400M. Specifically, we utilised the initial data shards
numbered from 00000 to 01241. Each shard contains approximately 25,000 samples. Our selection
of 1,242 shards (from 0 to 1,241 inclusive) results in a subset of approximately 31 million image-text
pairs that constitutes roughly 7.8% of the full 400-million-pair dataset. This selected portion is
substantial enough to ensure robust model training while remaining computationally manageable
compared to the previous datasets used in the baselines.


**RefCOCOg (UMD)** The RefCOCOg dataset was collected through an interactive game setting.
For our experiments, we use the version from the Hugging Face Hub under the identifier lmmslab/RefCOCOg. This specific dataset contains approximately 7.57k image-text pairs, each with a
corresponding segmentation mask for the target object. The referring expressions in this dataset are
significantly longer, more complex, and use more conversational language, posing a greater challenge
for language understanding.


12


A.2 EXPERIMENTAL SETTINGS


Our experimental setup is adapted from the protocol of M2IB (Zhu et al., 2025), as the setup for NIB
was not detailed in their work. We expand on M2IB (Wang et al., 2023)’s approach to create a more
robust evaluation framework. To ensure statistical reliability, all experiments were independently
repeated 5 times, using a different random seed for each run (42, 43, 44, 45, and 46). The final results
are consistently reported as the mean and standard deviation of these runs.


Our sampling strategy varies by dataset to balance thoroughness and computational cost. For the
smaller, task-specific datasets RefCOCOg and Flickr8k, we used the entirety of the data in each of
the 5 runs to conduct a comprehensive evaluation. For the large-scale CC3M and LAION-400M
datasets, we randomly sampled 2,000 pairs for each run to efficiently assess the model’s generalisation
performance on diverse data. In M2IB (Zhu et al., 2025), only 500 pairs were sampled per run.


B THEORETICAL PROOF


B.1 VARIATIONAL DERIVATION STEPS


**Step 1:** **Variational lower bound for** _I_ ( _Z_ ; _Y_ ) **.** For the sufficiency term, we introduce a variational
distribution _q_ ( _y|z_ ) to approximate the true posterior _p_ ( _y|z_ ).


= E _p_ ( _z,y_ )[log _q_ ( _y|z_ )] + _H_ ( _Y_ ) + E _p_ ( _z_ )�KL� _p_ ( _y|z_ ) _∥_ _q_ ( _y|z_ )�� _._ (5)


_I_ ( _Z_ ; _Y_ ) = E _p_ ( _z,y_ )


log _[p]_ [(] _[y][|][z]_ [)]

_p_ ( _y_ )


This identity follows by adding and subtracting log _q_ ( _y|z_ ) inside the expectation and using E _p_ ( _z,y_ )[log _p_ ( _y_ )] = E _p_ ( _y_ )[log _p_ ( _y_ )] = _−H_ ( _Y_ ) and E _p_ ( _z,y_ )[log _p_ ( _y|z_ ) _−_ log _q_ ( _y|z_ )] =
E _p_ ( _z_ )[KL( _p_ ( _y|z_ ) _∥_ _q_ ( _y|z_ ))]. By the non-negativity of KL divergence, dropping the last term yields
the lower bound.
_I_ ( _Z_ ; _Y_ ) _≥_ E _p_ ( _z,y_ )[log _q_ ( _y|z_ )] + _H_ ( _Y_ ) _._ (6)


**Step 2:** **Variational upper bound for** _I_ ( _Z_ ; _X_ ) **.** For the compression term, we approximate the
intractable marginal _p_ ( _z_ ) with a variational distribution _r_ ( _z_ ) (independent of _x_ ). Then


(7)


E _p_ ( _x_ )�KL( _p_ ( _z|x_ ) _∥_ _r_ ( _z_ ))� = E _p_ ( _x,z_ )


= E _p_ ( _x,z_ )


log _[p]_ [(] _[z][|][x]_ [)]

_r_ ( _z_ )


log _[p]_ [(] _[z][|][x]_ [)]

_p_ ( _z_ )


+ E _p_ ( _z_ )


log _[p]_ [(] _[z]_ [)]

_r_ ( _z_ )


(8)


= _I_ ( _Z_ ; _X_ ) + KL� _p_ ( _z_ ) _∥_ _r_ ( _z_ )� _._ (9)


By non-negativity of KL,


_I_ ( _Z_ ; _X_ ) _≤_ E _p_ ( _x_ )�KL( _p_ ( _z|x_ ) _∥_ _r_ ( _z_ ))� _,_ (10)

with equality when _r_ ( _z_ ) = _p_ ( _z_ ).


**Step 3:** **Variational lower bound for AdaIB.** Substituting the bounds from Steps 1–2 into the
AdaIB objective (Def. 1), and taking the outer expectation over ( _X, Y_ ), we obtain the lower bound.

_L_ [var] AdaIB [=][ E] _p_ ( _x,y_ )� _f_ ( _X, Y_ ) E _p_ ( _z|x_ )[log _q_ ( _y|z_ )] _−_ _g_   - _f_ ( _X, Y_ )� KL� _p_ ( _z|x_ ) _∥_ _r_ ( _z_ )� [�] _._ (11)


The additive term _H_ ( _Y_ ) E _p_ ( _x,y_ )[ _f_ ( _X, Y_ )] does not depend on _q, p, r_ and is omitted for optimisation
over these parameters.


**Step** **4:** **Empirical** **approximation.** Given _N_ i.i.d. samples _{_ ( _xi, yi_ ) _}_ _[N]_ _i_ =1 [,] [Eq.] [equation] [11] [is]
estimated by Monte Carlo:


_L_ ˆ = [1]

_N_


_N_

- - - - - - [�]

_f_ ( _xi, yi_ ) log _q_ ( _yi|zi_ ) _−_ _g_ _f_ ( _xi, yi_ ) KL _p_ ( _z|xi_ ) _∥_ _r_ ( _z_ ) _,_ _zi_ _∼_ _p_ ( _z|xi_ ) _,_ (12)


_i_ =1


where _zi_ is drawn via the reparameterization trick to obtain low-variance gradients.


13


**Step 5:** **Practical instantiations of** _f_ **and** _g_ **.** To complete the specification of AdaIB, we propose
stable and practical choices for the relevance function _f_ and the compression mapping _g_ . We consider
two types of _f_ that ensure larger values indicate stronger relevance:


**(i) Similarity-based:**


_s_ ( _x, y_ ) = [1+cos(] 2 _[x,y]_ [)] _∈_ [0 _,_ 1] _,_ _f_ ( _x, y_ ) = softplus( _τs_ ( _x, y_ )) + _ϵf_ _,_ _τ_ _>_ 0 _,_ _ϵf_ _>_ 0 _._


Here cos is computed on _ℓ_ 2-normalized features.


**(ii) Inverse-distance-based:**


1
_d_ ( _x, y_ ) = _∥x −_ _y∥p,_ _p ∈{_ 1 _,_ 2 _},_ _f_ ( _x, y_ ) = _,_ _ϵf_ _>_ 0 _._
_d_ ( _x, y_ ) + _ϵf_


For the compression mapping, we adopt


1
_g_ ( _f_ ) = _,_ _ϵg_ _>_ 0 _,_
_f_ + _ϵg_


which ensures stable optimisation by preventing excessively large weights when _f_ is small. Substituting these choices into the empirical objective equation 12 yields the final, concrete objective
function:


_Because_ _I_ ( _Z_ ; _X_ ) _<_ _∞,_ _we_ _have_ _ε_ ( _t_ ) _I_ ( _Z_ ; _X_ ) _→_ 0 _,_ _and_ _thus_ _the_ _bracketed_ _term_ _converges_ _to_
_I_ ( _Z_ ; _Y_ ) _._ _Multiplying by f_ ( _X, Y_ ) = _t yields_


_L_ AdaIB _∼_ _f_ ( _X, Y_ ) _· I_ ( _Z_ ; _Y_ ) _,_


_where “∼” denotes asymptotic equivalence (ratio →_ 1 _) as t →∞._


**Proof 2 (Minimality at low relevance)** _Rewrite L_ AdaIB( _Z_ ) = _−_ _g_ ( _t_ )� _IX_ ( _Z_ ) _−_ _δ_ ( _t_ ) _IY_ ( _Z_ )� _with_
_δ_ ( _t_ ) := _[f]_ _g_ ( [(] _t_ _[t]_ ) [)] _[.]_ _[Let][ ε >]_ [ 0] _[ and pick][ Z]_ [min] _[such that][ I][X]_ [(] _[Z]_ [min][)] _[ ≤]_ [inf] _[Z][ I][X]_ [(] _[Z]_ [) +] _[ ε/]_ [2] _[.]_ _[Optimality gives]_

_L_ ( _Z_ _[∗]_ ) _≥L_ ( _Z_ min) _, hence_

_IX_ ( _Z_ _[∗]_ ) _−_ _δ_ ( _t_ ) _IY_ ( _Z_ _[∗]_ ) _≤_ _IX_ ( _Z_ min) _−_ _δ_ ( _t_ ) _IY_ ( _Z_ min) _,_


_so_

_IX_ ( _Z_ _[∗]_ ) _−_ _IX_ ( _Z_ min) _≤_ _δ_ ( _t_ )   - _IY_ ( _Z_ _[∗]_ ) _−_ _IY_ ( _Z_ min)� _≤_ _δ_ ( _t_ ) �� _IY_ ( _Z_ _∗_ ) _−_ _IY_ ( _Z_ min)�� _._


_By_ _the_ _data_ _processing_ _inequality_ _I_ ( _Z_ ; _Y_ ) _≤_ _I_ ( _X_ ; _Y_ ) _,_ _we_ _have_ �� _IY_ ( _Z_ _∗_ ) _−_ _IY_ ( _Z_ min)�� _≤_
2 _I_ ( _X_ ; _Y_ ) =: 2 _MY ._ _Therefore_

_IX_ ( _Z_ _[∗]_ ) _−_ _IX_ ( _Z_ min) _≤_ 2 _δ_ ( _t_ ) _MY ._

_Since_ _δ_ ( _t_ ) = _f_ ( _t_ ) _/g_ ( _t_ ) _→_ 0 _as_ _f_ ( _t_ ) _→_ 0 [+] _, choose tε so that_ 2 _δ_ ( _t_ ) _MY_ _≤_ _ε/_ 2 _for t < tε._ _Then_

_IX_ ( _Z_ _[∗]_ ) _≤_ _IX_ ( _Z_ min) + _ε/_ 2 _≤_ inf [+] _[ε,]_
_Z_ _[I][X]_ [(] _[Z]_ [)]


14


_L_ ˆ = [1]

_N_


_N_


_i_ =1


- 1 - - [�]
_f_ ( _xi, yi_ ) log _q_ ( _yi|zi_ ) _−_ KL _p_ ( _z|xi_ ) _∥_ _r_ ( _z_ ) _._ (13)
_f_ ( _xi, yi_ ) + _ϵg_


B.2 SUFFICIENCY AND MINIMALITY PRINCIPALS


**Proof 1 (Sufficiency at high relevance)** _Rewrite the objective as_


                -                _L_ AdaIB = _f_ ( _X, Y_ ) _I_ ( _Z_ ; _Y_ ) _−_ _[g]_ [(] _[f]_ [(] _[X, Y]_ [ ))] _I_ ( _Z_ ; _X_ ) _._

_f_ ( _X, Y_ )


_Let t_ := _f_ ( _X, Y_ ) _and define ε_ ( _t_ ) := _[g]_ [(] _t_ _[t]_ [)] _[.]_ _[Since][ g][ is positive and non-increasing on]_ [ (0] _[,][ ∞]_ [)] _[, the limit]_

_c_ := lim _t→∞_ _g_ ( _t_ ) _∈_ [0 _, ∞_ ) _exists; hence_


_[t]_ [)]

_≤_ [max] _[{][c, g]_ [(1)] _[}]_
_t_ _t_


_ε_ ( _t_ ) = _[g]_ [(] _[t]_ [)]


_−→_ 0 _._
_t_


**Lemma B.1 (Monotonicity of the compression)** _Define the effective compression weight λ_ ( _f_ ) :=
_g_ ( _f_ )

_f_ _[.]_ _[If][ g]_ [: (0] _[,][ ∞]_ [)] _[ →]_ [(0] _[,][ ∞]_ [)] _[ is positive and non-increasing, then][ λ]_ [(] _[f]_ [)] _[ is non-increasing on]_ [ (0] _[,][ ∞]_ [)] _[.]_


**Proof 3 (Monotonicity of the compression)** _Take_ 0 _<_ _f_ 1 _<_ _f_ 2 _._ _Since_ _g_ _is_ _non-increasing_ _and_
_positive, g_ ( _f_ 1) _≥_ _g_ ( _f_ 2) _>_ 0 _._ _Then_


_hence λ_ ( _f_ 2) _≤_ _λ_ ( _f_ 1) _._


**Proof 4 (Adaptive sufficiency–minimality trade-off)** _Algebraically factor f_ _from Definition 1 to_
_obtain the displayed form._ _Monotonicity of λ follows from Lemma B.1._ _When f_ _grows, the bracketed_
_objective_ _reduces_ _the_ _penalty_ _coefficient_ _on_ _I_ ( _Z_ ; _X_ ) _while_ _scaling_ _I_ ( _Z_ ; _Y_ ) _by_ _a_ _larger_ _f_ _,_ _hence_
_shifting the balance toward sufficiency._ _The extremes follow from Theorems 1 and 2._


B.3 BOUNDED LEAKAGE


**Proposition 2 (No Gratuitous Leakage)** _Let_ _Zc_ := _{Z_ : _I_ ( _Z_ ; _Y_ ) = _c} be the set of representa-_
_tions achieving the same predictive information level c_ _≥_ 0 _._ _For any fixed relevance score f_ _>_ 0
_(and thus g_ ( _f_ ) _>_ 0 _by Definition 1), and for any Z_ 1 _, Z_ 2 _∈Zc_ _satisfying I_ ( _Z_ 1; _X_ ) _>_ _I_ ( _Z_ 2; _X_ ) _, it_
_holds that_
_L_ AdaIB( _Z_ 1) _< L_ AdaIB( _Z_ 2) _._


_Therefore, at a fixed sufficiency level, AdaIB always prefers the representation with smaller I_ ( _Z_ ; _X_ ) _._


**Proof 5 (No Gratuitous Leakage)** _For fixed f_ _>_ 0 _,_


_L_ AdaIB( _Z_ ) = _f_ _· I_ ( _Z_ ; _Y_ ) _−_ _g_ ( _f_ ) _· I_ ( _Z_ ; _X_ ) _._


_Given I_ ( _Z_ 1; _Y_ ) = _I_ ( _Z_ 2; _Y_ ) = _c,_


_L_ AdaIB( _Z_ 1) _−L_ AdaIB( _Z_ 2) = _−_ _g_ ( _f_ )     - _I_ ( _Z_ 1; _X_ ) _−_ _I_ ( _Z_ 2; _X_ )� _<_ 0 _,_


_since g_ ( _f_ ) _>_ 0 _and I_ ( _Z_ 1; _X_ ) _> I_ ( _Z_ 2; _X_ ) _._


**Corollary B.1 (Bounded leakage at high relevance)** _Let_ _Z_ _[⋆]_ _be_ _an_ _optimizer_ _of_ _L_ AdaIB _within_ _a_
_given_ _model_ _class,_ _and_ _write_ _c_ _[⋆]_ := _I_ ( _Z_ _[⋆]_ ; _Y_ ) _._ _Then_ _for_ _any_ _f_ _>_ 0 _there_ _is_ _no_ _Z_ _[′]_ _∈Zc⋆_ _with_
_I_ ( _Z_ _[′]_ ; _X_ ) _>_ _I_ ( _Z_ _[⋆]_ ; _X_ ) _._ _In particular, along any sequence f_ _→∞_ _(with g_ ( _f_ ) _>_ 0 _), AdaIB never_
_incentivizes increasing I_ ( _Z_ ; _X_ ) _without improving I_ ( _Z_ ; _Y_ ) _;_ _among equal-I_ ( _Z_ ; _Y_ ) _solutions, the_
_optimizer attains minimal I_ ( _Z_ ; _X_ ) _._


**Proof 6 (Bounded leakage at high relevance)** _By_ _Proposition_ _2,_ _for_ _any_ _fixed_ _f_ _>_ 0 _and_ _any_
_Z_ 1 _, Z_ 2 _∈Zc⋆,_ _if_ _I_ ( _Z_ 1; _X_ ) _>_ _I_ ( _Z_ 2; _X_ ) _then_ _L_ AdaIB( _Z_ 1) _<_ _L_ AdaIB( _Z_ 2) _because_ _g_ ( _f_ ) _>_ 0 _._
_Hence an optimizer Z_ _[⋆]_ _at that f_ _must minimize I_ ( _Z_ ; _X_ ) _within Zc⋆_ _._ _This argument holds for every_
_f_ _; therefore, it also holds along any sequence with f_ _→∞_ _(regardless of whether g_ ( _f_ ) _is bounded),_
_which proves the claim._


**Synthesis.** AdaIB provides a principled adaptive mechanism for balancing sufficiency and minimality:


    - _Prioritizes sufficiency_ when relevance is high (Theorem 1).


    - _Enforces minimality_ when relevance is low (Theorem 2).


    - _Protects against overfitting_ even at high relevance by preferring the smallest _I_ ( _Z_ ; _X_ ) among
equally sufficient solutions (Corollary B.1).


This adaptive behaviour, controlled by the relevance function _f_ ( _X, Y_ ), lets AdaIB adjust its learning
strategy to data quality, which is particularly useful in multimodal settings with heterogeneous and
noisy pairs.


15


_λ_ ( _f_ 2) - _g_ ( _f_ 2)

_[g]_ [(] _[f]_ [2][)] _[/f]_ [2] =
_λ_ ( _f_ 1) [=] _g_ ( _f_ 1) _/f_ 1 _g_ ( _f_ 1)


�� _f_ 1
_f_ 2


_≤_ 1 _·_ _[f]_ [1] _<_ 1 _,_

_f_ 2


B.4 EXISTENCE OF STATIONARY POINTS


**Theorem 4 (Existence of Stationary Points)** _Let_ _L_ [ˆ] ( _w_ ) _denote_ _the_ _empirical_ _AdaIB_ _objective_ _in_
_equation 12, with decoupled fθ, gϕ, and let w_ = ( _θ, ϕ, ψ_ ) _collect all trainable parameters._ _Assume:_


(A1) _**Positivity**_ _**and**_ _**boundedness.**_ _fθ_ ( _x, y_ ) _∈_ [ _εf_ _, Mf_ ] _and_ _gϕ_ ( _x, y_ ) _∈_ [ _εg, Mg_ ] _for_ _all_ ( _x, y_ ) _,_
_with εf_ _, εg_ _>_ 0 _._


(A2) _**Smoothness on compact domain.**_ _L_ [ˆ] _is continuously differentiable on a nonempty compact_
_convex set_ Ω _⊂_ R _[d]_ _, with ∇L_ [ˆ] _bounded on_ Ω _._


(A3) _**Projected gradient dynamics.**_ _Training uses T_ ( _w_ ) = ΠΩ( _w −_ _η∇L_ [ˆ] ( _w_ )) _for some η_ _>_ 0 _,_
_where_ ΠΩ _is Euclidean projection onto_ Ω _._


_Then_ _T_ : Ω _→_ Ω _is_ _continuous_ _and_ _admits_ _a_ _fixed_ _point_ _w_ _[∗]_ _∈_ Ω _by_ _Brouwer’s_ _fixed-point_ _theo-_
_rem (Brouwer, 1911)._ _Consequently, w_ _[∗]_ _satisfies the stationarity condition_ 0 _∈∇L_ [ˆ] ( _w_ _[∗]_ ) + _N_ Ω( _w_ _[∗]_ ) _,_
_where N_ Ω _is the normal cone._ _If w_ _[∗]_ _lies in the relative interior of_ Ω _, then ∇L_ [ˆ] ( _w_ _[∗]_ ) = 0 _._


**Proof 7 (Existence of Stationary Points)** _We verify the conditions for Brouwer’s fixed-point theo-_
_rem, which states that every continuous map from a nonempty, compact, convex set to itself has a_
_fixed point._


_By_ _assumption_ _(A3),_ _the_ _training_ _algorithm_ _uses_ _the_ _projected_ _gradient_ _map_ _T_ ( _w_ ) = ΠΩ( _w_ _−_
_η∇L_ [ˆ] ( _w_ )) _._ _Since_ ΠΩ _projects onto_ Ω _, we have T_ ( _w_ ) _∈_ Ω _for any w_ _∈_ Ω _, establishing that T_ _maps_
Ω _to itself._


_The_ _continuity_ _of_ _T_ _follows_ _from_ _the_ _continuous_ _differentiability_ _of_ _L_ [ˆ] _(by_ _A2)_ _and_ _the_ _fact_ _that_
_the projection operator_ ΠΩ _is nonexpansive (hence continuous) on the convex set_ Ω _._ _Thus, T_ _is a_
_composition of continuous maps._


_Since_ Ω _is_ _nonempty,_ _compact,_ _and_ _convex,_ _and_ _T_ : Ω _→_ Ω _is_ _continuous,_ _Brouwer’s_ _theorem_
_guarantees the existence of a fixed point w_ _[∗]_ _∈_ Ω _satisfying T_ ( _w_ _[∗]_ ) = _w_ _[∗]_ _._


_The_ _fixed_ _point_ _condition_ _w_ _[∗]_ = ΠΩ( _w_ _[∗]_ _−_ _η∇L_ [ˆ] ( _w_ _[∗]_ )) _implies,_ _by_ _the_ _projection_ _theorem,_ _that_
_−η∇L_ [ˆ] ( _w_ _[∗]_ ) _∈_ _N_ Ω( _w_ _[∗]_ ) _,_ _where_ _N_ Ω( _w_ _[∗]_ ) _is_ _the_ _normal_ _cone_ _to_ Ω _at_ _w_ _[∗]_ _._ _This_ _is_ _equivalent_ _to_
_the_ _stationarity_ _condition_ 0 _∈∇L_ [ˆ] ( _w_ _[∗]_ ) + _N_ Ω( _w_ _[∗]_ ) _._ _If_ _w_ _[∗]_ _lies_ _in_ _the_ _relative_ _interior_ _of_ Ω _,_ _then_
_N_ Ω( _w_ _[∗]_ ) = _{_ 0 _} and thus ∇L_ [ˆ] ( _w_ _[∗]_ ) = 0 _._


B.5 DECOUPLE PROPERTIES


Despite decoupling, the following properties still hold:


(i) **Pointwise reparameterization** (Prop. 1): with _λ_ ( _x, y_ ) := _gϕ_ ( _x, y_ ) _/fθ_ ( _x, y_ ) _>_ 0, _L_ AdaIB =
_fθ_ ( _x, y_ ) [ _I_ ( _Z_ ; _Y_ ) _−_ _λ_ ( _x, y_ ) _I_ ( _Z_ ; _X_ ) ].


(ii) **Sample-wise reweighting** : letting _fi_ = _fθ_ ( _xi, yi_ ), _gi_ = _gϕ_ ( _xi, yi_ ), _wi_ [(] _[y]_ [)] = _fi/_ [�] _j_ _[f][j]_ [,] _[ w]_ _i_ [(] _[x]_ [)] =

_gi/_ [�] _j_ _[g][j]_ [, and] _[β]_ [¯][eff] [=][ �] _i_ _[g][i][/]_ [ �] _i_ _[f][i]_ [, we obtain the same weighted IB form.]


(iii) **No** **gratuitous** **leakage** (Prop. 2): for fixed ( _x, y_ ), if _I_ ( _Z_ 1; _Y_ ) = _I_ ( _Z_ 2; _Y_ ) and _I_ ( _Z_ 1; _X_ ) _>_
_I_ ( _Z_ 2; _X_ ), then _L_ AdaIB( _Z_ 1) _< L_ AdaIB( _Z_ 2). This uses only _gϕ_ ( _x, y_ ) _>_ 0 and does not require any
functional dependence between _g_ and _f_ .


B.6 DECOUPLE SUFFICIENCY AND MINIMALITY


**Sufficiency** **and** **Minimality** **Balance** Recall the decoupled objective _L_ AdaIB =
_fθ_ ( _x, y_ ) [ _I_ ( _Z_ ; _Y_ ) _−_ _λ_ ( _x, y_ ) _I_ ( _Z_ ; _X_ ) ] with _λ_ ( _x, y_ ) := _gϕ_ ( _x, y_ ) _/fθ_ ( _x, y_ ) _>_ 0. We do _not_
assume any global monotonic relation between _gϕ_ and _fθ_ . The extreme-regime behaviours and their
quantitative, per-sample approximations remain valid under mild ratio conditions.


16


(ii) **Sample-wise reweighting** : letting _fi_ = _fθ_ ( _xi, yi_ ), _gi_ = _gϕ_ ( _xi, yi_ ), _wi_ [(] _[y]_ [)] = _fi/_ [�]


_j_ _[g][j]_ [, and] _[β]_ [¯][eff] [=][ �]


_i_ _[g][i][/]_ [ �]


_i_ _[f][i]_ [, we obtain the same weighted IB form.]


Even without a global monotone constraint on _g_, the two extreme behaviours are retained under local
ratio conditions. Let _KX_ _, KY_ _< ∞_ be uniform bounds on _I_ ( _Z_ ; _X_ ) and _I_ ( _Z_ ; _Y_ ) within the model
class, and define _λ_ ( _x, y_ ) := _gϕ_ ( _x, y_ ) _/fθ_ ( _x, y_ ) _>_ 0.


**Theorem 5 (** _ε_ **-sufficiency)** _Let λ_ ( _x, y_ ) := _gϕ_ ( _x, y_ ) _/fθ_ ( _x, y_ ) _>_ 0 _and KX_ _< ∞_ _bound I_ ( _Z_ ; _X_ ) _on_
_the model class._ _If a sample_ ( _x, y_ ) _satisfies λ_ ( _x, y_ ) _≤_ _ε for some ε >_ 0 _, then_

_L_ AdaIB _−_ _fθ_ ( _x, y_ ) _I_ ( _Z_ ; _Y_ ) _≤_ _ε fθ_ ( _x, y_ ) _KX_ _._
��� ���


_Hence, the objective behaves as fθ I_ ( _Z_ ; _Y_ ) _up to O_ ( _ε_ ) _; sufficiency dominates._


**Theorem 6 (** _η_ **-minimality)** _Let_ _KY_ _<_ _∞_ _bound_ _I_ ( _Z_ ; _Y_ ) _on_ _the_ _model_ _class._ _If_ _a_ _sample_ ( _x, y_ )
_satisfies_ _g_ _[f]_ _ϕ_ _[θ]_ [(] ( _[x,y]_ _x,y_ [)] ) _[≤]_ _[η][ for some][ η]_ _[>]_ [ 0] _[ (equivalently][ λ]_ [(] _[x, y]_ [)] _[ ≥]_ [1] _[/η][), then]_


_L_ AdaIB + _gϕ_ ( _x, y_ ) _I_ ( _Z_ ; _X_ ) _≤_ _η gϕ_ ( _x, y_ ) _KY ._
��� ���


_Hence maximizing L_ AdaIB _is approximately equivalent to minimizing I_ ( _Z_ ; _X_ ) _up to O_ ( _η_ ) _; minimality_
_dominates._


On a minibatch, let Πhi := _{i_ : _λ_ ( _xi, yi_ ) _≤_ _ε}_ and Πlo := _{i_ : _fθ_ ( _xi, yi_ ) _/gϕ_ ( _xi, yi_ ) _≤_ _η}_ . Then _L_ [ˆ]
can be viewed as the sum of a sufficiency-dominant average over Πhi (error _O_ ( _ε_ )) and a minimalitydominant average over Πlo (error _O_ ( _η_ )). Thus, the sufficiency–minimality balance is preserved in a
quantitative, per-sample manner.


Here is the operational behaviour under Definition 2.


    - **High relevance.** Along sequences where _fθ_ ( _x, y_ ) _→∞_ and _λ_ ( _x, y_ ) _→_ 0 (e.g. _gϕ_ is bounded
or grows sublinearly relative to _fθ_ ), we have _L_ AdaIB _∼_ _fθ_ ( _x, y_ ) _I_ ( _Z_ ; _Y_ ); compression
vanishes and sufficiency dominates.

    - **Low** **relevance.** Along sequences where _fθ_ ( _x, y_ ) _→_ 0 [+] and _λ_ ( _x, y_ ) _→∞_ (equivalently
_fθ/gϕ →_ 0), we have _L_ AdaIB = _−_ _gϕ_ ( _x, y_ ) _I_ ( _Z_ ; _X_ )+ _o_    - _gϕ_ ( _x, y_ )�; maximizing the objective
asymptotically minimizes _I_ ( _Z_ ; _X_ ); compression dominates.


C FURTHER ANALYSIS ON MODEL PERFORMANCE UNDER MISALIGNMENT


tively, and created a bal
|Col1|Col2|Col3|Col4|Col5|Col6|In|correct|
|---|---|---|---|---|---|---|---|
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||
|||||||||


|Col1|Col2|Col3|Col4|Col5|Correct<br>Incorrect|
|---|---|---|---|---|---|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


|Col1|Col2|Col3|Col4|Col5|I|ncorrect|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


|Col1|Col2|Col3|Col4|Col5|I|Correct<br>ncorrect|
|---|---|---|---|---|---|---|
||||||||
||||||||
||||||||
||||||||
||||||||
||||||||


|Col1|Col2|Col3|Col4|Col5|Correct|
|---|---|---|---|---|---|
||||||~~Incorrect~~|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||


|Col1|NIB|- Fitting|Loss|Col5|Correct|
|---|---|---|---|---|---|
||||||Correct<br>~~Incorrect~~|
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||
|||||||

anced set of correct (orig- Figure 3: Distributions of compression and fitting losses under matched
inal) and incorrect (ran- pairs vs. mismatched pairs. AdaIB enables separation, while M2IB
domly swapped captions) and NIB show significant overlap.
pairs. We then evaluated
whether the models could
distinguish between these two categories based on their final loss values, using the Area Under
the Curve (AUC) as a ranking metric. The results presented in Fig. 3, demonstrate the distributions of


To provide a more comprehensive analysis of our
model’s robustness against
image-text misalignment,
we conducted two supplementary experiments.


16


14


12


10


8


6


4


2


0


M2IB - Compression Loss


NIB - Compression Loss


AdaIB - Compression Loss


0.20 0.25 0.30 0.35 0.40 0.45 0.50 0.55 0.60
Loss Value


16


14


12


10


8


6


4


2


0


0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
Loss Value


18

16

14

12

10

8

6

4

2

0


0.1 0.2 0.3 0.4 0.5 0.6 0.7
Loss Value


M2IB - Fitting Loss


14


12


10


8


6


4


2


0


AdaIB - Fitting Loss


0.2 0.3 0.4 0.5 0.6 0.7
Loss Value


14


12


10


8


6


4


2


0


0.1 0.2 0.3 0.4 0.5 0.6 0.7
Loss Value


16


14


12


10


8


6


4


2


0


0.2 0.3 0.4 0.5 0.6 0.7
Loss Value


Figure 3: Distributions of compression and fitting losses under matched
pairs vs. mismatched pairs. AdaIB enables separation, while M2IB
and NIB show significant overlap.


17


Table 5: Performance comparison of M2IB, NIB, and Ours across noisy _⋆_, borderline _⋆_, and clean _⋆_
CC3M, Flickr8k, and Laion400m datasets. The asterisk ( _⋆_ ) indicates that the datasets have been
artificially partitioned into three distinct groups—noisy, borderline, and clean—based on the degree
of image-caption alignment. This categorisation was achieved by manually partitioning the data
according to the image-text similarity scores obtained from the CLIP model.


**Dataset** **Metric** **M2IB** **NIB** **Ours**


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


compression and fitting losses. Our model exhibits a clear separation between the loss distributions
for matched and mismatched pairs. In stark contrast, the distributions for M2IB and NIB show
significant overlap, suggesting their struggle to differentiate between well-matched and mismatched
inputs.


Then, we analysed performance on data with varying degrees of naturally occurring misalignment.
Leveraging CLIP similarity scores as a proxy for alignment quality, we partitioned the CC3M,
Flickr8k, and Laion400m datasets into three distinct subsets: ’Noisy’ (low similarity), ’Borderline’
(medium similarity), and ’Clean’ (high similarity). The detailed performance breakdown, presented
in Table 5, shows that our model consistently outperforms the M2IB and NIB baselines across these
partitions. Notably, our method’s superiority is evident even on the most challenging “noisy" subset
(in this case, the completely misaligned ones), highlighting its resilience to the imperfections inherent
in web-crawled data.


D ANALYSIS OF THE ADAPTIVE MECHANISM’S BEHAVIOR


A key finding, however, is

|Col1|Col2|Col3|Col4|Col5|Col6|Ca|tegory|Col9|
|---|---|---|---|---|---|---|---|---|
||||||Ma<br>Mis|Ma<br>Mis|tched<br>matched||
||||||Ma<br>Mis|Ma<br>Mis|||
||||||||||
||||||||||
||||||||||
||||||||||
||||||||||


Value of f

the corresponding _g_ values
for both matched and mismatched pairs. The _g_ value Figure 4: The visualisation of distribution of _f_ and _g_ values from
corresponding to the same the Laion400m dataset. Matched pairs represent the original, aligned
_f_ value is not fixed. This image-caption pairs, while Mismatched pairs are misaligned imagesuggests that the **AdaIB** caption pairs created by randomly shuffling the original image-caption
**optimisation process does** pairings.
**not** **rely** **solely** **on** **the** **ini-**
**tial** _f_ **value.** **Instead,** **the**
**framework appears to learn more nuanced, context-dependent characteristics of the data** . For
instance, a high _f_ value might still lead to high compression (high _g_ ) if the image contains irrelevant
background elements, while a low _f_ value could result in low compression if the model cannot find a
meaningful signal and defaults to a near-uniform distribution. This behaviour demonstrates that the
optimal compression weight _g_ is a function of deeper data properties beyond initial similarity metrics,
highlighting the adaptive and non-linear nature of the AdaIB framework.


The Fig. 4 visualises the
distribution of matched and
mismatched image-caption
pairs and their corresponding _f_ and _g_ values. The
_f_ value, representing the
L2 distance, demonstrates a
clear discriminative capability. Matched image-caption
pairs are predominantly associated with higher _f_ values, indicating that this metric effectively captures the
overall alignment between
the two modalities. In contrast, mismatched pairs are
more concentrated in the
lower range of _f_ values.


1.0


0.8


0.6


0.4


0.2


0.0


Value of f


Figure 4: The visualisation of distribution of _f_ and _g_ values from
the Laion400m dataset. Matched pairs represent the original, aligned
image-caption pairs, while Mismatched pairs are misaligned imagecaption pairs created by randomly shuffling the original image-caption
pairings.


19


**1026**


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


image cosine L2 L1 chebyshev


Figure 5: A qualitative comparison of attribution maps generated using various measurements for the
function _f_ ( _X, Y_ ). The results for L1, L2, Chebyshev, and Cosine functions illustrate that the metric
selection critically influences the model’s visual attribution, causing it to attend to different spatial
regions for the same textual concept.


E MORE ABLATION STUDY


E.1 THE ABLATION STUDY ON _f_ ( _X, Y_ )


To investigate the impact of the similarity function _f_ ( _X, Y_ ) on the model’s performance and its visual
attribution, an ablation study was conducted. The study systematically evaluated four different metric
choices for f: L1 distance, L2 distance, Chebyshev distance, and Cosine similarity. The quantitative
results are presented in Table 6.


Table 6: Ablation study on the choice of function _F_ and different metrics across datasets and ROAR
protocols.


(a) Performance on different datasets.


(b) Performance on different ROAR protocols.


|Metric|CC3M Flickr8k Laion400m|
|---|---|
|_F_=L1<br>vdrop_ ↓_<br>vincr_ ↑_<br>tdrop_ ↓_<br>tincr_ ↑_|**0.75**_±_**0.10**<br>0.78_±_0.04<br>0.92_±_0.03<br>22.50_±_3.54<br>22.50_±_4.95<br>31.66_±_0.49<br>1.67_±_0.26<br>2.44_±_0.47<br>1.89_±_0.07<br>31.50_±_3.54<br>30.00_±_12.73<br>27.14_±_1.61|
|_F_=L2<br>vdrop_ ↓_<br>vincr_ ↑_<br>tdrop_ ↓_<br>tincr_ ↑_|0.79_±_0.15<br>**0.56**_±_**0.04**<br>**0.64**_±_**0.06**<br>**53.00**_±_**4.24**<br>60.00_±_2.83<br>**55.76**_±_**6.00**<br>**1.20**_±_**0.08**<br>**1.61**_±_**0.10**<br>1.29_±_0.22<br>**36.00**_±_**2.83**<br>**37.50**_±_**0.71**<br>**28.63**_±_**3.35**|
|_F_=chebyshev<br>vdrop_ ↓_<br>vincr_ ↑_<br>tdrop_ ↓_<br>tincr_ ↑_|0.85_±_0.14<br>**0.64**_±_**0.09**<br>0.78_±_0.29<br>47.00_±_1.41<br>54.00_±_4.24<br>53.74_±_7.44<br>1.35_±_0.05<br>1.76_±_0.30<br>1.27_±_0.20<br>30.00_±_1.41<br>31.00_±_0.00<br>25.64_±_3.74|
|_F_=cosine<br>vdrop_ ↓_<br>vincr_ ↑_<br>tdrop_ ↓_<br>tincr_ ↑_|2.21_±_0.12<br>1.15_±_0.01<br>2.96_±_0.08<br>42.50_±_3.54<br>**61.00**_±_**1.41**<br>45.73_±_0.39<br>1.25_±_0.43<br>1.69_±_0.01<br>**1.25**_±_**0.13**<br>29.50_±_2.12<br>33.50_±_2.12<br>23.12_±_1.59|


|Metric|CC3M Flickr8k Laion400m|
|---|---|
|_F_=L1<br>i2t-oc_ ↑_<br>t2i-oc_ ↑_<br>i2t-co_ ↑_<br>t2i-co_ ↑_|45.53_±_1.95<br>46.00_±_1.59<br>44.06_±_3.89<br>46.45_±_0.42<br>48.75_±_2.60<br>44.85_±_6.55<br>31.18_±_9.17<br>17.60_±_0.50<br>31.14_±_5.61<br>27.78_±_8.47<br>19.23_±_2.75<br>22.45_±_1.00|
|_F_=L2<br>i2t-oc_ ↑_<br>t2i-oc_ ↑_<br>i2t-co_ ↑_<br>t2i-co_ ↑_|44.82_±_6.45<br>**82.40**_±_**0.50**<br>**60.74**_±_**1.36**<br>47.72_±_5.04<br>**85.56**_±_**1.21**<br>**59.35**_±_**0.92**<br>**57.55**_±_**2.41**<br>38.40_±_5.53<br>52.64_±_6.80<br>**46.75**_±_**6.05**<br>**42.90**_±_**3.72**<br>**41.13**_±_**4.36**|
|_F_=chebyshev<br>i2t-oc_ ↑_<br>t2i-oc_ ↑_<br>i2t-co_ ↑_<br>t2i-co_ ↑_|41.15_±_0.58<br>67.75_±_3.34<br>52.69_±_0.80<br>43.22_±_0.52<br>70.48_±_0.15<br>52.42_±_1.19<br>58.72_±_7.74<br>**40.36**_±_**6.31**<br>**53.19**_±_**4.51**<br>45.77_±_7.79<br>40.93_±_2.01<br>37.90_±_5.95|
|_F_=cosine<br>i2t-oc_ ↑_<br>t2i-oc_ ↑_<br>i2t-co_ ↑_<br>t2i-co_ ↑_|**53.72**_±_**4.34**<br>73.96_±_0.56<br>49.43_±_3.81<br>**55.47**_±_**3.14**<br>76.56_±_3.46<br>46.49_±_2.73<br>52.52_±_0.80<br>36.21_±_4.06<br>45.14_±_2.35<br>44.50_±_4.16<br>37.31_±_2.14<br>34.23_±_0.78|


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


Fig. 5 provides a qualitative comparison of attribution maps generated using the different similarity
functions. It is evident that the selection of _f_ ( _X, Y_ ) influences the model’s visual attention, guiding
it to focus on different spatial regions of the image that correspond to the textual query.


For the first example in Fig. 5, the L2 and Cosine similarity functions generate attribution maps that
are more precisely focused on the vehicle itself, aligning well with the key object in the text. In
contrast, the L1 and Chebyshev metrics produce more diffuse heatmaps, with the Chebyshev metric
highlighting the vehicle’s edges and surroundings more broadly. For the third instance in Fig 5, the
L2 and Cosine metrics again demonstrate a stronger ability to localise the subject (“deer"), whereas
the L1 and Chebyshev metrics result in more scattered attention, with Chebyshev’s attribution being
particularly dispersed across the background and foreground.


Overall, the qualitative results suggest that the L2 and Cosine similarity functions tend to produce
more semantically focused and cleaner attribution maps, which more accurately highlight the most
relevant objects described in the text. The quantitative results from the Table 6 further confirm this
trend, with the L2-based model demonstrating superior performance under various datasets. We
attribute that L2 distance takes into account the “strength" information of the feature (vector modulus),
it can make more accurate judgments than cosine similarity, which only looks at the “direction" of
the feature, thus achieving a consistent advantage in multimodal attribution.


E.2 THE ABLATION STUDY ON _g_ ( _f_ ( _X, Y_ ))


A comprehensive analysis of the updated ablation study, presented in Table 7, was conducted to
evaluate six different model architectures by varying their depth, width, and activation functions.


The results unequivocally identify Model 2, a shallow architecture with a single hidden layer (1 →
32 → 1) and a ReLU activation function, as the superior configuration. This model demonstrated
the most robust performance, achieving the highest scores on four of the eight metrics (VIncr, TIncr,
ROAR-t2t-oc, and ROAR-t2i-oc) and was the joint-best performer on a fifth metric (VDrop).


A detailed breakdown of the findings is as follows:


Table 7: Comprehensive results of the ablation study on model architectures, split into two parts. (a)
details the model configurations. (b) shows all corresponding performance metrics.


(a) Model architectures and configurations.


**ID** **Architecture** **Activation** **Purpose / Description**


1 1 _→_ 1 None Depth Ablation (Linear Model)


2 1 _→_ 32 _→_ 1 ReLU Depth Ablation (Shallower)


3 1 _→_ 8 _→_ 8 _→_ 1 ReLU Width Ablation (Narrower)


4 1 _→_ 64 _→_ 64 _→_ 1 ReLU Width Variation (Wider)


5 1 _→_ 32 _→_ 32 _→_ 1 None Activation Ablation (Removed)


6 1 _→_ 32 _→_ 32 _→_ 1 Tanh Activation Ablation (Replaced)


(b) Performance metrics for each model configuration on Flickr8k dataset. For Drop metrics, lower is better. For
all other metrics, higher is better. Best results are highlighted in **bold** .


**ID** **VDrop** **VIncr** **TDrop** **TIncr** **ROAR-i2t-oc** **ROAR-t2i-oc** **ROAR-i2t-co** **ROAR-t2i-co**


1 0 _._ 64 _±_ 0 _._ 09 54 _._ 00 _±_ 4 _._ 24 **1.76** _±_ **0.30** 31 _._ 00 _±_ 0 _._ 00 67 _._ 75 _±_ 3 _._ 34 70 _._ 48 _±_ 0 _._ 15 40 _._ 36 _±_ 6 _._ 31 40 _._ 93 _±_ 2 _._ 01


2 **0.56** _±_ **0.09** **57.00** _±_ **4.24** 1 _._ 81 _±_ 0 _._ 03 **35.50** _±_ **0.71** **73.12** _±_ **5.50** **76.45** _±_ **3.36** 37 _._ 62 _±_ 1 _._ 19 40 _._ 43 _±_ 3 _._ 93


3 0 _._ 58 _±_ 0 _._ 04 53 _._ 00 _±_ 4 _._ 24 1 _._ 96 _±_ 0 _._ 13 28 _._ 50 _±_ 0 _._ 71 64 _._ 74 _±_ 2 _._ 72 65 _._ 72 _±_ 3 _._ 09 39 _._ 41 _±_ 0 _._ 47 38 _._ 02 _±_ 3 _._ 85


4 **0.56** _±_ **0.06** 55 _._ 00 _±_ 1 _._ 41 1 _._ 90 _±_ 0 _._ 06 30 _._ 50 _±_ 0 _._ 71 67 _._ 24 _±_ 0 _._ 81 70 _._ 48 _±_ 0 _._ 15 38 _._ 64 _±_ 3 _._ 88 **44.60** _±_ **1.52**


5 0 _._ 58 _±_ 0 _._ 04 54 _._ 50 _±_ 2 _._ 12 **1.76** _±_ **0.09** 33 _._ 50 _±_ 0 _._ 71 67 _._ 62 _±_ 6 _._ 78 71 _._ 63 _±_ 3 _._ 52 40 _._ 36 _±_ 6 _._ 31 39 _._ 87 _±_ 6 _._ 47


6 0 _._ 57 _±_ 0 _._ 03 55 _._ 00 _±_ 1 _._ 41 1 _._ 85 _±_ 0 _._ 03 30 _._ 50 _±_ 0 _._ 71 67 _._ 75 _±_ 3 _._ 34 69 _._ 86 _±_ 1 _._ 03 38 _._ 77 _±_ 0 _._ 44 39 _._ 78 _±_ 1 _._ 36


**Impact of Depth and Non-linearity** : The study highlights the distinct advantage of a shallow,
non-linear architecture. Both the simple linear model (ID 1) and the deeper model without an


21


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


activation function (ID 5) performed poorly on most metrics, confirming that non-linearity is essential
for the task. Crucially, the shallow Model 2 significantly outperformed all deeper architectures (ID 3,
4, and 6), suggesting that increasing network depth beyond a single hidden layer is counterproductive.


**Impact of Network Width** : Network width was also a significant factor. The wider architecture
(ID 4, with 64 neurons) was a strong contender, achieving the top result for the ROAR-t2i-co metric
and a joint-best score for VDrop. However, its overall performance did not surpass that of the more
streamlined, shallower Model 2. In contrast, the narrower model (ID 3, with 8 neurons) proved to be
ineffective, showing mediocre performance across all metrics.


**Impact** **of** **Activation** **Function** : The choice of activation function was a key determinant of
performance. As noted, the absence of non-linearity (ID 5) was detrimental. In comparing non-linear
functions, the ReLU activation used in the top-performing Model 2 proved to be far more effective
than the Tanh activation used in Model 6, which yielded unremarkable results.


The ablation study demonstrates that a shallow network with a single hidden layer of 32 neurons,
activated by the ReLU function, provides the optimal architecture among the configurations tested.
This structure strikes the most effective balance, outperforming models that are deeper, wider, or that
utilise an alternative (Tanh) or no activation function.


F VISUALISATION OF ROAR


G MORE VISUALISATION OF ATTRIBUTION MAP


In this section, we provide the attribution maps generated by different methods, providing a visual
comparison of their ability to localise salient objects and actions within an image. Each row focuses
on a distinct scene. In comparison to established methods, our proposed AdalB consistently yields
sharper and more precise attribution maps. The visualisations demonstrate AdalB’s superior capacity
to focus on salient regions—such as the soccer player and ball, the dog’s limbs, and the duck’s
body—while in Fig. 7, effectively suppressing irrelevant background noise. This enhanced precision
contrasts with the more diffuse or scattered activations often observed in the outputs of other models.


22


We adapt the Remove
and Retrain (ROAR)

image attribution map attribution map corrupted image

benchmark (Hooker et al.,
2019) to create a more
computationally efficient
evaluation. Instead of the
costly process of retraining
the model from scratch,
we leverage the powerful

CLIP-based architecture.
The methodology follows

Figure 6: The visualisation process of ROAR.

the spirit of ROAR: we first
identify and remove the
most salient features to create a "corrupted" dataset. We then quantify the importance of these
removed features by measuring the degradation in zero-shot image-text retrieval performance. The
score is calculated by _[ACC]_ _Acc_ _[o][−][ACC]_ _o_ _[c]_ . In our setting, however, _Acco_ represents the baseline zero-shot

retrieval accuracy on the original data, whilst _Accc_ is the accuracy on the corrupted data. An effective
attribution method should identify features critical to the model’s performance; their removal will
therefore cause a substantial drop in _Accc_, yielding a score close to 1. A higher score is better.
Following the protocol of M2IB (Wang et al., 2023), we corrupt the inputs based on attribution score
percentiles. For images, we identify pixels with scores above the 75th percentile and replace their
values with the image’s mean channel values. For text, given its sparse nature, tokens with scores
exceeding the 90th percentile are replaced with the designated CLIP padding token (ID 49407). An
example can be found in Fig. 6.


image attribution map attribution map corrupted image


Figure 6: The visualisation process of ROAR.


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


Image GradCAM M2IB NIB AdaIB (Ours)


Figure 7: The visualisation of example attribution maps for image and text inputs.


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


Image GradCAM M2IB NIB AdaIB (Ours)


Figure 8: The visualisation of more example attribution maps for image and text inputs.


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


Image GradCAM M2IB NIB AdaIB (Ours)


Figure 9: The visualisation of more example attribution maps for image and text inputs.


25