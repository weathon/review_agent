# Transfer Learning in Infinite Width Feature Learning Networks

- Decision: Accept (Poster)
- Scores: 4, 4, 8, 8

## Abstract
We develop a theory of transfer learning in infinitely wide neural networks under gradient flow that quantifies when pretraining on a source task improves generalization on a target task. We analyze both (i) fine-tuning, when the downstream predictor is trained on top of source-induced features and (ii) a jointly rich setting, where both pretraining and downstream tasks can operate in a feature learning regime, but the downstream model is initialized with the features obtained after pre-training. In this setup, the summary statistics of randomly initialized networks after a rich pre-training are adaptive kernels which depend on both source data and labels. For (i), we analyze the performance of a readout for different pretraining data regimes. For (ii), the summary statistics after learning the target task are still adaptive kernels with features from both source and target tasks. We test our theory on linear and polynomial regression tasks as well as real datasets. Our theory allows interpretable conclusions on performance, which depend on the amount of data on both tasks, the alignment between tasks, and the feature learning strength.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops an analytical theory for transfer learning (TL) in infinitely wide neural networks trained under gradient flow. The authors extend the work on infinite-width feature learning (FL) to the TL setting, focusing on two main paradigms: (i) fine-tuning on top of source features, and (ii) a jointly rich setting where both pretraining and downstream tasks operate in the FL regime. The main contribution is quantifying the generalization benefit of pretraining as a function of the data and model "richness" parameters ($\gamma$) for both source and target tasks. The model is well-motivated by explaining empirical observations, which I believe is the right approach.

### Strengths
The main strength is the successful analytical quantification of the transfer learning process in the Feature Learning (FL) regime of infinite-width networks. Unlike the Neural Tangent Kernel (NTK) regime, which predicts no feature learning, this work captures how the source features influence the target task generalization. The results provide explicit, closed-form expressions for metrics like generalization error and the overlap with the label direction. The results are supported by numerical simulations. This is a significant theoretical advance over existing NTK-based TL analyses.
The distinction between fine-tuning and the "jointly rich" setting is interesting. The theory provides concrete and testable conditions, such as the relationship between source and target richness, for when pretraining is beneficial, which aligns well with the experimental results also tested on more realistic data.

### Weaknesses
1.	The author assumes the reader is fully familiar with their previous work and neglects to define many key terms in the main text. This makes the paper not accessible to many readers. In addition, common abbreviations and critical analysis components are not defined: DMFT (Line 275), MLP, NTK (Line 051), and the distinction between the limiting objects and their finite counterparts is often blurred. In addition, some of the quantities are not always defined, which makes it hard to read. 
2.	Although the theoretical analysis lacks rigor, I believe the method and analysis are robust in certain settings. The underlying model is highly constrained but captures the phenomenon nicely. I believe the analysis relies heavily on strong assumptions about input data isotropy (e.g., isotropic Gaussian inputs), which significantly simplifies real-world datasets. This reliance on simplified, isotropic features is somewhat hidden and must be explicitly stated and discussed as a key limitation, as it fundamentally affects the general applicability of the quantitative results. I believe that the structure of the features can significantly change the results.

The paper is technically sound and achieves a novel, closed-form result for transfer learning in the feature learning regime, which is a significant theoretical contribution, together with a detailed comparison to experiments. However, the highly inaccessible presentation, relying heavily on the reader's deep knowledge of the authors' previous work, is a major drawback. That being said, I’m willing to raise my score, provided the authors improve the presentation by addressing the numerous undefined terms, confusing notation, and clarify statements as I detailed in the questions.

### Questions
1.	DMFT, MLP, NTK: Please define these acronyms upon first use (e.g., Line 051, 275).
2.	Result 1 ($\langle \cdot \rangle$): What measure is average $\langle \cdot \rangle$ taken over? Does it include averages over $\phi \sim N(0, 1)$ and $\chi(x) \sim GP(0,K_x)$?
3.	Result 2 ($\chi^\ell$): Please define $\chi^\ell$ and provide a clear interpretation. Also, justify why the baseline for $\mathcal{L}_2$ is $1-\nu_2$.
4.	Error Definitions: Please define $\mathcal{L}(t)$ (Line 217) and $\hat{\mathcal{L}}$ (Eq. 10) explicitly. I assume $\mathcal{L}(t)$ is the loss at time $t$. Why does $\mathcal{L}(t)$ in line 217 have a $K^{1/2}$ on only one term? Also, is $\Delta(t)$ (Line 242) the train error residuals and not the train error? Finally, it could be helpful for readability to use a distinct letter for the limiting object (e.g. $\mathcal{L}(t)$) to avoid confusion.
5.	Appendix A: What is $\Delta(x,t)$? 
6.	Appendix B.1: Could you explain Eq. 27 also what is $\delta(t)$? Should it be $P_2$?
7.	Line 1068-1069: Is it clear that the response function is translation-invariant? Also what happens at the transition point? 
8.	Similar to 6, could you explain Eq. (98)? I’m probably missing something, but shouldn’t there be an additional $XX^\top$ based on the definition of the Kernel? 
9.  Assumption of Input Isotropy: Given the use of Statistical Physics methods, please clearly state the assumptions made about the input data distribution, particularly regarding isotropy (e.g., Gaussian inputs), and discuss how the results might qualitatively change if these assumptions were violated (e.g., for real-world, highly anisotropic datasets).
10. Generality of $\gamma$ Dependence: The main takeaway is that pretraining improves generalization if $\gamma_1 \ge \gamma_2$. Could the authors comment on whether this relationship is expected to hold qualitatively for deep, non-linear networks? For instance, in a deep non-linear network, what surrogate quantity would replace the analytical $\gamma$ to describe the "richness" of the source task/model?
11. Unbounded Feature Learning: Does the result imply that unbounded feature learning ($\gamma_2 \to \infty$) in the target task ($T_2$) does not matter for the final transfer benefit, or is this simply a case not fully explored?
12. Figure 1(c) and $\gamma_0$: What is the parameter $\gamma_0$ referenced in Figure 1(c) and Line 360?
13. In Section 3.1 (Line 377), it states that large $\gamma_2$ can be harmful. Should this be $\gamma_1$? Please clarify the dependence.
14. Optimality of $t_1, t_2$: Figure 3 is compelling. How do the results change if one varies the times $t_1, t_2$? Is there a theoretical way to determine the optimal time for transition to the new task? How do these times depend on the feature learning scales, $\gamma_1, \gamma_2$, and the overparameterization ratios, $\nu_1, \nu_2$?
15: Harmful Large FL Size: In section 3.3. You state in Fig. 4(a) that there is an improvement in the test loss for any value of $\gamma_2$, also in lines 444-445. However, it seems from the figure that for a large $\gamma_2$, it can be harmful. Could you clarify?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper develops a first-principles theory of transfer learning for infinitely wide neural networks operating in the feature-learning (mean-field/µP) regime. The analysis covers two scenarios: fine-tuning with frozen features and a "jointly rich" setting with feature adaptation on the target task. The core of the theoretical work involves deriving analytical results for increasingly simplified models (nonlinear $\rightarrow$ linear $\rightarrow$ linear two layer).The authors leverage the intuition gained from these simplified models to explain the broader phenomenology of transfer learning, identifying the conditions under which it succeeds or fails as a function of dataset sizes, task similarity, and the strength of feature learning.

### Strengths
This paper considers transfer learning, which is both an important problem in the field and difficult to analyze analytically. The approach of simplifying a complex system to a tractable toy model to build intuition is a valuable and necessary process. As deep learning models become increasingly complex, developing a single theoretical framework that explains their behavior in its entirety becomes improbable. For this reason, this type of work, which provides rigorous insights into a simplified but representative model, is an excellent and crucial step in the right direction. The theoretical results appear promising and provide a new lens through which to understand the interplay of factors governing transfer success.

Overall, this is a very interesting work. With improvements to the overall writing and clarity, particularly in the presentation of the main results, I would be happy to raise my score.

### Weaknesses
The primary weakness of the paper lies in its presentation, which, especially in Section 2, is exceptionally dense. This density hinders the paper's accessibility and, more importantly, obscures the key findings and novel contributions of the work. The following are concrete examples where the writing and structure could be improved:

1.The authors assume the audience is familiar with DMFT (to the point that the acronym is not defined in the main text), which may not be the case for many readers. The paper could greatly benefit from a short primer on DMFT in the main text or a longer appendix dedicated to a brief overview of the key principles relevant to this work. 

2. The italicized "result" paragraphs in Section 2 are long and convoluted, making it difficult for the reader to distill the actual new result, understand its specific implications, and separate it from established background knowledge. For example, in Result 2, Eq. 7 is an established result on infinite networks trained on infinite data, not a new finding specific to transfer learning, and should not have been presented as such (separately) . A similar issue occurs in Result 3, but this time it is further complicated with the introduction of a noise term that is not contextualized properly with respect to previous results.

3. The results are presented sequentially in a technically dense section with minimal guiding principle connecting them beyond the change in setting. A short paragraph giving an overview of these results and how they connect could be highly beneficial. Furthermore, while the titles of the results are meant to provide intuition, the link between the plain-English title and the complex mathematics is often not at all intuitive. A clearer bridge is needed to help the reader understand how the mathematical expressions support the high-level takeaway.

### Questions
1. Regarding **Result 1**: 
(a) The result is stated, but no derivation or methodological sketch is provided in the main text. It would be helpful to include a brief description of the key methods used to obtain this result to make the paper more self-contained.
(b) Can the authors explicitly explain what this result implies beyond the already established fact that the downstream task has a history dependence on the pre-training dynamics? The mathematical formulation is quite involved for what appears to be a known concept.

2. How are the DMFT curves in Figures 3 and 4 obtained? The text states this is from Result 1, but these are non-trivial high-order integral equations. An explanation in the main text of the numerical methods used to solve these equations and generate the curves would be very helpful for reproducibility and understanding.

3. In the phenomenology section (Section 3), it was often not clear which theoretical results were being used to support the empirical findings and simulations. For example, when discussing the CIFAR-10 experiments, it would be beneficial to explicitly state which result (e.g., Result 1, 3, or an ansatz based on them) provides the theoretical basis for the claims being made. This should be clearly emphasized throughout the section.

4. Minor point: In **Result 3**, the text states: “With this kernel, similarly to the sketch of Result 1…”. This seems to be a typo, as the preceding discussion in Result 2 is more relevant. Is it possible the authors were referring to Result 2? If not, it is unclear what sketch from Result 1 is being referenced.

### Soundness
4

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors theoretically analyze a wide variety of transfer learning setups in a feature-learning regime. The analysis relies on DMFT calculations; the resulting learning trajectories are typically history-dependent and expensive to numerically evaluate. In more tractable settings (shallow models, linear models) the authors analytically predict various measurables of the trained estimator, revealing conditions under which transfer learning helps or harms. The theory holds qualitatively in practical empirical settings.

### Strengths
This is a strong paper which uses powerful technical machinery to extract insights about learning trajectories in a transfer learning setup. It's really nice to have a predictive toy model of this phenomenon -- this is the first result (afaik) that establishes average-case predictions for this feature-learning behavior. The DMFT formalism is quite opaque (to me) but the takeaway messages are clearly communicated.

### Weaknesses
1) The authors don't treat the case where fine-tuning lazily updates all weights in the network (rather than just the readout). Do we expect quantitatively similar behavior in this regime?
2) The analytically tractable results rely on an isotropic data assumption which does not hold in many interesting settings. Data anisotropy can qualitatively change the behavior of learning algorithms, especially if $\Sigma_{xx}$ and $\Sigma_{xy}$ do not commute. I think it'd be really interesting to understand how the data geometry interacts with the task geometry, and how the transferability of learned features depends on this interaction! I wish the results touched on this.
3) It's unclear how to think about $\nu_1$. The data-rich limit appears to be $\nu_1\to\infty$, as stated in line 288 and Fig 1a. But the post-training NTK in Result 4 diverges in this limit, and the limit in line 313 seems to contradict the previously-defined data-rich regime.
4) I wish there was more empirics measuring task overlap on real targets :) can't have it all, I guess...

### Questions
1) Out of curiosity -- do you think there exist approximations that aren't as restrictive as linear models, but which might give more analytical insight than Result 1? Would it help at all to impose a lot of mutual structure on $\mathcal{T}_1$ and $\mathcal{T}_2$?
2) Fine-tuning can often be made parameter-efficient by using LoRA. Can DMFT-style results extend to the case where the updates on T2 are forced to be low-rank?
3) In result 2, do you assume that $\|\beta_{(\cdot)}\|^2=D$? That seems necessary to ensure $\alpha \in [-1, 1]$.
4) It would be interesting to extend this analysis to prescribe optimal curricula. Do you think it's possible?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
Summary

The manuscript presents two different theories of transfer learning in
two-layer neuronal networks in the limit of infinite width
where feature learning is assured by muP scaling.

The first theory studies gradient flow and technically employs
dynamical mean field theory to reduce the problem of transfer learning
to an effective single neuron problem on the background of some
self-consistently determined fields.

The second approach (mostly shown in the appendix) studies a more
idealized version of transfer learning in a Bayesian setting, where an
additional "elastic term" is added to the energy function that aligns
weights between the pre- and the post-training phase.

The main part of the manuscript explores the relevant parameters that
control transfer learning and explains qualitative properties, such as
the dependence of the amount of feature learning in the pre- and
post-training phase as well as the alignment of the targets and their
relative complexities (e.g., linear vs higher order polynomial
functions).

The general theoretical results are further illustrated in more
idealized settings (linear activation functions and Gaussian i.i.d.
data) to explain fundamental mechanisms which are subsequently
demonstrated on more real-world data sets (e.g. CIFAR-10, Fashion MNIST).

Overall the manuscript makes an important contribution to the theory
of transfer learning. My main points below are with regard to the
presentation that should still be improved so the work becomes
more accessible also for a broader set of readers.


Soundness

Most of the results appear to be sound and the authors support their
theoretical calculations by comparison to numerical results.

A few points to my mind deserve more explanation:

1.)
In the discussion of Figure 2 it would be important to explain
why panel a) for the theory of the linear model differs qualitatively
from panels b) and c) for the real dataset: Panel a) shows that there
is an optimal feature learning scale \gamma_1^\star (as also stated in the text)
whereas in panel b) and c) that the test loss monotonically
decreases with gamma_1.

2.)
In Figure 3a), the authors should comment on why the asymptotic
value of the test loss is so different between the case of
no pre-training and transfer learning. Should the pre-training
not ultimately become irrelevant at large step sizes of the
second task, so that the two curves are expected to reach the
same asymptotic value?

A similar question for Figure 3b): Here the initial transient
of the test loss seems to be almost identical until the
peak but ultimately at large step sizes, transfer learning
seems to lead to worse generalization than training from scratch.
Why is this?



Presentation

There are a couple of points that the authors should try to improve.
This is to the most extent due to the very dense presentation of
the main text due to the 8 page limit. Here are points I felt
deserve a some care:

1.)
It is confusing to use subscripts _t and _s in section 2.3 to denote
source and target, as the same letters appear as time points.
The authors may want to consider some alternative notation.
Why not use _1 and _2, as also for \mathcal{T}?

2.)
with regard to Result 4:
Please explain the idea of the balance condition in words in the
main text. The appendix is more explicit here.

3.)
Caption of Figure 1:
Remind reader that \alpha is source-target alingment.
Why are you using \alpha_s and \alpha, both of which are defined
in the same way?
Same for \nu_1 \nu_2: remind reader that these are the ratios of P/D.
Panel c: What is the value of \alpha_g for the shown plot?

4.)
A remark would be good explaining why the test loss in Fig 1a
is non-zero at \nu_2 = 0, even if \alpha=1, so source and target task
are identical. Should the loss then not be identical to the test
loss on task 1 alone, so rather small?

5.)
Neither at Eq 13 nor at the discussion of the results in relation
to Fig 1 it is mentioned what is the meaning of the constants c_1 and c_2.
They are explained in the appendix, but a few words in the main text
are needed to understand their meaning, in particular here:

"The simple alignment case (αs =1, αg =0) of Eq. 13 shows that there (i) larger c2 always helps, while (ii) c1 always hurt, since it rotates the high-gain direction towards the noise."

6.)
Figure 5: misalignment of symbol \gamma_2 in caption and \gamma_0
in figure legend.

Appendix B: Here the feature learning strength is defined as \gamma_0,
in the main text as \gamma_1 or \gamma_2, respectively. This should
be made consistent to minize confusion for the reader.

7.)
Eq. (25): not clear how right hand side depends on x' (what is x' btw?);
please clarify.

8.)
Please explain between Eq. (25) and Eq. (26) what is the motivation to
study the particular loss. Also please add some additional steps to
clarify how training the parameters a and W on task T_2 relates
to the dynamics of \hat{beta}.

9.)
When introducing the auxiliary fields in eq. (28) - (30) it would be
good to motivate this by their anticipated self-averaging due to the
summations over high-dimensional index spaces.

10.)
Are the results Eq. (84) - Eq. (95) used subsequently?
It seems you make an ansatz in Eq. (96) that is directly motivated
from Eq. (83) where effective parameters c_1, c_2, c_3 are determined
ad hoc. If this is the case, I would recommend removing Eqs. (84) - (95)
for clarity.

11.)
For the results shown in appendix B it does not become clear how
the numerical results were obtained. It is stated that they are
given by Langevin training.
My guess is that they are obtained by Langevin sampling from the
energy function given in Eq. (180), including the term \propto \delta.
Is this right? If so, it would be important to be stated, for example
in the figure caption.

In particular it should be clarified to the reader that the
results for the Bayesian case cannot be obtained by pre-training
the weights on task one first and subsequently switch to task two
(as is the case for the DMFT), because the posterior only captures the
stationary distribution which I believe should be agnostic to the
initialization. This qualitative difference should be explained
more clearly.


Contribution

The main contribution of the work is to present a theoretical treatment
of transfer learning. This is an important topic and the authors
present important theoretical results that explain qualitative
observation in networks trained on real data.

### Strengths
Transfer learning is an important topic of practical relevance
and theoretical progress is needed. The manuscript condenses this
complex question down to nicely analytically or semi-analytically
tractable settings.

### Weaknesses
The conciseness of the main part makes it hard to follow the main
text. The appendix, however, supplies all details as far as I see
(apart from points marked above).

Some connections between the idealized settings and the real-world
settings are still a bit loose and should be mentioned honestly
in the text (see points under "soundness" above).

### Questions
Please see itemized list above.

### Soundness
3

### Presentation
2

### Contribution
4
