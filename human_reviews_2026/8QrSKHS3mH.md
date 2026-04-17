# Mechanistic Study of Transformer In-Context Learning with Categorical Outputs

- Decision: Reject
- Scores: 6, 4, 4, 0

## Abstract
We study in-context learning (ICL) with Transformers for categorical outputs $y_i$, a setting largely unexplored compared to research on real-valued $y_i$. While attention-only Transformers can, in principle, perform functional gradient descent (GD) inference for real-valued outputs, we show that categorical $y_i$ introduce a non-linearity in GD that attention-only models cannot capture. This reveals a crucial role for the Transformer's multi-layered perceptron (MLP) layers, which we show are generally necessary for categorical ICL. However, we also analyze conditions under which attention-only models can, surprisingly, still perform well. Since training for categorical ICL requires substantial data, we propose a sparse Transformer parametrization linked to functional GD. This model trains far more efficiently with minimal performance degradation compared to an unconstrained Transformer. Our sparse design proves particularly valuable for data-limited applications, which we demonstrate through the ICL analysis of human surgical procedures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates in-context learning (ICL) with transformers for categorical outputs.
It shows that categorical targets introduce a nonlinear gradient update that attention-only models cannot represent, revealing a key role for the MLP layers.
The authors further identify conditions where attention-only models still perform well, propose sparse transformer parametrization linked to the functional gradient descent, and validate their theoretical findings through several experiments.

### Strengths
The paper makes theoretical and empirical contributions by focusing on categorical ICL, analyzing the functional gradient descent perspective to explain why MLP layers are necessary for nonlinear expectation updates, and validating these insights through several experiments. So, I think that this paper provide a meaningful findings supported by empirical evidence.

### Weaknesses
While this paper claims that MLPs play a key role in approximating the nonlinear expectation (lines 319–326), this point remains unclear to me from a mathematical perspective.  This may simply be because I have not yet fully understood why the functional gradient descent step is identified with Transformer layers (around line 297). Could you please provide more explanation of the theoretical role of the MLPs ? I elaborate on the specific question below.

### Questions
From a purely theoretical standpoint, if the goal is to approximate the nonlinear mapping $f \mapsto \mathbb{E}(w|f)$, is it truly necessary to rely on MLPs? Couldn’t a single-layer or deep self-attention approximate this mapping as well ?  I have seen recent approximation results (for instance, Hu et al, Universal Approximation with Softmax Attention, 2025; and  Kajitsuka and Sato, Are transformers with one layer self-attention using low-rank weight matrices universal approximators?, 2023; and Furuya et al, Transformers are universal in-context learners, 2024) show that even one-layer or deep attention architectures can achieve universal approximation  without relying on MLPs.

### Soundness
3

### Presentation
3

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
The paper studies the in-context learning of transformers for categorical outputs. Experiments are conducted to show that traditional transformers and attention-only models can indeed learn ICL tasks of categorical outputs. Theoretical analysis is also provided to show that a two-layer attention-only model can perform two steps of GD approximately, but further steps of GD wouldn't be possible without MLP layers.

### Strengths
1. The paper studies the ICL task of categorical output, which is more relevant to the NLP of transformers where transformers "choose" which token to output.
2. The idea of an attention layer approximating the expectation $\mathbb{E}(w|f)$ through first-order Taylor expansion is somewhat interesting.

### Weaknesses
1. The writing could be improved. In sec 3, some basic intuition on what the experiments are about could be included at the beginning. The  baselines should also be introduced in a more specific way. For example, I don't understand what the GD model is until I read line 230, which tells me it is a transformer "trained with constraints on its parameters", but the details are left to "the next section". Also, it's a bit strange to call a pretrained transformer model simply by "GD", which may cause some misunderstandings.

2. The conclusions drawn from experiments in sec 3 doesn't seem too surprising, given we already know that transformers can learn classification functions in-context [1,2,3,4].

3. The theoretical derivations in sec 4 largely follows [5] without much novelty, and the conclusioin is that a two layer attention only model can approximate two steps of GD, but further GD steps require MLP layers, which doesn't align with experiments in sec 3. 

4. The experiments in sec 6 doesn't seem to be very relevant to the point of the paper, which is to study the **mechanistic** of TF ICL on categorical data, not **whether** TF can learn the task or not.

[1] Edwards, A., & Camacho-Collados, J. (2024). Language models for text classification: Is in-context learning enough?. arXiv preprint arXiv:2403.17661.

[2] Milios, A., Reddy, S., & Bahdanau, D. (2023). In-context learning for text classification with many labels. arXiv preprint arXiv:2309.10954.

[3] Reddy, G. (2023). The mechanistic basis of data dependence and abrupt learning in an in-context classification task. arXiv preprint arXiv:2312.03002.

[4] Bai, Y., Chen, F., Wang, H., Xiong, C., & Mei, S. (2023). Transformers as statisticians: Provable in-context learning with in-context algorithm selection. Advances in neural information processing systems, 36, 57125-57211.

[5] Wang, A. T., Convertino, W., Cheng, X., Henao, R., & Carin, L. (2024). On Understanding Attention-Based In-Context Learning for Categorical Data. arXiv preprint arXiv:2405.17248.

### Questions
1. What is the "GD" model and what is its main difference compared with "Trained TF" besides initialization?

2. What is the takeaway message of this paper? The theoretical analysis seems to be inconsistent with the experiments, and the experiments seem to be unsurprising.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies "categorical ICL", a variant of the "ICL for regression setup" of Garg et al in which the labels are categorical rather than real valued. The authors show that an MLP layer can sometimes help in these settings, and suggest that if the training and test distributions match, it may not be as useful to have the MLP. The authors also compare full training with "GD" training, in which some of the weights are constrained in a way that is inspired by the transformers that learn gradient descent to solve regression problems. They demonstrate that this structure often significantly speeds up training. Finally, they demonstrate the effectiveness of GD training on a problem inspired by a medical application.

### Strengths
The application of GD based Transformer training to the real-world problem associated with surgery is very interesting, as is the demonstrated improvement in training speed when using GD training instead of full training.

### Weaknesses
The model of section 3.1 seems very contrived. Why are the anchor points $\tilde{x}(i)$ needed? Wouldnt it be natural to let $f(x)$ be "anything" (say lipschitz)? Is this just to make the analysis easier or is this fundamental?

Much of the discussion seems very specific to the "Gradient Descent Interpretation" of ICL (Cheng et al). For instance, the discussion around Line 319. How would these discussions extend to more general settings where the objective of the layers is not necessarily to perform gradient descent for a regression problem.

A comparison between full training and GD for the medical task would be useful to demonstrate the effectiveness of the proposed method.

### Questions
On line 155, do you mean "... for context $m$"?

It seems like there is a "MLP + attention vs attention-only" and "TF vs GD" comparison in the paper. How are these related? Is the final conclusion that "MLP+attention using GD" is the best when there is a mismatch in training and test distributions, whereas if those distributions match, the MLP is not useful? 

In line 199-204, I think it would be useful to hve a brief description of these datasets. Are they of the form (image, category)? Is the value of C so low to maintain a high N/C ratio?

I think it would be helpful to write out the CA model of Wang et al. for readers who are unfamiliar with it, especially since Figure 3 (left) and (center) agree so well.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper considers transformer's in-context learning (ICL) ability for categorical outputs. They claim that attention-only transformers are insufficient in this case and the MLP layers in traditional transformers are necessary. They claim they develop a sparse transformer parameterization that trains more efficiently. They also consider ICL in multi-question image analysis in the surgical contexts.

### Strengths
1. The problem considered is interesting and may potentially deepen our understanding of ICL ability of transformers.

### Weaknesses
1. The paper is very poorly written. It is very difficult to understand the mains results, both theoretical and empirical. For the empirical results, the experimental setup is described in great detail but not easy to understand. It is unclear what questions they are meant to answer, and why they can answer the questions. There is a list of empirical findings in Section 3.3, but they are not properly discussed and it is unclear if they are well supported by the experiments. For the theoretical analysis, the results are scattered in the informal discussions. It is hard to see what are the main results and their implications, why they are interesting and important. 

2. The claimed sparse transformer parameterization is not properly introduced. 

3. The purpose surgical example is unclear. It seems unrelated to the main theme of the paper other than it is an example of ICL.

### Questions
See weakness.

### Soundness
1

### Presentation
1

### Contribution
1
