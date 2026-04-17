# Convex Dominance in Deep Learning I: A Scaling Law of Loss and Learning Rate

- Decision: Accept (Poster)
- Scores: 6, 4, 8, 6

## Abstract
Deep learning has non-convex loss landscape and its optimization dynamics is hard to analyze or control. Nevertheless, the dynamics can be empirically convex-like across various tasks, models, optimizers, hyperparameters, etc. In this work, we examine the applicability of convexity and Lipschitz continuity in deep learning, in order to precisely control the loss dynamics via the learning rate schedules. We illustrate that deep learning quickly becomes weakly convex after a short period of training, and the loss is predicable by an upper bound on the last iterate, which further informs the scaling of optimal learning rate. Through the lens of convexity, we build scaling laws of learning rates and losses that extrapolate as much as $80\times$ across training horizons and $70\times$ across model sizes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper derives the dependency of the last-iterate loss on the sequence of the learning rate for the SGD algorithm under the assumption of convexity and bounded gradient. From the fine-grained upper bound based on the specific learning rate values, the paper derives $i)$ asymptotic bounds on the loss that depends on the peak learning rate, the optimal convergence rate, as well as a qualifying exam of the learning rate schedule that achieves the optimal convergence rate. The paper also presented an extension of the regulating dynamic to the setting of deep learning, and observed empirically the coincidence between the theoretical prediction and the experimental convergence. Lastly, the paper presented a scaling law governing the model size, the number of iterations, and the reference learning rate.

### Strengths
1. The result of the paper applies to any learning rate schedules, rather than just the cosine and WSD in Schaipp et al., 2025.
2. The paper empirically verified the similarity of the empirical training behavior of deep learning models and the theoretically derived last-iterate convergence.
3. The paper also derived a qualifying exam for the learning rate schedule and a scaling law of the learning rate schedules.

### Weaknesses
1. The paper assumes a uniform bound on the expected gradient, which in many case does not hold (e.g. in the training of neural networks based on MSE loss). The following theoretical bound are based on this assumption, which is probably the reason why in the experiment part the paper has to adopt the Adam optimizer instead of the regular SGD (to condition out the influence of the gradient norm). In the setting of the deep learning extension, this uniform upper bound factor $G$ is modified to be a more sophisticated term that may not solely depend on the gradient norm, which can be even harder to track.
2. The qualifying exam of the learning rate schedules takes a complicated form that requires several integrations. This can be hard to apply to a general learning rate schedules. The paper can consider to incorporate several examples that explicitly apply this qualifying exam and validate the result using experiments.
3. The dependency on $\eta_{\text{ref}}$ is not discussed thoroughly in the paper. Although is can be understood that the paper's focus is on the scaling of $T$, it is certainly the case that not all choices of $\eta_{\text{ref}}$ will lead to a meaningful convergence, which is not accounted for in this paper.
4. Some technical part lacks clarity. For instance, it is not clear how to obtain Eq. (2.4) from Eq. (2.3). Moreover, the definition of $N_{\text{small}}$ and $T_{\text{small}}$ in the scaling law part is not clear.

### Questions
None

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a series of upper bounds that are used to show—through regression fitting of their parameters—that different combinations of deep learning models and optimization algorithms display a convex-like behavior across the training path. In particular, this is used to predict asymptotic $O(\frac{1}{\sqrt{T}})$ loss convergence and learning rate. The upper bounds are characterized as “generalizations” of bounds derived from loss functions that otherwise are convex (or at least “star-convex”) and have bounded gradient.

### Strengths
The paper strengths are:
- The paper’s motivation of predicting convex-like behavior is relevant in the understanding of the training of deep learning models. 
- The scaling laws found by the paper, which establishes evidence towards convex-like behavior in the training of deep models, are extensively characterized in their data-driven approach.
- The paper has enough models and training procedures to properly demonstrate its claims. I really appreciate that from the authors.

### Weaknesses
Although the paper’s topic is relevant, there were multiple things that were unclear in the paper’s presentation (some of them could have been spotted after diligent proofreading). Also, there were confusing parts in the presentation of both theoretical and experimental results—I will detail this below. Given that there is still work to be done to improve the paper, I am giving the current score.

**>>Important things:**
- I find it odd that in the last sentence of the paragraph from line 057, there is reference to equations (2.3) and (2.4) and the work by Defazio et al. (2023), as if the reader could immediately recognize what the authors are referring to with such references. Moreover, it is hard to know from the sentence in lines 059-061 what exactly the authors are doing differently than Defazio et al. (2023).
- The inclusion of “Example 1.1.” is confusing. A few notes:
  - There is no explanation of the meaning of parameters $L_*$, $D$, $T$, and $G$. How is the reader supposed to understand Example 1.1 then?
  - Moreover, there is another problem: it is said that the shown equation “aligns with the empirical trade-off in deep learning that larger $\eta$ converges faster but to a higher loss, and vice versa”. Two problems with this: Firstly, **there is no citation** to back up this claim: the authors must add one. Secondly, the statement is misleading: what becomes higher is the **upper bound on the loss** that the authors have derived, and not the loss *itself* (but the reader has no way to know this, since no explanation is given in Example 1.1.).
  - Example 1.1 needs to be urgently fixed considering what I pointed out. It may need to be fully reformulated. Also, the title “Example” sounds odd–maybe “Finding” or even "Example of Findings” could be better.
- Line 116: Shouldn’t $\eta_{peak}$ be constrained to be positive, instead of being any real number?
- Equation (2.2): It should say “$\exists G$ such that $\forall w”.
- Remark 2.2: $w_*$ is stated as being **unique** because of the expression $w_*=argmin$ (instead of using the expression $w_*\in argmin$). However, convex functions **do not necessarily** have a unique minimum (I don’t think that Condition 2.1 implies uniqueness; please, correct me if I am wrong). Why is uniqueness assumed? Does uniqueness play an important role in **any** of the derivations in the paper?
- The paper goes through a series of elemental derivations to obtain equation (2.3). However, then the work (Defazio et al., 2023) is cited to obtain (2.4), without any mention of whether equation (2.3) was used to derive equation (2.4), and, if that is the case, there is no mention of how (2.4) came from (2.3). The question is: why derive equation (2.3) when at the end, as the paragraph that follows equation (2.4) states, only (2.4) is going to be used in the whole paper? Why not simply cite (Defazio et al., 2023) and avoid derivations that do not seem useful at all? The shown derivations would only be useful if an explicit connection and derivation is made between (2.3) and (2.4), but this is absent in the paper.
- In line 160 and in Table 1, the term “optimal loss” **does not** refer to the optimal loss itself, but it refers to the “**optimal upper bound on the loss**”. This needs to be fixed.
- Line 185: “Section 2.4” is mentioned when, in fact, the paper has no “Section 2.4”. What is supposed to go there?
- It is confusing that equation (2.2) of Condition 2.1 is included as an assumption for the **whole** paper, when in reality, according to Remark 2.2., only star-convexity is needed (the unnumbered equation inside Remark 2.2). So, if all the paper results only need star-convexity (assuming Defazio et al (2023) also used star convexity), why not replace equation (2.2) by star-convexity? One could just make a simple remark saying that convexity is a stronger notion than star-convexity. Do the authors have a strong reason for keeping (2.2) inside Condition 2.1 instead of being replaced by star convexity? 
- I **strongly suggest to emphasize** in Section 3 that: **>>>>** All the results described in Section 3.2 and Section 3.3 indicate convex-like behavior, and that such behavior is due to **two factors**: the model itself and the chosen optimization algorithm. These two factors were originally present in the upper bound where the generalized bound was derived from. In other words, though the loss function of the neural model is non-convex, its behavior **along the optimization path** behaves according to a convex-like upper bound. **<<<<**. This summarization is currently absent in the important Section 3. Also, the use of the term “optimization path” as *where* the convex-like behavior happens is very useful. 
- There seem to be figures missing or a mismatch between the text and the figures caption. For example, line 261 mentions that ResNet50 is present in Figure 2, however, Figure 2’s caption shows ResNet18. Likewise, line 282 mentions that ReNet18 is present in Figure 3, while Figure 3’s caption shows ResNet50. In any case, the paper **will greatly benefit** from the presentation of both SGD and AdamW for both ResNet18 and ResNet50 models.
- Section 4.1 does not mention which model is being trained for the experiments. I suspect that it is some sort of language model, but the model **must** be included.
- It is really cumbersome for the reader to have all types of generalizations spread out in the paper, when they are sort of related to each other. I **strongly suggest** to put all generalization bounds on a single table to aid the reader to better understand the paper. The table can include columns **describing** (i) where/when each generalization bound is used, and (ii) which parameters need to be obtained through regression.
- To better understand the experiments of Figure 8 and Figure 9 (something similar for Figure 10), it is important to include a text explaining that the reason why one cares about **fitting straight lines** is because the upper bound in Generalization 4 is just a line: for different values of $\frac{1}{\sqrt{T}}$, the slope $\tilde{Q}$ and the offset $\tilde{L}_*$ needs to be estimated. Unless I missed something, this explicit information of which **parameters the figures are fitting** is missed in Section 5.
- Many citations are **missing their publication year**. This needs to be fixed.

**>>Unclear things:**
- Lines 031-032: How is it that the Llama training being mentioned is similar to the other SGD training being mentioned? What kind of metric of similarity was used? Which citation presents such a claim?
- Line 076: uses the term “qualifying exam” without any reference of what this means and with respect to *what* something is being qualified. Moreover, the term “qualified” is used again in line 105 without any explanation of its meaning. The reader has to read past two pages to understand what these expressions mean. These expressions need to be specified early on if they are going to be used.
- Line 183: refers to Theorem 1. The problem is that Theorem 1 is in the Appendix, which makes the paper less self-contained. It would help to include a little description of the specific cases that Theorem 1 analyzes: in other words, to move the explanations from lines 216-217 to inside the sentence that starts in line 183.
- The important information that only half of the iterations are used to fit the bound in page 5 should be moved from footnote 1 to the main text of Section 3.2. This is an important experimental detail which is odd to relegate to a footnote.

**>>Other presentation issues:**
- Many references need to have parenthesis before the names of the cited authors; this is used when one refers to the *work* and not to the *authors themselves*. Use “\citep{}” for this. This occurs throughout the paper and needs attention. The first time this problem appears, for example, is in lines 026-027 (it should start with “(Garipov et al, 2028; …)”). Similarly, it should be “(Krogh & …, 1991)” and “(Hoerl & …, 1970)” in lines 054-055. Multiple other parts to fix exist throughout the paper.
 - Lines 045-046: How is it that the spectral properties of the Hessian indicate “convexity”? Does it have to do with local curvature or something related to it? Please explain.
- End of line 088: should say “for the fixed value $\alpha=0.5$”.
- Section 2.1: specify that $g_t:=g(w_t)$.
- Line 140: it should say “applying Jensen’s inequality”.
- Line 158-159: it should say “We show the upper bounds on the loss in terms of different learning rates (...)”.
- Line 160: it says “see appendix”. Which section of the appendix does it refer to?
- Line 161: it should say “in Table 1” instead of using the preposition “from”.
- Line 430: include citation from where Muon comes from.

### Questions
Please, see the Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper extends the discoveries in [1], which states that the upper bound driven from convex optimization for SGD matches the loss curve of deep learning in general. In other words, in deep learning SGD has coeffients A, B that satisfy
$$
L(w_T) - L^{*} \leq \frac{A}{T\eta_{max}} + B\eta_{max},
$$
where $\eta$ is learning rate, $A$ is an analogy of the diameter, and $B$ is an analogy of the maximal function value. The core contribution of this paper is extending this tendancy to other optimizers and find that the principle holds indeed. Different from SGD, the coefficients are not exactly given from convex optimization, so they propose a "data-driven" approach where they use certain points on the training curve to approximate $A$, $B$, and predict the rest. They validate their discoveries across various experiments: the tendancy across different max learning rates, upon the optimal learning rate, and a scaling law that predicts both loss and optimal learning rate.

[1] Schaipp, Fabian, et al. "The surprising agreement between convex optimization theory and learning-rate scheduling for large model training." arXiv preprint arXiv:2501.18965 (2025).

### Strengths
I believe it is rare to have a nice, predictable tendancy in neural network training, and such discovery as in this paper is useful because once we can predict what will work best in practice, e.g. can compute the optimal learning rate by hand. Even though this is an extension (at least in my view), the fact that the tendancy exists for different optimizers is an important discovery, because it is not obvious. So one strength of this paper is that it proposes a property of neural network training that seems actually useful.

Another strength of this paper is an extensive experimental result with very high correlation ($R^2 \geq 0.95$, mostly even higher). I think these results show a strong signal that the authors have indeed identified a tendancy.

### Weaknesses
One thing that was not clear to me was:

so we have an upper bound that looks like
$$
L(w_T) - L^{*} \leq \frac{A}{T\eta_{max}} + B\eta_{max}.
$$

So is $A, B$ a function of the optimizer, the model, and the learning rate schedule? In other words, if these three are fixed, is $A$ and $B$ fixed regardless of the maximal learning rate? If this is not the case, I think it would make the paper much weaker (and there is potential that I may lower the score), because even though there is a tendancy there is no practical benefit as for every maximal learning rate we should find $A$, $B$ again and there might not be a practical scheme. It would be great if the authors clarify this point, and it would be amazing to show a practical application of this result in the experiments section (e.g. actually tuning learning rate with the prediction). 

Eventually, it would be good to explicitly show how $A, B$ and the various different factors in training is related.

### Questions
One thing I got curious: why is this phenomenon happening? Is it really because the landscape becomes convex in the latter parts of training? If that is the case, is there a way to actually verify whether the landscape is convex or not?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores whether convex optimization insights can reliably predict and control loss dynamics in deep learning through learning-rate (LR) schedules. Starting from a known convex, bounded-gradient last-iterate bound, the authors compute schedule-specific last-iterate bounds and show that horizon-aware “qualified” schedules (e.g., linear, cosine, WSD) achieve $\mathcal{O}\left(\tfrac{1}{\sqrt{T}}\right)$ optimal last-iterate convergence with $\eta_{\text{peak}} \propto \tfrac{1}{\sqrt{T}}$ (Theorem 1, Table 1). They then propose a training-free “qualifying exam” (Condition 2.4) using integral approximations to identify such schedules. Moving beyond convex SGD, they introduce data-fitted coefficients $(\tilde{q}\_1, \tilde{q}\_2, L\_{\infty})$ and show empirically that (i) a sequence-to-sequence mapping from LR schedule to loss captures training trajectories across models, optimizers, and schedules ($R^2 \geq 0.95$ in most cases), and (ii) last-iterate loss scales approximately linearly in $1/\sqrt{T}$ under $\eta_{\text{peak}} \propto 1/\sqrt{T}$, with cross-horizon and cross-size extrapolations. Analyses include new runs in vision and language and external large-scale results to support a simple two-dimensional scaling relation.

### Strengths
- **Broad, consistent empirical signal.** The sequence-to-sequence fits trained on the first half of iterations and evaluated on the second half show high $R^2$ across vision (ResNet, ViT) and language (GPT-2), optimizers (SGD, AdamW, Muon-NSGD), and LR schedules.
- **Simple, actionable guidance.** Horizon-aware schedules with $\eta_{\text{peak}} \propto 1/\sqrt{T}$, paired with a few pilot runs to pick $\eta_{\text{ref}}^\star$, provide a practical recipe for early forecasting and planning.
- **Cross-horizon and cross-size extrapolation.** The paper demonstrates extrapolation up to $\sim 80 \times$ in horizon and $\sim 70\times$ in model size, and complements with analyses of external large-scale results.

### Weaknesses
- **Limited theoretical novelty.** Main results rely on standard convex SGD analysis; $\mathcal{O}(1/\sqrt{T})$ rates are classical. The main theoretical additions are schedule-wise constants (some via approximation).
- **Reproducibility and robustness gaps.** No code/logs are released. Experiments primarily vary LR and $T$; other key hyperparameters (weight decay, momentum/$\beta$’s, clipping, batch size) are mostly fixed (e.g., wd$=0.01$) with no ablations. 
- **Scope of applicability.** Appendix~D shows the $1/\sqrt{T}$ law may fail for test loss under overfitting; diagnostics to detect onset/breakdown of the “convex-like” regime are not provided.

### Questions
- Will you release code, configurations, and logs (including seeds) for all experiments and figures?
- How sensitive are $(\tilde{q}\_1,\tilde{q}_2, L\_{\infty})$ and fit quality to weight decay, momentum/$\beta$’s, gradient clipping, batch size, data order, and seeds? Please add ablations with error bars.
- Can you justify Condition 2.4’s sum $\to$ integral replacement or provide bounds connecting the integral “exam” to the discrete last-iterate behavior?
- In your experiments, how sensitive are fitted slopes/intercepts to batch size?
- Can you propose diagnostics to detect the onset and breakdown of the “convex-like” regime (e.g., via curvature/Hessian measures) and validate test-loss predictions where possible?

### Soundness
3

### Presentation
3

### Contribution
2
