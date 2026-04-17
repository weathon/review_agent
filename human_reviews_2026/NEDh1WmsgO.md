# A Fine-Grained Approach to Explaining Catastrophic Forgetting of Interactions in Class-Incremental Learning

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
This paper explains catastrophic forgetting in class incremental learning (CIL) from a novel perspective of interactions (non-linear relationship) between different input variables. Specifically, we make the first attempt to explicitly identify and quantify which interactions w.r.t. previous classes that are forgotten and preserved over incremental steps, and reveal their distinct behaviors, so as to provide a more fine-grained explanation of catastrophic forgetting. Based on the forgotten interactions, we provide a unified explanation for the effectiveness of different CIL methods in mitigating catastrophic forgetting, i.e., these methods all reduce the forgetting of interactions w.r.t. previous classes, particularly those of low complexities, although these methods are originally designed based on different intuitions and observations. Intrigued by this, we further propose a simple-yet-efficient method with theoretical guarantees to investigate the role of low-complexity interactions in the resistance of catastrophic forgetting, and discover that low-order interaction serves as an effective factor in resisting catastrophic forgetting. The code will be released if the paper is accepted.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors propose one different perspective to analyze the happening of catastrophic forgetting which is the interaction between input variables. The authors prove that the catastrophic forgetting happens because the interactions decrease and current CIL methods work by proving the decrease of interactions. Experiments on different backbones and CIL methods also show the generalization of authors' claim.

### Strengths
1. Different from previous methods, the authors analyze the happening of catastrophic forgetting in depth. This can be helpful for future works to deal with incremental learning.

2. The authors' claim is supported by both theoretical proof and experimental results. The experiments on different backbones and CIL methods also show the generalization of the claim.

### Weaknesses
1. The definition of interaction should be revised in the introduction. I think the current description is hard to understanding without referring to the concrete definition in Section 2. It will be hard for reader to capture the idea in the beginning.

2. According to that the reason of catastrophic forgetting is decrease of interactions, how to avoid it directly from the perspective of retaining interactions? Could you please conduct an experiment when adding one loss to retain interactions to further validate your claim?

### Questions
Please refer to the weakness part.

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
4

### Summary
This paper seeks to explain catastrophic forgetting observed in class incremental learning (CIL) through the lens of nonlinear interactions among input variables. By introducing the Harsanyi dividend, which is commonly used in cooperative game theory to quantify interactions among multiple variables, the authors measure the incremental gain in confidence for the output variable that arises when multiple parts of distinct input variables (e.g., image patches or text tokens) are combined. This approach makes it possible to identify how many input parts must be combined for the confidence increment to become large.

Using the proposed method, the paper quantitatively evaluates, via numerical experiments, how the order of input-part combinations and interactions among those orders change under CIL. Building on the observation that lower-order interaction effects are more susceptible to forgetting in CIL, the paper propose a regularization that suppresses changes in interactions within a specified range of orders. They show that suppressing changes in lower-order interactions mitigates catastrophic forgetting in terms of classification accuracy more effectively than suppressing changes in higher-order interactions.

### Strengths
- The approach of analyzing a trained model by focusing on interactions among parts of the input variables is interesting. In particular, it is suggestive that the confidence increments attributable to interactions exhibit consistent trends by interaction order.
- It is valuable that the paper quantifies which interaction orders are dominantly affected by catastrophic forgetting. The evaluation covers multiple models, datasets, and training methods, primarily on image datasets.
- The proposed method, which suppresses the learning of interactions in a specified range of orders based on insights from the experiments, is shown to be effective.
- The analysis framework sounds reasonable and widely applicable

### Weaknesses
- **Concerns about interpretation of results**: Several interpretations remain concerning, as detailed below.
    - **Is the claim that lower-order interactions are more easily forgotten than higher-order interactions justified?** From Figure 2, the paper interprets the observation that $\mathcal{I}^m_{\mathrm{forget}}$ is larger at lower orders as evidence that lower-order interactions are more prone to forgetting. This interpretation is questionable. A straightforward reading of the results suggests that the total numerical impact of interactions, $\mathcal{I}^m_{\mathrm{forget}} + \mathcal{I}^m_{\mathrm{preserve}}$, tends to be consistently larger at lower orders. Given this, it is unsurprising that $\mathcal{I}^m_{\mathrm{forget}}$ is also larger at lower orders, and this may not constitute a tendency that is specific to catastrophic forgetting.
    - **Concerns about quantifying the effect of CIL models**: To evaluate the effectiveness of existing methods for avoiding catastrophic forgetting (CIL models), the paper defines
        
        $$
        \Delta \mathcal{I}\_{\mathrm{forget}}^m = \mathcal{I}\_{\mathrm{forget, base}}^m - \mathcal{I}\_{\mathrm{forget, CIL}}^m 
        $$
        
        and argues that a larger value indicates a more effective method. However, these quantities do not appear to directly reflect effectiveness against catastrophic forgetting. Here, $\mathcal{I}^m$ measures how much the interactions change when moving from a model $f\_k$ trained up to step $k$ to a model $f\_t$ trained up to step $t$, and *both $f\_k$ and $f\_t$ are obtained under the same training method*. For example, computing both models under the base training yields $\mathcal{I}\_{\mathrm{forget, base}}^m$. In other words, the training procedure for the baseline model $f\_k$ differs between the base model and the CIL model settings. If the total amount of interaction already present at the stage of $f\_k$ differs between the two, then a direct comparison of the subsequent increments does not necessarily indicate superiority on catastrophic forgetting.
        
    - **Concerns on the theoretical soundness**: The proof of Theorem 3 is given in Appendix C. It says that
        
        $$
        \begin{aligned}\mathbb{E}\_{S\_1}\left[\mathbb{E}\_{S \subseteq S\_1,|S|=m}[I(S \mid \boldsymbol{x})]\right] & =\mathbb{E}\_{S\_2}\left[\mathbb{E}\_{S \subseteq S\_2,|S|=m}[I(S \mid \boldsymbol{x})]\right] =\mathbb{E}\_{S \subseteq N,|S|=m}[I(S \mid \boldsymbol{x})] .\end{aligned}
        $$
        
        I suspect this formula may not generally hold true for the condition $m>m_1 n$. Although I believe this should be fixed by some minor modifications, the proof should be precisely documented
        
- **Related work**: The interaction quantity used in the paper, $I(S \mid \boldsymbol{x})$, appears mathematically very close to partial information decomposition (PID) [1], which quantifies the information generated by interactions among multiple random variables. This concept has been widely used in computational biology [2] and, more recently, in machine learning for disentangled representations [3]. Discussing its relationship to PID would be highly beneficial to the ICLR community.
    1. Williams, P. L., & Beer, R. D. (2010). Nonnegative decomposition of multivariate information. arXiv.
    2. Griffith, V., & Koch, C. (2014). Quantifying synergistic mutual information. In Guided self-organization: inception (pp. 159-190).
    3. Tokui, S., & Sato, I. (2022). Disentanglement Analysis with Partial Information Decomposition. ICLR.
- **Concerns about exposition**: Although the overall logical flow is mostly clear, several issues remain, and the current structure is likely to confuse first-time readers.
    - **Order of presentation is suboptimal**: Some concepts defined later are used earlier without explanation. For example, in the paragraph starting at line 68, the quantities $S$, $\boldsymbol{x}\_{\mathrm{masked}}$, and $I(S)$ appear, but the relationship between $S$ and $\boldsymbol{x}\_{\mathrm{masked}}$ and the definition and interpretation of $I(S)$ are not understandable until Section 2.1 starting at line 128.
    - **Unclear mathematical definitions**: In several places, the notation is inconsistent or difficult to interpret, which hinders understanding.
        - **Ambiguity in the meaning of “interaction”**: The phrase “interaction of $S$” appears to be used with multiple meanings. At line 72, the statement “… an interaction $I(S)$ refers to an …” implies that “interaction” denotes *the change in confidence* caused by interactions among variables. In contrast, at line 198 the phrase “the DNN … encodes richest interactions relevant to …” suggests that “interaction” refers to information encoded by the DNN itself. The paper should adopt a consistent definition.
        - **Inconsistent definitions**: At line 201, the change in confidence due to an $m$th-order interaction is written as $I^m(S \mid \boldsymbol{x}_k, f_k)$, but placing $m$ as a superscript on $\mathcal{I}$ rather than on $S$ is potentially misleading. In Figure 1, the corresponding quantity is written as $I(S_m \mid \boldsymbol{x})$, which is inconsistent. Why not use $S_m$ consistently? In addition, in Equation 17, coefficients $m_1, m_2 \in [0, 1]$ appear to denote relative order, which is confusing when compared to $m \in \mathbb{N}$ denoting absolute order.
            
            Moreover, the symbol $f$ is used inconsistently. In Section 2.1, $f$ denotes model confidence, whereas in Section 2.4 the discussion appears to assume raw model outputs. The paper should either use different symbols or clearly state the intended meaning each time.
            
        - **Use of undefined symbols**: In Equation 5, the meaning of $\mathbb{E}_{t=2}^T$ is unclear. Expectation must be taken with respect to a random variable or a probability distribution, but no such specification is given. Undefined symbols also appear in proofs. For instance, Equation 14 uses $f(S_1)$ and Equation 17 uses $v(S_1)$ without explanation.

### Questions
- From Figures 2 and 3, the total effect of interactions appears concentrated at both the lower and higher orders, while medium-order interactions are generally small regardless of problem setup, model, or training method. What could explain this pattern? Is it merely due to the number of combinations in the summation, or is there another cause?
- In Equation 2, the subscript under the summation reads $T \in S$, but should this be $T \subseteq S$?
- In Equation 5, what does $\mathbb{E}_{t=2}^T$ mean? Since an expectation must be defined with respect to a random variable or probability distribution, with respect to which variable or distribution is this expectation taken?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper explains catastrophic forgetting in CIL from the perspective of interactions between different input variables. It tries to provide
an explanation of catastrophic forgetting by identifying and quantifying which interactions w.r.t. previous classes that are forgotten and preserved over incremental steps. The contributions if this paper ca be summarized as:

a. Providing an explanation for catastrophic forgetting by interactions.

b. Revealing the unified mechanism of different CIL methods in mitigating catastrophic forgetting.

c. Explaining the role of low-order interactions in resisting catastrophic forgetting.

### Strengths
a. This paper seems to be technically solid.

b. This paper is well written.

### Weaknesses
a. This paper is well written but not well organized. I think it would be better to be understood if there are more sections.

b. The idea of interactions seem to be similar with the feature learning. [1] is new way to understand catastrophic forgetting by feature learning. I think it is necessary to discuss [1] in details.

c. The notations are confusing. For example, in Section 2, $T$ is the number of tasks (i.e., $\mathcal{D}=\{\mathcal{D} _{1},\mathcal{D} _{2},\cdots,\mathcal{D} _{T}\}$). But in Section 2.1, $T$ is used as $x _T$.

[1] Towards Understanding Catastrophic Forgetting in Two-layer Convolutional Neural Networks. ICML 2025

### Questions
a. Explain what is $T$.

b. What is the shape of $x _T$? What is the relationship between $x _T$ and $S = \\{x _1, x _2 \\}$?

c. The most important concern is about [1]. Please discuss [1] in details and provide the similarities and differences.


[1] Towards Understanding Catastrophic Forgetting in Two-layer Convolutional Neural Networks. ICML 2025.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a novel interpretability approach based on quantifying "interactions" between input variables. The authors apply this method to the field of Class-Incremental Learning (CIL) to provide a fine-grained explanation for catastrophic forgetting, analyzing how interactions related to previous classes are forgotten or preserved. Based on this analysis, the authors propose a unified explanation for the effectiveness of several CIL methods and investigate the specific role that low-order interactions play in mitigating forgetting.

### Strengths
The core idea of applying interpretability methods to better understand the mechanisms of continual learning is a valuable direction for the field. The concept of empirically analyzing the effects of CL methods by observing the effect of impairing the model partially is interesting.  Furthermore, the authors are to be commended for conducting a thorough set of experiments to support their claims.

### Weaknesses
The main weakness of this paper is its presentation, which significantly hinders readability and comprehension. The proposed approach is highly complex, and the manuscript does not sufficiently break down the core concepts into accessible explanations, which seems counter-intuitive for a method intended to improve human interpretability.

Specific issues include:

* **Clarity of Writing**: The paper is very difficult to follow. Key terms, such as "interactions" and notation like $x_{masked}$, are used in the introduction without a clear, upfront definition, forcing the reader to guess their meaning.
* **Vague and inconsistent Definitions**: Several core definitions are unclear.
    * In Theorem 2, $S$ is described as a subset of $T$, whereas $T$ was previously a subset of $S$. Furthermore, $T$ is not defined in the context of Theorem 2.
    * The definition of $f(x)$ is not explicitly stated (e.g., whether it is a single scalar output dimension).
* **Lack of Intuition**: The central definition of an "interaction" (Eq. 2) is introduced without intuition. For example, the mathematical motivation for the $(-1)^{|S|-|T|}$ term is not provided. The claim that $I(S)$ becomes 0 if any patch in the interaction is masked (L155) is also not clearly justified.
* **Overly Complex Formulation**: Section 2.2, which details the quantification of forgotten interactions, is excessively dense and difficult to parse. Equations like Eq. 7 are very long and could be made more readable by using intermediate variables. The method described in Eq. 5 also appears to be computationally intensive.
* **Unclear Mechanism**: It is not clear how the authors' proposed method forces a DNN to encode interactions at specific orders. It is not obvious why maximizing the cross-entropy of $\Delta f(m_1, m_2)$ (Eq. 8) would penalize the orders *between* $m_1$ and $m_2$. The origin and meaning of the variable $n$ in this context are also unclear.
* **Unsupported Claims**: The assertion that defining knowledge in a NN is an interdisciplinary problem "spanning mathematics, cognitive science, neuroscience, and etc." seems overstated and is not substantiated.
* **Formatting Issues**: The PDF contains several formatting irregularities, such as paragraphs preceded by large black dots, which detract from the paper's professionalism.

### Questions
The authors are kindly requested to clarify the following points to improve the paper:

1.  Could the authors provide a clearer definition of $T$ in Section 2 and ensure its consistent use relative to $S$, particularly in light of Theorem 2?
2.  Could the authors offer more intuition for the interaction formula (Eq. 2)? Specifically, what is the reasoning for the $(-1)^{|S|-|T|}$ term, and why would the interaction value $I(S)$ necessarily be 0 if a single variable in $S$ is masked?
3.  What was the rationale for the choice of CL algorithms used in the experiments? They do not appear to be the most common or top-performing methods, so an explanation for their selection would be helpful.
4.  Regarding the method for penalizing interactions: could the authors elaborate on the mechanism by which maximizing the cross-entropy of $\Delta f(m_1, m_2)$ (Eq. 8) selectively penalizes interaction orders *between* $m_1$ and $m_2$?
5.  What do the $C$ variables represent in Theorem 3?

### Soundness
3

### Presentation
1

### Contribution
3
