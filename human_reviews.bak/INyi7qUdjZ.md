# Differential learning kinetics govern the transition from memorization to generalization during in-context learning

- Decision: Accept (Spotlight)
- Scores: 8, 10, 3, 8

## Abstract
Transformers exhibit in-context learning (ICL): the ability to use novel information presented in the context without additional weight updates. Recent work shows that ICL emerges when models are trained on a sufficiently diverse set of tasks and the transition from memorization to generalization is sharp with increasing task diversity. One interpretation is that a network's limited capacity to memorize favors generalization. Here, we examine the mechanistic underpinnings of this transition using a small transformer applied to a synthetic ICL task. Using theory and experiment, we show that the sub-circuits that memorize and generalize can be viewed as largely independent. The relative *rates* at which these sub-circuits learn explains the transition from memorization to generalization, rather than capacity constraints. We uncover a memorization scaling law, which determines the task diversity threshold at which the network generalizes. The theory quantitatively explains a variety of other ICL-related phenomena, including the long-tailed distribution of when ICL is acquired, the bimodal behavior of solutions close to the task diversity threshold, the influence of contextual and data distributional statistics on ICL, and the transient nature of ICL.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper provides a mechanistic interpretability analysis of ICL and IWL, explaining observed phenomena from the literature and presenting testable predictions for other spacing laws. The authors hypothesize and experimentally demonstrate that the sub-circuits responsible for memorization and generalization can largely be considered independent.

### Strengths
First of all, congratulations on your ICLR submission! I’d like to commend the author for this excellent piece of research. Below is my review of the work:

**Originality**
The paper is bringing a novel methodology to study ICL and IWL, and understand the trade of between the generalisation and memorisation.

**Quality**
The paper is of general great quality
- The paper effectively acknowledges its limitations.
- The experimental results are well-detailed and thoughtfully motivated.
- The findings demonstrate that the predictions made using the smaller model hold up in a slightly more general model used to study ICL and IWL.

**Clarity**
The paper is very well-structured, making it easy to follow. The figures are clear, informative, and complement the content effectively.

**Significance**
This work is highly relevant to the research community and plays a crucial role in improving the interpretability of neural networks. I would like to highlight its significance, particularly for the neuroscience community, where there is potential for fascinating cross-disciplinary ideas, as exemplified in this paper see . These paper explore bi level optimisation as a mechanism for effective learning and generalisation [1,2]. 

[1] Jensen, K.T., Hennequin, G. and Mattar, M.G., 2024. A recurrent network model of planning explains hippocampal replay and human behavior. Nature Neuroscience, pp.1-9.
[2] Murray, J.M. and Escola, G.S., 2020. Remembrance of things practiced with fast and slow learning in cortical and subcortical pathways. Nature Communications, 11(1), p.6441.

### Weaknesses
**Originality**
The topic is highly relevant and aligns with current research trends, yet it still offers an intriguing and fresh perspective.

**Quality**
A more detailed discussion of the limitations and assumptions would enrich the paper. Additionally, delving deeper into the comparison with capacity theory could provide further insights (see question below).

**Clarity**
Suggestion: Including a small diagram of the model would improve readability, especially considering its compact size.
Could the results be validated in a more step-by-step manner? This would clarify that the approximations are sound and the rationale behind them, potentially expanding on this in the appendix.
Providing full derivations in the appendix would enhance completeness and transparency.

**Significance**
While the authors have noted the need for validation with larger models, doing so would strengthen the paper, despite the acknowledged material constraints.

### Questions
**Questions**

- In previous works that observed the transient nature of the network, were weight regularizers applied?
- Could there be an alternative explanation beyond the dynamical and capacity arguments?
- For future work, it would be interesting to explore the dynamic effects under capacity limitations. Do you have any intuition about the phenomena that might arise, considering that both could occur in real-world models?
- How do these insights from ICL and IWL models compare with findings at scale in previous literature? Do you expect some of your results to hold at scale?

I hope this review is helpful for the further development of the paper. I encourage the author to continue this research and incorporate the feedback provided.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
The paper studies the transition behavior of in-context learning (ICL) and in-weight learning (IWL) in single-layer, single-head, nonlinear Transformer, trained on a variant of the binary labelling task introduced by Chan (2022). The paper then reduces this setting to an effective, “minimal” model that treats ICL and IWL as separate terms, leading to a pair of coupled scalar ODEs stemming from gradient flow on a data-averaged loss function. The minimal model is able to qualitatively explain empirical phenomena in the literature. Importantly, for the setting considered, the provided explanations match quantitatively, and with high precision. 

In particular, the model explains the following empirical phenomena

1. Sharp acquisition of ICL w.r.t. task diversity (modelled by number of training samples $K$), with bimodal IWL-ICL distribution at the threshold K\* (Kirsch, 2022\)  
2. Long-tailed distribution of ICL acquisition times (Reddy, 2023\)  
3. Transience of ICL (Singh, 2023\)  
4. (somewhat) The dependence on data statistics, in the paper modelled as label inbalance $n\_+/N$

In addition, the model makes two predictions: 1) That acquisition time $t_{ICL}$ scales linearly with context length $N$, and that "task diversity" $K^\star$ where ICL and IWL coexist depends on context length $N$ with a power law $ K^* \sim N^\{\nu=0.72\}$.

### Recommendation
The explanatory power, originality, and simplicity of the model are striking and well outweigh my criticisms. Moreover, the paper is highly relevant for the current discussion on in-context learning. I recommend acceptance and a prominent highlight at the conference.

### Strengths
- The modelling approach is plausible and original, and the power despite its simplicity of the minimal model is striking.

### Weaknesses
(more important points first)

- While the model qualitatively explains the literature as stated above, quantitative connections are missing. I understand that some quantitative predictions may be contingent on the one-layer-one-head architecture and task, but others may be transferable, such as the two predictions mentioned in the Summary. I would find it interesting to evaluate these on a naturalistic task. Two examples come to mind: The path-finding task introduces in (Brinkmann et al., 2024) and possibly also a lightweight NLP setting. 
- It is unclear to me how strong the dependence is in terms of data statistics. In particular, is there an intuition why the edge case of exactly $N/2$ positive $\ell=\+1$ tokens is so different? 
- It took me a while to understand the relation between the two losses at the end of Section 3.7. It would help the reader to find details, possibly in the Appendix. 
- Why does $\\beta=\\infty$ (or even just a large value) in Eq. (12) signify ICL dominance? I see this for $w$, but not for $\\beta$.   
- The claim that $c\_1(t)$ has a power-law up to $K=10^5$ is not supported by Fig. A.3. The claim needs to be weakened or additional simulations are required.   
- The title of the paper, “differential learning kinetics”, that also is used to label the distinction in Fig. 1, is not picked up much in the main text, and in my opinion also not very descriptive overall. I feel the paper would benefit from connecting the theory to this intuition that the authors developed.   
- In Fig. 1 and 2: At what time is $K^*$ evaluated?
- No code is published, limiting transparency and reproduciblity.

#### **Minor**

- The paper states in line 376  
> While previous work has shown that such skewed data distributions favor ICL acquisition (Chan et al. (2022)), our theory suggests that such data dependencies can be examined by simply measuring $I\_K$.  

  As I understand these findings are compatible, the wording here to me however makes it unclear whether they are or not.  
- It is not clear how good of a proxy $K$ is for data diversity, generally, so this caveat should be mentioned.
- Sometimes $\rangle\_{\\phi\_+}$ and sometimes  $\rangle\_{\\phi\_+ \\sim P\_+}$ are used, that was slightly confusing to me. Also, $P_\+$ and $P(n_+)$ could be differentiated more clearly notationally.
- In line 413 it says
> Equation 16 implies that $w$ after ICL acquisition tracks $w_{tr}$...

I assume $w_{tr}$ stems from gradient descent on Eq. (15), but this and why it bears this particular name is not clear.
- Data diversity is modelled by $K$, I would indicate this in Fig. 1 on the y-axis  
- In line 361, the case $-\beta\_0 \gg 1$ is discussed, and then a conclusion is being made on long-tailedness. I believe that the results show that long-tailedness should hold for both $-\beta\_0 \\gg 1$ and $-\\beta\_0 \\ll 1$, but the wording here made it seem like it only applies to the former.   
- Just before Section 3.3, the paper states   
> …three phenomena of interest, the transition from memorization to generalization, abrupt ICL learning and ICL transience,...  
  
  of which the first and second seem identical, and the (Reddy 2023\) long-tailedness property does not appear.
- The Gaussian on the y-axis s Fig. 6c is not referenced.

### Questions
I've expressed my questions in the Weaknesses section above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper presents a small transformer model performing a synthetic task aimed at separating in-context and in-weights learning. A surrogate model which is amenable to analysis is also presented. The paper aims to combine the analysis of this ``minimal model'' and the empirical analysis of the original transformer to understand under what conditions in-context learning emerges. The training dynamics of the minimal model are characterised and used to analyse the loss landscape of the in-context pathway of the model. Further, a number of phenomena of transformers performing in-context learning are considered, such as the transient nature of this capability.

### Strengths
# Originality
To my knowledge this is a new combination of ideas from the literature. The tasks and network structure similar to the work of Chan et. al. (2022) is combined with the analysis of the loss landscape and loss dynamics in various high-dimensional limits (while it is not mentioned in the work, the analysis has D and K very larger - the input dimension and number of datapoints - which constitutes the well known thermodynamic limit). By working in this limit much of the dynamics of training become tractable and this is exploited here.

# Quality
The tasks and model used here is well motivated and clearly relevant to in-context learning and useful for the aims of the work. The hypotheses or motivation of the work is clear.

#  Clarity
Overall use of language in this work is clear and it is well written. Figures are neat and legible.

# Significance
This work touches on a number of established phenomena on in-context learning but also introduces new phenomena which would be useful for further research. The model itself could likely be built upon in further work.

### Weaknesses
# Quality
The work is vague and lacks the necessary degree of rigor. The fact that no full derivations are presented even in the appendix is already grounds for rejection in my opinion. I recommend the authors provide a more thorough presentation of the derivations in this work. 

This somewhat casual approach to the mathematics means that some very important simplify assumptions are not described accurately and in some case the work changes assumptions or conditions to suit whatever point it is currently trying to make. The most clear example of this is the dimension of the input D. It is foundational to the analysis here that D is large enough that the key-query matrix will be diagonal. This appears to me to be a huge simplification, worthy of clear mention. But the work fails to even acknowledge this in Section 3.2 when describing the model, instead referencing the fact indirectly and terming it an ``independence ansatz''. This fact of needing infinite D is only brought up in Section 3.3. This lack of clear assumptions and formal definition results in the jump from Equation 3 to Equation 4 being unclear. A similar case can be made for the implicit simplicity of the setup of binary classification itself - which is necessary for the analysis which relies of a single variable specifying the probability of the  MLP being correct ($P^+$). Thus, the jump from Equation 4 to the equation after the sentence  "since the two labels are symmetric the average loss $\mathcal{L}$ can be written as" (which again is not actually shown) is also unclear and missing multiple steps. Another example is the treatment of $N$ right before Equation 5, where it is never mentioned that N follows a distribution before this point, but then it is just defined as being approximated by a normal distribution and the loss equation updated again without derivation or explanation. This happens throughout the work and needs to be correct, for example before Equation 6 it is merely stated "From Equation 5 a few steps of simplification leads to...". I think I make my point but for clarity I will point out some quicker examples:
- Lines 297 to 302: The conclusion of the loss from the MLP separating from the in-context pathway seems true by construction and if I am not mistaken is essentially the independence ansatz. I'm not certain why here it is being phrased as an emergent phenomenon.
- Figure 4: The landscape depends on the probability of the MLP being correct but this is omitted entirely.
- Lines 329 to 330: it says "if the MLP is unable to perfectly memorize the K item-label pairs" but on lines 225 and 226 it is said that "all the information required by the MLP to memorize the label of the target is present in $t_{N+1}$". These are contradictory statements. It seems the second sentence is an assumption which is used in the derivation (or else you cannot summarize the MLP contribution just with $P^+$ - it is difficult to tell without the derivation). This would mean that you cannot interpret the model in the case of the MLP not being able to learn.
- Lines 336 to 338: The ultimate conclusion of this section is that the model behaviour depends on the initial values of $\omega$ and $\beta$ (less the MLP cannot learn the dataset in which case in-context learn will clean up the remaining loss as best it can). Given this conclusion I am confused what the point of this section is given that the condition of $\omega, \beta >> 1$ resulting in ICL is already noted from Equation 3. It's possible I am missing some nuance around this point but without derivations it is difficult to say.
- Two Taylor expansions are used in this work with almost no mention of what sorts of terms or information is lost but this approximation.
- Lines 355 and 356: It is not clear at all that $c_1(t),c_2(t) \approx 1/2$ here. This is likely due to the information treatment and lack of explanation of $I_k(t)$.
- Lines 369 to 372: The relationship between $c_1$ and time is not clearly discussed and $c_1$ in it's own right is not clear explained. Why is the mean of the sigmoid of the negative of the MLP output for the positive class useful? Why would it's integral over time diverge? Surely for a fixed dataset the MLP output stabilises? Is this integral then just a repeated sum of this stable mean value? Why does the divergence of $I_K(t)$ then imply ICL will occur?
- For Section 3.7: How standard is it to regularize only the in-context pathway? Seems like a fairly trivial result that if only this pathway is regularized then it will disappear.
- For Section 3.7: Since the dynamics are only considered once ICL is acquired, are the results of this section only valid if L2 regularization is applied once ICL is acquired? Since Equation 5 is not derived with L2 regularization I don't see how it can be just dropped in here and claimed that it ``applies throughout training'' and then the regularization term is just appended. I find it hard to believe that if the regularization term is present throughout training then ICL will emerge in the same manner as the  model which is used to option Equation 5.
- Line 402: it is mentioned that the added regularization may be explicit or implicit. What justifies the conclusion that any sort of regularization will hold in a similar manner (or  is that not the implication of this statement?).  What sort of implicit regularization will only apply a weight normalization to t he in-context pathway?

Unfortunately, given my confusion with the derivation due to the lack of rigor and derivations it is difficult to even assess Section 4. Many of the phenomena which are considered in this section are either due to fairly vague conclusions drawn on equations without clear meaning or seem to be true by construction (again having sufficiently large D that the key-query matrix is diagonal is a large simplification not given sufficient attention). Thus, the conclusions and insights of this section seem premature.

It is possible I am missing something in the research - indeed it is possible this research is very impressive. I also appreciate that theory is difficult and requires simplifying assumptions, especially to get to grips with something as complex as in-context learning in transformers. But I am certain that the presentation of this research is significantly lacking for the current version.

### Questions
- What is the $P^+$ value used to plot Figure 4 and how does it affect the loss landscape?
- Figure 5: the power law is presented as being between $K$ and $I_K(t)$ but then Figure 6a presents it as being between $N$ and $K^*$. How do these two representations relate?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper explores the mechanism by which models acquire the ability to generalize and memorize. 
The authors test the hypothesis that generalization arises because, in models with limited capacity, a preference for generalization over memorization emerges when task diversity is sufficiently high.
The authors find that generalization is not solely the result of capacity constraints but also due to different learning kinetics between sub-circuits responsible for generalization and memorization.
At sufficient task diversity, the sub-circuits responsible for generalization can be acquired faster than those for memorization.
They also find that a network can lose generalization abilities in favor of memorization with sufficient training under $L_{2}$ regularization.
The paper identifies a number of properties about the dynamics of generalization and memorization. 
Among them are how MLP-memorization scales with task diversity, how the task diversity threshold scales with context length, and the sensitivity of generalization to initialization.

### Strengths
The way models acquire generalization during training is an interesting topic and the paper provides an insightful exploration of the mechanisms involved. 
The authors convincingly support the premise of the paper with theory and empirical studies.

### Weaknesses
Because the model used for the empirical validation is stripped down to a one layer transformer model the authors note that they cannot be sure that the dynamics are the same for much larger models.
The authors could consider supplementing the one-layer analysis with tests on a deeper transformer model. 
Even if constrained by computational resources, experiments on a 2 or 3 layer transformer would help assess the applicability of their findings to larger networks.

The paper does not recommend methods to facilitate generalization. If possible, a discussion of how the ideas in the paper may be used to develop initialization and regularization techniques would make the submission even stronger.

### Questions
Can the authors use this work to recommend methods to improve generalization? For example, can we choose an initialization method that might better support ICL?

The theory hints that ICL decays due to the use of regularization. Though using $L_2$ regularization makes the results more comparable with typical training, would it be possible to provide empirical results without regularization?

### Soundness
3

### Presentation
3

### Contribution
3
