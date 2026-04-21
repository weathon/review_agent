# Defining and Measuring Disentanglement for non-Independent Factors of Variation

- Avg Score: 5.75
- Decision: Reject
- Scores: 5, 5, 5, 8

## Abstract
Representation learning is an approach that allows to discover and extract the factors of variation from the data. Intuitively, a representation is said to be disentangled if it separates the different factors  of variation in a way that is understandable to humans. Definitions of disentanglement and metrics to measure it usually assume that the factors  of variation are independent of each other. However, this is generally false in the real world, which limits the use of these definitions and metrics to very specific and unrealistic scenarios. In this paper we give a definition of disentanglement based on information theory that is also valid when the factors are not independent. Furthermore, we demonstrate that this definition is equivalent to having a representation composed of minimal and sufficient variables. Finally, we propose a method to measure the degree of disentanglement from the given definition that works when the factors are not independent. We show through different experiments that the method proposed in this paper correctly measures disentanglement with independent and non-independent factors, while other methods fail in the latter scenario.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The authors propose measuring disentanglement through Minimality and Sufficiency. 
They propose two metrics based on upper bounds of the before mentioned quantities. 
They demonstrate their metrics on synthetic experiments. They claim that their metrics are better suited for correlated datasets.

### Strengths
The paper is well written in general. The introduction and related work section position the reader adequately. Using the concept of minimality and sufficiency is thought provoking.

### Weaknesses
- If I understand correctly, the authors decided to use an upper bound for their metric, but this design choice lacks motivation and is only noted in the Appendix.

- The experimental section is, unfortunately, not convincing to me. Multiple design choices of the authors are not motivated. The authors did not investigate their metrics's correlation with down-stream task utility. 
The authors never looked at actually learned representations, but only used simple linear (and in the first section, trigonometric) mappings. The discussion the authors give is mostly limited to extreme/edge cases of their metric. The metrics overall behaviour is not discussed.

### Questions
- Appendix B.1 is titled *DERIVATION OF THE ESTIMATOR FOR MINIMALITY* and B.2 is titled *DERIVATION OF THE ESTIMATOR FOR SUFFICIENCY.* However, the authors seem to derive upper bounds. Did I understand this correctly? If so, how tight is the bound? Can the authors please elaborate why they believe an upper bound is sufficient for a metric? 

- In Section 4.3 The authors introduce $\bar{m}$ and $\bar{s}$, writing "it is also interesting to have a single term that determines how minimal a representation z is." 
Why did the authors decide to go with the max here? Why not use some other kind of aggregation? Are the  $\bar{m}$, $\bar{s}$ metrics the ones used in the experimental section?

- In the experimental section 5.2, the authors generate a mapping $A \in \mathcal{R}^{n_y \times n_y}$ by fixing the diagonal values to $1- \alpha $ and sample the non-diagonal values from a uniform distribution between 0 and $\alpha$. 
Did I understand this correctly? Would then $\bar{m} = \frac{1}{n_y} \sum_{j=1}^{n_y}m_{jj}$ and $\bar{s} = \frac{1}{n_y} \sum_{j=1}^{n_y}s_{jj}$ for $\alpha < 0.5$ ? If so, why do we see a decrease for $0 < 0.5 < \alpha$ in (a), (b) and (c) of Figure 5 but not in (d) (e) and (f)?

- Why is the disentanglement metric DCI-D at 0 for dSprites and $\alpha = 0$? 

- In Appendix D, Figure 13, 14 and 15 seem to have the same caption? Is this a mistake? Intuitively I must say that the gradient in Sufficiency looks less pronounced than for other metrics (e.g. Disentanglement). Can the Authors please elaborate on the plots in Appendix D, how do they show the superiority of their metric?

- No standard deviations are provided in Figures 5 and 6 hinting at a single run, which, given the dynamics of the curves, seems insufficient. Either the authors add std's, or they move one of the Figures of Appendix D into the main paper, as it depicts many more runs.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes to define and measure disentanglement in a way that is more relevant for real-world scenarios, where (1) the true generative factors of variation are not necessarily independent, and (2) there are nuisance factors which are not relevant for a given task but may act as confounders. The metrics proposed in this paper are somewhat similar to those proposed by Ridgeway & Mozer (2018) and Eastwood & Williams (2018), but generalized/extended in the 2 ways mentioned above. Factors are still assumed to be independendent here, but only given the observed raw data, while in previous work they were unconditionally independent.

The authors consider 4 properties:
1. Factors-invariance: $z_j$ is factors-invariant for the factor $y_i$ if, given $y_i$, there is no mutual information between $z_j$ and all the other factors in $\mathbf{y}$. This is a direct extension of modularity/disentanglement that accounts for correlations.
2. Nuisances-invariance: this is the same as factors-invariances, except we replace "all the other factors" with "all nuisance variables".
3. Representations-invariance: $z_j$ is representations-invariant for the factor $y_i$ if, given $z_j$, there is no mutual information between $y_i$ and all the other representations in $\mathbf{z}$. This is a direct extension of compactness/completeness that accounts for correlations.
4. Explicitness: $\mathbf{z}$ is explicit for the factor $y_i$ if, given the full representation $\mathbf{z}$, the data $\mathbf{x}$ provides no additional information about $y_i$, i.e., $y_i$ is fully described by $\mathbf{z}$.

The authors then show that:
- Minimality is equivalent to (1) and (2) jointly.
- Sufficiency is equivalent to (3) and (4) jointly.

They also argue that it is reasonable to focus on minimality and sufficiency, and propose methods to estimate these quantities in practice, and showcase their metrics alongside classic disentanglement metrics in a few toy experiments.

### Strengths
- Very well-written introduction, with a good explanation of the motivation. In general the background and setup are nicely laid out, which is a service to the community.
- Overall pretty easy to follow, with clear and convincing arguments in the background/theory section.
- Interesting ideas that are relevant for the community.

### Weaknesses
**Clarification on minimality and sufficicency.** As far as I understand, minimality and sufficiency are always defined w.r.t. a task, and considering the entire representation. Here, the authors talk about "task" to distinguish between relevant factors $\mathbf{y}$ and nuisances $\mathbf{n}$. This is consistent with previous work on representation learning that concerns minimality and sufficiency. However, in this paper, the 4 properties (factors-invariance etc.) are linked to minimality and sufficiency in a different context, where the representations $z_j$ are considered separately, and the "tasks" are the single factors $y_i$. Is my interpretation correct? Regardless, I think this point should be clarified and made more explicit, to avoid potential misunderstandings. (This could also help give a very precise link between the 4 properties here and the classic disentanglement metrics by Ridgeway & Mozer (2018) and Eastwood & Williams (2018))

**Beginning of Sec. 5:**
> In this section we use minimality and sufficiency (lowercase) to refer to the properties of section 3 and Minimality and Sufficiency (uppercase) to refer to the metrics of section 4.3.

Two issues here:
1. Minimality and sufficiency are not mentioned in Section 3. If the authors mean to distinguish between properties and metrics, then I guess the reference should be to Section 4.1?
2. Distinguishing only with capitalization sounds like a recipe for misunderstandings. I would suggest using letters/symbols or acronyms, for example.

**Experiments.**

The theoretical arguments for why these metrics make more sense than existing ones are very clear. The empirical arguments, however, less so.

1. On the comparison between e.g. minimality and DCI disentanglement in Sec. 5.1: at least in the uncorrelated case, DCI should be good (especially with low nuisance strength). And in fact, I believe it is. For example, I disagree with "only takes high values when factors-invariance and nuisances-invariance levels are very low" being a problem. Absolute values of metrics are not necessarily as interesting as how they rank different representations. I don't think there's anything inherently more "accurate" or "correct" about the Minimality metric than e.g. Minimality$^2$. What about showing scatter plots and/or computing (rank) correlations between the metrics? Would it give a more complete picture perhaps?

2. My major concern is that, as I wrote above, I think the most interesting aspect is actually how different metrics rank different representations. And the ground truth to make comparisons with, should not be an abstract metric, but rather a concrete evaluation whose relevance/usefulness the community can agree on. So I think it would be important to investigate if these metrics can be useful in practice e.g. for model selection for downstream tasks (including e.g. generalization, fairness, sample efficiency etc.), as done by quite a few disentanglement papers especially around 2019--2021. So a research question could be: when using disentanglement metrics to cheaply select representations for downstream tasks, does our metric yield better results than others according to some evaluation metric?

3. Another concern I have is that sometimes these proposed metric don't seem to do a very good job, especially in the toy image datasets with correlations between factors. E.g. in Fig. 5 DCI disentanglement seems quite bad, but it's arguably better than Minimality on MPI3D (and it's not the only one). The situation is even worse for the sufficiency-like metrics in Fig. 6, where there's no particularly obvious advantage of using Sufficiency (most curves are relatively close to each other). I suspect the issue might be resolved if the authors clarify their interpretation of Figs. 5 and 6. Otherwise, this is clearly a limitation, since it's even in the correlated case, where these methods should shine. So it should be addressed upfront, and ideally there should be a bit more investigation into why this happens, what impact it might have (but see my paragraph above, i.e. to measure actual impact there's more work to be done), whether something can be done to mitigate it.

4. In addition, I think it would be interesting and useful to consider different ways/degrees of mixing as e.g. in [Eastwood et al. (2023)](https://arxiv.org/abs/2210.00364). But that's perhaps more of a side quest and it's more related to explicitness as defined in that paper (see comment on this below under "minor"). I wouldn't prioritize this, although maybe a short discussion in passing could be beneficial.

5. Regarding datasets, why use image datasets (Sec. 5.2) if the images as far as I can see are never used? I agree it doesn't make sense to use images, but then why not generate random low-dimensional data (just the factors) since they just need to be mixed artificially? Again, these datasets would be useful when doing model selection in a practical (toy) setting as in classic disentanglement papers, to see if these metrics can indeed be more useful than previous ones.

**Minor**:
- end of page 3: "connected to those described in 3" is a reference to the current section (also, it should be "Section 3" anyway)
- line 286: "since this gap can be low for correlated factors even when zj is minimal." I would maybe expand a bit on this to clarify
- There might be some issues/inconsistencies with notation. As far as I understand, $y_j$ is a factor, there are $n$ factors, and $\mathbf{y} = \{y_i\}_{i=1}^n$ is the set of factors -- all this for a single data point, because e.g. $y_i$ is not bold. But then when "datasets" appear, it seems that actually $y_i$ was the vector of $i$-th factors from the entire dataset, and when considering a single data point we have the notation $\mathbf{y}^{(k)}$, where $k=1,...,K$ and $K$ is the dataset size. I think this notation should be clarified earlier in the manuscript.
- Explicitness as in Ridgeway & Mozer (2018) is a bit misleading, and I think informativeness is much better (on the other hand, note that compactness in my opinion is more descriptive than completeness). Explicitness is also defined as additional metric by [Eastwood et al. (2023)](https://arxiv.org/abs/2210.00364) where perhaps the term explicitness makes more sense.

**In conclusion**, I think these are very interesting ideas and they are well explained. The writing is also overall good. I think the experimental validation is however rather lacking. Note that I'm not asking for any real-world or larger-scale experiments -- I just think to prove the points the authors want to prove, a wider and better-designed experimental study is necessary.

### Questions
See weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper addresses the topic of measuring the degree of disentanglement in learned representations. Existing metrics often rely on the assumption that underlying factors are independent, limiting their effectiveness in real-world applications. To address this, the authors propose two complementing metrics: Minimality and Sufficiency. These metrics enable disentanglement assessment without requiring factor independence. Experimental results indicate that Minimality and Sufficiency can identify the presence or absence of disentanglement in scenarios where previous metrics are ineffective.

### Strengths
The authors propose two predictor-based metrics, Minimality and Sufficiency, for assessing disentanglement without assuming factor independence. Existing approaches often rely on this independence assumption, which rarely holds in real-world data, limiting the applicability of such metrics to narrow and often unrealistic cases. Minimality and Sufficiency are introduced as complementary, yet opposing, principles that balance trade-offs in capturing disentanglement. The authors provide formal definitions and proofs grounded in Information Theory. Through experiments, they demonstrate that existing metrics struggle to measure disentanglement effectively when factors are not independent, whereas the proposed metrics perform better in these more realistic settings.

### Weaknesses
While the authors argue that Minimality and Sufficiency metrics are more applicable to real-world scenarios due to their independence from factor assumptions, this claim would benefit from validation on complex, real-world datasets. For example, using datasets like CelebA, which include nuanced features and dependencies, could illustrate the metrics' practical advantages.

Minimality and Sufficiency are properties associate with information bottleneck techniques, yet this connection is not addressed. Expanding the related work section to discuss information bottleneck approaches would strengthen the paper’s theoretical foundation by situating the metrics within established work. Including studies that use information bottlenecks for disentanglement would clarify how Minimality and Sufficiency build upon or differ from these techniques.

### Questions
1. The experiments involve a relatively small number of factors and simpler representations. Could the authors elaborate on how Minimality and Sufficiency perform in more complex scenarios with larger factor sets and higher-dimensional representations? Are there any known limitations in applying these metrics?

2. While Minimality and Sufficiency appear to offer distinct advantages, are there any scenarios where they might fail to capture disentanglement effectively, or where they might be less reliable than existing metrics?

3. In Section 4.3, the authors mention that Minimality and Sufficiency are better suited for cases where factors are not independent, as they focus on the most influential factors and not comparisons across factors. Could existing methods like DCI be modified to relax the independence assumption? If so, would Minimality and Sufficiency still provide a more effective disentanglement assessment, reinforcing the novelty of proposed metrics?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper explores disentangled representation learning, where the goal is to separate factors of variation in data into understandable parts. Traditional definitions assume these factors are independent, which is often unrealistic in real-world data, limiting their applicability. The authors introduce a new, information-theory-based definition of disentanglement that works even when factors are not independent. They show that this new definition aligns with representations made up of minimal and sufficient variables. Additionally, they propose a method to measure disentanglement in these scenarios and demonstrate its effectiveness in handling both independent and non-independent factors, outperforming existing metrics.

### Strengths
- The paper provides a great theoretical framework for understanding the nature of disentanglement. It presents a set of succinct conditions that are essential to disentanglement and condenses them into two highly general notions: minimality and sufficiency.
- The proposed disentanglement measure does not require independent latent factors, addressing a key limitation of previous approaches.
- The paper is very well written. It includes an extensive discussion of the related work and a nicely listed discussion on the desirable properties of a disentangled representation. The whole derivation process from the basic properties to the final algorithm is very clear and easy to follow.

### Weaknesses
- The proposed metrics (minimality and sufficiency) may not be very informative in cases where $y$ can’t be perfectly recovered from $x$. In such cases, no representation of $x$ can achieve 100% minimality and sufficiency, meaning that the factors can’t be perfectly disentangled. As a result, the optimal value of minimality/sufficiency may be different for different tasks. In general, this value is a priori unknown and may not be easy to estimate. This makes it difficult to tell if a representation is good or bad (in terms of disentanglement) if the measurement of minimality/sufficiency yields a medium value (not close to 0 or 1).
- The measurement algorithms require the ground-truth values of the causal factors $y$, which may be difficult to obtain for many real-world tasks. Moreover, in cases where $y$ has to be estimated from $x$, the estimation quality could affect the measurement of minimality/sufficiency.
- The metrics are defined as ratios between two mutual information terms in order to scale them between 0 and 1, but is this really a good choice? For example, do $\frac{I(z_j;y_i)}{I(z_j;x)}=\frac{0.01}{0.1}$ and $\frac{I(z_j;y_i)}{I(z_j;x)}=\frac{10}{100}$ really mean the same thing for minimality? Note that the values of $I(z_j;x|y_i)$ are quite different in these two cases. The definition deserves a more careful discussion.
It seems that the experiments only involve problems with a small number of causal factors. Is the proposed measure also accurate in much higher-dimensional spaces? Will there be computational issues? The time complexity of the measurement algorithm is O(mn).
- Why are the X, Y, and Z in the first paragraph of section 4 in capitals whereas the rest of the paper uses lower cases?
- Typos: 1. Line 69-70: missing “of” between “degree” and “minimality”; 2. Line 229: have *the* that; 3. Line 4 of Algorithm 4: n should be m.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
