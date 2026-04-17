# Uncertainty Quantification for Regression Using Proper Scoring Rules

- Decision: Reject
- Scores: 6, 4, 2, 4

## Abstract
Quantifying uncertainty of machine learning model predictions is essential for reliable decision-making, especially in safety-critical applications. Recently, uncertainty quantification (UQ) theory has advanced significantly, building on a firm basis of learning with proper scoring rules. However, these advances were focused on classification, while extending these ideas to regression remains challenging. In this work, we introduce a unified UQ framework for regression based on proper scoring rules, such as CRPS, logarithmic, squared error, and quadratic scores. We derive closed-form expressions for the resulting uncertainty measures under practical parametric assumptions and show how to estimate them using ensembles of models. In particular, the derived uncertainty measures naturally decompose into aleatoric and epistemic components. The framework recovers popular regression UQ measures based on predictive variance and differential entropy. Our broad evaluation on synthetic and real-world regression datasets provides guidance for selecting reliable UQ measures.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents a unified framework for uncertainty quantification with focus on regression problems using proper scoring rules. The authors present theoretical derivation of their uncertainty measures and evaluate all combinations empirically on a set of tasks.

### Strengths
- very clear writing. Especially the motivation and background is exceptionally clear and easy to read. This lowers the barrier for the reader to understand the paper greatly.
- an exhaustive and strong generalized framework that extends existing frameworks from classification to regression
- evaluation on a rigorous set of uncertainty quantification tasks

### Weaknesses
- Your focus lies in parametric predictive uncertainty quantification with a mixture of Gaussians. While this is a common approach, it limits the applicability of the results to some degree.
- Within your experiments you make empirical observations which metric performs best and should be used. However, there is a lack of connection between empirical results and theory. 
- In your experiments you compare performance between your metrics. However, there are no other baseline methods presented.

### Questions
Main questions:
- Can you approach be generalized to non-parametric uncertainty quantification or other types of parametetric functions? How do the results generalize?
- Can you connect your empirical findings to theory and therefore generate insights into why specific metrics should be preffered for specific tasks? Or more general, can you provide recommendations which metrics should be used for uncertainty quantificiation in regression in general?
- How do you results compare to other baseline methods?
- What are the limitations of your approach?


Here are some more detailled comments and questions in the order of appearance of the paper (not in order of importance). Some might overlap slightly with weaknesses from above.
- Table 1: For CRPS the symbol $F_\hat{P}$ is not introduced. Also it would be good to give small examples or intuition for each proper scoring rule; possibly in the appendix.
- line 185: with "ensemble of Gaussians", do you mean a mixture of Gaussians or is there a difference?
- line 188 following / estimation of risk: Can you justify why you use these three risk? Meaning what makes them useful and theoretically grounded?
- line 194: The posterior approximation $\bar{P}$ is not used in the written risk estimation $R(\hat R, \hat R_*)$.
- line 209: The "earlier work" under a new set of axioms. Can you explain its significance to show why your generalization is of interest?
- line 217: twice in one sentence "thus".
- line 268: Can you rigorously define the different shifts to the posterior for clarity?
- figure 1: column middle and right: 
  - font size too small to be readable. 
  - Possibly try to colorcode or otherwise highlight the different risks on the x-axis to simplify readability. 
  - What does the gray background color for the Bayes risks indicate?
  - For row 1 and 4 you do not explain what the results show in contrast to row 2/3.
- line 272: What do you mean specifically with "None of the measures considered changes under this shift"?
- section 5.1: I am missing an interpretation of the results and to put it into context with the developed theory. Why are the results the way they are and what does this imply for the application of the specific measures?
- Figure 2: 
  - missing description of what the red and blue lines mean and why the dotted risks are not continuous. This also makes the interpreation of the text hard since I do not know what I am looking at.
  - Font too small again.
- Section 5.2: The example with the figure and the text are quite hard to understand and follow. I do not really get what you are doing, what you are showing, and what the results mean. Can you please clarify this whole section?
- Section 5.3: you suggesto to use $\hat R_{Tot}^{1,1}$ for selective prediction. Can you connect that to theory why this specific risk is useful in this task? Are there any insights besides the purely empirical numbers? Similar questions hold for section 5.4

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
This paper puts forward a framework for uncertainty quantification in regression problems. The framework is based on proper scoring rules, which allow for the quantification of the total risk, which is, in this paper, equated to total uncertainty. Using a known decompositions of proper scoring rules into an entropy function and a divergence function, which represent Bayes risk and excess risk or aleatoric and epistemic uncertainty, respectively. The paper suggests three methods to estimate these risks and considers four loss functions as explicit instantiations. The framework is evaluated by looking at a characterization of the uncertainty measures, synthetic experiments, and common uncertainty tasks.

### Strengths
The paper uses a valuable framework that allows for a unified consideration of different known uncertainty measures.

The work includes a wide spectrum of empirical evaluations across different uncertainty applications.

In general, the work is very complete, providing a good background on the proper scoring rule literature, derivations, and many additional results.

### Weaknesses
The work is very similar to (Kotelevskii et al., 2025), but with different loss functions. It is, therefore, an overstatement to say that this work introduces a framework for uncertainty in regression. The framework already exists, the authors cite many previous works using the same framework, but it is used in a regression context. Thus, the novelty is limited.

The characterization of uncertainty measures section should reference the work by Bülte et al. (2025). Many of the “shifts” in the current work were introduced in (Bülte et al., 2025).

The characterization is not connected to the rest of the empirical findings. Specifically, it would be valuable to connect the performance of specific loss-approximation combinations on tasks such as selective prediction to specific properties that the measures satisfy. 

Additionally, the paper mostly reports how measures perform, but there is no discussion about *why* certain measures and approximations perform well.

### Questions
For the three tasks (selective prediction, out-of-distribution detection, and active learning), can you give reasons for the good performance of the best performing instantiations. Specifically, could you explain why using these particular losses and approximations work well for the respective tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper aims to extend the theoretical framework of proper scoring rules (which previously achieved more attention for classification) to the regression setting for uncertainty quantification (UQ). The authors give a systematic overview of different uncertainty measures derived from different scoring rules. These uncertainty measures can quantify the total risk and allow for decomposing the total risk into aleatoric and epistemic components. The paper also derives closed-form approximations under Gaussian assumptions, and provides empirical evaluations on synthetic and real-world datasets.

While the topic is relevant and potentially impactful, the novelty is limited. The paper largely reformulates existing ideas (such as CRPS, log score, Bayes risk, and related decompositions) under restrictive Gaussian assumptions. The empirical analysis reproduces expected behaviors but does not generate novel insights or practical guidance.

### Strengths
The paper tackles a relevant and timely topic. This is an area of active interest in both machine learning and statistics, and the authors’ attempt to build a framework is well-motivated. Uncertainty quantification and disentangling uncertainty into aleatoric and epistemic uncertainty are important for many applications. (However, the choice of the scoring rule does not seem to be the most pressing issue in this context.)

The empirical section is broad in scope, covering multiple datasets and several downstream applications such as selective prediction, OOD detection, and active learning. The results are consistent with theoretical expectations.

The paper nicely demonstrates that disentangling uncertainty into epistemic and aleatoric uncertainty does have a significant impact on downstream tasks for regression. This provides a useful counterargument against the framing by [Mucsányi et al] (https://proceedings.neurips.cc/paper_files/paper/2024/hash/5afa9cb1e917b898ad418216dc726fbd-Abstract-Datasets_and_Benchmarks_Track.html ). [Mucsányi et al] are sometimes interpreted as disentangling uncertainty into epistemic and aleatoric uncertainty doesn't work in practice (for classification). But the experiments by [Mucsányi et al] are limited and focus on classification only. Therefore the regression results by this paper help to put the results by [Mucsányi et al] into perspective.

### Weaknesses
Questionable novelty of the theoretical contribution. The theoretical sections largely restate known results (CRPS, log score, Bayes risk, total/excess risk) and rederive them under restrictive Gaussian assumptions. These notions are all well-established. It is unclear what genuinely new theoretical result the paper contributes beyond repackaging existing ideas and recomputing them explicitly for the Gaussian case.  Furthermore, the results in Tables 2–3 are mostly trivial as far as I understand, or am I wrong? I agree that these tables are a nice way to quickly provide an overview of the main intuition behind these different uncertainty measures; however, I would not count this as a major contribution. There is no Theorem that is novel, and I do not see any new insight for practitioners. I would summarize the main takeaway for practitioners as: the typically used decompositions by (Depeweg et al., 2018) work well (sometimes CRPS1,1 can be slightly better). No significant improvement over them was found, and in practice, it does not really matter which of them one should use. There are also some other possibilities to define it that behave almost identically. There are some other strange exotic ways to define it that nobody would ever use in practice, that actually don't work that well in practice. However, distinguishing between epistemic, aleatoric, and total uncertainty can have a significant difference on downstream tasks. Do you agree that this is the main summary of the results, or is there more to it?

When reading the paper for the first time, I thought: <<Strong and limiting assumptions. The framework relies entirely on Gaussian assumptions. While this simplifies derivations, it severely limits generality. Moreover, this procedure effectively measures the discrepancy between $\hat{P}$ and the Gaussian approximation, which is questionable if the Gaussian model is itself misspecified. You boldly claim that “it is necessary to make specific parametric assumptions.” Why should that be the case? There exist many nonparametric approaches to estimate P(Y|X) without imposing such restrictive assumptions.>>. Now reading it again, I think this was maybe a misunderstanding. Now, I think Tables 2 and 3 are rather an intuitive example of a more general framework. Maybe the paper could be clearer about what the main contributions are. 

Issue with main contributions 3: You wrote ``A broad empirical evaluation on synthetic and real-world datasets, culminating in practical recommendations for selecting uncertainty measures (Section 5).’’ The claimed “recommendations” are missing. They are neither summarized in the conclusion nor presented in any subsection. If they exist somewhere in Section 5 or the appendix, they are too well hidden to be useful.

Empirical evaluation offers little new insight. Although the experiments are extensive, they mainly confirm well-known qualitative behaviors (“as theory suggests”) rather than providing new understanding or guidelines. The “practical recommendations” promised in the abstract and contributions are not actually presented in the paper or are too buried to be actionable.

Presentation:

* Figure 2 is very unclear. The meaning of the colors (red, blue, and multicolored dots) is not explained; at least not in the main text. If one needs to consult the appendix to understand the figure, the figure itself probably belongs in the appendix. The colors of the dots are explained, but what do the x and y coordinates of the colorful dots correspond to. What are the red and blue lines? What was the training data? Section 5.2 is absolutely not understandable without the abstract.
* Figures 2 and 3 are also too small. Typical guidelines state that the text and axes labels should be readable in print without zooming in. As it stands, they are so insanely small that I can not even zoom in enough on my tablet. The quality of the figure is good, though. 
* Tables 2 and 3 include unexplained symbols (e.g. $A$). These should be defined in the captions or main text. I suppose they can be found in the appendix, but since they are used in the 'main contributions section', it should be at least explained why there are undefined symbols. 
* Sections 2 and 3.1 could be merged. Much of the material repeats: the definitions of total risk (Equation 1) and Bayes risk (Definition 1) overlap conceptually, but occupy substantial space. Just an idea. I leave the choice up to you.
* In my opinion, the goal of such a survey paper should be to clarify things also for people who are new to this field. I'm a bit worried that some of it creates more confusion than clarification. For example, $\hat{R}^2_{\text{Bayes}}$ for SE, is something which would usually be referred to as total uncertainty rather than aleatoric uncertainty. However, the Bayes risk is supposed to estimate the aleatoric uncertainty. This is confusing. I understand that in your hyper-general framework, when you try to write all possible combinations of ideas, then you end up with strange situations like this. But for someone who is new to the field, this could be very confusing. I think it would be good to add a discussion warning readers about these extremely counterintuitive uncertainty measures. It seems like this plug-and-play framework that you introduce does not always make sense for every plug-and-play combination.I think, especially for the readers who are new to these topics and are looking for a nice overview, there should be a clear message that the current state of the art is to simply use the well-established decompositions by (Depeweg et al., 2018) instead of putting too much thought into this hyper-general plug-and-play system. Also, you talk about 9 combinations, but (2,2), for example, obviously does not make any sense, and then you actually don't show (2,2) in your experiments, but you never explain this or write about this in the paper. Overall, it seems as if this framework should not be sold as a general plug-and-play system, but rather there are some combinations that make sense (the ones by  (Depeweg et al., 2018) and a couple more) and others that don't make sense.
* Line 195: Not really understandable without the appendix. When you write 3a and 3b, does this correspond to $\hat{P}\_\star$ and $\bar{P}_*$?
* Line 405: You write "substantially worse", but I would say "slightly worse", as the difference is not significant and really tiny. (For vegetation, arrow, and dots Log is even the best, not worse.)
* I think Table 4 is a bit misleading as it averages over multiple datasets with results on slightly different scales from Table 9. In Table 9, one can better see how inconsistent the rankings are across datasets.
* Line 128: There is a “with with” typo.

### Questions
1. Can you say very specifically what the paper’s novel theoretical contribution (Sections 1-4) is? Is it anything beyond Tables 2 and 3? I am just not really seeing which definition/computation / … can't be found in existing literature (note that I did not dig deep into the Appendix)
2. Why is Gaussianity treated as necessary? Couldn't the framework be extended to nonparametric predictive distributions?
3. Could you summarize the practical recommendations your experiments yield? What should the reader actually take away from your results?
4. In paragraph "Scale shift of predicted means," it sounds as if you want aleatoric uncertainty to increase for scale shifts of the predicted mean. Intuitively, I would say that the **aleatoric** uncertainty should not be affected by a Scale shift of predicted means. Please clarify which uncertainty component is expected to increase under a scale shift of the predicted means.
5. How do Eq. (3) and (4) fit into your framework as the logarithmic and SE special cases? As superscript 1,2? Why not state this more explicitly?

It was hard for me to decide whether to give a 2 or a 4. So, if you improve the presentation and soften the tone (i.e., don't claim that something is substantially worse, when it is actually only worse by ~1.5% on average, which is not even statistically significant)

### Soundness
2

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
This paper introduces measures of uncertainty for regression based on proper scoring rules. The derived measures naturally decompose (additively) into aleatoric and epistemic parts. The proper scoring framework recovers popular measures (variance- and entropy-based ones). The authors empirically evaluate on synthetic and real-world regression datasets.

### Strengths
The research direction itself seems to be very promising and of considerable importance for current developments in the (reliable) machine learning community. Given that loss based approaches to uncertainty quantification for classification purposes turned out to be quite successful in the recent past, a regression lens seems to be the logical next step.

In my opinion, the paper’s main contribution is a fairly original framing of uncertainty quantification through proper scoring rules and Bayes risk, coupled with a systematic comparison of three relevant (in particular for practical use cases) aggregation schemes (per-member averaging, mixture, and single-Gaussian fit). This indeed unifies several ad-hoc practices under one lens and connects them to familiar variance/entropy based uncertainty measures, which is useful for both theory and practice (for the latter see my later critical remarks).

Moreover, the presentation is generally speaking quite clear. The paper summarizes relevant formulas in tables, and explains computational trade-offs of the three aggregation schemes so practitioners can anticipate cost implications.  

Empirically, the evaluation covers synthetic and real-world settings and shows that the proposed aggregations are broadly comparable in practice, which is a valuable takeaway by itself.

### Weaknesses
First, let me elaborate on some more technical things I noticed. For brevity, let $\hat p_{\text{e}}=\frac{1}{M}\sum_{i=1}^M \hat p_i$. Then
\begin{align}
\hat{R}^{2}
= -\int \hat p_{\text{e}} \log \hat p_{\text{e}} 
= -\int \frac{1}{M}\Big(\sum_i \hat p_i\Big)\log \Big(\frac{1}{M}\sum_j \hat p_j\Big) 
= -\frac{1}{M}\int \Big(\sum_i \hat p_i\Big)\log \Big(\sum_j \hat p_j\Big) dy  +  \log M.
\end{align}

Thus, there seems to be a minus sign missing in Table 2, and Appendix 2.2. omits an additional $+ \log M$ term. Further, Appendix A.2.2 claims that "the logarithmic score is a proper scoring rule for discrete distributions". This is somehow misleading, since the log score is proper for both discrete and continuous cases.

In L. 216 the authors write that "CRPS is a non-negative metric". CRPS is a strictly proper scoring rule. The metric is the squared L^2 distance between CDFs that arises from expected CRPS, not the pointwise score itself. 


Is the third bullet point (in the Bayes risk derivations for the quadratic score) a typo? Should be $H(P_{\star})$, right?

An other weakness (at least from my perspective) is the following. While I agree that such a proper scoring rule framework in the regression setting is indeed suitable, the motivation for me is not immediate; by motivation I mean, for each such loss we get uncertainty measure, but  what do we get by such collection of measures? By a quick literature search, I found some related work (see e..g [1], or [2]), which seems to argue why such different instantiations might be useful (although in the classification case). Could the authors elaborate on this and if possible relate to their theory/experiments?

On the empirical side:

Across selective prediction and OOD tables, many entries across approximation strategies are numerically very close (frequently within one standard deviation), so any rank-ordering should be done very cautiously. It may help to add a short sentence stating that the strategies are broadly comparable empirically, with no consistent winner across datasets/scores. 

Several choices make it hard to draw comparative conclusions. More precisely, (i) PRR averages with overlapping SDs lead to near-ties among many methods, (ii) the OOD benchmark is so easy (by construction) that AUROCs saturate, and (iii) active-learning gains are modest and sometimes inconsistent. The setups as implemented provide limited discriminative power to justify those prescriptions beyond qualitative trends. I agree that this is not an easy task at all, but as executed, I am not sure I understand the actual take-away message from the empirics..

Minor things:

L. 128, duplicate "with with".

L. 663, typo "initilizations" instead of "initializations". 

[1] Hofman, P., Sale, Y., & Hüllermeier, E. (2025). Uncertainty Quantification with Proper Scoring Rules: Adjusting Measures to Prediction Tasks. arXiv preprint arXiv:2505.22538.

[2] Mucsányi, B., Kirchhof, M., & Oh, S. J. (2024). Benchmarking uncertainty disentanglement: Specialized uncertainties for specialized tasks. Advances in neural information processing systems, 37, 50972-51038.

### Questions
I already formulated (implicitly) some questions in the earlier questions, but will list some more that I had during reading the manuscript.

Could you distill decision rules for when to prefer per-member averaging, the mixture, or a single-Gaussian fit (e.g., multi-modal targets, heavy tails, low ensemble diversity)?

Members are trained under a Gaussian likelihood, evaluation spans CRPS, log, quadratic, and squared-error criteria. How sensitive are rankings to loss–evaluation mismatch? 

Do your takeaways extend to vector-valued regression?

### Soundness
2

### Presentation
3

### Contribution
2
