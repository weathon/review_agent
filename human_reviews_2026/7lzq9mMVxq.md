# Feature Hedging: Correlated Features Break Narrow Sparse Autoencoders

- Decision: Reject
- Scores: 4, 4, 4, 2

## Abstract
It is assumed that sparse autoencoders (SAEs) decompose polysemantic activations into interpretable linear directions, as long as the activations are composed of sparse linear combinations of underlying features. However, we find that if an SAE is more narrow than the number of underlying "true features" on which it is trained, and there is correlation between features, the SAE will merge components of correlated features together, thus destroying monosemanticity. In LLM SAEs, these two conditions are almost certainly true. This phenomenon, which we call feature hedging, is caused by SAE reconstruction loss, and is more severe the narrower the SAE. In this work, we introduce the problem of feature hedging and study it both theoretically in toy models and empirically in SAEs trained on LLMs. We suspect that feature hedging may be one of the core reasons that SAEs consistently underperform supervised baselines. Finally, we use our understanding of feature hedging to propose an improved variant of matryoshka SAEs. Importantly, our work shows that SAE width is not a neutral hyperparameter: narrower SAEs suffer more from hedging than wider SAEs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this work, the authors have defined and studied the feature hedging problem, theoretically in toy models and empirically in LLM SAEs. We show that hedging is worse the more narrow the SAE, and introduce a technique to characterise the amount of hedging present in a given SAE. They also studied hedging and absorption in matryoshka SAEs. They demonstrated that it is possible to improve the monosemanticity of matryoshka SAEs by adjusting the relative loss coefficients at each level of the matryoshka SAE better to balance the competing forces of absorption and hedging, while both issues remain present. It is also shown that the SAE width is not a neutral hyperparameter: narrow SAEs suffer more from hedging than wider SAEs.

### Strengths
Well written, good diagrams, the problem is clearly formulated
The methodology for the hedging degree is well explained
Feature hedging may not be a fully novel idea, buta  good and useful idea for SAE

### Weaknesses
Formal proofs are limited; relies mostly on empirical evidence and illustrative derivations.
Some assumptions—e.g., “parent latents are learned before child latents”—while intuitive, are not rigorously tested.
The broad idea that correlation can distort learned dictionaries has appeared in sparse coding literature so not so novel idea
Characterisation of Hedging is missing
\beta multiplier nonbelensing is not clear

### Questions
Give more rigorous theoretical proof
explain and reason out assumptions
Potential interactions between reconstruction loss and latent geometry could be explored more formally.
More experimentation can be performed

### Soundness
2

### Presentation
3

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
This paper describes the phenomenon of feature hedging, in some ways the opposite of the previously noted phenomena of feature absorption. Hedging is analyzed in four toy scenarios: independent, hierarchical, correlated, and anti-correlated features. Then, a metric, hedging degree, is proposed and used to measure hedging in SAEs for LLMs. Finally, the paper proposes balance matryoshka SAEs that scale the loss for each level to reduce hedging. This appears to be a simple approach that can help, although, as noted by the authors, not in all cases.

I learn towards rejecting the paper, mainly because feature hedging is not formally defined. This makes it difficult to judge whether the hedging degree is a reasonable metric, and limits formal understanding of the phenomenon.

Finally, I would like to disclose that I am not familiar with recent work on SAEs beyond the most well-known papers. There has been a surge of works in this area, so there could be closely related works to this paper that I am not aware of.

### Strengths
1. **Identifies a fundamental issue**. The paper identifies a fundamental issue in SAEs. I agree with the authors' conclusion in section 7 that understanding hedging is valuable for further work on SAEs.
2. **Balance matryoshka SAE**. This generalization of matryoshka SAEs is appealing. Weighing the levels is a small, simple change, and we can retrieve both SAEs and matryoshka SAEs as special cases, and the paper demonstrates that it can reduce hedging.

### Weaknesses
1. **Theoretical analysis**. Although the simple toy examples are useful for understanding when feature hedging may occur, some theoretical analysis of the phenomenon would be a useful complement. Would it be possible to explain why these feature-hedging optima occur in simple toy settings?
2. **Causes of hedging**. Throughout the paper, different causes for hedging are discussed. Some are related to the optimization problem itself (MSE, use of L1), others to the data (feature correlations), and even the learning process (in the paragraph on line 287). My impression is that the actual cause is not fully clear, and the discussion ends up slightly confusing. Again, some theoretical analysis could help clear things up.
3. **Hedging degree**. It is hard to tell whether this metric measures hedging, especially as hedging has not been formally defined. Moreover, it seems like a difficult metric to use in practice, as it depends on a hyperparameter N and training the network (and thus the training setup, such as the optimizer and length of extended training). A.5 states that the hedging degree increases with N, but does not provide a convincing reason for choosing some N. If there was, e.g., a formal definition of hedging and hedging degree was derived as an approximation of it, I would find its use more convincing, but as it stands, it does not appear to be a reliable metric.

### Questions
1. **Why not use more latents?**. Perhaps a naive question: why should we not simply make the model wider and focus on reducing feature absorption? If SAEs are "almost certainly narrower than the number of underlying features" (line 72), it would be natural to first make them less narrow, which would reduce hedging?
2. **$\beta$ multipliers**. On line 418, the $\beta$ parameters are parameterized relative to each other using a multiplier. How did you arrive at this particular form? That is: (ii) why is it decreasing, and (ii) why a constant multiplier and not some other function?

**Minor comments**
- Throughout the text, it would be good to disambiguate input features and latent features in the text so that they are not confused (as is done with $f$ and $l$ in mathematical notation).
- Line 17, "is caused by SAE reconstruction loss" -> "is caused by *the SAE's* reconstruction loss".

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces the problem of "feature hedging" in Sparse AutoEncoders (SAEs), motivating it first in the context of "toy" models, then empirically studying it in the context of larger-scale SAEs/LLMs. The authors proceed to introduce a tentative solution to hedging by incorporating a balancing term in the Matryoshka SAE loss function, finding that doing so presents empirical benefits for "Balance Matryoshka SAEs" relative to unbalanced baselines.

### Strengths
- The authors motivate, define, empirically study, and propose tentative solutions to a new and potentially important problem ("feature hedging") with leading interpretability methods (SAEs). 
    - Both defining/demonstrating this problem, and making substantive empirical progress towards understanding and resolving it, represent significant contributions to the interpretability community.
- The paper is generally well-written, with clear argumentation and strong intuition-building in both the introduction and "toy" experiments (sec 3).
- Hedging experiments (sec 4) are comprehensive, considering a range of LLMs/layers, SAE architectures and hyperparameters, etc.
    - And reporting several categories of SAEBench scores across hyperparameter values for the balancing multiplier (in sec 5) is also much appreciated for better understanding and contextualizing balancing with respect to standard (multiplier = 0) or unbalanced Matryoshka (multiplier = 1) SAEs.

### Weaknesses
The most substantial weakness is the lack of hedging results for Balance Matryoshka SAEs (elaborated below) -- given that the core motivation for the proposed balancing approach is to resolve the hedging problem, the absence of results showing whether balancing actually helps at all with this problem means that it cannot be taken seriously as a contribution. There are also more minor concerns regarding clarity on some important experimental details (discussed in the Questions section of the review) and citing prior interpretability work only within the narrow context of SAEs (elaborated below). Weaknesses by section are provided below.

Background/Related Work:
- There is already substantial work on closely-related problems in the context of probing classifiers -- see, e.g., [1-5]. (Note that this is not a serious novelty concern, as hedging is an SAE-specific problem and the most relevant works studying SAEs are already cited and discussed. Instead, I believe this is simply another instance of the "parallel community" studying mechanistic interpretability largely focused on SAEs that often fails to cite work from the broader, longstanding interpretability/model analysis literature, as highlighted in [6].)

[1] Kumar, A., Tan, C., & Sharma, A. (2022). Probing classifiers are unreliable for concept removal and detection. Advances in Neural Information Processing Systems, 35, 17994-18008.          
[2] Ravichander, A., Belinkov, Y., & Hovy, E. (2021, April). Probing the Probing Paradigm: Does Probing Accuracy Entail Task Relevance?. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume (pp. 3363-3377).        
[3] Elazar, Y., Ravfogel, S., Jacovi, A., & Goldberg, Y. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. Transactions of the Association for Computational Linguistics, 9, 160-175.            
[4] Canby, M., Davies, A., Rastogi, C., & Hockenmaier, J. (2024). How Reliable are Causal Probing Interventions?. arXiv preprint arXiv:2408.15510.     
[5] Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. arXiv preprint arXiv:1909.03368.           
[6] Saphra, Naomi, and Sarah Wiegreffe. "Mechanistic?." Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP. 2024.

Sections 4-5:
- In sec 5, it is claimed that "As we saw in Section 4.1, the more narrow an SAE is, the worse the hedging. Matryoshka SAEs thus solve feature absorption at the expense of exacerbating feature hedging." However, the experiments in sec 4.1 do not explicitly consider Matryoshka SAEs, weakening the motivation for Balance Matryoshka SAEs in sec 5. I suggest repeating the experiments in sec 4.1 (or a suitable subset, given compute constraints) with Matryoshka SAEs to determine the extent to which feature hedging is actually an issue for baseline Matryoshka SAEs.

Section 5:
- **(MOST IMPORTANT:)** There is no measurement of hedging for "real-world" balance Matryoshka SAEs (i.e., those trained on actual LLMs rather than toy models), so it is impossible to tell how well balancing actually solves this problem (relative to baseline Matryoshka SAEs without balancing). The results from "toy" models are useful for illustrating the tradeoff between hedging and absorption, but insufficient to claim that Balance Matryoshka SAEs present any real utility for resolving hedging in the target context of LLMs. I suggest repeating the experiments from sec 4.1 (or a suitable subset, given compute constraints) with the Balance Matryoshka SAEs trained in sec 5.

If the authors are able to perform the experiments noted above (perhaps at a smaller scale if this is infeasible during the discussion period) -- and if results show that Balance Matryoshka SAEs (at some multiplier between 0 and 1) do actually, nontrivially improve on baseline traditional SAEs (multiplier = 0) and Matryoshka SAEs (multiplier = 1) -- then I would be happy to increase my score.

### Questions
Intro:
- In the caption of fig 1b, the authors state that "Asymmetry between encoder and decoder is characteristic of absorption", which seems to be critical to the discussion in sec 4 (lines 291-296), and is also reiterated in tab 1. Can you elaborate on this point? Is this expected *a priori* (and in which case, what is the theoretical justification); or is it based on empirical findings in this or other work (in which case, can you cite the relevant work/section)?

Sec 3:
- In the paragraph "The implications of this for SAE performance are quite dire [...]" -- intuitively, I would tend to agree with that this is a problem, but I think the argument could be better fleshed out. For instance, what would the consequences be for practical interpretability desiderata, such as OOD detection or safety monitoring? Why is negative feature mixing an issue if two features really are strongly negatively correlated? 
    - The simple answer to the second question is that such correlations may be spurious and lead to poor OOD robustness of SAE features; but for SAE training datasets that are i.i.d. wrt model training data, such issues might still be indicative of the underlying representation learned by models, and thus useful for detecting spurious/shortcut feature learning. I am curious to hear how the authors would respond to this argument, and the degree to which it presents a real challenge to the core motivation of this work.

Sec 4:
- A central point supporting the experiments and interpretation throughout sec 4 is stated as: "Based on our understanding of hedging in toy models, we expect that when a new latent is added to an SAE, this should 'pull out' the component of the new feature from existing SAE latents, where it was previously hedged. Thus if hedging occurs, the change in existing latents after a new latent is added should project onto that new latent." I have a few questions on this point: 
    - To my understanding, the evidence in sec 3 and app A.1-2 comes from experiments where the number of latents is fixed prior to training the model -- which is a completely different setting from the expectation stated above. So:
        - Is my understanding here correct?
        - And if so, then what is the stated expectation based on? (I.e., please articulate precisely what it is in your "understanding of hedging in toy models" that would lead to this expectation, and why.)
    - Additionally, I feel that this argument would substantially benefit from a more rigorous formal presentation. For instance: (a) how could one mathematically state the expectation as a hypothesis to be tested, (b) how would one define a reasonable corresponding "null hypothesis" and measure the degree to which one is to be favored over the other, and (c) how could one formally state the evidentiary basis (per the previous questions) in relation to both hypotheses? (Note: I believe that (a) may already be covered by equation 7 and supporting text, but I would appreciate more clarification on precisely how it relates to the stated expectation/hypothesis; and I don't see anything corresponding to (b-c) in the main paper.)
- Similar to the previous question, what would one expect for a "null hypothesis" value of hedging degree h? (A control condition here would be useful -- if this is nontrivial to define in the context of LLM SAEs, then at least showing what h looks like for "toy" experiments like those in sec 3 (ideally across multiple dictionary sizes, k/L0 values, etc, at values closer to those in sec 4) would be helpful.)
- In fig 6,
    - In legends of all 3 plots, am I correct in assuming that "btk" refers to BatchTopK and non-btk is "l1" (meaning L1-regularized SAEs, such as those in Olah et al., 2024)? This was confusing to figure out.
    - What is being visualized for BatchTopK models on on the x-axis of fig 6c? To my understanding, the same k for BatchTopK is used for training all SAEs (with k=25, per appendix A.4) -- so how can there be multiple L0 values? Or are these separate models trained with different values of k?

Sec 5:
- The experiment whose results are reprted in fig 7 needs some clarification -- e.g., even after consulting the appendix, the following are still unclear:
    - What are the dictionary sizes $\mathcal{M} = m_1, ..., m_n$? My best guess is that there are just two dictionaries, $m_1, m_2 = 1, 4$, but this needs to be clarified.
    - $\beta = 0.25$ shows the best results. Does this mean that $\beta_{m_i} = 0.25$ for all $m_i$, or just $m_i : i > 1$?
- What is the basis for the multiplier formula used to obtain $\beta_m$ in lines 415-419? (I understand that the multiplier, such as 0.5, is a hyperparameter; but why set $\beta_m = \mu^{(n - m)}$ for multiplier $\mu$ rather than, e.g., sampling linearly between 0 and 1?)

Minor clarifications/nitpicks:
- In fig 6, the caption states "Hedging degree for SAEs trained on Gemma-2-2b layer 12" -- but both gemma and llama are tested, and fig 6b is over multiple layers, correct? Please update the caption accordingly.
- In fig 8, what is the purpose of setting the multiplier to a value greater than 1? I don't understand what this would correspond to theoretically; and empirically it seems that most variation occurs within [0, 1] (as one would intuitively expect).

### Soundness
2

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies LLM interpretability with sparse auto-encoders (SAEs). Previous works identify one phenomenon where one feature actives only when another does as the limiting factor for applying SAEs for interpretability. The authors introduce another limiting phenomenon which they call "feature hedging", that occurs when the SAE's latent space is too narrow and the underlying features are correlated. 

They demonstrate this through some toy experiments on 2 and 4 dimensional settings and they define a metric to detect this effect in LLMs. Their experiments on SAEs trained on LLM activations, and report their results using the metric (which they call hedging degree) as a function of the parameters of SAE and average number of active latents. Finally they propose a re-weighted Matryoshka training to overcome the phenomenon they proposed.

### Strengths
The idea of studying sparse autoencoders under the assumption of anti-correlated (or hierarchical) features is a good direction, the toy setup the authors propose is interesting and intuitive, and in the tests they show using LLMs, the hedging-degree metric appears to work.

### Weaknesses
While the direction of study is novel, the large scale experiments are limited in scope, and I am not fully convinced that the metric of hedging degree is the right one to measure this phenomenon. Is the $\lVert .\rVert$ an $\ell_2$ or $\ell_1$ norm? Further, what is $W_{rand}[0:N]$? are the latents initialized with a Uniform or a Gaussian distribution?

Can the authors give more reasoning behind selecting this metric besides it exceeding a "random" baseline? It would help their case to see tests using this metric on the toy setups. 

Overall, the writing can be greatly improved to a more precise way of defining terms. As an example, terms like polysemantic and monsemantic activations without their definitions, making this paper sort of inaccessible to users outside the area of interpretability. 

Overall, I do not recommend accepting this paper.

### Questions
Asked above.

### Soundness
2

### Presentation
1

### Contribution
2
